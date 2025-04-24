/* SPDX-License-Identifier: GPL-2.0 */

/*
 * transcode.c - convert audio file to WAVE
 *
 * Copyright (C) 2019		Andrew Clayton <andrew@digital-domain.net>
 * Copyright (C) 2024       William Tambellini <william.tambellini@gmail.com>
 */

// Just for conveninent C++ API
#include <vector>
#include <string>

// C
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

extern "C" {
#include <libavutil/opt.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswresample/swresample.h>
}

typedef uint64_t u64;
typedef int64_t  s64;
typedef uint32_t u32;
typedef int32_t  s32;
typedef uint16_t u16;
typedef int16_t  s16;
typedef uint8_t   u8;
typedef int8_t    s8;

#define WAVE_SAMPLE_RATE	16000
#define AVIO_CTX_BUF_SZ		 4096

static const char* ffmpegLog = getenv("FFMPEG_LOG");
// Todo: add __FILE__ __LINE__
#define LOG(...) \
  do { if (ffmpegLog) fprintf(stderr, __VA_ARGS__); } while(0) // C99

/*
 * WAVE file header based on definition from
 * https://gist.github.com/Jon-Schneider/8b7c53d27a7a13346a643dac9c19d34f
 *
 * We must ensure this structure doesn't have any holes or
 * padding so we can just map it straight to the WAVE data.
 */
struct wave_hdr {
	/* RIFF Header: "RIFF" */
	char riff_header[4];
	/* size of audio data + sizeof(struct wave_hdr) - 8 */
	int wav_size;
	/* "WAVE" */
	char wav_header[4];

	/* Format Header */
	/* "fmt " (includes trailing space) */
	char fmt_header[4];
	/* Should be 16 for PCM */
	int fmt_chunk_size;
	/* Should be 1 for PCM. 3 for IEEE Float */
	s16 audio_format;
	s16 num_channels;
	int sample_rate;
	/*
	 * Number of bytes per second
	 * sample_rate * num_channels * bit_depth/8
	 */
	int byte_rate;
	/* num_channels * bytes per sample */
	s16 sample_alignment;
	/* bits per sample */
	s16 bit_depth;

	/* Data Header */
	/* "data" */
	char data_header[4];
	/*
	 * size of audio
	 * number of samples * num_channels * bit_depth/8
	 */
	int data_bytes;
} __attribute__((__packed__));

struct audio_buffer {
	u8 *ptr;
	int size; /* size left in the buffer */
};

static void set_wave_hdr(wave_hdr& wh, size_t size) {
    memcpy(&wh.riff_header, "RIFF", 4);
    wh.wav_size = size + sizeof(struct wave_hdr) - 8;
    memcpy(&wh.wav_header, "WAVE", 4);
    memcpy(&wh.fmt_header, "fmt ", 4);
    wh.fmt_chunk_size = 16;
    wh.audio_format = 1;
    wh.num_channels = 1;
    wh.sample_rate = WAVE_SAMPLE_RATE;
    wh.sample_alignment = 2;
    wh.bit_depth = 16;
    wh.byte_rate = wh.sample_rate * wh.sample_alignment;
    memcpy(&wh.data_header, "data", 4);
    wh.data_bytes = size;
}

static void write_wave_hdr(int fd, size_t size) {
	struct wave_hdr wh;
    set_wave_hdr(wh, size);
	write(fd, &wh, sizeof(struct wave_hdr));
}

static int map_file(int fd, u8 **ptr, size_t *size)
{
	struct stat sb;

	fstat(fd, &sb);
	*size = sb.st_size;

    *ptr = (u8*)mmap(NULL, *size, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd, 0);
	if (*ptr == MAP_FAILED) {
		perror("mmap");
		return -1;
	}

	return 0;
}

static int read_packet(void *opaque, u8 *buf, int buf_size)
{
    struct audio_buffer *audio_buf = (audio_buffer*)opaque;

	buf_size = FFMIN(buf_size, audio_buf->size);

	/* copy internal buffer data to buf */
	memcpy(buf, audio_buf->ptr, buf_size);
	audio_buf->ptr += buf_size;
	audio_buf->size -= buf_size;

	return buf_size;
}

static void convert_frame(struct SwrContext *swr, AVCodecContext *codec,
			  AVFrame *frame, s16 **data, int *size, bool flush)
{
	int nr_samples;
	s64 delay;
	u8 *buffer;

	delay = swr_get_delay(swr, codec->sample_rate);
	nr_samples = av_rescale_rnd(delay + frame->nb_samples,
				    WAVE_SAMPLE_RATE, codec->sample_rate,
				    AV_ROUND_UP);
	av_samples_alloc(&buffer, NULL, 1, nr_samples, AV_SAMPLE_FMT_S16, 0);

	/*
	 * !flush is used to check if we are flushing any remaining
	 * conversion buffers...
	 */
	nr_samples = swr_convert(swr, &buffer, nr_samples,
				 !flush ? (const u8 **)frame->data : NULL,
				 !flush ? frame->nb_samples : 0);

    *data = (s16*)realloc(*data, (*size + nr_samples) * sizeof(s16));
	memcpy(*data + *size, buffer, nr_samples * sizeof(s16));
	*size += nr_samples;
	av_freep(&buffer);
}

static bool is_audio_stream(const AVStream *stream)
{
	if (stream->codecpar->codec_type == AVMEDIA_TYPE_AUDIO)
		return true;

	return false;
}

// Return non zero on error, 0 on success
// audio_buffer: input memory
// data: decoded output audio data (wav file)
// size: size of output data
static int decode_audio(struct audio_buffer *audio_buf, s16 **data, int *size)
{
    LOG("decode_audio: input size: %d\n", audio_buf->size);
	AVFormatContext *fmt_ctx;
	AVIOContext *avio_ctx;
	AVStream *stream;
	AVCodecContext *codec;
	AVPacket *packet;
	AVFrame *frame;
	struct SwrContext *swr;
	u8 *avio_ctx_buffer;
	unsigned int i;
	int stream_index = -1;
	int err;
    const size_t errbuffsize = 1024;
    char errbuff[errbuffsize];

    fmt_ctx = avformat_alloc_context();
    avio_ctx_buffer = (u8*)av_malloc(AVIO_CTX_BUF_SZ);
    LOG("Creating an avio context: AVIO_CTX_BUF_SZ=%d\n", AVIO_CTX_BUF_SZ);
    avio_ctx = avio_alloc_context(avio_ctx_buffer, AVIO_CTX_BUF_SZ, 0, audio_buf, &read_packet, NULL, NULL);
	fmt_ctx->pb = avio_ctx;

    // open the input stream and read header
	err = avformat_open_input(&fmt_ctx, NULL, NULL, NULL);
	if (err) {
        LOG("Could not read audio buffer: %d: %s\n", err, av_make_error_string(errbuff, errbuffsize, err));
        return err;
	}

	err = avformat_find_stream_info(fmt_ctx, NULL);
	if (err < 0) {
        LOG("Could not retrieve stream info from audio buffer: %d\n", err);
        return err;
	}

	for (i = 0; i < fmt_ctx->nb_streams; i++) {
		if (is_audio_stream(fmt_ctx->streams[i])) {
			stream_index = i;
			break;
		}
	}

	if (stream_index == -1) {
        LOG("Could not retrieve audio stream from buffer\n");
		return -1;
	}

	stream = fmt_ctx->streams[stream_index];
	codec = avcodec_alloc_context3(
			avcodec_find_decoder(stream->codecpar->codec_id));
	avcodec_parameters_to_context(codec, stream->codecpar);
	err = avcodec_open2(codec, avcodec_find_decoder(codec->codec_id),
							NULL);
	if (err) {
        LOG("Failed to open decoder for stream #%d in audio buffer\n", stream_index);
        return err;
	}

	/* prepare resampler */
	swr = swr_alloc();

#if LIBAVCODEC_VERSION_MAJOR > 60
	AVChannelLayout in_ch_layout = codec->ch_layout;
	AVChannelLayout out_ch_layout = AV_CHANNEL_LAYOUT_MONO;

	/* Set the source audio layout as-is */
	av_opt_set_chlayout(swr, "in_chlayout", &in_ch_layout, 0);
	av_opt_set_int(swr, "in_sample_rate", codec->sample_rate, 0);
	av_opt_set_sample_fmt(swr, "in_sample_fmt", codec->sample_fmt, 0);

	/* Convert it into 16khz Mono */
	av_opt_set_chlayout(swr, "out_chlayout", &out_ch_layout, 0);
	av_opt_set_int(swr, "out_sample_rate", WAVE_SAMPLE_RATE, 0);
	av_opt_set_sample_fmt(swr, "out_sample_fmt", AV_SAMPLE_FMT_S16, 0);
#else
	av_opt_set_int(swr, "in_channel_count", codec->channels, 0);
	av_opt_set_int(swr, "out_channel_count", 1, 0);
	av_opt_set_int(swr, "in_channel_layout", codec->channel_layout, 0);
	av_opt_set_int(swr, "out_channel_layout", AV_CH_LAYOUT_MONO, 0);
	av_opt_set_int(swr, "in_sample_rate", codec->sample_rate, 0);
	av_opt_set_int(swr, "out_sample_rate", WAVE_SAMPLE_RATE, 0);
	av_opt_set_sample_fmt(swr, "in_sample_fmt", codec->sample_fmt, 0);
	av_opt_set_sample_fmt(swr, "out_sample_fmt", AV_SAMPLE_FMT_S16, 0);
#endif

	swr_init(swr);
	if (!swr_is_initialized(swr)) {
        LOG("Resampler has not been properly initialized\n");
		return -1;
	}

	packet=av_packet_alloc();
	if (!packet) {
		LOG("Error allocating the packet\n");
		return -1;
	}
	frame = av_frame_alloc();
	if (!frame) {
        LOG("Error allocating the frame\n");
		return -1;
	}

	/* iterate through frames */
	*data = NULL;
	*size = 0;
	while (av_read_frame(fmt_ctx, packet) >= 0) {
		avcodec_send_packet(codec, packet);

		err = avcodec_receive_frame(codec, frame);
		if (err == AVERROR(EAGAIN))
			continue;

		convert_frame(swr, codec, frame, data, size, false);
	}
	/* Flush any remaining conversion buffers... */
	convert_frame(swr, codec, frame, data, size, true);

	av_packet_free(&packet);
	av_frame_free(&frame);
	swr_free(&swr);
    //avio_context_free(); // todo?
	avcodec_free_context(&codec);
	avformat_close_input(&fmt_ctx);
	avformat_free_context(fmt_ctx);

	if (avio_ctx) {
		av_freep(&avio_ctx->buffer);
		av_freep(&avio_ctx);
	}

	return 0;
}

// in mem decoding/conversion/resampling:
// ifname: input file path
// owav_data: in mem wav file. Can be forwarded as it to whisper/drwav
// return 0 on success
int ffmpeg_decode_audio(const std::string &ifname, std::vector<uint8_t>& owav_data) {
    LOG("ffmpeg_decode_audio: %s\n", ifname.c_str());
    int ifd = open(ifname.c_str(), O_RDONLY);
    if (ifd == -1) {
        fprintf(stderr, "Couldn't open input file %s\n", ifname.c_str());
        return -1;
    }
    u8 *ibuf = NULL;
    size_t ibuf_size;
    int err = map_file(ifd, &ibuf, &ibuf_size);
    if (err) {
        LOG("Couldn't map input file %s\n", ifname.c_str());
        return err;
    }
    LOG("Mapped input file: %s size: %d\n", ibuf, (int) ibuf_size);
    struct audio_buffer inaudio_buf;
    inaudio_buf.ptr = ibuf;
    inaudio_buf.size = ibuf_size;

    s16 *odata=NULL;
    int osize=0;

    err = decode_audio(&inaudio_buf, &odata, &osize);
    LOG("decode_audio returned %d \n", err);
    if (err != 0) {
        LOG("decode_audio failed\n");
        return err;
    }
    LOG("decode_audio output size: %d\n", osize);

    wave_hdr wh;
    const size_t outdatasize = osize * sizeof(s16);
    set_wave_hdr(wh, outdatasize);
    owav_data.resize(sizeof(wave_hdr) + outdatasize);
    // header:
    memcpy(owav_data.data(), &wh, sizeof(wave_hdr));
    // the data:
    memcpy(owav_data.data() + sizeof(wave_hdr), odata, osize* sizeof(s16));

    return 0;
}
