#ifdef WHISPER_COMMON_FFMPEG

#include <string>
#include <vector>
#include <cstdio>
#include <cstring>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswresample/swresample.h>
}

// Write a minimal WAV header into the output buffer.
// Returns the number of bytes written (44 for a standard PCM WAV header).
static size_t wav_header_write(uint8_t * buf, int num_channels, int sample_rate, int bits_per_sample, uint32_t data_size) {
    // RIFF header
    memcpy(buf, "RIFF", 4);
    uint32_t chunk_size = 36 + data_size;
    memcpy(buf + 4, &chunk_size, 4);
    memcpy(buf + 8, "WAVE", 4);

    // fmt subchunk
    memcpy(buf + 12, "fmt ", 4);
    uint32_t subchunk1_size = 16;
    memcpy(buf + 16, &subchunk1_size, 4);
    uint16_t audio_format = 1; // PCM
    memcpy(buf + 20, &audio_format, 2);
    memcpy(buf + 22, &num_channels, 2);
    memcpy(buf + 24, &sample_rate, 4);

    int bytes_per_sample = (bits_per_sample / 8) * num_channels;
    int byte_rate = sample_rate * bytes_per_sample;
    memcpy(buf + 28, &byte_rate, 4);
    memcpy(buf + 32, &bytes_per_sample, 2);
    memcpy(buf + 34, &bits_per_sample, 2);

    // data subchunk
    memcpy(buf + 36, "data", 4);
    memcpy(buf + 40, &data_size, 4);

    return 44;
}

bool ffmpeg_decode_audio(const std::string & ifname, std::vector<uint8_t> & wav_data, int out_sample_rate) {
    {
        const char * verbose = getenv("WHISPER_COMMON_FFMPEG_VERBOSE");
        if (verbose && strcmp(verbose, "2") == 0) {
            av_log_set_level(AV_LOG_DEBUG);
        } else if (verbose && strcmp(verbose, "1") == 0) {
            av_log_set_level(AV_LOG_VERBOSE);
        } else {
            av_log_set_level(AV_LOG_WARNING);
        }
    }

    AVFormatContext * fmt_ctx = nullptr;
    if (avformat_open_input(&fmt_ctx, ifname.c_str(), nullptr, nullptr) != 0) {
        fprintf(stderr, "error: failed to open input file '%s'\n", ifname.c_str());
        return true;
    }

    if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
        fprintf(stderr, "error: failed to find stream information\n");
        avformat_close_input(&fmt_ctx);
        return true;
    }

    // Find the first audio stream
    int audio_stream_idx = -1;
    for (unsigned int i = 0; i < fmt_ctx->nb_streams; i++) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            audio_stream_idx = i;
            break;
        }
    }

    if (audio_stream_idx == -1) {
        fprintf(stderr, "error: failed to find an audio stream in '%s'\n", ifname.c_str());
        avformat_close_input(&fmt_ctx);
        return true;
    }

    AVStream * audio_stream = fmt_ctx->streams[audio_stream_idx];

    // Open the decoder
    const AVCodec * codec = avcodec_find_decoder(audio_stream->codecpar->codec_id);
    if (!codec) {
        fprintf(stderr, "error: failed to find decoder for codec id %d\n", audio_stream->codecpar->codec_id);
        avformat_close_input(&fmt_ctx);
        return true;
    }

    AVCodecContext * codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        fprintf(stderr, "error: failed to allocate codec context\n");
        avformat_close_input(&fmt_ctx);
        return true;
    }

    if (avcodec_parameters_to_context(codec_ctx, audio_stream->codecpar) < 0) {
        fprintf(stderr, "error: failed to copy codec parameters to context\n");
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&fmt_ctx);
        return true;
    }

    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        fprintf(stderr, "error: failed to open codec\n");
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&fmt_ctx);
        return true;
    }

    // Setup resampler: convert to 16-bit signed PCM, mono, 16000 Hz
    const enum AVSampleFormat out_sample_fmt = AV_SAMPLE_FMT_S16;

    AVChannelLayout out_ch_layout = AV_CHANNEL_LAYOUT_MONO;

    SwrContext * swr_ctx = nullptr;
    if (swr_alloc_set_opts2(&swr_ctx, &out_ch_layout, out_sample_fmt, out_sample_rate,
                            &codec_ctx->ch_layout, codec_ctx->sample_fmt, codec_ctx->sample_rate,
                            0, nullptr) < 0) {
        fprintf(stderr, "error: failed to allocate swr context\n");
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&fmt_ctx);
        return true;
    }

    if (swr_init(swr_ctx) < 0) {
        fprintf(stderr, "error: failed to initialize swr context\n");
        swr_free(&swr_ctx);
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&fmt_ctx);
        return true;
    }

    // Decode and resample
    AVPacket * packet = av_packet_alloc();
    AVFrame * frame = av_frame_alloc();

    // Buffer to collect resampled output
    std::vector<int16_t> pcm_data;

    // Max output samples per swr_convert call
    const int max_out_samples = 16 * 1024;
    std::vector<int16_t> out_buffer(max_out_samples);

    while (av_read_frame(fmt_ctx, packet) >= 0) {
        if (packet->stream_index != audio_stream_idx) {
            av_packet_unref(packet);
            continue;
        }

        int ret = avcodec_send_packet(codec_ctx, packet);
        av_packet_unref(packet);

        if (ret < 0) {
            continue;
        }

        while (ret >= 0) {
            ret = avcodec_receive_frame(codec_ctx, frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                break;
            }
            if (ret < 0) {
                break;
            }

            // Resample
            int out_samples = av_rescale_rnd(swr_get_delay(swr_ctx, out_sample_rate) + frame->nb_samples,
                                              out_sample_rate, out_sample_rate, AV_ROUND_UP);
            if (out_samples > (int)out_buffer.size()) {
                out_buffer.resize(out_samples);
            }

            const uint8_t * in_data[16] = {0};
            for (int p = 0; p < (int)codec_ctx->ch_layout.nb_channels && p < 16; p++) {
                in_data[p] = frame->data[p];
            }
            uint8_t * out_data[16] = {0};
            out_data[0] = (uint8_t *)out_buffer.data();

            int got_samples = swr_convert(swr_ctx, out_data, out_samples, in_data, frame->nb_samples);
            if (got_samples > 0) {
                pcm_data.insert(pcm_data.end(), out_buffer.begin(), out_buffer.begin() + got_samples);
            }
        }
    }

    // Flush the decoder
    avcodec_send_packet(codec_ctx, nullptr);
    while (avcodec_receive_frame(codec_ctx, frame) >= 0) {
        int out_samples = av_rescale_rnd(swr_get_delay(swr_ctx, out_sample_rate) + frame->nb_samples,
                                          out_sample_rate, out_sample_rate, AV_ROUND_UP);
        if (out_samples > (int)out_buffer.size()) {
            out_buffer.resize(out_samples);
        }
        const uint8_t * in_data[16] = {0};
        for (int p = 0; p < (int)codec_ctx->ch_layout.nb_channels && p < 16; p++) {
            in_data[p] = frame->data[p];
        }
        uint8_t * out_data[16] = {0};
        out_data[0] = (uint8_t *)out_buffer.data();

        int got_samples = swr_convert(swr_ctx, out_data, out_samples, in_data, frame->nb_samples);
        if (got_samples > 0) {
            pcm_data.insert(pcm_data.end(), out_buffer.begin(), out_buffer.begin() + got_samples);
        }
    }

    // Flush the resampler
    uint8_t * out_data[16] = {0};
    out_data[0] = (uint8_t *)out_buffer.data();
    int flush_samples = swr_convert(swr_ctx, out_data, max_out_samples, nullptr, 0);
    if (flush_samples > 0) {
        pcm_data.insert(pcm_data.end(), out_buffer.begin(), out_buffer.begin() + flush_samples);
    }

    // Build WAV output
    uint32_t data_size = pcm_data.size() * sizeof(int16_t);
    wav_data.resize(44 + data_size);

    wav_header_write(wav_data.data(), 1, out_sample_rate, 16, data_size);
    memcpy(wav_data.data() + 44, pcm_data.data(), data_size);

    // Cleanup
    av_frame_free(&frame);
    av_packet_free(&packet);
    swr_free(&swr_ctx);
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&fmt_ctx);

    return false; // success
}

#endif // WHISPER_COMMON_FFMPEG
