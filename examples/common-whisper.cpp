#define _USE_MATH_DEFINES // for M_PI

#include "common-whisper.h"

#include "common.h"

#include "whisper.h"

// third-party utilities
// use your favorite implementations
#define STB_VORBIS_HEADER_ONLY
#include "stb_vorbis.c"    /* Enables Vorbis decoding. */

#ifdef _WIN32
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#endif

#define MA_NO_DEVICE_IO
#define MA_NO_THREADING
#define MA_NO_ENCODING
#define MA_NO_GENERATION
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include <cstring>
#include <fstream>

#ifdef WHISPER_COMMON_FFMPEG
// as implemented in ffmpeg-trancode.cpp only embedded in common lib if whisper built with ffmpeg support
extern bool ffmpeg_decode_audio(const std::string & ifname, std::vector<uint8_t> & wav_data, int out_sample_rate = WHISPER_SAMPLE_RATE);
#endif

// extract f32 PCM frames from an initialized decoder, downmix to mono and keep the stereo split
static bool read_audio_from_decoder(ma_decoder & decoder, std::vector<float> & pcmf32, std::vector<std::vector<float>> & pcmf32s, bool stereo) {
    ma_result result;
    ma_uint64 frame_count;
    ma_uint64 frames_read;

    if ((result = ma_decoder_get_length_in_pcm_frames(&decoder, &frame_count)) != MA_SUCCESS) {
        fprintf(stderr, "error: failed to retrieve the length of the audio data (%s)\n", ma_result_description(result));
        return false;
    }

    pcmf32.resize(stereo ? frame_count*2 : frame_count);

    if ((result = ma_decoder_read_pcm_frames(&decoder, pcmf32.data(), frame_count, &frames_read)) != MA_SUCCESS) {
        fprintf(stderr, "error: failed to read the frames of the audio data (%s)\n", ma_result_description(result));
        return false;
    }

    if (stereo) {
        std::vector<float> stereo_data = pcmf32;
        pcmf32.resize(frame_count);
        for (uint64_t i = 0; i < frame_count; i++) {
            pcmf32[i] = (stereo_data[2*i] + stereo_data[2*i + 1]);
        }
        pcmf32s.resize(2);
        pcmf32s[0].resize(frame_count);
        pcmf32s[1].resize(frame_count);
        for (uint64_t i = 0; i < frame_count; i++) {
            pcmf32s[0][i] = stereo_data[2*i];
            pcmf32s[1][i] = stereo_data[2*i + 1];
        }
    }

    return true;
}

bool read_audio_data(const std::string & fname, std::vector<float> & pcmf32, std::vector<std::vector<float>> & pcmf32s, bool stereo) {
    std::vector<uint8_t> audio_data; // used for pipe input from stdin or ffmpeg decoding output

    ma_result result;
    ma_decoder_config decoder_config;

    struct decoder_guard {
        ma_decoder decoder;
        bool initialized = false;
        ma_decoder * operator&() { return &decoder; }
        ~decoder_guard() {
            if (initialized) {
                ma_decoder_uninit(&decoder);
            }
        }
    };
    decoder_guard decoder{};

    decoder_config = ma_decoder_config_init(ma_format_f32, stereo ? 2 : 1, WHISPER_SAMPLE_RATE);

    if (fname == "-") {
#ifdef _WIN32
        _setmode(_fileno(stdin), _O_BINARY);
#endif

        uint8_t buf[1024];
        while (true)
        {
            const size_t n = fread(buf, 1, sizeof(buf), stdin);
            if (n == 0) {
                break;
            }
            audio_data.insert(audio_data.end(), buf, buf + n);
        }

        result = ma_decoder_init_memory(audio_data.data(), audio_data.size(), &decoder_config, &decoder);
        if (result != MA_SUCCESS) {
            fprintf(stderr, "%s: failed to open audio data from stdin (%s)\n", __func__, ma_result_description(result));
            return false;
        }
        decoder.initialized = true;

        fprintf(stderr, "%s: read %zu bytes from stdin\n", __func__, audio_data.size());
    } else {
        fprintf(stderr, "%s: reading audio data from '%s' ...\n", __func__, fname.c_str());

        // first try miniaudio. if it fails (or skipped) - try ffmpeg
        {
            const char * skip = getenv("WHISPER_COMMON_MINIAUDIO_SKIP");
            if (!skip || strlen(skip) == 0 || strcmp(skip, "0") == 0) {
                fprintf(stderr, "%s: trying to decode with miniaudio\n", __func__);

                result = ma_decoder_init_file(fname.c_str(), &decoder_config, &decoder);
                if (result == MA_SUCCESS) {
                    decoder.initialized = true;
                }
            } else {
                fprintf(stderr, "%s: skipping miniaudio\n", __func__);
            }
        }

#if defined(WHISPER_COMMON_FFMPEG)
        if (!decoder.initialized) {
            fprintf(stderr, "%s: trying to decode with ffmpeg\n", __func__);

            if (ffmpeg_decode_audio(fname, audio_data) != 0) {
                fprintf(stderr, "%s: failed to ffmpeg decode\n", __func__);
                return false;
            }
            result = ma_decoder_init_memory(audio_data.data(), audio_data.size(), &decoder_config, &decoder);
            if (result != MA_SUCCESS) {
                fprintf(stderr, "%s: failed to read audio data as wav (%s)\n", __func__, ma_result_description(result));
                return false;
            }
            decoder.initialized = true;
        }
#endif

        if (!decoder.initialized) {
            fprintf(stderr, "%s: failed to read audio data\n", __func__);
            return false;
        }
    }

    return read_audio_from_decoder(decoder.decoder, pcmf32, pcmf32s, stereo);
}

// decode audio bytes already held in memory
bool read_audio_data(const char * buffer, size_t buffer_size, std::vector<float> & pcmf32, std::vector<std::vector<float>> & pcmf32s, bool stereo) {
    ma_decoder_config decoder_config = ma_decoder_config_init(ma_format_f32, stereo ? 2 : 1, WHISPER_SAMPLE_RATE);
    ma_decoder decoder;

    if (ma_decoder_init_memory(buffer, buffer_size, &decoder_config, &decoder) != MA_SUCCESS) {
        fprintf(stderr, "error: failed to decode audio data from memory buffer\n");
        return false;
    }

    bool ok = read_audio_from_decoder(decoder, pcmf32, pcmf32s, stereo);
    ma_decoder_uninit(&decoder);
    return ok;
}

//  500 -> 00:05.000
// 6000 -> 01:00.000
std::string to_timestamp(int64_t t, bool comma) {
    int64_t msec = t * 10;
    int64_t hr = msec / (1000 * 60 * 60);
    msec = msec - hr * (1000 * 60 * 60);
    int64_t min = msec / (1000 * 60);
    msec = msec - min * (1000 * 60);
    int64_t sec = msec / 1000;
    msec = msec - sec * 1000;

    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d%s%03d", (int) hr, (int) min, (int) sec, comma ? "," : ".", (int) msec);

    return std::string(buf);
}

int timestamp_to_sample(int64_t t, int n_samples, int whisper_sample_rate) {
    return std::max(0, std::min((int) n_samples - 1, (int) ((t*whisper_sample_rate)/100)));
}

int utf8_trailing_bytes_needed(const std::string & s) {
    const int n = (int) s.size();
    int i = n - 1;
    while (i >= 0 && ((unsigned char) s[i] & 0xC0) == 0x80) {
        --i;
    }
    if (i < 0) {
        return 0;
    }

    const unsigned char c = (unsigned char) s[i];
    int expected;
    if ((c & 0x80) == 0x00) {
        expected = 1;
    } else if ((c & 0xE0) == 0xC0) {
        expected = 2;
    } else if ((c & 0xF0) == 0xE0) {
        expected = 3;
    } else if ((c & 0xF8) == 0xF0) {
        expected = 4;
    } else {
        return 0;
    }

    const int have = n - i;
    return have >= expected ? 0 : (expected - have);
}

bool speak_with_file(const std::string & command, const std::string & text, const std::string & path, int voice_id) {
    std::ofstream speak_file(path.c_str());
    if (speak_file.fail()) {
        fprintf(stderr, "%s: failed to open speak_file\n", __func__);
        return false;
    } else {
        speak_file.write(text.c_str(), text.size());
        speak_file.close();
        int ret = system((command + " " + std::to_string(voice_id) + " " + path).c_str());
        if (ret != 0) {
            fprintf(stderr, "%s: failed to speak\n", __func__);
            return false;
        }
    }
    return true;
}

#undef STB_VORBIS_HEADER_ONLY
#include "stb_vorbis.c"
