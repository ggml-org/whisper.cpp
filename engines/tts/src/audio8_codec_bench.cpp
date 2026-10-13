// Codec reconstruction only: no text LM, generation, gain normalization or lag search.
#include "audio8/codec_bench_cli.h"
#include "audio8/internal.h"
#include "voice_features.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace codec = tts_cpp::audio8::detail;
namespace {
void require(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error(message);
}
struct owned_codec {
    codec::codec_model model;
    ~owned_codec() { codec::free_codec(model); }
};

void check_headers(const codec::codec_header & enc, const codec::codec_header & dec) {
    require(enc.part == "encoder" && dec.part == "decoder", "wrong codec half supplied");
    const auto & a = enc.hp;
    const auto & b = dec.hp;
    require(a.num_codebooks == a.residual_codebooks + 1 &&
            b.num_codebooks == b.residual_codebooks + 1, "invalid codec codebook count");
    require(a.sample_rate == b.sample_rate && a.frame_size == b.frame_size &&
            a.num_codebooks == b.num_codebooks && a.codebook_dim == b.codebook_dim &&
            a.latent_dim == b.latent_dim && a.semantic_codebook_size == b.semantic_codebook_size &&
            a.residual_codebook_size == b.residual_codebook_size &&
            a.residual_codebooks == b.residual_codebooks, "incompatible codec encoder and decoder");
    require(a.sample_rate > 0 && a.frame_size > 0, "invalid codec rate or frame size");
}

void write_wav(const std::string & path, const std::vector<float> & pcm, int rate) {
    require(pcm.size() <= (UINT32_MAX - 36) / 2, "output exceeds WAV size limit");
    std::vector<unsigned char> bytes;
    bytes.reserve(44 + pcm.size() * 2);
    auto tag = [&](const char * s) { bytes.insert(bytes.end(), s, s + 4); };
    auto le = [&](uint32_t n, int width) {
        for (int i = 0; i < width; ++i) bytes.push_back(static_cast<unsigned char>(n >> (8 * i)));
    };
    tag("RIFF"); le(36 + static_cast<uint32_t>(pcm.size() * 2), 4); tag("WAVE");
    tag("fmt "); le(16, 4); le(1, 2); le(1, 2); le(rate, 4); le(static_cast<uint32_t>(rate) * 2, 4);
    le(2, 2); le(16, 2); tag("data"); le(static_cast<uint32_t>(pcm.size() * 2), 4);
    for (float sample : pcm) {
        require(std::isfinite(sample), "codec produced nonfinite audio");
        const auto value = static_cast<int16_t>(std::lrintf(std::max(-1.0f, std::min(1.0f, sample)) * 32767.0f));
        le(static_cast<uint16_t>(value), 2);
    }
    std::FILE * f = std::fopen(path.c_str(), "wb");
    require(f != nullptr, "cannot open output WAV: " + path);
    const bool written = std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
    const bool closed = std::fclose(f) == 0;
    require(written && closed, "failed writing output WAV: " + path);
}

void write_f32(const std::string & path, const std::vector<float> & pcm) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                  "raw PCM requires IEEE float32");
    std::vector<unsigned char> bytes;
    bytes.reserve(pcm.size() * sizeof(float));
    for (float sample : pcm) {
        uint32_t bits;
        std::memcpy(&bits, &sample, sizeof(bits));
        for (int i = 0; i < 4; ++i) bytes.push_back(static_cast<unsigned char>(bits >> (8 * i)));
    }
    std::FILE * f = std::fopen(path.c_str(), "wb");
    require(f != nullptr, "cannot open output F32: " + path);
    const bool written = std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
    const bool closed = std::fclose(f) == 0;
    require(written && closed, "failed writing output F32: " + path);
}

void check_pcm(const std::vector<float> & pcm, size_t expected) {
    require(pcm.size() == expected, "unexpected decoder output length");
    for (float sample : pcm) require(std::isfinite(sample), "codec produced nonfinite audio");
}

int run_decode_benchmark(const audio8_codec_bench::options & opts) {
    codec::codec_header header;
    std::string error;
    if (!codec::peek_codec_header(opts.decoder, header, &error)) throw std::runtime_error(error);
    const auto & hp = header.hp;
    require(header.part == "decoder", "decode mode requires a decoder codec half");
    require(hp.sample_rate > 0 && hp.frame_size > 0 && hp.num_codebooks > 0 &&
            hp.residual_codebooks == hp.num_codebooks - 1 && hp.codebook_dim > 0 && hp.latent_dim > 0,
            "invalid decoder codec geometry");
    require(opts.codebooks == 0 || opts.codebooks == hp.num_codebooks,
            "--codebooks does not match decoder metadata");
    require(opts.frames <= hp.max_frames && opts.frames <= INT_MAX / hp.frame_size,
            "--frames exceeds decoder frame or output size limit");
    const auto frame_codes = audio8_codec_bench::read_codes(opts.codes, opts.frames, hp.num_codebooks, opts.codes_format);
    audio8_codec_bench::validate_codes(frame_codes, opts.frames, hp.num_codebooks,
                                      hp.semantic_codebook_size, hp.residual_codebook_size);
    const auto codes = audio8_codec_bench::decoder_rows(frame_codes, opts.frames, hp.num_codebooks);
    const size_t expected = static_cast<size_t>(opts.frames) * hp.frame_size;
    require(expected <= std::numeric_limits<size_t>::max() / sizeof(float), "decoder PCM byte size overflows");
    owned_codec decoder;
    if (!codec::load_codec(opts.decoder, opts.backend == "cpu" ? 0 : 99,
                           opts.backend, decoder.model, &error)) throw std::runtime_error(error);
    std::printf("decoder backend: %s (requested=%s)\n", ggml_backend_name(decoder.model.backend), opts.backend.c_str());
    std::vector<float> pcm;
    for (int i = 0; i < opts.warmup; ++i) {
        if (!codec::decode_codes(decoder.model, codes.data(), opts.frames, opts.threads, {}, pcm, &error))
            throw std::runtime_error(error);
        check_pcm(pcm, expected);
    }
    std::vector<double> times;
    times.reserve(opts.repeat);
    std::vector<float> first_pcm;
    bool pcm_bitwise_stable = true;
    double pcm_max_abs_diff = 0;
    for (int i = 0; i < opts.repeat; ++i) {
        codec::decode_timing timing;
        const auto start = std::chrono::steady_clock::now();
        const bool ok = codec::decode_codes(decoder.model, codes.data(), opts.frames, opts.threads, {}, pcm, &error,
                                            nullptr, &timing);
        const auto end = std::chrono::steady_clock::now();
        if (!ok) throw std::runtime_error(error);
        const double ms = std::chrono::duration<double, std::milli>(end - start).count();
        require(std::isfinite(ms) && ms >= 0, "invalid decoder timing");
        times.push_back(ms);
        check_pcm(pcm, expected);
        if (first_pcm.empty()) {
            first_pcm = pcm;
        } else if (std::memcmp(first_pcm.data(), pcm.data(), expected * sizeof(float)) != 0) {
            pcm_bitwise_stable = false;
            for (size_t sample = 0; sample < expected; ++sample)
                pcm_max_abs_diff = std::max(pcm_max_abs_diff,
                    std::abs(static_cast<double>(pcm[sample]) - first_pcm[sample]));
        }
        // The scheduler bundle is created only when a graph takes its route.
        // Its presence detects that route, not the count of CPU-routed nodes.
        std::printf("decode iteration: repeat=%d decoder_ms=%.6f synthesis_backend=%s "
                    "scheduler_initialized=%s cpu_fallback_available=%s\n", i + 1, ms,
                    timing.synthesis_backend.c_str(), decoder.model.sched.sched ? "true" : "false",
                    decoder.model.sched.cpu_backend ? "true" : "false");
    }
    // The timed region is exactly decode_codes: graph construction/allocation,
    // execution and PCM readback. Loading, codes IO, checks and output IO are excluded.
    if (!opts.output_f32.empty()) write_f32(opts.output_f32, pcm);
    if (!opts.output.empty()) {
        // The first output now exists, so equivalent also detects aliases
        // under filesystem-specific Unicode case/normalization rules.
        if (!opts.output_f32.empty()) audio8_codec_bench::validate_output_paths(opts);
        write_wav(opts.output, pcm, hp.sample_rate);
    }
    const double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double variance = 0;
    for (double ms : times) variance += (ms - mean) * (ms - mean);
    std::sort(times.begin(), times.end());
    const size_t mid = times.size() / 2;
    const double median = times.size() % 2 ? times[mid] : (times[mid - 1] + times[mid]) / 2;
    const auto bounds = std::minmax_element(pcm.begin(), pcm.end());
    double squares = 0;
    for (float sample : pcm) squares += static_cast<double>(sample) * sample;
    std::printf("codec decode: frames=%d codebooks=%d codebook_dim=%d latent_dim=%d samples=%zu sample_rate=%d "
                "audio_seconds=%.6f finite=true pcm_min=%.9g pcm_max=%.9g pcm_rms=%.9g warmup=%d repeats=%d threads=%d\n",
                opts.frames, hp.num_codebooks, hp.codebook_dim, hp.latent_dim, pcm.size(), hp.sample_rate,
                static_cast<double>(pcm.size()) / hp.sample_rate, *bounds.first, *bounds.second,
                std::sqrt(squares / pcm.size()), opts.warmup, opts.repeat, opts.threads);
    std::printf("decoder timing: min_ms=%.6f mean_ms=%.6f median_ms=%.6f max_ms=%.6f stddev_ms=%.6f\n",
                times.front(), mean, median, times.back(), std::sqrt(variance / times.size()));
    std::printf("decoder repetitions: pcm_bitwise_stable=%s pcm_max_abs_diff=%.9g\n",
                pcm_bitwise_stable ? "true" : "false", pcm_max_abs_diff);
    return 0;
}
}

int main(int argc, char ** argv) {
    try {
        const auto opts = audio8_codec_bench::parse(argc, argv);
        if (opts.help) {
            std::puts("audio8-codec-bench --codec-encoder FILE --codec-decoder FILE --in WAV --out WAV [--threads N] [--n-gpu-layers N]\n"
                      "  Roundtrip (default --mode roundtrip): mono PCM16 at input rate and duration; removes codec end padding.\n"
                      "  Input channels are averaged to mono; sinc resampling and PCM16 clipping are included.\n"
                      "audio8-codec-bench --codes FILE --codec-decoder FILE --codec-backend cpu|hexagon|opencl --frames N\n"
                      "  [--mode decode] [--codes-format csv|i32] [--codebooks N] [--threads N]\n"
                      "  [--warmup N] [--repeat N] --out WAV and/or --out-f32 FILE\n"
                      "  Decode mode: default CSV matches audio8-cli --dump-codes (one frame per row).\n"
                      "  i32 is headerless little-endian int32; both formats are frame-major, semantic codebook first.\n"
                      "  Codebooks are inferred from decoder metadata; an explicit count must match.\n"
                      "  Defaults: threads=4, warmup=1, repeat=5. Warmup may be zero; frames/repeat must be positive.\n"
                      "  Times decode_codes including graph setup and PCM readback; excludes model load, file IO and checks.\n"
                      "  Outputs native-rate mono PCM16 WAV and/or unclipped little-endian IEEE float32 PCM.\n"
                      "  Aliases: --backend, --warmups, --repeats.");
            return 0;
        }
        audio8_codec_bench::validate_output_paths(opts);
        if (opts.mode == "decode") return run_decode_benchmark(opts);
        std::vector<float> pcm;
        int rate = 0;
        require(wav_load(opts.input, pcm, rate) && !pcm.empty(), "cannot read nonempty input WAV");
        require(rate >= 8000 && rate <= 192000, "input rate must be in [8000, 192000]");
        for (float v : pcm) require(std::isfinite(v), "input WAV contains nonfinite samples");
        const size_t input_samples = pcm.size();
        codec::codec_header enc_header, dec_header;
        std::string error;
        require(codec::peek_codec_header(opts.encoder, enc_header, &error), error);
        require(codec::peek_codec_header(opts.decoder, dec_header, &error), error);
        check_headers(enc_header, dec_header);
        const int native_rate = enc_header.hp.sample_rate;
        // resample_sinc rounds down. Retain the full input duration by adding
        // at most one zero sample at the end before the codec's frame padding.
        const size_t native_samples = static_cast<size_t>(std::ceil(
            static_cast<double>(input_samples) * native_rate / rate));
        require(native_samples <= static_cast<size_t>(INT_MAX - enc_header.hp.frame_size), "input too long for codec");
        pcm = resample_sinc(pcm, rate, native_rate);
        const size_t resample_padding = native_samples - pcm.size();
        require(resample_padding <= 1, "unexpected input resampler length");
        pcm.resize(native_samples, 0.0f);
        owned_codec encoder, decoder;
        require(codec::load_codec(opts.encoder, opts.gpu_layers, /*backend=*/"", encoder.model, &error), error);
        require(codec::load_codec(opts.decoder, opts.gpu_layers, /*backend=*/"", decoder.model, &error), error);
        std::printf("encoder backend: %s\ndecoder backend: %s\n", ggml_backend_name(encoder.model.backend), ggml_backend_name(decoder.model.backend));
        std::vector<int32_t> codes;
        int frames = 0;
        require(codec::encode_audio(encoder.model, pcm.data(), static_cast<int>(pcm.size()),
                                   opts.threads, {}, codes, frames, &error), error);
        std::vector<float> decoded;
        require(codec::decode_codes(decoder.model, codes.data(), frames, opts.threads, {}, decoded, &error), error);
        const size_t expected = static_cast<size_t>(frames) * dec_header.hp.frame_size;
        require(decoded.size() == expected && expected >= native_samples &&
                expected - native_samples < static_cast<size_t>(dec_header.hp.frame_size), "unexpected codec output length");
        // Keep the known zero-padded tail through resampling to avoid inserting
        // a new filter boundary into retained speech, then remove only that tail.
        decoded = resample_sinc(decoded, native_rate, rate);
        require(decoded.size() >= input_samples, "resampled output is shorter than input");
        decoded.resize(input_samples);
        write_wav(opts.output, decoded, rate);
        std::printf("codec roundtrip: input_samples=%zu sample_rate=%d native_samples=%zu native_rate=%d frames=%d codec_end_padding=%zu resample_end_padding=%zu output_samples=%zu\n",
                    input_samples, rate, native_samples, native_rate, frames, expected - native_samples, resample_padding, decoded.size());
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "audio8-codec-bench: %s\n", e.what());
        return 1;
    }
}
