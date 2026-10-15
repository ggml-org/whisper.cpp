#include "gguf_fixtures.h"

#include "moss/codec.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

using tts_cpp::moss::detail::Codec;
using tts_cpp::moss::detail::CodecAttention;
using tts_cpp::moss::detail::decode_after_context;
using tts_cpp::moss::detail::decode_in_windows;
using tts_cpp::moss::detail::decode_segments;
using namespace moss_fixtures;

namespace {

constexpr uint32_t WEIGHT_SEED = 20260923;
constexpr uint32_t CODE_SEED = 7;
constexpr int LONG_FRAMES = 40;
constexpr int SHORT_FRAMES = 11;
constexpr int WINDOW_FRAMES = 6;
constexpr float TOLERANCE = 1e-4f;
constexpr float SIGNAL_AMPLITUDE = 0.3f;
constexpr float SIGNAL_STEP = 0.37f;
const std::vector<int> CHUNK_PATTERN = {3, 5, 1, 7, 2};

int failures = 0;

void check(bool condition, const std::string & label) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", label.c_str());
        failures++;
    }
}

std::vector<int32_t> random_codes(int frames, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int32_t> draw(0, CODE_SIZE - 1);
    std::vector<int32_t> codes((size_t) frames * N_VQ);
    for (int32_t & code : codes) {
        code = draw(rng);
    }
    return codes;
}

std::vector<int32_t> frame_slice(const std::vector<int32_t> & codes, int first, int count) {
    return std::vector<int32_t>(codes.begin() + (std::ptrdiff_t) first * N_VQ,
            codes.begin() + (std::ptrdiff_t) (first + count) * N_VQ);
}

float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        return INFINITY;
    }
    float worst = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        worst = std::max(worst, std::fabs(a[i] - b[i]));
    }
    return worst;
}

float peak(const std::vector<float> & pcm) {
    float best = 0.0f;
    for (float sample : pcm) {
        best = std::max(best, std::fabs(sample));
    }
    return best;
}

std::vector<float> stream_in_chunks(Codec & codec, const std::vector<int32_t> & codes) {
    const int frames = (int) (codes.size() / N_VQ);
    std::vector<float> pcm;
    codec.begin_decode_stream();
    size_t turn = 0;
    for (int first = 0; first < frames; ++turn) {
        const int count = std::min(CHUNK_PATTERN[turn % CHUNK_PATTERN.size()], frames - first);
        const std::vector<float> piece = codec.decode_stream(frame_slice(codes, first, count));
        pcm.insert(pcm.end(), piece.begin(), piece.end());
        first += count;
    }
    return pcm;
}

void test_incremental_matches_full_decode(Codec & codec) {
    const std::vector<int32_t> codes = random_codes(LONG_FRAMES, CODE_SEED);
    const std::vector<float> full = codec.decode(codes);
    const std::vector<float> streamed = stream_in_chunks(codec, codes);
    check(peak(full) > 0.0f, "random weights produce audible output");
    check(streamed.size() == full.size(), "incremental decode emits every sample");
    check(max_abs_diff(streamed, full) < TOLERANCE,
            "incremental decode reproduces the full decode past the attention context");
}

void test_windows_match_full_decode(Codec & codec) {
    const std::vector<int32_t> codes = random_codes(LONG_FRAMES, CODE_SEED + 1);
    check(max_abs_diff(decode_in_windows(codec, codes, WINDOW_FRAMES), codec.decode(codes)) < TOLERANCE,
            "windowed batch decode matches a single decode");
}

void test_segments_decode_independently(Codec & codec) {
    const std::vector<int32_t> first = random_codes(SHORT_FRAMES, CODE_SEED + 2);
    const std::vector<int32_t> second = random_codes(SHORT_FRAMES, CODE_SEED + 3);
    std::vector<float> expected = codec.decode(first);
    const std::vector<float> second_alone = codec.decode(second);
    expected.insert(expected.end(), second_alone.begin(), second_alone.end());
    check(max_abs_diff(decode_segments(codec, {first, second}, LONG_FRAMES), expected) < TOLERANCE,
            "each segment starts from a fresh codec context");
    check(max_abs_diff(decode_segments(codec, {first, second}, WINDOW_FRAMES), expected) < TOLERANCE,
            "segments stay independent when decoded in windows");

    std::vector<int32_t> joined = first;
    joined.insert(joined.end(), second.begin(), second.end());
    check(max_abs_diff(codec.decode(joined), expected) > TOLERANCE,
            "decoding the concatenation would let the second segment attend to the first");
}

void test_restarted_stream_forgets_history(Codec & codec) {
    const std::vector<int32_t> first = random_codes(SHORT_FRAMES, CODE_SEED + 4);
    const std::vector<int32_t> second = random_codes(SHORT_FRAMES, CODE_SEED + 5);
    stream_in_chunks(codec, first);
    check(max_abs_diff(stream_in_chunks(codec, second), codec.decode(second)) < TOLERANCE,
            "begin_decode_stream discards the previous stream");
}

void test_stream_work_is_linear(Codec & codec) {
    const std::vector<int32_t> codes = random_codes(LONG_FRAMES, CODE_SEED + 6);
    const int64_t before = codec.frames_decoded();
    stream_in_chunks(codec, codes);
    check(codec.frames_decoded() - before == LONG_FRAMES,
            "every frame is decoded exactly once while streaming");
}

void test_context_is_decoded_then_trimmed(Codec & codec) {
    const std::vector<int32_t> context = random_codes(SHORT_FRAMES, CODE_SEED + 7);
    const std::vector<int32_t> generated = random_codes(SHORT_FRAMES, CODE_SEED + 8);
    std::vector<int32_t> joined = context;
    joined.insert(joined.end(), generated.begin(), generated.end());
    std::vector<float> expected = codec.decode(joined);
    expected.erase(expected.begin(),
            expected.begin() + (std::ptrdiff_t) SHORT_FRAMES * codec.samples_per_frame());

    check(max_abs_diff(decode_after_context(codec, {joined}, SHORT_FRAMES, LONG_FRAMES), expected) < TOLERANCE,
            "the generated suffix keeps the reference as causal history");
    check(max_abs_diff(decode_after_context(codec, {joined}, SHORT_FRAMES, WINDOW_FRAMES), expected) < TOLERANCE,
            "windowed decoding keeps the same history");
    check(max_abs_diff(codec.decode(generated), expected) > TOLERANCE,
            "decoding the suffix alone would lose the reference history");
}

void test_fused_attention_matches_unfused(const std::filesystem::path & path) {
    Codec automatic(path.string(), {}, 1);
    Codec unfused(path.string(), {}, 1, false, CodecAttention::Unfused);
    Codec fused(path.string(), {}, 1, false, CodecAttention::Fused);
    check(!automatic.fused_attention(), "the CPU keeps unfused codec attention");
    check(!unfused.fused_attention() && fused.fused_attention(), "codec attention layouts can be forced");
    const std::vector<int32_t> codes = random_codes(LONG_FRAMES, CODE_SEED);
    check(max_abs_diff(unfused.decode(codes), fused.decode(codes)) < TOLERANCE,
            "fused banded attention decodes like the unfused bands");
    check(max_abs_diff(stream_in_chunks(unfused, codes), stream_in_chunks(fused, codes)) < TOLERANCE,
            "fused attention streams through the same cached history");
}

std::vector<float> sine(size_t samples) {
    std::vector<float> pcm(samples);
    for (size_t i = 0; i < pcm.size(); ++i) {
        pcm[i] = SIGNAL_AMPLITUDE * std::sin(SIGNAL_STEP * (float) i);
    }
    return pcm;
}

void test_fused_attention_encodes_like_unfused() {
    std::mt19937 rng(WEIGHT_SEED);
    const auto path = temp_gguf("codec-fused-encoder");
    {
        GgufBuilder b;
        b.random = &rng;
        add_encoder_meta(b.file);
        add_encoder_tensors(b);
        b.write(path);
    }
    Codec unfused(path.string(), {}, 1, false, CodecAttention::Unfused);
    Codec fused(path.string(), {}, 1, false, CodecAttention::Fused);
    const std::vector<float> pcm = sine((size_t) LONG_FRAMES * OUT_DIM);
    check(unfused.encode(pcm) == fused.encode(pcm), "fused attention encodes the same codes");
    std::filesystem::remove(path);
}

void test_stream_requires_begin() {
    std::mt19937 rng(WEIGHT_SEED);
    const auto path = write_decoder("codec-stream-unstarted", N_VQ, 3 * D_MODEL,
            [](gguf_context *) {}, &rng);
    Codec codec(path.string(), {}, 1);
    try {
        codec.decode_stream(random_codes(1, CODE_SEED));
        check(false, "decode_stream before begin_decode_stream is rejected");
    } catch (const std::runtime_error & e) {
        check(std::string(e.what()).find("begin_decode_stream") != std::string::npos,
                std::string("unstarted stream: wrong failure: ") + e.what());
    }
    std::filesystem::remove(path);
}

} // namespace

int main() {
    std::mt19937 rng(WEIGHT_SEED);
    const auto path = write_decoder("codec-stream", N_VQ, 3 * D_MODEL, [](gguf_context *) {}, &rng);
    int rc = 0;
    try {
        Codec codec(path.string(), {}, 1);
        test_incremental_matches_full_decode(codec);
        test_windows_match_full_decode(codec);
        test_segments_decode_independently(codec);
        test_restarted_stream_forgets_history(codec);
        test_stream_work_is_linear(codec);
        test_context_is_decoded_then_trimmed(codec);
        test_stream_requires_begin();
        test_fused_attention_matches_unfused(path);
        test_fused_attention_encodes_like_unfused();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        rc = 1;
    }
    std::filesystem::remove(path);
    if (rc != 0) {
        return rc;
    }
    if (failures == 0) {
        std::printf("moss codec streaming: OK\n");
        return 0;
    }
    std::fprintf(stderr, "moss codec streaming: %d failures\n", failures);
    return 1;
}
