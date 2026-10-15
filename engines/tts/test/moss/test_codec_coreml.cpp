#include "gguf_fixtures.h"

#include "moss/codec.h"
#include "moss/codec_coreml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace tts_cpp::moss::detail;
using namespace moss_fixtures;

namespace {

constexpr uint32_t WEIGHT_SEED = 20261008;
constexpr uint32_t CODE_SEED = 11;
constexpr uint32_t LATENT_SEED = 5;
constexpr int FAKE_CHUNK = 4;
constexpr int NEVER_FAIL = -1;
constexpr int STREAM_FRAMES = 37;
constexpr int FALLBACK_CALL = 3;
constexpr float MASKED = -1e4f;
constexpr float TOLERANCE = 1e-5f;
constexpr float GGML_TOLERANCE = 1e-4f;
constexpr int FIXTURE_CONTEXT = 8;
constexpr int FIXTURE_HEAD_DIM = D_MODEL / 2;
constexpr const char * FAKE_LABEL = "coreml-fake";
const std::vector<int> SPLITS = {1, 4, 3, 9, 2, 7, 5, 6};

int failures = 0;

void check(bool condition, const std::string & label) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", label.c_str());
        failures++;
    }
}

template <typename Fn>
void expect_failure(Fn && fn, const std::string & needle, const std::string & label) {
    try {
        fn();
        check(false, label + ": accepted");
    } catch (const std::runtime_error & e) {
        check(std::string(e.what()).find(needle) != std::string::npos, label + ": wrong failure: " + e.what());
    }
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

class PrefixSumModel final : public CodecSidecarModel {
public:
    PrefixSumModel(CodecGeometry geometry, int fail_on_call, int * calls)
        : geometry_(std::move(geometry)), fail_on_call_(fail_on_call), calls_(calls) {}

    int chunk_frames() const override {
        return FAKE_CHUNK;
    }

    bool reset() override {
        state_ = 0.0;
        committed_ = 0;
        return true;
    }

    bool decode_chunk(const std::vector<float> & latents, float commit, const std::vector<std::vector<float>> & tables,
                      std::vector<float> & pcm) override {
        if ((*calls_)++ == fail_on_call_) {
            return false;
        }
        tables_match_ = tables_match_ && tables == codec_chunk_tables(geometry_, FAKE_CHUNK, committed_);
        pcm.clear();
        const double end = fill_prefix_sums(latents, pcm);
        if (commit == 1.0f) {
            state_ = end;
            committed_ += FAKE_CHUNK;
        }
        return true;
    }

    const char * label() const override {
        return FAKE_LABEL;
    }

    bool tables_match() const {
        return tables_match_;
    }

private:
    double fill_prefix_sums(const std::vector<float> & latents, std::vector<float> & pcm) const {
        double running = state_;
        for (int frame = 0; frame < FAKE_CHUNK; ++frame) {
            running += latents[(size_t) frame * geometry_.code_dim];
            pcm.insert(pcm.end(), (size_t) geometry_.hop, (float) running);
        }
        return running;
    }

    CodecGeometry geometry_;
    int fail_on_call_;
    int * calls_;
    double state_ = 0.0;
    int64_t committed_ = 0;
    bool tables_match_ = true;
};

CodecGeometry tiny_geometry() {
    CodecGeometry geometry;
    geometry.stages = {{2, 5, 4, 10000.0f}};
    geometry.code_dim = 3;
    geometry.hop = 2;
    return geometry;
}

std::vector<float> random_latents(int frames, int width, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> draw(0.0f, 1.0f);
    std::vector<float> latents((size_t) frames * width);
    for (float & value : latents) {
        value = draw(rng);
    }
    return latents;
}

std::vector<float> expected_prefix_sums(const std::vector<float> & latents, const CodecGeometry & geometry) {
    std::vector<float> pcm;
    double running = 0.0;
    for (size_t frame = 0; frame * geometry.code_dim < latents.size(); ++frame) {
        running += latents[frame * geometry.code_dim];
        pcm.insert(pcm.end(), (size_t) geometry.hop, (float) running);
    }
    return pcm;
}

std::vector<float> latent_frames(const std::vector<float> & latents, int width, int first, int count) {
    return std::vector<float>(latents.begin() + (std::ptrdiff_t) first * width,
                              latents.begin() + (std::ptrdiff_t) (first + count) * width);
}

std::vector<float> stream_latents(CodecSidecarStream & stream, const std::vector<float> & latents, int width, int hop) {
    std::vector<float> pcm;
    const int frames = (int) (latents.size() / (size_t) width);
    size_t turn = 0;
    for (int first = 0; first < frames; ++turn) {
        const int count = std::min(SPLITS[turn % SPLITS.size()], frames - first);
        std::vector<float> piece;
        check(stream.decode(latent_frames(latents, width, first, count), piece), "a streamed piece decodes");
        check(piece.size() == (size_t) count * hop, "a piece returns exactly its frames");
        pcm.insert(pcm.end(), piece.begin(), piece.end());
        first += count;
    }
    return pcm;
}

void test_band_mask() {
    const std::vector<float> start = codec_band_mask(0, 2, 3, 4);
    check(start == std::vector<float>({MASKED, MASKED, MASKED, 0, MASKED, MASKED, MASKED, MASKED, 0, 0}),
          "at the stream start the cache is masked and fresh keys are causal");
    const std::vector<float> later = codec_band_mask(5, 2, 3, 4);
    check(later == std::vector<float>({0, 0, 0, 0, MASKED, MASKED, 0, 0, 0, 0}),
          "cached keys stay visible only inside the attention window");
}

void test_rope_tables() {
    std::vector<float> cos;
    std::vector<float> sin;
    codec_rope_tables(3, 2, 4, 10000.0f, cos, sin);
    check(cos.size() == 8 && sin.size() == 8, "one rotary row per position");
    check(std::fabs(cos[0] - std::cos(3.0f)) < TOLERANCE && std::fabs(sin[1] - std::sin(0.03f)) < TOLERANCE,
          "rotary angles follow the absolute position");
    check(cos[0] == cos[2] && sin[1] == sin[3] && std::fabs(cos[4] - std::cos(4.0f)) < TOLERANCE,
          "the tables repeat each frequency over both halves");
}

void test_stream_scheduler() {
    const CodecGeometry geometry = tiny_geometry();
    int calls = 0;
    PrefixSumModel model(geometry, NEVER_FAIL, &calls);
    CodecSidecarStream stream(model, geometry);
    check(stream.begin(), "a stream begins");
    const std::vector<float> latents = random_latents(STREAM_FRAMES, geometry.code_dim, LATENT_SEED);
    const std::vector<float> pcm = stream_latents(stream, latents, geometry.code_dim, geometry.hop);
    check(max_abs_diff(pcm, expected_prefix_sums(latents, geometry)) < TOLERANCE,
          "uneven pieces decode exactly as one causal pass");
    check(model.tables_match(), "every chunk gets the tables of its committed position");
    check(stream.begin() && max_abs_diff(stream_latents(stream, latents, geometry.code_dim, geometry.hop),
          expected_prefix_sums(latents, geometry)) < TOLERANCE, "a restarted stream forgets its history");
}

std::vector<float> stream_codes(Codec & codec, const std::vector<int32_t> & codes, int piece_frames) {
    std::vector<float> pcm;
    codec.begin_decode_stream(0, piece_frames);
    const int frames = (int) (codes.size() / N_VQ);
    size_t turn = 0;
    for (int first = 0; first < frames; ++turn) {
        const int count = std::min(SPLITS[turn % SPLITS.size()], frames - first);
        const std::vector<float> piece = codec.decode_stream(std::vector<int32_t>(
                codes.begin() + (std::ptrdiff_t) first * N_VQ, codes.begin() + (std::ptrdiff_t) (first + count) * N_VQ));
        pcm.insert(pcm.end(), piece.begin(), piece.end());
        first += count;
    }
    return pcm;
}

std::vector<int32_t> random_codes(int frames) {
    std::mt19937 rng(CODE_SEED);
    std::uniform_int_distribution<int32_t> draw(0, CODE_SIZE - 1);
    std::vector<int32_t> codes((size_t) frames * N_VQ);
    for (int32_t & code : codes) {
        code = draw(rng);
    }
    return codes;
}

void test_codec_geometry(Codec & codec) {
    const CodecGeometry geometry = codec.sidecar_geometry();
    check(geometry.stages.size() == 1 && geometry.stages[0].rate == 1 && geometry.stages[0].context == FIXTURE_CONTEXT &&
          geometry.stages[0].head_dim == FIXTURE_HEAD_DIM, "the sidecar geometry mirrors the transformer chain");
    check(geometry.code_dim == OUT_DIM && geometry.hop == codec.samples_per_frame(), "latent width and hop");
    check(!codec.on_coreml(), "no staged sidecar keeps the ggml codec");
}

void test_codec_routing(const std::filesystem::path & path) {
    Codec reference(path.string(), {}, 1);
    Codec codec(path.string(), {}, 1);
    test_codec_geometry(codec);
    int calls = 0;
    codec.attach_sidecar(std::make_unique<PrefixSumModel>(codec.sidecar_geometry(), NEVER_FAIL, &calls), false);
    const std::vector<int32_t> codes = random_codes(STREAM_FRAMES);
    codec.begin_run();
    check(codec.decode(codes) == reference.decode(codes) && calls == 0,
          "a batch decode runs in one ggml pass with a sidecar attached");
    check(codec.on_coreml() && codec.run_backend() == GGML_STAGE_BACKEND, "the batch run reports ggml");
    codec.begin_run();
    const int64_t before = codec.frames_decoded();
    const std::vector<float> streamed = stream_codes(codec, codes, FAKE_CHUNK);
    check(calls > 0 && codec.run_backend() == FAKE_LABEL, "a stream of chunk-sized pieces runs on the sidecar");
    check(streamed.size() == (size_t) STREAM_FRAMES * codec.samples_per_frame(), "the sidecar renders every frame");
    check(codec.frames_decoded() - before == STREAM_FRAMES, "streaming counts every frame once");
    const int sidecar_calls = calls;
    codec.begin_run();
    check(stream_codes(codec, codes, FAKE_CHUNK + 1) == stream_codes(reference, codes, FAKE_CHUNK + 1) &&
          calls == sidecar_calls && codec.run_backend() == GGML_STAGE_BACKEND,
          "a stream of pieces longer than the sidecar chunk runs on ggml");
}

void test_codec_stream_fallback(const std::filesystem::path & path) {
    Codec reference(path.string(), {}, 1);
    const std::vector<int32_t> codes = random_codes(STREAM_FRAMES);
    const std::vector<float> ggml = stream_codes(reference, codes, FAKE_CHUNK);
    Codec codec(path.string(), {}, 1);
    int calls = 0;
    codec.attach_sidecar(std::make_unique<PrefixSumModel>(codec.sidecar_geometry(), FALLBACK_CALL, &calls), false);
    codec.begin_run();
    const int64_t before = codec.frames_decoded();
    const std::vector<float> mixed = stream_codes(codec, codes, FAKE_CHUNK);
    check(!codec.on_coreml() && codec.run_backend() == MIXED_STAGE_BACKEND, "a failed chunk retires the sidecar");
    check(mixed.size() == ggml.size() && codec.frames_decoded() - before == STREAM_FRAMES,
          "the fallback still renders and counts every frame once");
    const size_t tail = mixed.size() / 2;
    check(max_abs_diff(std::vector<float>(mixed.end() - (std::ptrdiff_t) tail, mixed.end()),
          std::vector<float>(ggml.end() - (std::ptrdiff_t) tail, ggml.end())) < GGML_TOLERANCE,
          "after a failure the ggml stream resumes with the full history");
}

void test_codec_strict(const std::filesystem::path & path) {
    Codec reference(path.string(), {}, 1);
    const std::vector<int32_t> codes = random_codes(STREAM_FRAMES);
    Codec codec(path.string(), {}, 1);
    int calls = 0;
    codec.attach_sidecar(std::make_unique<PrefixSumModel>(codec.sidecar_geometry(), 0, &calls), true);
    check(codec.decode(codes) == reference.decode(codes) && calls == 0,
          "a batch decode stays on ggml under MOSS_COREML_STRICT while a sidecar is attached");
    expect_failure([&] { stream_codes(codec, codes, FAKE_CHUNK); }, COREML_STRICT_ENV,
                   "a failed sidecar chunk under MOSS_COREML_STRICT");
    codec.attach_sidecar(nullptr, true);
    expect_failure([&] { codec.decode(codes); }, COREML_STRICT_ENV, "a batch decode without a sidecar under MOSS_COREML_STRICT");
    expect_failure([&] { stream_codes(codec, codes, FAKE_CHUNK); }, COREML_STRICT_ENV,
                   "a stream without a sidecar under MOSS_COREML_STRICT");
}

void test_sidecar_lookup(const std::filesystem::path & path) {
    check(codec_decoder_sidecar_path("models/moss-codec-decoder-q8_0.gguf") == "models/moss-codec-decoder.mlmodelc",
          "the decoder sidecar name drops the quantization tag");
    const std::filesystem::path staged = codec_decoder_sidecar_path(path.string());
    std::filesystem::create_directories(staged);
    Codec codec(path.string(), {}, 1);
    check(!codec.on_coreml(), "a directory that is not a model leaves the codec on ggml");
    std::filesystem::remove_all(staged);
}

} // namespace

int main() {
    std::mt19937 rng(WEIGHT_SEED);
    const auto path = write_decoder("codec-coreml", N_VQ, 3 * D_MODEL, [](gguf_context *) {}, &rng);
    int rc = 0;
    try {
        test_band_mask();
        test_rope_tables();
        test_stream_scheduler();
        test_codec_routing(path);
        test_codec_stream_fallback(path);
        test_codec_strict(path);
        test_sidecar_lookup(path);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        rc = 1;
    }
    std::filesystem::remove(path);
    if (rc != 0) {
        return rc;
    }
    if (failures == 0) {
        std::printf("moss codec Core ML routing: OK\n");
        return 0;
    }
    std::fprintf(stderr, "moss codec Core ML routing: %d failures\n", failures);
    return 1;
}
