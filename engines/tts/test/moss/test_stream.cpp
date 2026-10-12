#include "gguf_fixtures.h"
#include "../test_env_portable.h"

#include "tts-cpp/moss/engine.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using namespace moss_fixtures;

namespace {

constexpr uint32_t STREAM_WEIGHT_SEED = 11;

int failures = 0;

void check(bool condition, const std::string & label) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", label.c_str());
        failures++;
    }
}

tts_cpp::moss::EngineOptions fixture_options(const std::filesystem::path & backbone,
                                             const std::filesystem::path & decoder) {
    tts_cpp::moss::EngineOptions options;
    options.backbone_path = backbone.string();
    options.decoder_path = decoder.string();
    options.n_threads = 1;
    options.context = 512;
    options.max_new_tokens = 24;
    // Zero LM weights plus greedy text selection keep the generation slot
    // active until the cap, so the longer case really emits many chunks.
    options.text_temperature = 0.0f;
    options.stream_chunk_frames = 2;
    return options;
}

float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        return 1.0f;
    }
    float worst = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return INFINITY;
        worst = std::max(worst, std::fabs(a[i] - b[i]));
    }
    return worst;
}

void test_stream_matches_batch(tts_cpp::moss::Engine & engine, int limit, int chunk) {
    const tts_cpp::moss::SynthesisResult batch = engine.synthesize("hi");
    check(!batch.pcm.empty(), "batch synthesis produces PCM");
    check(batch.generated_frames == limit, "fixture exercises the full requested length");
    check(std::any_of(batch.pcm.begin(), batch.pcm.end(), [](float s) { return s != 0.0f; }),
            "random codec weights make the parity check meaningful");

    std::vector<float> streamed;
    size_t chunks = 0;
    const tts_cpp::moss::SynthesisResult stream = engine.synthesize_stream("hi",
            [&](const float * samples, size_t count, int rate) {
                check(rate == batch.sample_rate, "chunk sample rate matches batch");
                streamed.insert(streamed.end(), samples, samples + count);
                chunks++;
                return true;
            });
    check(!stream.cancelled, "streaming completes without cancellation");
    check(stream.pcm.empty(), "streaming leaves the result PCM to the callback");
    const size_t chunk_samples = (size_t) chunk * OUT_DIM;
    check(chunks == (batch.pcm.size() + chunk_samples - 1) / chunk_samples,
            "stream emits every full chunk and the final partial chunk");
    check(stream.first_audio_ms > 0, "first audio latency is measured");
    check(stream.generated_frames == batch.generated_frames,
            "stream and batch share the seeded trajectory");
    check(streamed.size() == batch.pcm.size(), "streamed sample count matches batch");
    double error = 0, reference = 0;
    for (size_t i = 0; i < std::min(streamed.size(), batch.pcm.size()); ++i) {
        const double diff = streamed[i] - batch.pcm[i];
        error += diff * diff;
        reference += (double) batch.pcm[i] * batch.pcm[i];
    }
    const double relative = std::sqrt(error / std::max(reference, 1e-20));
    std::printf("stream chunks=%zu samples=%zu max_abs=%g relative_l2=%g\n", chunks,
                streamed.size(), max_abs_diff(streamed, batch.pcm),
                relative);
    // Vulkan's F16 matmul conversion rounds differently for batch and chunk
    // shapes. Measured error is <0.04% L2 (and <4e-4 absolute), disappears
    // with F16 disabled, and stays bounded over 37 chunks. Keep both gates.
    const bool gpu = std::string(engine.backend_name()).find("Vulkan") == 0;
    check(max_abs_diff(streamed, batch.pcm) < (gpu ? 1e-3f : 1e-4f) && relative < 1e-3,
            "streamed audio matches the batch render");
}

void test_stream_callback_cancels(tts_cpp::moss::Engine & engine) {
    size_t chunks = 0;
    const tts_cpp::moss::SynthesisResult result = engine.synthesize_stream("hi",
            [&](const float *, size_t, int) {
                chunks++;
                return false;
            });
    check(result.cancelled, "returning false from the callback cancels the stream");
    check(chunks == 1, "no further chunks after the callback declines");
}

void test_stream_explicit_cancel(tts_cpp::moss::Engine & engine) {
    size_t chunks = 0;
    const auto result = engine.synthesize_stream("hi", [&](const float *, size_t, int) {
        ++chunks;
        engine.cancel();
        return true;
    });
    check(result.cancelled, "cancel() from callback cancels the stream");
    check(chunks == 1, "cancel() prevents subsequent callbacks");
}

} // namespace

int main(int argc, char ** argv) {
    const bool gpu = argc == 2 && std::string(argv[1]) == "--vulkan";
    if (argc != 1 && !gpu) return 1;
    if (gpu) {
        setenv("TTS_CPP_GPU_BACKEND", "vulkan", 1);
        unsetenv("GGML_VK_DISABLE_F16");
    }
    const auto backbone = write_backbone("stream-backbone", [](gguf_context *) {});
    std::mt19937 weights(STREAM_WEIGHT_SEED);
    const auto decoder = write_decoder("stream-decoder", N_VQ, 3 * D_MODEL,
            [](gguf_context *) {}, &weights);
    int rc = 0;
    try {
        // Include one-frame chunks, partial final chunks, one final callback,
        // and enough chunks to repeatedly wrap the codec's attention context.
        for (int chunk : {1, 2, 5, 25, 7}) {
            auto options = fixture_options(backbone, decoder);
            options.use_gpu = gpu;
            options.stream_chunk_frames = chunk;
            if (chunk == 7) options.max_new_tokens = 256;
            tts_cpp::moss::Engine engine(options);
            if (gpu && std::string(engine.backend_name()).find("Vulkan") != 0)
                throw std::runtime_error("Vulkan backend required");
            std::printf("stream backend=%s chunk=%d limit=%d\n", engine.backend_name(),
                        chunk, options.max_new_tokens);
            test_stream_matches_batch(engine, options.max_new_tokens, chunk);
            test_stream_callback_cancels(engine);
            test_stream_matches_batch(engine, options.max_new_tokens, chunk);
            test_stream_explicit_cancel(engine);
            test_stream_matches_batch(engine, options.max_new_tokens, chunk);
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        rc = 1;
    }
    std::filesystem::remove(backbone);
    std::filesystem::remove(decoder);
    if (rc != 0) {
        return rc;
    }
    if (failures == 0) {
        std::printf("moss streaming: OK\n");
        return 0;
    }
    std::fprintf(stderr, "moss streaming: %d failures\n", failures);
    return 1;
}
