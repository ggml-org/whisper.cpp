#include "sfx_fixtures.h"
#include "../../../test/moss_precision.h"
#include "moss/sfx_coreml.h"
#include "moss/sfx_networks.h"
#include "moss/sfx_sampler.h"
#include "backend_util.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

using namespace moss_sfx_fixtures;
using namespace tts_cpp::moss::detail;

namespace {

constexpr int CACHE_LAYERS = 2;
constexpr int TEST_FRAMES = 17;
constexpr uint32_t MODEL_SEED = 27311;
constexpr float PARITY_TOLERANCE = 1e-5f;
constexpr int FLASH_WIDTH = 128;
constexpr int FLASH_HEADS = 1;
constexpr int FLASH_FRAMES = 32;
constexpr int FLASH_PROBE_NODES = 8;
constexpr double FUSED_RELATIVE_TOLERANCE = 2e-3;
constexpr double NORM_FLOOR = 1e-20;

void require(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error(message);
}

void compare(const std::vector<float> & expected, const std::vector<float> & actual) {
    require(!expected.empty() && expected.size() == actual.size(), "cached velocity shape");
    for (size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(actual[i]) && std::fabs(actual[i] - expected[i]) < PARITY_TOLERANCE,
                "cached and uncached velocities differ");
    }
}

template <typename Action>
void refuses(Action action, const std::string & message) {
    try {
        action();
    } catch (const std::runtime_error & error) {
        require(std::string(error.what()).find(message) != std::string::npos, error.what());
        return;
    }
    throw std::runtime_error("expected failure: " + message);
}

struct Fixture {
    std::filesystem::path path;
    Fixture(int width = DIT_EMBD, int heads = DIT_HEADS)
        : path(write_sfx_model("sfx-cache", MODEL_SEED, [](gguf_context *) {}, {}, {}, CACHE_LAYERS, width, heads)) {}
    ~Fixture() { std::error_code error; std::filesystem::remove(path, error); }
};

struct ZeroWeight {
    ggml_tensor * tensor;
    std::vector<uint8_t> original;

    explicit ZeroWeight(ggml_tensor * weight) : tensor(weight), original(ggml_nbytes(weight)) {
        ggml_backend_tensor_get(tensor, original.data(), 0, original.size());
        const std::vector<uint8_t> zeros(original.size(), 0);
        ggml_backend_tensor_set(tensor, zeros.data(), 0, zeros.size());
    }
    ~ZeroWeight() { ggml_backend_tensor_set(tensor, original.data(), 0, original.size()); }
};

struct Inputs {
    std::vector<float> latents;
    std::vector<float> positive;
    std::vector<float> negative;
    std::vector<float> other;

    explicit Inputs(SfxModel & model, int frames = TEST_FRAMES)
        : latents(gaussian_noise(size_t(frames) * LATENT, MODEL_SEED)), positive(encode_text(model, {40, 41})),
          negative(encode_text(model, {})), other(encode_text(model, {42, 43})) {}
};

void cached_pair(SfxModel & model, const Inputs & input) {
    SfxDitSession reference(model, TEST_FRAMES);
    const auto positive = reference.velocity(input.latents, 999.0f, input.positive);
    const auto negative = reference.velocity(input.latents, 500.0f, input.negative);
    const auto last = reference.velocity(input.latents, 0.0f, input.positive);
    SfxDitSession cached(model, TEST_FRAMES, false, 2);
    refuses([&] { cached.velocity(input.latents, 999.0f, input.positive); }, "not prepared");
    refuses([&] { cached.prepare_contexts({input.positive}); }, "context count");
    refuses([&] { cached.prepare_contexts({input.positive, {1.0f}}); }, "wrong size");
    cached.prepare_contexts({input.positive, input.negative});
    compare(positive, cached.velocity(input.latents, 999.0f, input.positive));
    compare(negative, cached.velocity(input.latents, 500.0f, input.negative));
    compare(last, cached.velocity(input.latents, 0.0f, input.positive));
    refuses([&] { cached.velocity(input.latents, 0.0f, input.other); }, "not in the DiT cache");
    refuses([&] { cached.velocity({1.0f}, 0.0f, input.positive); }, "wrong size");
    ZeroWeight text(model.tensor("dit.text_embd_1.weight"));
    ZeroWeight key(model.tensor("dit.blk.0.cross_k.weight"));
    ZeroWeight value(model.tensor("dit.blk.1.cross_v.weight"));
    compare(positive, cached.velocity(input.latents, 999.0f, input.positive));
    compare(negative, cached.velocity(input.latents, 500.0f, input.negative));
}

void replace_context(SfxModel & model, const Inputs & input) {
    SfxDitSession reference(model, TEST_FRAMES);
    const auto expected = reference.velocity(input.latents, 200.0f, input.other);
    SfxDitSession cached(model, TEST_FRAMES, false, 1);
    cached.prepare_contexts({input.positive});
    cached.prepare_contexts({input.other});
    compare(expected, cached.velocity(input.latents, 200.0f, input.other));
    refuses([&] { cached.velocity(input.latents, 200.0f, input.positive); }, "not in the DiT cache");
    encode_text(model, {40});
    refuses([&] { cached.velocity(input.latents, 200.0f, input.other); }, "no longer allocated");
}

void diffusion_parity(SfxModel & model, const Inputs & input) {
    const auto sigmas = flow_sigmas(4, 5.0f);
    auto run = [&](bool cache) {
        SfxDitSession dit(model, TEST_FRAMES, false, cache ? 2 : 0);
        if (cache) dit.prepare_contexts({input.positive, input.negative});
        auto latents = input.latents;
        for (size_t i = 0; i < sigmas.size(); ++i) {
            const float time = sigmas[i] * 1000.0f;
            const auto pos = dit.velocity(latents, time, input.positive);
            const auto neg = dit.velocity(latents, time, input.negative);
            euler_step(latents, guided_velocity(pos, neg, 4.0f), sigmas[i], i + 1 < sigmas.size() ? sigmas[i + 1] : 0);
        }
        return latents;
    };
    const auto expected = run(false);
    compare(expected, run(true));
}

bool is_metal(const SfxModel & model) {
    return tts_cpp::detail::reg_name_is_metal(tts_cpp::detail::buft_reg_name(model.buffer_type()));
}

bool device_supports_flash(const SfxModel & model, int keys) {
    SfxGraph graph(FLASH_PROBE_NODES);
    auto * q = ggml_new_tensor_3d(graph.ctx(), GGML_TYPE_F32, FLASH_WIDTH, FLASH_FRAMES, FLASH_HEADS);
    auto * k = ggml_new_tensor_3d(graph.ctx(), GGML_TYPE_F32, FLASH_WIDTH, keys, FLASH_HEADS);
    auto * v = ggml_new_tensor_3d(graph.ctx(), GGML_TYPE_F32, FLASH_WIDTH, keys, FLASH_HEADS);
    auto * attention = ggml_flash_attn_ext(graph.ctx(), q, k, v, nullptr, 1.0f / std::sqrt(float(FLASH_WIDTH)), 0, 0);
    return ggml_backend_dev_supports_op(ggml_backend_buft_get_device(model.buffer_type()), attention);
}

void compare_fused(const std::vector<float> & expected, const std::vector<float> & actual) {
    require(expected.size() == actual.size() && !expected.empty(), "fused velocity shape");
    double error = 0.0, norm = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(actual[i]), "fused velocity is finite");
        const double delta = double(expected[i]) - actual[i];
        error += delta * delta;
        norm += double(expected[i]) * expected[i];
    }
    require(std::sqrt(error / std::max(norm, NORM_FLOOR)) <= FUSED_RELATIVE_TOLERANCE,
            "fused velocity differs from the unfused reference");
}

std::vector<std::vector<float>> flash_velocities(SfxModel & model, const Inputs & input, bool cached) {
    SfxDitSession session(model, FLASH_FRAMES, false, cached ? 2 : 0);
    if (cached) session.prepare_contexts({input.positive, input.negative});
    std::vector<std::vector<float>> result;
    for (float time : {1000.0f, 937.5f, 0.0f}) {
        result.push_back(session.velocity(input.latents, time, input.positive));
        result.push_back(session.velocity(input.latents, time, input.negative));
    }
    return result;
}

void test_flash_attention(ggml_type precision, const BackendRequest & backend) {
    Fixture fixture(FLASH_WIDTH, FLASH_HEADS);
    moss_fixture_precision(fixture.path, precision);
    SfxModel reference(fixture.path.string(), backend, 1, false, {TablePlacement::Automatic, AttentionLayout::Unfused});
    SfxModel automatic(fixture.path.string(), backend, 1);
    require(!reference.fused_dit_attention(), "explicit unfused DiT layout is preserved");
    const bool supported = is_metal(automatic) && device_supports_flash(automatic, FLASH_FRAMES) &&
                           device_supports_flash(automatic, TEXT_TOKENS);
    require(automatic.fused_dit_attention() == (automatic.fused_attention() || supported),
            "a supported Metal DiT selects flash attention");
    if (is_metal(automatic)) require(!automatic.fused_attention(), "Metal text attention keeps its existing layout");
    const Inputs input(reference, FLASH_FRAMES);
    const auto expected = flash_velocities(reference, input, false);
    const auto plain = flash_velocities(automatic, input, false);
    const auto cached = flash_velocities(automatic, input, true);
    for (size_t i = 0; i < expected.size(); ++i) {
        compare_fused(expected[i], cached[i]);
        compare(plain[i], cached[i]);
    }
    std::printf("sfx flash %s / %s: routing and cached parity passed\n", ggml_type_name(precision), automatic.backend_name());
}

void test_precision(ggml_type precision, const BackendRequest & backend) {
    Fixture fixture;
    moss_fixture_precision(fixture.path, precision);
    SfxModel model(fixture.path.string(), backend, 1);
    if (backend.name != "cpu") require(std::string(model.backend_name()) != "CPU", "requested device must be used");
    if (is_metal(model)) require(!model.fused_dit_attention(), "unsupported Metal heads keep unfused attention");
    const Inputs input(model);
    cached_pair(model, input);
    replace_context(model, input);
    diffusion_parity(model, input);
    std::printf("sfx cache %s / %s: parity and reuse passed\n", ggml_type_name(precision), model.backend_name());
}

} // namespace

int main() {
    try {
        const char * backend = std::getenv("MOSS_SFX_CACHE_BACKEND");
        const BackendRequest request{backend ? backend : "cpu", false};
        for (const auto precision : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}) {
            test_precision(precision, request);
            test_flash_attention(precision, request);
        }
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
