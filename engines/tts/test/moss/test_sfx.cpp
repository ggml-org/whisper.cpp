#include "sfx_fixtures.h"

#include "../test_env_portable.h"
#include "../../../test/moss_precision.h"

#include "moss/cli.h"
#include "moss/sfx_coreml.h"
#include "moss/sfx_model.h"
#include "moss/sfx_networks.h"
#include "moss/sfx_prompt.h"
#include "moss/sfx_sampler.h"
#include "moss/sfx_tokenizer.h"
#include "tts-cpp/moss/sound_effect.h"

#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace moss_sfx_fixtures;
using namespace tts_cpp::moss::detail;

namespace {

constexpr size_t UTF8_TWO_BYTE_LITERAL_SIZE = 3;
static_assert(sizeof("\u00e9") == UTF8_TWO_BYTE_LITERAL_SIZE,
              "prompt tests need UTF-8 narrow literals; compile with /utf-8 on MSVC");

constexpr uint32_t WEIGHT_SEED = 20260924;
constexpr float TOLERANCE = 1e-4f;
constexpr int SMALL_WINDOW = 16;
constexpr const char * UNKNOWN_UNITS = "neural_engine";
constexpr const char * ALL_UNITS = "all";
constexpr size_t NOISE_SAMPLES = 20000;
constexpr int KEPT_FRAMES = 100;
constexpr float OVERFLOW_EMBEDDING = 1.0f;
constexpr float OVERFLOW_PROJECTION = 12.5f;
constexpr float OVERFLOW_DOWN = 1.0f;
constexpr int NEVER_FAIL = -1;
constexpr int FAKE_VAE_WINDOW = 96;
constexpr float FAKE_VELOCITY_OFFSET = 1.0f;
constexpr float PROBE_TIMESTEP = 937.5f;
constexpr int TIMESTEP_MAX_PERIOD = 10000;
constexpr size_t SINUSOID_ARENA_BYTES = 1 << 20;
constexpr const char * FAKE_SIDECAR_LABEL = "coreml-fake";
constexpr const char * POSITIONAL_MODEL = "m.gguf";
constexpr int POSITIONAL_THREADS = 2;
constexpr const char * POSITIONAL_BACKENDS_DIR = "/plugins";
constexpr int TIER_DIT_WIDTH = 32;
constexpr double TIER_MIN_COSINE = 0.9;
constexpr const char * TIER_MATRIX = "dit.blk.0.self_q.weight";

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
        check(std::string(e.what()).find(needle) != std::string::npos,
                label + ": wrong failure: " + e.what());
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

bool rows_are_zero(const std::vector<float> & context, int first, int width) {
    return std::all_of(context.begin() + (std::ptrdiff_t) first * width, context.end(),
            [](float v) { return v == 0.0f; });
}

void test_schedule() {
    const std::vector<float> linear = flow_sigmas(4, 1.0f);
    check(max_abs_diff(linear, {1.0f, 0.75f, 0.5f, 0.25f}) < 1e-6f, "shift 1 keeps the linear schedule");
    const std::vector<float> shifted = flow_sigmas(4, 5.0f);
    check(shifted[0] == 1.0f, "the schedule starts at pure noise");
    check(std::fabs(shifted[2] - 5.0f * 0.5f / (1.0f + 4.0f * 0.5f)) < 1e-6f, "shift follows the flow-match formula");
    check(std::is_sorted(shifted.rbegin(), shifted.rend()), "sigmas decrease");
    expect_failure([] { flow_sigmas(0, 5.0f); }, "steps", "zero steps");
}

void test_euler_and_guidance() {
    std::vector<float> latents = {1.0f, 2.0f};
    euler_step(latents, {4.0f, -2.0f}, 1.0f, 0.75f);
    check(max_abs_diff(latents, {0.0f, 2.5f}) < 1e-6f, "euler moves along the velocity by sigma_next - sigma");
    check(max_abs_diff(guided_velocity({3.0f}, {1.0f}, 4.0f), {9.0f}) < 1e-6f, "cfg extrapolates from the negative");
    check(max_abs_diff(guided_velocity({3.0f}, {1.0f}, 1.0f), {3.0f}) < 1e-6f, "guidance 1 keeps the positive");
    expect_failure([] { guided_velocity({1.0f}, {1.0f, 2.0f}, 2.0f); }, "disagree", "mismatched branches");
}

void test_prompt_text() {
    check(clean_prompt("  Rain &amp;amp;  thunder\n\t") == "Rain & thunder", "cleaning unescapes and collapses spaces");
    check(clean_prompt("Tom &amp;amp;amp;amp; Jerry") == "Tom & Jerry", "plain text unescapes to a fixed point");
    check(clean_prompt("<b>&amp;amp;x</b>") == "<b>&x</b>", "markup-looking text unescapes exactly twice");
    check(clean_prompt("A dog\u2019s \u201cbark\u201d") == "A dog's \"bark\"", "curly quotes are straightened");
    check(clean_prompt("\uff26\uff55\uff4c\uff4c\u3000\uff57\uff49\uff44\uff54\uff48\uff0c\u3002") ==
            "Full width,\u3002", "full-width forms and the ideographic space narrow");
    check(clean_prompt("\ufb01ne\u00a0\u2009 \ufb02ow") == "fine flow", "ligatures split and Unicode spaces collapse");
    check(clean_prompt("x &#xD800; y") == "x \ufffd y", "surrogate references become the replacement character");
    check(clean_prompt("a&#65b") == "aAb" && clean_prompt("caf&#233;") == "caf\u00e9",
            "numeric references decode with or without a semicolon");
    check(clean_prompt("a&#+65;b") == "a&#+65;b", "signed numbers are not references");
    check(clean_prompt(std::string("bad\xff") + "byte") == "bad\ufffdbyte", "invalid UTF-8 becomes the replacement character");
    check(clean_prompt("\xc0\x80z") == "\ufffd\ufffdz", "overlong sequences are replaced byte by byte");
    check(clean_prompt("caf\xe9 au lait") == "caf\ufffd au lait", "a Latin-1 byte keeps the text after it");
    check(clean_prompt("\xc3z dog") == "\ufffdz dog", "a truncated two-byte sequence keeps the next character");
    check(clean_prompt("\xe2\x82z") == "\ufffd\ufffdz", "a truncated three-byte sequence keeps the next character");
    check(clean_prompt("\xc3\xe4\xb8\xad") == "\ufffd\u4e2d", "a truncated lead keeps the valid character after it");
    check(clean_prompt("end\xe4\xb8") == "end\ufffd\ufffd", "a sequence cut at the end of the text is replaced");
    check(clean_prompt("a\xed\xa0\x80" "b") == "a\ufffd\ufffd\ufffd" "b", "encoded surrogates are replaced");
    check(clean_prompt("\xf7\xbf\xbf\xbf") == "\ufffd\ufffd\ufffd\ufffd", "code points past U+10FFFF are replaced");
    check(clean_prompt("&lt;&amp;amp;amp;amp;amp;") == "<&", "unescaping that produces a tag stays on the fixed-point path");
    check(clean_prompt("a&lt;b&gt;c &quot;q&quot; &apos;s&apos; x&nbsp;y") == "a<b>c \"q\" 's' x y",
            "every supported named entity decodes");
    check(clean_prompt("a&#x41b") == "a\u041b", "hex references decode without a semicolon");
    check(clean_prompt("a&#0000000065;") == "a&#0000000065;", "references longer than eight digits stay literal");
    check(clean_prompt(duration_prompt(" dog  bark ", 50)) == "dog bark duration: 5.0s", "duration suffix matches training");
    check(seconds_in_tenths(4.96) == 50 && seconds_in_tenths(2.04) == 20, "seconds round to one decimal");
    check(seconds_in_tenths(0.25) == 2 && seconds_in_tenths(0.35) == 3 && seconds_in_tenths(1.25) == 12,
            "rounding follows Python on binary halves");
    check(seconds_in_tenths(0.45) == 5 && seconds_in_tenths(1.05) == 11 && seconds_in_tenths(0.15) == 1 &&
            seconds_in_tenths(2.35) == 24, "rounding works on the double the caller typed, as Python does");
    check(seconds_in_tenths(0.0) == 0 && seconds_in_tenths(-1.0) == 0, "non-positive seconds give no duration");
}

std::pair<double, double> mean_and_variance(const std::vector<float> & values) {
    double sum = 0.0, square = 0.0;
    for (float v : values) {
        sum += v;
        square += (double) v * v;
    }
    const double mean = sum / (double) values.size();
    return {mean, square / (double) values.size() - mean * mean};
}

void test_noise() {
    const std::vector<float> a = gaussian_noise(NOISE_SAMPLES, 7);
    check(a == gaussian_noise(NOISE_SAMPLES, 7), "noise is reproducible for a seed");
    check(a != gaussian_noise(NOISE_SAMPLES, 8), "seeds give different noise");
    const auto [mean, variance] = mean_and_variance(a);
    check(std::fabs(mean) < 0.05 && std::fabs(variance - 1.0) < 0.05, "noise is standard normal");
    check(gaussian_noise(3, 1).size() == 3, "odd counts are filled");
}

void test_model_config(const SfxModel & model) {
    const SfxConfig & config = model.config();
    check(config.latent_frames() == FRAMES, "latent frames follow sample rate, duration and hop");
    check(config.vae.hop_length() == HOP, "hop is the product of the decoder rates");
    check(config.default_steps == STEPS && config.text.max_tokens == TEXT_TOKENS, "metadata lands in the config");
}

void test_rejected_models() {
    const auto wrong_arch = write_sfx_model("sfx-arch", WEIGHT_SEED, [](gguf_context * f) {
        gguf_set_val_str(f, "general.architecture", "moss-tts-delay");
    });
    expect_failure([&] { SfxModel model(wrong_arch.string(), {}, 1); }, "architecture", "foreign architecture");
    const auto bad_geometry = write_sfx_model("sfx-geometry", WEIGHT_SEED, [](gguf_context * f) {
        gguf_set_val_u32(f, "moss-sfx.dit.out_channels", LATENT + 1);
    });
    expect_failure([&] { SfxModel model(bad_geometry.string(), {}, 1); }, "DiT geometry", "mismatched channels");
    const auto bad_rates = write_sfx_model("sfx-rates", WEIGHT_SEED, [](gguf_context * f) {
        const int rates[] = {0};
        gguf_set_arr_data(f, "moss-sfx.vae.decoder_rates", GGUF_TYPE_INT32, rates, 1);
    });
    expect_failure([&] { SfxModel model(bad_rates.string(), {}, 1); }, "decoder rates", "zero decoder rate");
    std::filesystem::remove(wrong_arch);
    std::filesystem::remove(bad_geometry);
    std::filesystem::remove(bad_rates);
}

void expect_model_rejects(const std::filesystem::path & path, const std::string & needle, const std::string & label) {
    expect_failure([&] { SfxModel model(path.string(), {}, 1); }, needle, label);
    std::filesystem::remove(path);
}

void test_rejected_metadata_bounds() {
    expect_model_rejects(write_sfx_model("sfx-hop", WEIGHT_SEED, [](gguf_context * f) {
        const int rates[] = {64, 64, 64, 64, 64, 64};
        gguf_set_arr_data(f, "moss-sfx.vae.decoder_rates", GGUF_TYPE_INT32, rates, 6);
        gguf_set_val_u32(f, "moss-sfx.vae.decoder_dim", 64);
    }), "hop length", "decoder rates whose product overflows");
    expect_model_rejects(write_sfx_model("sfx-frames", WEIGHT_SEED, [](gguf_context * f) {
        const int rates[] = {1};
        gguf_set_arr_data(f, "moss-sfx.vae.decoder_rates", GGUF_TYPE_INT32, rates, 1);
        gguf_set_val_u32(f, "moss-sfx.vae.sample_rate", 192000);
        gguf_set_val_f32(f, "moss-sfx.max_seconds", 600.0f);
    }), "generation metadata", "a latent longer than the engine runs");
    expect_model_rejects(write_sfx_model("sfx-odd-head", WEIGHT_SEED, [](gguf_context * f) {
        gguf_set_val_u32(f, "moss-sfx.text.attention.key_length", TEXT_HEAD_DIM - 1);
    }), "text encoder geometry", "an odd text head dimension");
    expect_model_rejects(write_sfx_model("sfx-deep", WEIGHT_SEED, [](gguf_context * f) {
        gguf_set_val_u32(f, "moss-sfx.dit.block_count", 1000);
    }), "DiT geometry", "more DiT layers than the graph budget");
    expect_model_rejects(write_sfx_model("sfx-deep-text", WEIGHT_SEED, [](gguf_context * f) {
        gguf_set_val_u32(f, "moss-sfx.text.block_count", 1000);
    }), "text encoder geometry", "more text layers than the graph budget");
    expect_model_rejects(write_sfx_model("sfx-many-rates", WEIGHT_SEED, [](gguf_context * f) {
        const int rates[17] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        gguf_set_arr_data(f, "moss-sfx.vae.decoder_rates", GGUF_TYPE_INT32, rates, 17);
    }), "too long", "more decoder rates than supported");
    expect_model_rejects(write_sfx_model("sfx-no-shift", WEIGHT_SEED, [](gguf_context * f) {
        gguf_remove_key(f, "moss-sfx.sigma_shift");
    }), "missing float metadata", "missing generation metadata");
    const auto healthy = write_sfx_model("sfx-threads", WEIGHT_SEED);
    expect_failure([&] { SfxModel model(healthy.string(), {}, 0); }, "threads", "zero threads");
    expect_failure([&] { SfxModel model(healthy.string(), {}, 1025); }, "threads", "too many threads");
    std::filesystem::remove(healthy);
}

void expect_engine_rejects(const std::filesystem::path & path, const std::string & needle, const std::string & label) {
    tts_cpp::moss::SoundEffectOptions options;
    options.model_path = path.string();
    options.n_threads = 1;
    expect_failure([&] { tts_cpp::moss::SoundEffectEngine engine(options); }, needle, label);
    std::filesystem::remove(path);
}

void test_rejected_tensors() {
    expect_engine_rejects(write_sfx_model("sfx-text-ff", WEIGHT_SEED, [](gguf_context * f) {
        gguf_set_val_u32(f, "moss-sfx.text.feed_forward_length", TEXT_FF + 8);
    }), "unexpected dimensions", "text weights that disagree with the metadata");
    expect_engine_rejects(write_sfx_model("sfx-dit-ff", WEIGHT_SEED, [](gguf_context * f) {
        gguf_set_val_u32(f, "moss-sfx.dit.feed_forward_length", DIT_FF + 8);
    }), "unexpected dimensions", "DiT weights that disagree with the metadata");
    expect_engine_rejects(write_sfx_model("sfx-vae-dim", WEIGHT_SEED, [](gguf_context * f) {
        gguf_set_val_u32(f, "moss-sfx.vae.decoder_dim", DECODER_DIM * 2);
    }), "unexpected dimensions", "VAE weights that disagree with the metadata");
    expect_engine_rejects(write_sfx_model("sfx-f16-snake", WEIGHT_SEED, [](gguf_context *) {},
            {{"vae.snake_out.alpha", {GGML_TYPE_F16, 1.0f}}}), "must be f32", "an f16 snake parameter");
    expect_engine_rejects(write_sfx_model("sfx-dit-embed", WEIGHT_SEED, [](gguf_context *) {},
            {}, {{"dit.text_embd_2.weight", {DIT_EMBD, DIT_EMBD + 1}}}), "unexpected dimensions",
            "a second text projection with the wrong width");
    expect_engine_rejects(write_sfx_model("sfx-i32-linear", WEIGHT_SEED, [](gguf_context *) {},
            {{"dit.blk.0.self_q.weight", {GGML_TYPE_I32, 1.0f}}}), "unsupported type", "an integer linear weight");
    expect_engine_rejects(write_sfx_model("sfx-f16-norm", WEIGHT_SEED, [](gguf_context *) {},
            {{"text.blk.0.ffn_norm.weight", {GGML_TYPE_F16, 1.0f}}}), "must be f32", "an f16 norm scale");
    f32_kernels() = true;
    const auto f32_path = write_sfx_model("sfx-f32-kernels", WEIGHT_SEED);
    f32_kernels() = false;
    expect_engine_rejects(f32_path, "must be f16", "f32 VAE kernels");
}

std::filesystem::path write_tier_model(const char * tag, ggml_type tier) {
    const auto path = write_sfx_model(tag, WEIGHT_SEED, [](gguf_context *) {}, {}, {}, 1, TIER_DIT_WIDTH);
    moss_fixture_precision(path, tier);
    return path;
}

double cosine(const std::vector<float> & a, const std::vector<float> & b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        dot += (double) a[i] * b[i];
        na += (double) a[i] * a[i];
        nb += (double) b[i] * b[i];
    }
    return a.size() == b.size() && na > 0.0 && nb > 0.0 ? dot / std::sqrt(na * nb) : -1.0;
}

std::vector<float> tier_velocity(const std::filesystem::path & path, ggml_type tier) {
    SfxModel model(path.string(), {}, 1);
    check(model.tensor(TIER_MATRIX)->type == tier,
          std::string(ggml_type_name(tier)) + " linear weights load without conversion");
    const std::vector<float> context = encode_text(model, {40, 41});
    SfxDitSession dit(model, FRAMES);
    return dit.velocity(gaussian_noise((size_t) FRAMES * LATENT, 3), PROBE_TIMESTEP, context);
}

void test_weight_tiers() {
    const auto reference_path = write_tier_model("sfx-tier-f32", GGML_TYPE_F32);
    const std::vector<float> reference = tier_velocity(reference_path, GGML_TYPE_F32);
    std::filesystem::remove(reference_path);
    for (ggml_type tier : {GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}) {
        const auto path = write_tier_model("sfx-tier", tier);
        check(cosine(reference, tier_velocity(path, tier)) > TIER_MIN_COSINE,
              std::string(ggml_type_name(tier)) + " linear weights follow the f32 velocity");
        std::filesystem::remove(path);
    }
    expect_engine_rejects(write_tier_model("sfx-tier-q5", GGML_TYPE_Q5_0), "unsupported type", "q5_0 linear weights");
}

void test_text_encoder(SfxModel & model) {
    check(rows_are_zero(encode_text(model, {}), 0, TEXT_EMBD), "an empty prompt encodes to zeros");
    const std::vector<float> two = encode_text(model, {40, 41});
    check(two.size() == (size_t) TEXT_TOKENS * TEXT_EMBD, "context always spans the full text length");
    check(rows_are_zero(two, 2, TEXT_EMBD), "rows past the prompt stay zero");
    check(!rows_are_zero(two, 0, TEXT_EMBD), "prompt rows carry hidden states");
    const std::vector<float> one = encode_text(model, {40});
    check(max_abs_diff(std::vector<float>(one.begin(), one.begin() + TEXT_EMBD),
            std::vector<float>(two.begin(), two.begin() + TEXT_EMBD)) < TOLERANCE,
            "attention is causal, so later tokens do not change earlier rows");
    expect_failure([&] { encode_text(model, std::vector<int32_t>(TEXT_TOKENS + 1, 40)); }, "context",
            "prompt longer than the text context");
    expect_failure([&] { encode_text(model, {-1}); }, "vocabulary", "negative token id");
}

bool all_finite(const std::vector<float> & values) {
    return std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); });
}

void test_feed_forward_beyond_half_precision() {
    const auto path = write_sfx_model("sfx-overflow", WEIGHT_SEED, [](gguf_context *) {}, {
        {"text.token_embd.weight", {GGML_TYPE_F32, OVERFLOW_EMBEDDING}},
        {"text.blk.0.ffn_norm.weight", {GGML_TYPE_F32, OVERFLOW_EMBEDDING}},
        {"text.blk.0.ffn_gate.weight", {GGML_TYPE_F16, OVERFLOW_PROJECTION}},
        {"text.blk.0.ffn_up.weight", {GGML_TYPE_F16, OVERFLOW_PROJECTION}},
        {"text.blk.0.ffn_down.weight", {GGML_TYPE_F16, OVERFLOW_DOWN}},
    });
    {
        SfxModel model(path.string(), {}, 1);
        check(all_finite(encode_text(model, {40, 41})), "a feed-forward sum past the f16 range keeps the context finite");
    }
    std::filesystem::remove(path);
}

void test_tokenizer(const SfxModel & model) {
    const SfxTokenizer tokenizer(model);
    check(tokenizer.encode("").empty(), "empty text has no tokens");
    check(tokenizer.encode("zyxwvutsrqponmlk").size() == (size_t) TEXT_TOKENS, "tokens truncate to the text length");
}

void test_dit(SfxModel & model) {
    const std::vector<float> latents = gaussian_noise((size_t) FRAMES * LATENT, 3);
    const std::vector<float> context = encode_text(model, {40, 41, 42});
    SfxDitSession dit(model, FRAMES);
    const std::vector<float> first = dit.velocity(latents, 999.0f, context);
    check(first.size() == latents.size(), "velocity matches the latent shape");
    check(dit.velocity(latents, 999.0f, context) == first, "a reused session is deterministic");
    check(max_abs_diff(dit.velocity(latents, 999.0f, encode_text(model, {})), first) > TOLERANCE,
            "the text context conditions the velocity");
    check(max_abs_diff(dit.velocity(latents, 10.0f, context), first) > TOLERANCE,
            "the timestep conditions the velocity");
    expect_failure([&] { dit.velocity({1.0f}, 1.0f, context); }, "wrong size", "short latent");
    encode_text(model, {40});
    expect_failure([&] { dit.velocity(latents, 999.0f, context); }, "no longer allocated",
            "a session whose scheduler another graph took is refused");
}

void test_vae(SfxModel & model) {
    const std::vector<float> latents = gaussian_noise((size_t) FRAMES * LATENT, 5);
    const int keep = KEPT_FRAMES;
    const std::vector<float> whole = decode_latents(model, latents, FRAMES, keep, FRAMES, {});
    check(whole.size() == (size_t) keep * HOP, "decode emits hop samples per kept frame");
    check(std::all_of(whole.begin(), whole.end(), [](float v) { return std::fabs(v) <= 1.0f; }),
            "tanh bounds the waveform");
    check(max_abs_diff(decode_latents(model, latents, FRAMES, keep, SMALL_WINDOW, {}), whole) < TOLERANCE,
            "windowed decoding matches a single window");
    int calls = 0;
    const std::vector<float> stopped = decode_latents(model, latents, FRAMES, keep, SMALL_WINDOW,
            [&](int, int) { return ++calls < 2; });
    check(stopped.empty() && calls == 2, "a false progress return stops decoding");
    expect_failure([&] { decode_latents(model, latents, FRAMES, FRAMES + 1, SMALL_WINDOW, {}); }, "invalid",
            "keeping more frames than exist");
}

std::vector<float> ggml_sinusoid(float timestep, int dim) {
    ggml_context * ctx = ggml_init({SINUSOID_ARENA_BYTES, nullptr, false});
    ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    *(float *) input->data = timestep;
    ggml_tensor * embedding = ggml_timestep_embedding(ctx, input, dim, TIMESTEP_MAX_PERIOD);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, embedding);
    ggml_graph_compute_with_ctx(ctx, graph, 1);
    std::vector<float> out((const float *) embedding->data, (const float *) embedding->data + dim);
    ggml_free(ctx);
    return out;
}

void test_timestep_sinusoid() {
    check(max_abs_diff(timestep_sinusoid(PROBE_TIMESTEP, FREQ_DIM), ggml_sinusoid(PROBE_TIMESTEP, FREQ_DIM)) == 0.0f,
          "the DiT sidecar's timestep embedding is ggml's");
}

void check_fixed_plan(int frames, int keep, int window, int context) {
    const std::string label = "fixed VAE windows " + std::to_string(frames) + "/" + std::to_string(keep) + "/" +
                              std::to_string(window) + ": ";
    int next = 0;
    for (const SfxVaeWindow & w : plan_fixed_vae_windows(frames, keep, window, context)) {
        check(w.count == window && w.first >= 0 && w.first + window <= frames, label + "every window has the width");
        check(w.first + w.keep_first == next, label + "kept frames tile the clip");
        check(w.keep_first >= context || w.first == 0, label + "left context or the clip start");
        check(w.keep_first + w.keep_count + context <= window || w.first + window == frames,
              label + "right context or the clip end");
        next += w.keep_count;
    }
    check(next == keep, label + "every kept frame is covered once");
}

void test_fixed_vae_plan() {
    check_fixed_plan(1500, 1500, 128, SFX_VAE_CONTEXT_FRAMES);
    check_fixed_plan(1500, 1, 128, SFX_VAE_CONTEXT_FRAMES);
    check_fixed_plan(1500, 97, 128, SFX_VAE_CONTEXT_FRAMES);
    check_fixed_plan(FRAMES, KEPT_FRAMES, FAKE_VAE_WINDOW, SFX_VAE_CONTEXT_FRAMES);
    check_fixed_plan(128, 128, 128, SFX_VAE_CONTEXT_FRAMES);
    check(plan_fixed_vae_windows(100, 50, 128, SFX_VAE_CONTEXT_FRAMES).empty(), "a window wider than the clip");
    check(plan_fixed_vae_windows(1500, 100, 64, SFX_VAE_CONTEXT_FRAMES).empty(), "a window with no room for a core");
}

class GgmlWindowVae final : public SfxVaeSidecar {
public:
    GgmlWindowVae(SfxModel & model, int fail_on_call, int * calls)
        : model_(model), fail_on_call_(fail_on_call), calls_(calls) {}

    int window() const override {
        return FAKE_VAE_WINDOW;
    }

    bool decode(const std::vector<float> & latents, std::vector<float> & pcm) override {
        if ((*calls_)++ == fail_on_call_) {
            return false;
        }
        pcm = decode_latents(model_, latents, FAKE_VAE_WINDOW, FAKE_VAE_WINDOW, FAKE_VAE_WINDOW, {});
        return true;
    }

    const char * label() const override {
        return FAKE_SIDECAR_LABEL;
    }

private:
    SfxModel & model_;
    int fail_on_call_;
    int * calls_;
};

void test_vae_sidecar_windows(SfxModel & model) {
    const std::vector<float> latents = gaussian_noise((size_t) FRAMES * LATENT, 5);
    const std::vector<float> whole = decode_latents(model, latents, FRAMES, KEPT_FRAMES, FRAMES, {});
    int calls = 0;
    std::unique_ptr<SfxVaeSidecar> sidecar = std::make_unique<GgmlWindowVae>(model, NEVER_FAIL, &calls);
    StageBackends backends;
    const std::vector<float> windowed = decode_sfx_latents(model, sidecar, false, backends, latents, FRAMES,
            KEPT_FRAMES, SMALL_WINDOW, {});
    check(max_abs_diff(windowed, whole) < TOLERANCE, "fixed sidecar windows reproduce the whole decode");
    check(sidecar && backends.summary() == FAKE_SIDECAR_LABEL, "the decode reports the sidecar");
    int stops = 0;
    check(decode_sfx_latents(model, sidecar, false, backends, latents, FRAMES, KEPT_FRAMES, SMALL_WINDOW,
            [&](int, int) { return ++stops < 2; }).empty() && sidecar, "a cancel stops the sidecar windows");
}

void test_vae_sidecar_fallback(SfxModel & model) {
    const std::vector<float> latents = gaussian_noise((size_t) FRAMES * LATENT, 5);
    const std::vector<float> ggml = decode_latents(model, latents, FRAMES, KEPT_FRAMES, SMALL_WINDOW, {});
    int calls = 0;
    std::unique_ptr<SfxVaeSidecar> sidecar = std::make_unique<GgmlWindowVae>(model, 1, &calls);
    StageBackends backends;
    check(decode_sfx_latents(model, sidecar, false, backends, latents, FRAMES, KEPT_FRAMES, SMALL_WINDOW, {}) ==
          ggml, "a failed window reruns the decode on ggml");
    check(!sidecar && backends.summary() == GGML_STAGE_BACKEND, "a failed VAE sidecar is retired");
    int strict_calls = 0;
    std::unique_ptr<SfxVaeSidecar> failing = std::make_unique<GgmlWindowVae>(model, 0, &strict_calls);
    expect_failure([&] { decode_sfx_latents(model, failing, true, backends, latents, FRAMES, KEPT_FRAMES,
            SMALL_WINDOW, {}); }, COREML_STRICT_ENV, "a failed window fails the decode under MOSS_COREML_STRICT");
    std::unique_ptr<SfxVaeSidecar> absent;
    expect_failure([&] { decode_sfx_latents(model, absent, true, backends, latents, FRAMES, KEPT_FRAMES,
            SMALL_WINDOW, {}); }, COREML_STRICT_ENV, "a missing VAE sidecar fails under MOSS_COREML_STRICT");
}

class OffsetDit final : public SfxDitSidecar {
public:
    OffsetDit(std::vector<float> velocity, int fail_on_call, int * calls, std::vector<float> * timestep)
        : velocity_(std::move(velocity)), fail_on_call_(fail_on_call), calls_(calls), timestep_(timestep) {}

    bool velocity(const std::vector<float> &, const std::vector<float> & timestep, const std::vector<float> &,
                  std::vector<float> & out) override {
        *timestep_ = timestep;
        if ((*calls_)++ == fail_on_call_) {
            return false;
        }
        out = velocity_;
        return true;
    }

    const char * label() const override {
        return FAKE_SIDECAR_LABEL;
    }

private:
    std::vector<float> velocity_;
    int fail_on_call_;
    int * calls_;
    std::vector<float> * timestep_;
};

std::vector<float> offset(const std::vector<float> & values, float delta) {
    std::vector<float> out(values);
    std::transform(out.begin(), out.end(), out.begin(), [delta](float v) { return v + delta; });
    return out;
}

void test_dit_runner(SfxModel & model) {
    const std::vector<float> latents = gaussian_noise((size_t) FRAMES * LATENT, 3);
    const std::vector<float> context = encode_text(model, {40, 41, 42});
    std::vector<float> ggml;
    const std::vector<float> negative(context.size(), 0.0f);
    std::vector<float> ggml_negative;
    {
        SfxDitSession reference(model, FRAMES);
        ggml = reference.velocity(latents, PROBE_TIMESTEP, context);
        ggml_negative = reference.velocity(latents, PROBE_TIMESTEP, negative);
    }
    int calls = 0;
    std::vector<float> seen_timestep;
    std::unique_ptr<SfxDitSidecar> sidecar =
            std::make_unique<OffsetDit>(offset(ggml, FAKE_VELOCITY_OFFSET), 1, &calls, &seen_timestep);
    StageBackends backends;
    SfxDitRunner runner(model, FRAMES, sidecar, false, backends);
    runner.prepare_contexts({context, negative});
    check(runner.velocity(latents, PROBE_TIMESTEP, context) == offset(ggml, FAKE_VELOCITY_OFFSET),
          "an attached DiT sidecar predicts the velocity");
    check(seen_timestep == timestep_sinusoid(PROBE_TIMESTEP, FREQ_DIM), "the sidecar gets the timestep embedding");
    check(max_abs_diff(runner.velocity(latents, PROBE_TIMESTEP, context), ggml) < TOLERANCE,
          "a failed step continues on the ggml DiT");
    check(!sidecar && backends.summary() == MIXED_STAGE_BACKEND, "a failed DiT sidecar is retired mid-run");
    check(calls == 2 && max_abs_diff(runner.velocity(latents, PROBE_TIMESTEP, context), ggml) < TOLERANCE,
          "later steps stay on ggml");
    check(max_abs_diff(runner.velocity(latents, PROBE_TIMESTEP, negative), ggml_negative) < TOLERANCE,
          "a failed DiT sidecar prepares and selects the negative cached branch");
    std::unique_ptr<SfxDitSidecar> absent;
    SfxDitRunner strict(model, FRAMES, absent, true, backends);
    expect_failure([&] { strict.velocity(latents, PROBE_TIMESTEP, context); }, COREML_STRICT_ENV,
                   "a missing DiT sidecar fails under MOSS_COREML_STRICT");
}

void test_sidecar_lookup(const std::filesystem::path & path, const SfxModel & model) {
    check(sfx_dit_sidecar_path("models/moss-sfx-v2-q8_0.gguf") == "models/moss-sfx-v2-dit.mlmodelc" &&
          sfx_vae_sidecar_path("models/moss-sfx-v2-f16.gguf") == "models/moss-sfx-v2-vae.mlmodelc",
          "the sidecar names drop the quantization tag");
    check(!open_sfx_dit_sidecar(path.string(), model.config(), {}) && !open_sfx_vae_sidecar(path.string(),
          model.config(), {}), "no staged sidecars keep both stages on ggml");
    const std::filesystem::path dit = sfx_dit_sidecar_path(path.string());
    std::filesystem::create_directories(dit);
    check(!open_sfx_dit_sidecar(path.string(), model.config(), {true, false}), "MOSS_COREML_DISABLE skips the DiT");
    check(!open_sfx_dit_sidecar(path.string(), model.config(), {}), "a directory that is not a model is ignored");
    std::filesystem::remove_all(dit);
}

void test_compute_units() {
    unsetenv(COREML_UNITS_ENV);
    check(requested_coreml_units() == COREML_DEFAULT_UNITS, "the MOSS sidecars default to the GPU");
    setenv(COREML_UNITS_ENV, ALL_UNITS, 1);
    check(requested_coreml_units() == ALL_UNITS, "MOSS_COREML_COMPUTE_UNITS picks the units");
    setenv(COREML_UNITS_ENV, UNKNOWN_UNITS, 1);
    check(requested_coreml_units() == COREML_DEFAULT_UNITS, "an unknown value keeps the default");
    unsetenv(COREML_UNITS_ENV);
}

tts_cpp::moss::SoundEffectRequest request(const char * prompt, double seconds, uint32_t seed) {
    tts_cpp::moss::SoundEffectRequest out;
    out.prompt = prompt;
    out.seconds = seconds;
    out.seed = seed;
    return out;
}

constexpr float SHORT_CLIP = 0.1f;
constexpr size_t SHORT_CLIP_SAMPLES = SAMPLE_RATE / 10;
constexpr float TOO_LONG_CLIP = 0.3f;
constexpr int TOO_MANY_STEPS = 5000;
constexpr float WEAK_GUIDANCE = 0.5f;
constexpr size_t MAX_PROMPT_BYTES_UNDER_TEST = 8192;
constexpr uint32_t SPEECH_DEFAULT_SEED = 1234;
constexpr uintmax_t WAV_HEADER_BYTES = 44;
constexpr uintmax_t PCM16_BYTES = 2;

tts_cpp::moss::SoundEffectOptions engine_options(const std::filesystem::path & path) {
    tts_cpp::moss::SoundEffectOptions options;
    options.model_path = path.string();
    options.n_threads = 1;
    return options;
}

void test_engine_generation(tts_cpp::moss::SoundEffectEngine & engine) {
    check(engine.sample_rate() == SAMPLE_RATE && engine.max_seconds() == MAX_SECONDS, "engine reports the model");
    int progress_calls = 0;
    const auto result = engine.generate(request("rain", SHORT_CLIP, 1), [&](int step, int total) {
        progress_calls++;
        return step <= total;
    });
    check(!result.cancelled && result.pcm.size() == SHORT_CLIP_SAMPLES, "output is cropped to the seconds asked");
    check(progress_calls == STEPS, "progress reports every diffusion step");
    check(engine.generate(request("rain", SHORT_CLIP, 1)).pcm == result.pcm, "the same seed reproduces the clip");
    check(engine.generate(request("rain", SHORT_CLIP, 2)).pcm != result.pcm, "a different seed changes the clip");
}

void test_engine_guidance(tts_cpp::moss::SoundEffectEngine & engine) {
    auto unguided = request("rain", SHORT_CLIP, 1);
    unguided.guidance = 1.0f;
    auto unguided_other_negative = unguided;
    unguided_other_negative.negative_prompt = "music";
    const auto plain = engine.generate(unguided).pcm;
    check(engine.generate(unguided_other_negative).pcm == plain, "guidance 1 ignores the negative prompt");
    auto guided = unguided_other_negative;
    guided.guidance = 4.0f;
    check(engine.generate(guided).pcm != plain, "guidance above 1 uses the negative prompt");
}

void test_engine_progress_cancel(tts_cpp::moss::SoundEffectEngine & engine) {
    const auto stopped = engine.generate(request("rain", SHORT_CLIP, 1), [](int, int) { return false; });
    check(stopped.cancelled && stopped.pcm.empty(), "returning false from progress cancels");
}

void test_engine_threaded_cancel(tts_cpp::moss::SoundEffectEngine & engine) {
    std::promise<void> started;
    std::promise<void> released;
    std::shared_future<void> release = released.get_future().share();
    std::thread worker;
    tts_cpp::moss::SoundEffectResult result;
    worker = std::thread([&] {
        result = engine.generate(request("rain", SHORT_CLIP, 1), [&](int step, int) {
            if (step == 1) {
                started.set_value();
                release.wait();
            }
            return true;
        });
    });
    started.get_future().wait();
    expect_failure([&] { engine.generate(request("rain", SHORT_CLIP, 1)); }, "already in progress",
            "a second generation while one runs");
    engine.cancel();
    released.set_value();
    worker.join();
    check(result.cancelled && result.pcm.empty(), "cancel() from another thread stops the running generation");
    check(!engine.generate(request("rain", SHORT_CLIP, 1)).cancelled, "the next generation starts uncancelled");
}

void test_engine_validation(tts_cpp::moss::SoundEffectEngine & engine) {
    expect_failure([&] { engine.generate(request("   ", SHORT_CLIP, 1)); }, "prompt", "blank prompt");
    expect_failure([&] { engine.generate(request("rain", TOO_LONG_CLIP, 1)); }, "seconds", "duration beyond the model");
    expect_failure([&] { engine.generate(request("rain", 0.0f, 1)); }, "seconds", "zero duration");
    expect_failure([&] { engine.generate(request("rain", NAN, 1)); }, "seconds", "NaN duration");
    auto weak = request("rain", SHORT_CLIP, 1);
    weak.guidance = WEAK_GUIDANCE;
    expect_failure([&] { engine.generate(weak); }, "guidance", "guidance below one");
    auto negative_guidance = request("rain", SHORT_CLIP, 1);
    negative_guidance.guidance = -2.0f;
    expect_failure([&] { engine.generate(negative_guidance); }, "guidance", "negative guidance is not the default");
    auto negative_steps = request("rain", SHORT_CLIP, 1);
    negative_steps.steps = -5;
    expect_failure([&] { engine.generate(negative_steps); }, "steps", "negative steps are not the default");
    auto strong = request("rain", SHORT_CLIP, 1);
    strong.guidance = 51.0f;
    expect_failure([&] { engine.generate(strong); }, "guidance", "guidance past the maximum");
    auto big_shift = request("rain", SHORT_CLIP, 1);
    big_shift.shift = 101.0f;
    expect_failure([&] { engine.generate(big_shift); }, "shift", "shift past the maximum");
    auto negative_shift = request("rain", SHORT_CLIP, 1);
    negative_shift.shift = -1.0f;
    expect_failure([&] { engine.generate(negative_shift); }, "shift", "negative shift is not the default");
    expect_failure([&] { engine.generate(request("rain", INFINITY, 1)); }, "seconds", "infinite duration");
    auto nan_shift = request("rain", SHORT_CLIP, 1);
    nan_shift.shift = NAN;
    expect_failure([&] { engine.generate(nan_shift); }, "shift", "NaN shift");
    const std::string huge(MAX_PROMPT_BYTES_UNDER_TEST + 1, 'a');
    expect_failure([&] { engine.generate(request(huge.c_str(), SHORT_CLIP, 1)); }, "bytes", "oversized prompt");
    auto huge_negative = request("rain", SHORT_CLIP, 1);
    huge_negative.negative_prompt = huge;
    expect_failure([&] { engine.generate(huge_negative); }, "bytes", "oversized negative prompt");
    auto too_many_steps = request("rain", SHORT_CLIP, 1);
    too_many_steps.steps = TOO_MANY_STEPS;
    expect_failure([&] { engine.generate(too_many_steps); }, "steps", "too many steps");
}

void test_engine_placement(tts_cpp::moss::SoundEffectEngine & engine) {
    const auto result = engine.generate(request("rain", SHORT_CLIP, 1));
    check(!engine.dit_on_coreml() && !engine.vae_on_coreml(), "an engine without sidecars stays on ggml");
    check(result.dit_backend == GGML_STAGE_BACKEND && result.vae_backend == GGML_STAGE_BACKEND,
          "the result reports where the DiT and the VAE ran");
}

void test_engine(const std::filesystem::path & path) {
    tts_cpp::moss::SoundEffectEngine engine(engine_options(path));
    test_engine_placement(engine);
    test_engine_generation(engine);
    test_engine_guidance(engine);
    test_engine_progress_cancel(engine);
    test_engine_threaded_cancel(engine);
    test_engine_validation(engine);
}

bool parse(const std::vector<const char *> & argv, tts_cpp::moss::cli::CliArgs & args) {
    return tts_cpp::moss::cli::parse_args((int) argv.size(), argv.data(), args);
}

void test_positional_options() {
    const tts_cpp::moss::SoundEffectOptions options{POSITIONAL_MODEL, POSITIONAL_THREADS, true,
                                                    POSITIONAL_BACKENDS_DIR};
    check(options.model_path == POSITIONAL_MODEL && options.n_threads == POSITIONAL_THREADS && options.use_gpu,
          "four-field initialization keeps the leading options");
    check(options.backends_dir == POSITIONAL_BACKENDS_DIR && options.backend.empty(),
          "four-field initialization sets backends_dir and leaves the backend automatic");
}

void test_cli(const std::filesystem::path & path) {
    using tts_cpp::moss::cli::CliArgs;
    using tts_cpp::moss::cli::Mode;
    CliArgs args;
    check(parse({"moss-cli", "--mode", "sfx", "--model", "m.gguf", "--text", "rain", "--seconds", "3",
            "--steps", "20", "--guidance", "3.5", "--shift", "4", "--seed", "9", "--negative-prompt", "music",
            "--threads", "2", "--gpu"}, args), "sfx flags parse");
    check(args.mode == Mode::SoundEffect && args.sound_options.model_path == "m.gguf", "sfx mode and model map");
    check(args.sound_request.prompt == "rain" && args.sound_request.seconds == 3.0f &&
            args.sound_request.steps == 20 && args.sound_request.guidance == 3.5f &&
            args.sound_request.shift == 4.0f && args.sound_request.seed == 9 &&
            args.sound_request.negative_prompt == "music", "sfx request flags map");
    check(args.sound_options.n_threads == 2 && args.sound_options.use_gpu, "runtime flags reach the sfx engine");

    CliArgs no_model;
    check(!parse({"moss-cli", "--mode", "sfx", "--text", "rain"}, no_model), "sfx needs --model");
    CliArgs streaming;
    check(!parse({"moss-cli", "--mode", "sfx", "--model", "m.gguf", "--text", "rain", "--stream"}, streaming),
            "sfx does not stream");
    CliArgs speech;
    check(parse({"moss-cli", "--mode", "tts", "--backbone", "b.gguf", "--decoder", "d.gguf", "--text", "hi"}, speech) &&
            speech.mode == Mode::Speech && speech.options.seed == SPEECH_DEFAULT_SEED,
            "--mode tts keeps the speech engine and its seed default");
    CliArgs sfx_without_speech_models;
    check(!parse({"moss-cli", "--mode", "tts", "--model", "m.gguf", "--text", "rain"}, sfx_without_speech_models),
            "tts mode still needs the backbone and decoder");
    CliArgs bad_mode;
    check(!parse({"moss-cli", "--mode", "music", "--text", "rain"}, bad_mode), "unknown modes are rejected");

    CliArgs run_args;
    const std::string out_wav = moss_fixtures::temp_gguf("sfx-cli").replace_extension(".wav").string();
    check(parse({"moss-cli", "--mode", "sfx", "--model", path.string().c_str(), "--text", "rain", "--seconds",
            "0.1", "--threads", "1", "--out", out_wav.c_str()}, run_args), "end-to-end flags parse");
    check(tts_cpp::moss::cli::run(run_args) == 0, "sfx mode runs on the fixture model");
    check(std::filesystem::exists(out_wav) && std::filesystem::file_size(out_wav) == WAV_HEADER_BYTES + SHORT_CLIP_SAMPLES * PCM16_BYTES,
            "sfx mode writes the requested duration as 16-bit mono");
    run_args.sound_options.backend = "cpu";
    check(tts_cpp::moss::cli::run(run_args) == 0, "sfx mode runs on an explicit cpu backend");
    run_args.sound_options.backend = "no-such-device";
    check(tts_cpp::moss::cli::run(run_args) != 0, "sfx mode fails on an unavailable backend instead of falling back");
    std::filesystem::remove(out_wav);
}

} // namespace

int main() {
    int rc = 0;
    try {
        test_schedule();
        test_euler_and_guidance();
        test_prompt_text();
        test_noise();
        test_rejected_models();
        test_rejected_tensors();
        test_weight_tiers();
        test_rejected_metadata_bounds();
        test_feed_forward_beyond_half_precision();
        test_timestep_sinusoid();
        test_fixed_vae_plan();
        test_compute_units();
        test_positional_options();
        const auto path = write_sfx_model("sfx-model", WEIGHT_SEED);
        {
            SfxModel model(path.string(), {}, 1);
            test_model_config(model);
            test_tokenizer(model);
            test_text_encoder(model);
            test_dit(model);
            test_vae(model);
            test_vae_sidecar_windows(model);
            test_vae_sidecar_fallback(model);
            test_dit_runner(model);
            test_sidecar_lookup(path, model);
        }
        test_engine(path);
        test_cli(path);
        std::filesystem::remove(path);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        rc = 1;
    }
    if (rc != 0) {
        return rc;
    }
    if (failures == 0) {
        std::printf("moss sfx: OK\n");
        return 0;
    }
    std::fprintf(stderr, "moss sfx: %d failures\n", failures);
    return 1;
}
