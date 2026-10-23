#include "minimax/logic.h"
#include "minimax/backend.h"
#include "minimax/bpe.h"
#include "minimax/mm3-depth-graph.h"
#include "minimax/mm3-dit-graph.h"
#include "minimax/mm3-flash-attn.h"
#include "minimax/mm3-flow-runtime.h"
#include "minimax/mm3-linear.h"
#include "minimax/mm3-lm-graph.h"
#include "minimax/mm3-replay-io.h"
#include "minimax/mm3-window-orchestrator.h"
#include "minimax/progress.h"
#include "minimax/request-utils.h"

#include "ggml-alloc.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;
int checks = 0;

#define CHECK(condition)                                                                      \
    do {                                                                                      \
        ++checks;                                                                             \
        if (!(condition)) {                                                                   \
            ++failures;                                                                       \
            std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #condition);          \
        }                                                                                     \
    } while (0)

template <typename Function>
bool throws_invalid_argument(Function function) {
    try {
        function();
    } catch (const std::invalid_argument &) {
        return true;
    }
    return false;
}

template <typename Function>
bool throws_runtime_error(Function function) {
    try {
        function();
    } catch (const std::runtime_error &) {
        return true;
    }
    return false;
}

// MSVC has no POSIX setenv/unsetenv; _putenv_s(key, "") removes the variable,
// which matches backend_configure_device treating an empty value as unset.
void set_env(const char * key, const char * value) {
#ifdef _WIN32
    _putenv_s(key, value ? value : "");
#else
    if (value) {
        setenv(key, value, 1);
    } else {
        unsetenv(key);
    }
#endif
}

bool close(float left, float right, float tolerance = 1e-6f) {
    return std::fabs(left - right) <= tolerance;
}

tts_cpp::minimax::detail::ModelCompatibility valid_compatibility() {
    using namespace tts_cpp::minimax::detail;
    ModelCompatibility model;
    model.lm_embedding = 4096;
    model.lm_codebooks = 8;
    model.lm_acoustic_vocab = 1024;
    model.frame_rate = 25;
    model.max_audio_frames = 9000;
    model.max_prompt_tokens = 5000;
    model.depth_embedding = 4096;
    model.depth_codebooks = 8;
    model.depth_acoustic_vocab = 1024;
    model.condition_layers = 8;
    model.condition_hidden = 4096;
    model.condition_out = 2048;
    model.condition_rate = {24000, 960, 44100, 512};
    model.dit_condition = 2048;
    model.dit_channels = 128;
    model.window_frames = 200;
    model.hop_frames = 100;
    model.window_latents = 689;
    model.hop_latents = 344;
    model.vocoder_latent_channels = 128;
    model.vocoder_sampling_rate = 44100;
    model.vocoder_channels = 2;
    model.vocoder_upsample = 512;
    model.components = {"depth", "cond", "dit", "vocoder"};
    return model;
}

void touch(const std::filesystem::path & path) {
    std::ofstream stream(path);
    stream << "test";
}

void test_frame_validation() {
    using namespace tts_cpp::minimax::detail;
    CHECK(kDefaultFrameRate == 25);
    CHECK(kDefaultMaxFrames == 9000);
    CHECK(frames_from_duration(12.0, 25, 9000) == 300);
    CHECK(frames_from_duration(1000.0, 25, 9000) == 9000);
    CHECK(validate_frames(1, 9000) == 1);
    CHECK(validate_frames(9001, 9000) == 9000);
    CHECK(throws_invalid_argument([] { frames_from_duration(0.0, 25, 9000); }));
    CHECK(throws_invalid_argument([] { frames_from_duration(std::numeric_limits<double>::infinity(), 25, 9000); }));
    CHECK(throws_invalid_argument([] { validate_frames(0, 9000); }));
}

void test_prompt() {
    using tts_cpp::minimax::detail::build_prompt;
    const std::string prompt = build_prompt("## **Bright** pop", "[Verse] ignored\nHello ^ world");
    CHECK(prompt == "<|im_start|><|caption_start|>Bright pop<|caption_end|><|lyrics_start|>"
                    "[start]\n[verse]\nHello\nworld<|lyrics_end|><|im_end|><|audio_start|>");
    CHECK(build_prompt("Instrumental piano", "") ==
          "<|im_start|><|caption_start|>Instrumental piano<|caption_end|><|lyrics_start|>"
          "[start]\n[instrumental]<|lyrics_end|><|im_end|><|audio_start|>");
    CHECK(throws_invalid_argument([] { build_prompt(" \n\t", "words"); }));
}

void test_request_text_primitives() {
    CHECK(mm3_is_space(' '));
    CHECK(mm3_is_space('\t'));
    CHECK(mm3_is_space('\r'));
    CHECK(mm3_is_space('\v'));
    CHECK(mm3_is_space('\f'));
    CHECK(!mm3_is_space('\n'));
    CHECK(!mm3_is_space('a'));

    CHECK(mm3_rstrip("a b \t\n") == "a b");
    CHECK(mm3_rstrip("  x") == "  x");
    CHECK(mm3_rstrip("").empty());

    CHECK(mm3_strip("\n\t hi \n") == "hi");
    CHECK(mm3_strip("hi") == "hi");
    CHECK(mm3_strip(" \t ").empty());
    CHECK(mm3_strip("").empty());

    CHECK(mm3_str_blank(""));
    CHECK(mm3_str_blank(" \t\n"));
    CHECK(!mm3_str_blank(" a "));
}

void test_request_line_splitting() {
    CHECK(mm3_split_lines("") == std::vector<std::string>({""}));
    CHECK(mm3_split_lines("a\nb") == std::vector<std::string>({"a", "b"}));
    CHECK(mm3_split_lines("a\n") == std::vector<std::string>({"a", ""}));
    CHECK(mm3_split_lines("a\r\nb") == std::vector<std::string>({"a\r", "b"}));

    CHECK(mm3_splitlines("").empty());
    CHECK(mm3_splitlines("a\r\nb") == std::vector<std::string>({"a", "b"}));
    CHECK(mm3_splitlines("a\rb") == std::vector<std::string>({"a", "b"}));
    CHECK(mm3_splitlines("a\n") == std::vector<std::string>({"a"}));
    CHECK(mm3_splitlines("\n\n") == std::vector<std::string>({"", ""}));

    CHECK(mm3_join_lines({}).empty());
    CHECK(mm3_join_lines({"a"}) == "a");
    CHECK(mm3_join_lines({"a", "", "b"}) == "a\n\nb");
    CHECK(mm3_join_lines(mm3_split_lines("a\nb\nc")) == "a\nb\nc");
}

void test_request_replace_and_tags() {
    std::string text = "aaa";
    mm3_replace_all(text, "aa", "b");
    CHECK(text == "ba");
    text = "ab";
    mm3_replace_all(text, "a", "aa");
    CHECK(text == "aab");
    text = "x";
    mm3_replace_all(text, "", "y");
    CHECK(text == "x");

    CHECK(mm3_rewrite_special_tags("tag: <|genre pop|> end") == "tag: genre is pop end");
    CHECK(mm3_rewrite_special_tags("<|instrumental|>") == "instrumental");
    CHECK(mm3_rewrite_special_tags("<|bpm  120|>") == "bpm is 120");
    CHECK(mm3_rewrite_special_tags("<|a b c|>") == "a is b c");
    CHECK(mm3_rewrite_special_tags("<||>").empty());
    CHECK(mm3_rewrite_special_tags("<|unclosed") == "<|unclosed");
    CHECK(mm3_rewrite_special_tags("plain") == "plain");
    CHECK(mm3_rewrite_special_tags("").empty());

    std::string tags;
    CHECK(mm3_leading_tags("[Verse] hello", &tags));
    CHECK(tags == "[Verse]");
    CHECK(mm3_leading_tags("  [A] [B]x", &tags));
    CHECK(tags == "[A] [B]");
    CHECK(!mm3_leading_tags("no tags", &tags));
    CHECK(!mm3_leading_tags("[]", &tags));
    CHECK(!mm3_leading_tags("[unclosed", &tags));
    CHECK(!mm3_leading_tags("", &tags));

    CHECK(mm3_lower_tags("[VERSE 1] La [CHORUS]") == "[verse 1] La [chorus]");
    CHECK(mm3_lower_tags("[]") == "[]");
    CHECK(mm3_lower_tags("[open") == "[open");
    CHECK(mm3_lower_tags("none") == "none");
}

void test_request_markdown_helpers() {
    std::string line = "## Title";
    mm3_strip_heading(line);
    CHECK(line == "Title");
    line = "  # T";
    mm3_strip_heading(line);
    CHECK(line == "T");
    line = "####### x";  // 7 hashes: beyond the h1-h6 window, kept verbatim
    mm3_strip_heading(line);
    CHECK(line == "####### x");
    line = "#Title";
    mm3_strip_heading(line);
    CHECK(line == "#Title");
    line = "    # x";  // 4-space indent is a code block, not a heading
    mm3_strip_heading(line);
    CHECK(line == "    # x");

    line = "- item";
    mm3_strip_bullet(line, "*+-");
    CHECK(line == "item");
    line = "  + item";
    mm3_strip_bullet(line, "*+-");
    CHECK(line == "item");
    line = "-item";
    mm3_strip_bullet(line, "*+-");
    CHECK(line == "-item");
    line = "*";
    mm3_strip_bullet(line, "*");
    CHECK(line == "*");

    CHECK(mm3_unwrap_bold_once("**bold**") == "bold");
    CHECK(mm3_unwrap_bold_once("a **b** c **d**") == "a b c d");
    CHECK(mm3_unwrap_bold_once("****") == "****");
    CHECK(mm3_unwrap_bold_once("**open") == "**open");
    CHECK(mm3_unwrap_bold_once("plain") == "plain");

    CHECK(mm3_unwrap_italic("*it*") == "it");
    CHECK(mm3_unwrap_italic("a*b*c") == "abc");
    CHECK(mm3_unwrap_italic("**x**") == "**x**");
    CHECK(mm3_unwrap_italic("*a\nb*") == "*a\nb*");
    CHECK(mm3_unwrap_italic("*") == "*");

    CHECK(mm3_is_hrule("---"));
    CHECK(mm3_is_hrule(" *** "));
    CHECK(mm3_is_hrule("___"));
    CHECK(mm3_is_hrule("-*_"));  // mixed marks count as one run
    CHECK(!mm3_is_hrule("--"));
    CHECK(!mm3_is_hrule("- - -"));
    CHECK(!mm3_is_hrule("--- x"));
    CHECK(!mm3_is_hrule(""));
}

void test_request_clean_caption() {
    CHECK(mm3_clean_caption("## **Bright** pop") == "Bright pop");
    CHECK(mm3_clean_caption("# Title\n\n- **bold** item\n---\n*soft* line   \n\n\nnext") ==
          "Title\nbold item\nsoft line\nnext");
    CHECK(mm3_clean_caption("\xE2\x80\xA2 point") == "point");
    CHECK(mm3_clean_caption("a    b") == "ab");
    CHECK(mm3_clean_caption("<|genre rock|>") == "genre is rock");
    CHECK(mm3_clean_caption("plain caption") == "plain caption");
    CHECK(mm3_clean_caption("").empty());
}

void test_request_normalize_lyrics() {
    CHECK(mm3_normalize_lyrics("[Verse] ignored\nHello ^ world") ==
          "[start]\n[verse]\nHello\nworld");
    CHECK(mm3_normalize_lyrics("[Intro]  \nLa la [Chorus] la\n\nend") ==
          "[start]\n[intro]\nLa la\n[chorus]\nla\n\nend");
    CHECK(mm3_normalize_lyrics("[instrumental]") == "[start]\n[instrumental]");
    CHECK(mm3_normalize_lyrics("a ^ b") == "[start]\na\nb");
    CHECK(mm3_normalize_lyrics("") == "[start]\n");
}

void test_request_assemble_prompt() {
    using tts_cpp::minimax::detail::build_prompt;

    bool instrumental = false;
    CHECK(mm3_assemble_prompt("Piano", " \n\t", &instrumental) ==
          "<|im_start|><|caption_start|>Piano<|caption_end|><|lyrics_start|>"
          "[start]\n[instrumental]<|lyrics_end|><|im_end|><|audio_start|>");
    CHECK(instrumental);

    CHECK(mm3_assemble_prompt("- **Jazz** trio", "[INTRO] hum", &instrumental) ==
          "<|im_start|><|caption_start|>Jazz trio<|caption_end|><|lyrics_start|>"
          "[start]\n[intro]<|lyrics_end|><|im_end|><|audio_start|>");
    CHECK(!instrumental);

    CHECK(build_prompt("Calm <|genre ambient|> pad", "la la") ==
          "<|im_start|><|caption_start|>Calm genre is ambient pad<|caption_end|><|lyrics_start|>"
          "[start]\nla la<|lyrics_end|><|im_end|><|audio_start|>");
}

void test_unconditional_mask() {
    using tts_cpp::minimax::detail::mask_unconditional;
    const std::vector<int32_t> conditional = {10, 11, 12, 13, 14, 15};
    CHECK(mask_unconditional(conditional, 99) == std::vector<int32_t>({10, 99, 99, 99, 14, 15}));
    CHECK(mask_unconditional({1, 2, 3}, 99) == std::vector<int32_t>({1, 2, 3}));
}

void test_noise() {
    using tts_cpp::minimax::detail::fill_noise;
    std::vector<float> first;
    std::vector<float> second;
    std::vector<float> other;
    fill_noise(42, 0, first, 8);
    fill_noise(42, 0, second, 8);
    fill_noise(42, 1, other, 8);
    CHECK(first == second);
    CHECK(first != other);
    CHECK(close(first[0], -0.19663458f));
    CHECK(close(first[1], -0.25129682f));
    CHECK(throws_invalid_argument([&] { fill_noise(42, -1, first, 8); }));
}

void test_depth_step_layout() {
    using tts_cpp::minimax::detail::depth_step_layout;
    // The seeding step writes the projected LM hidden and the semantic code;
    // every later step appends exactly one acoustic code.
    CHECK(depth_step_layout(1).window == 2);
    CHECK(depth_step_layout(1).tokens == 2);
    CHECK(depth_step_layout(1).first == 0);
    CHECK(depth_step_layout(2).window == 3);
    CHECK(depth_step_layout(2).tokens == 1);
    CHECK(depth_step_layout(2).first == 2);
    CHECK(depth_step_layout(7).window == 8);
    CHECK(depth_step_layout(7).tokens == 1);
    CHECK(depth_step_layout(7).first == 7);
    CHECK(throws_invalid_argument([] { depth_step_layout(0); }));

    // Each step may only read positions earlier steps of the same frame wrote,
    // so the steps must cover 0..NC exactly once with no gap: a gap would leave
    // a step attending over a position held from the previous frame.
    for (int codebooks = 1; codebooks < 16; codebooks++) {
        int64_t next = 0;
        for (int cb = 1; cb <= codebooks; cb++) {
            const auto layout = depth_step_layout(cb);
            CHECK(layout.first == next);
            CHECK(layout.window == layout.first + layout.tokens);
            next += layout.tokens;
        }
        CHECK(next == (int64_t) codebooks + 1);
    }
}

// The cached block must reproduce, position by position, what one full-prefix
// pass over the same tokens produces: seed two positions, append one, append
// one, and every output plus the cache contents must match the single pass.
// A wrong row index, window, mask or a stale position would all show here.
void test_depth_kv_cache_matches_full_prefix() {
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    CHECK(cpu != nullptr);
    if (!cpu) return;

    const int64_t H  = 8;
    const int64_t D  = 4;
    const int64_t Nh = 2;
    const int64_t FF = 16;
    const int64_t S  = 4;
    const int64_t B  = 2;

    MM3DepthConfig config;
    config.block_count         = 1;
    config.embedding_length    = static_cast<uint32_t>(H);
    config.feed_forward_length = static_cast<uint32_t>(FF);
    config.head_count          = static_cast<uint32_t>(Nh);
    config.head_dim            = static_cast<uint32_t>(D);
    config.rms_eps             = 1e-6f;

    ggml_init_params weight_params = { ggml_tensor_overhead() * 40, nullptr, true };
    ggml_context *   wctx          = ggml_init(weight_params);
    CHECK(wctx != nullptr);
    auto matrix = [&](int64_t in, int64_t out) { return ggml_new_tensor_2d(wctx, GGML_TYPE_F32, in, out); };
    auto vec    = [&](int64_t n) { return ggml_new_tensor_1d(wctx, GGML_TYPE_F32, n); };
    auto cache  = [&]() { return ggml_new_tensor_4d(wctx, GGML_TYPE_F32, D, S, Nh, B); };

    MM3DepthLayer layer;
    layer.attn_norm   = vec(H);
    layer.attn_q      = matrix(H, H);
    layer.attn_k      = matrix(H, H);
    layer.attn_v      = matrix(H, H);
    layer.attn_output = matrix(H, H);
    layer.ffn_norm    = vec(H);
    layer.ffn_gate    = matrix(H, FF);
    layer.ffn_up      = matrix(H, FF);
    layer.ffn_down    = matrix(FF, H);

    ggml_tensor * k_full = cache();
    ggml_tensor * v_full = cache();
    ggml_tensor * k_inc  = cache();
    ggml_tensor * v_inc  = cache();

    ggml_tensor * h_full  = ggml_new_tensor_3d(wctx, GGML_TYPE_F32, H, S, B);
    ggml_tensor * h_seed  = ggml_new_tensor_3d(wctx, GGML_TYPE_F32, H, 2, B);
    ggml_tensor * h_step2 = ggml_new_tensor_3d(wctx, GGML_TYPE_F32, H, 1, B);
    ggml_tensor * h_step3 = ggml_new_tensor_3d(wctx, GGML_TYPE_F32, H, 1, B);

    ggml_tensor * rows_full  = ggml_new_tensor_1d(wctx, GGML_TYPE_I64, S);
    ggml_tensor * rows_seed  = ggml_new_tensor_1d(wctx, GGML_TYPE_I64, 2);
    ggml_tensor * rows_step2 = ggml_new_tensor_1d(wctx, GGML_TYPE_I64, 1);
    ggml_tensor * rows_step3 = ggml_new_tensor_1d(wctx, GGML_TYPE_I64, 1);

    ggml_tensor * mask_full = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, S, S);
    ggml_tensor * mask_seed = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, 2, 2);

    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, cpu);
    CHECK(wbuf != nullptr);

    std::mt19937 random(2026);
    std::uniform_real_distribution<float> weight(-0.5f, 0.5f);
    auto fill = [&](ggml_tensor * t, float base) {
        std::vector<float> values(static_cast<size_t>(ggml_nelements(t)));
        for (float & value : values) value = base + weight(random);
        ggml_backend_tensor_set(t, values.data(), 0, values.size() * sizeof(float));
        return values;
    };
    for (ggml_tensor * t : { layer.attn_q, layer.attn_k, layer.attn_v, layer.attn_output,
                             layer.ffn_gate, layer.ffn_up, layer.ffn_down }) {
        fill(t, 0.0f);
    }
    fill(layer.attn_norm, 1.0f);
    fill(layer.ffn_norm, 1.0f);
    const std::vector<float> input = fill(h_full, 0.0f);

    auto slice = [&](ggml_tensor * dst, int64_t first, int64_t count) {
        std::vector<float> values(static_cast<size_t>(H * count * B));
        for (int64_t b = 0; b < B; b++) {
            for (int64_t s = 0; s < count; s++) {
                std::copy_n(input.begin() + (b * S + first + s) * H, H, values.begin() + (b * count + s) * H);
            }
        }
        ggml_backend_tensor_set(dst, values.data(), 0, values.size() * sizeof(float));
    };
    slice(h_seed, 0, 2);
    slice(h_step2, 2, 1);
    slice(h_step3, 3, 1);

    auto rows = [&](ggml_tensor * dst, std::vector<int64_t> values) {
        ggml_backend_tensor_set(dst, values.data(), 0, values.size() * sizeof(int64_t));
    };
    rows(rows_full, { 0, 1, 2, 3 });
    rows(rows_seed, { 0, 1 });
    rows(rows_step2, { 2 });
    rows(rows_step3, { 3 });

    auto causal = [&](ggml_tensor * dst, int64_t n) {
        std::vector<float> values(static_cast<size_t>(n * n));
        for (int64_t q = 0; q < n; q++) {
            for (int64_t k = 0; k < n; k++) values[static_cast<size_t>(k + q * n)] = k <= q ? 0.0f : -INFINITY;
        }
        ggml_backend_tensor_set(dst, values.data(), 0, values.size() * sizeof(float));
    };
    causal(mask_full, S);
    causal(mask_seed, 2);

    auto run = [&](ggml_tensor * h, ggml_tensor * mask, ggml_tensor * kc, ggml_tensor * vc, ggml_tensor * r,
                   int64_t window) {
        constexpr size_t MAX_NODES = 128;
        ggml_init_params graph_params = {
            ggml_tensor_overhead() * MAX_NODES + ggml_graph_overhead_custom(MAX_NODES, false), nullptr, true
        };
        ggml_context * gctx = ggml_init(graph_params);
        ggml_cgraph *  gf   = ggml_new_graph_custom(gctx, MAX_NODES, false);
        ggml_tensor *  y    = mm3_depth_block(gctx, gf, config, layer, h, mask, kc, vc, r, window);
        ggml_build_forward_expand(gf, y);
        ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
        CHECK(ggml_gallocr_alloc_graph(allocator, gf));
        CHECK(ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS);
        std::vector<float> out(static_cast<size_t>(ggml_nelements(y)));
        ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
        ggml_gallocr_free(allocator);
        ggml_free(gctx);
        return out;
    };
    const std::vector<float> full  = run(h_full, mask_full, k_full, v_full, rows_full, S);
    const std::vector<float> seed  = run(h_seed, mask_seed, k_inc, v_inc, rows_seed, 2);
    const std::vector<float> step2 = run(h_step2, nullptr, k_inc, v_inc, rows_step2, 3);
    const std::vector<float> step3 = run(h_step3, nullptr, k_inc, v_inc, rows_step3, 4);

    auto same = [](float a, float b) { return std::fabs(a - b) <= 1e-5f * (1.0f + std::fabs(a)); };
    auto position = [&](const std::vector<float> & out, int64_t count, int64_t s, int64_t b, int64_t i) {
        return out[static_cast<size_t>((b * count + s) * H + i)];
    };
    for (int64_t b = 0; b < B; b++) {
        for (int64_t i = 0; i < H; i++) {
            CHECK(same(position(full, S, 0, b, i), position(seed, 2, 0, b, i)));
            CHECK(same(position(full, S, 1, b, i), position(seed, 2, 1, b, i)));
            CHECK(same(position(full, S, 2, b, i), position(step2, 1, 0, b, i)));
            CHECK(same(position(full, S, 3, b, i), position(step3, 1, 0, b, i)));
        }
    }

    auto dump = [&](ggml_tensor * t) {
        std::vector<float> values(static_cast<size_t>(ggml_nelements(t)));
        ggml_backend_tensor_get(t, values.data(), 0, ggml_nbytes(t));
        return values;
    };
    auto same_cache = [&](ggml_tensor * a, ggml_tensor * b) {
        const std::vector<float> lhs = dump(a);
        const std::vector<float> rhs = dump(b);
        if (lhs.size() != rhs.size()) return false;
        for (size_t i = 0; i < lhs.size(); i++) {
            if (!same(lhs[i], rhs[i])) return false;
        }
        return true;
    };
    CHECK(same_cache(k_full, k_inc));
    CHECK(same_cache(v_full, v_inc));

    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    ggml_backend_free(cpu);
}

constexpr size_t kTestGraphNodes = 256;

struct CpuGraph {
    ggml_context * ctx   = nullptr;
    ggml_cgraph *  graph = nullptr;
};

CpuGraph new_cpu_graph() {
    ggml_init_params params = {
        ggml_tensor_overhead() * kTestGraphNodes + ggml_graph_overhead_custom(kTestGraphNodes, false), nullptr, true
    };
    CpuGraph graph;
    graph.ctx   = ggml_init(params);
    graph.graph = ggml_new_graph_custom(graph.ctx, kTestGraphNodes, false);
    return graph;
}

std::vector<float> compute_cpu_graph(ggml_backend_t cpu, const CpuGraph & graph, ggml_tensor * output) {
    ggml_build_forward_expand(graph.graph, output);
    ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
    CHECK(ggml_gallocr_alloc_graph(allocator, graph.graph));
    CHECK(ggml_backend_graph_compute(cpu, graph.graph) == GGML_STATUS_SUCCESS);
    std::vector<float> values(static_cast<size_t>(ggml_nelements(output)));
    ggml_backend_tensor_get(output, values.data(), 0, ggml_nbytes(output));
    ggml_gallocr_free(allocator);
    ggml_free(graph.ctx);
    return values;
}

std::vector<float> fill_uniform(ggml_tensor * tensor, std::mt19937 & random, float base) {
    std::uniform_real_distribution<float> spread(-0.5f, 0.5f);
    std::vector<float> values(static_cast<size_t>(ggml_nelements(tensor)));
    for (float & value : values) {
        value = base + spread(random);
    }
    ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
    return values;
}

bool all_close(const std::vector<float> & left, const std::vector<float> & right, float tolerance) {
    if (left.size() != right.size()) {
        return false;
    }
    for (size_t i = 0; i < left.size(); i++) {
        if (std::fabs(left[i] - right[i]) > tolerance * (1.0f + std::fabs(left[i]))) {
            return false;
        }
    }
    return true;
}

std::vector<float> host_matvec(const std::vector<float> & weight, const std::vector<float> & x, int64_t rows,
                               int64_t inner, int64_t column) {
    std::vector<float> y(static_cast<size_t>(rows), 0.0f);
    for (int64_t r = 0; r < rows; r++) {
        for (int64_t k = 0; k < inner; k++) {
            y[static_cast<size_t>(r)] += weight[static_cast<size_t>(r * inner + k)] *
                                         x[static_cast<size_t>(column * inner + k)];
        }
    }
    return y;
}

// The CFG rows must reach ggml_mul_mat as columns of one 2-D operand, not as a
// broadcast batch: GPU matrix-vector kernels stream the weights once per batch
// entry, which is what doubled the AR stage's weight traffic.
void test_linear_folds_cfg_rows_into_columns() {
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    CHECK(cpu != nullptr);
    if (!cpu) return;

    constexpr int64_t K = 6;
    constexpr int64_t M = 3;
    constexpr int64_t B = 2;

    ggml_init_params weight_params = { ggml_tensor_overhead() * 4, nullptr, true };
    ggml_context *   wctx          = ggml_init(weight_params);
    ggml_tensor *    weight        = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, K, M);
    ggml_tensor *    x             = ggml_new_tensor_3d(wctx, GGML_TYPE_F32, K, 1, B);
    ggml_backend_buffer_t buffer   = ggml_backend_alloc_ctx_tensors(wctx, cpu);

    std::mt19937             random(7);
    const std::vector<float> weight_values = fill_uniform(weight, random, 0.0f);
    const std::vector<float> x_values      = fill_uniform(x, random, 0.0f);

    CpuGraph      graph   = new_cpu_graph();
    ggml_tensor * y       = mm3_linear(graph.ctx, weight, x);
    ggml_tensor * product = y->src[0];
    CHECK(y->ne[0] == M && y->ne[1] == 1 && y->ne[2] == B);
    CHECK(product->op == GGML_OP_MUL_MAT);
    CHECK(product->src[1]->ne[1] == B && product->src[1]->ne[2] == 1);

    const std::vector<float> values = compute_cpu_graph(cpu, graph, y);
    std::vector<float>       expected;
    for (int64_t column = 0; column < B; column++) {
        const std::vector<float> row = host_matvec(weight_values, x_values, M, K, column);
        expected.insert(expected.end(), row.begin(), row.end());
    }
    CHECK(all_close(values, expected, 1e-5f));

    ggml_backend_buffer_free(buffer);
    ggml_free(wctx);
    ggml_backend_free(cpu);
}

// The f32 request reaches ggml only for float weights: a quantized weight keeps
// the default precision and with it its integer kernels.
void test_linear_precision_follows_weight_type() {
    ggml_init_params params = { ggml_tensor_overhead() * kTestGraphNodes, nullptr, true };
    ggml_context *   ctx    = ggml_init(params);
    ggml_tensor *    x      = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 1, 2);
    for (ggml_type type : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16 }) {
        ggml_tensor * y = mm3_linear(ctx, ggml_new_tensor_2d(ctx, type, 64, 8), x, GGML_PREC_F32);
        CHECK(y->src[0]->op_params[0] == GGML_PREC_F32);
    }
    ggml_tensor * wide = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 1, 2);
    for (ggml_type type : { GGML_TYPE_Q8_0, GGML_TYPE_Q4_K }) {
        ggml_tensor * y = mm3_linear(ctx, ggml_new_tensor_2d(ctx, type, 256, 8), wide, GGML_PREC_F32);
        CHECK(y->src[0]->op_params[0] == GGML_PREC_DEFAULT);
    }
    ggml_tensor * y = mm3_linear(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 64, 8), x);
    CHECK(y->src[0]->op_params[0] == GGML_PREC_DEFAULT);
    ggml_free(ctx);
}

struct DitBlockFixture {
    MM3DitConfig   config;
    MM3DitBlock    block;
    ggml_context * ctx       = nullptr;
    ggml_tensor *  pair      = nullptr;
    ggml_tensor *  first     = nullptr;
    ggml_tensor *  second    = nullptr;
    ggml_tensor *  positions = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
};

constexpr int64_t kDitTestEmbedding = 8;
constexpr int64_t kDitTestHeads     = 2;
constexpr int64_t kDitTestFfInner   = 6;
constexpr int64_t kDitTestTokens    = 5;

void define_dit_block_weights(DitBlockFixture & f) {
    const int64_t E  = kDitTestEmbedding;
    const int64_t FI = kDitTestFfInner;
    f.block.attn_norm_w = ggml_new_tensor_1d(f.ctx, GGML_TYPE_F32, E);
    f.block.attn_norm_b = ggml_new_tensor_1d(f.ctx, GGML_TYPE_F32, E);
    f.block.attn_qkv    = ggml_new_tensor_2d(f.ctx, GGML_TYPE_F32, E, 3 * E);
    f.block.attn_output = ggml_new_tensor_2d(f.ctx, GGML_TYPE_F32, E, E);
    f.block.ffn_norm_w  = ggml_new_tensor_1d(f.ctx, GGML_TYPE_F32, E);
    f.block.ffn_norm_b  = ggml_new_tensor_1d(f.ctx, GGML_TYPE_F32, E);
    f.block.ffn_in_w    = ggml_new_tensor_2d(f.ctx, GGML_TYPE_F32, E, 2 * FI);
    f.block.ffn_in_b    = ggml_new_tensor_1d(f.ctx, GGML_TYPE_F32, 2 * FI);
    f.block.ffn_out_w   = ggml_new_tensor_2d(f.ctx, GGML_TYPE_F32, FI, E);
    f.block.ffn_out_b   = ggml_new_tensor_1d(f.ctx, GGML_TYPE_F32, E);
}

void fill_dit_block_weights(DitBlockFixture & f, std::mt19937 & random) {
    for (ggml_tensor * t : { f.block.attn_norm_b, f.block.attn_qkv, f.block.attn_output, f.block.ffn_norm_b,
                             f.block.ffn_in_w, f.block.ffn_in_b, f.block.ffn_out_w, f.block.ffn_out_b }) {
        fill_uniform(t, random, 0.0f);
    }
    fill_uniform(f.block.attn_norm_w, random, 1.0f);
    fill_uniform(f.block.ffn_norm_w, random, 1.0f);
}

void fill_dit_block_inputs(DitBlockFixture & f, std::mt19937 & random) {
    const std::vector<float> pair       = fill_uniform(f.pair, random, 0.0f);
    const size_t             per_branch = pair.size() / 2;
    ggml_backend_tensor_set(f.first, pair.data(), 0, per_branch * sizeof(float));
    ggml_backend_tensor_set(f.second, pair.data() + per_branch, 0, per_branch * sizeof(float));
    std::vector<int32_t> positions(static_cast<size_t>(kDitTestTokens));
    for (size_t i = 0; i < positions.size(); i++) {
        positions[i] = static_cast<int32_t>(i);
    }
    ggml_backend_tensor_set(f.positions, positions.data(), 0, positions.size() * sizeof(int32_t));
}

DitBlockFixture make_dit_block_fixture(ggml_backend_t cpu) {
    DitBlockFixture f;
    f.config.embedding_length = static_cast<uint32_t>(kDitTestEmbedding);
    f.config.head_count       = static_cast<uint32_t>(kDitTestHeads);
    f.config.head_dim         = static_cast<uint32_t>(kDitTestEmbedding / kDitTestHeads);
    f.config.ff_inner         = static_cast<uint32_t>(kDitTestFfInner);
    f.config.rope_dim         = f.config.head_dim;
    f.config.rope_theta       = 10000.0f;
    f.config.layer_norm_eps   = 1e-5f;

    ggml_init_params params = { ggml_tensor_overhead() * 16, nullptr, true };
    f.ctx                   = ggml_init(params);
    define_dit_block_weights(f);
    f.pair      = ggml_new_tensor_3d(f.ctx, GGML_TYPE_F32, kDitTestEmbedding, kDitTestTokens, 2);
    f.first     = ggml_new_tensor_3d(f.ctx, GGML_TYPE_F32, kDitTestEmbedding, kDitTestTokens, 1);
    f.second    = ggml_new_tensor_3d(f.ctx, GGML_TYPE_F32, kDitTestEmbedding, kDitTestTokens, 1);
    f.positions = ggml_new_tensor_1d(f.ctx, GGML_TYPE_I32, kDitTestTokens);
    f.buffer    = ggml_backend_alloc_ctx_tensors(f.ctx, cpu);

    std::mt19937 random(2026);
    fill_dit_block_weights(f, random);
    fill_dit_block_inputs(f, random);
    return f;
}

std::vector<float> run_dit_block(ggml_backend_t cpu, const DitBlockFixture & f, const MM3DitGraph & dit,
                                 ggml_tensor * h) {
    CpuGraph graph = new_cpu_graph();
    return compute_cpu_graph(cpu, graph, mm3_dit_block(graph.ctx, dit, f.config, f.block, h, f.positions));
}

ggml_tensor * split_dit_rows(ggml_context * ctx, ggml_tensor * x, int64_t rows, int64_t offset) {
    return ggml_cont(ctx, ggml_view_2d(ctx, x, rows, x->ne[1], x->nb[1], (size_t) offset * x->nb[0]));
}

// One branch of the DiT block written the explicit way: every projection and
// GLU half copied out before use. mm3_dit_block reads them through views and a
// fused SwiGLU instead, and must compute the same thing.
ggml_tensor * explicit_dit_block(ggml_context * ctx, const MM3DitGraph & dit, const DitBlockFixture & f,
                                 ggml_tensor * h) {
    const MM3DitConfig & c  = f.config;
    const MM3DitBlock &  w  = f.block;
    const int64_t        E  = (int64_t) c.embedding_length;
    const int64_t        D  = (int64_t) c.head_dim;
    const int64_t        Nh = (int64_t) c.head_count;
    const int64_t        FI = (int64_t) c.ff_inner;
    const int64_t        S  = h->ne[1];

    ggml_tensor * x   = ggml_reshape_2d(ctx, h, E, S);
    ggml_tensor * n   = mm3_dit_ln(ctx, x, w.attn_norm_w, w.attn_norm_b, c.layer_norm_eps);
    ggml_tensor * qkv = ggml_mul_mat(ctx, w.attn_qkv, n);
    ggml_tensor * q   = ggml_reshape_3d(ctx, split_dit_rows(ctx, qkv, E, 0), D, Nh, S);
    ggml_tensor * k   = ggml_reshape_3d(ctx, split_dit_rows(ctx, qkv, E, E), D, Nh, S);
    ggml_tensor * v   = ggml_reshape_3d(ctx, split_dit_rows(ctx, qkv, E, 2 * E), D, Nh, S);
    q = ggml_permute(ctx, mm3_dit_rope(ctx, c, q, f.positions), 0, 2, 1, 3);
    k = ggml_permute(ctx, mm3_dit_rope(ctx, c, k, f.positions), 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    const float   scale = 1.0f / sqrtf((float) D);
    ggml_tensor * attn  = nullptr;
    if (dit.use_flash_attn) {
        attn = ggml_flash_attn_ext(ctx, q, ggml_cast(ctx, k, GGML_TYPE_F16), ggml_cast(ctx, v, GGML_TYPE_F16), NULL,
                                   scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    } else {
        attn = mm3_dit_attn_f32(ctx, q, k, v, scale);
    }
    x = ggml_add(ctx, x, ggml_mul_mat(ctx, w.attn_output, ggml_reshape_2d(ctx, attn, Nh * D, S)));

    ggml_tensor * n2 = mm3_dit_ln(ctx, x, w.ffn_norm_w, w.ffn_norm_b, c.layer_norm_eps);
    ggml_tensor * ff = ggml_add(ctx, ggml_mul_mat(ctx, w.ffn_in_w, n2), w.ffn_in_b);
    ggml_tensor * y  = ggml_mul(ctx, split_dit_rows(ctx, ff, FI, 0), ggml_silu(ctx, split_dit_rows(ctx, ff, FI, FI)));
    return ggml_add(ctx, x, ggml_add(ctx, ggml_mul_mat(ctx, w.ffn_out_w, y), w.ffn_out_b));
}

std::vector<float> run_explicit_dit_block(ggml_backend_t cpu, const DitBlockFixture & f, const MM3DitGraph & dit,
                                          ggml_tensor * h) {
    CpuGraph graph = new_cpu_graph();
    return compute_cpu_graph(cpu, graph, explicit_dit_block(graph.ctx, dit, f, h));
}

// Both CFG branches now share one DiT forward; a branch of the batched block
// must equal the same branch run on its own and the explicit single-branch
// formulation, with and without flash attention.
void test_dit_block_batches_cfg_branches() {
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    CHECK(cpu != nullptr);
    if (!cpu) return;

    DitBlockFixture fixture = make_dit_block_fixture(cpu);
    for (bool flash : { false, true }) {
        MM3DitGraph dit;
        dit.use_flash_attn = flash;
        const std::vector<float> pair   = run_dit_block(cpu, fixture, dit, fixture.pair);
        std::vector<float>       joined = run_dit_block(cpu, fixture, dit, fixture.first);
        const std::vector<float> second = run_dit_block(cpu, fixture, dit, fixture.second);
        CHECK(all_close(joined, run_explicit_dit_block(cpu, fixture, dit, fixture.first), 1e-5f));
        CHECK(all_close(second, run_explicit_dit_block(cpu, fixture, dit, fixture.second), 1e-5f));
        joined.insert(joined.end(), second.begin(), second.end());
        CHECK(all_close(pair, joined, 1e-5f));
    }

    ggml_backend_buffer_free(fixture.buffer);
    ggml_free(fixture.ctx);
    ggml_backend_free(cpu);
}

// Uploads a stacked copy of the parts, rows in the given order.
void stack_rows(ggml_tensor * stacked, const std::vector<ggml_tensor *> & parts) {
    size_t offset = 0;
    for (ggml_tensor * part : parts) {
        std::vector<uint8_t> bytes(ggml_nbytes(part));
        ggml_backend_tensor_get(part, bytes.data(), 0, bytes.size());
        ggml_backend_tensor_set(stacked, bytes.data(), offset, bytes.size());
        offset += bytes.size();
    }
}

void test_lm_positions_tile_cfg_rows() {
    MM3LmGraph graph;
    graph.kv_pos = 3;
    graph.n_ctx  = 8;
    mm3_lm_fill_positions(&graph, 2);
    CHECK(graph.pos_host == std::vector<int32_t>({ 3, 4, 3, 4 }));
    CHECK(graph.rows_host == std::vector<int64_t>({ 3, 4, 11, 12 }));
}

constexpr int64_t kLmTestHidden   = 8;
constexpr int64_t kLmTestHeadDim  = 4;
constexpr int64_t kLmTestHeads    = 2;
constexpr int64_t kLmTestKvHeads  = 1;
constexpr int64_t kLmTestFfn      = 12;
constexpr int64_t kLmTestContext  = 8;
constexpr int64_t kLmTestTokens   = 3;
constexpr int64_t kLmTestRows     = MM3_LM_CFG_ROWS;
constexpr int     kLmTestCaches   = 4;

struct LmBlockFixture {
    MM3LmConfig    config;
    MM3LmLayer     separate;
    MM3LmLayer     stacked;
    ggml_context * ctx = nullptr;
    ggml_tensor *  full   = nullptr;
    ggml_tensor *  prefix = nullptr;
    ggml_tensor *  last   = nullptr;
    ggml_tensor *  swapped = nullptr;
    ggml_tensor *  k_cache[kLmTestCaches] = {};
    ggml_tensor *  v_cache[kLmTestCaches] = {};
    std::vector<float> input;
    ggml_backend_buffer_t buffer = nullptr;
};

void define_lm_block_weights(LmBlockFixture & f) {
    const int64_t H = kLmTestHidden;
    const int64_t Q = kLmTestHeadDim * kLmTestHeads;
    const int64_t K = kLmTestHeadDim * kLmTestKvHeads;
    const int64_t F = kLmTestFfn;
    auto matrix = [&](int64_t in, int64_t out) { return ggml_new_tensor_2d(f.ctx, GGML_TYPE_F32, in, out); };
    auto vec    = [&](int64_t n) { return ggml_new_tensor_1d(f.ctx, GGML_TYPE_F32, n); };
    f.separate.attn_norm   = vec(H);
    f.separate.attn_q      = matrix(H, Q);
    f.separate.attn_k      = matrix(H, K);
    f.separate.attn_v      = matrix(H, K);
    f.separate.attn_output = matrix(Q, H);
    f.separate.attn_q_norm = vec(kLmTestHeadDim);
    f.separate.attn_k_norm = vec(kLmTestHeadDim);
    f.separate.ffn_norm    = vec(H);
    f.separate.ffn_gate    = matrix(H, F);
    f.separate.ffn_up      = matrix(H, F);
    f.separate.ffn_down    = matrix(F, H);
    f.stacked              = f.separate;
    f.stacked.attn_q = f.stacked.attn_k = f.stacked.attn_v = nullptr;
    f.stacked.ffn_gate = f.stacked.ffn_up = nullptr;
    f.stacked.attn_qkv     = matrix(H, Q + 2 * K);
    f.stacked.ffn_gate_up  = matrix(H, 2 * F);
}

void fill_lm_block_weights(LmBlockFixture & f, std::mt19937 & random) {
    const MM3LmLayer & w = f.separate;
    for (ggml_tensor * t : { w.attn_q, w.attn_k, w.attn_v, w.attn_output, w.ffn_gate, w.ffn_up, w.ffn_down }) {
        fill_uniform(t, random, 0.0f);
    }
    for (ggml_tensor * t : { w.attn_norm, w.attn_q_norm, w.attn_k_norm, w.ffn_norm }) {
        fill_uniform(t, random, 1.0f);
    }
    stack_rows(f.stacked.attn_qkv, { w.attn_q, w.attn_k, w.attn_v });
    stack_rows(f.stacked.ffn_gate_up, { w.ffn_gate, w.ffn_up });
}

// Copies tokens [first, first + count) of both CFG rows of the full input.
void slice_lm_tokens(const LmBlockFixture & f, ggml_tensor * dst, int64_t first, int64_t count) {
    std::vector<float> values;
    for (int64_t b = 0; b < kLmTestRows; b++) {
        const auto begin = f.input.begin() + (b * kLmTestTokens + first) * kLmTestHidden;
        values.insert(values.end(), begin, begin + count * kLmTestHidden);
    }
    ggml_backend_tensor_set(dst, values.data(), 0, values.size() * sizeof(float));
}

LmBlockFixture make_lm_block_fixture(ggml_backend_t cpu) {
    LmBlockFixture f;
    f.config.embedding_length = (uint32_t) kLmTestHidden;
    f.config.key_length       = (uint32_t) kLmTestHeadDim;
    f.config.head_count       = (uint32_t) kLmTestHeads;
    f.config.head_count_kv    = (uint32_t) kLmTestKvHeads;
    f.config.rms_eps          = 1e-6f;
    f.config.rope_freq_base   = 10000.0f;

    ggml_init_params params = { ggml_tensor_overhead() * 40, nullptr, true };
    f.ctx                   = ggml_init(params);
    define_lm_block_weights(f);
    f.full   = ggml_new_tensor_3d(f.ctx, GGML_TYPE_F32, kLmTestHidden, kLmTestTokens, kLmTestRows);
    f.prefix = ggml_new_tensor_3d(f.ctx, GGML_TYPE_F32, kLmTestHidden, kLmTestTokens - 1, kLmTestRows);
    f.last   = ggml_new_tensor_3d(f.ctx, GGML_TYPE_F32, kLmTestHidden, 1, kLmTestRows);
    f.swapped = ggml_new_tensor_3d(f.ctx, GGML_TYPE_F32, kLmTestHidden, kLmTestTokens, kLmTestRows);
    for (int i = 0; i < kLmTestCaches; i++) {
        const int64_t width = kLmTestHeadDim * kLmTestKvHeads;
        f.k_cache[i]        = ggml_new_tensor_2d(f.ctx, GGML_TYPE_F16, width, kLmTestContext * kLmTestRows);
        f.v_cache[i]        = ggml_new_tensor_2d(f.ctx, GGML_TYPE_F16, width, kLmTestContext * kLmTestRows);
    }
    f.buffer = ggml_backend_alloc_ctx_tensors(f.ctx, cpu);
    ggml_backend_buffer_clear(f.buffer, 0);

    std::mt19937 random(2027);
    fill_lm_block_weights(f, random);
    f.input = fill_uniform(f.full, random, 0.0f);
    slice_lm_tokens(f, f.prefix, 0, kLmTestTokens - 1);
    slice_lm_tokens(f, f.last, kLmTestTokens - 1, 1);
    const size_t row_bytes = (size_t) (kLmTestHidden * kLmTestTokens) * sizeof(float);
    ggml_backend_tensor_set(f.swapped, f.input.data() + row_bytes / sizeof(float), 0, row_bytes);
    ggml_backend_tensor_set(f.swapped, f.input.data(), row_bytes, row_bytes);
    return f;
}

struct LmStep {
    ggml_tensor * h      = nullptr;
    int64_t       kv_pos = 0;
    int           cache  = 0;
};

// Runs one block over h (T tokens starting at kv_pos) through the production
// position, row and mask helpers, writing the step into cache `cache`.
std::vector<float> run_lm_block(ggml_backend_t cpu, const LmBlockFixture & f, const MM3LmLayer & w, LmStep step,
                                bool flash) {
    const int64_t T    = step.h->ne[1];
    const int64_t n_kv = step.kv_pos + T;
    MM3LmGraph    host;
    host.kv_pos = step.kv_pos;
    host.n_ctx  = kLmTestContext;
    mm3_lm_fill_positions(&host, T);
    mm3_lm_fill_mask(&host, T, n_kv);

    CpuGraph      graph     = new_cpu_graph();
    ggml_tensor * positions = ggml_new_tensor_1d(graph.ctx, GGML_TYPE_I32, T * kLmTestRows);
    ggml_tensor * rows      = ggml_new_tensor_1d(graph.ctx, GGML_TYPE_I64, T * kLmTestRows);
    ggml_tensor * mask      = ggml_new_tensor_2d(graph.ctx, GGML_TYPE_F16, n_kv, T);
    for (ggml_tensor * t : { positions, rows, mask }) {
        ggml_set_input(t);
    }
    ggml_tensor * y = mm3_lm_block(graph.ctx, graph.graph, f.config, w, step.h, positions, mask, rows,
                                   f.k_cache[step.cache], f.v_cache[step.cache], n_kv, flash, GGML_PREC_F32);
    ggml_build_forward_expand(graph.graph, y);
    ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
    CHECK(ggml_gallocr_alloc_graph(allocator, graph.graph));
    ggml_backend_tensor_set(positions, host.pos_host.data(), 0, ggml_nbytes(positions));
    ggml_backend_tensor_set(rows, host.rows_host.data(), 0, ggml_nbytes(rows));
    ggml_backend_tensor_set(mask, host.mask_host.data(), 0, ggml_nbytes(mask));
    CHECK(ggml_backend_graph_compute(cpu, graph.graph) == GGML_STATUS_SUCCESS);
    std::vector<float> values((size_t) ggml_nelements(y));
    ggml_backend_tensor_get(y, values.data(), 0, ggml_nbytes(y));
    ggml_gallocr_free(allocator);
    ggml_free(graph.ctx);
    return values;
}

std::vector<float> lm_token(const std::vector<float> & out, int64_t tokens, int64_t t, int64_t b) {
    const auto begin = out.begin() + (b * tokens + t) * kLmTestHidden;
    return std::vector<float>(begin, begin + kLmTestHidden);
}

// The LM block must give the same result with stacked and separate q/k/v and
// gate/up weights, a cache filled incrementally (prefix, then one token) must
// reproduce a single pass over the whole sequence, and each CFG row must only
// see its own history: swapping the rows' inputs swaps their outputs. The
// cache holds one [D * heads] row per position with CFG row b at b * n_ctx.
void test_lm_block_stacking_and_cache() {
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    CHECK(cpu != nullptr);
    if (!cpu) return;

    for (bool flash : { false, true }) {
        LmBlockFixture f = make_lm_block_fixture(cpu);
        const std::vector<float> full    = run_lm_block(cpu, f, f.separate, { f.full, 0, 0 }, flash);
        const std::vector<float> stacked = run_lm_block(cpu, f, f.stacked, { f.full, 0, 1 }, flash);
        const std::vector<float> prefix  = run_lm_block(cpu, f, f.stacked, { f.prefix, 0, 2 }, flash);
        const std::vector<float> last    = run_lm_block(cpu, f, f.stacked, { f.last, kLmTestTokens - 1, 2 }, flash);
        const std::vector<float> swapped = run_lm_block(cpu, f, f.stacked, { f.swapped, 0, 3 }, flash);
        CHECK(all_close(full, stacked, 1e-5f));
        for (int64_t t = 0; t < kLmTestTokens; t++) {
            CHECK(all_close(lm_token(full, kLmTestTokens, t, 0), lm_token(swapped, kLmTestTokens, t, 1), 1e-5f));
            CHECK(all_close(lm_token(full, kLmTestTokens, t, 1), lm_token(swapped, kLmTestTokens, t, 0), 1e-5f));
        }
        for (int64_t b = 0; b < kLmTestRows; b++) {
            for (int64_t t = 0; t + 1 < kLmTestTokens; t++) {
                CHECK(all_close(lm_token(full, kLmTestTokens, t, b), lm_token(prefix, kLmTestTokens - 1, t, b), 1e-4f));
            }
            CHECK(all_close(lm_token(full, kLmTestTokens, kLmTestTokens - 1, b), lm_token(last, 1, 0, b), 1e-4f));
        }
        ggml_backend_buffer_free(f.buffer);
        ggml_free(f.ctx);
    }
    ggml_backend_free(cpu);
}

struct DepthStackFixture {
    MM3DepthConfig config;
    MM3DepthLayer  separate;
    MM3DepthLayer  stacked;
    ggml_context * ctx   = nullptr;
    ggml_tensor *  h     = nullptr;
    ggml_tensor *  rows  = nullptr;
    ggml_tensor *  mask  = nullptr;
    ggml_tensor *  k_cache[2] = {};
    ggml_tensor *  v_cache[2] = {};
    ggml_backend_buffer_t buffer = nullptr;
};

constexpr int64_t kDepthTestHidden = 8;
constexpr int64_t kDepthTestHeads  = 2;
constexpr int64_t kDepthTestFfn    = 12;
constexpr int64_t kDepthTestSeed   = 2;

void define_depth_stack_weights(DepthStackFixture & f) {
    const int64_t H = kDepthTestHidden;
    const int64_t F = kDepthTestFfn;
    auto matrix = [&](int64_t in, int64_t out) { return ggml_new_tensor_2d(f.ctx, GGML_TYPE_F32, in, out); };
    f.separate.attn_norm   = ggml_new_tensor_1d(f.ctx, GGML_TYPE_F32, H);
    f.separate.ffn_norm    = ggml_new_tensor_1d(f.ctx, GGML_TYPE_F32, H);
    f.separate.attn_q      = matrix(H, H);
    f.separate.attn_k      = matrix(H, H);
    f.separate.attn_v      = matrix(H, H);
    f.separate.attn_output = matrix(H, H);
    f.separate.ffn_gate    = matrix(H, F);
    f.separate.ffn_up      = matrix(H, F);
    f.separate.ffn_down    = matrix(F, H);
    f.stacked              = f.separate;
    f.stacked.attn_q = f.stacked.attn_k = f.stacked.attn_v = nullptr;
    f.stacked.ffn_gate = f.stacked.ffn_up = nullptr;
    f.stacked.attn_qkv     = matrix(H, 3 * H);
    f.stacked.ffn_gate_up  = matrix(H, 2 * F);
}

void fill_depth_stack_inputs(DepthStackFixture & f, std::mt19937 & random) {
    const MM3DepthLayer & w = f.separate;
    for (ggml_tensor * t : { w.attn_q, w.attn_k, w.attn_v, w.attn_output, w.ffn_gate, w.ffn_up, w.ffn_down }) {
        fill_uniform(t, random, 0.0f);
    }
    fill_uniform(w.attn_norm, random, 1.0f);
    fill_uniform(w.ffn_norm, random, 1.0f);
    stack_rows(f.stacked.attn_qkv, { w.attn_q, w.attn_k, w.attn_v });
    stack_rows(f.stacked.ffn_gate_up, { w.ffn_gate, w.ffn_up });
    fill_uniform(f.h, random, 0.0f);
    const int64_t rows[kDepthTestSeed] = { 0, 1 };
    ggml_backend_tensor_set(f.rows, rows, 0, sizeof(rows));
    const float mask[kDepthTestSeed * kDepthTestSeed] = { 0.0f, -INFINITY, 0.0f, 0.0f };
    ggml_backend_tensor_set(f.mask, mask, 0, sizeof(mask));
}

DepthStackFixture make_depth_stack_fixture(ggml_backend_t cpu) {
    DepthStackFixture f;
    f.config.embedding_length    = (uint32_t) kDepthTestHidden;
    f.config.feed_forward_length = (uint32_t) kDepthTestFfn;
    f.config.head_count          = (uint32_t) kDepthTestHeads;
    f.config.head_dim            = (uint32_t) (kDepthTestHidden / kDepthTestHeads);
    f.config.rms_eps             = 1e-6f;

    ggml_init_params params = { ggml_tensor_overhead() * 32, nullptr, true };
    f.ctx                   = ggml_init(params);
    define_depth_stack_weights(f);
    f.h    = ggml_new_tensor_3d(f.ctx, GGML_TYPE_F32, kDepthTestHidden, kDepthTestSeed, 2);
    f.rows = ggml_new_tensor_1d(f.ctx, GGML_TYPE_I64, kDepthTestSeed);
    f.mask = ggml_new_tensor_2d(f.ctx, GGML_TYPE_F32, kDepthTestSeed, kDepthTestSeed);
    for (int i = 0; i < 2; i++) {
        const int64_t D = kDepthTestHidden / kDepthTestHeads;
        f.k_cache[i]    = ggml_new_tensor_4d(f.ctx, GGML_TYPE_F32, D, kDepthTestSeed, kDepthTestHeads, 2);
        f.v_cache[i]    = ggml_new_tensor_4d(f.ctx, GGML_TYPE_F32, D, kDepthTestSeed, kDepthTestHeads, 2);
    }
    f.buffer = ggml_backend_alloc_ctx_tensors(f.ctx, cpu);
    ggml_backend_buffer_clear(f.buffer, 0);
    std::mt19937 random(2028);
    fill_depth_stack_inputs(f, random);
    return f;
}

std::vector<float> run_depth_stack_block(ggml_backend_t cpu, const DepthStackFixture & f, const MM3DepthLayer & w,
                                         int cache) {
    CpuGraph      graph = new_cpu_graph();
    ggml_tensor * y     = mm3_depth_block(graph.ctx, graph.graph, f.config, w, f.h, f.mask, f.k_cache[cache],
                                          f.v_cache[cache], f.rows, kDepthTestSeed);
    return compute_cpu_graph(cpu, graph, y);
}

// The depth block must give the same result with stacked and separate q/k/v
// and gate/up weights.
void test_depth_block_stacked_matches_separate() {
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    CHECK(cpu != nullptr);
    if (!cpu) return;
    DepthStackFixture f = make_depth_stack_fixture(cpu);
    CHECK(all_close(run_depth_stack_block(cpu, f, f.separate, 0), run_depth_stack_block(cpu, f, f.stacked, 1), 1e-5f));
    ggml_backend_buffer_free(f.buffer);
    ggml_free(f.ctx);
    ggml_backend_free(cpu);
}

void test_flow_schedule() {
    using tts_cpp::minimax::detail::flow_schedule;
    using tts_cpp::minimax::detail::resolve_flow_steps;
    CHECK(tts_cpp::minimax::detail::kDefaultFlowSteps == 20);
    // An explicit request always wins; otherwise the engine's own recommendation.
    CHECK(resolve_flow_steps(8) == 8);
    CHECK(resolve_flow_steps(0) == 20);
    CHECK(resolve_flow_steps(-3) == 20);
    CHECK(resolve_flow_steps(50) == 50);
    CHECK(close(tts_cpp::minimax::detail::kDefaultCfgScale, 1.7f));
    std::vector<float> sigmas;
    std::vector<float> timesteps;
    flow_schedule(4, sigmas, timesteps);
    CHECK(sigmas.size() == 5);
    CHECK(timesteps.size() == 4);
    CHECK(close(timesteps[0], 0.0f));
    CHECK(close(timesteps[1], 0.25f));
    CHECK(close(timesteps[2], 0.5f));
    CHECK(close(timesteps[3], 0.75f));
    CHECK(close(sigmas[4], 1.0f));
    CHECK(throws_invalid_argument([&] { flow_schedule(0, sigmas, timesteps); }));
}

void test_production_dit_readback_preserves_velocity() {
    const std::vector<float> raw = {1.0f, -2.0f, 0.5f};
    size_t requested_bytes = 0;
    const MM3DitReadback readback =
        [&raw, &requested_bytes](float * output, size_t bytes, std::string *) {
            requested_bytes = bytes;
            std::copy(raw.begin(), raw.end(), output);
            return true;
        };
    std::vector<float> output(raw.size());
    std::string error;
    CHECK(mm3_read_dit_output(output.data(), output.size(), readback, &error));
    CHECK(error.empty());
    CHECK(requested_bytes == raw.size() * sizeof(float));
    CHECK(output == raw);
}

void test_depth_step_fused_readback_matches_source_tensors() {
    // Mirrors mm3-depth-graph.h's per-step output fusion: reshape the hidden
    // [H,1,2] and logits [V,1,2] CFG outputs to 1-D and concat along dim 0,
    // then confirm one download splits back into the exact values each
    // tensor was set to. H != V so a size mixup would size-mismatch, and
    // distinct value ranges so a block swap or misaligned offset would show.
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    CHECK(cpu != nullptr);
    if (!cpu) return;

    const int64_t H = 5;
    const int64_t V = 7;
    constexpr size_t MAX_NODES = 8;

    const size_t context_size =
        ggml_tensor_overhead() * MAX_NODES + ggml_graph_overhead_custom(MAX_NODES, false);
    ggml_init_params params = { context_size, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    CHECK(ctx != nullptr);

    ggml_tensor * hidden = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H, 1, 2);
    ggml_tensor * logits = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, V, 1, 2);
    ggml_set_input(hidden);
    ggml_set_input(logits);

    ggml_tensor * hidden_flat = ggml_reshape_1d(ctx, hidden, ggml_nelements(hidden));
    ggml_tensor * logits_flat = ggml_reshape_1d(ctx, logits, ggml_nelements(logits));
    ggml_tensor * fused       = ggml_concat(ctx, hidden_flat, logits_flat, 0);
    ggml_set_output(fused);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, MAX_NODES, false);
    ggml_build_forward_expand(graph, fused);

    ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
    CHECK(ggml_gallocr_alloc_graph(allocator, graph));

    std::vector<float> hidden_values(static_cast<size_t>(H * 2));
    std::vector<float> logits_values(static_cast<size_t>(V * 2));
    for (size_t i = 0; i < hidden_values.size(); i++) hidden_values[i] = static_cast<float>(i + 1);
    for (size_t i = 0; i < logits_values.size(); i++) logits_values[i] = static_cast<float>(100 + i);
    ggml_backend_tensor_set(hidden, hidden_values.data(), 0, hidden_values.size() * sizeof(float));
    ggml_backend_tensor_set(logits, logits_values.data(), 0, logits_values.size() * sizeof(float));

    CHECK(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> fused_buf(hidden_values.size() + logits_values.size());
    ggml_backend_tensor_get(fused, fused_buf.data(), 0, fused_buf.size() * sizeof(float));

    const auto split = fused_buf.begin() + static_cast<std::ptrdiff_t>(hidden_values.size());
    const std::vector<float> hidden_part(fused_buf.begin(), split);
    const std::vector<float> logits_part(split, fused_buf.end());
    CHECK(hidden_part == hidden_values);
    CHECK(logits_part == logits_values);

    ggml_gallocr_free(allocator);
    ggml_free(ctx);
    ggml_backend_free(cpu);
}

void test_production_cfg_euler_step() {
    std::vector<float> latents = {1.0f, -1.0f};
    const std::vector<float> conditional = {2.0f, 4.0f};
    const std::vector<float> unconditional = {0.5f, -2.0f};
    const std::vector<float> sigmas = {0.0f, 0.25f};
    std::string error;
    CHECK(mm3_integrate_flow_step(latents, conditional, unconditional, sigmas, 0,
                                  1.5f, &error));
    CHECK(error.empty());
    CHECK(latents == std::vector<float>({1.6875f, 0.75f}));
}

tts_cpp::minimax::detail::SynthesisContract valid_synthesis_contract() {
    tts_cpp::minimax::detail::SynthesisContract contract;
    contract.scheduler = "FlowMatchEulerDiscrete";
    contract.invert_sigmas = true;
    contract.shift = 1.0f;
    contract.train_timesteps = 1;
    contract.rope_type = "neox";
    contract.glu_order = "value_gate";
    contract.timestep_token_prepended = true;
    contract.pre_post_conv_residual = true;
    contract.attn_bias = false;
    return contract;
}

void test_malformed_synthesis_metadata() {
    using namespace tts_cpp::minimax::detail;
    CHECK(vocoder_upsample_error({8, 8, 2, 4}, 512).empty());
    CHECK(!vocoder_upsample_error({}, 512).empty());
    CHECK(!vocoder_upsample_error({8, 0, 2, 4}, 512).empty());
    CHECK(!vocoder_upsample_error({8, -1, 2, 4}, 512).empty());
    CHECK(!vocoder_upsample_error({8, 8, 2, 4}, 256).empty());
    CHECK(!vocoder_upsample_error({std::numeric_limits<int32_t>::max(), 3}, 1).empty());

    SynthesisContract contract = valid_synthesis_contract();
    CHECK(validate_synthesis_contract(contract).empty());
    contract.scheduler = "Other";
    CHECK(!validate_synthesis_contract(contract).empty());
    contract = valid_synthesis_contract();
    contract.invert_sigmas = false;
    CHECK(!validate_synthesis_contract(contract).empty());
    contract = valid_synthesis_contract();
    contract.shift = std::numeric_limits<float>::quiet_NaN();
    CHECK(!validate_synthesis_contract(contract).empty());
    contract = valid_synthesis_contract();
    contract.shift = 1.0001f;
    CHECK(!validate_synthesis_contract(contract).empty());
    contract = valid_synthesis_contract();
    contract.train_timesteps = 1000;
    CHECK(!validate_synthesis_contract(contract).empty());
    contract = valid_synthesis_contract();
    contract.rope_type = "interleaved";
    CHECK(!validate_synthesis_contract(contract).empty());
    contract = valid_synthesis_contract();
    contract.glu_order = "gate_value";
    CHECK(!validate_synthesis_contract(contract).empty());
    contract = valid_synthesis_contract();
    contract.timestep_token_prepended = false;
    CHECK(!validate_synthesis_contract(contract).empty());
    contract = valid_synthesis_contract();
    contract.pre_post_conv_residual = false;
    CHECK(!validate_synthesis_contract(contract).empty());
    contract = valid_synthesis_contract();
    contract.attn_bias = true;
    CHECK(!validate_synthesis_contract(contract).empty());
}

void test_vocoder_output_shape() {
    using tts_cpp::minimax::detail::vocoder_output_shape_error;
    CHECK(vocoder_output_shape_error(8, 1, 1, 1, 8, 8 * sizeof(float), 8).empty());
    CHECK(!vocoder_output_shape_error(7, 1, 1, 1, 8, 8 * sizeof(float), 8).empty());
    CHECK(!vocoder_output_shape_error(8, 2, 1, 1, 8, 8 * sizeof(float), 8).empty());
    CHECK(!vocoder_output_shape_error(8, 1, 1, 1, 7, 8 * sizeof(float), 8).empty());
    CHECK(!vocoder_output_shape_error(8, 1, 1, 1, 8, 7 * sizeof(float), 8).empty());
    CHECK(!vocoder_output_shape_error(0, 1, 1, 1, 0, 0, 0).empty());
}

void test_copy_ranges() {
    using namespace tts_cpp::minimax::detail;
    CHECK(copy_range_fits(10, 8, 2, 1, 4));
    CHECK(!copy_range_fits(10, 8, -1, 1, 4));
    CHECK(!copy_range_fits(10, 8, 8, 1, 3));
    CHECK(!copy_range_fits(10, 8, 2, 7, 2));
    CHECK(stereo_copy_ranges_fit(20, 16, 10, 2, 8, 1, 4));
    CHECK(!stereo_copy_ranges_fit(19, 16, 10, 2, 8, 1, 4));
    CHECK(!stereo_copy_ranges_fit(20, 16, 10, 7, 8, 1, 4));
    CHECK(!stereo_copy_ranges_fit(20, 16, 10, 2, 8, 6, 4));
}

void test_condition_length() {
    using namespace tts_cpp::minimax::detail;
    ConditionRate rate;
    CHECK(condition_latent_length(rate, 200) == 689);
    CHECK(condition_latent_length(rate, 100) == 344);
    CHECK(condition_latent_length(rate, 1) == 3);
    CHECK(throws_invalid_argument([&] { condition_latent_length(rate, 0); }));
}

void test_window_arithmetic() {
    using namespace tts_cpp::minimax::detail;
    CHECK(window_starts(200, kWindowFrames, 100) == std::vector<int64_t>({0}));
    CHECK(window_starts(300, kWindowFrames, 100) == std::vector<int64_t>({0, 100}));
    CHECK(window_starts(301, kWindowFrames, 100) == std::vector<int64_t>({0, 100, 200}));
    // Every window layout must cover the last frame, whatever the hop. The old
    // `frames - hop` bound only did so when a window spanned exactly two hops.
    for (int64_t hop = 20; hop <= kWindowFrames; hop += 10) {
        for (int64_t frames = 1; frames <= 1200; frames += 7) {
            const std::vector<int64_t> starts = window_starts(frames, kWindowFrames, hop);
            CHECK(!starts.empty());
            CHECK(starts.front() == 0);
            CHECK(starts.back() + kWindowFrames >= frames);
            CHECK(starts.size() == 1 || starts.back() < frames);
        }
    }
    const FlowWindowGeometry hop100 = flow_window_geometry(689, 344);
    const CropSpan first = crop_span(689, 0, 2, 512, hop100);
    const CropSpan second = crop_span(689, 1, 2, 512, hop100);
    CHECK(first.left == 0);
    CHECK(first.length == 431 * 512);
    CHECK(second.left == 86 * 512);
    CHECK(second.length == 603 * 512);
    CHECK(stitched_sample_count({689, 689}, 512, hop100) == 529408);
    // The hop-100 geometry must reproduce the values that were hardcoded before
    // it was derived, and the defining relationships must hold for any hop.
    CHECK(hop100.carry_span == 344);
    CHECK(hop100.overlap == 172);
    CHECK(hop100.crop_left == 86);
    CHECK(hop100.crop_right == 258);
    for (int64_t hop_latents = 4; hop_latents <= 685; hop_latents += 1) {
        const FlowWindowGeometry geometry = flow_window_geometry(689, hop_latents);
        const int64_t shared = 689 - hop_latents;
        CHECK(geometry.carry_span <= shared && geometry.carry_span > shared - 4);
        CHECK(geometry.crop_left + geometry.crop_right == geometry.carry_span);
        CHECK(geometry.carry_span == 2 * geometry.overlap);
        CHECK(geometry.crop_left * 3 == geometry.crop_right);
    }
    CHECK(throws_invalid_argument([] { flow_window_geometry(689, 0); }));
    CHECK(throws_invalid_argument([] { flow_window_geometry(689, 689); }));
    CHECK(throws_invalid_argument([] { flow_window_geometry(689, 687); }));

    // The kept spans must tile the output exactly: if they do not, generation is
    // silently truncated rather than failing. Checked across the frame counts a
    // caller can ask for, at the shipped hop.
    const ConditionRate rate;
    for (int64_t hop = 50; hop <= 190; hop += 10) {
        const int64_t hop_latents = condition_latent_length(rate, hop);
        const int64_t window_latents = condition_latent_length(rate, kWindowFrames);
        const FlowWindowGeometry geometry = flow_window_geometry(window_latents, hop_latents);
        for (int64_t frames = 1; frames <= 2000; frames += 3) {
            const std::vector<int64_t> starts = window_starts(frames, kWindowFrames, hop);
            std::vector<int64_t> latent_lengths;
            for (size_t index = 0; index < starts.size(); ++index) {
                const int64_t length = std::min(starts[index] + kWindowFrames, frames) - starts[index];
                latent_lengths.push_back(condition_latent_length(rate, length));
            }
            // Rounding the shared region down to a multiple of four, and
            // latent_length's own truncation per window, can move each seam by
            // up to three latents in either direction; a single window is exact.
            const int64_t stitched = stitched_sample_count(latent_lengths, 1, geometry);
            const int64_t expected = condition_latent_length(rate, frames);
            const int64_t seams = (int64_t) starts.size() - 1;
            CHECK(std::llabs(stitched - expected) <= 3 * seams);
        }
    }
}

void test_overlap_blend_and_pin() {
    using namespace tts_cpp::minimax::detail;
    std::vector<float> noise = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};
    std::vector<float> previous = {20.0f, 40.0f, 60.0f, 80.0f};
    std::vector<float> latents = noise;
    blend_latent_overlap(latents.data(), noise.data(), previous.data(), 2, 3, 2, 2,
                         0.5f);
    CHECK(close(latents[0], 11.000001f));
    CHECK(close(latents[1], 22.000002f));
    CHECK(close(latents[3], 34.000004f));
    CHECK(close(latents[4], 45.000005f));
    pin_latent_overlap(latents.data(), previous.data(), 2, 3, 2, 2);
    CHECK(latents == std::vector<float>({20.0f, 40.0f, 6.0f, 60.0f, 80.0f, 12.0f}));
}

void test_carry_layout() {
    using namespace tts_cpp::minimax::detail;
    std::vector<float> latents(2 * 689);
    std::vector<float> condition(3 * 689);
    for (size_t index = 0; index < latents.size(); ++index) {
        latents[index] = static_cast<float>(index);
    }
    for (size_t index = 0; index < condition.size(); ++index) {
        condition[index] = static_cast<float>(index);
    }
    const CarryRange range = carry_range(689, 344, 172);
    CHECK(range.start == 345);
    CHECK(range.end == 517);
    std::vector<float> carry_latents;
    std::vector<float> carry_condition;
    CHECK(copy_carry_layout(latents, condition, 2, 3, 689, range, carry_latents,
                            carry_condition));
    CHECK(carry_latents.size() == 344);
    CHECK(carry_latents[0] == 345.0f);
    CHECK(carry_latents[171] == 516.0f);
    CHECK(carry_latents[172] == 1034.0f);
    CHECK(carry_condition.size() == 516);
    CHECK(carry_condition.front() == 1035.0f);
    CHECK(carry_condition.back() == 1550.0f);
}

void test_planar_stitch() {
    using tts_cpp::minimax::detail::copy_planar_window;
    const std::vector<float> source = {0.0f, 1.0f, 2.0f, 3.0f,
                                       10.0f, 11.0f, 12.0f, 13.0f};
    std::vector<float> destination(12, -1.0f);
    CHECK(copy_planar_window(source, 2, 4, 1, 6, 2, 2, destination));
    CHECK(destination == std::vector<float>({-1.0f, -1.0f, 1.0f, 2.0f, -1.0f, -1.0f,
                                             -1.0f, -1.0f, 11.0f, 12.0f, -1.0f, -1.0f}));
}

void emit_flow_steps(int64_t steps,
                     const std::function<void(int64_t, int64_t)> & on_step) {
    for (int64_t step = 1; step <= steps; ++step) {
        on_step(step, steps);
    }
}

int count_stage(const std::vector<std::string> & stages, const std::string & stage) {
    return static_cast<int>(std::count(stages.begin(), stages.end(), stage));
}

void fill_fake_latents(int64_t window, int64_t channels, int64_t latent_length,
                       int64_t overlap, std::vector<float> & latents) {
    for (int64_t channel = 0; channel < channels; ++channel) {
        for (int64_t index = overlap; index < latent_length; ++index) {
            latents[static_cast<size_t>(channel * latent_length + index)] =
                static_cast<float>(window * 10000 + channel * 1000 + index);
        }
    }
}

constexpr int64_t kTwoWindowChannels = 2;
constexpr int64_t kTwoWindowConditionDimension = 2;
constexpr int64_t kTwoWindowUpsample = 2;
constexpr int64_t kTwoWindowFlowSteps = 2;
constexpr int64_t kTwoWindowFrames = 201;

struct TwoWindowCapture {
    std::vector<std::string> stages;
    std::vector<float> first_condition;
    std::vector<float> blended;
    std::vector<float> pinned;
};

MM3WindowDimensions make_two_window_dimensions() {
    MM3WindowDimensions dimensions;
    dimensions.latent_channels = kTwoWindowChannels;
    dimensions.audio_channels = kTwoWindowChannels;
    dimensions.condition_dimension = kTwoWindowConditionDimension;
    dimensions.upsample = kTwoWindowUpsample;
    dimensions.window_frames = 200;
    dimensions.hop_frames = 100;
    dimensions.carry_span = 344;
    dimensions.overlap = 172;
    dimensions.crop_left = 86;
    dimensions.crop_right = 258;
    dimensions.flow_steps = kTwoWindowFlowSteps;
    return dimensions;
}

bool capture_two_window_progress(TwoWindowCapture & capture,
                                 const std::string & stage) {
    capture.stages.push_back(stage);
    return true;
}

void fill_two_window_condition(int64_t window, std::vector<float> & condition) {
    for (size_t index = 0; index < condition.size(); ++index) {
        condition[index] =
            static_cast<float>(window * 10000 + static_cast<int64_t>(index));
    }
}

bool inject_two_window_condition(TwoWindowCapture & capture, int64_t window,
                                 int64_t frames, std::vector<float> & condition,
                                 int64_t & latent_length) {
    latent_length =
        tts_cpp::minimax::detail::condition_latent_length({}, frames);
    condition.resize(static_cast<size_t>(
        latent_length * kTwoWindowConditionDimension));
    fill_two_window_condition(window, condition);
    if (window == 0) {
        capture.first_condition = condition;
    }
    return true;
}

bool inject_two_window_noise(int64_t window, int64_t count,
                             std::vector<float> & noise) {
    noise.assign(static_cast<size_t>(count), window == 0 ? 0.25f : 0.5f);
    return window == 1;
}

void capture_two_window_overlap(TwoWindowCapture & capture, int64_t latent_length,
                                const std::vector<float> & noise,
                                const std::vector<float> & carry_latents,
                                int64_t carry_length,
                                const std::vector<float> & condition,
                                std::vector<float> & latents) {
    CHECK(carry_length == 172);
    CHECK(condition[0] ==
          capture.first_condition[345 * kTwoWindowConditionDimension]);
    CHECK(condition[343] ==
          capture.first_condition[345 * kTwoWindowConditionDimension + 343]);
    tts_cpp::minimax::detail::blend_latent_overlap(
        latents.data(), noise.data(), carry_latents.data(), kTwoWindowChannels,
        latent_length, carry_length, carry_length, 0.5f);
    capture.blended = latents;
    tts_cpp::minimax::detail::pin_latent_overlap(
        latents.data(), carry_latents.data(), kTwoWindowChannels, latent_length,
        carry_length, carry_length);
    capture.pinned = latents;
}

bool inject_two_window_flow(
    TwoWindowCapture & capture, int64_t window, int64_t latent_length,
    const std::vector<float> & noise, const std::vector<float> & carry_latents,
    int64_t carry_length, const std::vector<float> & condition,
    std::vector<float> & latents,
    const std::function<void(int64_t, int64_t)> & on_step) {
    latents = noise;
    if (window == 1) {
        capture_two_window_overlap(capture, latent_length, noise, carry_latents,
                                   carry_length, condition, latents);
    }
    fill_fake_latents(window, kTwoWindowChannels, latent_length, carry_length,
                      latents);
    emit_flow_steps(kTwoWindowFlowSteps, on_step);
    return true;
}

bool inject_two_window_vocoder(int64_t window, int64_t latent_length,
                               std::vector<float> & waveform) {
    const int64_t samples = latent_length * kTwoWindowUpsample;
    waveform.resize(static_cast<size_t>(samples * kTwoWindowChannels));
    std::fill(waveform.begin(), waveform.begin() + samples,
              window == 0 ? 0.1f : 0.3f);
    std::fill(waveform.begin() + samples, waveform.end(),
              window == 0 ? 0.2f : 0.4f);
    return true;
}

MM3WindowOperations make_two_window_operations(TwoWindowCapture & capture) {
    MM3WindowOperations operations;
    operations.progress =
        [&capture](const std::string & stage, int64_t, int64_t, int64_t, int64_t,
                   std::string *) {
            return capture_two_window_progress(capture, stage);
        };
    operations.condition =
        [&capture](int64_t window, int64_t, int64_t, int64_t frames,
                   std::vector<float> & condition, int64_t & latent_length,
                   std::string *) {
            return inject_two_window_condition(capture, window, frames, condition,
                                               latent_length);
        };
    operations.noise =
        [](int64_t window, int64_t count, std::vector<float> & noise) {
            return inject_two_window_noise(window, count, noise);
        };
    operations.flow =
        [&capture](
            int64_t window, int64_t, int64_t latent_length,
            const std::vector<float> & noise,
            const std::vector<float> & carry_latents, int64_t carry_length,
            const std::vector<float> & condition, std::vector<float> & latents,
            const std::function<void(int64_t, int64_t)> & on_step,
            std::string *) {
            return inject_two_window_flow(
                capture, window, latent_length, noise, carry_latents, carry_length,
                condition, latents, on_step);
        };
    operations.vocoder =
        [](int64_t window, int64_t, const std::vector<float> &, int64_t latent_length,
           std::vector<float> & waveform, std::string *) {
            return inject_two_window_vocoder(window, latent_length, waveform);
        };
    return operations;
}

void assert_two_window_expected_state(const MM3WindowOrchestration & result,
                                      const std::string & error) {
    CHECK(error.empty());
    CHECK(result.starts == std::vector<int64_t>({0, 100}));
    CHECK(result.frame_lengths == std::vector<int64_t>({200, 101}));
    CHECK(result.latent_lengths == std::vector<int64_t>({689, 347}));
    CHECK(result.overlaps == std::vector<int64_t>({0, 172}));
    CHECK(result.carry_starts[0] == 345);
    CHECK(result.carry_ends[0] == 517);
    CHECK(result.vocoder_calls == 2);
    CHECK(result.samples_per_channel == 1384);
    CHECK(result.audio.size() == 2768);
    CHECK(result.forced_noise == std::vector<int64_t>({0, 1}));
}

void assert_two_window_crops(const MM3WindowDimensions & dimensions) {
    const auto first_crop = mm3_window_crop_span(dimensions, 689, 0, 2);
    const auto second_crop = mm3_window_crop_span(dimensions, 347, 1, 2);
    CHECK(first_crop.left == 0);
    CHECK(first_crop.length == 862);
    CHECK(second_crop.left == 172);
    CHECK(second_crop.length == 522);
}

void assert_two_window_seams(const MM3WindowOrchestration & result,
                             const TwoWindowCapture & capture) {
    CHECK(close(result.audio[861], 0.1f));
    CHECK(close(result.audio[862], 0.3f));
    CHECK(close(result.audio[1384 + 861], 0.2f));
    CHECK(close(result.audio[1384 + 862], 0.4f));
    CHECK(!capture.blended.empty());
    CHECK(!capture.pinned.empty());
    CHECK(!close(capture.blended[0], capture.pinned[0]));
    CHECK(capture.pinned[0] == result.latents[0][345]);
}

void assert_two_window_progress(const TwoWindowCapture & capture) {
    CHECK(count_stage(capture.stages, "cond") == 2);
    CHECK(count_stage(capture.stages, "flow") == 6);
    CHECK(count_stage(capture.stages, "vocode") == 2);
    CHECK(count_stage(capture.stages, "stitch") == 1);
    CHECK(count_stage(capture.stages, "done") == 1);
}

void test_two_window_orchestration() {
    const MM3WindowDimensions dimensions = make_two_window_dimensions();
    TwoWindowCapture capture;
    const MM3WindowOperations operations = make_two_window_operations(capture);
    MM3WindowOrchestration result;
    std::string error;
    CHECK(mm3_orchestrate_windows(kTwoWindowFrames, dimensions, operations, &result,
                                  &error));
    assert_two_window_expected_state(result, error);
    assert_two_window_crops(dimensions);
    assert_two_window_seams(result, capture);
    assert_two_window_progress(capture);
}

void test_nonfinite_vocoder_orchestration(float value) {
    const MM3WindowDimensions dimensions = make_two_window_dimensions();
    TwoWindowCapture capture;
    MM3WindowOperations operations = make_two_window_operations(capture);
    operations.vocoder =
        [value](int64_t, int64_t, const std::vector<float> &, int64_t latent_length,
                std::vector<float> & waveform, std::string *) {
            const int64_t samples = latent_length * kTwoWindowUpsample;
            waveform.assign(static_cast<size_t>(samples * kTwoWindowChannels), 0.25f);
            waveform[4] = value;
            return true;
        };
    MM3WindowOrchestration result;
    std::string error;
    CHECK(!mm3_orchestrate_windows(1, dimensions, operations, &result, &error));
    CHECK(error == "non-finite audio sample at index 4");
    CHECK(result.audio.size() == 12);
    CHECK(!std::isfinite(result.audio[4]));
}

void test_nonfinite_vocoder_orchestration() {
    test_nonfinite_vocoder_orchestration(std::numeric_limits<float>::quiet_NaN());
    test_nonfinite_vocoder_orchestration(std::numeric_limits<float>::infinity());
    test_nonfinite_vocoder_orchestration(-std::numeric_limits<float>::infinity());
}

void test_finite_vocoder_orchestration_clipping() {
    const MM3WindowDimensions dimensions = make_two_window_dimensions();
    TwoWindowCapture capture;
    MM3WindowOperations operations = make_two_window_operations(capture);
    operations.vocoder =
        [](int64_t, int64_t, const std::vector<float> &, int64_t latent_length,
           std::vector<float> & waveform, std::string *) {
            const int64_t samples = latent_length * kTwoWindowUpsample;
            waveform.assign(static_cast<size_t>(samples * kTwoWindowChannels), 0.0f);
            waveform[0] = 2.0f;
            waveform[1] = -2.0f;
            waveform[2] = 0.5f;
            return true;
        };
    MM3WindowOrchestration result;
    std::string error;
    CHECK(mm3_orchestrate_windows(1, dimensions, operations, &result, &error));
    CHECK(error.empty());
    CHECK(result.audio[0] == 1.0f);
    CHECK(result.audio[1] == -1.0f);
    CHECK(result.audio[2] == 0.5f);
    CHECK(close(result.metrics.peak, 1.0f));
    CHECK(std::fabs(result.metrics.rms - std::sqrt(0.1875)) <= 1e-12);
}

void test_ar_candidate_helpers() {
    using namespace tts_cpp::minimax::detail;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const std::vector<float> logits = {
        0.0f, 9.0f, 0.0f, nan, inf, -inf,
        0.0f, 4.0f, 0.0f, 1.0f, 2.0f, 3.0f,
    };
    ArCandidates candidates;
    int64_t nonfinite = 0;
    CHECK(collect_ar_candidates(logits.data(), 6, 1, 3, 3, candidates, nonfinite));
    CHECK(candidates.conditional.size() == 4);
    CHECK(candidates.conditional[0] == 9.0f);
    CHECK(std::isinf(candidates.conditional[1]) && candidates.conditional[1] < 0.0f);
    CHECK(std::isinf(candidates.conditional[2]) && candidates.conditional[2] < 0.0f);
    CHECK(std::isinf(candidates.conditional[3]) && candidates.conditional[3] < 0.0f);
    CHECK(candidates.unconditional == std::vector<float>({4.0f, 1.0f, 2.0f, 3.0f}));
    CHECK(nonfinite == 2);

    candidates.conditional = {4.0f, 3.0f, 3.0f, 1.0f};
    candidates.unconditional = {0.0f, 0.0f, 0.0f, 0.0f};
    guide_ar_candidates(candidates, 2.0f);
    apply_conditional_top_k(candidates, 2);
    CHECK(candidates.guided[0] == 8.0f);
    CHECK(candidates.guided[1] == 6.0f);
    CHECK(candidates.guided[2] == 6.0f);
    CHECK(std::isinf(candidates.guided[3]) && candidates.guided[3] < 0.0f);
}

void test_compact_head_layout() {
    using namespace tts_cpp::minimax::detail;

    CHECK(compact_head_row_count(16384) == 16385);
    CHECK(compact_head_row_count(0) == 1);

    // A synthetic 10-row, 4-byte-row "vocabulary": semantic block at rows
    // [3,6), EOS at row 1.
    std::array<CompactHeadCopy, 2> plan{};
    CHECK(compact_head_copy_plan(10, 3, 3, 1, 4, plan));
    CHECK(plan[0].src_offset == 12 && plan[0].dst_offset == 0 && plan[0].nbytes == 12);
    CHECK(plan[1].src_offset == 4 && plan[1].dst_offset == 12 && plan[1].nbytes == 4);
    CHECK(!compact_head_copy_plan(10, 8, 3, 1, 4, plan));
    CHECK(!compact_head_copy_plan(10, 3, 3, 10, 4, plan));
    CHECK(!compact_head_copy_plan(10, 3, 3, 1, 0, plan));

    // Apply the plan to a byte buffer and confirm the compact output holds
    // exactly the semantic block followed by the EOS row.
    std::vector<uint8_t> source(10 * 4);
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<uint8_t>(i);
    }
    std::vector<uint8_t> compact(static_cast<size_t>(compact_head_row_count(3)) * 4, 0xAA);
    memcpy(compact.data() + plan[0].dst_offset, source.data() + plan[0].src_offset, plan[0].nbytes);
    memcpy(compact.data() + plan[1].dst_offset, source.data() + plan[1].src_offset, plan[1].nbytes);
    CHECK(compact == std::vector<uint8_t>({12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 4, 5, 6, 7}));

    // Index mapping: candidates collected from the compact layout must match
    // candidates collected directly from the full-vocab layout, bit-for-bit.
    const int64_t vocab = 20;
    const int64_t semantic_offset = 5;
    const int64_t semantic_vocab = 4;
    const int64_t eos = 2;
    std::vector<float> full(static_cast<size_t>(vocab) * 2);
    for (size_t i = 0; i < full.size(); ++i) {
        full[i] = static_cast<float>(i);
    }
    ArCandidates from_full;
    int64_t nonfinite_full = 0;
    CHECK(collect_ar_candidates(full.data(), vocab, eos, semantic_offset, semantic_vocab, from_full,
                                nonfinite_full));

    const int64_t compact_vocab = compact_head_row_count(semantic_vocab);
    std::vector<float> compact_logits(static_cast<size_t>(compact_vocab) * 2);
    for (int64_t row = 0; row < 2; ++row) {
        const float * src_row = full.data() + row * vocab;
        float *       dst_row = compact_logits.data() + row * compact_vocab;
        memcpy(dst_row, src_row + semantic_offset, static_cast<size_t>(semantic_vocab) * sizeof(float));
        dst_row[semantic_vocab] = src_row[eos];
    }
    ArCandidates from_compact;
    int64_t nonfinite_compact = 0;
    CHECK(collect_ar_candidates(compact_logits.data(), compact_vocab, semantic_vocab, kCompactHeadSemanticOffset,
                                semantic_vocab, from_compact, nonfinite_compact));
    CHECK(from_full.conditional == from_compact.conditional);
    CHECK(from_full.unconditional == from_compact.unconditional);
}

void test_ar_acoustic_rows_and_frame_cap() {
    using namespace tts_cpp::minimax::detail;
    const int32_t codes[] = {2, 3, 4};
    std::vector<int32_t> rows;
    build_acoustic_rows(codes, 3, 10, rows);
    CHECK(rows == std::vector<int32_t>({2, 13, 24}));
    std::string error;
    CHECK(resolve_ar_frame_cap(20, 12, true, 6, error) == 5);
    CHECK(error.empty());
    CHECK(resolve_ar_frame_cap(20, 12, false, 0, error) == 12);
    CHECK(resolve_ar_frame_cap(20, 12, true, 1, error) == 0);
    CHECK(!error.empty());
}

void test_sampling_edges() {
    using tts_cpp::minimax::detail::sample_top_k;
    std::mt19937_64 random(42);
    CHECK(sample_top_k(nullptr, 0, 50, random) == 0);
    const float logits[] = {-5.0f, 3.0f, 2.0f};
    CHECK(sample_top_k(logits, 3, 1, random) == 1);
    const float unusual[] = {std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity()};
    CHECK(sample_top_k(unusual, 3, 1, random) == 1);
    std::mt19937_64 first(7);
    std::mt19937_64 second(7);
    CHECK(sample_top_k(logits, 3, 3, first) == sample_top_k(logits, 3, 3, second));
}

void test_model_compatibility() {
    using namespace tts_cpp::minimax::detail;
    ModelCompatibility model = valid_compatibility();
    CHECK(validate_model_compatibility(model).empty());
    model.condition_out = 1024;
    CHECK(!validate_model_compatibility(model).empty());
    model = valid_compatibility();
    model.dit_channels = 64;
    CHECK(!validate_model_compatibility(model).empty());
    model = valid_compatibility();
    model.depth_codebooks = 7;
    CHECK(!validate_model_compatibility(model).empty());
    model = valid_compatibility();
    model.components.push_back("vocoder");
    CHECK(!validate_model_compatibility(model).empty());
    model = valid_compatibility();
    model.condition_rate.output_sampling_rate = 32000;
    CHECK(!validate_model_compatibility(model).empty());
}

void test_unicode_categories() {
    CHECK(is_digit(0x0661));
    CHECK(is_letter(0x4E2D));
    CHECK(is_letter(0x00E9));
    CHECK(!is_letter(0x1F600));
    CHECK(!is_digit(0x1F600));
    CHECK(is_letter(0x10400));
    CHECK(is_whitespace(0x2003));
    CHECK(!is_whitespace(0x200B));
    CHECK(gpt2_pre_tokenize(u8"\u00E9\u4E2D\U00010400") ==
          std::vector<std::string>({u8"\u00E9\u4E2D\U00010400"}));
    CHECK(gpt2_pre_tokenize(u8"\u0661\u0662\u0663\u0664") ==
          std::vector<std::string>({u8"\u0661", u8"\u0662", u8"\u0663", u8"\u0664"}));
    CHECK(gpt2_pre_tokenize(u8"\U0001F600\u00E9") ==
          std::vector<std::string>({u8"\U0001F600\u00E9"}));
    CHECK(gpt2_pre_tokenize(u8"\u2003\u4E2D") ==
          std::vector<std::string>({u8"\u2003\u4E2D"}));
}

void test_pre_tokenization_branches() {
    CHECK(gpt2_pre_tokenize(u8"I'LL we\u2019RE they've dog's can't I'm he'd") ==
          std::vector<std::string>({"I", "'LL", " we", u8"\u2019RE", " they", "'ve",
                                    " dog", "'s", " can", "'t", " I", "'m", " he", "'d"}));
    CHECK(gpt2_pre_tokenize("'llama") == std::vector<std::string>({"'llama"}));
    CHECK(gpt2_pre_tokenize(u8"Hello\u00E9\u4E2D\U00010400") ==
          std::vector<std::string>({u8"Hello\u00E9\u4E2D\U00010400"}));
    CHECK(gpt2_pre_tokenize(u8".world") == std::vector<std::string>({u8".world"}));
    CHECK(gpt2_pre_tokenize(u8"\u0667") == std::vector<std::string>({u8"\u0667"}));
    CHECK(gpt2_pre_tokenize("!?\r\n") == std::vector<std::string>({"!?\r\n"}));
    CHECK(gpt2_pre_tokenize("  hello") == std::vector<std::string>({" ", " hello"}));
    CHECK(gpt2_pre_tokenize(" \t!") == std::vector<std::string>({" ", "\t!"}));
    CHECK(gpt2_pre_tokenize("  7") == std::vector<std::string>({" ", " ", "7"}));
    CHECK(gpt2_pre_tokenize(" \t\n") == std::vector<std::string>({" \t\n"}));
    CHECK(gpt2_pre_tokenize(u8"\u2003\u2003\u4E2D") ==
          std::vector<std::string>({u8"\u2003", u8"\u2003\u4E2D"}));
}

void test_malformed_utf8() {
    const std::string truncated_four_byte("\xF0", 1);
    const std::string truncated_sequence("\xF0\x9F", 2);
    const std::string invalid_prefix("\x80" "A", 2);
    CHECK(gpt2_pre_tokenize(truncated_four_byte) ==
          std::vector<std::string>({truncated_four_byte}));
    CHECK(gpt2_pre_tokenize(truncated_sequence) ==
          std::vector<std::string>({std::string("\xF0", 1), std::string("\x9F", 1)}));
    CHECK(gpt2_pre_tokenize(invalid_prefix) ==
          std::vector<std::string>({invalid_prefix}));
    int advance = 0;
    CHECK(utf8_codepoint(truncated_sequence.data(),
                         static_cast<int>(truncated_sequence.size()), &advance) == 0xF0);
    CHECK(advance == 1);
}

void test_model_pair_resolution() {
    namespace fs = std::filesystem;
    using tts_cpp::minimax::detail::ModelPair;
    using tts_cpp::minimax::detail::resolve_model_pair;
    const fs::path root = fs::temp_directory_path() /
                          ("minimax-model-pair-" + std::to_string(std::random_device{}()));
    fs::create_directories(root / "mm3");
    touch(root / "mm3-lm-f16.gguf");
    touch(root / "mm3-synth-f16.gguf");
    touch(root / "MM3-LM-Q8_0.GGUF");
    touch(root / "mm3-SYNTH-q8_0.gguf");
    ModelPair pair = resolve_model_pair(root.string(), "", "");
    CHECK(pair.quant == "q8_0");
    CHECK(fs::path(pair.lm).filename() == "MM3-LM-Q8_0.GGUF");
    touch(root / "mm3" / "mm3-lm-q8_0.gguf");
    bool ambiguous = false;
    try {
        resolve_model_pair(root.string(), "", "");
    } catch (const std::runtime_error &) {
        ambiguous = true;
    }
    CHECK(ambiguous);
    fs::remove_all(root);
}

void test_model_pair_resolution_q4() {
    namespace fs = std::filesystem;
    using tts_cpp::minimax::detail::ModelPair;
    using tts_cpp::minimax::detail::resolve_model_pair;
    const fs::path root = fs::temp_directory_path() /
                          ("minimax-model-pair-q4-" + std::to_string(std::random_device{}()));
    fs::create_directories(root);
    touch(root / "mm3-lm-q4_k_m.gguf");
    touch(root / "mm3-synth-q4_k_m.gguf");
    ModelPair q4_only = resolve_model_pair(root.string(), "", "");
    CHECK(q4_only.quant == "q4_k_m");

    touch(root / "mm3-lm-q8_0.gguf");
    touch(root / "mm3-synth-q8_0.gguf");
    ModelPair prefers_q8 = resolve_model_pair(root.string(), "", "");
    CHECK(prefers_q8.quant == "q8_0");

    fs::remove_all(root);
}

void test_model_pair_resolution_f32() {
    namespace fs = std::filesystem;
    using tts_cpp::minimax::detail::ModelPair;
    using tts_cpp::minimax::detail::resolve_model_pair;
    const fs::path root = fs::temp_directory_path() /
                          ("minimax-model-pair-f32-" + std::to_string(std::random_device{}()));
    fs::create_directories(root);
    touch(root / "mm3-lm-f32.gguf");
    touch(root / "mm3-synth-f32.gguf");
    ModelPair f32_only = resolve_model_pair(root.string(), "", "");
    CHECK(f32_only.quant == "f32");

    touch(root / "mm3-lm-q4_k_m.gguf");
    touch(root / "mm3-synth-q4_k_m.gguf");
    ModelPair prefers_q4 = resolve_model_pair(root.string(), "", "");
    CHECK(prefers_q4.quant == "q4_k_m");

    touch(root / "mm3-lm-q6_k.gguf");
    touch(root / "mm3-synth-q6_k.gguf");
    ModelPair prefers_q6 = resolve_model_pair(root.string(), "", "");
    CHECK(prefers_q6.quant == "q6_k");

    touch(root / "mm3-lm-q8_0.gguf");
    touch(root / "mm3-synth-q8_0.gguf");
    ModelPair prefers_q8 = resolve_model_pair(root.string(), "", "");
    CHECK(prefers_q8.quant == "q8_0");

    fs::remove_all(root);
}

void test_backend_configuration() {
    backend_set_abort_handler(nullptr, nullptr, nullptr);
    using tts_cpp::minimax::detail::backend_configuration_matches;
    CHECK(backend_configuration_matches(0, 4, "first", 8, "second"));
    CHECK(backend_configuration_matches(1, 4, "backends/.", 4, "backends"));
    CHECK(!backend_configuration_matches(1, 4, "backends", 8, "backends"));
    CHECK(!backend_configuration_matches(1, 4, "first", 4, "second"));
}

void test_device_configuration() {
    CHECK(throws_runtime_error([] { backend_configure_device("fast"); }));

    set_env("MM3_DEVICE", "auto");
    backend_configure_device("cpu");
    CHECK(g_backend_device == "cpu");

    backend_configure_device("");
    CHECK(g_backend_device == "auto");

    set_env("MM3_DEVICE", "vulkan");
    CHECK(throws_runtime_error([] { backend_configure_device(""); }));

    set_env("MM3_DEVICE", nullptr);
    backend_configure_device("");
    CHECK(g_backend_device == "cpu");
}

void test_device_backend_init() {
    ggml_backend_load_all();
    ggml_backend_t probe = tts_cpp::acestep::backend_gpu_init();
    const bool gpu_available = probe != nullptr;
    if (probe) {
        ggml_backend_free(probe);
    }

    backend_configure_device("cpu");
    BackendPair cpu_pair = backend_init("test");
    CHECK(cpu_pair.cpu_backend != nullptr);
    CHECK(!cpu_pair.has_gpu);
    CHECK(cpu_pair.backend == cpu_pair.cpu_backend);
    CHECK(g_backend_gpu_fallback_reason == tts_cpp::GpuFallbackReason::not_requested);
    CHECK(throws_runtime_error([] { backend_configure_device("auto"); }));
    backend_release(cpu_pair.backend, cpu_pair.cpu_backend);

    backend_configure_device("auto");
    BackendPair auto_pair = backend_init("test");
    CHECK(auto_pair.cpu_backend != nullptr);
    CHECK(auto_pair.has_gpu == gpu_available);
    if (gpu_available) {
        CHECK(g_backend_gpu_fallback_reason == tts_cpp::GpuFallbackReason::none);
    } else {
        CHECK(g_backend_gpu_fallback_reason == tts_cpp::GpuFallbackReason::no_devices ||
              g_backend_gpu_fallback_reason == tts_cpp::GpuFallbackReason::init_failed);
    }
    backend_release(auto_pair.backend, auto_pair.cpu_backend);

    backend_configure_device("gpu");
    if (gpu_available) {
        BackendPair gpu_pair = backend_init("test");
        CHECK(gpu_pair.has_gpu);
        CHECK(gpu_pair.backend != gpu_pair.cpu_backend);
        backend_release(gpu_pair.backend, gpu_pair.cpu_backend);
    } else {
        CHECK(throws_runtime_error([] { backend_init("test"); }));
    }
    backend_configure_device("cpu");
}

void test_flash_attn_policy() {
    CHECK(!mm3_use_flash_attn(false, false, "MM3_LM_NO_FLASH", "MM3_LM_FLASH"));
    CHECK(!mm3_use_flash_attn(false, true, "MM3_DIT_NO_FLASH", nullptr));

    set_env("MM3_LM_NO_FLASH", nullptr);
    set_env("MM3_LM_FLASH", nullptr);
    set_env("MM3_DIT_NO_FLASH", nullptr);

    CHECK(!mm3_use_flash_attn(true, false, "MM3_LM_NO_FLASH", "MM3_LM_FLASH"));
    CHECK(mm3_use_flash_attn(true, true, "MM3_DIT_NO_FLASH", nullptr));

    set_env("MM3_LM_FLASH", "1");
    CHECK(mm3_use_flash_attn(true, false, "MM3_LM_NO_FLASH", "MM3_LM_FLASH"));

    set_env("MM3_LM_NO_FLASH", "1");
    CHECK(!mm3_use_flash_attn(true, false, "MM3_LM_NO_FLASH", "MM3_LM_FLASH"));

    set_env("MM3_DIT_NO_FLASH", "1");
    CHECK(!mm3_use_flash_attn(true, true, "MM3_DIT_NO_FLASH", nullptr));

    set_env("MM3_DIT_NO_FLASH", "0");
    CHECK(mm3_use_flash_attn(true, true, "MM3_DIT_NO_FLASH", nullptr));

    set_env("MM3_LM_NO_FLASH", nullptr);
    set_env("MM3_LM_FLASH", nullptr);
    set_env("MM3_DIT_NO_FLASH", nullptr);

    CHECK(mm3_lm_flash_attn_default("CUDA"));
    CHECK(mm3_lm_flash_attn_default("Vulkan"));
    CHECK(!mm3_lm_flash_attn_default("Metal"));
    CHECK(!mm3_lm_flash_attn_default("MTL"));
    CHECK(!mm3_lm_flash_attn_default("OpenCL"));
    CHECK(!mm3_lm_flash_attn_default("CPU"));
    CHECK(!mm3_lm_flash_attn_default(""));
    CHECK(!mm3_lm_flash_attn_default(nullptr));
}

void test_replay_io() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "mm3-replay-io-test";
    fs::remove_all(dir);
    std::string error;
    CHECK(mm3_replay_prepare_output_dir(dir.string(), &error));
    CHECK(mm3_replay_prepare_output_dir(dir.string(), &error));

    const std::vector<float> payload = {1.0f, -2.5f, 0.25f};
    const std::string raw_path = (dir / "data.f32").string();
    CHECK(mm3_replay_write_raw(raw_path, payload.data(), payload.size()));
    std::vector<float> loaded;
    CHECK(mm3_replay_read_raw(raw_path, loaded));
    CHECK(loaded == payload);

    const std::vector<float> planar = {0.5f, -0.5f, 1.5f, -1.5f};
    const std::string wav_path = (dir / "audio.wav").string();
    CHECK(mm3_replay_write_wav(wav_path, planar, 2, 44100));
    CHECK(fs::file_size(wav_path) == 44 + 2 * 2 * sizeof(int16_t));

    const std::string missing = (dir / "missing-subdir" / "data.f32").string();
    CHECK(!mm3_replay_write_raw(missing, payload.data(), payload.size()));
    CHECK(!mm3_replay_write_wav(missing, planar, 2, 44100));

    std::string not_a_dir_error;
    CHECK(!mm3_replay_prepare_output_dir(raw_path, &not_a_dir_error));
    CHECK(!not_a_dir_error.empty());

    std::vector<float> absent;
    CHECK(!mm3_replay_read_raw((dir / "absent.bin").string(), absent));

    const std::string empty_path = (dir / "empty.f32").string();
    CHECK(mm3_replay_write_raw(empty_path, payload.data(), 0));
    std::vector<float> from_empty;
    CHECK(!mm3_replay_read_raw(empty_path, from_empty));

    const std::string truncated_path = (dir / "truncated.f32").string();
    const char truncated_bytes[5] = {1, 2, 3, 4, 5};
    CHECK(mm3_replay_write_raw(truncated_path, truncated_bytes, sizeof(truncated_bytes)));
    std::vector<float> from_truncated;
    CHECK(!mm3_replay_read_raw(truncated_path, from_truncated));

    CHECK(mm3_replay_mode_is_supported("full"));
    CHECK(mm3_replay_mode_is_supported("replay"));
    CHECK(mm3_replay_mode_is_supported("condcheck"));
    CHECK(!mm3_replay_mode_is_supported("ful"));
    CHECK(!mm3_replay_mode_is_supported(""));
    fs::remove_all(dir);
}

void test_engine_instance_limit() {
    using tts_cpp::minimax::detail::engine_instance_available;
    CHECK(engine_instance_available(0));
    CHECK(!engine_instance_available(1));
}

void test_cancellation() {
    using tts_cpp::minimax::detail::cancellation_requested;
    CHECK(!cancellation_requested({}));
    CHECK(cancellation_requested([] { return true; }));
    CHECK(!cancellation_requested([] { return false; }));
}

void test_progress_cancellation() {
    bool cancelled = false;
    std::string error;
    const MM3ProgressCb progress = [&cancelled](const MM3GenProgress &) {
        cancelled = true;
    };
    CHECK(!mm3_emit_progress(progress, {"stitch", -1, 1, 0, 1},
                             [&cancelled] { return cancelled; }, &error));
    CHECK(error == MM3_ERR_CANCELLED);
}

}

int main() {
    test_frame_validation();
    test_prompt();
    test_request_text_primitives();
    test_request_line_splitting();
    test_request_replace_and_tags();
    test_request_markdown_helpers();
    test_request_clean_caption();
    test_request_normalize_lyrics();
    test_request_assemble_prompt();
    test_unconditional_mask();
    test_noise();
    test_flow_schedule();
    test_depth_step_layout();
    test_depth_kv_cache_matches_full_prefix();
    test_depth_block_stacked_matches_separate();
    test_lm_positions_tile_cfg_rows();
    test_lm_block_stacking_and_cache();
    test_linear_folds_cfg_rows_into_columns();
    test_linear_precision_follows_weight_type();
    test_dit_block_batches_cfg_branches();
    test_production_dit_readback_preserves_velocity();
    test_depth_step_fused_readback_matches_source_tensors();
    test_production_cfg_euler_step();
    test_malformed_synthesis_metadata();
    test_vocoder_output_shape();
    test_copy_ranges();
    test_condition_length();
    test_window_arithmetic();
    test_overlap_blend_and_pin();
    test_carry_layout();
    test_planar_stitch();
    test_two_window_orchestration();
    test_nonfinite_vocoder_orchestration();
    test_finite_vocoder_orchestration_clipping();
    test_ar_candidate_helpers();
    test_compact_head_layout();
    test_ar_acoustic_rows_and_frame_cap();
    test_sampling_edges();
    test_model_compatibility();
    test_unicode_categories();
    test_pre_tokenization_branches();
    test_malformed_utf8();
    test_model_pair_resolution();
    test_model_pair_resolution_q4();
    test_model_pair_resolution_f32();
    test_backend_configuration();
    test_device_configuration();
    test_device_backend_init();
    test_flash_attn_policy();
    test_replay_io();
    test_engine_instance_limit();
    test_cancellation();
    test_progress_cancellation();
    std::fprintf(stderr, "[test-minimax-units] %d/%d checks passed\n", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
