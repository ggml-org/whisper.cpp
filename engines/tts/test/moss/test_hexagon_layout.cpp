#include "gguf_fixtures.h"
#include "sfx_fixtures.h"
#include "../../../test/moss_precision.h"

#include "moss/delay_lm.h"
#include "moss/sfx_networks.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tts_cpp::moss::detail;
using namespace moss_fixtures;

namespace {

constexpr int THREADS = 2;
constexpr int CONTEXT = 64;
constexpr uint32_t FIXTURE_SEED = 26896;
constexpr float SLOT_RELATIVE_TOLERANCE = 1e-2f;
constexpr int SLOT_STEPS = 3;
constexpr int FUSED_STEPS = 5;
constexpr double FUSED_RELATIVE_TOLERANCE = 2e-3;
constexpr double NORM_FLOOR = 1e-20;
constexpr int WIDE_EMBEDDING = 32;
constexpr size_t GATE_CONTEXT_BYTES = 4096;
constexpr int64_t GATE_TENSOR_ELEMENTS = 32;
constexpr int SFX_FRAMES = 17;
constexpr float SFX_TIMESTEPS[] = {1000.0f, 937.5f, 0.0f};
constexpr float SFX_LATENT_AMPLITUDE = 0.2f;
constexpr float SFX_LATENT_FREQUENCY = 0.31f;
const std::vector<int32_t> SFX_PROMPT = {1, 8, 9, 2};

int failures = 0;

void check(bool condition, const std::string & label) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", label.c_str());
        failures++;
    }
}

struct Fixture {
    std::filesystem::path path;
    ~Fixture() { std::error_code ec; std::filesystem::remove(path, ec); }
    std::string str() const { return path.string(); }
};

std::filesystem::path random_backbone(const char * tag, ggml_type precision, int n_embd = N_EMBD) {
    std::mt19937 rng(FIXTURE_SEED);
    GgufBuilder b;
    b.random = &rng;
    add_backbone_meta(b.file, n_embd);
    add_backbone_tensors(b, n_embd);
    const auto path = temp_gguf(tag);
    b.write(path);
    moss_fixture_precision(path, precision);
    return path;
}

bool close_to(float expected, float actual) {
    return std::fabs(expected - actual) <= SLOT_RELATIVE_TOLERANCE * std::max(1.0f, std::fabs(expected));
}

bool slots_match(const std::vector<float> & full, const std::vector<float> & slots) {
    return close_to(full[GEN_SLOT], slots[GEN_SLOT]) && close_to(full[DELAY_SLOT], slots[DELAY_SLOT]);
}

bool finite_only_at_slots(const std::vector<float> & text) {
    for (size_t id = 0; id < text.size(); ++id) {
        const bool slot = id == (size_t) GEN_SLOT || id == (size_t) DELAY_SLOT;
        if (std::isfinite(text[id]) != slot) {
            return false;
        }
    }
    return true;
}

bool all_masked(const std::vector<float> & text) {
    return std::all_of(text.begin(), text.end(), [](float value) { return std::isinf(value) && value < 0; });
}

std::vector<DelayRow> prompt_rows() {
    return {{1, {0, 1}}, {3, {2, 3}}, {6, {4, 5}}};
}

void check_slot_steps(DelayLM & device, DelayLM & host, const DelayRow & row, const std::string & tier) {
    for (int i = 0; i < SLOT_STEPS; ++i) {
        const DelayLogits full = device.step(row);
        const DelayLogits slots = host.step(row, TextLogits::Slots);
        check(slots_match(full.text, slots.text), tier + ": slot head matches the whole head's rows");
        check(finite_only_at_slots(slots.text), tier + ": slot logits leave every other token masked");
        check(full.audio == slots.audio, tier + ": slot steps keep the audio heads");
    }
}

void test_placement_matches(ggml_type precision, int n_embd) {
    const std::string tier = std::string(ggml_type_name(precision)) + " width " + std::to_string(n_embd);
    Fixture file{random_backbone("host-tables", precision, n_embd)};
    DelayLM device(file.str(), {}, THREADS, CONTEXT, false, {TablePlacement::Device, AttentionLayout::Unfused});
    DelayLM host(file.str(), {}, THREADS, CONTEXT, false, {TablePlacement::Host, AttentionLayout::Unfused});
    check(!device.tables_on_host() && host.tables_on_host(), tier + ": forced placements");

    const std::vector<DelayRow> rows = prompt_rows();
    const DelayLogits expected = device.prefill(rows);
    const DelayLogits actual = host.prefill(rows, TextLogits::Full);
    check(expected.text == actual.text, tier + ": host tables give the whole text head");
    check(expected.audio == actual.audio, tier + ": host tables give the same audio heads");

    check_slot_steps(device, host, rows.back(), tier);

    const DelayLogits drained_full = device.step(rows.back(), TextLogits::None);
    const DelayLogits drained = host.step(rows.back(), TextLogits::None);
    check(all_masked(drained.text), tier + ": forced steps project no text logits");
    check(!all_masked(drained_full.text), tier + ": device tables always project the whole head");
    check(drained_full.audio == drained.audio, tier + ": forced steps keep the audio heads");

    device.reset();
    host.reset();
    check(device.prefill(rows).text == host.prefill(rows).text, tier + ": reset replays the prompt");
}

double relative_l2(const std::vector<float> & expected, const std::vector<float> & actual) {
    if (expected.size() != actual.size()) return INFINITY;
    double error = 0, norm = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double d = (double) expected[i] - actual[i];
        error += d * d;
        norm += (double) expected[i] * expected[i];
    }
    return std::sqrt(error / std::max(norm, NORM_FLOOR));
}

bool audio_close(const DelayLogits & expected, const DelayLogits & actual) {
    for (size_t channel = 0; channel < expected.audio.size(); ++channel) {
        if (relative_l2(expected.audio[channel], actual.audio[channel]) > FUSED_RELATIVE_TOLERANCE) return false;
    }
    return expected.audio.size() == actual.audio.size();
}

bool logits_close(const DelayLogits & expected, const DelayLogits & actual) {
    return relative_l2(expected.text, actual.text) <= FUSED_RELATIVE_TOLERANCE && audio_close(expected, actual);
}

void check_fused_steps(DelayLM & unfused, DelayLM & fused, const std::vector<DelayRow> & rows,
                       const std::string & tier) {
    for (int i = 0; i < FUSED_STEPS; ++i) {
        const DelayRow & row = rows[(size_t) i % rows.size()];
        check(logits_close(unfused.step(row), fused.step(row)), tier + ": fused decoding reads the key-layout value cache");
    }
}

void test_fused_attention_matches_unfused(ggml_type precision) {
    const std::string tier = ggml_type_name(precision);
    Fixture file{random_backbone("fused-attention", precision)};
    DelayLM unfused(file.str(), {}, THREADS, CONTEXT, false, {TablePlacement::Device, AttentionLayout::Unfused});
    DelayLM fused(file.str(), {}, THREADS, CONTEXT, false, {TablePlacement::Device, AttentionLayout::Fused});
    check(!unfused.fused_attention() && fused.fused_attention(), tier + ": forced attention layouts");

    const std::vector<DelayRow> rows = prompt_rows();
    check(logits_close(unfused.prefill(rows), fused.prefill(rows)), tier + ": fused prefill matches");
    check_fused_steps(unfused, fused, rows, tier);
    unfused.reset();
    fused.reset();
    check(logits_close(unfused.prefill(rows), fused.prefill(rows)), tier + ": fused prefill after reset");
}

int64_t tensor_count(const std::filesystem::path & path) {
    gguf_context * file = gguf_init_from_file(path.string().c_str(), {true, nullptr});
    if (!file) throw std::runtime_error("cannot read " + path.string());
    const int64_t count = gguf_get_n_tensors(file);
    gguf_free(file);
    return count;
}

void test_precision_fixture_keeps_the_tensor_list() {
    Fixture f32{random_backbone("tensor-list-f32", GGML_TYPE_F32)};
    Fixture f16{random_backbone("tensor-list-f16", GGML_TYPE_F16)};
    check(tensor_count(f16.path) == tensor_count(f32.path), "re-encoded fixtures add no tensors");
}

void test_hexagon_weight_types() {
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}) {
        check(hexagon_runs_weight_type(type), std::string("Hexagon runs ") + ggml_type_name(type) + " weights");
    }
    for (ggml_type type : {GGML_TYPE_BF16, GGML_TYPE_Q5_0, GGML_TYPE_Q6_K}) {
        check(!hexagon_runs_weight_type(type), std::string("Hexagon refuses ") + ggml_type_name(type) + " weights");
    }
}

void test_first_refused_weight() {
    ggml_context * ctx = ggml_init({GATE_CONTEXT_BYTES, nullptr, true});
    ggml_set_name(ggml_new_tensor_1d(ctx, GGML_TYPE_Q4_0, GATE_TENSOR_ELEMENTS), "accepted.weight");
    check(first_tensor_hexagon_cannot_run(ctx) == nullptr, "a q4_0 context passes the Hexagon gate");
    ggml_tensor * refused = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, GATE_TENSOR_ELEMENTS);
    ggml_set_name(refused, "refused.weight");
    check(first_tensor_hexagon_cannot_run(ctx) == refused, "the gate names the first weight Hexagon cannot run");
    const std::string message = hexagon_weight_refusal(refused, HEXAGON_TTS_WEIGHT_TIERS);
    check(message.find("refused.weight (bf16)") != std::string::npos && message.find("q4_0") != std::string::npos,
          "the refusal names the tensor and the supported tiers");
    check(hexagon_weight_refusal(refused, HEXAGON_SFX_WEIGHT_TIERS).find("q4_0") == std::string::npos,
          "the sound-effect refusal offers no tier its loader rejects");
    ggml_free(ctx);
}

std::filesystem::path random_sfx(const char * tag, ggml_type precision) {
    const auto path = moss_sfx_fixtures::write_sfx_model(tag, FIXTURE_SEED);
    moss_fixture_precision(path, precision);
    return path;
}

std::vector<float> sfx_latents() {
    std::vector<float> values((size_t) SFX_FRAMES * moss_sfx_fixtures::LATENT);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = SFX_LATENT_AMPLITUDE * std::sin((float) i * SFX_LATENT_FREQUENCY);
    }
    return values;
}

void test_sfx_table_placement(ggml_type precision) {
    const std::string tier = std::string("sfx ") + ggml_type_name(precision);
    Fixture file{random_sfx("sfx-host-table", precision)};
    SfxModel device(file.str(), {}, THREADS, false, {TablePlacement::Device, AttentionLayout::Unfused});
    SfxModel host(file.str(), {}, THREADS, false, {TablePlacement::Host, AttentionLayout::Unfused});
    check(!device.table_on_host() && host.table_on_host(), tier + ": forced token table placements");
    check(encode_text(device, SFX_PROMPT) == encode_text(host, SFX_PROMPT),
          tier + ": the host token table gives the same text context");
}

void check_sfx_velocities(SfxModel & unfused, SfxModel & fused, const std::vector<float> & context,
                          const std::string & tier) {
    const std::vector<float> latents = sfx_latents();
    SfxDitSession expected(unfused, SFX_FRAMES);
    SfxDitSession actual(fused, SFX_FRAMES);
    for (float timestep : SFX_TIMESTEPS) {
        check(relative_l2(expected.velocity(latents, timestep, context),
                          actual.velocity(latents, timestep, context)) <= FUSED_RELATIVE_TOLERANCE,
              tier + ": fused DiT self and cross attention match at t=" + std::to_string(timestep));
    }
}

void test_sfx_fused_attention(ggml_type precision) {
    const std::string tier = std::string("sfx ") + ggml_type_name(precision);
    Fixture file{random_sfx("sfx-fused-attention", precision)};
    SfxModel unfused(file.str(), {}, THREADS, false, {TablePlacement::Device, AttentionLayout::Unfused});
    SfxModel fused(file.str(), {}, THREADS, false, {TablePlacement::Device, AttentionLayout::Fused});
    check(!unfused.fused_attention() && fused.fused_attention(), tier + ": forced attention layouts");
    const std::vector<float> context = encode_text(unfused, SFX_PROMPT);
    check(relative_l2(context, encode_text(fused, SFX_PROMPT)) <= FUSED_RELATIVE_TOLERANCE,
          tier + ": fused causal text attention matches");
    check_sfx_velocities(unfused, fused, context, tier);
}

void test_sfx_measured_table() {
    Fixture file{random_sfx("sfx-measured-table", GGML_TYPE_F16)};
    SfxModel device(file.str(), {}, THREADS, true, {TablePlacement::Device, AttentionLayout::Unfused});
    SfxModel host(file.str(), {}, THREADS, true, {TablePlacement::Host, AttentionLayout::Fused});
    const uint64_t table = ggml_nbytes(host.tensor(SFX_TOKEN_TABLE));
    const tts_cpp::FitResult on_device = device.measure_weights();
    const tts_cpp::FitResult on_host = host.measure_weights();
    check(on_device.device.weights_bytes >= on_host.device.weights_bytes + table,
          "a host token table leaves the device weights");
    check(on_host.host_bytes >= on_device.host_bytes + table, "a host token table is priced in host memory");
    check(measure_text(host, SFX_PROMPT).device_bytes > 0, "the text graph measures with a host token table");
    SfxDitSession dit(host, SFX_FRAMES, true);
    check(dit.measure().device_bytes > 0, "the fused DiT graph measures");
}

void test_sfx_cpu_layout() {
    Fixture file{random_sfx("sfx-auto-layout", GGML_TYPE_F32)};
    SfxModel model(file.str(), {}, THREADS);
    check(!model.table_on_host() && !model.fused_attention(), "the CPU keeps the sfx token table and attention");
}

void test_cpu_keeps_tables_on_device() {
    Fixture file{random_backbone("auto-tables", GGML_TYPE_F32)};
    DelayLM model(file.str(), {}, THREADS, CONTEXT);
    check(!model.tables_on_host(), "the CPU keeps every table in its weight buffer");
    check(!model.fused_attention(), "the CPU keeps unfused attention");
}

} // namespace

int main() {
    try {
        test_precision_fixture_keeps_the_tensor_list();
        test_hexagon_weight_types();
        test_first_refused_weight();
        test_cpu_keeps_tables_on_device();
        test_sfx_cpu_layout();
        test_sfx_measured_table();
        for (ggml_type precision : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0}) {
            test_placement_matches(precision, N_EMBD);
            test_fused_attention_matches_unfused(precision);
            test_sfx_table_placement(precision);
            test_sfx_fused_attention(precision);
        }
        for (ggml_type precision : {GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}) {
            test_placement_matches(precision, WIDE_EMBEDDING);
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    if (failures == 0) {
        std::printf("moss hexagon layout: OK\n");
        return 0;
    }
    std::fprintf(stderr, "moss hexagon layout: %d failures\n", failures);
    return 1;
}
