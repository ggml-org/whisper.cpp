#include "gguf_fixtures.h"
#include "../../../test/moss_precision.h"
#include "moss/delay_lm.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>

using namespace moss_fixtures;
using namespace tts_cpp::moss::detail;

namespace {

constexpr uint32_t WEIGHT_SEED = 20261023;
constexpr int EMBEDDING_WIDTH = 32;
constexpr int CONTEXT = 64;
constexpr int THREADS = 2;
constexpr int STEPS = 3;
constexpr int CONTROL_TOKENS = 2;
constexpr int PROJECTION_TOKENS = 1;
constexpr size_t PROBE_TENSORS = 5;
constexpr float LOGIT_TOLERANCE = 1e-2f;

void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

struct Fixture {
    std::filesystem::path path = temp_gguf("metal-text-projection");
    ~Fixture() { std::error_code error; std::filesystem::remove(path, error); }
};

void write_fixture(const Fixture & fixture, ggml_type precision, int delay_slot) {
    std::mt19937 rng(WEIGHT_SEED);
    GgufBuilder builder;
    builder.random = &rng;
    add_backbone_meta(builder.file, EMBEDDING_WIDTH);
    gguf_set_val_u32(builder.file, "moss-tts-delay.audio_assistant_delay_slot_token_id", delay_slot);
    add_backbone_tensors(builder, EMBEDDING_WIDTH);
    builder.write(fixture.path);
    moss_fixture_precision(fixture.path, precision);
}

void check_slots(const DelayLogits & full, const DelayLogits & selected, const DelayConfig & config) {
    require(full.text.size() == selected.text.size(), "text vocabulary size changed");
    require(full.audio == selected.audio, "selecting text logits changed the audio heads");
    for (size_t id = 0; id < full.text.size(); ++id) {
        if (id == (size_t) config.audio_assistant_gen_slot_token_id ||
            id == (size_t) config.audio_assistant_delay_slot_token_id) {
            require(std::isfinite(selected.text[id]), "control logit is not finite");
            require(std::fabs(full.text[id] - selected.text[id]) <=
                    LOGIT_TOLERANCE * std::max(1.0f, std::fabs(full.text[id])),
                    "selected control logit diverges from the full head");
        } else {
            require(std::isinf(selected.text[id]) && selected.text[id] < 0,
                    "a text token outside the requested slots is unmasked");
        }
    }
}

void check_steps(DelayLM & full, DelayLM & selected, const DelayRow & row) {
    for (int i = 0; i < STEPS; ++i) {
        check_slots(full.step(row), selected.step(row, TextLogits::Slots), full.config());
    }
}

void check_no_text(DelayLM & full, DelayLM & selected, const DelayRow & row) {
    const auto expected = full.step(row);
    const auto actual = selected.step(row, TextLogits::None);
    require(expected.audio == actual.audio, "omitting text logits changed the audio heads");
    require(actual.text.size() == expected.text.size(), "drain changed the text vocabulary size");
    require(std::all_of(actual.text.begin(), actual.text.end(), [](float value) {
        return std::isinf(value) && value < 0;
    }), "drain projected text logits");
}

void check_native_projection(ggml_type precision, int delay_slot) {
    auto * device = ggml_backend_dev_by_name("MTL0");
    require(device != nullptr, "Metal device is unavailable");
    auto * ctx = ggml_init({PROBE_TENSORS * ggml_tensor_overhead(), nullptr, true});
    require(ctx != nullptr, "projection probe allocation failed");
    const int stride = std::max(1, delay_slot - GEN_SLOT);
    auto * source = ggml_new_tensor_2d(ctx, precision, EMBEDDING_WIDTH, stride + 1);
    auto * head = ggml_view_2d(ctx, source, EMBEDDING_WIDTH, CONTROL_TOKENS, source->nb[1] * stride, 0);
    auto * hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, EMBEDDING_WIDTH, PROJECTION_TOKENS);
    const bool supported = ggml_backend_dev_supports_op(device, ggml_mul_mat(ctx, head, hidden));
    ggml_free(ctx);
    require(supported, "the selective projection would fall back from Metal");
}

void test_precision(ggml_type precision, int delay_slot) {
    Fixture fixture;
    write_fixture(fixture, precision, delay_slot);
    const BackendRequest metal{"MTL0", true};
    DelayLM full(fixture.path.string(), metal, THREADS, CONTEXT);
    DelayLM selected(fixture.path.string(), metal, THREADS, CONTEXT);
    require(std::string(full.backend_name()) == "MTL0", "test is not running on Metal");
    require(!full.tables_on_host() && !selected.tables_on_host(), "embedding tables moved to the CPU");
    check_native_projection(precision, delay_slot);
    const std::vector<DelayRow> rows{{1, {0, 1}}, {3, {2, 3}}, {6, {4, 5}}};
    check_slots(full.prefill(rows), selected.prefill(rows, TextLogits::Slots), full.config());
    check_steps(full, selected, rows.back());
    check_no_text(full, selected, rows.back());
    const auto expected = full.step(rows.back());
    const auto actual = selected.step(rows.back(), TextLogits::Full);
    require(expected.text == actual.text && expected.audio == actual.audio,
            "switching back to full projection changed decoder state");
    full.reset();
    selected.reset();
    check_slots(full.prefill(rows), selected.prefill(rows, TextLogits::Slots), full.config());
}

void test_precisions() {
    for (auto precision : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0}) {
        std::printf("precision: %s\n", ggml_type_name(precision));
        test_precision(precision, DELAY_SLOT);
        test_precision(precision, DELAY_SLOT + 1);
        test_precision(precision, GEN_SLOT - 1);
    }
}

} // namespace

int main() {
    try {
        test_precisions();
        std::puts("MOSS Metal selective text projection: OK");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
