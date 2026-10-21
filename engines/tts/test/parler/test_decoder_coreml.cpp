// Standalone native probe. Generate the fixture with
// prepare_decoder_coreml_fixture.py, then compile this and decoder_coreml.mm
// with -std=c++17 -fobjc-arc -framework CoreML -framework Foundation.
// Usage: test_decoder_coreml /outside/repo/readonly_cache.mlmodelc [cpu|ane|all]
#include "../../src/parler/decoder_coreml.h"
#include "../../src/parler/decoder_update_plan.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tts_cpp::parler::detail;

namespace {

void check(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error(message);
}

void near(float a, float b, const std::string & message) {
    check(std::fabs(a - b) <= 0.001f, message + ": got " + std::to_string(a) + ", expected " + std::to_string(b));
}

decoder_coreml_buffer view(void * data, size_t bytes, decoder_coreml_dtype type,
                           std::vector<int64_t> shape, std::vector<int64_t> strides) {
    decoder_coreml_buffer b;
    b.data = data; b.bytes = bytes; b.dtype = type;
    b.shape = std::move(shape); b.strides = std::move(strides);
    return b;
}

void test_layout() {
    std::string error;
    float source[30];
    std::fill_n(source, 30, -99.f);
    // Padded source and destination, and a noncontiguous innermost axis.
    for (int h = 0; h < 2; ++h) for (int d = 0; d < 3; ++d) source[h * 17 + d * 2] = h * 10.f + d + 0.25f;
    auto src = view(source, sizeof(source), decoder_coreml_dtype::float32, {2, 1, 3}, {17, 8, 2});
    uint16_t destination[110];
    std::fill_n(destination, 110, (uint16_t) 0);
    auto dst = view(destination, sizeof(destination), decoder_coreml_dtype::float16, {2, 4, 3}, {60, 13, 3});
    check(decoder_coreml_copy_region(src, dst, {0, 2, 0}, error), error);
    float output[24];
    auto out = view(output, sizeof(output), decoder_coreml_dtype::float32, {2, 4, 3}, {12, 3, 1});
    check(decoder_coreml_copy_region(dst, out, {0, 0, 0}, error), error);
    for (int h = 0; h < 2; ++h) for (int t = 0; t < 4; ++t) for (int d = 0; d < 3; ++d) {
        near(output[(h * 4 + t) * 3 + d], t == 2 ? h * 10.f + d + 0.25f : 0.f, "padded fp16 row roundtrip");
    }
    check(!decoder_coreml_copy_region(src, dst, {0, 4, 0}, error), "out-of-bounds row accepted");
    check(!decoder_coreml_copy_region(src, dst, {0, -1, 0}, error), "negative row accepted");
    check(!decoder_coreml_copy_region(src, dst, {0, 0}, error), "rank mismatch accepted");
    auto bad = dst;
    bad.bytes = 10;
    check(!decoder_coreml_copy_region(src, bad, {0, 0, 0}, error), "short storage accepted");
    bad = dst; bad.strides[0] = std::numeric_limits<int64_t>::max();
    check(!decoder_coreml_copy_region(src, bad, {0, 0, 0}, error), "overflow stride accepted");
    bad = dst; bad.strides[0] = -1;
    check(!decoder_coreml_copy_region(src, bad, {0, 0, 0}, error), "negative stride accepted");
    bad = dst; bad.dtype = decoder_coreml_dtype::int32;
    check(!decoder_coreml_copy_region(src, bad, {0, 0, 0}, error), "float/int32 conversion accepted");
    check(!decoder_coreml_copy_region(src, src, {0, 0, 0}, error), "alias accepted");

    float flat[3] = {1.f, -2.5f, 9.25f}, flat_out[3] = {};
    auto a = view(flat, sizeof(flat), decoder_coreml_dtype::float32, {3}, {1});
    auto b = view(flat_out, sizeof(flat_out), decoder_coreml_dtype::float32, {3}, {1});
    check(decoder_coreml_copy_region(a, b, {0}, error), error);
    for (int i = 0; i < 3; ++i) near(flat_out[i], flat[i], "rank-one copy");

    // Slice a padded first axis without packing; select a grouped destination
    // layer and token row while retaining rank and all actual strides.
    float stacked[100] = {}, grouped[300] = {}, unpacked[48] = {};
    for (int h = 0; h < 2; ++h) for (int d = 0; d < 3; ++d) stacked[53 + h * 17 + d * 2] = h * 3.f + d + 1.f;
    auto selected = view(stacked + 53, sizeof(stacked) - 53 * sizeof(float),
                         decoder_coreml_dtype::float32, {1, 2, 1, 3}, {53, 17, 8, 2});
    auto group = view(grouped, sizeof(grouped), decoder_coreml_dtype::float32,
                      {2, 2, 4, 3}, {150, 60, 13, 3});
    check(decoder_coreml_copy_region(selected, group, {1, 0, 2, 0}, error), error);
    auto unpack = view(unpacked, sizeof(unpacked), decoder_coreml_dtype::float32,
                       {2, 2, 4, 3}, {24, 12, 3, 1});
    check(decoder_coreml_copy_region(group, unpack, {0, 0, 0, 0}, error), error);
    for (int l = 0; l < 2; ++l) for (int h = 0; h < 2; ++h) for (int t = 0; t < 4; ++t) for (int d = 0; d < 3; ++d) {
        near(unpacked[((l * 2 + h) * 4 + t) * 3 + d], l == 1 && t == 2 ? h * 3.f + d + 1.f : 0.f,
             "padded source and grouped destination layer selection");
    }
    // Whole-output updates retain both layer axes and the actual padded
    // source/destination strides, including fp32 -> fp16 conversion.
    for (int l = 0; l < 2; ++l) for (int h = 0; h < 2; ++h) for (int d = 0; d < 3; ++d)
        stacked[l * 53 + h * 17 + d * 2] = l * 12.f + h * 3.f + d + 0.25f;
    uint16_t whole_storage[300] = {};
    auto whole_source = view(stacked, sizeof(stacked), decoder_coreml_dtype::float32,
                             {2, 2, 1, 3}, {53, 17, 8, 2});
    auto whole_destination = view(whole_storage, sizeof(whole_storage), decoder_coreml_dtype::float16,
                                  {2, 2, 4, 3}, {150, 60, 13, 3});
    check(decoder_coreml_copy_region(whole_source, whole_destination, {0, 0, 3, 0}, error), error);
    check(decoder_coreml_copy_region(whole_destination, unpack, {0, 0, 0, 0}, error), error);
    for (int l = 0; l < 2; ++l) for (int h = 0; h < 2; ++h) for (int t = 0; t < 4; ++t) for (int d = 0; d < 3; ++d)
        near(unpacked[((l * 2 + h) * 4 + t) * 3 + d],
             t == 3 ? l * 12.f + h * 3.f + d + 0.25f : 0.f, "whole padded mixed-dtype row copy");
    std::puts("PASS strided layout, fp16 conversions, bounds, overflow and alias checks");
}

void test_fullgroup_session(const std::string & path, decoder_coreml_units units) {
    std::string error;
    auto legacy = decoder_coreml_session::open(path, units, error);
    check(!!legacy, error);
    auto whole = decoder_coreml_session::open(path, units, error);
    check(!!whole, error);
    for (auto * session : {legacy.get(), whole.get()}) {
        check(session->metadata("parler.test.metadata") == "readonly-fullgroup-l24-g24-v1",
              "fresh fullgroup fixture metadata");
        check(session->states().size() == 2, "fullgroup state count");
        for (const auto & state : session->states())
            check(state.shape == std::vector<int64_t>({24, 2, 8, 3}) &&
                  state.dtype == decoder_coreml_dtype::float16, "fullgroup native fp16 geometry");
    }
    std::vector<decoder_coreml_row_update> legacy_rows;
    for (int l = 0; l < 24; ++l) for (int v = 0; v < 2; ++v)
        legacy_rows.push_back({v ? "new_v" : "new_k", v ? "self_v_00" : "self_k_00", 2, 0, l, l});
    auto whole_rows = parler_decoder_update_plan(24, 24);
    const auto compare = [&](bool outputs) {
        for (const auto & state : legacy->states()) {
            std::vector<float> a(state.elements()), b(state.elements());
            check(legacy->read_state(state.name, a.data(), a.size(), error), error);
            check(whole->read_state(state.name, b.data(), b.size(), error), error);
            check(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0,
                  "fullgroup complete state memcmp: " + state.name);
        }
        if (outputs) for (const auto & output : legacy->outputs()) {
            std::vector<float> a(output.elements()), b(output.elements());
            check(legacy->copy_output(output.name, a.data(), a.size(), error), error);
            check(whole->copy_output(output.name, b.data(), b.size(), error), error);
            check(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0,
                  "fullgroup logits/K/V output memcmp: " + output.name);
        }
    };
    for (int repeat = 0; repeat < 2; ++repeat) {
        for (auto * session : {legacy.get(), whole.get()}) {
            check(session->reset(error), error);
            float old_output[6];
            check(!session->copy_output("logits", old_output, 6, error), "fullgroup reset retained output");
            check(!session->predict({}, nullptr, error), "fullgroup reset retained input initialization");
            for (int v = 0; v < 2; ++v) {
                std::vector<float> seed(24 * 2 * 2 * 3);
                for (size_t i = 0; i < seed.size(); ++i)
                    seed[i] = ((int) (i % 17) - 8) * (v ? -0.125f : 0.25f);
                check(session->write_state(v ? "self_v_00" : "self_k_00", {0, 0, 0, 0},
                                          {24, 2, 2, 3}, seed.data(), seed.size(), error), error);
            }
        }
        compare(false);
        for (int32_t row : {2, 3, 7, 0, 6}) {
            const float hidden[4] = {0.5f, 1.f, (float) row / 4, 0.25f};
            const float cross[2] = {0.5f, 1.5f};
            for (auto * session : {legacy.get(), whole.get()}) {
                check(session->set_input("hidden", hidden, 4, error), error);
                check(session->set_input("cross", cross, 2, error), error);
                check(session->set_input_i32("position", &row, 1, error), error);
            }
            for (auto & update : legacy_rows) update.row = row;
            for (auto & update : whole_rows) update.row = row;
            decoder_coreml_timing a, b;
            check(legacy->predict(legacy_rows, &a, error), error);
            check(whole->predict(whole_rows, &b, error), error);
            check(a.row_bytes_written == 24 * 2 * 3 * 2 * 2 && a.row_bytes_written == b.row_bytes_written,
                  "legacy48 and whole2 write identical destination payload bytes");
            compare(true);
        }
    }
    std::puts("PASS actual fullgroup L=G=24 legacy48/whole2 complete state and logits/K/V memcmp, seeded history, positions and reset/repeat (finite fp16 state read as fp32)");
}

void test_session(const std::string & path, decoder_coreml_units units) {
    std::string error;
    check(!decoder_coreml_session::open(path + ".missing", units, error), "invalid model path accepted");
    auto session = decoder_coreml_session::open(path, units, error);
    check(!!session, error);
    check(session->metadata("parler.test.metadata") == "readonly-state-fixture", "creator metadata read");
    check(session->metadata("missing.key").empty(), "missing metadata is empty");
    check(session->inputs().size() == 3 && session->outputs().size() == 3 && session->states().size() == 2, "model introspection");
    std::vector<float> k(96), v(96);
    check(session->read_state("cache_k", k.data(), k.size(), error), error);
    check(std::all_of(k.begin(), k.end(), [](float x) { return x == 0.f; }), "initial state not zero");
    const float hidden[4] = {1, 2, 3, 4}; // sum=10
    const float cross[2] = {0.5f, 1.5f};
    int32_t position = 0;
    const std::vector<decoder_coreml_row_update> rows = {{"new_k", "cache_k", 2, 0}, {"new_v", "cache_v", 2, 0}};
    decoder_coreml_timing timing;
    check(!session->predict(rows, &timing, error), "unset inputs accepted");
    check(!session->set_input("missing", hidden, 4, error), "unknown input accepted");
    check(!session->set_input("hidden", hidden, 3, error), "wrong input size accepted");
    check(!session->set_input("position", hidden, 1, error), "wrong input dtype accepted");
    check(session->set_input("hidden", hidden, 4, error), error);
    check(session->set_input("cross", cross, 2, error), error);
    check(session->set_input_i32("position", &position, 1, error), error);
    auto invalid = rows; invalid[1].row = 8;
    check(!session->predict(invalid, &timing, error), "invalid complete row plan accepted");
    check(session->read_state("cache_k", k.data(), k.size(), error), error);
    check(std::all_of(k.begin(), k.end(), [](float x) { return x == 0.f; }), "bad plan changed first state");

    float logits[6];
    check(!session->copy_output("logits", logits, 6, error), "output before prediction accepted");
    check(session->predict(rows, &timing, error), error);
    check(timing.row_bytes_written == 48, "row write copied more than 2 * 12 fp16 scalars");
    check(timing.total_ms >= timing.predict_ms + timing.row_write_ms, "inclusive timing inconsistent");
    check(session->copy_output("logits", logits, 6, error), error);
    for (int i = 0; i < 6; ++i) near(logits[i], 12.f + i, "zero-history prediction");
    check(session->read_state("cache_k", k.data(), k.size(), error), error);
    check(session->read_state("cache_v", v.data(), v.size(), error), error);
    for (int h = 0; h < 4; ++h) for (int t = 0; t < 8; ++t) for (int d = 0; d < 3; ++d) {
        const int i = (h * 8 + t) * 3 + d, base = h * 3 + d;
        near(k[i], t == 0 ? base + 10.f : 0.f, "only K row zero changed");
        near(v[i], t == 0 ? base - 20.f : 0.f, "only V row zero changed");
    }
    // Reuse hidden/cross input allocations and values; update only position.
    position = 1;
    check(session->set_input_i32("position", &position, 1, error), error);
    auto second = rows; second[0].row = second[1].row = 7;
    check(session->predict(second, &timing, error), error);
    check(session->copy_output("logits", logits, 6, error), error);
    // sum(K)=186, sum(V)=-174; history=186-522=-336.
    for (int i = 0; i < 6; ++i) near(logits[i], -323.f + i, "host-written read-only state visible on second prediction");
    check(session->predict({}, &timing, error), error);
    check(session->copy_output("logits", logits, 6, error), error);
    for (int i = 0; i < 6; ++i) near(logits[i], -659.f + i, "two rows persist across consecutive predictions");

    // Independent sessions must never share synthesis state.
    auto independent = decoder_coreml_session::open(path, units, error);
    check(!!independent, error);
    check(independent->read_state("cache_k", k.data(), k.size(), error), error);
    check(std::all_of(k.begin(), k.end(), [](float x) { return x == 0.f; }), "session states shared");

    check(session->reset(error), error);
    check(!session->copy_output("logits", logits, 6, error), "reset retained old outputs");
    check(!session->predict({}, &timing, error), "reset did not invalidate input initialization");
    check(session->read_state("cache_k", k.data(), k.size(), error), error);
    check(session->read_state("cache_v", v.data(), v.size(), error), error);
    check(std::all_of(k.begin(), k.end(), [](float x) { return x == 0.f; }) &&
          std::all_of(v.begin(), v.end(), [](float x) { return x == 0.f; }), "reset states not zero");

    // Import a packed [L,H,N,HD] prefill into the larger state once.
    std::vector<float> packed(24);
    for (size_t i = 0; i < packed.size(); ++i) packed[i] = (float) i / 4.f;
    check(!session->write_state("cache_k", {0, 0, 7, 0}, {2, 2, 2, 3}, packed.data(), packed.size(), error), "bad seed bounds accepted");
    check(session->write_state("cache_k", {0, 0, 0, 0}, {2, 2, 2, 3}, packed.data(), packed.size(), error), error);
    check(session->read_state("cache_k", k.data(), k.size(), error), error);
    for (int h = 0; h < 4; ++h) for (int t = 0; t < 8; ++t) for (int d = 0; d < 3; ++d) {
        near(k[(h * 8 + t) * 3 + d], t < 2 ? packed[(h * 2 + t) * 3 + d] : 0.f, "packed prefill import");
    }
    check(session->set_input("hidden", hidden, 4, error), error);
    check(session->set_input("cross", cross, 2, error), error);
    check(session->set_input_i32("position", &position, 1, error), error);
    check(session->predict({}, &timing, error), error);
    check(session->copy_output("logits", logits, 6, error), error);
    for (int i = 0; i < 6; ++i) near(logits[i], 82.f + i, "seeded state visible to inference");

    // Tiny inclusive timing only; constant bounded state by overwriting row 0.
    check(session->reset(error), error);
    check(session->set_input("hidden", hidden, 4, error), error);
    check(session->set_input("cross", cross, 2, error), error);
    check(session->set_input_i32("position", &position, 1, error), error);
    for (int i = 0; i < 5; ++i) check(session->predict(rows, &timing, error), error);
    double predict = 0, write = 0, total = 0;
    std::vector<double> times;
    for (int i = 0; i < 100; ++i) {
        check(session->predict(rows, &timing, error), error);
        check(timing.row_bytes_written == 48, "unexpected bytes written");
        predict += timing.predict_ms; write += timing.row_write_ms; total += timing.total_ms;
        times.push_back(timing.total_ms);
    }
    std::sort(times.begin(), times.end());
    std::printf("PASS actual Core ML read-only state, two host rows, input reuse, independent sessions, reset, packed seed\n"
                "100 warm tiny calls: predict_mean_ms=%.6f row_write_mean_ms=%.6f inclusive_mean_ms=%.6f inclusive_p50_ms=%.6f p95_ms=%.6f bytes_per_call=48\n",
                predict / 100, write / 100, total / 100, times[50], times[95]);
}

void test_layered_session(const std::string & path, decoder_coreml_units units) {
    std::string error;
    auto session = decoder_coreml_session::open(path, units, error);
    check(!!session, error);
    check(session->states().size() == 4, "per-layer state introspection");
    const float hidden[4] = {1, 2, 3, 4}, cross[2] = {0.5f, 1.5f};
    int32_t position = 0;
    check(session->set_input("hidden", hidden, 4, error), error);
    check(session->set_input("cross", cross, 2, error), error);
    check(session->set_input_i32("position", &position, 1, error), error);
    const std::vector<decoder_coreml_row_update> rows = {
        {"new_k", "self_k_00", 2, 0, 0}, {"new_k", "self_k_01", 2, 0, 1},
        {"new_v", "self_v_00", 2, 0, 0}, {"new_v", "self_v_01", 2, 0, 1},
    };
    auto invalid = rows;
    invalid.back().source_layer = 2;
    decoder_coreml_timing timing;
    check(!session->predict(invalid, &timing, error), "out-of-bounds source layer accepted");
    invalid.back().source_layer = -2;
    check(!session->predict(invalid, &timing, error), "negative source layer accepted");
    invalid.back().source_layer = -1;
    check(!session->predict(invalid, &timing, error), "full stacked source accepted for per-layer state");
    float state[48];
    check(session->read_state("self_k_00", state, 48, error), error);
    for (float x : state) near(x, 0.f, "invalid layer plan changed state");
    check(session->predict(rows, &timing, error), error);
    check(timing.row_bytes_written == 48, "layer slices copied incorrect byte count");
    for (int layer = 0; layer < 2; ++layer) {
        const std::string suffix = layer == 0 ? "00" : "01";
        for (const std::string kind : {"k", "v"}) {
            check(session->read_state("self_" + kind + "_" + suffix, state, 48, error), error);
            for (int h = 0; h < 2; ++h) for (int t = 0; t < 8; ++t) for (int d = 0; d < 3; ++d) {
                const float expected = (layer * 2 + h) * 3.f + d + (kind == "k" ? 10.f : -20.f);
                near(state[(h * 8 + t) * 3 + d], t == 0 ? expected : 0.f, "source layer copy");
            }
        }
    }
    float new_k[12], logits[6];
    check(session->copy_output("new_k", new_k, 12, error), error);
    for (int i = 0; i < 12; ++i) near(new_k[i], 10.f + i, "stacked output retained after row updates");
    check(session->predict({}, &timing, error), error);
    check(session->copy_output("logits", logits, 6, error), error);
    for (int i = 0; i < 6; ++i) near(logits[i], -324.f + i, "per-layer host state visibility");
    const float seed[6] = {1, 1, 1, 1, 1, 1};
    check(session->write_state("self_k_01", {0, 0, 4, 0}, {1, 2, 1, 3}, seed, 6, error), error);
    check(session->copy_output("new_k", new_k, 12, error), error);
    for (int i = 0; i < 12; ++i) near(new_k[i], 10.f + i, "output retained after explicit host write");
    check(session->predict({}, &timing, error), error);
    check(session->copy_output("logits", logits, 6, error), error);
    for (int i = 0; i < 6; ++i) near(logits[i], -318.f + i, "per-layer explicit host write visible");
    // G=1 accepts an explicit destination layer zero as well as the legacy
    // default -1 used above, with identical row contents and byte count.
    auto explicit_layer = rows;
    for (auto & row : explicit_layer) row.destination_layer = 0;
    check(session->predict(explicit_layer, &timing, error), error);
    check(timing.row_bytes_written == 48, "G1 explicit destination byte count");
    check(session->read_state("self_k_01", state, 48, error), error);
    for (int h = 0; h < 2; ++h) for (int d = 0; d < 3; ++d) {
        near(state[h * 24 + d], 16.f + h * 3 + d, "G1 explicit destination layer row");
    }
    check(session->reset(error), error);
    check(session->read_state("self_k_01", state, 48, error), error);
    for (float x : state) near(x, 0.f, "layered reset");
    check(session->set_input("hidden", hidden, 4, error), error);
    check(session->set_input("cross", cross, 2, error), error);
    check(session->set_input_i32("position", &position, 1, error), error);
    for (int i = 0; i < 5; ++i) check(session->predict(rows, &timing, error), error);
    double total = 0, write = 0;
    for (int i = 0; i < 100; ++i) {
        check(session->predict(rows, &timing, error), error);
        check(timing.row_bytes_written == 48, "incorrect sliced write byte count");
        total += timing.total_ms; write += timing.row_write_ms;
    }
    std::printf("PASS actual per-layer Core ML state, source_layer bounds, direct slices, retained outputs, host writes, reset\n"
                "100 warm per-layer tiny calls: row_write_mean_ms=%.6f inclusive_mean_ms=%.6f bytes_per_call=48\n", write / 100, total / 100);
}

void test_grouped_session(const std::string & path, decoder_coreml_units units) {
    std::string error;
    auto session = decoder_coreml_session::open(path, units, error);
    check(!!session, error);
    check(session->states().size() == 4, "grouped state introspection");
    for (const auto & state : session->states()) check(state.shape == std::vector<int64_t>({2, 2, 8, 3}), "grouped state shape");
    const float hidden[4] = {1, 2, 3, 4}, cross[2] = {0.5f, 1.5f};
    int32_t position = 0;
    auto initialize = [&]() {
        check(session->set_input("hidden", hidden, 4, error), error);
        check(session->set_input("cross", cross, 2, error), error);
        check(session->set_input_i32("position", &position, 1, error), error);
    };
    initialize();
    // Global layers 1 and 2 cross the group boundary; other slots stay zero.
    const std::vector<decoder_coreml_row_update> rows = {
        {"new_k", "group_k_00", 2, 3, 1, 1}, {"new_v", "group_v_00", 2, 3, 1, 1},
        {"new_k", "group_k_01", 2, 7, 2, 0}, {"new_v", "group_v_01", 2, 7, 2, 0},
    };
    decoder_coreml_timing timing;
    float logits[6], output[24], state[96];
    auto assert_zero = [&]() {
        for (const auto & desc : session->states()) {
            check(session->read_state(desc.name, state, 96, error), error);
            for (float x : state) near(x, 0.f, "invalid grouped plan changed state");
        }
    };
    for (int kind = 0; kind < 15; ++kind) {
        auto invalid = rows;
        auto & last = invalid.back(); // Valid earlier entries must not write.
        switch (kind) {
            case 0: last.destination_layer = -2; break;
            case 1: last.destination_layer = 2; break;
            case 2: last.destination_layer = std::numeric_limits<int64_t>::max(); break;
            case 3: last.source_layer = -2; break;
            case 4: last.source_layer = 4; break;
            case 5: last.source_layer = std::numeric_limits<int64_t>::max(); break;
            case 6: last.axis = 0; break;
            case 7: last.axis = std::numeric_limits<size_t>::max(); break;
            case 8: last.row = -1; break;
            case 9: last.row = 8; break;
            case 10: last.row = std::numeric_limits<int64_t>::max(); break;
            case 11: last.source_layer = -1; break;
            case 12: last.destination_layer = -1; break;
            case 13: last.output = "missing"; break;
            case 14: last.state = "missing"; break;
        }
        check(!session->predict(invalid, &timing, error), "invalid grouped plan accepted: " + std::to_string(kind));
        check(!error.empty(), "missing grouped plan error");
        if (kind == 6) check(error.find("conflicts") != std::string::npos, "axis0/layer conflict not diagnosed");
        check(timing.predict_ms == 0 && timing.row_bytes_written == 0, "invalid grouped plan reached inference/write");
        check(!session->copy_output("logits", logits, 6, error), "invalid grouped plan ran inference");
        assert_zero();
    }
    check(session->predict(rows, &timing, error), error);
    check(timing.row_bytes_written == 48, "grouped selected rows byte count");
    check(session->copy_output("logits", logits, 6, error), error);
    for (int i = 0; i < 6; ++i) near(logits[i], 12.f + i, "grouped zero history");
    check(session->copy_output("new_k", output, 24, error), error);
    for (int i = 0; i < 24; ++i) near(output[i], 10.f + i, "grouped writes retained stacked output");
    auto invalid = rows; invalid.back().destination_layer = 2;
    check(!session->predict(invalid, &timing, error), "invalid plan after success accepted");
    check(session->copy_output("logits", logits, 6, error), error);
    for (int i = 0; i < 6; ++i) near(logits[i], 12.f + i, "invalid plan discarded/changed retained output");

    auto verify_rows = [&](bool all_layers) {
        for (int group = 0; group < 2; ++group) for (const std::string kind : {"k", "v"}) {
            const std::string name = "group_" + kind + (group == 0 ? "_00" : "_01");
            check(session->read_state(name, state, 96, error), error);
            for (int slot = 0; slot < 2; ++slot) for (int h = 0; h < 2; ++h) for (int t = 0; t < 8; ++t) for (int d = 0; d < 3; ++d) {
                const int layer = group * 2 + slot;
                const int row = layer == 0 ? 0 : layer == 1 ? 3 : layer == 2 ? 7 : 1;
                const bool written = layer == 1 || layer == 2 || all_layers;
                const float expected = (layer * 2 + h) * 3.f + d + (kind == "k" ? 10.f : -20.f);
                near(state[((slot * 2 + h) * 8 + t) * 3 + d], written && t == row ? expected : 0.f,
                     "grouped selected destination layer/row, untouched neighbors");
            }
        }
    };
    verify_rows(false);
    const std::vector<decoder_coreml_row_update> remaining = {
        {"new_k", "group_k_00", 2, 0, 0, 0}, {"new_v", "group_v_00", 2, 0, 0, 0},
        {"new_k", "group_k_01", 2, 1, 3, 1}, {"new_v", "group_v_01", 2, 1, 3, 1},
    };
    check(session->predict(remaining, &timing, error), error);
    check(timing.row_bytes_written == 48, "second grouped rows byte count");
    check(session->copy_output("logits", logits, 6, error), error);
    // Layers 1/2: sum(K)=258, sum(V)=-102, so history=-48.
    for (int i = 0; i < 6; ++i) near(logits[i], -36.f + i, "future prediction sees first grouped writes");
    verify_rows(true);
    check(session->predict({}, &timing, error), error);
    check(timing.row_bytes_written == 0, "no-update prediction wrote state");
    check(session->copy_output("logits", logits, 6, error), error);
    for (int i = 0; i < 6; ++i) near(logits[i], -84.f + i, "future prediction sees all grouped writes");

    auto independent = decoder_coreml_session::open(path, units, error);
    check(!!independent, error);
    for (const auto & desc : independent->states()) {
        check(independent->read_state(desc.name, state, 96, error), error);
        for (float x : state) near(x, 0.f, "grouped sessions shared state");
    }
    verify_rows(true); // Opening a second session must not affect the first.
    check(session->reset(error), error);
    check(!session->copy_output("logits", logits, 6, error), "grouped reset retained outputs");
    check(!session->predict(rows, &timing, error), "grouped reset retained input initialization");
    assert_zero();
    initialize();
    check(session->predict(rows, &timing, error), error);
    check(session->copy_output("logits", logits, 6, error), error);
    for (int i = 0; i < 6; ++i) near(logits[i], 12.f + i, "grouped reset history");
    verify_rows(false);
    std::puts("PASS actual G2 Core ML state, both layer selectors, complete-plan rejection before inference, retained outputs, future reads, untouched rows/layers, independent sessions and reset");
}

void test_inference_failure(const std::string & path) {
    std::string error;
    auto session = decoder_coreml_session::open(path, decoder_coreml_units::cpu_only, error);
    check(!!session, error);
    const float hidden[4] = {1, 2, 3, 4};
    int32_t index = 0;
    check(session->set_input("hidden", hidden, 4, error), error);
    check(session->set_input_i32("position", &index, 1, error), error);
    check(session->predict({}, nullptr, error), error);
    float logits = 0;
    check(session->copy_output("logits", &logits, 1, error), error);
    near(logits, 1.f, "valid checked gather");
    index = 4; // A real invalid gather in Core ML, not a mocked failure.
    check(session->set_input_i32("position", &index, 1, error), error);
    check(!session->predict({}, nullptr, error), "invalid checked gather did not fail inference");
    check(!error.empty(), "missing inference error description");
    const std::string inference_error = error;
    check(!session->copy_output("logits", &logits, 1, error), "failed inference retained accessible output");
    check(!session->predict({}, nullptr, error) && error.find("requires reset") != std::string::npos, "failed inference did not require reset");
    check(!session->set_input("hidden", hidden, 4, error), "failed session accepted new input");
    check(session->reset(error), error);
    index = 0;
    check(session->set_input("hidden", hidden, 4, error), error);
    check(session->set_input_i32("position", &index, 1, error), error);
    check(session->predict({}, nullptr, error), error);
    check(session->copy_output("logits", &logits, 1, error), error);
    near(logits, 1.f, "reset after inference failure");
    std::printf("PASS real Core ML inference error, explicit failure, poisoned-session gate and reset recovery\n"
                "Expected error: %s\n", inference_error.c_str());
}

} // namespace

int main(int argc, char ** argv) {
    try {
        test_layout();
        if (argc < 2) throw std::runtime_error("usage: test_decoder_coreml model.mlmodelc [cpu|ane|all]");
        const std::string units = argc > 2 ? argv[2] : "cpu";
        check(units == "cpu" || units == "ane" || units == "all", "unknown compute units");
        test_session(argv[1], units == "ane" ? decoder_coreml_units::cpu_and_ane :
                     units == "all" ? decoder_coreml_units::all : decoder_coreml_units::cpu_only);
        const std::string path(argv[1]);
        test_layered_session(path.substr(0, path.find_last_of('/') + 1) + "readonly_layers.mlmodelc",
                             units == "ane" ? decoder_coreml_units::cpu_and_ane :
                             units == "all" ? decoder_coreml_units::all : decoder_coreml_units::cpu_only);
        test_grouped_session(path.substr(0, path.find_last_of('/') + 1) + "readonly_grouped_g2.mlmodelc",
                             units == "ane" ? decoder_coreml_units::cpu_and_ane :
                             units == "all" ? decoder_coreml_units::all : decoder_coreml_units::cpu_only);
        test_fullgroup_session(path.substr(0, path.find_last_of('/') + 1) + "readonly_fullgroup_l24.mlmodelc",
                             units == "ane" ? decoder_coreml_units::cpu_and_ane :
                             units == "all" ? decoder_coreml_units::all : decoder_coreml_units::cpu_only);
        test_inference_failure(path.substr(0, path.find_last_of('/') + 1) + "inference_error.mlmodelc");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
