#include "parler/decoder_update_plan.h"

#include <cstdio>

using namespace tts_cpp::parler::detail;

namespace {
int failures = 0;

void check_plan(int layers, int per_state, const std::vector<decoder_coreml_row_update> & expected) {
    const auto actual = parler_decoder_update_plan(layers, per_state);
    bool ok = actual.size() == expected.size();
    if (ok) for (size_t i = 0; i < expected.size(); ++i) {
        const auto & a = actual[i];
        const auto & e = expected[i];
        if (a.output != e.output || a.state != e.state || a.source_layer != e.source_layer ||
            a.destination_layer != e.destination_layer || a.axis != e.axis || a.row != e.row) {
            fprintf(stderr, "FAIL L=%d G=%d update %zu: %s -> %s, source=%lld destination=%lld axis=%zu row=%lld\n",
                    layers, per_state, i, a.output.c_str(), a.state.c_str(),
                    (long long) a.source_layer, (long long) a.destination_layer, a.axis, (long long) a.row);
            ok = false;
        }
    }
    fprintf(stderr, "[%s] constructor plan L=%d G=%d: count=%zu expected=%zu and all update fields\n",
            ok ? "PASS" : "FAIL", layers, per_state, actual.size(), expected.size());
    if (!ok) ++failures;
}
} // namespace

int main() {
    check_plan(24, 24, {
        {"new_k", "self_k_00", 2, 0, -1, -1}, {"new_v", "self_v_00", 2, 0, -1, -1}});
    check_plan(4, 2, {
        {"new_k", "self_k_00", 2, 0, 0, 0}, {"new_v", "self_v_00", 2, 0, 0, 0},
        {"new_k", "self_k_00", 2, 0, 1, 1}, {"new_v", "self_v_00", 2, 0, 1, 1},
        {"new_k", "self_k_01", 2, 0, 2, 0}, {"new_v", "self_v_01", 2, 0, 2, 0},
        {"new_k", "self_k_01", 2, 0, 3, 1}, {"new_v", "self_v_01", 2, 0, 3, 1}});
    check_plan(3, 1, {
        {"new_k", "self_k_00", 2, 0, 0, 0}, {"new_v", "self_v_00", 2, 0, 0, 0},
        {"new_k", "self_k_01", 2, 0, 1, 0}, {"new_v", "self_v_01", 2, 0, 1, 0},
        {"new_k", "self_k_02", 2, 0, 2, 0}, {"new_v", "self_v_02", 2, 0, 2, 0}});
    // Both conditions hold here; the full-group branch must take precedence.
    check_plan(1, 1, {
        {"new_k", "self_k_00", 2, 0, -1, -1}, {"new_v", "self_v_00", 2, 0, -1, -1}});
    return failures ? 1 : 0;
}
