// TDD harness for Phase 2D — `SUPERTONIC_PROFILE_CSV` machine-
// readable timing emitter.
//
// Background:
//   Each Supertonic stage already emits human-readable profile
//   timing to stderr when its per-stage env var is set
//   (`SUPERTONIC_VECTOR_PROFILE`, `SUPERTONIC_VOCODER_PROFILE`,
//   `SUPERTONIC_TEXT_PROFILE`).  Those are great for eyeballing
//   what just happened on a single run but useless for the next
//   optimization round — we need a stable schema that a small
//   Python script can ingest, group by (stage, island), and
//   surface as "top 10 hot spots by p95 latency" over a 100-synth
//   benchmark.  This finding adds `SUPERTONIC_PROFILE_CSV=PATH`
//   that hooks into the same call sites and emits one row per
//   `supertonic_graph_compute` invocation.
//
// Schema (one header row, then one data row per compute call):
//
//   stage,island,step,wall_ms,unix_us
//   vector,attn0_flash,0,1.234,1715517000123456
//   vector,style0_residual,0,0.412,1715517000125678
//   ...
//
// The unit harness here verifies the writer mechanics without
// requiring a model load.  It:
//
//   1. Points `SUPERTONIC_PROFILE_CSV` at a temp file.
//   2. Calls `supertonic_profile_csv_record(...)` for a handful
//      of synthetic rows.
//   3. Calls `supertonic_profile_csv_flush()` to force the
//      buffered writes to disk.
//   4. Reopens the file and parses each row.
//   5. Asserts the header is correct, the row count + ordering
//      matches what was recorded, and the per-field types are
//      well-formed (numeric where they should be).
//
// Registered with `LABEL "unit"` in CMakeLists.txt — no GGUF
// required.

#include "supertonic_internal.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace tts_cpp::supertonic::detail;

namespace {

int g_failures = 0;
int g_checks   = 0;

#define CHECK(cond) do {                                              \
    ++g_checks;                                                       \
    if (!(cond)) {                                                    \
        ++g_failures;                                                 \
        std::fprintf(stderr, "FAIL %s:%d  %s\n",                     \
                     __FILE__, __LINE__, #cond);                      \
    }                                                                 \
} while (0)

// Split a CSV row on commas.  Pragmatic, doesn't handle quoting —
// the emitter's schema doesn't use commas in any field.
std::vector<std::string> split_csv(const std::string & line) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : line) {
        if (c == ',') {
            out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    out.push_back(cur);
    return out;
}

bool is_numeric(const std::string & s) {
    if (s.empty()) return false;
    bool seen_digit = false;
    bool seen_dot = false;
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (c == '-' && i == 0) continue;
        if (c >= '0' && c <= '9') { seen_digit = true; continue; }
        if (c == '.' && !seen_dot) { seen_dot = true; continue; }
        return false;
    }
    return seen_digit;
}

std::vector<std::string> read_lines(const std::string & path) {
    std::vector<std::string> out;
    std::ifstream f(path);
    if (!f.good()) return out;
    std::string line;
    while (std::getline(f, line)) out.push_back(line);
    return out;
}

// Test 1 — Disabled by default.
//
// With `SUPERTONIC_PROFILE_CSV` unset, recording must be a no-op:
// any subsequent `record` call returns without touching disk, and
// `flush` is similarly inert.  Otherwise the env-gated overhead
// would land in every production synth.
void test_disabled_by_default() {
    std::fprintf(stderr, "[Phase 2D disabled-by-default]\n");
    // Make absolutely sure the env var isn't set from the parent
    // shell (CI hygiene).
#if defined(_WIN32)
    _putenv_s("SUPERTONIC_PROFILE_CSV", "");
#else
    unsetenv("SUPERTONIC_PROFILE_CSV");
#endif
    // No env var, no path-set.  Recording is a no-op.
    supertonic_profile_csv_record("vector", "attn0_flash", /*step=*/0, /*wall_ms=*/1.0);
    supertonic_profile_csv_flush();
    CHECK(!supertonic_profile_csv_enabled());
}

// Test 2 — End-to-end round-trip via the explicit path override.
//
// Pointing the emitter at a temp file (via the test-only
// `_set_path` helper that bypasses the env-var probe) records a
// few rows, flushes, then re-reads the file to verify the
// schema + values.  Avoids touching the parent process env state
// to keep the test thread-safe against other unit tests.
void test_csv_round_trip() {
    std::fprintf(stderr, "[Phase 2D CSV round-trip]\n");

    // Allocate a fresh path inside the build dir so multiple
    // concurrent ctest runs don't collide.  Using `/tmp` directly
    // also works on Linux + macOS; on Windows the test would need
    // GetTempPathA, but our CI matrix runs the unit label on
    // Linux + macOS where /tmp exists.
    char path_buf[L_tmpnam];
    if (!std::tmpnam(path_buf)) {
        std::fprintf(stderr, "  SKIP: tmpnam failed\n");
        return;
    }
    const std::string path = path_buf;
    supertonic_profile_csv_set_path(path.c_str());
    CHECK(supertonic_profile_csv_enabled());

    // Record a few rows that exercise the schema:
    //   - vector stage with a step != 0.
    //   - vocoder stage with step = 0.
    //   - text stage with negative step (sentinel for "not a
    //     denoise step" — emitter should still accept and emit).
    supertonic_profile_csv_record("vector",  "attn0_flash",       0,  1.234);
    supertonic_profile_csv_record("vector",  "style0_residual",   0,  0.412);
    supertonic_profile_csv_record("vector",  "attn0_flash",       1,  1.198);
    supertonic_profile_csv_record("vocoder", "compute",           0, 42.0);
    supertonic_profile_csv_record("text",    "convnext_front",   -1,  6.7);
    supertonic_profile_csv_flush();

    // Read it back.
    auto lines = read_lines(path);
    CHECK(lines.size() == 6); // header + 5 data rows

    if (lines.size() >= 1) {
        // Header row.  Exact order matters because the analysis
        // script keys columns by position, not name.
        const std::string expected_header = "stage,island,step,wall_ms,unix_us";
        CHECK(lines[0] == expected_header);
    }

    if (lines.size() >= 6) {
        // Per-row checks.
        struct Expected {
            const char * stage;
            const char * island;
            int          step;
            double       wall_ms;
        };
        const Expected expected[] = {
            { "vector",  "attn0_flash",      0,  1.234 },
            { "vector",  "style0_residual",  0,  0.412 },
            { "vector",  "attn0_flash",      1,  1.198 },
            { "vocoder", "compute",          0, 42.0   },
            { "text",    "convnext_front",  -1,  6.7   },
        };
        for (int i = 0; i < 5; ++i) {
            auto cols = split_csv(lines[i + 1]);
            CHECK(cols.size() == 5);
            if (cols.size() != 5) continue;

            CHECK(cols[0] == expected[i].stage);
            CHECK(cols[1] == expected[i].island);
            CHECK(std::atoi(cols[2].c_str()) == expected[i].step);

            // wall_ms is a double; tolerate the emitter's print
            // formatting (e.g. "%.3f" rounding).  Use parse +
            // numeric tolerance instead of string match.
            CHECK(is_numeric(cols[3]));
            const double parsed = std::atof(cols[3].c_str());
            const double err    = std::abs(parsed - expected[i].wall_ms);
            CHECK(err <= 0.01); // 10 µs slack for "%.3f"-style formatting

            // unix_us is opaque to us — emitter records the wall
            // clock at record time — but must be numeric and
            // non-negative.
            CHECK(is_numeric(cols[4]));
            const long long us = std::atoll(cols[4].c_str());
            CHECK(us >= 0);
        }
    }

    // Disable + clean up.
    supertonic_profile_csv_set_path(nullptr);
    CHECK(!supertonic_profile_csv_enabled());
    std::remove(path.c_str());
}

// Test 3 — Multiple records appended, not overwritten.
//
// Re-enabling the same path and recording more rows must append
// to the existing file (not truncate it).  This matches the
// expected pattern: a bench harness runs many synths with the
// env var set, and the CSV accumulates one row per
// `supertonic_graph_compute` call across the whole run.
void test_append_semantics() {
    std::fprintf(stderr, "[Phase 2D append semantics]\n");
    char path_buf[L_tmpnam];
    if (!std::tmpnam(path_buf)) { std::fprintf(stderr, "  SKIP\n"); return; }
    const std::string path = path_buf;

    supertonic_profile_csv_set_path(path.c_str());
    supertonic_profile_csv_record("vector", "x", 0, 1.0);
    supertonic_profile_csv_flush();
    supertonic_profile_csv_set_path(nullptr); // close

    supertonic_profile_csv_set_path(path.c_str()); // reopen
    supertonic_profile_csv_record("vector", "x", 1, 2.0);
    supertonic_profile_csv_flush();
    supertonic_profile_csv_set_path(nullptr);

    auto lines = read_lines(path);
    // One header + two data rows.  Re-opening must NOT re-write
    // the header (or the analysis script will trip on it).
    CHECK(lines.size() == 3);
    if (lines.size() >= 3) {
        CHECK(lines[0] == "stage,island,step,wall_ms,unix_us");
        CHECK(split_csv(lines[1])[2] == "0");
        CHECK(split_csv(lines[2])[2] == "1");
    }
    std::remove(path.c_str());
}

} // namespace

int main() {
    test_disabled_by_default();
    test_csv_round_trip();
    test_append_semantics();

    std::fprintf(stderr,
                 "test_supertonic_profile_csv: %d / %d checks passed\n",
                 g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
