#include "parler/decoder_profile.h"
#include "json.hpp"

#include <cstdio>
#include <cmath>
#include <iomanip>
#include <sstream>

using namespace tts_cpp::parler::detail;

int main() {
    int failures = 0;
    const auto check = [&](bool ok, const char * message) {
        fprintf(stderr, "[%s] %s\n", ok ? "PASS" : "FAIL", message);
        if (!ok) ++failures;
    };
    const auto near = [](double actual, double expected) {
        return std::isfinite(actual) &&
               std::fabs(actual - expected) <= 1e-12 * std::max(1.0, std::fabs(expected));
    };
    parler_decoder_profile_summary summary;
    parler_decoder_step_profile p;
    p.valid = true;
    p.embedding_ms = 1;
    p.input_prepare_ms = 2;
    p.predict_ms = 3;
    p.row_write_ms = 4;
    p.session_total_ms = 8;
    p.logit_readback_ms = 5;
    p.step_wall_ms = 20;
    p.row_bytes_written = 128;
    summary.add(p, false);
    check(summary.requested_steps == 0 && summary.successful_steps == 0,
          "warmup contributes neither requested steps nor successful native steps");
    p.valid = false;
    summary.add(p, true);
    std::ostringstream empty_json, empty_text;
    empty_json << std::scientific << std::setprecision(3);
    const auto empty_flags = empty_json.flags();
    summary.write_json(empty_json);
    summary.write_text(empty_text);
    auto report = nlohmann::json::parse(empty_json.str());
    check(report.at("requested_steps") == 1 && report.at("successful_steps") == 0 &&
          report.at("unprofiled_steps") == 1 && report.at("totals_ms").is_null() &&
          report.at("row_bytes_written") == 0 && empty_text.str().find("unavailable") != std::string::npos,
          "invalid/ggml steps count toward coverage but produce unavailable native timings");
    check(empty_json.flags() == empty_flags && empty_json.precision() == 3,
          "empty-profile JSON restores caller formatting on the early-return path");
    parler_decoder_profile_summary run;
    p.valid = true;
    run.add(p, true);
    p.step_wall_ms = 40;
    run.add(p, true);
    p.valid = false;
    run.add(p, true);
    check(summary.requested_steps == 1 && summary.successful_steps == 0 && summary.step_wall_samples.empty(),
          "per-run measurements stay buffered until the accepted-run merge");
    summary.merge(std::move(run));
    check(run.requested_steps == 0 && run.successful_steps == 0 && run.step_wall_samples.empty() &&
          summary.step_wall_samples.size() == 2,
          "accepted-run merge transfers counts, totals and samples without retaining duplicates");
    std::ostringstream json, text;
    summary.write_json(json);
    summary.write_text(text);
    report = nlohmann::json::parse(json.str());
    const auto & totals = report.at("totals_ms");
    check(report.at("requested_steps") == 4 && report.at("successful_steps") == 2 &&
          report.at("unprofiled_steps") == 2 && report.at("row_bytes_written") == 256 &&
          near(totals.at("embedding_ms"), 2) && near(totals.at("input_prepare_ms"), 4) &&
          near(totals.at("predict_ms"), 6) && near(totals.at("row_write_ms"), 8) &&
          near(totals.at("logit_readback_ms"), 10),
          "only successful measured steps contribute timing totals and destination bytes");
    check(near(totals.at("session_other_ms"), 2) && near(totals.at("session_total_ms"), 16) &&
          near(totals.at("step_other_ms"), 28) && near(totals.at("step_wall_ms"), 60),
          "nested totals and disjoint remainders reconcile without double counting");
    check(text.str().find("mean=30 ms/step") != std::string::npos &&
          near(report.at("step_wall_p50_ms"), 30) && near(report.at("step_wall_p95_ms"), 39) &&
          report.at("nesting").get<std::string>().find("overlap") != std::string::npos,
          "text reports per-step means and JSON documents inclusive timing scope");
    // A second accepted run exercises appending samples to a nonempty summary.
    p.valid = true;
    p.embedding_ms = 1.2345678901234567;
    p.step_wall_ms = 60.125;
    run.add(p, true);
    summary.merge(std::move(run));
    check(summary.requested_steps == 5 && summary.successful_steps == 3 &&
          summary.unprofiled_steps() == 2 && summary.step_wall_samples.size() == 3 &&
          near(summary.step_wall_percentile(0.5), 40) &&
          near(summary.step_wall_percentile(0.95), 58.1125),
          "multiple accepted runs preserve coverage and interpolated per-step percentiles");
    parler_decoder_profile_summary precision_summary;
    precision_summary.add(p, true);
    std::ostringstream precise_json;
    precise_json << std::hex << std::scientific << std::showpos << std::setprecision(3);
    const auto flags = precise_json.flags();
    precision_summary.write_json(precise_json);
    report = nlohmann::json::parse(precise_json.str());
    check(std::fabs(report.at("totals_ms").at("embedding_ms").get<double>() - p.embedding_ms) <= 1e-15 &&
          precise_json.flags() == flags && precise_json.precision() == 3,
          "JSON uses local 17-digit precision and decimal formatting then restores the caller");
    return failures ? 1 : 0;
}
