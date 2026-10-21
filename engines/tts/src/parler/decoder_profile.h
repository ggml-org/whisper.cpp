#pragma once

#include <algorithm>
#include <cstddef>
#include <ostream>
#include <utility>
#include <vector>

namespace tts_cpp { namespace parler { namespace detail {

// Internal, opt-in timings in milliseconds. A runtime step publishes valid
// only after native prediction, row writes, output geometry and finite logits
// all succeed. Failed attempts, replay, ggml, prefill and seeding are excluded.
struct parler_decoder_step_profile {
    bool valid = false;
    // Set by the native sidecar; runtime alone promotes this to valid after
    // checking the returned logits. Unknown step implementations leave false.
    bool native_timing_available = false;
    double embedding_ms = 0;
    // Sidecar position validation, mask/update-plan construction and set_input.
    double input_prepare_ms = 0;
    double predict_ms = 0;
    // Output retrieval/validation and state handlers, including the row copies.
    double row_write_ms = 0;
    double session_total_ms = 0;
    double logit_readback_ms = 0;
    double step_wall_ms = 0;
    // Destination scalar bytes, using the native state's dtype; no padding.
    size_t row_bytes_written = 0;

    // predict and row_write are disjoint children of session_total. The
    // session remainder covers validation/provider/bookkeeping overhead.
    double session_other_ms() const { return session_total_ms - predict_ms - row_write_ms; }
    // step_wall encloses runtime validation/history, embedding, preparation,
    // session_total, readback and finite-logit validation/bookkeeping.
    double step_other_ms() const {
        return step_wall_ms - embedding_ms - input_prepare_ms - session_total_ms - logit_readback_ms;
    }
};

// Count every requested measured step, but accumulate timings only for valid
// native steps. Merge each run only when the pipeline accepts that same run.
// Profile totals overlap the inclusive pipeline stages.
struct parler_decoder_profile_summary {
    size_t requested_steps = 0;
    size_t successful_steps = 0;
    parler_decoder_step_profile total;
    std::vector<double> step_wall_samples;

    double step_wall_percentile(double fraction) const {
        if (step_wall_samples.empty()) return 0;
        auto values = step_wall_samples;
        std::sort(values.begin(), values.end());
        const double index = fraction * (values.size() - 1);
        const size_t lo = (size_t) index, hi = std::min(lo + 1, values.size() - 1);
        return values[lo] + (values[hi] - values[lo]) * (index - lo);
    }

    void add(const parler_decoder_step_profile & p, bool measured) {
        if (!measured) return;
        ++requested_steps;
        if (!p.valid) return;
        ++successful_steps;
        step_wall_samples.push_back(p.step_wall_ms);
        accumulate(p);
    }

    size_t unprofiled_steps() const { return requested_steps - successful_steps; }

    void merge(parler_decoder_profile_summary && run) {
        requested_steps += run.requested_steps;
        successful_steps += run.successful_steps;
        accumulate(run.total);
        if (step_wall_samples.empty()) step_wall_samples = std::move(run.step_wall_samples);
        else step_wall_samples.insert(step_wall_samples.end(),
                                      run.step_wall_samples.begin(), run.step_wall_samples.end());
        run = {};
    }

    void accumulate(const parler_decoder_step_profile & p) {
        total.embedding_ms += p.embedding_ms;
        total.input_prepare_ms += p.input_prepare_ms;
        total.predict_ms += p.predict_ms;
        total.row_write_ms += p.row_write_ms;
        total.session_total_ms += p.session_total_ms;
        total.logit_readback_ms += p.logit_readback_ms;
        total.step_wall_ms += p.step_wall_ms;
        total.row_bytes_written += p.row_bytes_written;
    }

    void write_json(std::ostream & os) const {
        struct format_guard {
            std::ostream & stream;
            std::ios_base::fmtflags flags;
            std::streamsize precision;
            ~format_guard() { stream.flags(flags); stream.precision(precision); }
        } restore{os, os.flags(), os.precision()};
        os.flags(std::ios_base::dec);
        os.precision(17);
        os << "{\"requested_steps\": " << requested_steps
           << ", \"successful_steps\": " << successful_steps
           << ", \"unprofiled_steps\": " << unprofiled_steps()
           << ", \"row_bytes_written\": " << total.row_bytes_written
           << ", \"timing_scope\": \"coverage counts all requested steps in accepted measured runs; timings include successful CoreML steps only; excludes warmup, prefill, seed, failures and ggml replay\""
           << ", \"input_prepare_scope\": \"sidecar position validation, mask/update-plan construction and input conversion/copy\""
           << ", \"row_write_scope\": \"output retrieval/validation and state handlers including row copies\""
           << ", \"synchronization_note\": \"deferred Core ML synchronization may shift work between predict, row writes and readback; compare these intervals with session total and step wall\""
           << ", \"nesting\": \"session_total_ms contains predict_ms, row_write_ms, session_other_ms; step_wall_ms contains embedding_ms, input_prepare_ms, session_total_ms, logit_readback_ms, step_other_ms; totals overlap pipeline stages\""
           << ", \"totals_ms\": ";
        if (!successful_steps) { os << "null}"; return; }
        os << "{\"embedding_ms\": " << total.embedding_ms
           << ", \"input_prepare_ms\": " << total.input_prepare_ms
           << ", \"predict_ms\": " << total.predict_ms
           << ", \"row_write_ms\": " << total.row_write_ms
           << ", \"session_other_ms\": " << total.session_other_ms()
           << ", \"session_total_ms\": " << total.session_total_ms
           << ", \"logit_readback_ms\": " << total.logit_readback_ms
           << ", \"step_other_ms\": " << total.step_other_ms()
           << ", \"step_wall_ms\": " << total.step_wall_ms << "}"
           << ", \"step_wall_p50_ms\": " << step_wall_percentile(0.5)
           << ", \"step_wall_p95_ms\": " << step_wall_percentile(0.95) << "}";
    }

    void write_text(std::ostream & os) const {
        os << "\n  Core ML decoder profile: requested measured steps=" << requested_steps
           << ", successful measured steps=" << successful_steps
           << ", unprofiled steps=" << unprofiled_steps()
           << ", row bytes written=" << total.row_bytes_written << "\n";
        if (!successful_steps) {
            os << "    No successful measured Core ML steps; timings unavailable.\n";
            return;
        }
        const auto line = [&](const char * name, double ms) {
            os << "    " << name << ": total=" << ms << " ms, mean=" << ms / successful_steps << " ms/step\n";
        };
        line("embedding", total.embedding_ms);
        line("input/update preparation (position validation, mask/plan construction, input copy)", total.input_prepare_ms);
        line("predict", total.predict_ms);
        line("cache row writes (output retrieval/validation, state handlers and copies)", total.row_write_ms);
        line("session other", total.session_other_ms());
        line("session total (inclusive)", total.session_total_ms);
        line("logit readback", total.logit_readback_ms);
        line("step other", total.step_other_ms());
        line("step wall (inclusive)", total.step_wall_ms);
        os << "    step wall: p50=" << step_wall_percentile(0.5)
           << " ms, p95=" << step_wall_percentile(0.95) << " ms\n";
        os << "    Session total = predict + row writes + session other.\n"
              "    Step wall = embedding + preparation + session total + readback + step other.\n"
              "    Excludes warmup, prefill, seed, failures and ggml replay; overlaps pipeline stages.\n"
              "    Deferred Core ML synchronization may shift work between predict, row writes and readback; compare these intervals with session total and step wall.\n";
    }
};

}}} // namespace tts_cpp::parler::detail
