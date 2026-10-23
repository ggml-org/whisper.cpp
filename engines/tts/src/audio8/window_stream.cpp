#include "audio8/window_stream.h"

#include <algorithm>
#include <chrono>
#include <utility>

namespace tts_cpp {
namespace audio8 {
namespace detail {

namespace {

double elapsed_ms(std::chrono::steady_clock::time_point since) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - since)
        .count();
}

size_t columns(int frames, int width) {
    return static_cast<size_t>(frames) * static_cast<size_t>(width);
}

}  // namespace

coreml_window leading_coreml_window(int index, int window, int context) {
    coreml_window span;
    span.begin = index * (window - context);
    span.filled = window;
    span.core_begin = index == 0 ? 0 : span.begin + context;
    span.core_end = span.begin + window;
    return span;
}

window_stream::window_stream(const window_stream_shape & shape, window_stream_hooks hooks)
    : shape_(shape), hooks_(std::move(hooks)) {
    if (shape_.asynchronous) worker_ = std::thread([this] { run_worker(); });
}

window_stream::~window_stream() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_.clear();
    }
    close();
    if (worker_.joinable()) worker_.join();
}

bool window_stream::advance(int n_frames, std::string * error) {
    rethrow_callback_error();
    while (leading_coreml_window(queued_, shape_.window, shape_.context).core_end < n_frames) {
        if (!queue_leading(queued_, error)) return false;
        rethrow_callback_error();
    }
    return true;
}

window_stream_status window_stream::finish(int n_frames, std::vector<float> & pcm,
                                           std::string * error) {
    const std::vector<coreml_window> plan =
        plan_coreml_windows(n_frames, shape_.window, shape_.context);
    const bool planned = !plan.empty() && static_cast<size_t>(queued_) < plan.size();
    const bool queued = planned && queue_remaining(plan, error);
    close();
    const window_stream_status status = wait_for_worker();
    if (!planned) return window_stream_status::unavailable;
    if (!queued) return window_stream_status::failed;
    if (status == window_stream_status::done) stitch(n_frames, pcm);
    return status;
}

bool window_stream::queue_leading(int index, std::string * error) {
    const coreml_window span = leading_coreml_window(index, shape_.window, shape_.context);
    if (!compute_post(span.begin + span.filled, error)) return false;
    queue_span(span);
    ++streamed_;
    return true;
}

bool window_stream::queue_remaining(const std::vector<coreml_window> & plan,
                                    std::string * error) {
    if (!compute_post(plan.back().core_end, error)) return false;
    for (size_t index = static_cast<size_t>(queued_); index < plan.size(); ++index) {
        queue_span(plan[index]);
    }
    return true;
}

bool window_stream::compute_post(int end, std::string * error) {
    rethrow_callback_error();
    const auto started = std::chrono::steady_clock::now();
    const bool ok = hooks_.extend_post(end, post_, error);
    post_ms_ += elapsed_ms(started);
    rethrow_callback_error();
    if (!ok && error && error->empty()) *error = "audio8: the streamed post pass failed";
    return ok;
}

void window_stream::queue_span(const coreml_window & span) {
    job work;
    work.index = queued_++;
    work.span = span;
    work.input.assign(columns(shape_.window, shape_.latent_dim), 0.0f);
    const auto from =
        post_.begin() + static_cast<std::ptrdiff_t>(columns(span.begin, shape_.latent_dim));
    std::copy(from, from + static_cast<std::ptrdiff_t>(columns(span.filled, shape_.latent_dim)),
              work.input.begin());
    push(std::move(work));
}

void window_stream::push(job work) {
    if (!shape_.asynchronous) {
        std::vector<float> out(columns(shape_.window, shape_.frame_size));
        if (!failed_ && !cancelled_) process_job(work, out);
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_.push_back(std::move(work));
    }
    changed_.notify_all();
}

bool window_stream::pop(job & work) {
    std::unique_lock<std::mutex> lock(mutex_);
    changed_.wait(lock, [this] { return !pending_.empty() || closing_; });
    if (pending_.empty()) return false;
    work = std::move(pending_.front());
    pending_.pop_front();
    return true;
}

void window_stream::run_worker() {
    std::vector<float> out(columns(shape_.window, shape_.frame_size));
    job work;
    while (pop(work)) {
        if (!process_job(work, out)) return;
    }
}

bool window_stream::process_job(const job & work, std::vector<float> & out) {
    if (hooks_.cancelled && hooks_.cancelled()) {
        cancelled_ = true;
        return false;
    }
    const auto started = std::chrono::steady_clock::now();
    const bool synthesized = run_job(work, out);
    synthesis_ms_ += elapsed_ms(started);
    if (!synthesized) {
        failed_ = true;
        return false;
    }
    try {
        keep_core(work, out);
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            callback_error_ = std::current_exception();
        }
        failed_ = true;
        return false;
    }
    return true;
}

bool window_stream::run_job(const job & work, std::vector<float> & out) {
    try {
        return hooks_.synthesize(work.input.data(), out.data());
    } catch (...) {
        return false;
    }
}

void window_stream::keep_core(const job & work, const std::vector<float> & out) {
    const size_t skip = columns(work.span.core_begin - work.span.begin, shape_.frame_size);
    const size_t keep = columns(work.span.core_end - work.span.core_begin, shape_.frame_size);
    std::vector<float> core(out.begin() + static_cast<std::ptrdiff_t>(skip),
                            out.begin() + static_cast<std::ptrdiff_t>(skip + keep));
    if (hooks_.cancelled && hooks_.cancelled()) {
        cancelled_ = true;
        return;
    }
    if (hooks_.emit) {
        emitted_ = true;
        hooks_.emit(core);
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (cores_.size() <= static_cast<size_t>(work.index)) {
        cores_.resize(static_cast<size_t>(work.index) + 1);
    }
    cores_[static_cast<size_t>(work.index)] = std::move(core);
    ++finished_;
}

void window_stream::rethrow_callback_error() {
    std::exception_ptr error;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        error = callback_error_;
    }
    if (error) std::rethrow_exception(error);
}

void window_stream::close() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        closing_ = true;
    }
    changed_.notify_all();
}

window_stream_status window_stream::wait_for_worker() {
    if (worker_.joinable()) worker_.join();
    rethrow_callback_error();
    if (cancelled_) return window_stream_status::cancelled;
    if (failed_ || finished_ != queued_) return window_stream_status::unavailable;
    return window_stream_status::done;
}

void window_stream::stitch(int n_frames, std::vector<float> & pcm) const {
    pcm.clear();
    pcm.reserve(columns(n_frames, shape_.frame_size));
    for (const std::vector<float> & core : cores_) {
        pcm.insert(pcm.end(), core.begin(), core.end());
    }
}

}  // namespace detail
}  // namespace audio8
}  // namespace tts_cpp
