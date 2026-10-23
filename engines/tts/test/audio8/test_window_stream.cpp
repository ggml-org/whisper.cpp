// Model-free: the synthesis stream (window_stream.h) with a fake causal
// synthesizer standing in for the Core ML sidecar.

#include "audio8/coreml_windows.h"
#include "audio8/window_stream.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <stdexcept>
#include <thread>
#include <vector>

using tts_cpp::audio8::detail::coreml_window;
using tts_cpp::audio8::detail::leading_coreml_window;
using tts_cpp::audio8::detail::plan_coreml_windows;
using tts_cpp::audio8::detail::window_stream;
using tts_cpp::audio8::detail::window_stream_hooks;
using tts_cpp::audio8::detail::window_stream_shape;
using tts_cpp::audio8::detail::window_stream_status;

namespace {

constexpr int LATENT = 2;
constexpr int SAMPLES = 3;
constexpr float SAMPLE_STEP = 0.25f;
constexpr float SECOND_CHANNEL_WEIGHT = 0.5f;
constexpr auto WORKER_TIMEOUT = std::chrono::seconds(10);
constexpr auto POLL = std::chrono::milliseconds(1);

int g_failures = 0;

void fail(const std::string & what) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
}

void expect(bool ok, const std::string & what) {
    if (!ok) fail(what);
}

std::string tag(int n_frames, int window, int context) {
    return "n=" + std::to_string(n_frames) + " window=" + std::to_string(window) +
           " context=" + std::to_string(context);
}

float post_value(int frame, int channel) {
    return static_cast<float>(frame * 7 + channel * 3 + 1);
}

float frame_weight(int frame) {
    return post_value(frame, 0) + SECOND_CHANNEL_WEIGHT * post_value(frame, 1);
}

struct fake_codec {
    window_stream_shape shape;
    std::thread::id producer = std::this_thread::get_id();
    std::atomic<int> synthesized{0};
    std::atomic<bool> on_producer{false};
    int fail_at = -1;
    bool cancel = false;
    bool fail_post = false;
    int computed = 0;
    bool went_back = false;
};

void append_column(int frame, std::vector<float> & post) {
    for (int channel = 0; channel < LATENT; ++channel) post.push_back(post_value(frame, channel));
}

// Appends the columns the stream does not hold yet, and counts them, so a
// stream that asked for a frame twice would show it.
bool fake_post(fake_codec & codec, int end, std::vector<float> & post, std::string * error) {
    if (codec.fail_post) {
        if (error) *error = "fake post failure";
        return false;
    }
    const int held = static_cast<int>(post.size()) / LATENT;
    if (end < held) codec.went_back = true;
    for (int frame = held; frame < end; ++frame) append_column(frame, post);
    codec.computed += std::max(0, end - held);
    return true;
}

float window_weight(const float * post, int column) {
    return post[column * LATENT] + SECOND_CHANNEL_WEIGHT * post[column * LATENT + 1];
}

// Frame t of a window sums the weights of its own column and the `context`
// before it, clipped at the window's start: causal with exactly that reach.
float causal_sum(const float * post, int frame, int context) {
    float total = 0.0f;
    for (int column = std::max(0, frame - context); column <= frame; ++column) {
        total += window_weight(post, column);
    }
    return total;
}

void write_frame(float * pcm, int frame, float value) {
    for (int sample = 0; sample < SAMPLES; ++sample) {
        pcm[frame * SAMPLES + sample] = value + SAMPLE_STEP * sample;
    }
}

void synthesize_frames(const fake_codec & codec, const float * post, float * pcm) {
    for (int frame = 0; frame < codec.shape.window; ++frame) {
        write_frame(pcm, frame, causal_sum(post, frame, codec.shape.context));
    }
}

bool fake_synthesize(fake_codec & codec, const float * post, float * pcm) {
    if (std::this_thread::get_id() == codec.producer) codec.on_producer = true;
    const int call = codec.synthesized++;
    if (call == codec.fail_at) return false;
    synthesize_frames(codec, post, pcm);
    return true;
}

window_stream_hooks hooks_for(fake_codec & codec) {
    window_stream_hooks hooks;
    hooks.extend_post = [&codec](int end, std::vector<float> & post, std::string * error) {
        return fake_post(codec, end, post, error);
    };
    hooks.synthesize = [&codec](const float * post, float * pcm) {
        return fake_synthesize(codec, post, pcm);
    };
    hooks.cancelled = [&codec] { return codec.cancel; };
    return hooks;
}

window_stream_shape shape_of(int window, int context) {
    window_stream_shape shape;
    shape.window = window;
    shape.context = context;
    shape.latent_dim = LATENT;
    shape.frame_size = SAMPLES;
    return shape;
}

float reference_frame(int frame, int context) {
    float total = 0.0f;
    for (int column = std::max(0, frame - context); column <= frame; ++column) {
        total += frame_weight(column);
    }
    return total;
}

std::vector<float> reference_pcm(int n_frames, int context) {
    std::vector<float> pcm(static_cast<size_t>(n_frames) * SAMPLES);
    for (int frame = 0; frame < n_frames; ++frame) {
        write_frame(pcm.data(), frame, reference_frame(frame, context));
    }
    return pcm;
}

bool same_window(const coreml_window & a, const coreml_window & b) {
    return a.begin == b.begin && a.filled == b.filled && a.core_begin == b.core_begin &&
           a.core_end == b.core_end;
}

void check_leading(int n_frames, int window, int context) {
    const std::vector<coreml_window> plan = plan_coreml_windows(n_frames, window, context);
    for (size_t index = 0; index + 1 < plan.size(); ++index) {
        expect(same_window(leading_coreml_window(static_cast<int>(index), window, context),
                           plan[index]),
               tag(n_frames, window, context) + ": leading window " +
                   std::to_string(index) + " differs from the plan");
    }
}

bool feed_frames(window_stream & stream, int n_frames, std::string * error) {
    for (int frame = 1; frame <= n_frames; ++frame) {
        if (!stream.advance(frame, error)) return false;
    }
    return true;
}

bool wait_for_worker(const fake_codec & codec, int windows) {
    const auto deadline = std::chrono::steady_clock::now() + WORKER_TIMEOUT;
    while (codec.synthesized.load() < windows) {
        if (std::chrono::steady_clock::now() > deadline) return false;
        std::this_thread::sleep_for(POLL);
    }
    return true;
}

void check_stream(int n_frames, int window, int context) {
    const std::string where = tag(n_frames, window, context);
    fake_codec codec;
    codec.shape = shape_of(window, context);
    window_stream stream(codec.shape, hooks_for(codec));
    std::string error;
    expect(feed_frames(stream, n_frames, &error), where + ": advance failed: " + error);

    const int leading = static_cast<int>(plan_coreml_windows(n_frames, window, context).size()) - 1;
    expect(stream.streamed_windows() == leading,
           where + ": " + std::to_string(stream.streamed_windows()) +
               " windows streamed, expected every leading one (" + std::to_string(leading) + ")");
    expect(wait_for_worker(codec, leading),
           where + ": the worker did not synthesise the streamed windows before finish");

    std::vector<float> pcm;
    const window_stream_status status = stream.finish(n_frames, pcm, &error);
    expect(status == window_stream_status::done, where + ": finish did not complete");
    expect(pcm == reference_pcm(n_frames, context),
           where + ": stitched waveform differs from the whole-sequence reference");
    expect(!codec.on_producer, where + ": a window was synthesised on the producer thread");
    expect(codec.computed == n_frames && !codec.went_back,
           where + ": " + std::to_string(codec.computed) + " post columns computed for " +
               std::to_string(n_frames) + " frames; each must be computed once");
}

window_stream_status finish_with(fake_codec & codec, int n_frames) {
    window_stream stream(codec.shape, hooks_for(codec));
    std::string error;
    if (!feed_frames(stream, n_frames, &error)) return window_stream_status::failed;
    std::vector<float> pcm;
    return stream.finish(n_frames, pcm, &error);
}

void check_failed_window_is_unavailable() {
    fake_codec codec;
    codec.shape = shape_of(16, 4);
    codec.fail_at = 1;
    expect(finish_with(codec, 100) == window_stream_status::unavailable,
           "a failed window must report the stream unavailable");
}

void check_cancel() {
    fake_codec codec;
    codec.shape = shape_of(16, 4);
    codec.cancel = true;
    expect(finish_with(codec, 100) == window_stream_status::cancelled,
           "a cancelled stream must report cancelled");
}

void check_post_failure() {
    fake_codec codec;
    codec.shape = shape_of(16, 4);
    codec.fail_post = true;
    window_stream stream(codec.shape, hooks_for(codec));
    std::string error;
    expect(stream.advance(16, &error), "no window is final at exactly one window of frames");
    expect(!stream.advance(17, &error), "a failed post pass must fail advance");
    expect(error == "fake post failure", "advance must pass the post pass's reason on");
}

// Destroying a stream that never finished must join its worker, not hang.
void check_abandoned_stream() {
    fake_codec codec;
    codec.shape = shape_of(16, 4);
    window_stream stream(codec.shape, hooks_for(codec));
    std::string error;
    expect(feed_frames(stream, 80, &error), "abandoned stream: advance failed");
}

void check_delivery(bool asynchronous) {
    fake_codec codec;
    codec.shape = shape_of(16, 4);
    codec.shape.asynchronous = asynchronous;
    auto hooks = hooks_for(codec);
    std::vector<float> delivered;
    std::atomic<int> callbacks{0};
    hooks.emit = [&](const std::vector<float> & pcm) {
        delivered.insert(delivered.end(), pcm.begin(), pcm.end());
        ++callbacks;
    };
    window_stream stream(codec.shape, hooks);
    std::string error;
    expect(feed_frames(stream, 17, &error), "delivery: advance failed");
    const auto deadline = std::chrono::steady_clock::now() + WORKER_TIMEOUT;
    while (!callbacks.load() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(POLL);
    }
    expect(callbacks.load() == 1, "first PCM must arrive before finish");
    expect(feed_frames(stream, 35, &error), "delivery: remaining advance failed");
    std::vector<float> pcm;
    expect(stream.finish(35, pcm, &error) == window_stream_status::done, "delivery finish failed");
    expect(delivered == reference_pcm(35, 4), "callbacks must preserve exact samples without gaps or overlap");
    expect(delivered == pcm, "callbacks and result must agree");
    expect(codec.on_producer == !asynchronous, "synthesis ran on the wrong thread");
}

void check_callback_exception() {
    fake_codec codec;
    codec.shape = shape_of(16, 4);
    auto hooks = hooks_for(codec);
    hooks.emit = [](const std::vector<float> &) { throw std::runtime_error("callback failure"); };
    window_stream stream(codec.shape, hooks);
    std::vector<float> pcm;
    std::string error;
    try {
        stream.finish(10, pcm, &error);
        fail("callback exception must propagate from finish");
    } catch (const std::runtime_error & ex) {
        expect(std::string(ex.what()) == "callback failure", "callback exception changed");
    }
}

constexpr int EXCEPTION_WINDOW = 16;
constexpr int EXCEPTION_CONTEXT = 4;
constexpr int EXCEPTION_FRAME_LIMIT = 100;
constexpr const char * CALLBACK_FAILURE = "callback failure";

void generate_until_callback_failure(window_stream & stream, int & generated) {
    std::string error;
    for (int frame = 1; frame <= EXCEPTION_FRAME_LIMIT; ++frame) {
        ++generated;
        if (!stream.advance(frame, &error)) throw std::runtime_error(error);
    }
}

bool observe_callback_failure(window_stream & stream, int frames) {
    std::string error;
    try {
        stream.advance(frames, &error);
    } catch (const std::runtime_error & ex) {
        expect(std::string(ex.what()) == CALLBACK_FAILURE, "advance changed the callback error");
        return true;
    }
    return false;
}

bool await_callback_failure(window_stream & stream) {
    const auto deadline = std::chrono::steady_clock::now() + WORKER_TIMEOUT;
    while (std::chrono::steady_clock::now() < deadline) {
        if (observe_callback_failure(stream, EXCEPTION_WINDOW + 1)) return true;
        std::this_thread::sleep_for(POLL);
    }
    return false;
}

void check_callback_stops_generation() {
    fake_codec codec;
    codec.shape = shape_of(EXCEPTION_WINDOW, EXCEPTION_CONTEXT);
    codec.shape.asynchronous = false;
    auto hooks = hooks_for(codec);
    hooks.emit = [](const std::vector<float> &) { throw std::runtime_error(CALLBACK_FAILURE); };
    window_stream stream(codec.shape, hooks);
    int generated = 0;
    try {
        generate_until_callback_failure(stream, generated);
        fail("throwing PCM callback must stop generation through advance");
    } catch (const std::runtime_error & ex) {
        expect(std::string(ex.what()) == CALLBACK_FAILURE, "generation changed the callback error");
    }
    expect(generated == EXCEPTION_WINDOW + 1, "generation continued after the first callback failed");
    expect(codec.computed == EXCEPTION_WINDOW, "post work continued after callback failure");
    expect(codec.synthesized == 1, "synthesis continued after callback failure");
}

void check_worker_callback_stops_advance() {
    fake_codec codec;
    codec.shape = shape_of(EXCEPTION_WINDOW, EXCEPTION_CONTEXT);
    auto hooks = hooks_for(codec);
    hooks.emit = [](const std::vector<float> &) { throw std::runtime_error(CALLBACK_FAILURE); };
    window_stream stream(codec.shape, hooks);
    std::string error;
    try {
        stream.advance(EXCEPTION_WINDOW + 1, &error);
    } catch (const std::runtime_error & ex) {
        expect(std::string(ex.what()) == CALLBACK_FAILURE, "worker callback error changed");
    }
    expect(await_callback_failure(stream), "worker callback exception did not reach advance");
    codec.fail_post = true;
    expect(observe_callback_failure(stream, EXCEPTION_FRAME_LIMIT),
           "a later post failure masked the callback exception");
    expect(codec.computed == EXCEPTION_WINDOW, "worker failure allowed further post work");
    expect(codec.synthesized == 1, "worker failure allowed further synthesis");
}

void check_delivery_failure() {
    fake_codec codec;
    codec.shape = shape_of(16, 4);
    codec.fail_at = 1;
    auto hooks = hooks_for(codec);
    int callbacks = 0;
    hooks.emit = [&](const std::vector<float> &) { ++callbacks; };
    window_stream stream(codec.shape, hooks);
    std::vector<float> pcm;
    std::string error;
    expect(stream.finish(35, pcm, &error) == window_stream_status::unavailable, "failed synthesis must be reported");
    expect(callbacks == 1 && stream.emitted(), "failure must retain delivery state for fallback protection");
}

void check_callback_cancellation() {
    fake_codec codec;
    codec.shape = shape_of(16, 4);
    auto hooks = hooks_for(codec);
    int callbacks = 0;
    hooks.emit = [&](const std::vector<float> &) {
        ++callbacks;
        codec.cancel = true;
    };
    window_stream stream(codec.shape, hooks);
    std::vector<float> pcm;
    std::string error;
    expect(stream.finish(35, pcm, &error) == window_stream_status::cancelled,
           "callback cancellation must stop the remaining windows");
    expect(callbacks == 1, "cancelled callback must not receive later PCM");
}

void check_sizes(int window, int context) {
    for (int n_frames = 1; n_frames <= 4 * window + 3; ++n_frames) {
        check_leading(n_frames, window, context);
        check_stream(n_frames, window, context);
    }
    check_stream(517, window, context);
}

}  // namespace

int main() {
    check_delivery(true);
    check_delivery(false);
    check_callback_exception();
    check_callback_stops_generation();
    check_worker_callback_stops_advance();
    check_delivery_failure();
    check_callback_cancellation();
    check_sizes(16, 4);
    check_sizes(16, 0);
    check_sizes(64, 10);
    check_sizes(8, 7);
    check_failed_window_is_unavailable();
    check_cancel();
    check_post_failure();
    check_abandoned_stream();

    if (g_failures == 0) {
        std::printf("[window-stream] PASS\n");
        return 0;
    }
    std::printf("[window-stream] FAIL: %d case(s)\n", g_failures);
    return 1;
}
