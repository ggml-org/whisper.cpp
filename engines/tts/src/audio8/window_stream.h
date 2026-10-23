#pragma once
#include "audio8/coreml_windows.h"

#include <condition_variable>
#include <deque>
#include <functional>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace tts_cpp {
namespace audio8 {
namespace detail {

struct window_stream_shape {
    int window = 0;
    int context = 0;
    int latent_dim = 0;
    int frame_size = 0;
    bool asynchronous = true;
};

struct window_stream_hooks {
    // Appends post columns up to `end`, latent_dim each, to the ones `post`
    // holds, on the producer's thread.
    std::function<bool(int end, std::vector<float> & post, std::string * error)> extend_post;
    std::function<bool(const float * post, float * pcm)> synthesize;
    std::function<bool()> cancelled;
    std::function<void(const std::vector<float> &)> emit;
};

enum class window_stream_status { done, unavailable, cancelled, failed };

// Window `index` of every plan long enough to hold a window after it.
coreml_window leading_coreml_window(int index, int window, int context);

class window_stream {
public:
    window_stream(const window_stream_shape & shape, window_stream_hooks hooks);
    ~window_stream();
    window_stream(const window_stream &) = delete;
    window_stream & operator=(const window_stream &) = delete;

    // n_frames are final; queues every leading window they complete. False
    // when the post pass failed, with the reason in `error`.
    bool advance(int n_frames, std::string * error);

    // Queues the rest of plan_coreml_windows(n_frames) and waits for the worker.
    window_stream_status finish(int n_frames, std::vector<float> & pcm, std::string * error);

    // Windows queued by advance(), i.e. synthesised while frames were coming.
    int streamed_windows() const { return streamed_; }
    int window_frames() const { return shape_.window; }
    double post_ms() const { return post_ms_; }
    bool emitted() const { return emitted_; }
    bool asynchronous() const { return shape_.asynchronous; }
    double synthesis_ms() const { return synthesis_ms_; }

private:
    struct job {
        int index = 0;
        coreml_window span;
        std::vector<float> input;
    };

    void queue_span(const coreml_window & span);
    bool queue_leading(int index, std::string * error);
    bool queue_remaining(const std::vector<coreml_window> & plan, std::string * error);
    bool compute_post(int end, std::string * error);
    void push(job work);
    bool pop(job & work);
    void run_worker();
    bool run_job(const job & work, std::vector<float> & out);
    bool process_job(const job & work, std::vector<float> & out);
    void keep_core(const job & work, const std::vector<float> & out);
    void rethrow_callback_error();
    void close();
    window_stream_status wait_for_worker();
    void stitch(int n_frames, std::vector<float> & pcm) const;

    window_stream_shape shape_;
    window_stream_hooks hooks_;
    int queued_ = 0;
    int streamed_ = 0;
    double post_ms_ = 0.0;
    double synthesis_ms_ = 0.0;
    std::vector<float> post_;

    std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<job> pending_;
    std::vector<std::vector<float>> cores_;
    int finished_ = 0;
    bool closing_ = false;
    bool failed_ = false;
    bool cancelled_ = false;
    bool emitted_ = false;
    std::exception_ptr callback_error_;
    std::thread worker_;
};

}  // namespace detail
}  // namespace audio8
}  // namespace tts_cpp
