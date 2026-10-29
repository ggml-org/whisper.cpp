#include "chatterbox_coreml.h"

#include "coreml_sidecar_path.h"

#ifdef TTS_CPP_USE_COREML
#include "moss/coreml/runner.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace tts_cpp::chatterbox::detail {
namespace {

#ifdef TTS_CPP_USE_COREML
constexpr int MEL = 80;
constexpr int EMBED = 512;
constexpr int TIME_EMBED = 1024;
constexpr int MAX_RANK = 8;
constexpr int FUNCTION_LIST_BYTES = 4096;
constexpr const char * STAGE_NAMES[] = {"encoder", "CFM", "HiFT"};

std::vector<int64_t> dims_of(moss_coreml_model * model, const char * name) {
    int64_t dims[MAX_RANK] = {};
    const int rank = moss_coreml_feature_dims(model, name, dims, MAX_RANK);
    return std::vector<int64_t>(dims, dims + std::max(rank, 0));
}

std::string dims_text(const std::vector<int64_t> & dims) {
    std::ostringstream text;
    text << '[';
    for (size_t i = 0; i < dims.size(); ++i) text << (i ? ", " : "") << dims[i];
    text << ']';
    return text.str();
}

// One prediction feature: its name, the shape this call packs it as, and a
// buffer of exactly that many floats.
struct input_feature {
    const char * name;
    std::vector<int64_t> dims;
    const std::vector<float> * data;
};
struct output_feature {
    const char * name;
    std::vector<int64_t> dims;
    std::vector<float> * data;
};

// The shared runner copies every feature by the shape the model declares, so
// a sidecar declaring any other shape than the buffer packed for it would be
// read or written out of bounds. Every feature is checked first; a mismatch
// is an incompatible export and throws, which retires the stage (or fails a
// strict call) before anything is copied.
void predict(moss_coreml_model * model, const std::vector<input_feature> & inputs,
             const std::vector<output_feature> & outputs, const char * stage) {
    auto check = [&](const char * name, const std::vector<int64_t> & dims, size_t size) {
        const std::vector<int64_t> declared = dims_of(model, name);
        int64_t elements = 1;
        for (int64_t d : dims) elements *= d;
        if (declared != dims || (int64_t) size != elements) {
            throw std::runtime_error(std::string("chatterbox: the Core ML ") + stage + " sidecar declares " + name +
                                     " " + dims_text(declared) + " where this call packs " + dims_text(dims));
        }
    };
    std::vector<const char *> in_names, out_names;
    std::vector<const float *> in_data;
    std::vector<float *> out_data;
    for (const input_feature & f : inputs) {
        check(f.name, f.dims, f.data->size());
        in_names.push_back(f.name);
        in_data.push_back(f.data->data());
    }
    for (const output_feature & f : outputs) {
        check(f.name, f.dims, f.data->size());
        out_names.push_back(f.name);
        out_data.push_back(f.data->data());
    }
    if (moss_coreml_predict(model, in_names.data(), in_data.data(), (int) in_names.size(), out_names.data(),
                            out_data.data(), (int) out_names.size()) != 0) {
        throw std::runtime_error(std::string("chatterbox: Core ML ") + stage + " prediction failed");
    }
}

void require_finite(const std::vector<float> & values, const char * stage) {
    if (!std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); })) {
        throw std::runtime_error(std::string("chatterbox: non-finite Core ML ") + stage + " output");
    }
}
#endif

} // namespace

bool coreml_valid_units(const std::string & units) {
    return units == "all" || units == "cpu_and_ane" || units == "cpu_and_gpu" || units == "cpu_only";
}

coreml_policy resolve_coreml_policy(const std::string & compute_units, std::optional<bool> strict) {
    coreml_policy policy;
    policy.disabled = std::getenv(COREML_DISABLE_ENV) != nullptr;
    policy.strict = strict.has_value() ? *strict : std::getenv(COREML_STRICT_ENV) != nullptr;
    if (!compute_units.empty()) {
        if (!coreml_valid_units(compute_units)) {
            throw std::runtime_error("chatterbox: invalid coreml_compute_units='" + compute_units +
                                     "' (expected all, cpu_and_ane, cpu_and_gpu or cpu_only)");
        }
        policy.compute_units = compute_units;
    } else if (const char * env = std::getenv(COREML_UNITS_ENV); env && *env) {
        if (coreml_valid_units(env)) {
            policy.compute_units = env;
        } else {
            std::fprintf(stderr, "chatterbox: ignoring %s=%s (expected all, cpu_and_ane, cpu_and_gpu or cpu_only)\n",
                         COREML_UNITS_ENV, env);
        }
    }
    return policy;
}

std::string coreml_stage_path(const std::string & s3gen_gguf, const char * suffix) {
    return ::tts_cpp::detail::coreml_sidecar_path(s3gen_gguf, suffix);
}

bool coreml_sidecars_enabled(const std::string & s3gen_gguf, const coreml_policy & policy) {
#ifdef TTS_CPP_USE_COREML
    if (policy.disabled || s3gen_gguf.empty()) return false;
    for (const char * suffix : {COREML_ENCODER_SUFFIX, COREML_CFM_SUFFIX, COREML_HIFT_SUFFIX}) {
        if (::tts_cpp::detail::coreml_sidecar_exists(coreml_stage_path(s3gen_gguf, suffix))) return true;
    }
#else
    (void) s3gen_gguf;
    (void) policy;
#endif
    return false;
}

bool coreml_offloads_s3gen(const std::string & s3gen_gguf, const coreml_policy & policy) {
#ifdef TTS_CPP_USE_COREML
    return !policy.disabled && !s3gen_gguf.empty() &&
           ::tts_cpp::detail::coreml_sidecar_exists(coreml_stage_path(s3gen_gguf, COREML_CFM_SUFFIX)) &&
           ::tts_cpp::detail::coreml_sidecar_exists(coreml_stage_path(s3gen_gguf, COREML_HIFT_SUFFIX));
#else
    (void) s3gen_gguf;
    (void) policy;
    return false;
#endif
}

std::vector<int> coreml_buckets(const std::string & function_names, const std::string & prefix) {
    std::vector<int> buckets;
    std::stringstream list(function_names);
    std::string name;
    while (std::getline(list, name, ',')) {
        if (name.size() <= prefix.size() || name.compare(0, prefix.size(), prefix) != 0) continue;
        const std::string digits = name.substr(prefix.size());
        if (!std::all_of(digits.begin(), digits.end(), [](char c) { return c >= '0' && c <= '9'; })) continue;
        if (digits.size() > 6) continue;
        const int width = std::atoi(digits.c_str());
        if (width > 0) buckets.push_back(width);
    }
    std::sort(buckets.begin(), buckets.end());
    buckets.erase(std::unique(buckets.begin(), buckets.end()), buckets.end());
    return buckets;
}

int coreml_pick_bucket(const std::vector<int> & buckets, int needed) {
    for (int width : buckets) {
        if (width >= needed) return width;
    }
    return -1;
}

std::vector<hift_window> coreml_hift_windows(int frames, int window, int context) {
    std::vector<hift_window> plan;
    if (frames <= 0 || window <= 2 * context) return plan;
    for (int begin = 0; begin < frames;) {
        const int start = std::max(0, begin - context);
        const int end = start + window >= frames ? frames : start + window - context;
        plan.push_back({start, begin, end});
        begin = end;
    }
    return plan;
}

s3gen_coreml::s3gen_coreml(const std::string & s3gen_gguf, const std::string & compute_units, int token_mel_ratio)
    : units_(compute_units), ratio_(token_mel_ratio) {
#ifdef TTS_CPP_USE_COREML
    label_ = units_ == "cpu_and_ane" ? "coreml-ane" : units_ == "cpu_and_gpu" ? "coreml-gpu"
           : units_ == "cpu_only"    ? "coreml-cpu" : "coreml-all";
    const char * suffixes[STAGES] = {COREML_ENCODER_SUFFIX, COREML_CFM_SUFFIX, COREML_HIFT_SUFFIX};
    const char * prefixes[STAGES] = {COREML_ENCODER_PREFIX, COREML_CFM_PREFIX, nullptr};
    for (int s = 0; s < STAGES; ++s) {
        stage_state & st = stages_[s];
        st.path = coreml_stage_path(s3gen_gguf, suffixes[s]);
        if (!::tts_cpp::detail::coreml_sidecar_exists(st.path)) continue;
        present_[s] = true;
        if (prefixes[s] != nullptr) {
            char names[FUNCTION_LIST_BYTES] = {};
            if (moss_coreml_function_names(st.path.c_str(), names, sizeof(names)) <= 0) {
                std::fprintf(stderr, "chatterbox: cannot list the functions of %s (needs macOS 15 / iOS 18), "
                                     "retaining ggml for the %s\n", st.path.c_str(), STAGE_NAMES[s]);
                continue;
            }
            st.buckets = coreml_buckets(names, prefixes[s]);
            if (st.buckets.empty()) {
                std::fprintf(stderr, "chatterbox: %s has no %s* functions, retaining ggml for the %s\n",
                             st.path.c_str(), prefixes[s], STAGE_NAMES[s]);
                continue;
            }
        }
        st.retired = false;
    }
#else
    (void) s3gen_gguf;
#endif
}

s3gen_coreml::~s3gen_coreml() = default;

bool s3gen_coreml::available(stage which) const {
    return !stages_[which].retired.load(std::memory_order_acquire);
}

void s3gen_coreml::retire(stage which) {
    std::lock_guard<std::mutex> lock(mutex_);
    stages_[which].retired.store(true, std::memory_order_release);
    stages_[which].loaded.clear();
    loaded_.notify_all();
}

std::shared_ptr<moss_coreml_model> s3gen_coreml::model_for(stage which, int bucket) {
#ifdef TTS_CPP_USE_COREML
    stage_state & st = stages_[which];
    {
        // A load takes ~0.2-0.6 s once compiled; never hold the lock across
        // one, so a warm-up thread and a synthesis call can load in parallel.
        std::unique_lock<std::mutex> lock(mutex_);
        loaded_.wait(lock, [&] { return st.loading.count(bucket) == 0 || st.retired.load(); });
        if (st.retired.load()) return {};
        auto found = st.loaded.find(bucket);
        if (found != st.loaded.end()) return found->second;
        if (which != HIFT && std::find(st.buckets.begin(), st.buckets.end(), bucket) == st.buckets.end()) return {};
        st.loading.insert(bucket);
    }
    const std::string function = which == ENCODER ? COREML_ENCODER_PREFIX + std::to_string(bucket)
                               : which == CFM     ? COREML_CFM_PREFIX + std::to_string(bucket)
                                                  : std::string();
    moss_coreml_model * raw = function.empty()
        ? moss_coreml_load(st.path.c_str(), units_.c_str())
        : moss_coreml_load_function(st.path.c_str(), units_.c_str(), function.c_str());
    std::shared_ptr<moss_coreml_model> model = raw ? std::shared_ptr<moss_coreml_model>(raw, moss_coreml_free) : nullptr;
    std::lock_guard<std::mutex> lock(mutex_);
    st.loading.erase(bucket);
    loaded_.notify_all();
    if (!model) {
        std::fprintf(stderr, "chatterbox: cannot load Core ML %s %s%s, retaining ggml for it\n",
                     STAGE_NAMES[which], st.path.c_str(), function.empty() ? "" : (" (" + function + ")").c_str());
        if (!function.empty()) {
            st.buckets.erase(std::remove(st.buckets.begin(), st.buckets.end(), bucket), st.buckets.end());
        }
        if (function.empty() || st.buckets.empty()) st.retired.store(true);
        return {};
    }
    if (st.retired.load()) return {};
    st.loaded.emplace(bucket, model);
    return model;
#else
    (void) which;
    (void) bucket;
    return {};
#endif
}

void s3gen_coreml::warm(const std::vector<int> & tokens, const std::atomic<bool> * stop) {
    // HiFT first (every call uses one window model), then the encoder and
    // CFM functions each expected call needs, in call order.  A function
    // load grows with the whole multi-function program (~1 s for the CFM),
    // so only the predicted widths are loaded, not every width.
    std::vector<std::pair<stage, int>> order;
    if (available(HIFT)) order.emplace_back(HIFT, 0);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (int n : tokens) {
            const int width = coreml_pick_bucket(stages_[ENCODER].buckets, n);
            const int frames = coreml_pick_bucket(stages_[CFM].buckets, ratio_ * n);
            if (width > 0) order.emplace_back(ENCODER, width);
            if (frames > 0) order.emplace_back(CFM, frames);
        }
    }
    for (const auto & [which, bucket] : order) {
        if (stop != nullptr && stop->load(std::memory_order_relaxed)) return;
        if (available(which)) model_for(which, bucket);
    }
}

int coreml_expected_speech_tokens(size_t text_bytes) {
    return (int) std::lround(1.15 * (double) text_bytes);
}

std::vector<int> coreml_stream_windows(int speech_tokens, int first_chunk, int chunk, int left_context) {
    std::vector<int> windows;
    if (speech_tokens <= 0 || chunk <= 0) return windows;
    int start = 0;
    for (int end = std::min(speech_tokens, first_chunk > 0 ? first_chunk : chunk); start < speech_tokens;
         end = std::min(speech_tokens, end + chunk)) {
        const int from = left_context > 0 ? std::max(0, start - left_context) : 0;
        windows.push_back(end - from);
        start = end;
    }
    return windows;
}

bool s3gen_coreml::encode(const std::vector<float> & embedding, int tokens, int ratio, std::vector<float> & mu) {
#ifdef TTS_CPP_USE_COREML
    if (!available(ENCODER) || tokens <= 0 || (int64_t) embedding.size() != (int64_t) tokens * EMBED) return false;
    int width = -1;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        width = coreml_pick_bucket(stages_[ENCODER].buckets, tokens);
    }
    if (width < 0) return false;
    auto model = model_for(ENCODER, width);
    if (!model) return false;
    std::vector<float> padded((size_t) width * EMBED, 0.0f);
    std::copy(embedding.begin(), embedding.end(), padded.begin());
    std::vector<float> valid((size_t) width, 0.0f);
    std::fill(valid.begin(), valid.begin() + tokens, 1.0f);
    std::vector<float> out((size_t) width * ratio * MEL);
    predict(model.get(),
            {{"embedding", {1, width, EMBED}, &padded}, {"valid", {1, 1, width}, &valid}},
            {{"mu", {1, (int64_t) width * ratio, MEL}, &out}}, "encoder");
    out.resize((size_t) tokens * ratio * MEL);
    require_finite(out, "encoder");
    mu = std::move(out);
    return true;
#else
    (void) embedding;
    (void) tokens;
    (void) ratio;
    (void) mu;
    return false;
#endif
}

bool s3gen_coreml::velocity(int batch, int frames, const std::vector<const float *> & x,
                            const std::vector<const float *> & mu, const std::vector<const float *> & cond,
                            const std::vector<const float *> & spks, const std::vector<const float *> & t_emb,
                            std::vector<std::vector<float>> & out) {
#ifdef TTS_CPP_USE_COREML
    // Callers pass the CFG pair; a batch-1 (meanflow) sidecar uses the first.
    const size_t planes = (size_t) std::max(batch, 0);
    if (!available(CFM) || frames <= 0 || batch <= 0 || x.size() < planes || mu.size() < planes ||
        cond.size() < planes || spks.size() < planes || t_emb.size() < planes) {
        return false;
    }
    int width = -1;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        width = coreml_pick_bucket(stages_[CFM].buckets, frames);
    }
    if (width < 0) return false;
    auto model = model_for(CFM, width);
    if (!model) return false;
    const std::vector<int64_t> plane = {batch, MEL, width};
    const size_t plane_size = (size_t) MEL * width;
    auto pack = [&](const std::vector<const float *> & planes) {
        std::vector<float> packed((size_t) batch * plane_size, 0.0f);
        for (int b = 0; b < batch; ++b) {
            for (int m = 0; m < MEL; ++m) {
                std::memcpy(packed.data() + b * plane_size + (size_t) m * width, planes[b] + (size_t) m * frames,
                            (size_t) frames * sizeof(float));
            }
        }
        return packed;
    };
    auto rows = [&](const std::vector<const float *> & vectors, int width_each) {
        std::vector<float> packed((size_t) batch * width_each);
        for (int b = 0; b < batch; ++b) std::memcpy(packed.data() + (size_t) b * width_each, vectors[b],
                                                    (size_t) width_each * sizeof(float));
        return packed;
    };
    const std::vector<float> xs = pack(x), mus = pack(mu), conds = pack(cond);
    const std::vector<float> speakers = rows(spks, MEL), times = rows(t_emb, TIME_EMBED);
    std::vector<float> valid((size_t) width, 0.0f);
    std::fill(valid.begin(), valid.begin() + frames, 1.0f);
    std::vector<float> velocity((size_t) batch * plane_size);
    predict(model.get(),
            {{"x", plane, &xs}, {"mu", plane, &mus}, {"cond", plane, &conds}, {"spks", {batch, MEL}, &speakers},
             {"t_emb", {batch, TIME_EMBED}, &times}, {"valid", {1, 1, width}, &valid}},
            {{"velocity", plane, &velocity}}, "CFM");
    out.assign(batch, std::vector<float>((size_t) MEL * frames));
    for (int b = 0; b < batch; ++b) {
        for (int m = 0; m < MEL; ++m) {
            std::memcpy(out[b].data() + (size_t) m * frames, velocity.data() + b * plane_size + (size_t) m * width,
                        (size_t) frames * sizeof(float));
        }
        require_finite(out[b], "CFM");
    }
    return true;
#else
    (void) batch;
    (void) frames;
    (void) x;
    (void) mu;
    (void) cond;
    (void) spks;
    (void) t_emb;
    (void) out;
    return false;
#endif
}

bool s3gen_coreml::vocode(const std::vector<float> & mel, int frames, const std::vector<float> & source_stft,
                          std::vector<float> & spectrum) {
#ifdef TTS_CPP_USE_COREML
    const int64_t stft = (int64_t) frames * HIFT_STFT_PER_FRAME + 1;
    if (!available(HIFT) || frames <= 0 || (int64_t) mel.size() != (int64_t) MEL * frames ||
        (int64_t) source_stft.size() != HIFT_STFT_BINS * stft) {
        return false;
    }
    auto model = model_for(HIFT, 0);
    if (!model) return false;
    // The window is the sidecar's; predict() checks every feature against it.
    const std::vector<int64_t> mel_dims = dims_of(model.get(), "mel");
    const int window = mel_dims.size() == 3 ? (int) mel_dims[2] : 0;
    if (window <= 2 * HIFT_CONTEXT_FRAMES) {
        throw std::runtime_error("chatterbox: the Core ML HiFT sidecar declares mel " + dims_text(mel_dims) +
                                 ", narrower than its " + std::to_string(2 * HIFT_CONTEXT_FRAMES) +
                                 " context frames");
    }
    const int64_t window_stft = (int64_t) window * HIFT_STFT_PER_FRAME + 1;
    const std::vector<int64_t> spectrum_dims = {1, HIFT_STFT_BINS, window_stft};
    std::vector<float> mel_in((size_t) MEL * window), source_in((size_t) HIFT_STFT_BINS * window_stft);
    std::vector<float> valid((size_t) window), out((size_t) HIFT_STFT_BINS * window_stft);
    std::vector<float> result((size_t) HIFT_STFT_BINS * stft, 0.0f);
    for (const hift_window & w : coreml_hift_windows(frames, window, HIFT_CONTEXT_FRAMES)) {
        const int count = std::min(window, frames - w.start);
        const int64_t source_count = (int64_t) count * HIFT_STFT_PER_FRAME + 1;
        const int64_t source_start = (int64_t) w.start * HIFT_STFT_PER_FRAME;
        std::fill(mel_in.begin(), mel_in.end(), 0.0f);
        std::fill(source_in.begin(), source_in.end(), 0.0f);
        std::fill(valid.begin(), valid.end(), 0.0f);
        std::fill(valid.begin(), valid.begin() + count, 1.0f);
        for (int m = 0; m < MEL; ++m) {
            std::memcpy(mel_in.data() + (size_t) m * window, mel.data() + (size_t) m * frames + w.start,
                        (size_t) count * sizeof(float));
        }
        for (int c = 0; c < HIFT_STFT_BINS; ++c) {
            std::memcpy(source_in.data() + (size_t) c * window_stft,
                        source_stft.data() + (size_t) c * stft + source_start, (size_t) source_count * sizeof(float));
        }
        predict(model.get(),
                {{"mel", {1, MEL, window}, &mel_in}, {"source", spectrum_dims, &source_in},
                 {"valid", {1, 1, window}, &valid}},
                {{"spectrum", spectrum_dims, &out}}, "HiFT");
        const int64_t keep_begin = (int64_t) (w.keep_begin - w.start) * HIFT_STFT_PER_FRAME;
        const int64_t keep_count = (int64_t) (w.keep_end - w.keep_begin) * HIFT_STFT_PER_FRAME +
                                   (w.keep_end == frames ? 1 : 0);
        for (int c = 0; c < HIFT_STFT_BINS; ++c) {
            std::memcpy(result.data() + (size_t) c * stft + (size_t) w.keep_begin * HIFT_STFT_PER_FRAME,
                        out.data() + (size_t) c * window_stft + keep_begin, (size_t) keep_count * sizeof(float));
        }
    }
    require_finite(result, "HiFT");
    spectrum = std::move(result);
    return true;
#else
    (void) mel;
    (void) frames;
    (void) source_stft;
    (void) spectrum;
    return false;
#endif
}

} // namespace tts_cpp::chatterbox::detail
