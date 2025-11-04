#include "napi.h"
#include "common.h"
#include "common-whisper.h"

#include "whisper.h"

#include <string>
#include <thread>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cfloat>

struct whisper_params {
    int32_t n_threads    = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_processors = 1;
    int32_t offset_t_ms  = 0;
    int32_t offset_n     = 0;
    int32_t duration_ms  = 0;
    int32_t max_context  = -1;
    int32_t max_len      = 0;
    int32_t best_of      = 5;
    int32_t beam_size    = -1;
    int32_t audio_ctx    = 0;

    float word_thold    = 0.01f;
    float entropy_thold = 2.4f;
    float logprob_thold = -1.0f;

    bool translate      = false;
    bool diarize        = false;
    bool output_txt     = false;
    bool output_vtt     = false;
    bool output_srt     = false;
    bool output_wts     = false;
    bool output_csv     = false;
    bool print_special  = false;
    bool print_colors   = false;
    bool print_progress = false;
    bool no_timestamps  = false;
    bool no_prints      = false;
    bool detect_language= false;
    bool use_gpu        = true;
    bool flash_attn     = false;
    bool comma_in_time  = true;

    std::string language = "en";
    std::string prompt;
    std::string model    = "../../ggml-large.bin";

    std::vector<std::string> fname_inp = {};
    std::vector<std::string> fname_out = {};

    std::vector<float> pcmf32 = {}; // mono-channel F32 PCM

    // Voice Activity Detection (VAD) parameters
    bool        vad           = false;
    std::string vad_model     = "";
    float       vad_threshold = 0.5f;
    int         vad_min_speech_duration_ms = 250;
    int         vad_min_silence_duration_ms = 100;
    float       vad_max_speech_duration_s = FLT_MAX;
    int         vad_speech_pad_ms = 30;
    float       vad_samples_overlap = 0.1f;
};

struct whisper_print_user_data {
    const whisper_params * params;

    const std::vector<std::vector<float>> * pcmf32s;
};

void whisper_print_segment_callback(struct whisper_context * ctx, struct whisper_state * state, int n_new, void * user_data) {
    const auto & params  = *((whisper_print_user_data *) user_data)->params;
    const auto & pcmf32s = *((whisper_print_user_data *) user_data)->pcmf32s;

    const int n_segments = whisper_full_n_segments(ctx);

    std::string speaker = "";

    int64_t t0;
    int64_t t1;

    // print the last n_new segments
    const int s0 = n_segments - n_new;

    if (s0 == 0) {
        printf("\n");
    }

    for (int i = s0; i < n_segments; i++) {
        if (!params.no_timestamps || params.diarize) {
            t0 = whisper_full_get_segment_t0(ctx, i);
            t1 = whisper_full_get_segment_t1(ctx, i);
        }

        if (!params.no_timestamps && !params.no_prints) {
            printf("[%s --> %s]  ", to_timestamp(t0).c_str(), to_timestamp(t1).c_str());
        }

        if (params.diarize && pcmf32s.size() == 2) {
            const int64_t n_samples = pcmf32s[0].size();

            const int64_t is0 = timestamp_to_sample(t0, n_samples, WHISPER_SAMPLE_RATE);
            const int64_t is1 = timestamp_to_sample(t1, n_samples, WHISPER_SAMPLE_RATE);

            double energy0 = 0.0f;
            double energy1 = 0.0f;

            for (int64_t j = is0; j < is1; j++) {
                energy0 += fabs(pcmf32s[0][j]);
                energy1 += fabs(pcmf32s[1][j]);
            }

            if (energy0 > 1.1*energy1) {
                speaker = "(speaker 0)";
            } else if (energy1 > 1.1*energy0) {
                speaker = "(speaker 1)";
            } else {
                speaker = "(speaker ?)";
            }

            //printf("is0 = %lld, is1 = %lld, energy0 = %f, energy1 = %f, %s\n", is0, is1, energy0, energy1, speaker.c_str());
        }

        // colorful print bug
        //
        if (!params.no_prints) {
            const char * text = whisper_full_get_segment_text(ctx, i);
            printf("%s%s", speaker.c_str(), text);
        }


        // with timestamps or speakers: each segment on new line
        if ((!params.no_timestamps || params.diarize) && !params.no_prints) {
            printf("\n");
        }

        fflush(stdout);
    }
}

void cb_log_disable(enum ggml_log_level, const char *, void *) {}

struct whisper_result {
    std::vector<std::vector<std::string>> segments;
    std::string language;
};

class ProgressWorker : public Napi::AsyncWorker {
 public:
    ProgressWorker(Napi::Function& callback, whisper_params params, Napi::Function progress_callback, Napi::Env env)
        : Napi::AsyncWorker(callback), params(params), env(env) {
        // Create thread-safe function
        if (!progress_callback.IsEmpty()) {
            tsfn = Napi::ThreadSafeFunction::New(
                env,
                progress_callback,
                "Progress Callback",
                0,
                1
            );
        }
    }

    ~ProgressWorker() {
        if (tsfn) {
            // Make sure to release the thread-safe function on destruction
            tsfn.Release();
        }
    }

    void Execute() override {
        // Use custom run function with progress callback support
        run_with_progress(params, result);
    }

    void OnOK() override {
        Napi::HandleScope scope(Env());

        if (params.detect_language) {
            Napi::Object resultObj = Napi::Object::New(Env());
            resultObj.Set("language", Napi::String::New(Env(), result.language));
            Callback().Call({Env().Null(), resultObj});
        }

        Napi::Object returnObj = Napi::Object::New(Env());
        if (!result.language.empty()) {
            returnObj.Set("language", Napi::String::New(Env(), result.language));
        }
        Napi::Array transcriptionArray = Napi::Array::New(Env(), result.segments.size());
        for (uint64_t i = 0; i < result.segments.size(); ++i) {
            Napi::Object tmp = Napi::Array::New(Env(), 3);
            for (uint64_t j = 0; j < 3; ++j) {
                tmp[j] = Napi::String::New(Env(), result.segments[i][j]);
            }
            transcriptionArray[i] = tmp;
         }
         returnObj.Set("transcription", transcriptionArray);
         Callback().Call({Env().Null(), returnObj});
    }

    // Progress callback function - using thread-safe function
    void OnProgress(int progress) {
        if (tsfn) {
            // Use thread-safe function to call JavaScript callback
            auto callback = [progress](Napi::Env env, Napi::Function jsCallback) {
                jsCallback.Call({Napi::Number::New(env, progress)});
            };
            
            tsfn.BlockingCall(callback);
        }
    }

 private:
    whisper_params params;
    whisper_result result;
    Napi::Env env;
    Napi::ThreadSafeFunction tsfn;

    // Custom run function with progress callback support
    int run_with_progress(whisper_params &params, whisper_result & result) {
        if (params.no_prints) {
            whisper_log_set(cb_log_disable, NULL);
        }

        if (params.fname_inp.empty() && params.pcmf32.empty()) {
            fprintf(stderr, "error: no input files or audio buffer specified\n");
            return 2;
        }

        if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1) {
            fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
            exit(0);
        }

        // whisper init
        struct whisper_context_params cparams = whisper_context_default_params();
        cparams.use_gpu = params.use_gpu;
        cparams.flash_attn = params.flash_attn;
        struct whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);

        if (ctx == nullptr) {
            fprintf(stderr, "error: failed to initialize whisper context\n");
            return 3;
        }

        // If params.pcmf32 provides, set params.fname_inp as "buffer"
        if (!params.pcmf32.empty()) {
            fprintf(stderr, "info: using audio buffer as input\n");
            params.fname_inp.clear();
            params.fname_inp.emplace_back("buffer");
        }

        for (int f = 0; f < (int) params.fname_inp.size(); ++f) {
            const auto fname_inp = params.fname_inp[f];
            const auto fname_out = f < (int)params.fname_out.size() && !params.fname_out[f].empty() ? params.fname_out[f] : params.fname_inp[f];

            std::vector<float> pcmf32; // mono-channel F32 PCM
            std::vector<std::vector<float>> pcmf32s; // stereo-channel F32 PCM

            // If params.pcmf32 is empty, read input audio file
            if (params.pcmf32.empty()) {
                if (!::read_audio_data(fname_inp, pcmf32, pcmf32s, params.diarize)) {
                    fprintf(stderr, "error: failed to read audio file '%s'\n", fname_inp.c_str());
                    continue;
                }
            } else {
                pcmf32 = params.pcmf32;
            }

            // Print system info
            if (!params.no_prints) {
                fprintf(stderr, "\n");
                fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                        params.n_threads*params.n_processors, std::thread::hardware_concurrency(), whisper_print_system_info());
            }

            // Print processing info
            if (!params.no_prints) {
                fprintf(stderr, "\n");
                if (!whisper_is_multilingual(ctx)) {
                    if (params.language != "en" || params.translate) {
                        params.language = "en";
                        params.translate = false;
                        fprintf(stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
                    }
                }
                fprintf(stderr, "%s: processing '%s' (%d samples, %.1f sec), %d threads, %d processors, lang = %s, task = %s, timestamps = %d, audio_ctx = %d ...\n",
                        __func__, fname_inp.c_str(), int(pcmf32.size()), float(pcmf32.size())/WHISPER_SAMPLE_RATE,
                        params.n_threads, params.n_processors,
                        params.language.c_str(),
                        params.translate ? "translate" : "transcribe",
                        params.no_timestamps ? 0 : 1,
                        params.audio_ctx);

                fprintf(stderr, "\n");
            }

            // Run inference
            {
                whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

                wparams.strategy = params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY;

                wparams.print_realtime   = false;
                wparams.print_progress   = params.print_progress;
                wparams.print_timestamps = !params.no_timestamps;
                wparams.print_special    = params.print_special;
                wparams.translate        = params.translate;
                wparams.language         = params.detect_language ? "auto" : params.language.c_str();
                wparams.detect_language  = params.detect_language;
                wparams.n_threads        = params.n_threads;
                wparams.n_max_text_ctx   = params.max_context >= 0 ? params.max_context : wparams.n_max_text_ctx;
                wparams.offset_ms        = params.offset_t_ms;
                wparams.duration_ms      = params.duration_ms;

                wparams.token_timestamps = params.output_wts || params.max_len > 0;
                wparams.thold_pt         = params.word_thold;
                wparams.entropy_thold    = params.entropy_thold;
                wparams.logprob_thold    = params.logprob_thold;
                wparams.max_len          = params.output_wts && params.max_len == 0 ? 60 : params.max_len;
                wparams.audio_ctx        = params.audio_ctx;

                wparams.greedy.best_of        = params.best_of;
                wparams.beam_search.beam_size = params.beam_size;

                wparams.initial_prompt   = params.prompt.c_str();

                wparams.no_timestamps    = params.no_timestamps;

                whisper_print_user_data user_data = { &params, &pcmf32s };

                // This callback is called for each new segment
                if (!wparams.print_realtime) {
                    wparams.new_segment_callback           = whisper_print_segment_callback;
                    wparams.new_segment_callback_user_data = &user_data;
                }

                // Set progress callback
                wparams.progress_callback = [](struct whisper_context * /*ctx*/, struct whisper_state * /*state*/, int progress, void * user_data) {
                    ProgressWorker* worker = static_cast<ProgressWorker*>(user_data);
                    worker->OnProgress(progress);
                };
                wparams.progress_callback_user_data = this;

                // Set VAD parameters
                wparams.vad            = params.vad;
                wparams.vad_model_path = params.vad_model.c_str();

                wparams.vad_params.threshold               = params.vad_threshold;
                wparams.vad_params.min_speech_duration_ms  = params.vad_min_speech_duration_ms;
                wparams.vad_params.min_silence_duration_ms = params.vad_min_silence_duration_ms;
                wparams.vad_params.max_speech_duration_s   = params.vad_max_speech_duration_s;
                wparams.vad_params.speech_pad_ms           = params.vad_speech_pad_ms;
                wparams.vad_params.samples_overlap         = params.vad_samples_overlap;

                if (whisper_full_parallel(ctx, wparams, pcmf32.data(), pcmf32.size(), params.n_processors) != 0) {
                    fprintf(stderr, "failed to process audio\n");
                    return 10;
                }
            }
        }

        if (params.detect_language || params.language == "auto") {
            result.language = whisper_lang_str(whisper_full_lang_id(ctx));
        }
        const int n_segments = whisper_full_n_segments(ctx);
        result.segments.resize(n_segments);

        for (int i = 0; i < n_segments; ++i) {
            const char * text = whisper_full_get_segment_text(ctx, i);
            const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
            const int64_t t1 = whisper_full_get_segment_t1(ctx, i);

            result.segments[i].emplace_back(to_timestamp(t0, params.comma_in_time));
            result.segments[i].emplace_back(to_timestamp(t1, params.comma_in_time));
            result.segments[i].emplace_back(text);
        }

        whisper_print_timings(ctx);
        whisper_free(ctx);

        return 0;
    }
};

Napi::Value whisper(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() <= 0 || !info[0].IsObject()) {
    Napi::TypeError::New(env, "object expected").ThrowAsJavaScriptException();
  }
  whisper_params params;

  Napi::Object whisper_params = info[0].As<Napi::Object>();
  std::string language = whisper_params.Get("language").As<Napi::String>();
  std::string model = whisper_params.Get("model").As<Napi::String>();
  std::string input = whisper_params.Get("fname_inp").As<Napi::String>();
  
  bool use_gpu = true;
  if (whisper_params.Has("use_gpu") && whisper_params.Get("use_gpu").IsBoolean()) {
    use_gpu = whisper_params.Get("use_gpu").As<Napi::Boolean>();
  }
  
  bool flash_attn = false;
  if (whisper_params.Has("flash_attn") && whisper_params.Get("flash_attn").IsBoolean()) {
    flash_attn = whisper_params.Get("flash_attn").As<Napi::Boolean>();
  }
  
  bool no_prints = false;
  if (whisper_params.Has("no_prints") && whisper_params.Get("no_prints").IsBoolean()) {
    no_prints = whisper_params.Get("no_prints").As<Napi::Boolean>();
  }
  
  bool no_timestamps = false;
  if (whisper_params.Has("no_timestamps") && whisper_params.Get("no_timestamps").IsBoolean()) {
    no_timestamps = whisper_params.Get("no_timestamps").As<Napi::Boolean>();
  }
  
  bool detect_language = false;
  if (whisper_params.Has("detect_language") && whisper_params.Get("detect_language").IsBoolean()) {
    detect_language = whisper_params.Get("detect_language").As<Napi::Boolean>();
  }
  
  int32_t audio_ctx = 0;
  if (whisper_params.Has("audio_ctx") && whisper_params.Get("audio_ctx").IsNumber()) {
    audio_ctx = whisper_params.Get("audio_ctx").As<Napi::Number>();
  }
  
  bool comma_in_time = true;
  if (whisper_params.Has("comma_in_time") && whisper_params.Get("comma_in_time").IsBoolean()) {
    comma_in_time = whisper_params.Get("comma_in_time").As<Napi::Boolean>();
  }
  
  int32_t max_len = 0;
  if (whisper_params.Has("max_len") && whisper_params.Get("max_len").IsNumber()) {
    max_len = whisper_params.Get("max_len").As<Napi::Number>();
  }
  
  // Add support for max_context
  int32_t max_context = -1;
  if (whisper_params.Has("max_context") && whisper_params.Get("max_context").IsNumber()) {
    max_context = whisper_params.Get("max_context").As<Napi::Number>();
  }
  
  // support prompt
  std::string prompt = "";
  if (whisper_params.Has("prompt") && whisper_params.Get("prompt").IsString()) {
    prompt = whisper_params.Get("prompt").As<Napi::String>();
  }
  
  // Add support for print_progress
  bool print_progress = false;
  if (whisper_params.Has("print_progress") && whisper_params.Get("print_progress").IsBoolean()) {
    print_progress = whisper_params.Get("print_progress").As<Napi::Boolean>();
  }
  // Add support for progress_callback
  Napi::Function progress_callback;
  if (whisper_params.Has("progress_callback") && whisper_params.Get("progress_callback").IsFunction()) {
    progress_callback = whisper_params.Get("progress_callback").As<Napi::Function>();
  }

  // Add support for VAD parameters
  bool vad = false;
  if (whisper_params.Has("vad") && whisper_params.Get("vad").IsBoolean()) {
    vad = whisper_params.Get("vad").As<Napi::Boolean>();
  }
  
  std::string vad_model = "";
  if (whisper_params.Has("vad_model") && whisper_params.Get("vad_model").IsString()) {
    vad_model = whisper_params.Get("vad_model").As<Napi::String>();
  }
  
  float vad_threshold = 0.5f;
  if (whisper_params.Has("vad_threshold") && whisper_params.Get("vad_threshold").IsNumber()) {
    vad_threshold = whisper_params.Get("vad_threshold").As<Napi::Number>();
  }
  
  int vad_min_speech_duration_ms = 250;
  if (whisper_params.Has("vad_min_speech_duration_ms") && whisper_params.Get("vad_min_speech_duration_ms").IsNumber()) {
    vad_min_speech_duration_ms = whisper_params.Get("vad_min_speech_duration_ms").As<Napi::Number>();
  }
  
  int vad_min_silence_duration_ms = 100;
  if (whisper_params.Has("vad_min_silence_duration_ms") && whisper_params.Get("vad_min_silence_duration_ms").IsNumber()) {
    vad_min_silence_duration_ms = whisper_params.Get("vad_min_silence_duration_ms").As<Napi::Number>();
  }
  
  float vad_max_speech_duration_s = FLT_MAX;
  if (whisper_params.Has("vad_max_speech_duration_s") && whisper_params.Get("vad_max_speech_duration_s").IsNumber()) {
    vad_max_speech_duration_s = whisper_params.Get("vad_max_speech_duration_s").As<Napi::Number>();
  }
  
  int vad_speech_pad_ms = 30;
  if (whisper_params.Has("vad_speech_pad_ms") && whisper_params.Get("vad_speech_pad_ms").IsNumber()) {
    vad_speech_pad_ms = whisper_params.Get("vad_speech_pad_ms").As<Napi::Number>();
  }
  
  float vad_samples_overlap = 0.1f;
  if (whisper_params.Has("vad_samples_overlap") && whisper_params.Get("vad_samples_overlap").IsNumber()) {
    vad_samples_overlap = whisper_params.Get("vad_samples_overlap").As<Napi::Number>();
  }

  Napi::Value pcmf32Value = whisper_params.Get("pcmf32");
  std::vector<float> pcmf32_vec;
  if (pcmf32Value.IsTypedArray()) {
    Napi::Float32Array pcmf32 = pcmf32Value.As<Napi::Float32Array>();
    size_t length = pcmf32.ElementLength();
    pcmf32_vec.reserve(length);
    for (size_t i = 0; i < length; i++) {
      pcmf32_vec.push_back(pcmf32[i]);
    }
  }

  params.language = language;
  params.model = model;
  params.fname_inp.emplace_back(input);
  params.use_gpu = use_gpu;
  params.flash_attn = flash_attn;
  params.no_prints = no_prints;
  params.no_timestamps = no_timestamps;
  params.audio_ctx = audio_ctx;
  params.pcmf32 = pcmf32_vec;
  params.comma_in_time = comma_in_time;
  params.max_len = max_len;
  params.max_context = max_context;
  params.print_progress = print_progress;
  params.prompt = prompt;
  params.detect_language = detect_language;

  // Set VAD parameters
  params.vad = vad;
  params.vad_model = vad_model;
  params.vad_threshold = vad_threshold;
  params.vad_min_speech_duration_ms = vad_min_speech_duration_ms;
  params.vad_min_silence_duration_ms = vad_min_silence_duration_ms;
  params.vad_max_speech_duration_s = vad_max_speech_duration_s;
  params.vad_speech_pad_ms = vad_speech_pad_ms;
  params.vad_samples_overlap = vad_samples_overlap;

  Napi::Function callback = info[1].As<Napi::Function>();
  // Create a new Worker class with progress callback support
  ProgressWorker* worker = new ProgressWorker(callback, params, progress_callback, env);
  worker->Queue();
  return env.Undefined();
}


Napi::Object Init(Napi::Env env, Napi::Object exports) {
  exports.Set(
      Napi::String::New(env, "whisper"),
      Napi::Function::New(env, whisper)
  );
  return exports;
}

NODE_API_MODULE(whisper, Init);
