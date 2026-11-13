#include "addon.h" // Your header file for WhisperStreamWrapper
#include "whisper-stream.h" // Your header file for the WhisperStream class

// NOTE: The N-API wrapper handles errors by throwing JS exceptions, so this macro is not needed.
// #define CHECK_STATUS(env, status, msg) ...

// --- Implementation of the Wrapper ---

Napi::Object WhisperStreamWrapper::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "WhisperStream", {
        InstanceMethod("startModel", &WhisperStreamWrapper::startModel),
        InstanceMethod("processChunk", &WhisperStreamWrapper::ProcessChunk),
        InstanceMethod("freeModel", &WhisperStreamWrapper::freeModel),
    });

    exports.Set("WhisperStream", func);
    return exports;
}

WhisperStreamWrapper::WhisperStreamWrapper(const Napi::CallbackInfo& info) : Napi::ObjectWrap<WhisperStreamWrapper>(info) {
}

Napi::Value WhisperStreamWrapper::startModel(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsObject()) {
        Napi::TypeError::New(env, "Expected a configuration object").ThrowAsJavaScriptException();
        return env.Null();
    }

    Napi::Object js_params = info[0].As<Napi::Object>();
    StreamParams params;

    if (js_params.Has("modelPath")) {
        params.model = js_params.Get("modelPath").As<Napi::String>();
    } else {
        Napi::TypeError::New(env, "Missing required parameter 'model'").ThrowAsJavaScriptException();
        return env.Null();
    }

    if (js_params.Has("language")) params.language = js_params.Get("language").As<Napi::String>();
    if (js_params.Has("nThreads")) params.n_threads = js_params.Get("nThreads").As<Napi::Number>();
    if (js_params.Has("stepMs")) params.step_ms = js_params.Get("stepMs").As<Napi::Number>();
    if (js_params.Has("lengthMs")) params.length_ms = js_params.Get("lengthMs").As<Napi::Number>();
    if (js_params.Has("keepMs")) params.keep_ms = js_params.Get("keepMs").As<Napi::Number>();
    if (js_params.Has("maxTokens")) params.max_tokens = js_params.Get("maxTokens").As<Napi::Number>();
    if (js_params.Has("audioCtx")) params.audio_ctx = js_params.Get("audioCtx").As<Napi::Number>();
    if (js_params.Has("vadThold")) params.vad_thold = js_params.Get("vadThold").As<Napi::Number>();
    if (js_params.Has("beamSize")) params.beam_size = js_params.Get("beamSize").As<Napi::Number>();
    if (js_params.Has("freqThold")) params.freq_thold = js_params.Get("freqThold").As<Napi::Number>();
    if (js_params.Has("translate")) params.translate = js_params.Get("translate").As<Napi::Boolean>();
    if (js_params.Has("noFallback")) params.no_fallback = js_params.Get("noFallback").As<Napi::Boolean>();
    if (js_params.Has("printSpecial")) params.print_special = js_params.Get("printSpecial").As<Napi::Boolean>();
    if (js_params.Has("noContext")) params.no_context = js_params.Get("noContext").As<Napi::Boolean>();
    if (js_params.Has("noTimestamps")) params.no_timestamps = js_params.Get("noTimestamps").As<Napi::Boolean>();
    if (js_params.Has("tinydiarize")) params.tinydiarize = js_params.Get("tinydiarize").As<Napi::Boolean>();
    if (js_params.Has("saveAudio")) params.save_audio = js_params.Get("saveAudio").As<Napi::Boolean>();
    if (js_params.Has("useGpu")) params.use_gpu = js_params.Get("useGpu").As<Napi::Boolean>();
    if (js_params.Has("flashAttn")) params.flash_attn = js_params.Get("flashAttn").As<Napi::Boolean>();

    if (this->whisperStream_) {
        delete this->whisperStream_;
    }

    try {
        this->whisperStream_ = new WhisperStream(params);
        this->whisperStream_->init();
    } catch (const std::runtime_error& e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
        return env.Null();
    }

    return env.Undefined();
}

Napi::Value WhisperStreamWrapper::ProcessChunk(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (!this->whisperStream_) {
        Napi::Error::New(env, "Model not started. Call startModel() first.").ThrowAsJavaScriptException();
        return env.Null();
    }

    if (info.Length() < 1 || !info[0].IsTypedArray() || info[0].As<Napi::TypedArray>().TypedArrayType() != napi_float32_array) {
        Napi::TypeError::New(env, "Argument must be a Float32Array").ThrowAsJavaScriptException();
        return env.Null();
    }

    Napi::Float32Array pcmf32_array = info[0].As<Napi::Float32Array>();
    std::vector<float> pcmf32_new(pcmf32_array.Data(), pcmf32_array.Data() + pcmf32_array.ElementLength());

    TranscriptionResult result = this->whisperStream_->process(pcmf32_new);

    Napi::Object resultObj = Napi::Object::New(env);
    resultObj.Set("text", Napi::String::New(env, result.text));
    resultObj.Set("isFinal", Napi::Boolean::New(env, result.final));

    return resultObj;
}

Napi::Value WhisperStreamWrapper::freeModel(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (this->whisperStream_) {
        delete this->whisperStream_;
        this->whisperStream_ = nullptr;
    }
    return env.Undefined();
}

Napi::Object InitAll(Napi::Env env, Napi::Object exports) {
    return WhisperStreamWrapper::Init(env, exports);
}

NODE_API_MODULE(whisper, InitAll)