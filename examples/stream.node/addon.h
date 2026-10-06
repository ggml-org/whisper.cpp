#pragma once

#include <napi.h>
#include "whisper-stream.h"

class WhisperStreamWrapper : public Napi::ObjectWrap<WhisperStreamWrapper> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    WhisperStreamWrapper(const Napi::CallbackInfo& info);

private:
    Napi::Value startModel(const Napi::CallbackInfo& info);
    Napi::Value ProcessChunk(const Napi::CallbackInfo& info);
    Napi::Value freeModel(const Napi::CallbackInfo& info);

    WhisperStream* whisperStream_ = nullptr;
};