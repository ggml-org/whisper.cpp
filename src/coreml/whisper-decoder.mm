#if !__has_feature(objc_arc)
#error This file must be compiled with automatic reference counting enabled (-fobjc-arc)
#endif

#import "whisper-decoder.h"

#import <CoreML/CoreML.h>

#if __has_include(<CoreML/MLModel+MLState.h>)
#import <CoreML/MLModel+MLState.h>
#define WHISPER_COREML_HAS_STATE 1
#else
#define WHISPER_COREML_HAS_STATE 0
#endif

#include <cstdio>
#include <cstring>
#include <cmath>

#if __cplusplus
extern "C" {
#endif

struct whisper_coreml_decoder_context {
    const void * model;
    const void * state;
    bool audio_ane_layout;
    bool stateful;
    bool uses_audio_input;
    int64_t state_pos;
    int64_t max_tokens;
};

struct whisper_coreml_decoder_context * whisper_coreml_decoder_init(const char * path_model) {
    NSString * path_model_str = [[NSString alloc] initWithUTF8String:path_model];
    NSURL * url_model = [NSURL fileURLWithPath:path_model_str];

    MLModelConfiguration * config = [[MLModelConfiguration alloc] init];
    config.computeUnits = MLComputeUnitsAll;

    NSError * error = nil;
    MLModel * model = [MLModel modelWithContentsOfURL:url_model configuration:config error:&error];
    if (model == nil) {
        if (error != nil) {
            fprintf(stderr, "%s: failed to load Core ML decoder: %s\n", __func__, error.localizedDescription.UTF8String);
        }
        return nullptr;
    }

    whisper_coreml_decoder_context * ctx = new whisper_coreml_decoder_context;
    ctx->model = CFBridgingRetain(model);
    ctx->state = nullptr;
    ctx->audio_ane_layout = false;
    ctx->stateful = false;
    ctx->uses_audio_input = false;
    ctx->state_pos = 0;
    ctx->max_tokens = 0;

    MLFeatureDescription * audioDescription = model.modelDescription.inputDescriptionsByName[@"audio_data"];
    if (audioDescription != nil) {
        ctx->uses_audio_input = true;
        NSArray<NSNumber *> * audioShape = audioDescription.multiArrayConstraint.shape;
        if (audioShape.count == 4) {
            ctx->audio_ane_layout = true;
        }
    }

#if WHISPER_COREML_HAS_STATE
    if (@available(macOS 15.0, *)) {
        if (model.modelDescription.stateDescriptionsByName.count > 0) {
            ctx->stateful = true;
            ctx->state = CFBridgingRetain([model newState]);

            for (NSString * name in model.modelDescription.stateDescriptionsByName) {
                MLFeatureDescription * desc = model.modelDescription.stateDescriptionsByName[name];
                NSArray<NSNumber *> * shape = desc.stateConstraint.bufferShape;
                if ([name hasPrefix:@"k_cache_"] && shape.count >= 2) {
                    ctx->max_tokens = shape[1].longLongValue;
                    break;
                }
            }
        }
    }
#endif

    if (!ctx->uses_audio_input && !ctx->stateful) {
        fprintf(stderr, "%s: Core ML decoder model has no audio input and no usable state; stateful decoders require macOS 15 SDK/runtime support\n", __func__);
        whisper_coreml_decoder_free(ctx);
        return nullptr;
    }

    return ctx;
}

void whisper_coreml_decoder_free(struct whisper_coreml_decoder_context * ctx) {
    if (ctx == nullptr) {
        return;
    }

    if (ctx->state != nullptr) {
        CFRelease(ctx->state);
    }
    CFRelease(ctx->model);
    delete ctx;
}

void whisper_coreml_decoder_reset(struct whisper_coreml_decoder_context * ctx) {
    if (ctx == nullptr || !ctx->stateful) {
        return;
    }

#if WHISPER_COREML_HAS_STATE
    if (@available(macOS 15.0, *)) {
        if (ctx->state != nullptr) {
            CFRelease(ctx->state);
        }
        ctx->state = CFBridgingRetain([(__bridge MLModel *) ctx->model newState]);
        ctx->state_pos = 0;
    }
#endif
}

bool whisper_coreml_decoder_is_stateful(const struct whisper_coreml_decoder_context * ctx) {
    return ctx != nullptr && ctx->stateful;
}

bool whisper_coreml_decoder_uses_audio_input(const struct whisper_coreml_decoder_context * ctx) {
    return ctx != nullptr && ctx->uses_audio_input;
}

int64_t whisper_coreml_decoder_state_pos(const struct whisper_coreml_decoder_context * ctx) {
    return ctx != nullptr ? ctx->state_pos : -1;
}

bool whisper_coreml_decoder_set_state_f16(struct whisper_coreml_decoder_context * ctx, const char * name, const void * data, int64_t n_elems) {
    if (ctx == nullptr || name == nullptr || data == nullptr || !ctx->stateful || ctx->state == nullptr) {
        return false;
    }

#if WHISPER_COREML_HAS_STATE
    if (@available(macOS 15.0, *)) {
        NSString * stateName = [[NSString alloc] initWithUTF8String:name];
        __block bool ok = false;

        [(__bridge MLState *) ctx->state getMultiArrayForStateNamed:stateName handler:^(MLMultiArray * buffer) {
            if (buffer.dataType == MLMultiArrayDataTypeFloat16 && buffer.count >= n_elems) {
                memcpy(buffer.dataPointer, data, n_elems*sizeof(uint16_t));
                ok = true;
            }
        }];

        return ok;
    }
#endif

    return false;
}

static MLMultiArray * whisper_coreml_make_token_array(const int32_t * tokens, const int64_t n_tokens, NSError ** error) {
    MLMultiArray * result = [[MLMultiArray alloc] initWithShape:@[@1, @(n_tokens)]
                                                       dataType:MLMultiArrayDataTypeInt32
                                                          error:error];
    if (result == nil) {
        return nil;
    }

    int32_t * data = (int32_t *) result.dataPointer;
    const int64_t stride = result.strides.count == 2 ? result.strides[1].longLongValue : 1;

    for (int64_t i = 0; i < n_tokens; ++i) {
        data[i*stride] = tokens[i];
    }

    return result;
}

static bool whisper_coreml_copy_last_logits(MLMultiArray * logits, const int64_t n_tokens, const int64_t n_vocab, float * out_logits) {
    if (logits == nil || logits.dataPointer == nullptr) {
        return false;
    }

    if (logits.dataType != MLMultiArrayDataTypeFloat32) {
        fprintf(stderr, "%s: expected Float32 logits, got MLMultiArray data type %ld\n", __func__, (long) logits.dataType);
        return false;
    }

    if (logits.shape.count < 2 || logits.shape.lastObject.longLongValue != n_vocab) {
        fprintf(stderr, "%s: unexpected logits shape\n", __func__);
        return false;
    }

    int64_t n_rows = 1;
    int64_t base = 0;
    if (logits.shape.count == 3) {
        n_rows = logits.shape[1].longLongValue;
        base = (n_tokens - 1) * logits.strides[1].longLongValue;
    } else if (logits.shape.count == 2) {
        n_rows = logits.shape[0].longLongValue;
        base = (n_tokens - 1) * logits.strides[0].longLongValue;
    }

    const int64_t vocab_stride = logits.strides.lastObject.longLongValue;
    const int64_t last_index = base + (n_vocab - 1)*vocab_stride;
    if (n_rows < n_tokens || base < 0 || last_index >= logits.count) {
        fprintf(stderr, "%s: logits output is smaller than the requested token row\n", __func__);
        return false;
    }

    const float * data = (const float *) logits.dataPointer;

    if (vocab_stride == 1) {
        memcpy(out_logits, data + base, n_vocab*sizeof(float));
    } else {
        for (int64_t i = 0; i < n_vocab; ++i) {
            out_logits[i] = data[base + i*vocab_stride];
        }
    }

    return true;
}

static MLMultiArray * whisper_coreml_make_audio_array(
        const whisper_coreml_decoder_context * ctx,
                                const float * audio,
                                      int64_t   n_audio_ctx,
                                      int64_t   n_audio_state,
                                   NSError ** error) {
    if (!ctx->uses_audio_input) {
        return nil;
    }

    if (!ctx->audio_ane_layout) {
        return [[MLMultiArray alloc] initWithDataPointer:(void *) audio
                                                   shape:@[@1, @(n_audio_ctx), @(n_audio_state)]
                                                dataType:MLMultiArrayDataTypeFloat32
                                                 strides:@[@(n_audio_ctx*n_audio_state), @(n_audio_state), @1]
                                             deallocator:nil
                                                   error:error];
    }

    MLMultiArray * result = [[MLMultiArray alloc] initWithShape:@[@1, @(n_audio_state), @1, @(n_audio_ctx)]
                                                       dataType:MLMultiArrayDataTypeFloat32
                                                          error:error];
    if (result == nil) {
        return nil;
    }

    float * dst = (float *) result.dataPointer;
    const int64_t stride_c = result.strides[1].longLongValue;
    const int64_t stride_w = result.strides[3].longLongValue;

    for (int64_t t = 0; t < n_audio_ctx; ++t) {
        for (int64_t c = 0; c < n_audio_state; ++c) {
            dst[c*stride_c + t*stride_w] = audio[t*n_audio_state + c];
        }
    }

    return result;
}

static MLMultiArray * whisper_coreml_make_i32_array(NSArray<NSNumber *> * shape, const int32_t * values, const int64_t n_values, NSError ** error) {
    MLMultiArray * result = [[MLMultiArray alloc] initWithShape:shape
                                                       dataType:MLMultiArrayDataTypeInt32
                                                          error:error];
    if (result == nil) {
        return nil;
    }

    int32_t * data = (int32_t *) result.dataPointer;
    for (int64_t i = 0; i < n_values; ++i) {
        data[i] = values[i];
    }

    return result;
}

static MLMultiArray * whisper_coreml_make_state_step_mask(const int64_t n_step, NSError ** error) {
    MLMultiArray * result = [[MLMultiArray alloc] initWithShape:@[@1, @1, @(n_step)]
                                                       dataType:MLMultiArrayDataTypeFloat32
                                                          error:error];
    if (result == nil) {
        return nil;
    }

    memset(result.dataPointer, 0, result.count*sizeof(float));
    return result;
}

static MLMultiArray * whisper_coreml_make_state_self_mask(const int64_t pos, const int64_t max_tokens, NSError ** error) {
    MLMultiArray * result = [[MLMultiArray alloc] initWithShape:@[@1, @1, @(max_tokens)]
                                                       dataType:MLMultiArrayDataTypeFloat32
                                                          error:error];
    if (result == nil) {
        return nil;
    }

    float * data = (float *) result.dataPointer;
    for (int64_t i = 0; i < max_tokens; ++i) {
        data[i] = i <= pos ? 0.0f : -INFINITY;
    }

    return result;
}

static bool whisper_coreml_decoder_predict(
        struct whisper_coreml_decoder_context * ctx,
                              MLMultiArray * tokenArray,
                              MLMultiArray * audioArray,
                              MLMultiArray * posArray,
                              MLMultiArray * stepMask,
                              MLMultiArray * selfMask,
                                   int64_t   n_tokens,
                                   int64_t   n_vocab,
                                    float * out_logits,
                                  NSError ** error) {
    NSMutableDictionary<NSString *, MLFeatureValue *> * features = [NSMutableDictionary dictionary];
    features[@"token_data"] = [MLFeatureValue featureValueWithMultiArray:tokenArray];
    if (ctx->uses_audio_input) {
        features[@"audio_data"] = [MLFeatureValue featureValueWithMultiArray:audioArray];
    }

    if (ctx->stateful) {
        features[@"pos_data"] = [MLFeatureValue featureValueWithMultiArray:posArray];
        features[@"step_mask"] = [MLFeatureValue featureValueWithMultiArray:stepMask];
        features[@"self_mask"] = [MLFeatureValue featureValueWithMultiArray:selfMask];
    }

    MLDictionaryFeatureProvider * input = [[MLDictionaryFeatureProvider alloc] initWithDictionary:features error:error];
    if (input == nil) {
        return false;
    }

    id<MLFeatureProvider> output = nil;
    if (ctx->stateful) {
#if WHISPER_COREML_HAS_STATE
        if (@available(macOS 15.0, *)) {
            output = [(__bridge MLModel *) ctx->model predictionFromFeatures:input usingState:(__bridge MLState *) ctx->state error:error];
        }
#endif
    } else {
        output = [(__bridge MLModel *) ctx->model predictionFromFeatures:input error:error];
    }

    if (output == nil) {
        return false;
    }

    MLFeatureValue * logitsValue = [output featureValueForName:@"logits"];
    if (logitsValue == nil) {
        for (NSString * name in output.featureNames) {
            logitsValue = [output featureValueForName:name];
            break;
        }
    }

    return whisper_coreml_copy_last_logits(logitsValue.multiArrayValue, n_tokens, n_vocab, out_logits);
}

bool whisper_coreml_decoder_decode(
        struct whisper_coreml_decoder_context * ctx,
                                     int64_t   n_tokens,
                                     int64_t   n_vocab,
                                     int64_t   n_audio_ctx,
                                     int64_t   n_audio_state,
                              const int32_t * tokens,
                                const float * audio,
                                      float * out_logits) {
    if (ctx == nullptr || tokens == nullptr || audio == nullptr || out_logits == nullptr || n_tokens <= 0) {
        return false;
    }

    @autoreleasepool {
        NSError * error = nil;

        MLMultiArray * tokenArray = whisper_coreml_make_token_array(tokens, n_tokens, &error);
        if (tokenArray == nil) {
            if (error != nil) {
                fprintf(stderr, "%s: failed to create token input: %s\n", __func__, error.localizedDescription.UTF8String);
            }
            return false;
        }

        MLMultiArray * audioArray = whisper_coreml_make_audio_array(ctx, audio, n_audio_ctx, n_audio_state, &error);
        if (ctx->uses_audio_input && audioArray == nil) {
            if (error != nil) {
                fprintf(stderr, "%s: failed to create audio input: %s\n", __func__, error.localizedDescription.UTF8String);
            }
            return false;
        }

        if (!ctx->stateful) {
            if (!whisper_coreml_decoder_predict(ctx, tokenArray, audioArray, nil, nil, nil, n_tokens, n_vocab, out_logits, &error)) {
                if (error != nil) {
                    fprintf(stderr, "%s: Core ML decoder prediction failed: %s\n", __func__, error.localizedDescription.UTF8String);
                }
                return false;
            }

            return true;
        }

        if (ctx->max_tokens <= 0 || ctx->state_pos + n_tokens > ctx->max_tokens) {
            fprintf(stderr, "%s: Core ML decoder state capacity exceeded\n", __func__);
            return false;
        }

        for (int64_t i = 0; i < n_tokens; ++i) {
            const int32_t token = tokens[i];
            const int32_t pos = (int32_t) ctx->state_pos;

            MLMultiArray * tokenStep = whisper_coreml_make_i32_array(@[@1, @1], &token, 1, &error);
            MLMultiArray * posStep = whisper_coreml_make_i32_array(@[@1], &pos, 1, &error);
            MLMultiArray * stepMask = whisper_coreml_make_state_step_mask(ctx->state_pos + 1, &error);
            MLMultiArray * selfMask = whisper_coreml_make_state_self_mask(ctx->state_pos, ctx->max_tokens, &error);

            if (tokenStep == nil || posStep == nil || stepMask == nil || selfMask == nil) {
                if (error != nil) {
                    fprintf(stderr, "%s: failed to create stateful decoder inputs: %s\n", __func__, error.localizedDescription.UTF8String);
                }
                return false;
            }

            if (!whisper_coreml_decoder_predict(ctx, tokenStep, audioArray, posStep, stepMask, selfMask, 1, n_vocab, out_logits, &error)) {
                if (error != nil) {
                    fprintf(stderr, "%s: Core ML decoder prediction failed: %s\n", __func__, error.localizedDescription.UTF8String);
                }
                return false;
            }

            ctx->state_pos++;
        }

        return true;
    }
}

#if __cplusplus
}
#endif
