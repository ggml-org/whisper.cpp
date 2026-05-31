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

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if __cplusplus
extern "C" {
#endif

struct whisper_coreml_decoder_shard {
    const void * model;
    const void * state;
    std::vector<const void *> cross_inputs;
    int64_t start_layer;
    int64_t n_layers;
    int64_t n_state;
    int64_t n_audio_ctx;
    bool token_input;
    bool output_logits;
    bool cross_kv_input;
};

struct whisper_coreml_decoder_context {
    const void * model;
    const char * compute_units_name;
    int64_t state_pos;
    int64_t max_tokens;
    std::vector<whisper_coreml_decoder_shard> shards;
};

static bool whisper_coreml_decoder_compute_units(MLComputeUnits * units, const char ** name) {
    if (@available(macOS 13.0, iOS 16.0, watchOS 9.0, tvOS 16.0, *)) {
        *units = MLComputeUnitsCPUAndNeuralEngine;
        *name = "cpu_ne";
        return true;
    }

    fprintf(stderr, "%s: Core ML decoder requires CPU and Neural Engine compute units\n", __func__);
    return false;
}

static MLModel * whisper_coreml_decoder_load_model_url(NSURL * url_model, MLModelConfiguration * config, NSError ** error) {
    MLModel * model = [MLModel modelWithContentsOfURL:url_model configuration:config error:error];
    if (model != nil) {
        return model;
    }

    NSString * ext = url_model.pathExtension.lowercaseString;
    if (![ext isEqualToString:@"mlmodelc"]) {
        NSError * compileError = nil;
        NSURL * compiledURL = [MLModel compileModelAtURL:url_model error:&compileError];
        if (compiledURL != nil) {
            *error = nil;
            model = [MLModel modelWithContentsOfURL:compiledURL configuration:config error:error];
            if (model != nil) {
                return model;
            }
        } else if (compileError != nil) {
            *error = compileError;
        }
    }

    return nil;
}

static bool whisper_coreml_decoder_model_has_input(MLModel * model, NSString * name) {
    return model.modelDescription.inputDescriptionsByName[name] != nil;
}

static bool whisper_coreml_decoder_model_has_output(MLModel * model, NSString * name) {
    return model.modelDescription.outputDescriptionsByName[name] != nil;
}

static bool whisper_coreml_decoder_parse_layer_suffix(NSString * name, NSString * prefix, int64_t * layer) {
    if (![name hasPrefix:prefix]) {
        return false;
    }

    NSString * suffix = [name substringFromIndex:prefix.length];
    if (suffix.length == 0) {
        return false;
    }

    *layer = suffix.longLongValue;
    return true;
}

static bool whisper_coreml_decoder_check_ane_kv_shape(NSArray<NSNumber *> * shape, const int64_t n_state, int64_t * n_ctx) {
    if (shape.count != 4 ||
            shape[0].longLongValue != 1 ||
            shape[1].longLongValue != n_state ||
            shape[2].longLongValue != 1 ||
            shape[3].longLongValue <= 0) {
        return false;
    }

    if (n_ctx != nullptr) {
        if (*n_ctx == 0) {
            *n_ctx = shape[3].longLongValue;
        } else if (*n_ctx != shape[3].longLongValue) {
            return false;
        }
    }

    return true;
}

static bool whisper_coreml_decoder_parse_no_write_shard(
        MLModel * model,
        whisper_coreml_decoder_shard * shard,
        int64_t * max_tokens) {
    if (!whisper_coreml_decoder_model_has_input(model, @"slot_mask")) {
        return false;
    }

    const bool token_input = whisper_coreml_decoder_model_has_input(model, @"token_data");
    const bool x_input = whisper_coreml_decoder_model_has_input(model, @"x_data");
    if (token_input == x_input) {
        return false;
    }

    int64_t min_layer = INT64_MAX;
    int64_t max_layer = -1;
    int64_t n_layers = 0;
    for (NSString * name in model.modelDescription.outputDescriptionsByName) {
        int64_t layer = -1;
        if (whisper_coreml_decoder_parse_layer_suffix(name, @"k_out_", &layer)) {
            min_layer = std::min(min_layer, layer);
            max_layer = std::max(max_layer, layer);
            n_layers++;
        }
    }

    if (min_layer == INT64_MAX || max_layer < min_layer || n_layers != max_layer - min_layer + 1) {
        return false;
    }

    const bool output_logits = whisper_coreml_decoder_model_has_output(model, @"logits");
    const bool output_hidden = whisper_coreml_decoder_model_has_output(model, @"x_out");
    if (output_logits == output_hidden) {
        return false;
    }

    if (shard != nullptr) {
        shard->model = nullptr;
        shard->state = nullptr;
        shard->cross_inputs.clear();
        shard->start_layer = min_layer;
        shard->n_layers = n_layers;
        shard->n_state = 0;
        shard->n_audio_ctx = 0;
        shard->token_input = token_input;
        shard->output_logits = output_logits;
        shard->cross_kv_input = false;
    }

#if WHISPER_COREML_HAS_STATE
    if (@available(macOS 15.0, *)) {
        NSString * stateName = [[NSString alloc] initWithFormat:@"k_cache_%lld", (long long) min_layer];
        MLFeatureDescription * desc = model.modelDescription.stateDescriptionsByName[stateName];
        if (desc == nil || desc.stateConstraint.dataType != MLMultiArrayDataTypeFloat16) {
            return false;
        }

        NSArray<NSNumber *> * shape = desc.stateConstraint.bufferShape;
        if (shape.count != 4 ||
                shape[0].longLongValue != 1 ||
                shape[2].longLongValue != 1 ||
                shape[1].longLongValue <= 0 ||
                shape[3].longLongValue <= 0) {
            return false;
        }

        const int64_t n_state = shape[1].longLongValue;
        const int64_t n_self_ctx = shape[3].longLongValue;
        bool cross_source_known = false;
        bool cross_kv_input = false;
        int64_t n_audio_ctx = 0;

        for (int64_t layer = min_layer; layer <= max_layer; ++layer) {
            NSString * kOut = [[NSString alloc] initWithFormat:@"k_out_%lld", (long long) layer];
            NSString * vOut = [[NSString alloc] initWithFormat:@"v_out_%lld", (long long) layer];
            if (!whisper_coreml_decoder_model_has_output(model, kOut) ||
                    !whisper_coreml_decoder_model_has_output(model, vOut)) {
                return false;
            }

            for (NSString * prefix in @[@"k_cache", @"v_cache"]) {
                NSString * checkName = [[NSString alloc] initWithFormat:@"%@_%lld", prefix, (long long) layer];
                MLFeatureDescription * checkDesc = model.modelDescription.stateDescriptionsByName[checkName];
                if (checkDesc == nil || checkDesc.stateConstraint.dataType != MLMultiArrayDataTypeFloat16) {
                    return false;
                }

                int64_t self_ctx = n_self_ctx;
                if (!whisper_coreml_decoder_check_ane_kv_shape(checkDesc.stateConstraint.bufferShape, n_state, &self_ctx)) {
                    return false;
                }
            }

            NSString * crossKName = [[NSString alloc] initWithFormat:@"cross_k_%lld", (long long) layer];
            NSString * crossVName = [[NSString alloc] initWithFormat:@"cross_v_%lld", (long long) layer];
            MLFeatureDescription * crossKState = model.modelDescription.stateDescriptionsByName[crossKName];
            MLFeatureDescription * crossVState = model.modelDescription.stateDescriptionsByName[crossVName];
            MLFeatureDescription * crossKInput = model.modelDescription.inputDescriptionsByName[crossKName];
            MLFeatureDescription * crossVInput = model.modelDescription.inputDescriptionsByName[crossVName];
            const bool has_cross_state = crossKState != nil || crossVState != nil;
            const bool has_cross_input = crossKInput != nil || crossVInput != nil;

            if (has_cross_state == has_cross_input ||
                    (crossKState == nil) != (crossVState == nil) ||
                    (crossKInput == nil) != (crossVInput == nil)) {
                return false;
            }

            if (!cross_source_known) {
                cross_source_known = true;
                cross_kv_input = has_cross_input;
            } else if (cross_kv_input != has_cross_input) {
                return false;
            }

            if (!has_cross_input) {
                return false;
            }

            if (crossKInput.multiArrayConstraint.dataType != MLMultiArrayDataTypeFloat16 ||
                    crossVInput.multiArrayConstraint.dataType != MLMultiArrayDataTypeFloat16 ||
                    !whisper_coreml_decoder_check_ane_kv_shape(crossKInput.multiArrayConstraint.shape, n_state, &n_audio_ctx) ||
                    !whisper_coreml_decoder_check_ane_kv_shape(crossVInput.multiArrayConstraint.shape, n_state, &n_audio_ctx)) {
                return false;
            }
        }

        if (!cross_kv_input) {
            return false;
        }

        if (shard != nullptr) {
            shard->n_state = n_state;
            shard->n_audio_ctx = n_audio_ctx;
            shard->cross_kv_input = true;
        }
        if (max_tokens != nullptr) {
            *max_tokens = n_self_ctx;
        }
    } else {
        return false;
    }
#else
    return false;
#endif

    return true;
}

static std::string whisper_coreml_decoder_no_write_shard_path(
        const std::string & first_path,
        const int64_t       start_layer,
        const int64_t       n_layers,
        const bool          token_input,
        const bool          logits) {
    const std::string marker = "-no-write-s";
    const size_t marker_pos = first_path.rfind(marker);
    if (marker_pos == std::string::npos) {
        return "";
    }

    size_t ext_pos = first_path.rfind('.');
    if (ext_pos == std::string::npos || ext_pos < marker_pos) {
        ext_pos = first_path.size();
    }

    std::string suffix = "-no-write-s" + std::to_string(start_layer) + "-l" + std::to_string(n_layers);
    if (token_input) {
        suffix += "-token";
    }
    if (logits) {
        suffix += "-logits";
    }

    return first_path.substr(0, marker_pos) + suffix + first_path.substr(ext_pos);
}

static bool whisper_coreml_decoder_path_exists(const std::string & path) {
    if (path.empty()) {
        return false;
    }

    NSString * nsPath = [[NSString alloc] initWithUTF8String:path.c_str()];
    return [[NSFileManager defaultManager] fileExistsAtPath:nsPath];
}

static void whisper_coreml_decoder_zero_f16_multiarray(const void * ptr) {
    if (ptr == nullptr) {
        return;
    }

    MLMultiArray * array = (__bridge MLMultiArray *) ptr;
    if (array.dataPointer != nullptr && array.dataType == MLMultiArrayDataTypeFloat16 && array.count > 0) {
        memset(array.dataPointer, 0, array.count*sizeof(uint16_t));
    }
}

static bool whisper_coreml_decoder_add_no_write_shard(
        whisper_coreml_decoder_context * ctx,
        MLModel * model,
        const whisper_coreml_decoder_shard & parsed) {
#if WHISPER_COREML_HAS_STATE
    if (@available(macOS 15.0, *)) {
        MLState * state = [model newState];
        if (state == nil) {
            return false;
        }

        whisper_coreml_decoder_shard shard = parsed;
        shard.model = CFBridgingRetain(model);
        shard.state = CFBridgingRetain(state);

        for (int64_t i = 0; i < shard.n_layers*2; ++i) {
            NSError * error = nil;
            MLMultiArray * input = [[MLMultiArray alloc] initWithShape:@[@1, @(shard.n_state), @1, @(shard.n_audio_ctx)]
                                                              dataType:MLMultiArrayDataTypeFloat16
                                                                 error:&error];
            if (input == nil) {
                for (const void * ptr : shard.cross_inputs) {
                    CFRelease(ptr);
                }
                CFRelease(shard.state);
                CFRelease(shard.model);
                return false;
            }

            shard.cross_inputs.push_back(CFBridgingRetain(input));
            whisper_coreml_decoder_zero_f16_multiarray(shard.cross_inputs.back());
        }

        ctx->shards.push_back(shard);
        return true;
    }
#endif

    return false;
}

static bool whisper_coreml_decoder_load_no_write_shards(
        whisper_coreml_decoder_context * ctx,
        MLModel * first_model,
        const std::string & first_path,
        MLModelConfiguration * config) {
    whisper_coreml_decoder_shard first;
    int64_t max_tokens = 0;
    if (!whisper_coreml_decoder_parse_no_write_shard(first_model, &first, &max_tokens)) {
        fprintf(stderr, "%s: first Core ML decoder shard has an incompatible interface\n", __func__);
        return false;
    }

    if (!first.token_input || first.start_layer != 0 || max_tokens <= 0) {
        fprintf(stderr, "%s: first Core ML decoder shard must start at layer 0 and consume token input\n", __func__);
        return false;
    }

    ctx->max_tokens = max_tokens;

    if (!whisper_coreml_decoder_add_no_write_shard(ctx, first_model, first)) {
        fprintf(stderr, "%s: failed to create Core ML state for first decoder shard\n", __func__);
        return false;
    }

    if (first.output_logits) {
        return true;
    }

    const int64_t max_layers_per_shard = first.n_layers;
    for (int64_t start = first.start_layer + first.n_layers; start < 512; ) {
        std::string shard_path;
        bool logits = false;
        int64_t n_layers = 0;
        int found = 0;

        for (int64_t candidate_layers = max_layers_per_shard; candidate_layers > 0; --candidate_layers) {
            const std::string hidden_path = whisper_coreml_decoder_no_write_shard_path(first_path, start, candidate_layers, false, false);
            if (whisper_coreml_decoder_path_exists(hidden_path)) {
                shard_path = hidden_path;
                logits = false;
                n_layers = candidate_layers;
                found++;
            }

            const std::string logits_path = whisper_coreml_decoder_no_write_shard_path(first_path, start, candidate_layers, false, true);
            if (whisper_coreml_decoder_path_exists(logits_path)) {
                shard_path = logits_path;
                logits = true;
                n_layers = candidate_layers;
                found++;
            }
        }

        if (found > 1) {
            fprintf(stderr, "%s: ambiguous Core ML decoder shard for start layer %lld\n", __func__, (long long) start);
            return false;
        }

        if (found == 0) {
            fprintf(stderr, "%s: missing Core ML decoder shard for start layer %lld\n", __func__, (long long) start);
            return false;
        }

        NSString * path = [[NSString alloc] initWithUTF8String:shard_path.c_str()];
        NSURL * url = [NSURL fileURLWithPath:path];
        NSError * error = nil;
        MLModel * model = whisper_coreml_decoder_load_model_url(url, config, &error);
        if (model == nil) {
            if (error != nil) {
                fprintf(stderr, "%s: failed to load Core ML decoder shard '%s': %s\n", __func__, shard_path.c_str(), error.localizedDescription.UTF8String);
            }
            return false;
        }

        whisper_coreml_decoder_shard parsed;
        int64_t shard_max_tokens = 0;
        if (!whisper_coreml_decoder_parse_no_write_shard(model, &parsed, &shard_max_tokens) ||
                parsed.start_layer != start ||
                parsed.n_layers != n_layers ||
                parsed.token_input ||
                parsed.output_logits != logits ||
                parsed.n_state != first.n_state ||
                parsed.n_audio_ctx != first.n_audio_ctx ||
                shard_max_tokens != ctx->max_tokens) {
            fprintf(stderr, "%s: Core ML decoder shard '%s' has an incompatible interface\n", __func__, shard_path.c_str());
            return false;
        }

        if (!whisper_coreml_decoder_add_no_write_shard(ctx, model, parsed)) {
            fprintf(stderr, "%s: failed to create Core ML state for decoder shard '%s'\n", __func__, shard_path.c_str());
            return false;
        }

        if (parsed.output_logits) {
            return true;
        }

        start += parsed.n_layers;
    }

    fprintf(stderr, "%s: Core ML decoder shard sequence has no logits shard\n", __func__);
    return false;
}

struct whisper_coreml_decoder_context * whisper_coreml_decoder_init(const char * path_model) {
    if (path_model == nullptr || path_model[0] == '\0') {
        return nullptr;
    }

    MLComputeUnits compute_units = MLComputeUnitsAll;
    const char * compute_units_name = nullptr;
    if (!whisper_coreml_decoder_compute_units(&compute_units, &compute_units_name)) {
        return nullptr;
    }

    NSString * path_model_str = [[NSString alloc] initWithUTF8String:path_model];
    NSURL * url_model = [NSURL fileURLWithPath:path_model_str];

    MLModelConfiguration * config = [[MLModelConfiguration alloc] init];
    config.computeUnits = compute_units;

    NSError * error = nil;
    MLModel * model = whisper_coreml_decoder_load_model_url(url_model, config, &error);
    if (model == nil) {
        if (error != nil) {
            fprintf(stderr, "%s: failed to load Core ML decoder: %s\n", __func__, error.localizedDescription.UTF8String);
        }
        return nullptr;
    }

    whisper_coreml_decoder_context * ctx = new whisper_coreml_decoder_context;
    ctx->model = CFBridgingRetain(model);
    ctx->compute_units_name = compute_units_name;
    ctx->state_pos = 0;
    ctx->max_tokens = 0;

    if (!whisper_coreml_decoder_load_no_write_shards(ctx, model, path_model, config)) {
        whisper_coreml_decoder_free(ctx);
        return nullptr;
    }

    return ctx;
}

void whisper_coreml_decoder_free(struct whisper_coreml_decoder_context * ctx) {
    if (ctx == nullptr) {
        return;
    }

    for (auto & shard : ctx->shards) {
        for (const void * ptr : shard.cross_inputs) {
            if (ptr != nullptr) {
                CFRelease(ptr);
            }
        }
        if (shard.state != nullptr) {
            CFRelease(shard.state);
        }
        if (shard.model != nullptr) {
            CFRelease(shard.model);
        }
    }

    if (ctx->model != nullptr) {
        CFRelease(ctx->model);
    }
    delete ctx;
}

void whisper_coreml_decoder_reset(struct whisper_coreml_decoder_context * ctx) {
    if (ctx == nullptr) {
        return;
    }

#if WHISPER_COREML_HAS_STATE
    if (@available(macOS 15.0, *)) {
        for (auto & shard : ctx->shards) {
            for (const void * ptr : shard.cross_inputs) {
                whisper_coreml_decoder_zero_f16_multiarray(ptr);
            }

            if (shard.state != nullptr) {
                CFRelease(shard.state);
                shard.state = nullptr;
            }

            MLState * state = [(__bridge MLModel *) shard.model newState];
            shard.state = CFBridgingRetain(state);
        }

        ctx->state_pos = 0;
    }
#endif
}

const char * whisper_coreml_decoder_compute_units_name(const struct whisper_coreml_decoder_context * ctx) {
    return ctx != nullptr && ctx->compute_units_name != nullptr ? ctx->compute_units_name : "cpu_ne";
}

int64_t whisper_coreml_decoder_state_pos(const struct whisper_coreml_decoder_context * ctx) {
    return ctx != nullptr ? ctx->state_pos : -1;
}

static bool whisper_coreml_decoder_copy_f16_to_multiarray(MLMultiArray * buffer, const void * data, const int64_t n_elems) {
    if (buffer == nil ||
            buffer.dataPointer == nullptr ||
            data == nullptr ||
            n_elems <= 0 ||
            buffer.dataType != MLMultiArrayDataTypeFloat16 ||
            buffer.count < n_elems) {
        return false;
    }

    if (buffer.shape.count == 4 && buffer.shape[2].longLongValue == 1) {
        const int64_t n_state = buffer.shape[1].longLongValue;
        const int64_t n_ctx = buffer.shape[3].longLongValue;
        if (n_state <= 0 || n_elems % n_state != 0 || n_elems/n_state > n_ctx) {
            return false;
        }

        const uint16_t * src = (const uint16_t *) data;
        uint16_t * dst = (uint16_t *) buffer.dataPointer;
        const int64_t src_n_ctx = n_elems/n_state;
        const int64_t stride_c = buffer.strides[1].longLongValue;
        const int64_t stride_t = buffer.strides[3].longLongValue;
        for (int64_t t = 0; t < src_n_ctx; ++t) {
            for (int64_t c = 0; c < n_state; ++c) {
                dst[c*stride_c + t*stride_t] = src[t*n_state + c];
            }
        }
        return true;
    }

    memcpy(buffer.dataPointer, data, n_elems*sizeof(uint16_t));
    return true;
}

bool whisper_coreml_decoder_set_state_f16(struct whisper_coreml_decoder_context * ctx, const char * name, const void * data, int64_t n_elems) {
    if (ctx == nullptr || name == nullptr || data == nullptr || n_elems <= 0) {
        return false;
    }

    NSString * stateName = [[NSString alloc] initWithUTF8String:name];
    int64_t layer = -1;
    bool is_v = false;
    if (!whisper_coreml_decoder_parse_layer_suffix(stateName, @"cross_k_", &layer)) {
        if (!whisper_coreml_decoder_parse_layer_suffix(stateName, @"cross_v_", &layer)) {
            return false;
        }
        is_v = true;
    }

    for (const auto & shard : ctx->shards) {
        if (!shard.cross_kv_input || layer < shard.start_layer || layer >= shard.start_layer + shard.n_layers) {
            continue;
        }

        const int64_t rel = layer - shard.start_layer;
        const int64_t index = rel*2 + (is_v ? 1 : 0);
        if (index < 0 || index >= (int64_t) shard.cross_inputs.size()) {
            return false;
        }

        if (n_elems != shard.n_state*shard.n_audio_ctx) {
            return false;
        }

        return whisper_coreml_decoder_copy_f16_to_multiarray((__bridge MLMultiArray *) shard.cross_inputs[index], data, n_elems);
    }

    return false;
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

static MLMultiArray * whisper_coreml_make_ane_slot_mask(const int64_t pos, const int64_t max_tokens, NSError ** error) {
    MLMultiArray * result = [[MLMultiArray alloc] initWithShape:@[@1, @1, @1, @(max_tokens)]
                                                       dataType:MLMultiArrayDataTypeFloat32
                                                          error:error];
    if (result == nil) {
        return nil;
    }

    memset(result.dataPointer, 0, result.count*sizeof(float));
    if (pos >= 0 && pos < max_tokens) {
        float * data = (float *) result.dataPointer;
        const int64_t stride_t = result.strides[3].longLongValue;
        data[pos*stride_t] = 1.0f;
    }

    return result;
}

static MLMultiArray * whisper_coreml_make_ane_self_mask(const int64_t pos, const int64_t max_tokens, NSError ** error) {
    MLMultiArray * result = [[MLMultiArray alloc] initWithShape:@[@1, @(max_tokens), @1, @1]
                                                       dataType:MLMultiArrayDataTypeFloat32
                                                          error:error];
    if (result == nil) {
        return nil;
    }

    float * data = (float *) result.dataPointer;
    const int64_t stride_t = result.strides[1].longLongValue;
    for (int64_t i = 0; i < max_tokens; ++i) {
        data[i*stride_t] = i <= pos ? 0.0f : -1.0e4f;
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
        base = (n_tokens - 1)*logits.strides[1].longLongValue;
    } else if (logits.shape.count == 2) {
        n_rows = logits.shape[0].longLongValue;
        base = (n_tokens - 1)*logits.strides[0].longLongValue;
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

static bool whisper_coreml_decoder_write_state_slot_f16(
        const void * statePtr,
        const char * name,
        MLMultiArray * value,
        const int64_t pos) {
    if (statePtr == nullptr ||
            name == nullptr ||
            value == nil ||
            value.dataPointer == nullptr ||
            value.dataType != MLMultiArrayDataTypeFloat16 ||
            pos < 0) {
        return false;
    }

#if WHISPER_COREML_HAS_STATE
    if (@available(macOS 15.0, *)) {
        NSString * stateName = [[NSString alloc] initWithUTF8String:name];
        __block bool ok = false;
        [(__bridge MLState *) statePtr getMultiArrayForStateNamed:stateName handler:^(MLMultiArray * buffer) {
            if (buffer.dataType != MLMultiArrayDataTypeFloat16 ||
                    buffer.shape.count != 4 ||
                    buffer.shape[2].longLongValue != 1 ||
                    pos >= buffer.shape[3].longLongValue) {
                return;
            }

            const int64_t n_state = buffer.shape[1].longLongValue;
            if (value.shape.count != 4 ||
                    value.shape[1].longLongValue != n_state ||
                    value.shape[2].longLongValue != 1 ||
                    value.shape[3].longLongValue != 1) {
                return;
            }

            const uint16_t * src = (const uint16_t *) value.dataPointer;
            uint16_t * dst = (uint16_t *) buffer.dataPointer;
            const int64_t src_stride_c = value.strides[1].longLongValue;
            const int64_t dst_stride_c = buffer.strides[1].longLongValue;
            const int64_t dst_stride_t = buffer.strides[3].longLongValue;
            for (int64_t c = 0; c < n_state; ++c) {
                dst[c*dst_stride_c + pos*dst_stride_t] = src[c*src_stride_c];
            }
            ok = true;
        }];

        return ok;
    }
#endif

    return false;
}

static bool whisper_coreml_decoder_write_no_write_outputs(
        const whisper_coreml_decoder_shard & shard,
        id<MLFeatureProvider> output,
        const int64_t pos) {
    if (output == nil || shard.state == nullptr) {
        return false;
    }

    for (int64_t i = 0; i < shard.n_layers; ++i) {
        const int64_t il = shard.start_layer + i;
        NSString * kName = [[NSString alloc] initWithFormat:@"k_out_%lld", (long long) il];
        NSString * vName = [[NSString alloc] initWithFormat:@"v_out_%lld", (long long) il];
        NSString * kState = [[NSString alloc] initWithFormat:@"k_cache_%lld", (long long) il];
        NSString * vState = [[NSString alloc] initWithFormat:@"v_cache_%lld", (long long) il];

        MLFeatureValue * kValue = [output featureValueForName:kName];
        MLFeatureValue * vValue = [output featureValueForName:vName];
        if (kValue == nil ||
                vValue == nil ||
                !whisper_coreml_decoder_write_state_slot_f16(shard.state, kState.UTF8String, kValue.multiArrayValue, pos) ||
                !whisper_coreml_decoder_write_state_slot_f16(shard.state, vState.UTF8String, vValue.multiArrayValue, pos)) {
            return false;
        }
    }

    return true;
}

static bool whisper_coreml_decoder_decode_no_write(
        struct whisper_coreml_decoder_context * ctx,
                                     int64_t   n_tokens,
                                     int64_t   n_vocab,
                              const int32_t * tokens,
                                      float * out_logits,
                                        bool   is_prompt,
                                  NSError ** error) {
    if (ctx == nullptr ||
            tokens == nullptr ||
            out_logits == nullptr ||
            ctx->shards.empty()) {
        return false;
    }

    for (int64_t i = 0; i < n_tokens; ++i) {
        const int32_t token = tokens[i];
        const int32_t pos = (int32_t) ctx->state_pos;

        MLMultiArray * tokenStep = whisper_coreml_make_i32_array(@[@1, @1], &token, 1, error);
        MLMultiArray * posStep = whisper_coreml_make_i32_array(@[@1], &pos, 1, error);
        MLMultiArray * slotMask = whisper_coreml_make_ane_slot_mask(ctx->state_pos, ctx->max_tokens, error);
        MLMultiArray * selfMask = whisper_coreml_make_ane_self_mask(ctx->state_pos, ctx->max_tokens, error);
        if (tokenStep == nil || posStep == nil || slotMask == nil || selfMask == nil) {
            return false;
        }

        MLMultiArray * xArray = nil;
        for (const auto & shard : ctx->shards) {
            NSMutableDictionary<NSString *, MLFeatureValue *> * features = [NSMutableDictionary dictionary];
            if (shard.token_input) {
                features[@"token_data"] = [MLFeatureValue featureValueWithMultiArray:tokenStep];
                features[@"pos_data"] = [MLFeatureValue featureValueWithMultiArray:posStep];
            } else {
                if (xArray == nil) {
                    return false;
                }
                features[@"x_data"] = [MLFeatureValue featureValueWithMultiArray:xArray];
            }

            features[@"slot_mask"] = [MLFeatureValue featureValueWithMultiArray:slotMask];
            features[@"self_mask"] = [MLFeatureValue featureValueWithMultiArray:selfMask];

            if ((int64_t) shard.cross_inputs.size() != shard.n_layers*2) {
                return false;
            }

            for (int64_t i_layer = 0; i_layer < shard.n_layers; ++i_layer) {
                const int64_t il = shard.start_layer + i_layer;
                NSString * kName = [[NSString alloc] initWithFormat:@"cross_k_%lld", (long long) il];
                NSString * vName = [[NSString alloc] initWithFormat:@"cross_v_%lld", (long long) il];
                features[kName] = [MLFeatureValue featureValueWithMultiArray:(__bridge MLMultiArray *) shard.cross_inputs[2*i_layer]];
                features[vName] = [MLFeatureValue featureValueWithMultiArray:(__bridge MLMultiArray *) shard.cross_inputs[2*i_layer + 1]];
            }

            MLDictionaryFeatureProvider * input = [[MLDictionaryFeatureProvider alloc] initWithDictionary:features error:error];
            if (input == nil) {
                return false;
            }

            id<MLFeatureProvider> output = nil;
#if WHISPER_COREML_HAS_STATE
            if (@available(macOS 15.0, *)) {
                output = [(__bridge MLModel *) shard.model predictionFromFeatures:input usingState:(__bridge MLState *) shard.state error:error];
            }
#endif
            if (output == nil) {
                return false;
            }

            if (!whisper_coreml_decoder_write_no_write_outputs(shard, output, ctx->state_pos)) {
                return false;
            }

            if (shard.output_logits) {
                MLFeatureValue * logitsValue = [output featureValueForName:@"logits"];
                if (logitsValue == nil || !whisper_coreml_copy_last_logits(logitsValue.multiArrayValue, 1, n_vocab, out_logits)) {
                    return false;
                }
            } else {
                MLFeatureValue * xValue = [output featureValueForName:@"x_out"];
                if (xValue == nil || xValue.multiArrayValue == nil) {
                    return false;
                }
                xArray = xValue.multiArrayValue;
            }
        }

        ctx->state_pos++;
    }

    (void) is_prompt;
    return true;
}

bool whisper_coreml_decoder_decode(
        struct whisper_coreml_decoder_context * ctx,
                                     int64_t   n_tokens,
                                     int64_t   n_vocab,
                                     int64_t   n_audio_ctx,
                                     int64_t   n_audio_state,
                              const int32_t * tokens,
                                const float * audio,
                                      float * out_logits,
                                        bool   is_prompt) {
    if (ctx == nullptr || tokens == nullptr || out_logits == nullptr || n_tokens <= 0) {
        return false;
    }

    (void) n_audio_ctx;
    (void) n_audio_state;
    (void) audio;

    @autoreleasepool {
        NSError * error = nil;

        if (ctx->max_tokens <= 0 || ctx->state_pos + n_tokens > ctx->max_tokens) {
            fprintf(stderr, "%s: Core ML decoder state capacity exceeded\n", __func__);
            return false;
        }

        if (!whisper_coreml_decoder_decode_no_write(ctx, n_tokens, n_vocab, tokens, out_logits, is_prompt, &error)) {
            if (error != nil) {
                fprintf(stderr, "%s: Core ML decoder prediction failed: %s\n", __func__, error.localizedDescription.UTF8String);
            }
            return false;
        }

        return true;
    }
}

#if __cplusplus
}
#endif
