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
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

#if __cplusplus
extern "C" {
#endif

struct whisper_coreml_decoder_shard {
    const void * model;
    const void * state;
    int64_t start_layer;
    int64_t n_layers;
    int64_t n_state;
    bool token_input;
    bool output_logits;
};

struct whisper_coreml_decoder_context {
    const void * model;
    const void * model_prefill;
    const void * state;
    MLComputeUnits compute_units;
    const char * compute_units_name;
    bool audio_ane_layout;
    bool stateful;
    bool no_write_sharded;
    bool uses_audio_input;
    bool prewarm_state;
    bool reuse_state;
    int64_t state_pos;
    int64_t max_tokens;
    int64_t prefill_tokens;
    std::vector<whisper_coreml_decoder_shard> shards;
    whisper_coreml_decoder_trace trace;
    std::vector<whisper_coreml_decoder_step_trace> trace_steps;
    int64_t trace_generation_steps_recorded;
};

static int64_t whisper_coreml_decoder_time_us() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration_cast<std::chrono::microseconds>(clock::now().time_since_epoch()).count();
}

static bool whisper_coreml_decoder_env_is_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0' &&
        std::strcmp(value, "0") != 0 &&
        std::strcmp(value, "false") != 0 &&
        std::strcmp(value, "FALSE") != 0;
}

static MLComputeUnits whisper_coreml_decoder_compute_units_from_env(const char ** name) {
    const char * value = std::getenv("WHISPER_COREML_DECODER_COMPUTE_UNITS");
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "all") == 0) {
        *name = "all";
        return MLComputeUnitsAll;
    }

    if (std::strcmp(value, "cpu_gpu") == 0) {
        *name = "cpu_gpu";
        return MLComputeUnitsCPUAndGPU;
    }

    if (std::strcmp(value, "cpu_ne") == 0) {
        if (@available(macOS 13.0, iOS 16.0, watchOS 9.0, tvOS 16.0, *)) {
            *name = "cpu_ne";
            return MLComputeUnitsCPUAndNeuralEngine;
        }

        fprintf(stderr, "%s: WHISPER_COREML_DECODER_COMPUTE_UNITS=cpu_ne requires macOS 13/iOS 16 or newer; using all\n", __func__);
        *name = "all";
        return MLComputeUnitsAll;
    }

    if (std::strcmp(value, "cpu_only") == 0) {
        *name = "cpu_only";
        return MLComputeUnitsCPUOnly;
    }

    fprintf(stderr, "%s: unknown WHISPER_COREML_DECODER_COMPUTE_UNITS='%s'; using all\n", __func__, value);
    *name = "all";
    return MLComputeUnitsAll;
}

static bool whisper_coreml_decoder_compute_units_may_use_ne(const char * name) {
    return name != nullptr && (std::strcmp(name, "all") == 0 || std::strcmp(name, "cpu_ne") == 0);
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

static bool whisper_coreml_decoder_model_has_state(MLModel * model, NSString * name) {
#if WHISPER_COREML_HAS_STATE
    if (@available(macOS 15.0, *)) {
        return model.modelDescription.stateDescriptionsByName[name] != nil;
    }
#endif
    (void) model;
    (void) name;
    return false;
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
        shard->start_layer = min_layer;
        shard->n_layers = n_layers;
        shard->n_state = 0;
        shard->token_input = token_input;
        shard->output_logits = output_logits;
    }

#if WHISPER_COREML_HAS_STATE
    if (@available(macOS 15.0, *)) {
        NSString * stateName = [[NSString alloc] initWithFormat:@"k_cache_%lld", (long long) min_layer];
        MLFeatureDescription * desc = model.modelDescription.stateDescriptionsByName[stateName];
        NSArray<NSNumber *> * shape = desc.stateConstraint.bufferShape;
        if (shape.count != 4 || shape[0].longLongValue != 1 || shape[2].longLongValue != 1 ||
                shape[1].longLongValue <= 0 || shape[3].longLongValue <= 0) {
            return false;
        }

        const int64_t n_state = shape[1].longLongValue;
        const int64_t n_self_ctx = shape[3].longLongValue;
        for (int64_t layer = min_layer; layer <= max_layer; ++layer) {
            NSString * kOut = [[NSString alloc] initWithFormat:@"k_out_%lld", (long long) layer];
            NSString * vOut = [[NSString alloc] initWithFormat:@"v_out_%lld", (long long) layer];
            if (!whisper_coreml_decoder_model_has_output(model, kOut) ||
                    !whisper_coreml_decoder_model_has_output(model, vOut)) {
                return false;
            }

            for (NSString * prefix in @[@"k_cache", @"v_cache", @"cross_k", @"cross_v"]) {
                NSString * checkName = [[NSString alloc] initWithFormat:@"%@_%lld", prefix, (long long) layer];
                MLFeatureDescription * checkDesc = model.modelDescription.stateDescriptionsByName[checkName];
                NSArray<NSNumber *> * checkShape = checkDesc.stateConstraint.bufferShape;
                if (checkShape.count != 4 || checkShape[0].longLongValue != 1 ||
                        checkShape[1].longLongValue != n_state || checkShape[2].longLongValue != 1 ||
                        checkShape[3].longLongValue <= 0) {
                    return false;
                }
                if (([prefix isEqualToString:@"k_cache"] || [prefix isEqualToString:@"v_cache"]) &&
                        checkShape[3].longLongValue != n_self_ctx) {
                    return false;
                }
            }
        }

        if (shard != nullptr) {
            shard->n_state = n_state;
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

static bool whisper_coreml_decoder_add_no_write_shard(
        whisper_coreml_decoder_context * ctx,
        MLModel * model,
        const whisper_coreml_decoder_shard & parsed) {
#if WHISPER_COREML_HAS_STATE
    if (@available(macOS 15.0, *)) {
        const int64_t t_state_us = ctx->trace.enabled ? whisper_coreml_decoder_time_us() : 0;
        MLState * state = [model newState];
        if (state == nil) {
            return false;
        }
        if (ctx->trace.enabled) {
            ctx->trace.mlstate_create_us += whisper_coreml_decoder_time_us() - t_state_us;
        }

        whisper_coreml_decoder_shard shard = parsed;
        shard.model = CFBridgingRetain(model);
        shard.state = CFBridgingRetain(state);
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
        return false;
    }

    if (!first.token_input || first.start_layer != 0 || first.output_logits || max_tokens <= 0) {
        fprintf(stderr, "%s: first no-write shard must be s0 token-input hidden-output with known capacity\n", __func__);
        return false;
    }

    ctx->no_write_sharded = true;
    ctx->stateful = true;
    ctx->uses_audio_input = false;
    ctx->prewarm_state = false;
    ctx->max_tokens = max_tokens;

    if (!whisper_coreml_decoder_add_no_write_shard(ctx, first_model, first)) {
        fprintf(stderr, "%s: failed to create Core ML state for first no-write shard\n", __func__);
        return false;
    }

    const int64_t n_layers_per_shard = first.n_layers;
    for (int64_t start = first.start_layer + n_layers_per_shard; start < 512; start += n_layers_per_shard) {
        const std::string hidden_path = whisper_coreml_decoder_no_write_shard_path(first_path, start, n_layers_per_shard, false, false);
        const std::string logits_path = whisper_coreml_decoder_no_write_shard_path(first_path, start, n_layers_per_shard, false, true);
        const bool has_hidden = whisper_coreml_decoder_path_exists(hidden_path);
        const bool has_logits = whisper_coreml_decoder_path_exists(logits_path);

        if (has_hidden && has_logits) {
            fprintf(stderr, "%s: ambiguous no-write decoder shard for start layer %lld; both hidden and logits variants exist\n", __func__, (long long) start);
            return false;
        }

        if (!has_hidden && !has_logits) {
            fprintf(stderr, "%s: missing no-write decoder shard for start layer %lld\n", __func__, (long long) start);
            return false;
        }

        const bool logits = has_logits;
        const std::string & shard_path = logits ? logits_path : hidden_path;
        NSString * path = [[NSString alloc] initWithUTF8String:shard_path.c_str()];
        NSURL * url = [NSURL fileURLWithPath:path];
        NSError * error = nil;
        MLModel * model = whisper_coreml_decoder_load_model_url(url, config, &error);
        if (model == nil) {
            if (error != nil) {
                fprintf(stderr, "%s: failed to load no-write decoder shard '%s': %s\n", __func__, shard_path.c_str(), error.localizedDescription.UTF8String);
            }
            return false;
        }

        whisper_coreml_decoder_shard parsed;
        int64_t shard_max_tokens = 0;
        if (!whisper_coreml_decoder_parse_no_write_shard(model, &parsed, &shard_max_tokens) ||
                parsed.start_layer != start ||
                parsed.n_layers != n_layers_per_shard ||
                parsed.token_input ||
                parsed.output_logits != logits ||
                shard_max_tokens != ctx->max_tokens) {
            fprintf(stderr, "%s: no-write decoder shard '%s' has an incompatible interface\n", __func__, shard_path.c_str());
            return false;
        }

        if (!whisper_coreml_decoder_add_no_write_shard(ctx, model, parsed)) {
            fprintf(stderr, "%s: failed to create Core ML state for no-write decoder shard '%s'\n", __func__, shard_path.c_str());
            return false;
        }

        if (parsed.output_logits) {
            return true;
        }
    }

    fprintf(stderr, "%s: no-write decoder shard sequence has no logits shard\n", __func__);
    return false;
}

struct whisper_coreml_decoder_context * whisper_coreml_decoder_init(const char * path_model) {
    NSString * path_model_str = [[NSString alloc] initWithUTF8String:path_model];
    NSURL * url_model = [NSURL fileURLWithPath:path_model_str];

    const char * compute_units_name = nullptr;
    const MLComputeUnits compute_units = whisper_coreml_decoder_compute_units_from_env(&compute_units_name);

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
    ctx->model_prefill = nullptr;
    ctx->state = nullptr;
    ctx->compute_units = compute_units;
    ctx->compute_units_name = compute_units_name;
    ctx->audio_ane_layout = false;
    ctx->stateful = false;
    ctx->no_write_sharded = false;
    ctx->uses_audio_input = false;
    ctx->prewarm_state = whisper_coreml_decoder_env_is_enabled("WHISPER_COREML_DECODER_PREWARM_STATE");
    ctx->reuse_state = whisper_coreml_decoder_env_is_enabled("WHISPER_COREML_DECODER_REUSE_STATE");
    ctx->state_pos = 0;
    ctx->max_tokens = 0;
    ctx->prefill_tokens = 0;
    std::memset(&ctx->trace, 0, sizeof(ctx->trace));
    ctx->trace.enabled = whisper_coreml_decoder_env_is_enabled("WHISPER_COREML_DECODER_TRACE");
    ctx->trace_generation_steps_recorded = 0;

    if (whisper_coreml_decoder_parse_no_write_shard(model, nullptr, nullptr)) {
        if (whisper_coreml_decoder_compute_units_may_use_ne(compute_units_name) &&
                !whisper_coreml_decoder_env_is_enabled("WHISPER_COREML_DECODER_ALLOW_UNVERIFIED_ANE")) {
            fprintf(stderr, "%s: no-write decoder shards are known to diverge when Neural Engine is eligible; use WHISPER_COREML_DECODER_COMPUTE_UNITS=cpu_only or cpu_gpu, or set WHISPER_COREML_DECODER_ALLOW_UNVERIFIED_ANE=1 for diagnostics\n", __func__);
            whisper_coreml_decoder_free(ctx);
            return nullptr;
        }

        if (!whisper_coreml_decoder_load_no_write_shards(ctx, model, path_model, config)) {
            whisper_coreml_decoder_free(ctx);
            return nullptr;
        }
        ctx->trace.no_write_sharded = true;
        return ctx;
    }

#if WHISPER_COREML_HAS_STATE
    if (@available(macOS 15.0, *)) {
        const bool prefill_disabled = whisper_coreml_decoder_env_is_enabled("WHISPER_COREML_DECODER_DISABLE_PREFILL");
        MLModelConfiguration * prefillConfig = [[MLModelConfiguration alloc] init];
        prefillConfig.computeUnits = compute_units;
        prefillConfig.functionName = @"prefill";

        NSError * prefillError = nil;
        MLModel * model_prefill = prefill_disabled ? nil : whisper_coreml_decoder_load_model_url(url_model, prefillConfig, &prefillError);
        if (model_prefill != nil) {
            MLFeatureDescription * tokenDescription = model_prefill.modelDescription.inputDescriptionsByName[@"token_data"];
            NSArray<NSNumber *> * tokenShape = tokenDescription.multiArrayConstraint.shape;
            if (tokenShape.count == 2) {
                ctx->model_prefill = CFBridgingRetain(model_prefill);
                ctx->prefill_tokens = tokenShape[1].longLongValue;
            }
        }
    }
#endif

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
            const int64_t t_state_us = ctx->trace.enabled ? whisper_coreml_decoder_time_us() : 0;
            ctx->state = CFBridgingRetain([model newState]);
            if (ctx->trace.enabled) {
                ctx->trace.mlstate_create_us += whisper_coreml_decoder_time_us() - t_state_us;
            }

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
    for (auto & shard : ctx->shards) {
        if (shard.state != nullptr) {
            CFRelease(shard.state);
        }
        if (shard.model != nullptr) {
            CFRelease(shard.model);
        }
    }
    if (ctx->model_prefill != nullptr) {
        CFRelease(ctx->model_prefill);
    }
    if (ctx->model != nullptr) {
        CFRelease(ctx->model);
    }
    delete ctx;
}

void whisper_coreml_decoder_reset(struct whisper_coreml_decoder_context * ctx) {
    if (ctx == nullptr || !ctx->stateful) {
        return;
    }

#if WHISPER_COREML_HAS_STATE
    if (@available(macOS 15.0, *)) {
        if (ctx->no_write_sharded) {
            if (ctx->reuse_state) {
                ctx->state_pos = 0;
                return;
            }

            for (auto & shard : ctx->shards) {
                if (shard.state != nullptr) {
                    CFRelease(shard.state);
                    shard.state = nullptr;
                }

                const int64_t t_state_us = ctx->trace.enabled ? whisper_coreml_decoder_time_us() : 0;
                MLState * state = [(__bridge MLModel *) shard.model newState];
                if (ctx->trace.enabled) {
                    ctx->trace.mlstate_create_us += whisper_coreml_decoder_time_us() - t_state_us;
                }
                shard.state = CFBridgingRetain(state);
            }

            ctx->state_pos = 0;
            return;
        }

        if (ctx->reuse_state && ctx->state != nullptr) {
            ctx->state_pos = 0;
            return;
        }

        if (ctx->state != nullptr) {
            CFRelease(ctx->state);
            ctx->state = nullptr;
        }
        const int64_t t_state_us = ctx->trace.enabled ? whisper_coreml_decoder_time_us() : 0;
        ctx->state = CFBridgingRetain([(__bridge MLModel *) ctx->model newState]);
        if (ctx->trace.enabled) {
            ctx->trace.mlstate_create_us += whisper_coreml_decoder_time_us() - t_state_us;
        }
        ctx->state_pos = 0;
    }
#endif
}

bool whisper_coreml_decoder_is_stateful(const struct whisper_coreml_decoder_context * ctx) {
    return ctx != nullptr && ctx->stateful;
}

bool whisper_coreml_decoder_is_no_write_sharded(const struct whisper_coreml_decoder_context * ctx) {
    return ctx != nullptr && ctx->no_write_sharded;
}

bool whisper_coreml_decoder_uses_audio_input(const struct whisper_coreml_decoder_context * ctx) {
    return ctx != nullptr && ctx->uses_audio_input;
}

const char * whisper_coreml_decoder_compute_units_name(const struct whisper_coreml_decoder_context * ctx) {
    return ctx != nullptr && ctx->compute_units_name != nullptr ? ctx->compute_units_name : "all";
}

bool whisper_coreml_decoder_prewarms_state(const struct whisper_coreml_decoder_context * ctx) {
    return ctx != nullptr && ctx->prewarm_state;
}

bool whisper_coreml_decoder_reuses_state(const struct whisper_coreml_decoder_context * ctx) {
    return ctx != nullptr && ctx->reuse_state;
}

int64_t whisper_coreml_decoder_state_pos(const struct whisper_coreml_decoder_context * ctx) {
    return ctx != nullptr ? ctx->state_pos : -1;
}

void whisper_coreml_decoder_set_state_pos(struct whisper_coreml_decoder_context * ctx, int64_t state_pos) {
    if (ctx != nullptr) {
        ctx->state_pos = state_pos;
    }
}

bool whisper_coreml_decoder_trace_enabled(const struct whisper_coreml_decoder_context * ctx) {
    return ctx != nullptr && ctx->trace.enabled;
}

void whisper_coreml_decoder_trace_reset(struct whisper_coreml_decoder_context * ctx) {
    if (ctx == nullptr || !ctx->trace.enabled) {
        return;
    }

    const bool enabled = ctx->trace.enabled;
    const bool no_write_sharded = ctx->trace.no_write_sharded;
    std::memset(&ctx->trace, 0, sizeof(ctx->trace));
    ctx->trace.enabled = enabled;
    ctx->trace.no_write_sharded = no_write_sharded;
    ctx->trace.state_pos = ctx->state_pos;
    ctx->trace_steps.clear();
    ctx->trace_generation_steps_recorded = 0;
}

void whisper_coreml_decoder_trace_get(const struct whisper_coreml_decoder_context * ctx, struct whisper_coreml_decoder_trace * trace) {
    if (trace == nullptr) {
        return;
    }

    std::memset(trace, 0, sizeof(*trace));
    if (ctx == nullptr) {
        return;
    }

    *trace = ctx->trace;
    trace->state_pos = ctx->state_pos;
}

int64_t whisper_coreml_decoder_trace_step_count(const struct whisper_coreml_decoder_context * ctx) {
    if (ctx == nullptr || !ctx->trace.enabled) {
        return 0;
    }

    return (int64_t) ctx->trace_steps.size();
}

bool whisper_coreml_decoder_trace_step_get(const struct whisper_coreml_decoder_context * ctx, int64_t index, struct whisper_coreml_decoder_step_trace * step) {
    if (ctx == nullptr || step == nullptr || !ctx->trace.enabled || index < 0 || index >= (int64_t) ctx->trace_steps.size()) {
        return false;
    }

    *step = ctx->trace_steps[index];
    return true;
}

static bool whisper_coreml_decoder_copy_state_f16(
        whisper_coreml_decoder_context * ctx,
        const void * statePtr,
        const char * name,
        const void * data,
        int64_t n_elems) {
    if (ctx == nullptr || statePtr == nullptr || name == nullptr || data == nullptr || n_elems <= 0) {
        return false;
    }

#if WHISPER_COREML_HAS_STATE
    if (@available(macOS 15.0, *)) {
        NSString * stateName = [[NSString alloc] initWithUTF8String:name];
        __block bool ok = false;
        const int64_t t_write_us = ctx->trace.enabled ? whisper_coreml_decoder_time_us() : 0;

        [(__bridge MLState *) statePtr getMultiArrayForStateNamed:stateName handler:^(MLMultiArray * buffer) {
            if (buffer.dataType != MLMultiArrayDataTypeFloat16 || buffer.count < n_elems) {
                return;
            }

            if (buffer.shape.count == 4 && buffer.shape[2].longLongValue == 1) {
                const int64_t n_state = buffer.shape[1].longLongValue;
                const int64_t n_ctx = buffer.shape[3].longLongValue;
                if (n_state > 0 && n_elems % n_state == 0 && n_elems/n_state <= n_ctx) {
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
                    ok = true;
                }
            } else {
                memcpy(buffer.dataPointer, data, n_elems*sizeof(uint16_t));
                ok = true;
            }
        }];

        if (ctx->trace.enabled) {
            const int64_t elapsed_us = whisper_coreml_decoder_time_us() - t_write_us;
            if (ok) {
                if (std::strncmp(name, "cross_", 6) == 0) {
                    ctx->trace.cross_kv_write_count++;
                    ctx->trace.cross_kv_bytes_written += n_elems*sizeof(uint16_t);
                    ctx->trace.cross_kv_write_us += elapsed_us;
                } else if (std::strncmp(name, "k_cache_", 8) == 0 || std::strncmp(name, "v_cache_", 8) == 0) {
                    ctx->trace.self_kv_write_count++;
                    ctx->trace.self_kv_bytes_written += n_elems*sizeof(uint16_t);
                    ctx->trace.self_kv_write_us += elapsed_us;
                }
            }
        }

        return ok;
    }
#endif

    return false;
}

bool whisper_coreml_decoder_set_state_f16(struct whisper_coreml_decoder_context * ctx, const char * name, const void * data, int64_t n_elems) {
    if (ctx == nullptr || name == nullptr || data == nullptr || !ctx->stateful) {
        return false;
    }

#if WHISPER_COREML_HAS_STATE
    if (@available(macOS 15.0, *)) {
        if (ctx->no_write_sharded) {
            NSString * stateName = [[NSString alloc] initWithUTF8String:name];
            for (const auto & shard : ctx->shards) {
                if (shard.state == nullptr || shard.model == nullptr) {
                    continue;
                }

                if (!whisper_coreml_decoder_model_has_state((__bridge MLModel *) shard.model, stateName)) {
                    continue;
                }

                return whisper_coreml_decoder_copy_state_f16(ctx, shard.state, name, data, n_elems);
            }

            return false;
        }
    }
#endif

    if (ctx->state == nullptr) {
        return false;
    }

    return whisper_coreml_decoder_copy_state_f16(ctx, ctx->state, name, data, n_elems);
}

static void whisper_coreml_decoder_trace_prediction(
        whisper_coreml_decoder_context * ctx,
                           const bool   is_prompt,
                           const bool   is_prefill,
                           const bool   is_prewarm,
                        const int64_t   state_pos,
                        const int64_t   n_tokens,
                        const int64_t   prediction_us) {
    if (ctx == nullptr || !ctx->trace.enabled) {
        return;
    }

    if (!is_prewarm && !is_prompt) {
        if (ctx->trace_generation_steps_recorded >= WHISPER_COREML_DECODER_TRACE_GENERATION_STEPS) {
            return;
        }
        ctx->trace_generation_steps_recorded++;
    }

    whisper_coreml_decoder_step_trace step;
    step.is_prompt    = is_prompt;
    step.is_prefill   = is_prefill;
    step.is_prewarm   = is_prewarm;
    step.step_index   = (int64_t) ctx->trace_steps.size();
    step.shard_index  = -1;
    step.shard_start_layer = -1;
    step.state_pos    = state_pos;
    step.n_tokens     = n_tokens;
    step.prediction_us = prediction_us;
    step.state_write_us = 0;
    step.logits_copy_us = 0;
    ctx->trace_steps.push_back(step);
}

static void whisper_coreml_decoder_trace_shard_prediction(
        whisper_coreml_decoder_context * ctx,
                           const bool   is_prompt,
                        const int64_t   state_pos,
                        const int64_t   n_tokens,
                        const int64_t   shard_index,
                        const int64_t   shard_start_layer,
                        const int64_t   prediction_us,
                        const int64_t   state_write_us,
                        const int64_t   logits_copy_us) {
    if (ctx == nullptr || !ctx->trace.enabled) {
        return;
    }

    whisper_coreml_decoder_step_trace step;
    step.is_prompt    = is_prompt;
    step.is_prefill   = false;
    step.is_prewarm   = false;
    step.step_index   = (int64_t) ctx->trace_steps.size();
    step.shard_index  = shard_index;
    step.shard_start_layer = shard_start_layer;
    step.state_pos    = state_pos;
    step.n_tokens     = n_tokens;
    step.prediction_us = prediction_us;
    step.state_write_us = state_write_us;
    step.logits_copy_us = logits_copy_us;
    ctx->trace_steps.push_back(step);
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

static MLMultiArray * whisper_coreml_make_state_self_mask_batch(
        const int64_t pos,
        const int64_t n_tokens,
        const int64_t max_tokens,
          NSError ** error) {
    MLMultiArray * result = [[MLMultiArray alloc] initWithShape:@[@1, @(n_tokens), @(max_tokens)]
                                                       dataType:MLMultiArrayDataTypeFloat32
                                                          error:error];
    if (result == nil) {
        return nil;
    }

    float * data = (float *) result.dataPointer;
    const int64_t stride_t = result.strides[1].longLongValue;
    const int64_t stride_i = result.strides[2].longLongValue;
    for (int64_t t = 0; t < n_tokens; ++t) {
        const int64_t visible_pos = pos + t;
        for (int64_t i = 0; i < max_tokens; ++i) {
            data[t*stride_t + i*stride_i] = i <= visible_pos ? 0.0f : -INFINITY;
        }
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

static bool whisper_coreml_decoder_write_state_slot_f16(
        const void * statePtr,
        const char * name,
        MLMultiArray * value,
        const int64_t pos) {
    if (statePtr == nullptr || name == nullptr || value == nil || value.dataPointer == nullptr ||
            value.dataType != MLMultiArrayDataTypeFloat16 || pos < 0) {
        return false;
    }

#if WHISPER_COREML_HAS_STATE
    if (@available(macOS 15.0, *)) {
        NSString * stateName = [[NSString alloc] initWithUTF8String:name];
        __block bool ok = false;
        [(__bridge MLState *) statePtr getMultiArrayForStateNamed:stateName handler:^(MLMultiArray * buffer) {
            if (buffer.dataType != MLMultiArrayDataTypeFloat16 || buffer.shape.count != 4 ||
                    buffer.shape[2].longLongValue != 1 || pos >= buffer.shape[3].longLongValue) {
                return;
            }

            const int64_t n_state = buffer.shape[1].longLongValue;
            if (value.shape.count != 4 || value.shape[1].longLongValue != n_state ||
                    value.shape[2].longLongValue != 1 || value.shape[3].longLongValue != 1) {
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
        whisper_coreml_decoder_context * ctx,
        const whisper_coreml_decoder_shard & shard,
        id<MLFeatureProvider> output,
        const int64_t pos) {
    if (ctx == nullptr || output == nil || shard.state == nullptr) {
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
        if (kValue == nil || vValue == nil ||
                !whisper_coreml_decoder_write_state_slot_f16(shard.state, kState.UTF8String, kValue.multiArrayValue, pos) ||
                !whisper_coreml_decoder_write_state_slot_f16(shard.state, vState.UTF8String, vValue.multiArrayValue, pos)) {
            return false;
        }
    }

    return true;
}

static bool whisper_coreml_decoder_predict(
        struct whisper_coreml_decoder_context * ctx,
                                const void * modelPtr,
                              MLMultiArray * tokenArray,
                              MLMultiArray * audioArray,
                              MLMultiArray * posArray,
                              MLMultiArray * stepMask,
                              MLMultiArray * selfMask,
                                   int64_t   n_tokens,
                                   int64_t   n_vocab,
                                    float * out_logits,
                                      bool   is_prompt,
                                      bool   is_prefill,
                                      bool   is_prewarm,
                                   int64_t   state_pos,
                                  NSError ** error) {
    const int64_t t_feature_provider_us = ctx->trace.enabled ? whisper_coreml_decoder_time_us() : 0;
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
    if (ctx->trace.enabled) {
        ctx->trace.feature_provider_create_us += whisper_coreml_decoder_time_us() - t_feature_provider_us;
    }
    if (input == nil) {
        return false;
    }

    id<MLFeatureProvider> output = nil;
    MLModel * model = (__bridge MLModel *) (modelPtr != nullptr ? modelPtr : ctx->model);
    const int64_t t_prediction_us = ctx->trace.enabled ? whisper_coreml_decoder_time_us() : 0;
    if (ctx->stateful) {
#if WHISPER_COREML_HAS_STATE
        if (@available(macOS 15.0, *)) {
            output = [model predictionFromFeatures:input usingState:(__bridge MLState *) ctx->state error:error];
        }
#endif
    } else {
        output = [model predictionFromFeatures:input error:error];
    }
    if (ctx->trace.enabled) {
        const int64_t elapsed_us = whisper_coreml_decoder_time_us() - t_prediction_us;
        ctx->trace.prediction_us += elapsed_us;
        whisper_coreml_decoder_trace_prediction(ctx, is_prompt, is_prefill, is_prewarm, state_pos, n_tokens, elapsed_us);
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

    const int64_t t_logits_copy_us = ctx->trace.enabled ? whisper_coreml_decoder_time_us() : 0;
    const bool ok = whisper_coreml_copy_last_logits(logitsValue.multiArrayValue, n_tokens, n_vocab, out_logits);
    if (ctx->trace.enabled) {
        ctx->trace.logits_copy_us += whisper_coreml_decoder_time_us() - t_logits_copy_us;
    }

    return ok;
}

static bool whisper_coreml_decoder_decode_no_write(
        struct whisper_coreml_decoder_context * ctx,
                                     int64_t   n_tokens,
                                     int64_t   n_vocab,
                              const int32_t * tokens,
                                      float * out_logits,
                                        bool   is_prompt,
                                  NSError ** error) {
    if (ctx == nullptr || !ctx->no_write_sharded || tokens == nullptr || out_logits == nullptr || ctx->shards.empty()) {
        return false;
    }

    for (int64_t i = 0; i < n_tokens; ++i) {
        const int32_t token = tokens[i];
        const int32_t pos = (int32_t) ctx->state_pos;
        const bool record_generation = is_prompt || ctx->trace_generation_steps_recorded < WHISPER_COREML_DECODER_TRACE_GENERATION_STEPS;

        const int64_t t_step_input_us = ctx->trace.enabled ? whisper_coreml_decoder_time_us() : 0;
        MLMultiArray * tokenStep = whisper_coreml_make_i32_array(@[@1, @1], &token, 1, error);
        MLMultiArray * posStep = whisper_coreml_make_i32_array(@[@1], &pos, 1, error);
        MLMultiArray * slotMask = whisper_coreml_make_ane_slot_mask(ctx->state_pos, ctx->max_tokens, error);
        MLMultiArray * selfMask = whisper_coreml_make_ane_self_mask(ctx->state_pos, ctx->max_tokens, error);
        if (ctx->trace.enabled) {
            ctx->trace.input_array_create_us += whisper_coreml_decoder_time_us() - t_step_input_us;
            if (is_prompt) {
                ctx->trace.prompt_step_count++;
            } else {
                ctx->trace.generation_step_count++;
            }
        }

        if (tokenStep == nil || posStep == nil || slotMask == nil || selfMask == nil) {
            return false;
        }

        MLMultiArray * xArray = nil;

        for (int64_t is = 0; is < (int64_t) ctx->shards.size(); ++is) {
            const auto & shard = ctx->shards[is];
            const int64_t t_feature_provider_us = ctx->trace.enabled ? whisper_coreml_decoder_time_us() : 0;
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

            MLDictionaryFeatureProvider * input = [[MLDictionaryFeatureProvider alloc] initWithDictionary:features error:error];
            if (ctx->trace.enabled) {
                ctx->trace.feature_provider_create_us += whisper_coreml_decoder_time_us() - t_feature_provider_us;
            }
            if (input == nil) {
                return false;
            }

            id<MLFeatureProvider> output = nil;
            const int64_t t_prediction_us = ctx->trace.enabled ? whisper_coreml_decoder_time_us() : 0;
#if WHISPER_COREML_HAS_STATE
            if (@available(macOS 15.0, *)) {
                output = [(__bridge MLModel *) shard.model predictionFromFeatures:input usingState:(__bridge MLState *) shard.state error:error];
            }
#endif
            const int64_t prediction_us = ctx->trace.enabled ? whisper_coreml_decoder_time_us() - t_prediction_us : 0;
            if (ctx->trace.enabled) {
                ctx->trace.prediction_us += prediction_us;
            }
            if (output == nil) {
                return false;
            }

            const int64_t t_state_write_us = ctx->trace.enabled ? whisper_coreml_decoder_time_us() : 0;
            if (!whisper_coreml_decoder_write_no_write_outputs(ctx, shard, output, ctx->state_pos)) {
                return false;
            }
            const int64_t state_write_us = ctx->trace.enabled ? whisper_coreml_decoder_time_us() - t_state_write_us : 0;
            if (ctx->trace.enabled) {
                const int64_t n_state_elems = shard.n_layers*2*shard.n_state;
                ctx->trace.self_kv_write_count += shard.n_layers*2;
                ctx->trace.self_kv_bytes_written += n_state_elems*sizeof(uint16_t);
                ctx->trace.self_kv_write_us += state_write_us;
                ctx->trace.shard_state_write_us += state_write_us;
            }

            int64_t logits_copy_us = 0;
            if (shard.output_logits) {
                MLFeatureValue * logitsValue = [output featureValueForName:@"logits"];
                const int64_t t_logits_copy_us = ctx->trace.enabled ? whisper_coreml_decoder_time_us() : 0;
                const bool ok = whisper_coreml_copy_last_logits(logitsValue.multiArrayValue, 1, n_vocab, out_logits);
                logits_copy_us = ctx->trace.enabled ? whisper_coreml_decoder_time_us() - t_logits_copy_us : 0;
                if (ctx->trace.enabled) {
                    ctx->trace.logits_copy_us += logits_copy_us;
                }
                if (!ok) {
                    return false;
                }
            } else {
                MLFeatureValue * xValue = [output featureValueForName:@"x_out"];
                if (xValue == nil || xValue.multiArrayValue == nil) {
                    return false;
                }
                xArray = xValue.multiArrayValue;
            }

            if (record_generation && ctx->trace.enabled) {
                whisper_coreml_decoder_trace_shard_prediction(
                        ctx,
                        is_prompt,
                        ctx->state_pos,
                        1,
                        is,
                        shard.start_layer,
                        prediction_us,
                        state_write_us,
                        logits_copy_us);
            }
        }

        if (!is_prompt && ctx->trace.enabled) {
            ctx->trace_generation_steps_recorded++;
        }
        ctx->state_pos++;
    }

    return true;
}

bool whisper_coreml_decoder_prewarm_state(
        struct whisper_coreml_decoder_context * ctx,
                                      int64_t   n_vocab,
                                      int64_t   n_audio_ctx,
                                      int64_t   n_audio_state,
                                const float * audio) {
    if (ctx == nullptr || !ctx->stateful || ctx->state == nullptr || n_vocab <= 0) {
        return false;
    }

    if (ctx->uses_audio_input && audio == nullptr) {
        return false;
    }

    if (ctx->max_tokens <= 0 || ctx->state_pos >= ctx->max_tokens) {
        fprintf(stderr, "%s: Core ML decoder state capacity exceeded\n", __func__);
        return false;
    }

    @autoreleasepool {
        NSError * error = nil;
        const int32_t token = 0;
        const int32_t pos = (int32_t) ctx->state_pos;
        const int64_t t_input_us = ctx->trace.enabled ? whisper_coreml_decoder_time_us() : 0;

        MLMultiArray * tokenArray = whisper_coreml_make_i32_array(@[@1, @1], &token, 1, &error);
        MLMultiArray * audioArray = whisper_coreml_make_audio_array(ctx, audio, n_audio_ctx, n_audio_state, &error);
        MLMultiArray * posArray = whisper_coreml_make_i32_array(@[@1], &pos, 1, &error);
        MLMultiArray * stepMask = whisper_coreml_make_state_step_mask(ctx->state_pos + 1, &error);
        MLMultiArray * selfMask = whisper_coreml_make_state_self_mask(ctx->state_pos, ctx->max_tokens, &error);

        if (ctx->trace.enabled) {
            ctx->trace.input_array_create_us += whisper_coreml_decoder_time_us() - t_input_us;
        }

        if (tokenArray == nil || (ctx->uses_audio_input && audioArray == nil) || posArray == nil || stepMask == nil || selfMask == nil) {
            if (error != nil) {
                fprintf(stderr, "%s: failed to create Core ML decoder prewarm inputs: %s\n", __func__, error.localizedDescription.UTF8String);
            }
            return false;
        }

        std::vector<float> logits(n_vocab);
        if (!whisper_coreml_decoder_predict(ctx, ctx->model, tokenArray, audioArray, posArray, stepMask, selfMask, 1, n_vocab, logits.data(), false, false, true, ctx->state_pos, &error)) {
            if (error != nil) {
                fprintf(stderr, "%s: Core ML decoder prewarm prediction failed: %s\n", __func__, error.localizedDescription.UTF8String);
            }
            return false;
        }

        return true;
    }
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
    if (ctx == nullptr || tokens == nullptr || audio == nullptr || out_logits == nullptr || n_tokens <= 0) {
        return false;
    }

    @autoreleasepool {
        NSError * error = nil;

        if (ctx->stateful && (ctx->max_tokens <= 0 || ctx->state_pos + n_tokens > ctx->max_tokens)) {
            fprintf(stderr, "%s: Core ML decoder state capacity exceeded\n", __func__);
            return false;
        }

        if (ctx->no_write_sharded) {
            if (!whisper_coreml_decoder_decode_no_write(ctx, n_tokens, n_vocab, tokens, out_logits, is_prompt, &error)) {
                if (error != nil) {
                    fprintf(stderr, "%s: Core ML no-write shard decoder prediction failed: %s\n", __func__, error.localizedDescription.UTF8String);
                }
                return false;
            }

            return true;
        }

        const int64_t t_input_us = ctx->trace.enabled ? whisper_coreml_decoder_time_us() : 0;

        MLMultiArray * tokenArray = whisper_coreml_make_token_array(tokens, n_tokens, &error);
        if (tokenArray == nil) {
            if (error != nil) {
                fprintf(stderr, "%s: failed to create token input: %s\n", __func__, error.localizedDescription.UTF8String);
            }
            return false;
        }

        MLMultiArray * audioArray = whisper_coreml_make_audio_array(ctx, audio, n_audio_ctx, n_audio_state, &error);
        if (ctx->trace.enabled) {
            ctx->trace.input_array_create_us += whisper_coreml_decoder_time_us() - t_input_us;
        }
        if (ctx->uses_audio_input && audioArray == nil) {
            if (error != nil) {
                fprintf(stderr, "%s: failed to create audio input: %s\n", __func__, error.localizedDescription.UTF8String);
            }
            return false;
        }

        if (!ctx->stateful) {
            if (ctx->trace.enabled) {
                if (is_prompt) {
                    ctx->trace.prompt_step_count++;
                } else {
                    ctx->trace.generation_step_count++;
                }
            }
            if (!whisper_coreml_decoder_predict(ctx, ctx->model, tokenArray, audioArray, nil, nil, nil, n_tokens, n_vocab, out_logits, is_prompt, false, false, ctx->state_pos, &error)) {
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

        if (is_prompt && ctx->model_prefill != nullptr && n_tokens == ctx->prefill_tokens) {
            const int64_t state_pos = ctx->state_pos;
            std::vector<int32_t> pos(n_tokens);
            for (int64_t i = 0; i < n_tokens; ++i) {
                pos[i] = (int32_t) (state_pos + i);
            }

            const int64_t t_prefill_input_us = ctx->trace.enabled ? whisper_coreml_decoder_time_us() : 0;
            MLMultiArray * posArray = whisper_coreml_make_i32_array(@[@(n_tokens)], pos.data(), n_tokens, &error);
            MLMultiArray * stepMask = whisper_coreml_make_state_step_mask(state_pos + n_tokens, &error);
            MLMultiArray * selfMask = whisper_coreml_make_state_self_mask_batch(state_pos, n_tokens, ctx->max_tokens, &error);
            if (ctx->trace.enabled) {
                ctx->trace.input_array_create_us += whisper_coreml_decoder_time_us() - t_prefill_input_us;
                ctx->trace.prompt_step_count++;
            }

            if (posArray == nil || stepMask == nil || selfMask == nil) {
                if (error != nil) {
                    fprintf(stderr, "%s: failed to create stateful decoder prefill inputs: %s\n", __func__, error.localizedDescription.UTF8String);
                }
                return false;
            }

            if (!whisper_coreml_decoder_predict(ctx, ctx->model_prefill, tokenArray, audioArray, posArray, stepMask, selfMask, n_tokens, n_vocab, out_logits, is_prompt, true, false, state_pos, &error)) {
                if (error != nil) {
                    fprintf(stderr, "%s: Core ML decoder prefill prediction failed: %s\n", __func__, error.localizedDescription.UTF8String);
                }
                return false;
            }

            ctx->state_pos += n_tokens;
            return true;
        }

        for (int64_t i = 0; i < n_tokens; ++i) {
            const int32_t token = tokens[i];
            const int32_t pos = (int32_t) ctx->state_pos;

            const int64_t t_step_input_us = ctx->trace.enabled ? whisper_coreml_decoder_time_us() : 0;
            MLMultiArray * tokenStep = whisper_coreml_make_i32_array(@[@1, @1], &token, 1, &error);
            MLMultiArray * posStep = whisper_coreml_make_i32_array(@[@1], &pos, 1, &error);
            MLMultiArray * stepMask = whisper_coreml_make_state_step_mask(ctx->state_pos + 1, &error);
            MLMultiArray * selfMask = whisper_coreml_make_state_self_mask(ctx->state_pos, ctx->max_tokens, &error);
            if (ctx->trace.enabled) {
                ctx->trace.input_array_create_us += whisper_coreml_decoder_time_us() - t_step_input_us;
                if (is_prompt) {
                    ctx->trace.prompt_step_count++;
                } else {
                    ctx->trace.generation_step_count++;
                }
            }

            if (tokenStep == nil || posStep == nil || stepMask == nil || selfMask == nil) {
                if (error != nil) {
                    fprintf(stderr, "%s: failed to create stateful decoder inputs: %s\n", __func__, error.localizedDescription.UTF8String);
                }
                return false;
            }

            if (!whisper_coreml_decoder_predict(ctx, ctx->model, tokenStep, audioArray, posStep, stepMask, selfMask, 1, n_vocab, out_logits, is_prompt, false, false, ctx->state_pos, &error)) {
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
