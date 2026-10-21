#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include "minimax/coreml-conditioner.h"
#include "minimax/conditioner-layout.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <mutex>

struct MM3CoreMLConditioner {
    MLModel * model = nil;
    MLMultiArray * input = nil;
    MLMultiArray * output = nil;
    MLDictionaryFeatureProvider * provider = nil;
    MLPredictionOptions * options = nil;
    int64_t hidden = 0;
    int64_t channels = 0;
    int64_t window = 0;
    std::mutex mutex;
};

namespace {
bool fail(std::string * error, const std::string & message) {
    if (error) *error = "MiniMax Core ML conditioner: " + message;
    return false;
}

std::string describe(NSError * error) {
    const char * message = error.localizedDescription.UTF8String;
    return message ? message : "unknown Core ML error";
}

bool valid_shape(NSArray<NSNumber *> * shape, int64_t channels, int64_t window) {
    return shape.count == 4 && shape[0].longLongValue == 1 &&
           shape[1].longLongValue == channels && shape[2].longLongValue == 1 &&
           shape[3].longLongValue == window;
}
}

MM3CoreMLConditioner * mm3_coreml_conditioner_load(const std::string & path,
                                                 int64_t hidden, int64_t output,
                                                 std::string * error) {
    @autoreleasepool {
        if (hidden <= 0 || output <= 0) {
            fail(error, "invalid model dimensions");
            return nullptr;
        }
        NSString * name = [[NSString alloc] initWithUTF8String:path.c_str()];
        if (!name || path.empty()) {
            fail(error, "invalid compiled model path");
            return nullptr;
        }
        MLModelConfiguration * config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        NSError * load_error = nil;
        MLModel * model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:name]
                                           configuration:config error:&load_error];
        if (!model) {
            fail(error, describe(load_error));
            return nullptr;
        }
        auto inputs = model.modelDescription.inputDescriptionsByName;
        auto outputs = model.modelDescription.outputDescriptionsByName;
        MLMultiArrayConstraint * in = inputs[@"mixed"].multiArrayConstraint;
        MLMultiArrayConstraint * out = outputs[@"projected"].multiArrayConstraint;
        const int64_t window = in.shape.count == 4 ? in.shape[3].longLongValue : 0;
        NSDictionary * metadata = model.modelDescription.metadata[MLModelCreatorDefinedKey];
        if (inputs.count != 1 || outputs.count != 1 || !in || !out || window < 3 ||
            in.dataType != MLMultiArrayDataTypeFloat32 || out.dataType != MLMultiArrayDataTypeFloat32 ||
            !valid_shape(in.shape, hidden, window) || !valid_shape(out.shape, output, window) ||
            ![metadata[@"minimax.component"] isEqualToString:@"conditioner-projection-k3-p1-v1"]) {
            fail(error, "incompatible projection interface or metadata");
            return nullptr;
        }
        auto context = std::make_unique<MM3CoreMLConditioner>();
        context->model = model;
        context->hidden = hidden;
        context->channels = output;
        context->window = window;
        context->input = [[MLMultiArray alloc] initWithShape:in.shape
                                                  dataType:MLMultiArrayDataTypeFloat32 error:&load_error];
        context->output = [[MLMultiArray alloc] initWithShape:out.shape
                                                   dataType:MLMultiArrayDataTypeFloat32 error:&load_error];
        if (!context->input || !context->output) {
            fail(error, describe(load_error));
            return nullptr;
        }
        context->provider = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{@"mixed": context->input} error:&load_error];
        if (!context->provider) {
            fail(error, describe(load_error));
            return nullptr;
        }
        context->options = [[MLPredictionOptions alloc] init];
        context->options.outputBackings = @{@"projected": context->output};
        return context.release();
    }
}

void mm3_coreml_conditioner_free(MM3CoreMLConditioner * context) {
    delete context;
}

bool mm3_coreml_conditioner_project(MM3CoreMLConditioner * context, const float * mixed,
                                   int64_t frames, std::vector<float> & projected,
                                   std::string * error) {
    if (!context || !mixed || frames <= 0 ||
        static_cast<uint64_t>(frames) > std::vector<float>().max_size() /
                                             static_cast<uint64_t>(context->channels)) {
        return fail(error, "invalid projection input");
    }
    std::lock_guard<std::mutex> lock(context->mutex);
    @autoreleasepool {
        projected.resize(static_cast<size_t>(frames) * context->channels);
        const int64_t payload = context->window - 2;
        for (int64_t first = 0; first < frames; first += payload) {
            const int64_t count = std::min(payload, frames - first);
            tts_cpp::minimax::detail::pack_conditioner_window(
                mixed, frames, context->hidden, first, context->window,
                static_cast<float *>(context->input.dataPointer),
                context->input.strides[1].longLongValue, context->input.strides[3].longLongValue);
            NSError * prediction_error = nil;
            id<MLFeatureProvider> result = [context->model predictionFromFeatures:context->provider
                                                                        options:context->options
                                                                          error:&prediction_error];
            MLMultiArray * output = [result featureValueForName:@"projected"].multiArrayValue;
            if (!result || !output || output.dataType != MLMultiArrayDataTypeFloat32 ||
                !valid_shape(output.shape, context->channels, context->window)) {
                return fail(error, prediction_error ? describe(prediction_error) : "invalid prediction output");
            }
            tts_cpp::minimax::detail::unpack_conditioner_window(
                static_cast<const float *>(output.dataPointer), context->channels, count,
                output.strides[1].longLongValue, output.strides[3].longLongValue,
                projected.data() + first * context->channels);
        }
        return true;
    }
}
