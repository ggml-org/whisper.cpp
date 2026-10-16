#if !defined(__APPLE__)
#error "runner.mm is Apple-only; compile it solely under TTS_CPP_USE_COREML"
#endif

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include "moss/coreml/runner.h"

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

#if (defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000) || \
    (defined(__IPHONE_OS_VERSION_MAX_ALLOWED) && __IPHONE_OS_VERSION_MAX_ALLOWED >= 180000)
#define MOSS_COREML_STATE_SDK 1
#else
#define MOSS_COREML_STATE_SDK 0
#endif

namespace {

struct placement {
    MLComputeUnits units;
    const char *   label;
};

constexpr placement CPU_PLACEMENT = {MLComputeUnitsCPUOnly, "coreml-cpu"};
constexpr placement GPU_PLACEMENT = {MLComputeUnitsCPUAndGPU, "coreml-gpu"};
constexpr placement ANE_PLACEMENT = {MLComputeUnitsCPUAndNeuralEngine, "coreml-ane"};
constexpr placement ALL_PLACEMENT = {MLComputeUnitsAll, "coreml-all"};

struct Feature {
    std::string          name;
    std::vector<int64_t> dims;
    const void *         array = nullptr;
};

struct View {
    void *               base = nullptr;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    bool                 f32 = true;
};

float decode_float16(uint16_t h) {
    __fp16 v;
    std::memcpy(&v, &h, sizeof(v));
    return (float) v;
}

uint16_t encode_float16(float f) {
    __fp16 v = (__fp16) f;
    uint16_t out;
    std::memcpy(&out, &v, sizeof(out));
    return out;
}

placement placement_named(const char * compute_units) {
    const std::string requested = compute_units != nullptr ? compute_units : "";
    if (requested == "cpu_only") return CPU_PLACEMENT;
    if (requested == "cpu_and_gpu") return GPU_PLACEMENT;
    if (requested == "cpu_and_ane") return ANE_PLACEMENT;
    return ALL_PLACEMENT;
}

MLModel * load_model(const char * path_mlmodelc, const placement & where) {
    NSString * path = [[NSString alloc] initWithUTF8String:path_mlmodelc];
    if (path == nil) return nil;
    MLModelConfiguration * config = [[MLModelConfiguration alloc] init];
    config.computeUnits = where.units;
    NSError * err   = nil;
    MLModel * model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:path]
                                        configuration:config
                                                error:&err];
    if (model == nil && err != nil) std::fprintf(stderr, "Core ML: %s\n", err.localizedDescription.UTF8String);
    return (model == nil || err != nil) ? nil : model;
}

std::vector<int64_t> dims_of(NSArray<NSNumber *> * shape) {
    std::vector<int64_t> dims;
    for (NSNumber * n in shape) dims.push_back(n.longLongValue);
    return dims;
}

int64_t element_count(const std::vector<int64_t> & dims) {
    int64_t count = dims.empty() ? 0 : 1;
    for (int64_t d : dims) count = d > 0 ? count * d : 0;
    return count;
}

bool supported_data_type(MLMultiArrayDataType dt) {
    return dt == MLMultiArrayDataTypeFloat32 || dt == MLMultiArrayDataTypeFloat16;
}

bool describe_feature(NSString * name, MLFeatureDescription * description, Feature * feature) {
    if (description.type != MLFeatureTypeMultiArray) return false;
    MLMultiArrayConstraint * constraint = description.multiArrayConstraint;
    if (constraint == nil || !supported_data_type(constraint.dataType)) return false;
    feature->name = name.UTF8String;
    feature->dims = dims_of(constraint.shape);
    if (element_count(feature->dims) <= 0) return false;
    NSError *      err   = nil;
    MLMultiArray * array = [[MLMultiArray alloc] initWithShape:constraint.shape
                                                      dataType:constraint.dataType
                                                         error:&err];
    if (array == nil || err != nil) return false;
    feature->array = CFBridgingRetain(array);
    return true;
}

bool describe_features(NSDictionary<NSString *, MLFeatureDescription *> * descriptions,
                       std::vector<Feature> * features) {
    for (NSString * name in descriptions) {
        Feature feature;
        const bool described = describe_feature(name, descriptions[name], &feature);
        features->push_back(feature);
        if (!described) return false;
    }
    return !features->empty();
}

void release_features(std::vector<Feature> * features) {
    for (Feature & feature : *features) {
        if (feature.array != nullptr) CFRelease(feature.array);
        feature.array = nullptr;
    }
}

const Feature * find_feature(const std::vector<Feature> & features, const char * name) {
    for (const Feature & feature : features) {
        if (name != nullptr && feature.name == name) return &feature;
    }
    return nullptr;
}

NSDictionary<NSString *, id> * feature_dictionary(const std::vector<Feature> & features) {
    NSMutableDictionary<NSString *, id> * dict = [NSMutableDictionary dictionaryWithCapacity:features.size()];
    for (const Feature & feature : features) {
        dict[[NSString stringWithUTF8String:feature.name.c_str()]] = (__bridge MLMultiArray *) feature.array;
    }
    return dict;
}

View view_of(MLMultiArray * array) {
    View view;
    view.base    = array.dataPointer;
    view.shape   = dims_of(array.shape);
    view.strides = dims_of(array.strides);
    view.f32     = array.dataType == MLMultiArrayDataTypeFloat32;
    return view;
}

int64_t row_offset(const View & view, int64_t row) {
    int64_t offset = 0;
    for (int d = (int) view.shape.size() - 2; d >= 0; --d) {
        offset += (row % view.shape[(size_t) d]) * view.strides[(size_t) d];
        row /= view.shape[(size_t) d];
    }
    return offset;
}

void write_row_f32(float * dst, int64_t stride, const float * src, int64_t width) {
    for (int64_t i = 0; i < width; ++i) dst[i * stride] = src[i];
}

void write_row_f16(uint16_t * dst, int64_t stride, const float * src, int64_t width) {
    for (int64_t i = 0; i < width; ++i) dst[i * stride] = encode_float16(src[i]);
}

void read_row_f32(const float * src, int64_t stride, float * dst, int64_t width) {
    for (int64_t i = 0; i < width; ++i) dst[i] = src[i * stride];
}

void read_row_f16(const uint16_t * src, int64_t stride, float * dst, int64_t width) {
    for (int64_t i = 0; i < width; ++i) dst[i] = decode_float16(src[i * stride]);
}

void write_view(const View & view, const float * data) {
    const int64_t width = view.shape.back();
    const int64_t rows  = element_count(view.shape) / width;
    const int64_t inner = view.strides.back();
    for (int64_t row = 0; row < rows; ++row) {
        const int64_t offset = row_offset(view, row);
        if (view.f32) write_row_f32((float *) view.base + offset, inner, data + row * width, width);
        else write_row_f16((uint16_t *) view.base + offset, inner, data + row * width, width);
    }
}

void read_view(const View & view, float * data) {
    const int64_t width = view.shape.back();
    const int64_t rows  = element_count(view.shape) / width;
    const int64_t inner = view.strides.back();
    for (int64_t row = 0; row < rows; ++row) {
        const int64_t offset = row_offset(view, row);
        if (view.f32) read_row_f32((const float *) view.base + offset, inner, data + row * width, width);
        else read_row_f16((const uint16_t *) view.base + offset, inner, data + row * width, width);
    }
}

bool model_declares_state(MLModel * model) {
#if MOSS_COREML_STATE_SDK
    if (@available(macOS 15.0, iOS 18.0, *)) {
        return model.modelDescription.stateDescriptionsByName.count > 0;
    }
#endif
    (void) model;
    return false;
}

#if MOSS_COREML_STATE_SDK
API_AVAILABLE(macos(15.0), ios(18.0))
void zero_state(MLModel * model, MLState * state) {
    for (NSString * name in model.modelDescription.stateDescriptionsByName) {
        [state getMultiArrayForStateNamed:name
                                  handler:^(MLMultiArray * buffer) {
                                      [buffer getMutableBytesWithHandler:^(void * bytes, NSInteger size,
                                                                           NSArray<NSNumber *> * strides) {
                                          (void) strides;
                                          std::memset(bytes, 0, (size_t) size);
                                      }];
                                  }];
    }
}
#endif

const void * fresh_state(MLModel * model) {
#if MOSS_COREML_STATE_SDK
    if (@available(macOS 15.0, iOS 18.0, *)) {
        MLState * state = [model newState];
        if (state == nil) return nullptr;
        zero_state(model, state);
        return CFBridgingRetain(state);
    }
#endif
    (void) model;
    return nullptr;
}

}  // namespace

struct moss_coreml_model {
    const void *         model    = nullptr;
    const void *         provider = nullptr;
    const void *         options  = nullptr;
    const void *         state    = nullptr;
    bool                 stateful = false;
    std::vector<Feature> inputs;
    std::vector<Feature> outputs;
    std::string          label;
    std::mutex           mutex;
};

namespace {

void release_model(moss_coreml_model * model) {
    release_features(&model->inputs);
    release_features(&model->outputs);
    if (model->state != nullptr) CFRelease(model->state);
    if (model->options != nullptr) CFRelease(model->options);
    if (model->provider != nullptr) CFRelease(model->provider);
    if (model->model != nullptr) CFRelease(model->model);
    delete model;
}

bool bind_io(moss_coreml_model * model, MLModel * mlmodel) {
    if (!describe_features(mlmodel.modelDescription.inputDescriptionsByName, &model->inputs)) return false;
    if (!describe_features(mlmodel.modelDescription.outputDescriptionsByName, &model->outputs)) return false;
    NSError * err = nil;
    MLDictionaryFeatureProvider * provider =
        [[MLDictionaryFeatureProvider alloc] initWithDictionary:feature_dictionary(model->inputs) error:&err];
    if (provider == nil || err != nil) return false;
    MLPredictionOptions * options = [[MLPredictionOptions alloc] init];
    options.outputBackings = feature_dictionary(model->outputs);
    model->provider = CFBridgingRetain(provider);
    model->options  = CFBridgingRetain(options);
    return true;
}

bool bind_state(moss_coreml_model * model, MLModel * mlmodel) {
    model->stateful = model_declares_state(mlmodel);
    if (!model->stateful) return true;
    model->state = fresh_state(mlmodel);
    return model->state != nullptr;
}

bool write_inputs(moss_coreml_model * model, const char * const * names, const float * const * data, int count) {
    if (count != (int) model->inputs.size()) return false;
    for (int i = 0; i < count; ++i) {
        const Feature * feature = find_feature(model->inputs, names[i]);
        if (feature == nullptr || data[i] == nullptr) return false;
        write_view(view_of((__bridge MLMultiArray *) feature->array), data[i]);
    }
    return true;
}

bool read_outputs(moss_coreml_model * model, id<MLFeatureProvider> result, const char * const * names,
                  float * const * data, int count) {
    for (int i = 0; i < count; ++i) {
        const Feature * feature = find_feature(model->outputs, names[i]);
        if (feature == nullptr || data[i] == nullptr) return false;
        MLMultiArray * array = [result featureValueForName:[NSString stringWithUTF8String:names[i]]].multiArrayValue;
        if (array == nil || element_count(dims_of(array.shape)) != element_count(feature->dims)) return false;
        read_view(view_of(array), data[i]);
    }
    return true;
}

id<MLFeatureProvider> run_prediction(moss_coreml_model * model, NSError ** err) {
    MLModel *                     mlmodel  = (__bridge MLModel *) model->model;
    MLDictionaryFeatureProvider * provider = (__bridge MLDictionaryFeatureProvider *) model->provider;
    MLPredictionOptions *         options  = (__bridge MLPredictionOptions *) model->options;
#if MOSS_COREML_STATE_SDK
    if (model->stateful) {
        if (@available(macOS 15.0, iOS 18.0, *)) {
            return [mlmodel predictionFromFeatures:provider
                                        usingState:(__bridge MLState *) model->state
                                           options:options
                                             error:err];
        }
        return nil;
    }
#endif
    return [mlmodel predictionFromFeatures:provider options:options error:err];
}

}  // namespace

struct moss_coreml_model * moss_coreml_load(const char * path_mlmodelc, const char * compute_units) {
    if (path_mlmodelc == nullptr) return nullptr;
    @autoreleasepool {
        const placement where   = placement_named(compute_units);
        MLModel *       mlmodel = load_model(path_mlmodelc, where);
        if (mlmodel == nil) return nullptr;
        auto * model  = new moss_coreml_model();
        model->model  = CFBridgingRetain(mlmodel);
        model->label  = where.label;
        if (!bind_io(model, mlmodel) || !bind_state(model, mlmodel)) {
            std::fprintf(stderr, "Core ML: cannot bind model inputs, outputs or state (%s)\n", path_mlmodelc);
            release_model(model);
            return nullptr;
        }
        return model;
    }
}

void moss_coreml_free(struct moss_coreml_model * model) {
    if (model != nullptr) release_model(model);
}

int moss_coreml_feature_dims(const struct moss_coreml_model * model, const char * name, int64_t * dims,
                             int max_dims) {
    if (model == nullptr || dims == nullptr) return 0;
    const Feature * feature = find_feature(model->inputs, name);
    if (feature == nullptr) feature = find_feature(model->outputs, name);
    if (feature == nullptr || (int) feature->dims.size() > max_dims) return 0;
    std::copy(feature->dims.begin(), feature->dims.end(), dims);
    return (int) feature->dims.size();
}

int moss_coreml_has_state(const struct moss_coreml_model * model) {
    return model != nullptr && model->stateful ? 1 : 0;
}

int moss_coreml_reset_state(struct moss_coreml_model * model) {
    if (model == nullptr || !model->stateful) return 1;
    @autoreleasepool {
        std::lock_guard<std::mutex> lock(model->mutex);
        const void * state = fresh_state((__bridge MLModel *) model->model);
        if (state == nullptr) return 2;
        CFRelease(model->state);
        model->state = state;
        return 0;
    }
}

int moss_coreml_predict(struct moss_coreml_model * model,
                        const char * const * input_names, const float * const * inputs, int n_inputs,
                        const char * const * output_names, float * const * outputs, int n_outputs) {
    if (model == nullptr || input_names == nullptr || inputs == nullptr || output_names == nullptr ||
        outputs == nullptr) return 1;
    @autoreleasepool {
        std::lock_guard<std::mutex> lock(model->mutex);
        if (!write_inputs(model, input_names, inputs, n_inputs)) return 2;
        NSError *             err    = nil;
        id<MLFeatureProvider> result = run_prediction(model, &err);
        if (result == nil || err != nil) return 3;
        return read_outputs(model, result, output_names, outputs, n_outputs) ? 0 : 4;
    }
}

const char * moss_coreml_label(const struct moss_coreml_model * model) {
    return model != nullptr ? model->label.c_str() : "coreml";
}
