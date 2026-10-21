#if !__has_feature(objc_arc)
#error "parakeet-diarizer.mm must be compiled with -fobjc-arc"
#endif

#import "coreml/parakeet-diarizer.h"
#include "coreml/coreml_diarizers.h"
#include "coreml/parakeet_coreml_units.h"
#include "parakeet_coreml_path.h"
#include "parakeet_log.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <dirent.h>
#include <sys/stat.h>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {

constexpr const char * kValidMaskName = "valid_mask";
constexpr int kTrailingDims = 2;

enum DiarizerStatus {
    kOk = 0,
    kBadArguments = 1,
    kBadSlice = 2,
    kPredictionFailed = 3,
    kMissingOutput = 4,
};

struct Binding {
    std::string name;
    const void * array = nullptr;
    int64_t rows = 0;
    int64_t columns = 0;
};

MLMultiArray * bridged(const void * array) {
    return (__bridge MLMultiArray *) array;
}

bool trailing_rows_and_columns(NSArray<NSNumber *> * shape, int64_t * rows, int64_t * columns) {
    if (shape.count == 0) return false;
    if (shape.count == 1) {
        *rows = shape[0].longLongValue;
        *columns = 1;
        return *rows > 0;
    }
    *rows = shape[shape.count - kTrailingDims].longLongValue;
    *columns = shape[shape.count - 1].longLongValue;
    return *rows > 0 && *columns > 0;
}

bool is_contiguous(MLMultiArray * array, int64_t columns) {
    NSArray<NSNumber *> * strides = array.strides;
    if (strides.count == 0 || strides[strides.count - 1].longLongValue != 1) return false;
    return strides.count == 1 || strides[strides.count - kTrailingDims].longLongValue == columns;
}

MLMultiArray * float_array(MLMultiArrayConstraint * constraint) {
    if (constraint == nil || constraint.dataType != MLMultiArrayDataTypeFloat32) return nil;
    NSError * error = nil;
    MLMultiArray * array = [[MLMultiArray alloc] initWithShape:constraint.shape
                                                      dataType:MLMultiArrayDataTypeFloat32
                                                         error:&error];
    return error == nil ? array : nil;
}

bool bind_array(MLMultiArrayConstraint * constraint, NSString * name, Binding & binding,
                MLMultiArray ** array) {
    *array = float_array(constraint);
    if (*array == nil) return false;
    if (!trailing_rows_and_columns(constraint.shape, &binding.rows, &binding.columns)) return false;
    if (!is_contiguous(*array, binding.columns)) return false;
    binding.name = name.UTF8String;
    binding.array = CFBridgingRetain(*array);
    return true;
}

void release_bindings(std::vector<Binding> & bindings) {
    for (Binding & binding : bindings) {
        if (binding.array != nullptr) CFRelease(binding.array);
        binding.array = nullptr;
    }
    bindings.clear();
}

const parakeet_coreml_rows * find_input(const parakeet_coreml_rows * inputs, int count,
                                        const std::string & name) {
    for (int index = 0; index < count; ++index) {
        if (inputs[index].name != nullptr && name == inputs[index].name) return &inputs[index];
    }
    return nullptr;
}

const Binding * find_binding(const std::vector<Binding> & bindings, const char * name) {
    for (const Binding & binding : bindings) {
        if (name != nullptr && binding.name == name) return &binding;
    }
    return nullptr;
}

bool slice_fits(int64_t row_offset, int64_t rows, int64_t columns, const Binding & binding) {
    return row_offset >= 0 && rows >= 0 && row_offset + rows <= binding.rows &&
           columns == binding.columns;
}

bool fill_input(const Binding & binding, const parakeet_coreml_rows * slice) {
    if (slice == nullptr) return false;
    float * destination = (float *) bridged(binding.array).dataPointer;
    const int64_t total = binding.rows * binding.columns;
    if (slice->rows == 0) {
        std::fill(destination, destination + total, 0.0f);
        return true;
    }
    if (slice->data == nullptr ||
        !slice_fits(slice->row_offset, slice->rows, slice->columns, binding)) return false;
    const int64_t begin = slice->row_offset * binding.columns;
    const int64_t end = begin + slice->rows * binding.columns;
    std::fill(destination, destination + begin, 0.0f);
    std::memcpy(destination + begin, slice->data, (size_t) (end - begin) * sizeof(float));
    std::fill(destination + end, destination + total, 0.0f);
    return true;
}

void fill_mask(const Binding & mask, int64_t valid_frames) {
    float * destination = (float *) bridged(mask.array).dataPointer;
    std::fill(destination, destination + mask.rows, 0.0f);
    std::fill(destination, destination + valid_frames, 1.0f);
}

void copy_row(const float * line, int64_t column_stride, int64_t columns, float * destination) {
    for (int64_t column = 0; column < columns; ++column) {
        destination[column] = line[column * column_stride];
    }
}

bool copy_rows(MLMultiArray * array, const parakeet_coreml_rows_out & output) {
    if (array == nil || array.dataType != MLMultiArrayDataTypeFloat32) return false;
    NSArray<NSNumber *> * strides = array.strides;
    const int64_t row_stride = strides.count >= kTrailingDims
        ? strides[strides.count - kTrailingDims].longLongValue : 1;
    const int64_t column_stride = strides[strides.count - 1].longLongValue;
    const float * source = (const float *) array.dataPointer;
    for (int64_t row = 0; row < output.rows; ++row) {
        copy_row(source + (output.row_offset + row) * row_stride, column_stride,
                 output.columns, output.data + row * output.columns);
    }
    return true;
}

}  // namespace

struct parakeet_coreml_diarizer {
    const void * model = nullptr;
    const void * provider = nullptr;
    const void * options = nullptr;
    std::vector<Binding> inputs;
    std::vector<Binding> outputs;
    Binding mask;
    std::string label;
    std::mutex mutex;
};

namespace {

bool bind_inputs(parakeet_coreml_diarizer * ctx, MLModel * model,
                 NSMutableDictionary<NSString *, MLMultiArray *> * features) {
    NSDictionary<NSString *, MLFeatureDescription *> * descriptions =
        model.modelDescription.inputDescriptionsByName;
    for (NSString * name in descriptions) {
        Binding binding;
        MLMultiArray * array = nil;
        if (!bind_array(descriptions[name].multiArrayConstraint, name, binding, &array)) return false;
        features[name] = array;
        if (binding.name == kValidMaskName) {
            ctx->mask = binding;
        } else {
            ctx->inputs.push_back(binding);
        }
    }
    return ctx->mask.array != nullptr;
}

bool inputs_share_capacity(const parakeet_coreml_diarizer * ctx) {
    for (const Binding & binding : ctx->inputs) {
        if (binding.rows != ctx->mask.rows) return false;
    }
    return true;
}

bool bind_outputs(parakeet_coreml_diarizer * ctx, MLModel * model,
                  NSMutableDictionary<NSString *, id> * backings) {
    NSDictionary<NSString *, MLFeatureDescription *> * descriptions =
        model.modelDescription.outputDescriptionsByName;
    for (NSString * name in descriptions) {
        Binding binding;
        MLMultiArray * array = nil;
        if (!bind_array(descriptions[name].multiArrayConstraint, name, binding, &array)) return false;
        backings[name] = array;
        ctx->outputs.push_back(binding);
    }
    return !ctx->outputs.empty();
}

bool bind_model(parakeet_coreml_diarizer * ctx, MLModel * model) {
    NSMutableDictionary<NSString *, MLMultiArray *> * features = [NSMutableDictionary dictionary];
    NSMutableDictionary<NSString *, id> * backings = [NSMutableDictionary dictionary];
    if (!bind_inputs(ctx, model, features) || !inputs_share_capacity(ctx)) return false;
    if (!bind_outputs(ctx, model, backings)) return false;
    NSError * error = nil;
    MLDictionaryFeatureProvider * provider =
        [[MLDictionaryFeatureProvider alloc] initWithDictionary:features error:&error];
    if (provider == nil || error != nil) return false;
    MLPredictionOptions * options = [[MLPredictionOptions alloc] init];
    options.outputBackings = backings;
    ctx->provider = CFBridgingRetain(provider);
    ctx->options = CFBridgingRetain(options);
    return true;
}

bool inputs_bound(const parakeet_coreml_diarizer * ctx, const parakeet_coreml_rows * inputs,
                  int count) {
    for (int index = 0; index < count; ++index) {
        if (find_binding(ctx->inputs, inputs[index].name) == nullptr) return false;
    }
    return true;
}

bool fill_inputs(parakeet_coreml_diarizer * ctx, const parakeet_coreml_rows * inputs, int count) {
    for (const Binding & binding : ctx->inputs) {
        if (!fill_input(binding, find_input(inputs, count, binding.name))) return false;
    }
    return true;
}

bool outputs_fit(const parakeet_coreml_diarizer * ctx, const parakeet_coreml_rows_out * outputs,
                 int count) {
    for (int index = 0; index < count; ++index) {
        const Binding * binding = find_binding(ctx->outputs, outputs[index].name);
        if (binding == nullptr || outputs[index].data == nullptr ||
            !slice_fits(outputs[index].row_offset, outputs[index].rows, outputs[index].columns,
                        *binding)) return false;
    }
    return true;
}

bool read_outputs(id<MLFeatureProvider> result, const parakeet_coreml_rows_out * outputs,
                  int count) {
    for (int index = 0; index < count; ++index) {
        NSString * name = [NSString stringWithUTF8String:outputs[index].name];
        if (!copy_rows([result featureValueForName:name].multiArrayValue, outputs[index])) return false;
    }
    return true;
}

}  // namespace

struct parakeet_coreml_diarizer * parakeet_coreml_diarizer_init(const char * path_mlmodelc,
                                                                const char * units_env) {
    if (path_mlmodelc == nullptr) return nullptr;
    @autoreleasepool {
        std::string label;
        MLModelConfiguration * config = [[MLModelConfiguration alloc] init];
        config.computeUnits = parakeet_coreml_requested_units(
            units_env, /*prefer_neural_engine=*/true, &label);
        NSError * error = nil;
        NSURL * url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path_mlmodelc]];
        MLModel * model = [MLModel modelWithContentsOfURL:url configuration:config error:&error];
        if (model == nil || error != nil) return nullptr;
        auto * ctx = new parakeet_coreml_diarizer();
        ctx->model = CFBridgingRetain(model);
        ctx->label = label;
        if (!bind_model(ctx, model)) {
            parakeet_coreml_diarizer_free(ctx);
            return nullptr;
        }
        return ctx;
    }
}

void parakeet_coreml_diarizer_free(struct parakeet_coreml_diarizer * ctx) {
    if (ctx == nullptr) return;
    release_bindings(ctx->inputs);
    release_bindings(ctx->outputs);
    if (ctx->mask.array != nullptr) CFRelease(ctx->mask.array);
    if (ctx->options != nullptr) CFRelease(ctx->options);
    if (ctx->provider != nullptr) CFRelease(ctx->provider);
    if (ctx->model != nullptr) CFRelease(ctx->model);
    delete ctx;
}

int64_t parakeet_coreml_diarizer_capacity(const struct parakeet_coreml_diarizer * ctx) {
    return ctx != nullptr ? ctx->mask.rows : 0;
}

int parakeet_coreml_diarizer_run(struct parakeet_coreml_diarizer * ctx,
                                 int64_t valid_frames,
                                 const struct parakeet_coreml_rows * inputs,
                                 int n_inputs,
                                 const struct parakeet_coreml_rows_out * outputs,
                                 int n_outputs) {
    if (ctx == nullptr || valid_frames <= 0 || valid_frames > ctx->mask.rows ||
        n_inputs < 0 || n_outputs <= 0 || outputs == nullptr) return kBadArguments;
    @autoreleasepool {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        if (!inputs_bound(ctx, inputs, n_inputs) || !fill_inputs(ctx, inputs, n_inputs) ||
            !outputs_fit(ctx, outputs, n_outputs)) {
            return kBadSlice;
        }
        fill_mask(ctx->mask, valid_frames);
        NSError * error = nil;
        MLModel * model = (__bridge MLModel *) ctx->model;
        id<MLFeatureProvider> result =
            [model predictionFromFeatures:(__bridge MLDictionaryFeatureProvider *) ctx->provider
                                  options:(__bridge MLPredictionOptions *) ctx->options
                                    error:&error];
        if (result == nil || error != nil) return kPredictionFailed;
        return read_outputs(result, outputs, n_outputs) ? kOk : kMissingOutput;
    }
}

const char * parakeet_coreml_diarizer_label(const struct parakeet_coreml_diarizer * ctx) {
    return ctx != nullptr ? ctx->label.c_str() : "coreml";
}

namespace parakeet {
namespace {

constexpr int kNoSidecar = -1;
constexpr char kPathSeparator = '/';
constexpr const char * kCurrentDirectory = ".";

std::string directory_of(const std::string & path) {
    const std::size_t slash = path.find_last_of(kPathSeparator);
    return slash == std::string::npos ? std::string() : path.substr(0, slash + 1);
}

bool is_directory(const std::string & path) {
    struct stat status {};
    return ::stat(path.c_str(), &status) == 0 && S_ISDIR(status.st_mode);
}

std::vector<int> listed_capacities(DIR * listing, const std::string & directory,
                                   const std::string & gguf_path) {
    std::vector<int> capacities;
    while (const dirent * entry = ::readdir(listing)) {
        const std::string candidate = directory + entry->d_name;
        const int capacity = coreml_diarizer_sidecar_capacity(gguf_path, candidate);
        if (capacity > 0 && is_directory(candidate)) capacities.push_back(capacity);
    }
    return capacities;
}

std::vector<int> scan_capacities(const std::string & gguf_path) {
    const std::string directory = directory_of(gguf_path);
    DIR * listing = ::opendir(directory.empty() ? kCurrentDirectory : directory.c_str());
    if (listing == nullptr) return {};
    std::vector<int> capacities = listed_capacities(listing, directory, gguf_path);
    ::closedir(listing);
    std::sort(capacities.begin(), capacities.end());
    capacities.erase(std::unique(capacities.begin(), capacities.end()), capacities.end());
    return capacities;
}

constexpr const char * kWeightsFile = "/weights/weight.bin";

uint64_t file_bytes(const std::string & path) {
    struct stat status {};
    return ::stat(path.c_str(), &status) == 0 ? static_cast<uint64_t>(status.st_size) : 0;
}

CoremlDiarizerBundle bundle_of(const std::string & gguf_path, int capacity) {
    return {capacity, file_bytes(coreml_diarizer_sidecar_path(gguf_path, capacity) + kWeightsFile)};
}

}

bool coreml_diarizers_disabled() {
    return std::getenv(kCoremlDisableEnv) != nullptr ||
           std::getenv(kCoremlDiarizerDisableEnv) != nullptr;
}

std::vector<CoremlDiarizerBundle> list_coreml_diarizer_bundles(const std::string & gguf_path) {
    const std::vector<int> capacities = scan_capacities(gguf_path);
    std::vector<CoremlDiarizerBundle> bundles(capacities.size());
    std::transform(capacities.begin(), capacities.end(), bundles.begin(),
                   [&](int capacity) { return bundle_of(gguf_path, capacity); });
    return bundles;
}

CoremlDiarizers::CoremlDiarizers(std::string gguf_path, std::vector<int> capacities)
    : gguf_path_(std::move(gguf_path)), slots_(capacities.size()) {
    std::transform(capacities.begin(), capacities.end(), slots_.begin(),
                   [](int capacity) { return Slot{capacity, nullptr, false}; });
}

CoremlDiarizers::~CoremlDiarizers() {
    std::for_each(slots_.begin(), slots_.end(),
                  [](Slot & slot) { parakeet_coreml_diarizer_free(slot.context); });
}

CoremlDiarizers::Slot * CoremlDiarizers::slot_for(int frames) {
    for (Slot & slot : slots_) {
        if (!slot.failed && slot.capacity >= frames) return &slot;
    }
    return nullptr;
}

bool CoremlDiarizers::load(Slot & slot) {
    if (slot.context != nullptr) return true;
    const std::string path = coreml_diarizer_sidecar_path(gguf_path_, slot.capacity);
    slot.context = parakeet_coreml_diarizer_init(path.c_str(), kCoremlDiarizerUnitsEnv);
    if (slot.context == nullptr ||
        parakeet_coreml_diarizer_capacity(slot.context) != slot.capacity) {
        PARAKEET_LOG_WARN("parakeet: failed to load Core ML diarizer at '%s'; using ggml\n",
                          path.c_str());
        quarantine(slot);
        return false;
    }
    label_ = parakeet_coreml_diarizer_label(slot.context);
    PARAKEET_LOG_INFO("parakeet: Core ML diarizer loaded from '%s'\n", path.c_str());
    return true;
}

void CoremlDiarizers::quarantine(Slot & slot) {
    parakeet_coreml_diarizer_free(slot.context);
    slot.context = nullptr;
    slot.failed = true;
}

CoremlDiarizers::Slot * CoremlDiarizers::loaded_slot_for(int frames) {
    for (Slot * slot = slot_for(frames); slot != nullptr; slot = slot_for(frames)) {
        if (load(*slot)) return slot;
    }
    return nullptr;
}

bool CoremlDiarizers::covers(int frames) {
    std::lock_guard<std::mutex> lock(mutex_);
    return frames > 0 && slot_for(frames) != nullptr;
}

bool CoremlDiarizers::prepare(int frames) {
    std::lock_guard<std::mutex> lock(mutex_);
    return frames > 0 && loaded_slot_for(frames) != nullptr;
}

int CoremlDiarizers::run(int valid_frames,
                         const parakeet_coreml_rows * inputs, int n_inputs,
                         const parakeet_coreml_rows_out * outputs, int n_outputs) {
    std::lock_guard<std::mutex> lock(mutex_);
    Slot * slot = valid_frames > 0 ? loaded_slot_for(valid_frames) : nullptr;
    if (slot == nullptr) return kNoSidecar;
    const int status = parakeet_coreml_diarizer_run(slot->context, valid_frames,
                                                    inputs, n_inputs, outputs, n_outputs);
    if (status != 0) {
        PARAKEET_LOG_WARN("parakeet: Core ML diarizer (%d frames) failed (rc=%d); "
                          "quarantining it and using ggml\n", slot->capacity, status);
        quarantine(*slot);
    }
    return status;
}

std::string CoremlDiarizers::label() {
    std::lock_guard<std::mutex> lock(mutex_);
    return label_.empty() ? parakeet_coreml_diarizer_label(nullptr) : label_;
}

std::vector<int> CoremlDiarizers::capacities() const {
    std::vector<int> result(slots_.size());
    std::transform(slots_.begin(), slots_.end(), result.begin(),
                   [](const Slot & slot) { return slot.capacity; });
    return result;
}

std::unique_ptr<CoremlDiarizers> discover_coreml_diarizers(const std::string & gguf_path) {
    std::vector<int> capacities = scan_capacities(gguf_path);
    if (capacities.empty()) return nullptr;
    return std::make_unique<CoremlDiarizers>(gguf_path, std::move(capacities));
}

}
