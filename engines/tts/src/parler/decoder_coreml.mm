#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include "decoder_coreml.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <exception>
#include <limits>
#include <utility>

#if !__has_feature(objc_arc)
#error "decoder_coreml.mm requires -fobjc-arc"
#endif

namespace tts_cpp {
namespace parler {
namespace detail {
namespace {

using clock_type = std::chrono::steady_clock;
using dtype = decoder_coreml_dtype;
using buffer = decoder_coreml_buffer;

bool fail(std::string & error, const std::string & text) {
    error = text;
    return false;
}

template<class F> bool guarded(std::string & error, F && f) {
    error.clear();
    @try {
        try {
            return f();
        } catch (const std::exception & e) {
            return fail(error, e.what());
        }
    } @catch (NSException * e) {
        return fail(error, std::string("Core ML exception: ") + (e.reason.UTF8String ?: "unknown"));
    }
}

size_t scalar_bytes(dtype type) { return type == dtype::float16 ? 2 : 4; }

size_t element_count(const std::vector<int64_t> & shape) {
    if (shape.empty()) return 0;
    size_t count = 1;
    for (int64_t d : shape) {
        if (d <= 0 || (uint64_t) d > std::numeric_limits<size_t>::max() / count) return 0;
        count *= (size_t) d;
    }
    return count;
}

bool layout_valid(const buffer & b, std::string & error) {
    if (!b.data || !element_count(b.shape) || b.shape.size() != b.strides.size()) {
        return fail(error, "invalid tensor pointer, shape or strides");
    }
    size_t last = 0;
    for (size_t d = 0; d < b.shape.size(); ++d) {
        if (b.strides[d] < 0 || (b.shape[d] > 1 && b.strides[d] == 0)) {
            return fail(error, "negative or zero tensor stride");
        }
        const size_t extent = (size_t) (b.shape[d] - 1);
        const size_t stride = (size_t) b.strides[d];
        if (extent && stride > (std::numeric_limits<size_t>::max() - last) / extent) {
            return fail(error, "tensor stride overflow");
        }
        last += extent * stride;
    }
    if (b.bytes < scalar_bytes(b.dtype) || last > (b.bytes - scalar_bytes(b.dtype)) / scalar_bytes(b.dtype)) {
        return fail(error, "tensor layout exceeds storage");
    }
    return true;
}

buffer contiguous(void * data, size_t count, dtype type, const std::vector<int64_t> & shape) {
    buffer b;
    b.data = data;
    b.bytes = count <= std::numeric_limits<size_t>::max() / scalar_bytes(type) ? count * scalar_bytes(type) : 0;
    b.dtype = type;
    b.shape = shape;
    b.strides.resize(shape.size());
    int64_t stride = 1;
    for (size_t d = shape.size(); d-- > 0;) {
        b.strides[d] = stride;
        if (shape[d] <= 0 || stride > std::numeric_limits<int64_t>::max() / shape[d]) {
            b.bytes = 0;
            break;
        }
        stride *= shape[d];
    }
    return b;
}

float read_float(const unsigned char * src, dtype type) {
    if (type == dtype::float16) {
        __fp16 x;
        std::memcpy(&x, src, sizeof(x));
        return (float) x;
    }
    float x;
    std::memcpy(&x, src, sizeof(x));
    return x;
}

void write_float(unsigned char * dst, dtype type, float x) {
    if (type == dtype::float16) {
        const __fp16 h = (__fp16) x;
        std::memcpy(dst, &h, sizeof(h));
    } else {
        std::memcpy(dst, &x, sizeof(x));
    }
}

std::vector<int64_t> numbers(NSArray<NSNumber *> * values) {
    std::vector<int64_t> result;
    result.reserve(values.count);
    for (NSNumber * n in values) result.push_back(n.longLongValue);
    return result;
}

bool data_type(MLMultiArrayDataType type, dtype & result) {
    if (type == MLMultiArrayDataTypeFloat16) result = dtype::float16;
    else if (type == MLMultiArrayDataTypeFloat32) result = dtype::float32;
    else if (type == MLMultiArrayDataTypeInt32) result = dtype::int32;
    else return false;
    return true;
}

buffer array_buffer(MLMultiArray * array, void * bytes, NSInteger size, NSArray<NSNumber *> * strides) {
    buffer result;
    result.data = bytes;
    result.bytes = size > 0 ? (size_t) size : 0;
    result.shape = numbers(array.shape);
    result.strides = numbers(strides);
    // Callers verify the data type before constructing a view.
    data_type(array.dataType, result.dtype);
    return result;
}

bool matches(MLMultiArray * array, const decoder_coreml_tensor & description, std::string & error) {
    dtype type;
    if (!array || !data_type(array.dataType, type) || type != description.dtype ||
        numbers(array.shape) != description.shape) {
        return fail(error, "unexpected runtime shape or dtype for " + description.name);
    }
    return true;
}

double elapsed_ms(clock_type::time_point start, clock_type::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

} // namespace

size_t decoder_coreml_tensor::elements() const { return element_count(shape); }

bool decoder_coreml_copy_region(const buffer & source, const buffer & destination,
                                const std::vector<int64_t> & origin, std::string & error) {
    error.clear();
    if (!layout_valid(source, error) || !layout_valid(destination, error)) return false;
    const size_t rank = source.shape.size();
    if (rank != destination.shape.size() || rank != origin.size()) return fail(error, "tensor region rank mismatch");
    if ((source.dtype == dtype::int32) != (destination.dtype == dtype::int32)) {
        return fail(error, "int32/float conversion is not supported");
    }
    size_t dst_base = 0;
    for (size_t d = 0; d < rank; ++d) {
        if (origin[d] < 0 || source.shape[d] > destination.shape[d] ||
            origin[d] > destination.shape[d] - source.shape[d]) return fail(error, "tensor region out of bounds");
        dst_base += (size_t) origin[d] * (size_t) destination.strides[d];
    }
    const uintptr_t src_addr = (uintptr_t) source.data, dst_addr = (uintptr_t) destination.data;
    if ((src_addr <= dst_addr && dst_addr - src_addr < source.bytes) ||
        (dst_addr <= src_addr && src_addr - dst_addr < destination.bytes)) return fail(error, "tensor buffers alias");

    const size_t src_size = scalar_bytes(source.dtype), dst_size = scalar_bytes(destination.dtype);
    const bool same_type = source.dtype == destination.dtype;
    const bool row_contiguous = source.strides.back() == 1 && destination.strides.back() == 1;
    const size_t width = (size_t) source.shape.back();
    const size_t rows = element_count(source.shape) / width;
    const auto * src = static_cast<const unsigned char *>(source.data);
    auto * dst = static_cast<unsigned char *>(destination.data);
    for (size_t r = 0; r < rows; ++r) {
        size_t remaining = r, src_offset = 0, dst_offset = dst_base;
        for (size_t d = rank - 1; d-- > 0;) {
            const size_t coordinate = remaining % (size_t) source.shape[d];
            remaining /= (size_t) source.shape[d];
            src_offset += coordinate * (size_t) source.strides[d];
            dst_offset += coordinate * (size_t) destination.strides[d];
        }
        if (same_type && row_contiguous) {
            std::memcpy(dst + dst_offset * dst_size, src + src_offset * src_size, width * src_size);
        } else {
            for (size_t c = 0; c < width; ++c) {
                const auto * from = src + (src_offset + c * (size_t) source.strides.back()) * src_size;
                auto * to = dst + (dst_offset + c * (size_t) destination.strides.back()) * dst_size;
                if (same_type) std::memcpy(to, from, src_size);
                else write_float(to, destination.dtype, read_float(from, source.dtype));
            }
        }
    }
    return true;
}

#ifndef PARLER_COREML_STATE_SDK
#if (defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000) || \
    (defined(__IPHONE_OS_VERSION_MAX_ALLOWED) && __IPHONE_OS_VERSION_MAX_ALLOWED >= 180000)
#define PARLER_COREML_STATE_SDK 1
#else
#define PARLER_COREML_STATE_SDK 0
#endif
#endif

#if PARLER_COREML_STATE_SDK

struct decoder_coreml_session::impl {
    MLModel * model = nil;
    MLState * state API_AVAILABLE(macos(15.0), ios(18.0)) = nil;
    MLDictionaryFeatureProvider * provider = nil;
    id<MLFeatureProvider> result = nil;
    NSMutableDictionary<NSString *, MLMultiArray *> * arrays = nil;
    std::vector<decoder_coreml_tensor> in, out, state_info;
    std::vector<bool> initialized;
    bool healthy = true;

    const decoder_coreml_tensor * find(const std::vector<decoder_coreml_tensor> & list, const std::string & name) const {
        for (const auto & t : list) if (t.name == name) return &t;
        return nullptr;
    }

    bool ready(std::string & error) const {
        return healthy || fail(error, "Core ML session requires reset after prediction/state-write failure");
    }

    bool describe(NSDictionary<NSString *, MLFeatureDescription *> * descriptions,
                  std::vector<decoder_coreml_tensor> & destination, bool is_state,
                  std::string & error) API_AVAILABLE(macos(15.0), ios(18.0)) {
        for (NSString * name in [[descriptions allKeys] sortedArrayUsingSelector:@selector(compare:)]) {
            MLFeatureDescription * desc = descriptions[name];
            decoder_coreml_tensor tensor;
            tensor.name = name.UTF8String;
            MLMultiArrayDataType type;
            if (is_state) {
                if (!desc.stateConstraint) return fail(error, "missing state constraint: " + tensor.name);
                tensor.shape = numbers(desc.stateConstraint.bufferShape);
                type = desc.stateConstraint.dataType;
            } else {
                if (desc.type != MLFeatureTypeMultiArray || !desc.multiArrayConstraint) {
                    return fail(error, "expected multi-array feature: " + tensor.name);
                }
                tensor.shape = numbers(desc.multiArrayConstraint.shape);
                type = desc.multiArrayConstraint.dataType;
            }
            if (!data_type(type, tensor.dtype) || !tensor.elements() ||
                tensor.elements() > std::numeric_limits<size_t>::max() / scalar_bytes(tensor.dtype)) {
                return fail(error, "unsupported shape/dtype: " + tensor.name);
            }
            destination.push_back(std::move(tensor));
        }
        return true;
    }

    bool set_input(const std::string & name, const void * data, size_t count, dtype type, std::string & error) {
        if (!ready(error)) return false;
        const auto * desc = find(in, name);
        if (!desc) return fail(error, "unknown input: " + name);
        if (!data || count != desc->elements()) return fail(error, "input element count mismatch: " + name);
        if ((type == dtype::int32) != (desc->dtype == dtype::int32)) return fail(error, "input dtype mismatch: " + name);
        MLMultiArray * array = arrays[[NSString stringWithUTF8String:name.c_str()]];
        if (!matches(array, *desc, error)) return false;
        const buffer src = contiguous(const_cast<void *>(data), count, type, desc->shape);
        __block bool ok = false;
        std::string * err = &error;
        [array getMutableBytesWithHandler:^(void * bytes, NSInteger size, NSArray<NSNumber *> * strides) {
            ok = decoder_coreml_copy_region(src, array_buffer(array, bytes, size, strides),
                                            std::vector<int64_t>(desc->shape.size(), 0), *err);
        }];
        if (ok) initialized[(size_t) (desc - in.data())] = true;
        return ok;
    }

    bool copy_array(MLMultiArray * array, const decoder_coreml_tensor & desc,
                    float * data, size_t count, std::string & error) const {
        if (!data || count != desc.elements()) return fail(error, "output/state element count mismatch: " + desc.name);
        if (desc.dtype == dtype::int32) return fail(error, "float read requires fp16/fp32: " + desc.name);
        if (!matches(array, desc, error)) return false;
        const buffer dst = contiguous(data, count, dtype::float32, desc.shape);
        __block bool ok = false;
        std::string * err = &error;
        [array getBytesWithHandler:^(const void * bytes, NSInteger size) {
            ok = decoder_coreml_copy_region(array_buffer(array, const_cast<void *>(bytes), size, array.strides),
                                            dst, std::vector<int64_t>(desc.shape.size(), 0), *err);
        }];
        return ok;
    }
};

std::unique_ptr<decoder_coreml_session> decoder_coreml_session::open(
    const std::string & path, decoder_coreml_units units, std::string & error) {
    std::unique_ptr<decoder_coreml_session> session;
    guarded(error, [&]() {
        if (@available(macOS 15.0, iOS 18.0, *)) {
            @autoreleasepool {
                auto p = std::unique_ptr<impl>(new impl);
                MLModelConfiguration * config = [[MLModelConfiguration alloc] init];
                switch (units) {
                    case decoder_coreml_units::cpu_only: config.computeUnits = MLComputeUnitsCPUOnly; break;
                    case decoder_coreml_units::cpu_and_gpu: config.computeUnits = MLComputeUnitsCPUAndGPU; break;
                    case decoder_coreml_units::cpu_and_ane: config.computeUnits = MLComputeUnitsCPUAndNeuralEngine; break;
                    case decoder_coreml_units::all: config.computeUnits = MLComputeUnitsAll; break;
                    default: return fail(error, "invalid Core ML compute units");
                }
                NSString * ns_path = [NSString stringWithUTF8String:path.c_str()];
                if (!ns_path || ![ns_path.pathExtension isEqualToString:@"mlmodelc"]) {
                    return fail(error, "expected compiled .mlmodelc path");
                }
                NSError * err = nil;
                p->model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:ns_path] configuration:config error:&err];
                if (!p->model) return fail(error, std::string("Core ML load failed: ") + (err.localizedDescription.UTF8String ?: "unknown"));
                MLModelDescription * desc = p->model.modelDescription;
                if (!p->describe(desc.inputDescriptionsByName, p->in, false, error) ||
                    !p->describe(desc.outputDescriptionsByName, p->out, false, error) ||
                    !p->describe(desc.stateDescriptionsByName, p->state_info, true, error)) return false;
                if (p->state_info.empty()) return fail(error, "model has no state buffers");
                p->arrays = [[NSMutableDictionary alloc] init];
                NSMutableDictionary<NSString *, MLFeatureValue *> * features = [[NSMutableDictionary alloc] init];
                for (const auto & input : p->in) {
                    NSString * name = [NSString stringWithUTF8String:input.name.c_str()];
                    MLFeatureDescription * feature = desc.inputDescriptionsByName[name];
                    MLMultiArray * array = [[MLMultiArray alloc] initWithShape:feature.multiArrayConstraint.shape
                                                                 dataType:feature.multiArrayConstraint.dataType error:&err];
                    if (!array) return fail(error, std::string("input allocation failed: ") + (err.localizedDescription.UTF8String ?: "unknown"));
                    p->arrays[name] = array;
                    features[name] = [MLFeatureValue featureValueWithMultiArray:array];
                }
                p->provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:features error:&err];
                if (!p->provider) return fail(error, std::string("feature provider failed: ") + (err.localizedDescription.UTF8String ?: "unknown"));
                p->initialized.resize(p->in.size(), false);
                p->state = [p->model newState];
                if (!p->state) return fail(error, "Core ML state allocation failed");
                session.reset(new decoder_coreml_session(std::move(p)));
                return true;
            }
        }
        return fail(error, "stateful Core ML requires macOS 15 / iOS 18");
    });
    return session;
}

decoder_coreml_session::decoder_coreml_session(std::unique_ptr<impl> p) : p_(std::move(p)) {}
decoder_coreml_session::~decoder_coreml_session() = default;
const std::vector<decoder_coreml_tensor> & decoder_coreml_session::inputs() const { return p_->in; }
const std::vector<decoder_coreml_tensor> & decoder_coreml_session::outputs() const { return p_->out; }
const std::vector<decoder_coreml_tensor> & decoder_coreml_session::states() const { return p_->state_info; }
std::string decoder_coreml_session::metadata(const std::string & key) const {
    @autoreleasepool {
        id creator = p_->model.modelDescription.metadata[MLModelCreatorDefinedKey];
        if (![creator isKindOfClass:[NSDictionary class]]) return {};
        id value = creator[[NSString stringWithUTF8String:key.c_str()]];
        return [value isKindOfClass:[NSString class]] ? std::string([value UTF8String]) : std::string();
    }
}

bool decoder_coreml_session::set_input(const std::string & name, const float * data, size_t count, std::string & error) {
    return guarded(error, [&]() { @autoreleasepool { return p_->set_input(name, data, count, dtype::float32, error); } });
}

bool decoder_coreml_session::set_input_i32(const std::string & name, const int32_t * data, size_t count, std::string & error) {
    return guarded(error, [&]() { @autoreleasepool { return p_->set_input(name, data, count, dtype::int32, error); } });
}

bool decoder_coreml_session::predict(const std::vector<decoder_coreml_row_update> & updates,
                                    decoder_coreml_timing * timing, std::string & error) {
    if (timing) *timing = {};
    const auto start = clock_type::now();
    const bool ok = guarded(error, [&]() {
        if (@available(macOS 15.0, iOS 18.0, *)) {
            @autoreleasepool {
                if (!p_->ready(error)) return false;
                for (size_t i = 0; i < p_->in.size(); ++i) {
                    if (!p_->initialized[i]) return fail(error, "input has not been set: " + p_->in[i].name);
                }
                size_t planned_bytes = 0;
                for (const auto & u : updates) {
                    const auto * src = p_->find(p_->out, u.output);
                    const auto * dst = p_->find(p_->state_info, u.state);
                    if (!src || !dst) return fail(error, "row update has unknown output/state");
                    if (!src->elements() || !dst->elements() ||
                        src->elements() > std::numeric_limits<size_t>::max() / scalar_bytes(src->dtype) ||
                        dst->elements() > std::numeric_limits<size_t>::max() / scalar_bytes(dst->dtype) ||
                        src->shape.size() != dst->shape.size() || u.axis >= src->shape.size()) {
                        return fail(error, "row update has invalid tensor metadata or axis");
                    }
                    if (u.source_layer < -1 || (u.source_layer >= 0 && u.source_layer >= src->shape[0])) {
                        return fail(error, "row update source layer out of bounds");
                    }
                    if (u.destination_layer < -1 ||
                        (u.destination_layer >= 0 && u.destination_layer >= dst->shape[0])) {
                        return fail(error, "row update destination layer out of bounds");
                    }
                    if (u.axis == 0 && u.destination_layer >= 0) {
                        return fail(error, "row update axis 0 conflicts with destination layer");
                    }
                    if (src->dtype == dtype::int32 || dst->dtype == dtype::int32 ||
                        (u.axis == 0 && u.source_layer >= 0 ? 1 : src->shape[u.axis]) != 1 ||
                        u.row < 0 || u.row >= dst->shape[u.axis]) {
                        return fail(error, "row update has invalid dtype, shape, axis or row");
                    }
                    for (size_t d = 0; d < src->shape.size(); ++d) {
                        const int64_t extent = d == 0 && u.source_layer >= 0 ? 1 : src->shape[d];
                        const int64_t destination_extent = d == 0 && u.destination_layer >= 0 ? 1 : dst->shape[d];
                        if (d != u.axis && extent != destination_extent) return fail(error, "row update dimensions do not match");
                    }
                    const size_t elements = src->elements() / (u.source_layer >= 0 ? (size_t) src->shape[0] : 1);
                    if (elements > (std::numeric_limits<size_t>::max() - planned_bytes) / scalar_bytes(dst->dtype)) {
                        return fail(error, "row update byte count overflow");
                    }
                    planned_bytes += elements * scalar_bytes(dst->dtype);
                }
                // Any error after this point can leave model state partially
                // changed. Require reset even if Core ML itself wrote nothing.
                p_->healthy = false;
                p_->result = nil;
                NSError * err = nil;
                const auto predict_start = clock_type::now();
                p_->result = [p_->model predictionFromFeatures:p_->provider usingState:p_->state error:&err];
                const auto predict_end = clock_type::now();
                if (timing) timing->predict_ms = elapsed_ms(predict_start, predict_end);
                if (!p_->result) return fail(error, std::string("Core ML prediction failed: ") + (err.localizedDescription.UTF8String ?: "unknown"));
                for (const auto & u : updates) {
                    const auto * src_desc = p_->find(p_->out, u.output);
                    const auto * dst_desc = p_->find(p_->state_info, u.state);
                    MLMultiArray * src = [p_->result featureValueForName:[NSString stringWithUTF8String:u.output.c_str()]].multiArrayValue;
                    if (!matches(src, *src_desc, error)) return false;
                    std::vector<int64_t> origin(dst_desc->shape.size(), 0);
                    if (u.destination_layer >= 0) origin[0] = u.destination_layer;
                    origin[u.axis] = u.row;
                    __block bool copied = false;
                    std::string * err_text = &error;
                    [src getBytesWithHandler:^(const void * src_bytes, NSInteger src_size) {
                        buffer source = array_buffer(src, const_cast<void *>(src_bytes), src_size, src.strides);
                        if (!layout_valid(source, *err_text)) return;
                        if (u.source_layer >= 0) {
                            // The validated full layout bounds this offset,
                            // including multiplication by scalar byte width.
                            const size_t offset = (size_t) u.source_layer * (size_t) source.strides[0] * scalar_bytes(source.dtype);
                            source.data = static_cast<unsigned char *>(source.data) + offset;
                            source.bytes -= offset;
                            source.shape[0] = 1;
                        }
                        [p_->state getMultiArrayForStateNamed:[NSString stringWithUTF8String:u.state.c_str()] handler:^(MLMultiArray * dst) {
                            if (!matches(dst, *dst_desc, *err_text)) return;
                            [dst getMutableBytesWithHandler:^(void * dst_bytes, NSInteger dst_size, NSArray<NSNumber *> * strides) {
                                copied = decoder_coreml_copy_region(source, array_buffer(dst, dst_bytes, dst_size, strides), origin, *err_text);
                            }];
                        }];
                    }];
                    if (!copied) return false;
                    if (timing) timing->row_bytes_written += src_desc->elements() /
                        (u.source_layer >= 0 ? (size_t) src_desc->shape[0] : 1) * scalar_bytes(dst_desc->dtype);
                }
                if (timing) timing->row_write_ms = elapsed_ms(predict_end, clock_type::now());
                p_->healthy = true;
                return true;
            }
        }
        return fail(error, "stateful Core ML requires macOS 15 / iOS 18");
    });
    if (timing) timing->total_ms = elapsed_ms(start, clock_type::now());
    return ok;
}

bool decoder_coreml_session::copy_output(const std::string & name, float * data, size_t count, std::string & error) const {
    return guarded(error, [&]() {
        @autoreleasepool {
            if (!p_->ready(error)) return false;
            const auto * desc = p_->find(p_->out, name);
            if (!desc) return fail(error, "unknown output: " + name);
            if (!p_->result) return fail(error, "no successful prediction available");
            return p_->copy_array([p_->result featureValueForName:[NSString stringWithUTF8String:name.c_str()]].multiArrayValue,
                                  *desc, data, count, error);
        }
    });
}

bool decoder_coreml_session::write_state(const std::string & name, const std::vector<int64_t> & origin,
                                        const std::vector<int64_t> & shape, const float * data,
                                        size_t count, std::string & error) {
    return guarded(error, [&]() {
        if (@available(macOS 15.0, iOS 18.0, *)) {
            @autoreleasepool {
                if (!p_->ready(error)) return false;
                const auto * desc = p_->find(p_->state_info, name);
                if (!desc) return fail(error, "unknown state: " + name);
                if (!data || !element_count(shape) || count != element_count(shape) ||
                    shape.size() != desc->shape.size() || origin.size() != shape.size() || desc->dtype == dtype::int32) {
                    return fail(error, "state write shape/dtype/count mismatch");
                }
                for (size_t d = 0; d < shape.size(); ++d) {
                    if (origin[d] < 0 || shape[d] > desc->shape[d] || origin[d] > desc->shape[d] - shape[d]) {
                        return fail(error, "state write out of bounds");
                    }
                }
                const buffer src = contiguous(const_cast<float *>(data), count, dtype::float32, shape);
                __block bool copied = false;
                std::string * err_text = &error;
                p_->healthy = false;
                [p_->state getMultiArrayForStateNamed:[NSString stringWithUTF8String:name.c_str()] handler:^(MLMultiArray * dst) {
                    if (!matches(dst, *desc, *err_text)) return;
                    [dst getMutableBytesWithHandler:^(void * bytes, NSInteger size, NSArray<NSNumber *> * strides) {
                        copied = decoder_coreml_copy_region(src, array_buffer(dst, bytes, size, strides), origin, *err_text);
                    }];
                }];
                p_->healthy = copied;
                return copied;
            }
        }
        return fail(error, "stateful Core ML requires macOS 15 / iOS 18");
    });
}

bool decoder_coreml_session::read_state(const std::string & name, float * data, size_t count, std::string & error) const {
    return guarded(error, [&]() {
        if (@available(macOS 15.0, iOS 18.0, *)) {
            @autoreleasepool {
                if (!p_->ready(error)) return false;
                const auto * desc = p_->find(p_->state_info, name);
                if (!desc) return fail(error, "unknown state: " + name);
                __block bool copied = false;
                std::string * err_text = &error;
                [p_->state getMultiArrayForStateNamed:[NSString stringWithUTF8String:name.c_str()] handler:^(MLMultiArray * src) {
                    copied = p_->copy_array(src, *desc, data, count, *err_text);
                }];
                return copied;
            }
        }
        return fail(error, "stateful Core ML requires macOS 15 / iOS 18");
    });
}

bool decoder_coreml_session::reset(std::string & error) {
    return guarded(error, [&]() {
        if (@available(macOS 15.0, iOS 18.0, *)) {
            @autoreleasepool {
                p_->healthy = false;
                p_->result = nil;
                p_->state = [p_->model newState];
                if (!p_->state) return fail(error, "Core ML state allocation failed");
                std::fill(p_->initialized.begin(), p_->initialized.end(), false);
                p_->healthy = true;
                return true;
            }
        }
        return fail(error, "stateful Core ML requires macOS 15 / iOS 18");
    });
}

#else
// Keep stateless Core ML stages buildable with older SDKs. The opt-in decoder
// is unavailable, while DAC and the other sidecars retain their old targets.
struct decoder_coreml_session::impl {};
decoder_coreml_session::decoder_coreml_session(std::unique_ptr<impl> p) : p_(std::move(p)) {}
decoder_coreml_session::~decoder_coreml_session() = default;
std::unique_ptr<decoder_coreml_session> decoder_coreml_session::open(
    const std::string &, decoder_coreml_units, std::string & error) {
    error = "decoder state requires a macOS 15 / iOS 18 SDK";
    return nullptr;
}
const std::vector<decoder_coreml_tensor> & decoder_coreml_session::inputs() const {
    static const std::vector<decoder_coreml_tensor> empty;
    return empty;
}
const std::vector<decoder_coreml_tensor> & decoder_coreml_session::outputs() const { return inputs(); }
const std::vector<decoder_coreml_tensor> & decoder_coreml_session::states() const { return inputs(); }
std::string decoder_coreml_session::metadata(const std::string &) const { return {}; }
bool decoder_coreml_session::set_input(const std::string &, const float *, size_t, std::string & e) {
    return fail(e, "decoder state requires a macOS 15 / iOS 18 SDK");
}
bool decoder_coreml_session::set_input_i32(const std::string &, const int32_t *, size_t, std::string & e) {
    return fail(e, "decoder state requires a macOS 15 / iOS 18 SDK");
}
bool decoder_coreml_session::predict(const std::vector<decoder_coreml_row_update> &,
                                    decoder_coreml_timing *, std::string & e) {
    return fail(e, "decoder state requires a macOS 15 / iOS 18 SDK");
}
bool decoder_coreml_session::copy_output(const std::string &, float *, size_t, std::string & e) const {
    return fail(e, "decoder state requires a macOS 15 / iOS 18 SDK");
}
bool decoder_coreml_session::write_state(const std::string &, const std::vector<int64_t> &,
                                        const std::vector<int64_t> &, const float *, size_t, std::string & e) {
    return fail(e, "decoder state requires a macOS 15 / iOS 18 SDK");
}
bool decoder_coreml_session::read_state(const std::string &, float *, size_t, std::string & e) const {
    return fail(e, "decoder state requires a macOS 15 / iOS 18 SDK");
}
bool decoder_coreml_session::reset(std::string & e) {
    return fail(e, "decoder state requires a macOS 15 / iOS 18 SDK");
}
#endif

} // namespace detail
} // namespace parler
} // namespace tts_cpp
