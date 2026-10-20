// Core ML CTC/IndicConformer/Unified RNN-T/TDT/EOU/Nemotron/Sortformer encoder sidecars for
// parakeet-cpp.
//
// Derived from the whisper.cpp Core ML wrapper pattern (src/coreml/whisper-encoder.mm),
// generalised to load an arbitrary compiled encoder via the generic MLModel API so it
// works with whatever `.mlmodelc` the mobius export produces (no code-generated model
// interface required). Tensor orientation is discovered from the model description, so
// the export only has to agree on dimension sizes, not their order.

#if !__has_feature(objc_arc)
#error "parakeet-encoder.mm must be compiled with -fobjc-arc"
#endif

#import "coreml/parakeet-encoder.h"
#include "coreml/parakeet_coreml_shape.h"
#include "coreml/parakeet_coreml_units.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {

// Input names the fixed-window Neural Engine exporter
// (scripts/export-encoder-coreml-ane.py) declares. Their presence marks that
// export; any other model is bound as a single-input export.
NSString * const kAneMelInput     = @"mel";
NSString * const kAnePadMaskInput = @"pad_mask";
NSString * const kAneKeyBiasInput = @"key_bias";
// Optional: exports that mask every subsampling resolution take the
// mel-frame validity too; the published TranscribePsy export does not.
NSString * const kAneMelMaskInput = @"mel_mask";

// Additive attention bias for pad keys. Matches KEY_BIAS_PAD in the exporter:
// fp16 -inf would make an all-pad softmax row NaN, -1e4 saturates just as hard.
constexpr float kAneKeyBiasPad = -1.0e4f;

// Seam context the fixed-window export was validated with, in encoder frames.
constexpr int64_t kAneSeamContextFramesDefault = 16;
constexpr const char * kAneSeamContextEnv = "PARAKEET_COREML_ANE_CONTEXT_FRAMES";
constexpr const char * kComputeUnitsEnv   = "PARAKEET_COREML_COMPUTE_UNITS";

float decode_float16(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int      shift = 0;
            uint32_t m     = mant;
            do { m <<= 1; ++shift; } while ((m & 0x400u) == 0);
            bits = sign | ((uint32_t) (127 - 15 - shift + 1) << 23) | ((m & 0x3ffu) << 13);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

uint16_t encode_float16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t exp  = (bits >> 23) & 0xffu;
    uint32_t       mant = bits & 0x7fffffu;
    if (exp == 0xffu) {
        return (uint16_t) (sign | (mant == 0 ? 0x7c00u : 0x7e00u));
    }

    const int32_t half_exp = (int32_t) exp - 127 + 15;
    if (half_exp >= 31) return (uint16_t) (sign | 0x7c00u);
    if (half_exp <= 0) {
        if (half_exp < -10) return (uint16_t) sign;
        mant |= 0x800000u;
        const uint32_t shift   = (uint32_t) (14 - half_exp);
        uint32_t       rounded = mant >> shift;
        const uint32_t halfway = 1u << (shift - 1);
        if ((mant & (halfway - 1)) != 0 || ((mant & halfway) != 0 && (rounded & 1u) != 0)) {
            ++rounded;
        }
        return (uint16_t) (sign | rounded);
    }

    uint32_t rounded = mant + 0xfffu + ((mant >> 13) & 1u);
    if ((rounded & 0x800000u) != 0) {
        rounded = 0;
        if (half_exp + 1 >= 31) return (uint16_t) (sign | 0x7c00u);
        return (uint16_t) (sign | ((uint32_t) (half_exp + 1) << 10));
    }
    return (uint16_t) (sign | ((uint32_t) half_exp << 10) | (rounded >> 13));
}

NSString * sole_multiarray_feature(NSDictionary<NSString *, MLFeatureDescription *> * descs) {
    NSString * found = nil;
    for (NSString * key in descs) {
        if (descs[key].type == MLFeatureTypeMultiArray) {
            if (found != nil) return nil;
            found = key;
        }
    }
    return found;
}

std::vector<int64_t> dims_of(NSArray<NSNumber *> * shape) {
    std::vector<int64_t> dims;
    if (shape != nil) {
        dims.reserve(shape.count);
        for (NSNumber * dim in shape) {
            dims.push_back(dim.longLongValue);
        }
    }
    return dims;
}

bool constraint_has_one_fixed_shape(MLMultiArrayConstraint * constraint) {
    if (constraint == nil) return false;

    MLMultiArrayShapeConstraint * shape_constraint = constraint.shapeConstraint;
    if (shape_constraint == nil) return false;

    if (shape_constraint.type == MLMultiArrayShapeConstraintTypeEnumerated) {
        return shape_constraint.enumeratedShapes.count == 1;
    }
    if (shape_constraint.type != MLMultiArrayShapeConstraintTypeRange) {
        return false;
    }

    NSArray<NSValue *> * ranges = shape_constraint.sizeRangeForDimension;
    if (ranges.count != constraint.shape.count) return false;
    for (NSValue * value in ranges) {
        if (value.rangeValue.length != 1) return false;
    }
    return true;
}

// Resolves the concrete input shape to allocate and whether it is feature-major.
// Thin Objective-C adapter over parakeet::coreml_resolve_input_dims (unit-tested in
// test-coreml-shapes): a fixed-shape model already sized to this input is honoured
// verbatim, a flexible export is rebuilt at the requested mel length in the model's
// declared orientation, and an unknown/non-concrete shape falls back to
// features-major [1, n_mels, n_mel_frames].
NSArray<NSNumber *> * resolve_input_shape(NSArray<NSNumber *> * declared,
                                          int64_t n_mel_frames, int64_t n_mels,
                                          bool * transpose) {
    const parakeet::CoremlInputDims resolved =
        parakeet::coreml_resolve_input_dims(dims_of(declared), n_mel_frames, n_mels);
    *transpose = resolved.transpose;
    NSMutableArray<NSNumber *> * shape = [NSMutableArray arrayWithCapacity:resolved.dims.size()];
    for (int64_t dim : resolved.dims) {
        [shape addObject:@(dim)];
    }
    return shape;
}

MLMultiArray * build_array(NSArray<NSNumber *> * shape, MLMultiArrayDataType data_type,
                           NSError ** err) {
    if (data_type != MLMultiArrayDataTypeFloat32 && data_type != MLMultiArrayDataTypeFloat16) {
        return nil;
    }
    return [[MLMultiArray alloc] initWithShape:shape dataType:data_type error:err];
}

bool fill_input_array(
    MLMultiArray * arr,
    bool transpose,
    const float * mel,
    int64_t actual_mel_frames,
    int64_t allocated_mel_frames,
    int64_t n_mels) {
    if (arr == nil || mel == nullptr ||
        actual_mel_frames <= 0 ||
        allocated_mel_frames < actual_mel_frames ||
        n_mels <= 0) {
        return false;
    }

    const int64_t allocated_count = allocated_mel_frames * n_mels;
    if ((int64_t) arr.count != allocated_count) return false;

    if (arr.dataType == MLMultiArrayDataTypeFloat32) {
        float * dst = (float *) arr.dataPointer;
        std::fill(dst, dst + allocated_count, 0.0f);

        if (!transpose) {
            std::memcpy(
                dst,
                mel,
                (size_t) actual_mel_frames * n_mels * sizeof(float));
            return true;
        }

        for (int64_t m = 0; m < n_mels; ++m) {
            for (int64_t t = 0; t < actual_mel_frames; ++t) {
                dst[m * allocated_mel_frames + t] =
                    mel[t * n_mels + m];
            }
        }
        return true;
    }

    if (arr.dataType != MLMultiArrayDataTypeFloat16) return false;

    uint16_t * dst = (uint16_t *) arr.dataPointer;
    std::fill(dst, dst + allocated_count, uint16_t{0});

    if (!transpose) {
        const int64_t actual_count = actual_mel_frames * n_mels;
        for (int64_t i = 0; i < actual_count; ++i) {
            dst[i] = encode_float16(mel[i]);
        }
        return true;
    }

    for (int64_t m = 0; m < n_mels; ++m) {
        for (int64_t t = 0; t < actual_mel_frames; ++t) {
            dst[m * allocated_mel_frames + t] =
                encode_float16(mel[t * n_mels + m]);
        }
    }
    return true;
}

bool fill_valid_mask(MLMultiArray * arr, int64_t valid_frames, int64_t capacity) {
    if (arr == nil || valid_frames <= 0 || capacity < valid_frames ||
        (int64_t) arr.count != capacity) {
        return false;
    }

    if (arr.dataType == MLMultiArrayDataTypeFloat32) {
        float * dst = (float *) arr.dataPointer;
        std::fill(dst, dst + capacity, 0.0f);
        std::fill(dst, dst + valid_frames, 1.0f);
        return true;
    }
    if (arr.dataType == MLMultiArrayDataTypeFloat16) {
        uint16_t * dst = (uint16_t *) arr.dataPointer;
        std::fill(dst, dst + capacity, uint16_t{0});
        std::fill(dst, dst + valid_frames, encode_float16(1.0f));
        return true;
    }
    return false;
}

bool read_scalar(const void * base, MLMultiArrayDataType dt, int64_t off, float * out) {
    switch (dt) {
        case MLMultiArrayDataTypeFloat32: *out = ((const float *)    base)[off];                 return true;
        case MLMultiArrayDataTypeDouble:  *out = (float) ((const double *) base)[off];            return true;
        case MLMultiArrayDataTypeFloat16: *out = decode_float16(((const uint16_t *) base)[off]);  return true;
        default:                          return false;
    }
}

bool write_scalar(void * base, MLMultiArrayDataType dt, int64_t off, float value) {
    switch (dt) {
        case MLMultiArrayDataTypeFloat32: ((float *)    base)[off] = value;                 return true;
        case MLMultiArrayDataTypeDouble:  ((double *)   base)[off] = (double) value;        return true;
        case MLMultiArrayDataTypeFloat16: ((uint16_t *) base)[off] = encode_float16(value); return true;
        default:                          return false;
    }
}

// Fills a dense array of `capacity` scalars with `valid_value` for the first
// `valid` entries and `pad_value` after them.
bool fill_split_array(MLMultiArray * arr, int64_t valid, int64_t capacity,
                      float valid_value, float pad_value) {
    if (arr == nil || valid < 0 || capacity < valid || (int64_t) arr.count != capacity) {
        return false;
    }
    void * base = arr.dataPointer;
    for (int64_t i = 0; i < capacity; ++i) {
        if (!write_scalar(base, arr.dataType, i, i < valid ? valid_value : pad_value)) return false;
    }
    return true;
}

// Copies the first `n_rows` frames of a channel-first [1, d_model, 1, T] output
// into row-major (frame, d_model) rows of `dst`, honouring the declared strides.
bool copy_channel_first_output(MLMultiArray * arr, float * dst, int64_t n_rows, int64_t d_model) {
    if (arr == nil || arr.shape.count != 4) return false;

    const std::vector<int64_t> dims = dims_of(arr.shape);
    if (dims[1] != d_model || dims[3] < n_rows) return false;

    const void               * base    = arr.dataPointer;
    const MLMultiArrayDataType dt      = arr.dataType;
    NSArray<NSNumber *>      * strides = arr.strides;
    const int64_t              s_chan  = strides[1].longLongValue;
    const int64_t              s_time  = strides[3].longLongValue;

    for (int64_t t = 0; t < n_rows; ++t) {
        for (int64_t f = 0; f < d_model; ++f) {
            float value;
            if (!read_scalar(base, dt, f * s_chan + t * s_time, &value)) return false;
            dst[t * d_model + f] = value;
        }
    }
    return true;
}

// The compiled bundle's metadata.json lists the model's inputs by name, so the
// export kind is known before the (expensive) model load has to pick its
// compute units. A bundle without readable metadata is treated as single-input.
bool mlmodelc_declares_ane_inputs(NSString * path) {
    NSString * metadata = [path stringByAppendingPathComponent:@"metadata.json"];
    NSString * text = [NSString stringWithContentsOfFile:metadata
                                                encoding:NSUTF8StringEncoding
                                                   error:nil];
    if (text == nil) return false;
    NSString * pad_mask = [NSString stringWithFormat:@"\"%@\"", kAnePadMaskInput];
    NSString * key_bias = [NSString stringWithFormat:@"\"%@\"", kAneKeyBiasInput];
    return [text containsString:pad_mask] && [text containsString:key_bias];
}

bool declares_ane_inputs(NSDictionary<NSString *, MLFeatureDescription *> * ins) {
    return ins[kAneMelInput].type == MLFeatureTypeMultiArray &&
           ins[kAnePadMaskInput].type == MLFeatureTypeMultiArray &&
           ins[kAneKeyBiasInput].type == MLFeatureTypeMultiArray;
}

// Copies the model output into `dst` as row-major (n_enc_frames, d_model), adapting to the
// output tensor orientation and dtype and honouring its element strides.
bool copy_output_array(MLMultiArray * arr, float * dst, int64_t n_enc_frames, int64_t d_model) {
    if (arr == nil) return false;

    const void               * base  = arr.dataPointer;
    const MLMultiArrayDataType dt    = arr.dataType;
    const int64_t              total = n_enc_frames * d_model;

    const parakeet::CoremlTrailingMatch match =
        parakeet::coreml_match_trailing_capacity(dims_of(arr.shape), n_enc_frames, d_model);
    if (match.matched) {
        const bool            transpose = match.transpose;
        const NSUInteger      n       = arr.shape.count;
        NSArray<NSNumber *> * strides = arr.strides;
        const int64_t         s_outer = strides[n - 2].longLongValue;
        const int64_t         s_inner = strides[n - 1].longLongValue;
        for (int64_t t = 0; t < n_enc_frames; ++t) {
            for (int64_t f = 0; f < d_model; ++f) {
                const int64_t outer = transpose ? f : t;
                const int64_t inner = transpose ? t : f;
                float value;
                if (!read_scalar(base, dt, outer * s_outer + inner * s_inner, &value)) return false;
                dst[t * d_model + f] = value;
            }
        }
        return true;
    }

    // Unmatched shape but matching element count: assume a contiguous row-major
    // (n_enc_frames, d_model) buffer -- the natural layout for a flattened export.
    if ((int64_t) arr.count != total) return false;
    for (int64_t i = 0; i < total; ++i) {
        float value;
        if (!read_scalar(base, dt, i, &value)) return false;
        dst[i] = value;
    }
    return true;
}

}  // namespace

// Geometry of a fixed-window Neural Engine export; `active` is false for every
// other export.
struct AneWindow {
    bool    active     = false;
    int64_t n_mels     = 0;
    int64_t window_mel = 0;
    int64_t enc_frames = 0;
    int64_t d_model    = 0;
};

struct parakeet_coreml_context {
    const void * model              = nullptr;  // CFBridgingRetain'd MLModel *
    const void * input_array        = nullptr;  // retained MLMultiArray *
    const void * valid_mask_array   = nullptr;  // retained MLMultiArray * (AOSC)
    const void * pad_mask_array     = nullptr;  // retained MLMultiArray * (fixed-window export)
    const void * key_bias_array     = nullptr;  // retained MLMultiArray * (fixed-window export)
    const void * mel_mask_array     = nullptr;  // retained MLMultiArray * (fixed-window export, optional)
    const void * input_provider     = nullptr;  // retained MLDictionaryFeatureProvider *
    const void * output_array       = nullptr;  // retained MLMultiArray * (when shape is fixed)
    const void * prediction_options = nullptr;  // retained MLPredictionOptions *
    std::string  input_name;
    std::string  valid_mask_name;
    std::string  output_name;
    std::string  label;
    std::vector<int64_t> declared_input_dims;
    std::vector<int64_t> declared_output_dims;
    bool         input_shape_fixed = false;
    int64_t      cached_mel_frames = 0;
    int64_t      cached_mels       = 0;
    int64_t      cached_d_model    = 0;
    bool         input_transpose   = false;
    AneWindow    ane;
    std::mutex   mutex;
};

static int64_t trailing_partner(const std::vector<int64_t> & dims, int64_t known) {
    if (dims.size() < 2 || known <= 0) return 0;
    const std::size_t n = dims.size();
    if (dims[n - 1] == known && dims[n - 2] > 0) return dims[n - 2];
    if (dims[n - 2] == known && dims[n - 1] > 0) return dims[n - 1];
    return 0;
}

int64_t fixed_mel_frames_for(
    const parakeet_coreml_context * ctx,
    int64_t n_mels) {
    if (ctx == nullptr || !ctx->input_shape_fixed || n_mels <= 0) return 0;
    if (ctx->ane.active) {
        return n_mels == ctx->ane.n_mels ? ctx->ane.window_mel : 0;
    }

    const std::vector<int64_t> & dims = ctx->declared_input_dims;
    if (dims.size() < 2) return 0;

    const std::size_t n = dims.size();
    if (dims[n - 1] == n_mels && dims[n - 2] > 0) {
        return dims[n - 2];
    }
    if (dims[n - 2] == n_mels && dims[n - 1] > 0) {
        return dims[n - 1];
    }
    return 0;
}

namespace {

void release_cached_io(parakeet_coreml_context * ctx) {
    if (ctx->prediction_options != nullptr) CFRelease(ctx->prediction_options);
    if (ctx->output_array       != nullptr) CFRelease(ctx->output_array);
    if (ctx->input_provider     != nullptr) CFRelease(ctx->input_provider);
    if (ctx->valid_mask_array   != nullptr) CFRelease(ctx->valid_mask_array);
    if (ctx->pad_mask_array     != nullptr) CFRelease(ctx->pad_mask_array);
    if (ctx->key_bias_array     != nullptr) CFRelease(ctx->key_bias_array);
    if (ctx->mel_mask_array     != nullptr) CFRelease(ctx->mel_mask_array);
    if (ctx->input_array        != nullptr) CFRelease(ctx->input_array);
    ctx->prediction_options = nullptr;
    ctx->output_array       = nullptr;
    ctx->input_provider     = nullptr;
    ctx->valid_mask_array   = nullptr;
    ctx->pad_mask_array     = nullptr;
    ctx->key_bias_array     = nullptr;
    ctx->mel_mask_array     = nullptr;
    ctx->input_array        = nullptr;
    ctx->cached_mel_frames  = 0;
    ctx->cached_mels        = 0;
    ctx->cached_d_model     = 0;
}

// Allocates the fixed-window export's three inputs, its provider and its
// output backing once; every shape is a property of the compiled graph, so
// later predictions only overwrite the contents.
bool prepare_cached_ane_io(parakeet_coreml_context * ctx, MLModel * model, NSString * out_name) {
    if (ctx->input_array != nullptr && ctx->pad_mask_array != nullptr &&
        ctx->key_bias_array != nullptr) {
        return true;
    }
    release_cached_io(ctx);

    NSDictionary<NSString *, MLFeatureDescription *> * ins =
        model.modelDescription.inputDescriptionsByName;
    MLMultiArrayConstraint * mel_c  = ins[kAneMelInput].multiArrayConstraint;
    MLMultiArrayConstraint * mask_c = ins[kAnePadMaskInput].multiArrayConstraint;
    MLMultiArrayConstraint * bias_c = ins[kAneKeyBiasInput].multiArrayConstraint;
    MLMultiArrayConstraint * mel_mask_c = ins[kAneMelMaskInput].multiArrayConstraint;
    MLMultiArrayConstraint * out_c  =
        model.modelDescription.outputDescriptionsByName[out_name].multiArrayConstraint;
    if (mel_c == nil || mask_c == nil || bias_c == nil || out_c == nil) return false;

    NSError * err = nil;
    MLMultiArray * mel_arr  = build_array(mel_c.shape,  mel_c.dataType,  &err);
    if (mel_arr == nil || err != nil) return false;
    MLMultiArray * mask_arr = build_array(mask_c.shape, mask_c.dataType, &err);
    if (mask_arr == nil || err != nil) return false;
    MLMultiArray * bias_arr = build_array(bias_c.shape, bias_c.dataType, &err);
    if (bias_arr == nil || err != nil) return false;
    MLMultiArray * out_arr  = build_array(out_c.shape,  out_c.dataType,  &err);
    if (out_arr == nil || err != nil) return false;

    NSMutableDictionary<NSString *, id> * inputs = [NSMutableDictionary dictionaryWithDictionary:@{
        kAneMelInput     : mel_arr,
        kAnePadMaskInput : mask_arr,
        kAneKeyBiasInput : bias_arr,
    }];
    MLMultiArray * mel_mask_arr = nil;
    if (mel_mask_c != nil) {
        mel_mask_arr = build_array(mel_mask_c.shape, mel_mask_c.dataType, &err);
        if (mel_mask_arr == nil || err != nil) return false;
        inputs[kAneMelMaskInput] = mel_mask_arr;
    }

    MLDictionaryFeatureProvider * provider =
        [[MLDictionaryFeatureProvider alloc] initWithDictionary:inputs error:&err];
    if (provider == nil || err != nil) return false;

    MLPredictionOptions * options = [[MLPredictionOptions alloc] init];
    options.outputBackings = @{ out_name : out_arr };

    ctx->input_array        = CFBridgingRetain(mel_arr);
    ctx->pad_mask_array     = CFBridgingRetain(mask_arr);
    ctx->key_bias_array     = CFBridgingRetain(bias_arr);
    if (mel_mask_arr != nil) ctx->mel_mask_array = CFBridgingRetain(mel_mask_arr);
    ctx->input_provider     = CFBridgingRetain(provider);
    ctx->prediction_options = CFBridgingRetain(options);
    ctx->output_array       = CFBridgingRetain(out_arr);
    ctx->cached_mel_frames  = ctx->ane.window_mel;
    ctx->cached_mels        = ctx->ane.n_mels;
    ctx->cached_d_model     = ctx->ane.d_model;
    ctx->input_transpose    = true;
    return true;
}

// Binds the three-input fixed-window export. Returns false when its declared
// shapes are not the ones the exporter promises.
bool bind_ane_window(parakeet_coreml_context * ctx, MLModel * model, NSString * out_name) {
    NSDictionary<NSString *, MLFeatureDescription *> * ins =
        model.modelDescription.inputDescriptionsByName;
    const std::vector<int64_t> mel_dims  = dims_of(ins[kAneMelInput].multiArrayConstraint.shape);
    const std::vector<int64_t> mask_dims = dims_of(ins[kAnePadMaskInput].multiArrayConstraint.shape);
    const std::vector<int64_t> out_dims  =
        dims_of(model.modelDescription.outputDescriptionsByName[out_name].multiArrayConstraint.shape);
    if (mel_dims.size() != 4 || mask_dims.size() != 4 || out_dims.size() != 4) return false;

    AneWindow ane;
    ane.active     = true;
    ane.n_mels     = mel_dims[1];
    ane.window_mel = mel_dims[3];
    ane.enc_frames = mask_dims[3];
    ane.d_model    = out_dims[1];
    if (ane.n_mels <= 0 || ane.window_mel <= 0 || ane.enc_frames <= 0 || ane.d_model <= 0 ||
        out_dims[3] != ane.enc_frames) {
        return false;
    }

    ctx->ane                  = ane;
    ctx->input_name           = kAneMelInput.UTF8String;
    ctx->output_name          = out_name.UTF8String;
    ctx->declared_input_dims  = mel_dims;
    ctx->declared_output_dims = out_dims;
    ctx->input_shape_fixed    = true;
    return true;
}

// One window of the fixed-window export: a shorter input is zero-padded to the
// window and its padded encoder frames are masked out of attention and the
// convolution module (and, when the export takes `mel_mask`, out of every
// subsampling resolution), so the valid frames come out as if the input had
// ended there. The masks follow `n_mel_valid`, not the slab length, so the
// mel front-end's zeroed trailing frames count as padding too.
// `n_enc_frames` is the caller's frame count for `n_mel_frames`.
int encode_ane_window(parakeet_coreml_context * ctx,
                      int64_t n_mel_frames, int64_t n_mel_valid, int64_t n_mels, const float * mel,
                      int64_t n_enc_frames, int64_t d_model, float * encoder_out) {
    const AneWindow & ane = ctx->ane;
    if (n_mels != ane.n_mels || d_model != ane.d_model) return 2;
    if (n_mel_frames > ane.window_mel || n_enc_frames > ane.enc_frames) return 2;
    const parakeet::CoremlAneValidity valid =
        parakeet::coreml_ane_valid_lengths(n_mel_frames, n_mel_valid, n_enc_frames);

    MLModel  * model    = (__bridge MLModel *) ctx->model;
    NSString * out_name = [NSString stringWithUTF8String:ctx->output_name.c_str()];
    if (!prepare_cached_ane_io(ctx, model, out_name)) return 2;

    MLMultiArray * mel_arr  = (__bridge MLMultiArray *) ctx->input_array;
    MLMultiArray * mask_arr = (__bridge MLMultiArray *) ctx->pad_mask_array;
    MLMultiArray * bias_arr = (__bridge MLMultiArray *) ctx->key_bias_array;
    if (!fill_input_array(mel_arr, /*transpose=*/true, mel, n_mel_frames, ane.window_mel, n_mels)) return 3;
    if (!fill_split_array(mask_arr, valid.enc_frames, ane.enc_frames, 1.0f, 0.0f)) return 3;
    if (!fill_split_array(bias_arr, valid.enc_frames, ane.enc_frames, 0.0f, kAneKeyBiasPad)) return 3;
    if (ctx->mel_mask_array != nullptr) {
        MLMultiArray * mel_mask_arr = (__bridge MLMultiArray *) ctx->mel_mask_array;
        if (!fill_split_array(mel_mask_arr, valid.mel_frames, ane.window_mel, 1.0f, 0.0f)) return 3;
    }

    NSError * err = nil;
    MLDictionaryFeatureProvider * provider =
        (__bridge MLDictionaryFeatureProvider *) ctx->input_provider;
    MLPredictionOptions * options = (__bridge MLPredictionOptions *) ctx->prediction_options;
    id<MLFeatureProvider> result = [model predictionFromFeatures:provider options:options error:&err];
    if (result == nil || err != nil) return 5;

    MLMultiArray * out_arr = [result featureValueForName:out_name].multiArrayValue;
    if (out_arr == nil) return 6;
    if (!copy_channel_first_output(out_arr, encoder_out, n_enc_frames, d_model)) return 7;
    return 0;
}

bool prepare_cached_io(parakeet_coreml_context * ctx, MLModel * model,
                       NSString * in_name, NSString * out_name,
                       int64_t n_mel_frames, int64_t n_mels,
                       int64_t n_enc_frames, int64_t d_model) {
    const int64_t fixed_frames = fixed_mel_frames_for(ctx, n_mels);
    if (fixed_frames > 0 && n_mel_frames > fixed_frames) {
        return false;
    }

    const int64_t input_frames = fixed_frames > 0 ? fixed_frames : n_mel_frames;

    if (ctx->input_array != nullptr &&
        ctx->cached_mel_frames == input_frames &&
        ctx->cached_mels == n_mels &&
        ctx->cached_d_model == d_model) {
        return true;
    }

    release_cached_io(ctx);
    MLMultiArrayConstraint * in_constraint =
        model.modelDescription.inputDescriptionsByName[in_name].multiArrayConstraint;
    bool                  in_transpose = true;
    NSArray<NSNumber *> * in_shape =
        resolve_input_shape(in_constraint.shape, input_frames, n_mels, &in_transpose);
    if (in_shape == nil) return false;

    NSError      * err    = nil;
    MLMultiArray * in_arr = build_array(in_shape, in_constraint.dataType, &err);
    if (in_arr == nil || err != nil) return false;
    MLDictionaryFeatureProvider * provider =
        [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{ in_name : in_arr } error:&err];
    if (provider == nil || err != nil) return false;

    MLPredictionOptions * options = [[MLPredictionOptions alloc] init];
    MLMultiArrayConstraint * out_constraint =
        model.modelDescription.outputDescriptionsByName[out_name].multiArrayConstraint;
    MLMultiArray * out_arr = nil;
    const parakeet::CoremlTrailingMatch out_match =
        parakeet::coreml_match_trailing_capacity(dims_of(out_constraint.shape), n_enc_frames, d_model);
    if (out_match.matched) {
        out_arr = build_array(out_constraint.shape, out_constraint.dataType, &err);
        if (out_arr == nil || err != nil) return false;
        options.outputBackings = @{ out_name : out_arr };
    }

    ctx->input_array        = CFBridgingRetain(in_arr);
    ctx->input_provider     = CFBridgingRetain(provider);
    ctx->prediction_options = CFBridgingRetain(options);
    if (out_arr != nil) ctx->output_array = CFBridgingRetain(out_arr);
    ctx->cached_mel_frames = input_frames;
    ctx->cached_mels       = n_mels;
    ctx->cached_d_model    = d_model;
    ctx->input_transpose   = in_transpose;
    return true;
}

bool prepare_cached_bypass_io(parakeet_coreml_context * ctx, MLModel * model,
                              NSString * in_name, NSString * mask_name,
                              NSString * out_name, int64_t capacity,
                              int64_t d_model) {
    if (ctx->input_array != nullptr && ctx->valid_mask_array != nullptr &&
        ctx->cached_mel_frames == capacity && ctx->cached_mels == d_model &&
        ctx->cached_d_model == d_model) {
        return true;
    }

    release_cached_io(ctx);
    MLMultiArrayConstraint * in_constraint =
        model.modelDescription.inputDescriptionsByName[in_name].multiArrayConstraint;
    MLMultiArrayConstraint * mask_constraint =
        model.modelDescription.inputDescriptionsByName[mask_name].multiArrayConstraint;
    MLMultiArrayConstraint * out_constraint =
        model.modelDescription.outputDescriptionsByName[out_name].multiArrayConstraint;
    if (in_constraint == nil || mask_constraint == nil || out_constraint == nil) return false;

    bool transpose = true;
    NSArray<NSNumber *> * in_shape =
        resolve_input_shape(in_constraint.shape, capacity, d_model, &transpose);
    NSError * err = nil;
    MLMultiArray * in_arr = build_array(in_shape, in_constraint.dataType, &err);
    if (in_arr == nil || err != nil) return false;
    MLMultiArray * mask_arr = build_array(
        mask_constraint.shape, mask_constraint.dataType, &err);
    if (mask_arr == nil || err != nil || (int64_t) mask_arr.count != capacity) return false;

    MLDictionaryFeatureProvider * provider =
        [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{ in_name : in_arr, mask_name : mask_arr }
            error:&err];
    if (provider == nil || err != nil) return false;

    MLPredictionOptions * options = [[MLPredictionOptions alloc] init];
    MLMultiArray * out_arr = nil;
    const parakeet::CoremlTrailingMatch out_match =
        parakeet::coreml_match_trailing_capacity(
            dims_of(out_constraint.shape), capacity, d_model);
    if (out_match.matched) {
        out_arr = build_array(out_constraint.shape, out_constraint.dataType, &err);
        if (out_arr == nil || err != nil) return false;
        options.outputBackings = @{ out_name : out_arr };
    }

    ctx->input_array        = CFBridgingRetain(in_arr);
    ctx->valid_mask_array   = CFBridgingRetain(mask_arr);
    ctx->input_provider     = CFBridgingRetain(provider);
    ctx->prediction_options = CFBridgingRetain(options);
    if (out_arr != nil) ctx->output_array = CFBridgingRetain(out_arr);
    ctx->cached_mel_frames = capacity;
    ctx->cached_mels       = d_model;
    ctx->cached_d_model    = d_model;
    ctx->input_transpose   = transpose;
    return true;
}

}  // namespace

struct parakeet_coreml_context * parakeet_coreml_init(const char * path_mlmodelc) {
    return parakeet_coreml_init_with_units_env(path_mlmodelc, kComputeUnitsEnv);
}

struct parakeet_coreml_context * parakeet_coreml_init_with_units_env(const char * path_mlmodelc,
                                                                     const char * units_env) {
    if (path_mlmodelc == nullptr || units_env == nullptr) return nullptr;

    @autoreleasepool {
        NSString * path = [[NSString alloc] initWithUTF8String:path_mlmodelc];
        if (path == nil) return nullptr;

        // The fixed-window export is fully ANE-shaped, so it is pinned to CPU
        // and Neural Engine; the generic exports keep every processor
        // available, since their few GPU-preferred operations would otherwise
        // fall back to the CPU and erase the encoder speedup.
        MLModelConfiguration * config = [[MLModelConfiguration alloc] init];
        std::string label;
        config.computeUnits = parakeet_coreml_requested_units(
            units_env, mlmodelc_declares_ane_inputs(path), &label);

        NSError * err   = nil;
        MLModel * model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:path]
                                            configuration:config
                                                    error:&err];
        if (model == nil || err != nil) return nullptr;

        NSDictionary<NSString *, MLFeatureDescription *> * input_descs =
            model.modelDescription.inputDescriptionsByName;
        if (declares_ane_inputs(input_descs)) {
            NSString * ane_out = sole_multiarray_feature(model.modelDescription.outputDescriptionsByName);
            if (ane_out == nil) return nullptr;
            auto * ctx = new parakeet_coreml_context();
            ctx->model = CFBridgingRetain(model);
            ctx->label = std::move(label);
            if (!bind_ane_window(ctx, model, ane_out)) {
                parakeet_coreml_free(ctx);
                return nullptr;
            }
            return ctx;
        }
        NSString * in_name = sole_multiarray_feature(input_descs);
        NSString * valid_mask_name = nil;
        if (in_name == nil && input_descs.count == 2 &&
            input_descs[@"pre_encode"].type == MLFeatureTypeMultiArray &&
            input_descs[@"valid_mask"].type == MLFeatureTypeMultiArray) {
            in_name = @"pre_encode";
            valid_mask_name = @"valid_mask";
        }
        NSString * out_name = sole_multiarray_feature(model.modelDescription.outputDescriptionsByName);
        if (in_name == nil || out_name == nil) return nullptr;

        MLMultiArrayConstraint * input_constraint = model.modelDescription.inputDescriptionsByName[in_name].multiArrayConstraint;
        MLMultiArrayConstraint * output_constraint = model.modelDescription.outputDescriptionsByName[out_name].multiArrayConstraint;
        if (input_constraint == nil) return nullptr;

        auto * ctx = new parakeet_coreml_context();
        ctx->model       = CFBridgingRetain(model);
        ctx->input_name  = in_name.UTF8String;
        if (valid_mask_name != nil) ctx->valid_mask_name = valid_mask_name.UTF8String;
        ctx->output_name = out_name.UTF8String;
        ctx->label       = std::move(label);
        ctx->declared_input_dims = dims_of(input_constraint.shape);
        if (output_constraint != nil) ctx->declared_output_dims = dims_of(output_constraint.shape);
        ctx->input_shape_fixed = constraint_has_one_fixed_shape(input_constraint);
        return ctx;
    }
}

void parakeet_coreml_free(struct parakeet_coreml_context * ctx) {
    if (ctx == nullptr) return;
    release_cached_io(ctx);
    if (ctx->model != nullptr) CFRelease(ctx->model);
    delete ctx;
}

int64_t parakeet_coreml_fixed_mel_frames(
    const struct parakeet_coreml_context * ctx,
    int64_t n_mels) {
    return fixed_mel_frames_for(ctx, n_mels);
}

int64_t parakeet_coreml_fixed_output_rows(
    const struct parakeet_coreml_context * ctx,
    int64_t cols) {
    return ctx != nullptr ? trailing_partner(ctx->declared_output_dims, cols) : 0;
}

int parakeet_coreml_encode(struct parakeet_coreml_context * ctx,
                           int64_t       n_mel_frames,
                           int64_t       n_mel_valid,
                           int64_t       n_mels,
                           const float * mel,
                           int64_t       n_enc_frames,
                           int64_t       d_model,
                           float       * encoder_out) {
    if (ctx == nullptr || ctx->model == nullptr || mel == nullptr || encoder_out == nullptr) return 1;
    if (!ctx->valid_mask_name.empty()) return 1;
    if (n_mel_frames <= 0 || n_mels <= 0 || n_enc_frames <= 0 || d_model <= 0) return 1;

    @autoreleasepool {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        if (ctx->ane.active) {
            return encode_ane_window(ctx, n_mel_frames, n_mel_valid, n_mels, mel, n_enc_frames,
                                     d_model, encoder_out);
        }
        MLModel  * model    = (__bridge MLModel *) ctx->model;
        NSString * in_name  = [NSString stringWithUTF8String:ctx->input_name.c_str()];
        NSString * out_name = [NSString stringWithUTF8String:ctx->output_name.c_str()];

        if (!prepare_cached_io(ctx, model, in_name, out_name, n_mel_frames, n_mels,
                               n_enc_frames, d_model)) return 2;
        MLMultiArray * in_arr = (__bridge MLMultiArray *) ctx->input_array;
        if (!fill_input_array(in_arr, ctx->input_transpose, mel, n_mel_frames, ctx->cached_mel_frames, n_mels)) return 3;

        NSError * err = nil;
        MLDictionaryFeatureProvider * provider =
            (__bridge MLDictionaryFeatureProvider *) ctx->input_provider;
        MLPredictionOptions * options = (__bridge MLPredictionOptions *) ctx->prediction_options;
        id<MLFeatureProvider> result = [model predictionFromFeatures:provider options:options error:&err];
        if (result == nil || err != nil) return 5;

        MLMultiArray * out_arr = [result featureValueForName:out_name].multiArrayValue;
        if (out_arr == nil) return 6;
        if (!copy_output_array(out_arr, encoder_out, n_enc_frames, d_model)) return 7;
        return 0;
    }
}

int parakeet_coreml_encode_embeddings(
        struct parakeet_coreml_context * ctx,
        int64_t n_encoder_frames,
        int64_t d_model,
        const float * pre_encode,
        float * encoder_out) {
    if (ctx == nullptr || ctx->model == nullptr || pre_encode == nullptr ||
        encoder_out == nullptr || ctx->valid_mask_name.empty()) return 1;
    if (n_encoder_frames <= 0 || d_model <= 0) return 1;

    @autoreleasepool {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        MLModel * model = (__bridge MLModel *) ctx->model;
        NSString * in_name = [NSString stringWithUTF8String:ctx->input_name.c_str()];
        NSString * mask_name = [NSString stringWithUTF8String:ctx->valid_mask_name.c_str()];
        NSString * out_name = [NSString stringWithUTF8String:ctx->output_name.c_str()];

        const int64_t capacity = fixed_mel_frames_for(ctx, d_model);
        if (capacity <= 0 || n_encoder_frames > capacity) return 2;

        if (!prepare_cached_bypass_io(ctx, model, in_name, mask_name, out_name,
                                      capacity, d_model)) return 2;
        MLMultiArray * in_arr = (__bridge MLMultiArray *) ctx->input_array;
        MLMultiArray * mask_arr = (__bridge MLMultiArray *) ctx->valid_mask_array;
        if (!fill_input_array(in_arr, ctx->input_transpose, pre_encode, n_encoder_frames,
                              capacity, d_model)) return 3;
        if (!fill_valid_mask(mask_arr, n_encoder_frames, capacity)) return 3;

        NSError * err = nil;
        MLDictionaryFeatureProvider * provider =
            (__bridge MLDictionaryFeatureProvider *) ctx->input_provider;
        MLPredictionOptions * options =
            (__bridge MLPredictionOptions *) ctx->prediction_options;
        id<MLFeatureProvider> result =
            [model predictionFromFeatures:provider options:options error:&err];
        if (result == nil || err != nil) return 5;

        MLMultiArray * out_arr = [result featureValueForName:out_name].multiArrayValue;
        if (out_arr == nil) return 6;
        if (!copy_output_array(out_arr, encoder_out, n_encoder_frames, d_model)) return 7;
        return 0;
    }
}

const char * parakeet_coreml_backend_label(const struct parakeet_coreml_context * ctx) {
    return ctx != nullptr ? ctx->label.c_str() : "coreml";
}

int64_t parakeet_coreml_seam_context_frames(const struct parakeet_coreml_context * ctx) {
    if (ctx == nullptr || !ctx->ane.active) return 0;
    const int64_t cap = ctx->ane.enc_frames / 4;
    int64_t frames = std::min<int64_t>(kAneSeamContextFramesDefault, cap);
    if (const char * env = std::getenv(kAneSeamContextEnv)) {
        frames = std::min<int64_t>(std::max<int64_t>(0, std::atoll(env)), cap);
    }
    return frames;
}
