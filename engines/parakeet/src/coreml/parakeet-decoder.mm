// Core ML TDT decoder sidecar for parakeet-cpp.
//
// Companion to parakeet-encoder.mm. The decoder is dispatched once per joint
// step, so the input MLMultiArrays, the feature provider and the output
// backings are allocated once in the context and reused by every dispatch.
//
// The lane semantics, the I/O contract, and what stays on the host are all
// documented in parakeet-decoder.h; this file only implements them.

#if !__has_feature(objc_arc)
#error "parakeet-decoder.mm must be compiled with -fobjc-arc"
#endif

#import "coreml/parakeet-decoder.h"
#include "coreml/parakeet_coreml_units.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Feature names the exporter (scripts/export-decoder-coreml-ane.py) declares.
NSString * const kEmbedInput   = @"embed";
NSString * const kHInInput     = @"h_in";
NSString * const kCInInput     = @"c_in";
NSString * const kEncStepInput = @"enc_step";
NSString * const kLogitsOutput = @"logits";
NSString * const kHOutOutput   = @"h_out";
NSString * const kCOutOutput   = @"c_out";

// fp16 conversion, in bulk. Uses the hardware conversion where the compiler
// provides an IEEE half type, with a portable bit-level fallback for builds
// that lack one (e.g. x86_64 macOS on an older toolchain).
#if defined(__ARM_FP16_FORMAT_IEEE)

using half_t = __fp16;

void f32_to_f16(const float * src, void * dst, size_t n) {
    half_t * out = (half_t *) dst;
    for (size_t i = 0; i < n; ++i) out[i] = (half_t) src[i];
}

void f16_to_f32(const void * src, float * dst, size_t n) {
    const half_t * in = (const half_t *) src;
    for (size_t i = 0; i < n; ++i) dst[i] = (float) in[i];
}

#else

uint16_t encode_float16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t        exp  = (int32_t) ((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t       mant = bits & 0x7fffffu;

    if (exp >= 0x1f) return (uint16_t) (sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t) sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t) (14 - exp);
        return (uint16_t) (sign | (mant >> shift));
    }
    return (uint16_t) (sign | ((uint32_t) exp << 10) | (mant >> 13));
}

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

void f32_to_f16(const float * src, void * dst, size_t n) {
    uint16_t * out = (uint16_t *) dst;
    for (size_t i = 0; i < n; ++i) out[i] = encode_float16(src[i]);
}

void f16_to_f32(const void * src, float * dst, size_t n) {
    const uint16_t * in = (const uint16_t *) src;
    for (size_t i = 0; i < n; ++i) dst[i] = decode_float16(in[i]);
}

#endif  // __ARM_FP16_FORMAT_IEEE

std::vector<int64_t> dims_of(NSArray<NSNumber *> * shape) {
    std::vector<int64_t> dims;
    for (NSNumber * dim in shape) dims.push_back(dim.longLongValue);
    return dims;
}

// The declared shape of a named MultiArray feature, or empty if the feature is
// missing, not a MultiArray, or not the rank the exporter promises.
std::vector<int64_t> feature_dims(NSDictionary<NSString *, MLFeatureDescription *> * descs,
                                  NSString * name) {
    MLFeatureDescription * desc = descs[name];
    if (desc == nil || desc.type != MLFeatureTypeMultiArray) return {};
    std::vector<int64_t> dims = dims_of(desc.multiArrayConstraint.shape);
    if (dims.size() != 4) return {};
    for (int64_t d : dims) {
        if (d <= 0) return {};  // flexible / symbolic: this graph is fixed-shape
    }
    return dims;
}

MLMultiArray * new_f16_array(NSArray<NSNumber *> * shape) {
    NSError * err = nil;
    MLMultiArray * arr = [[MLMultiArray alloc] initWithShape:shape
                                                   dataType:MLMultiArrayDataTypeFloat16
                                                      error:&err];
    return err == nil ? arr : nil;
}

// True when the array's element strides are the dense row-major strides for its
// shape, which is what lets a whole tensor move in one conversion pass. Freshly
// allocated MLMultiArrays are dense; an array Core ML hands back need not be
// (the ANE may pad channel counts to a hardware multiple), so nothing here
// assumes it.
bool is_dense(MLMultiArray * arr) {
    const std::vector<int64_t> dims    = dims_of(arr.shape);
    const std::vector<int64_t> strides = dims_of(arr.strides);
    if (dims.size() != strides.size() || dims.empty()) return false;
    int64_t expect = 1;
    for (size_t i = dims.size(); i-- > 0; ) {
        if (strides[i] != expect) return false;
        expect *= dims[i];
    }
    return true;
}

void read_strided_output(const uint16_t * src, const std::vector<int64_t> & strides, float * dst,
                         int64_t lanes, int64_t channels, int64_t width) {
    for (int64_t n = 0; n < lanes; ++n) {
        for (int64_t c = 0; c < channels; ++c) {
            const int64_t row = n * strides[0] + c * strides[1];
            float * out = dst + (n * channels + c) * width;
            for (int64_t t = 0; t < width; ++t) {
                const uint16_t h = src[row + t * strides[3]];
                f16_to_f32(&h, out + t, 1);
            }
        }
    }
}

// Reads a [lanes, C, 1, W] fp16 output into a densely packed float32
// [lanes][C][W] host buffer. Takes the dense fast path when it can and walks the
// declared strides when it cannot.
bool read_output(MLMultiArray * arr, float * dst,
                 int64_t lanes, int64_t channels, int64_t width) {
    if (arr == nil) return false;
    if (arr.dataType != MLMultiArrayDataTypeFloat16) return false;

    const std::vector<int64_t> dims = dims_of(arr.shape);
    if (dims.size() != 4 || dims[0] != lanes || dims[1] != channels ||
        dims[2] != 1 || dims[3] != width) {
        return false;
    }

    const void * base = arr.dataPointer;
    if (is_dense(arr)) {
        f16_to_f32(base, dst, (size_t) (lanes * channels * width));
        return true;
    }
    read_strided_output((const uint16_t *) base, dims_of(arr.strides), dst, lanes, channels, width);
    return true;
}

}  // namespace

struct parakeet_coreml_decoder {
    // All CFBridgingRetain'd, released in parakeet_coreml_decoder_free.
    const void * model    = nullptr;  // MLModel *
    const void * provider = nullptr;  // MLDictionaryFeatureProvider *
    const void * options  = nullptr;  // MLPredictionOptions *

    // Input arrays the provider points at. Their contents are overwritten in
    // place per dispatch: the provider holds references, not copies, so one
    // provider built at load time serves every step.
    const void * embed    = nullptr;  // MLMultiArray *
    const void * h_in     = nullptr;
    const void * c_in     = nullptr;
    const void * enc_step = nullptr;

    // Output backings handed to Core ML so it writes into buffers we own and
    // keeps its declared (dense) strides, instead of allocating three arrays per
    // dispatch. Core ML is free to ignore them, so the read path still checks.
    const void * logits   = nullptr;  // MLMultiArray *
    const void * h_out    = nullptr;
    const void * c_out    = nullptr;

    int64_t lanes  = 0;
    int64_t h_pred = 0;
    int64_t h_dim  = 0;
    int64_t d_enc  = 0;
    int64_t width  = 0;
    int64_t v_out  = 0;

    bool inputs_dense = false;

    std::string label;
};

namespace {

// Everything the wrapper will index has to line up, or a mismatched export
// would be discovered as garbage transcripts rather than as a refusal to load.
// h_dim is a whole number of h_pred-sized layers by construction (the exporter
// concatenates the LSTM layers on the channel axis), and the LSTM's own state
// has no time axis.
bool decoder_geometry_ok(const std::vector<int64_t> & embed_d, const std::vector<int64_t> & h_in_d,
                         const std::vector<int64_t> & c_in_d, const std::vector<int64_t> & enc_d,
                         const std::vector<int64_t> & logits_d, const std::vector<int64_t> & h_out_d,
                         const std::vector<int64_t> & c_out_d) {
    const int64_t lanes  = embed_d[0];
    const int64_t h_pred = embed_d[1];
    const int64_t h_dim  = h_in_d[1];
    const int64_t d_enc  = enc_d[1];
    const int64_t width  = enc_d[3];
    const int64_t v_out  = logits_d[1];
    return lanes > 0 && h_pred > 0 && h_dim > 0 && d_enc > 0 && width > 0 && v_out > 0 &&
           h_dim % h_pred == 0 &&
           embed_d[2] == 1 && embed_d[3] == 1 &&
           h_in_d == c_in_d && h_out_d == h_in_d && c_out_d == c_in_d &&
           h_in_d[0] == lanes && h_in_d[2] == 1 && h_in_d[3] == 1 &&
           enc_d[0] == lanes && enc_d[2] == 1 &&
           logits_d[0] == lanes && logits_d[2] == 1 && logits_d[3] == width;
}

}  // namespace

struct parakeet_coreml_decoder * parakeet_coreml_decoder_init(const char * path_mlmodelc) {
    if (path_mlmodelc == nullptr) return nullptr;

    @autoreleasepool {
        NSString * path = [[NSString alloc] initWithUTF8String:path_mlmodelc];
        if (path == nil) return nullptr;

        // The graph is exported for the Neural Engine, so it is pinned to CPU
        // and Neural Engine unless PARAKEET_COREML_COMPUTE_UNITS says otherwise;
        // MLComputeUnitsAll is only a preference and may land it on the GPU.
        MLModelConfiguration * config = [[MLModelConfiguration alloc] init];
        std::string label;
        config.computeUnits = parakeet_coreml_requested_units(
            "PARAKEET_COREML_COMPUTE_UNITS", /*prefer_neural_engine=*/true, &label);

        NSError * err   = nil;
        MLModel * model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:path]
                                            configuration:config
                                                    error:&err];
        if (model == nil || err != nil) return nullptr;

        NSDictionary<NSString *, MLFeatureDescription *> * ins =
            model.modelDescription.inputDescriptionsByName;
        NSDictionary<NSString *, MLFeatureDescription *> * outs =
            model.modelDescription.outputDescriptionsByName;

        const std::vector<int64_t> embed_d  = feature_dims(ins,  kEmbedInput);
        const std::vector<int64_t> h_in_d   = feature_dims(ins,  kHInInput);
        const std::vector<int64_t> c_in_d   = feature_dims(ins,  kCInInput);
        const std::vector<int64_t> enc_d    = feature_dims(ins,  kEncStepInput);
        const std::vector<int64_t> logits_d = feature_dims(outs, kLogitsOutput);
        const std::vector<int64_t> h_out_d  = feature_dims(outs, kHOutOutput);
        const std::vector<int64_t> c_out_d  = feature_dims(outs, kCOutOutput);
        if (embed_d.empty() || h_in_d.empty() || c_in_d.empty() || enc_d.empty() ||
            logits_d.empty() || h_out_d.empty() || c_out_d.empty()) {
            return nullptr;
        }
        if (!decoder_geometry_ok(embed_d, h_in_d, c_in_d, enc_d, logits_d, h_out_d, c_out_d)) {
            return nullptr;
        }

        // Geometry comes off the model description, never from a constant here:
        // the lane count is a property of the export, and the engine's job is to
        // cross-check it against the GGUF rather than to assume it.
        const int64_t lanes  = embed_d[0];
        const int64_t h_pred = embed_d[1];
        const int64_t h_dim  = h_in_d[1];
        const int64_t d_enc  = enc_d[1];
        const int64_t width  = enc_d[3];
        const int64_t v_out  = logits_d[1];

        MLMultiArray * embed    = new_f16_array(@[@(lanes), @(h_pred), @1, @1]);
        MLMultiArray * h_in     = new_f16_array(@[@(lanes), @(h_dim),  @1, @1]);
        MLMultiArray * c_in     = new_f16_array(@[@(lanes), @(h_dim),  @1, @1]);
        MLMultiArray * enc_step = new_f16_array(@[@(lanes), @(d_enc),  @1, @(width)]);
        MLMultiArray * logits   = new_f16_array(@[@(lanes), @(v_out),  @1, @(width)]);
        MLMultiArray * h_out    = new_f16_array(@[@(lanes), @(h_dim),  @1, @1]);
        MLMultiArray * c_out    = new_f16_array(@[@(lanes), @(h_dim),  @1, @1]);
        if (embed == nil || h_in == nil || c_in == nil || enc_step == nil ||
            logits == nil || h_out == nil || c_out == nil) {
            return nullptr;
        }

        MLDictionaryFeatureProvider * provider =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
                kEmbedInput   : embed,
                kHInInput     : h_in,
                kCInInput     : c_in,
                kEncStepInput : enc_step,
            } error:&err];
        if (provider == nil || err != nil) return nullptr;

        MLPredictionOptions * options = [[MLPredictionOptions alloc] init];
        options.outputBackings = @{
            kLogitsOutput : logits,
            kHOutOutput   : h_out,
            kCOutOutput   : c_out,
        };

        auto * ctx = new parakeet_coreml_decoder();
        ctx->model    = CFBridgingRetain(model);
        ctx->provider = CFBridgingRetain(provider);
        ctx->options  = CFBridgingRetain(options);
        ctx->embed    = CFBridgingRetain(embed);
        ctx->h_in     = CFBridgingRetain(h_in);
        ctx->c_in     = CFBridgingRetain(c_in);
        ctx->enc_step = CFBridgingRetain(enc_step);
        ctx->logits   = CFBridgingRetain(logits);
        ctx->h_out    = CFBridgingRetain(h_out);
        ctx->c_out    = CFBridgingRetain(c_out);
        ctx->lanes  = lanes;
        ctx->h_pred = h_pred;
        ctx->h_dim  = h_dim;
        ctx->d_enc  = d_enc;
        ctx->width  = width;
        ctx->v_out  = v_out;
        // Checked once rather than per dispatch: these are our own allocations,
        // so this is invariant for the life of the context.
        ctx->inputs_dense = is_dense(embed) && is_dense(h_in) && is_dense(c_in) &&
                            is_dense(enc_step);
        ctx->label = std::move(label);
        return ctx;
    }
}

void parakeet_coreml_decoder_free(struct parakeet_coreml_decoder * ctx) {
    if (ctx == nullptr) return;
    for (const void * obj : {ctx->model, ctx->provider, ctx->options,
                             ctx->embed, ctx->h_in, ctx->c_in, ctx->enc_step,
                             ctx->logits, ctx->h_out, ctx->c_out}) {
        if (obj != nullptr) CFRelease(obj);
    }
    delete ctx;
}

int parakeet_coreml_decoder_geometry(const struct parakeet_coreml_decoder * ctx,
                                     int64_t * lanes,
                                     int64_t * h_pred,
                                     int64_t * h_dim,
                                     int64_t * d_enc,
                                     int64_t * width,
                                     int64_t * v_out) {
    if (ctx == nullptr) return 1;
    if (lanes  != nullptr) *lanes  = ctx->lanes;
    if (h_pred != nullptr) *h_pred = ctx->h_pred;
    if (h_dim  != nullptr) *h_dim  = ctx->h_dim;
    if (d_enc  != nullptr) *d_enc  = ctx->d_enc;
    if (width  != nullptr) *width  = ctx->width;
    if (v_out  != nullptr) *v_out  = ctx->v_out;
    return 0;
}

int parakeet_coreml_decoder_step(struct parakeet_coreml_decoder * ctx,
                                 const float * embed,
                                 const float * h_in,
                                 const float * c_in,
                                 const float * enc_step,
                                 float       * logits,
                                 float       * h_out,
                                 float       * c_out) {
    if (ctx == nullptr || ctx->model == nullptr) return 1;
    if (embed == nullptr || h_in == nullptr || c_in == nullptr || enc_step == nullptr ||
        logits == nullptr || h_out == nullptr || c_out == nullptr) {
        return 1;
    }
    if (!ctx->inputs_dense) return 2;

    @autoreleasepool {
        MLMultiArray * embed_arr = (__bridge MLMultiArray *) ctx->embed;
        MLMultiArray * h_in_arr  = (__bridge MLMultiArray *) ctx->h_in;
        MLMultiArray * c_in_arr  = (__bridge MLMultiArray *) ctx->c_in;
        MLMultiArray * enc_arr   = (__bridge MLMultiArray *) ctx->enc_step;

        // Every input is read into the model's own fp16 buffers before the
        // prediction runs, which is what makes h_out/c_out aliasing h_in/c_in
        // safe for the caller: the documented way to step state in place.
        f32_to_f16(embed,    embed_arr.dataPointer, (size_t) (ctx->lanes * ctx->h_pred));
        f32_to_f16(h_in,     h_in_arr.dataPointer,  (size_t) (ctx->lanes * ctx->h_dim));
        f32_to_f16(c_in,     c_in_arr.dataPointer,  (size_t) (ctx->lanes * ctx->h_dim));
        f32_to_f16(enc_step, enc_arr.dataPointer,
                   (size_t) (ctx->lanes * ctx->d_enc * ctx->width));

        MLModel * model = (__bridge MLModel *) ctx->model;
        id<MLFeatureProvider> provider = (__bridge id<MLFeatureProvider>) ctx->provider;
        MLPredictionOptions * options = (__bridge MLPredictionOptions *) ctx->options;

        NSError * err = nil;
        id<MLFeatureProvider> result = [model predictionFromFeatures:provider
                                                            options:options
                                                              error:&err];
        if (result == nil || err != nil) return 5;

        MLMultiArray * logits_arr = [result featureValueForName:kLogitsOutput].multiArrayValue;
        MLMultiArray * h_out_arr  = [result featureValueForName:kHOutOutput].multiArrayValue;
        MLMultiArray * c_out_arr  = [result featureValueForName:kCOutOutput].multiArrayValue;
        if (logits_arr == nil || h_out_arr == nil || c_out_arr == nil) return 6;

        if (!read_output(logits_arr, logits, ctx->lanes, ctx->v_out, ctx->width)) return 7;
        if (!read_output(h_out_arr,  h_out,  ctx->lanes, ctx->h_dim, 1))          return 7;
        if (!read_output(c_out_arr,  c_out,  ctx->lanes, ctx->h_dim, 1))          return 7;
        return 0;
    }
}

const char * parakeet_coreml_decoder_backend_label(const struct parakeet_coreml_decoder * ctx) {
    return ctx != nullptr ? ctx->label.c_str() : "coreml";
}
