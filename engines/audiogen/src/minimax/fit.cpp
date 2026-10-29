#include "audiogen-cpp/minimax/fit.h"

#include "acestep/fit_pools.h"
#include "acestep/fit_util.h"
#include "minimax/backend.h"
#include "minimax/logic.h"
#include "minimax/mm3-pipeline.h"
#include "minimax/model-files.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tts_cpp::minimax {
namespace {

using acestep::FIT_HOST_POOL;
using acestep::FitBudgetCheck;
using acestep::FitPool;
using acestep::PoolCharge;
using acestep::fitutil::sat_add;
using acestep::fitutil::sat_mul;

constexpr const char * kFitTag           = "MM3-Fit";
constexpr const char * kDefaultModelName = "MiniMax-Music3";
constexpr const char * kStageLm          = "lm";
constexpr const char * kStageDepth       = "depth";
constexpr const char * kStageCond        = "cond";
constexpr const char * kStageDit         = "dit";
constexpr const char * kStageVocoder     = "vocoder";
constexpr const char * kDepthPrefix      = "depth.";
constexpr const char * kCondPrefix       = "cond.";
constexpr const char * kDitPrefix        = "dit.";
constexpr const char * kVocoderPrefix    = "voc.";

constexpr const char * kReasonFits        = "fits";
constexpr const char * kReasonDoesNotFit  = "does-not-fit";
constexpr const char * kReasonInvalid     = "invalid-arguments";
constexpr const char * kReasonUnreadable  = "model-unreadable";
constexpr const char * kReasonNoBackend   = "no-backend-device";
constexpr const char * kReasonMeasurement = "measurement-failed";
constexpr const char * kReasonTooLarge    = "workload-too-large";
constexpr const char * kReasonCoreML      = "coreml-memory-estimate-unavailable";

constexpr int64_t kDitBranches       = 2;
constexpr int64_t kLmSlots           = 2;
constexpr int64_t kInterleavedCopies = 2;
constexpr size_t  kSchedulerBackends = 2;
constexpr size_t  kHostSchedulerSlot = 1;
constexpr double  kBytesPerMib       = 1024.0 * 1024.0;
constexpr int     kMibPrecision      = 1;

constexpr uintptr_t kPlaceholderBase = 4096;

struct FitError : std::runtime_error {
    const char * reason;

    FitError(const char * fit_reason, const std::string & message) : std::runtime_error(message), reason(fit_reason) {}
};

struct Workload {
    int64_t              frames       = 0;
    int64_t              prompt       = 0;
    int64_t              kv_positions = 0;
    int64_t              prefill_pad  = 0;
    int64_t              decode_pad   = 0;
    int64_t              samples      = 0;
    std::vector<int64_t> window_latents;
    std::set<int64_t>    condition_frames;
    std::set<int64_t>    dit_latents;
    std::set<int64_t>    vocoder_tiles;
};

struct StageBytes {
    uint64_t weights = 0;
    uint64_t state   = 0;
    uint64_t compute = 0;
    uint64_t staging = 0;
    uint64_t arena   = 0;
};

struct GraphBytes {
    uint64_t device = 0;
    uint64_t host   = 0;
};

struct Placement {
    BackendPair                pair;
    ggml_backend_buffer_type_t buft    = nullptr;
    ggml_backend_buffer_t      weights = nullptr;
    ggml_backend_buffer_t      state   = nullptr;
    uintptr_t                  next    = kPlaceholderBase;

    ~Placement() {
        ggml_backend_buffer_free(weights);
        ggml_backend_buffer_free(state);
    }
};

struct Stages {
    StageBytes lm;
    StageBytes depth;
    StageBytes cond;
    StageBytes dit;
    StageBytes vocoder;
    uint64_t   vocoder_prep = 0;
};

struct WeightBytes {
    uint64_t lm      = 0;
    uint64_t depth   = 0;
    uint64_t cond    = 0;
    uint64_t dit     = 0;
    uint64_t vocoder = 0;
};

struct HostPhases {
    uint64_t persistent       = 0;
    uint64_t lm               = 0;
    uint64_t window           = 0;
    uint64_t vocoder_prepare  = 0;
    uint64_t stitch           = 0;
    uint64_t interleave       = 0;
};

struct FitModel {
    MM3Model model;

    ~FitModel() {
        wctx_free(&model.wctx_lm);
        wctx_free(&model.wctx_synth);
    }
};

struct FitBackends {
    BackendPair pair;

    ~FitBackends() { backend_free_pair(pair); }
};

struct SchedulerGuard {
    ggml_backend_sched_t sched = nullptr;

    ~SchedulerGuard() {
        if (sched) {
            ggml_backend_sched_free(sched);
        }
    }
};

struct ContextGuard {
    ggml_context * ctx = nullptr;

    ~ContextGuard() {
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

struct GraphArena {
    ggml_context * ctx  = nullptr;
    uint8_t **     gbuf = nullptr;

    ~GraphArena() {
        if (ctx) {
            ggml_free(ctx);
        }
        free(*gbuf);
        *gbuf = nullptr;
    }
};

[[noreturn]] void fail(const char * reason, const std::string & message) {
    throw FitError(reason, message);
}

uint64_t f32_bytes(uint64_t count) {
    return sat_mul(count, sizeof(float));
}

uint64_t to_u64(int64_t value) {
    return value > 0 ? static_cast<uint64_t>(value) : 0;
}

void validate_workload(const FitWorkload & workload) {
    if (workload.max_frames <= 0) {
        fail(kReasonInvalid, "max_frames must be positive");
    }
    if (workload.prompt_tokens <= 0) {
        fail(kReasonInvalid, "prompt_tokens must be positive");
    }
}

std::string resolve_device(const EngineOptions & options) {
    const std::string device = backend_resolve_device_request(options.device);
    if (!backend_device_request_valid(device)) {
        fail(kReasonInvalid, "device must be cpu, gpu or auto, got '" + device + "'");
    }
    return device;
}

EngineOptions resolve_paths(const EngineOptions & options) {
    if (options.model_dir.empty() && (options.lm_model_path.empty() || options.synth_model_path.empty())) {
        fail(kReasonInvalid, "set model_dir or both the LM and synth GGUF paths");
    }
    try {
        return detail::resolve_model_paths(options);
    } catch (const std::exception & error) {
        fail(kReasonUnreadable, error.what());
    }
}

void probe_metadata(MM3Model & model, const EngineOptions & options) {
    try {
        detail::probe_model_files(model, options, gf_load_metadata);
    } catch (const std::exception & error) {
        fail(kReasonUnreadable, error.what());
    }
}

using TensorLoader = bool (*)(MM3Model *, const GGUFModel &, std::vector<std::string> *);

void load_tensor_metadata(MM3Model & model, const std::string & path, TensorLoader loader) {
    GGUFModel gguf = {};
    if (!gf_load_metadata(&gguf, path.c_str())) {
        fail(kReasonUnreadable, "cannot open " + path);
    }
    std::vector<std::string> errors;
    const bool               loaded = loader(&model, gguf, &errors);
    gf_close(&gguf);
    if (!loaded) {
        fail(kReasonUnreadable, errors.empty() ? "cannot read the tensors of " + path : errors.front());
    }
}

void load_model_metadata(MM3Model & model) {
    load_tensor_metadata(model, model.lm_file.path, mm3_load_lm_tensors);
    load_tensor_metadata(model, model.synth_file.path, mm3_load_synth_tensors);
}

void validate_runtime_contract(const MM3Model & model) {
    std::string error;
    if (!mm3_lm_validate(model, &error) || !mm3_depth_validate(model, &error) || !mm3_cond_validate(model, &error) ||
        !mm3_dit_validate(model, &error) || !mm3_voc_validate(model, &error)) {
        fail(kReasonUnreadable, error);
    }
}

void derive_lm_positions(const MM3LmConfig & config, Workload & workload) {
    if (workload.prompt > static_cast<int64_t>(config.max_prompt_tokens)) {
        fail(kReasonTooLarge, "the prompt exceeds the model token limit");
    }
    if (!mm3_lm_positions_fit(config.context_length, workload.prompt, workload.frames)) {
        fail(kReasonTooLarge, "the prompt and generated frames exceed qwen3.context_length");
    }
    std::string error;
    if (!mm3_lm_kv_positions(config, workload.prompt + workload.frames, &workload.kv_positions, &error)) {
        fail(kReasonTooLarge, error);
    }
    mm3_lm_bucket_for_context(workload.prompt, workload.kv_positions, &workload.prefill_pad);
    mm3_lm_bucket_for_context(workload.prompt + workload.frames, workload.kv_positions, &workload.decode_pad);
}

void add_vocoder_tiles(int64_t latents, std::set<int64_t> & tiles) {
    if (mm3_voc_single_shot(latents, MM3_VOC_CHUNK)) {
        tiles.insert(latents);
        return;
    }
    const int64_t core = mm3_voc_tile_core(MM3_VOC_CHUNK, MM3_VOC_OVERLAP);
    for (int64_t start = 0; start < latents; start += core) {
        const MM3VocTile tile = mm3_voc_tile(start, latents, core, MM3_VOC_OVERLAP);
        tiles.insert(tile.end - tile.start);
    }
}

void add_window(const MM3Model & model, int64_t frames, Workload & workload) {
    const int64_t latents = mm3_cond_latent_length(model.synth_cfg.cond, frames);
    workload.window_latents.push_back(latents);
    workload.condition_frames.insert(frames);
    workload.dit_latents.insert(latents);
    add_vocoder_tiles(latents, workload.vocoder_tiles);
}

void derive_windows(const MM3Model & model, Workload & workload) {
    const MM3DitConfig & dit    = model.synth_cfg.dit;
    const int64_t        window = static_cast<int64_t>(dit.window_frames);
    for (const int64_t start : detail::window_starts(workload.frames, window, static_cast<int64_t>(dit.hop_frames))) {
        add_window(model, std::min(start + window, workload.frames) - start, workload);
    }
    const detail::FlowWindowGeometry geometry = detail::flow_window_geometry(
        static_cast<int64_t>(dit.window_latents), static_cast<int64_t>(dit.hop_latents));
    workload.samples = detail::stitched_sample_count(
        workload.window_latents, static_cast<int64_t>(model.synth_cfg.voc.total_upsample), geometry);
}

Workload derive_workload(const MM3Model & model, const FitWorkload & request) {
    Workload workload;
    workload.frames = std::min<int64_t>(request.max_frames, static_cast<int64_t>(model.lm_cfg.max_audio_frames));
    workload.prompt = request.prompt_tokens;
    derive_lm_positions(model.lm_cfg, workload);
    derive_windows(model, workload);
    return workload;
}

BackendPair create_backends(const std::string & device, const std::string & backends_dir) {
    backend_load_modules(backends_dir);
    tts_cpp::GpuFallbackReason reason = tts_cpp::GpuFallbackReason::not_requested;
    try {
        return backend_create_pair(device, kFitTag, &reason);
    } catch (const std::exception & error) {
        fail(kReasonNoBackend, error.what());
    }
}

uint64_t tensor_bytes(ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    return ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, buft);
}

uintptr_t padded_size(const Placement & placement, const ggml_tensor * tensor) {
    const size_t alignment = ggml_backend_buft_get_alignment(placement.buft);
    return std::max<size_t>(GGML_PAD(ggml_backend_buft_get_alloc_size(placement.buft, tensor), alignment), alignment);
}

void place_tensor(Placement & placement, ggml_tensor * tensor, ggml_backend_buffer_t buffer) {
    tensor->data   = reinterpret_cast<void *>(placement.next);
    tensor->buffer = buffer;
    placement.next += padded_size(placement, tensor);
}

void mark_allocated(ggml_context * ctx, Placement & placement, ggml_backend_buffer_t buffer) {
    for (ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor; tensor = ggml_get_next_tensor(ctx, tensor)) {
        if (!tensor->data && !tensor->view_src) {
            place_tensor(placement, tensor, buffer);
        }
    }
}

uint64_t size_and_mark(ggml_context * ctx, Placement & placement, ggml_backend_buffer_t buffer) {
    const uint64_t bytes = tensor_bytes(ctx, placement.buft);
    mark_allocated(ctx, placement, buffer);
    return bytes;
}

ggml_backend_buffer_t placeholder_buffer(ggml_backend_buffer_type_t buft, ggml_backend_buffer_usage usage) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, 0);
    if (!buffer) {
        fail(kReasonMeasurement, "placeholder buffer allocation failed");
    }
    ggml_backend_buffer_set_usage(buffer, usage);
    return buffer;
}

bool has_prefix(const ggml_tensor * tensor, const char * prefix) {
    return std::strncmp(ggml_get_name(tensor), prefix, std::strlen(prefix)) == 0;
}

uint64_t prefixed_tensor_bytes(ggml_context * ctx, ggml_backend_buffer_type_t buft, const char * prefix) {
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    uint64_t     total     = 0;
    for (ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor; tensor = ggml_get_next_tensor(ctx, tensor)) {
        if (has_prefix(tensor, prefix)) {
            total = sat_add(total, GGML_PAD(ggml_backend_buft_get_alloc_size(buft, tensor), alignment));
        }
    }
    return total;
}

GraphBytes graph_bytes(const Placement & placement, int max_nodes, ggml_cgraph * graph) {
    SchedulerGuard scheduler{ backend_sched_new(placement.pair, max_nodes) };
    size_t         sizes[kSchedulerBackends] = {};
    if (!ggml_backend_sched_reserve_size(scheduler.sched, graph, sizes)) {
        fail(kReasonMeasurement, "the scheduler cannot place the graph");
    }
    return { sizes[0], placement.pair.has_gpu ? sizes[kHostSchedulerSlot] : 0 };
}

GraphBytes add_graph_bytes(const GraphBytes & left, const GraphBytes & right) {
    return { sat_add(left.device, right.device), sat_add(left.host, right.host) };
}

GraphBytes max_graph_bytes(const GraphBytes & left, const GraphBytes & right) {
    return { std::max(left.device, right.device), std::max(left.host, right.host) };
}

void store_compute(StageBytes & stage, const GraphBytes & bytes) {
    stage.compute = bytes.device;
    stage.staging = bytes.host;
}

void require_context(const ggml_context * ctx, const std::string & error) {
    if (!ctx) {
        fail(kReasonMeasurement, error);
    }
}

GraphBytes lm_slot_bytes(const MM3Model & model, const MM3LmGraph & lm, int64_t tokens, int64_t kv_pad, bool decode,
                         const Placement & placement) {
    MM3LmSlot   slot;
    std::string error;
    GraphArena  arena{ mm3_lm_slot_context(&slot, &error), &slot.gbuf };
    require_context(arena.ctx, error);
    return graph_bytes(placement, MM3_LM_MAX_NODES * 2,
                       mm3_lm_slot_graph(arena.ctx, model, lm, &slot, tokens, kv_pad, decode));
}

StageBytes measure_lm(const MM3Model & model, Placement & placement, uint64_t weights,
                      const Workload & workload) {
    MM3LmGraph lm;
    lm.bp             = placement.pair;
    lm.backend        = placement.pair.backend;
    lm.cpu_backend    = placement.pair.cpu_backend;
    lm.use_flash_attn = mm3_lm_use_flash_attn(placement.pair);
    std::string error;
    if (!mm3_lm_define_kv(model.lm_cfg, &lm, workload.kv_positions, &error)) {
        fail(kReasonMeasurement, error);
    }
    ContextGuard kv{ lm.kv_ctx };
    StageBytes   bytes;
    bytes.weights = weights;
    bytes.state   = size_and_mark(lm.kv_ctx, placement, placement.state);
    store_compute(bytes, add_graph_bytes(
                             lm_slot_bytes(model, lm, workload.prompt, workload.prefill_pad, false, placement),
                             lm_slot_bytes(model, lm, 1, workload.decode_pad, true, placement)));
    bytes.arena = sat_mul(mm3_lm_slot_context_bytes(), kLmSlots);
    return bytes;
}

GraphBytes depth_step_bytes(const MM3Model & model, MM3DepthGraph & depth, int codebook,
                            const Placement & placement) {
    MM3DepthStep * step = &depth.step[codebook - 1];
    std::string    error;
    GraphArena     arena{ mm3_depth_step_context(step, &error), &step->gbuf };
    require_context(arena.ctx, error);
    return graph_bytes(placement, MM3_DEPTH_MAX_NODES * 2,
                       mm3_depth_step_graph(arena.ctx, model, depth, step, codebook));
}

GraphBytes depth_steps_bytes(const MM3Model & model, MM3DepthGraph & depth, int steps, const Placement & placement) {
    GraphBytes total;
    for (int codebook = 1; codebook <= steps; codebook++) {
        total = add_graph_bytes(total, depth_step_bytes(model, depth, codebook, placement));
    }
    return total;
}

StageBytes measure_depth(const MM3Model & model, uint64_t weights, Placement & placement) {
    MM3DepthGraph depth;
    wctx_init(&depth.prep, 1);
    ContextGuard prep{ depth.prep.ctx };
    depth.step[0].mask = mm3_depth_mask_tensor(&depth.prep);
    const int   steps  = mm3_depth_step_count(model.synth_cfg.depth);
    std::string error;
    if (!mm3_depth_define_kv(model.synth_cfg.depth, &depth, steps, &error)) {
        fail(kReasonMeasurement, error);
    }
    ContextGuard kv{ depth.kv_ctx };
    StageBytes   bytes;
    bytes.weights = sat_add(weights, size_and_mark(depth.prep.ctx, placement, placement.weights));
    bytes.state   = size_and_mark(depth.kv_ctx, placement, placement.state);
    store_compute(bytes, depth_steps_bytes(model, depth, steps, placement));
    bytes.arena = sat_mul(mm3_depth_step_context_bytes(), static_cast<uint64_t>(steps));
    return bytes;
}

GraphBytes cond_graph_bytes(const MM3Model & model, MM3CondGraph & cond, int64_t frames,
                            const Placement & placement) {
    std::string error;
    GraphArena  arena{ mm3_cond_graph_context(&cond, &error), &cond.gbuf };
    require_context(arena.ctx, error);
    return graph_bytes(placement, MM3_COND_MAX_NODES * 2, mm3_cond_define_graph(arena.ctx, model, &cond, frames));
}

GraphBytes largest_cond_graph_bytes(const MM3Model & model, MM3CondGraph & cond, const std::set<int64_t> & frames,
                                    const Placement & placement) {
    GraphBytes largest;
    for (const int64_t count : frames) {
        largest = max_graph_bytes(largest, cond_graph_bytes(model, cond, count, placement));
    }
    return largest;
}

StageBytes measure_cond(const MM3Model & model, uint64_t weights, const Workload & workload,
                        Placement & placement) {
    MM3CondGraph cond;
    wctx_init(&cond.prep, 1);
    ContextGuard prep{ cond.prep.ctx };
    cond.mix = mm3_cond_mix_tensor(&cond.prep, ggml_nelements(model.synth.cond.layer_logits));
    StageBytes bytes;
    bytes.weights = sat_add(weights, size_and_mark(cond.prep.ctx, placement, placement.weights));
    store_compute(bytes, largest_cond_graph_bytes(model, cond, workload.condition_frames, placement));
    bytes.arena = mm3_cond_graph_context_bytes();
    return bytes;
}

GraphBytes dit_graph_bytes(const MM3Model & model, MM3DitGraph & dit, int64_t latents, const Placement & placement) {
    std::string error;
    GraphArena  arena{ mm3_dit_graph_context(&dit, &error), &dit.gbuf };
    require_context(arena.ctx, error);
    return graph_bytes(placement, MM3_DIT_MAX_NODES * 2,
                       mm3_dit_define_graph(arena.ctx, model, &dit, latents, kDitBranches));
}

GraphBytes largest_dit_graph_bytes(const MM3Model & model, MM3DitGraph & dit, const std::set<int64_t> & latents,
                                   const Placement & placement) {
    GraphBytes largest;
    for (const int64_t count : latents) {
        largest = max_graph_bytes(largest, dit_graph_bytes(model, dit, count, placement));
    }
    return largest;
}

StageBytes measure_dit(const MM3Model & model, uint64_t weights, const Workload & workload,
                       const Placement & placement) {
    MM3DitGraph dit;
    dit.use_flash_attn = mm3_dit_use_flash_attn(placement.pair.has_gpu);
    StageBytes bytes;
    bytes.weights = weights;
    store_compute(bytes, largest_dit_graph_bytes(model, dit, workload.dit_latents, placement));
    bytes.arena = sat_add(mm3_dit_graph_context_bytes(), f32_bytes(to_u64(ggml_nelements(model.synth.dit.time_fourier))));
    return bytes;
}

GraphBytes vocoder_graph_bytes(const MM3Model & model, MM3VocGraph & vocoder, int64_t latents,
                               const Placement & placement) {
    std::string error;
    GraphArena  arena{ mm3_voc_graph_context(&vocoder, &error), &vocoder.gbuf };
    require_context(arena.ctx, error);
    return graph_bytes(placement, MM3_VOC_MAX_NODES * 2, mm3_voc_define_graph(arena.ctx, model, &vocoder, latents));
}

GraphBytes largest_vocoder_graph_bytes(const MM3Model & model, MM3VocGraph & vocoder, const std::set<int64_t> & tiles,
                                       const Placement & placement) {
    GraphBytes largest;
    for (const int64_t tile : tiles) {
        largest = max_graph_bytes(largest, vocoder_graph_bytes(model, vocoder, tile, placement));
    }
    return largest;
}

StageBytes measure_vocoder(const MM3Model & model, uint64_t weights, const Workload & workload,
                           Placement & placement, uint64_t & prep_bytes) {
    MM3VocGraph vocoder;
    mm3_voc_define_prep(model, &vocoder);
    ContextGuard prep{ vocoder.prep.ctx };
    prep_bytes = size_and_mark(vocoder.prep.ctx, placement, placement.weights);
    StageBytes bytes;
    bytes.weights = sat_add(weights, prep_bytes);
    store_compute(bytes, largest_vocoder_graph_bytes(model, vocoder, workload.vocoder_tiles, placement));
    bytes.arena = mm3_voc_graph_context_bytes();
    return bytes;
}

WeightBytes measure_weights(const MM3Model & model, Placement & placement) {
    ggml_context * synth = model.wctx_synth.ctx;
    WeightBytes    weights;
    weights.lm      = tensor_bytes(model.wctx_lm.ctx, placement.buft);
    weights.depth   = prefixed_tensor_bytes(synth, placement.buft, kDepthPrefix);
    weights.cond    = prefixed_tensor_bytes(synth, placement.buft, kCondPrefix);
    weights.dit     = prefixed_tensor_bytes(synth, placement.buft, kDitPrefix);
    weights.vocoder = prefixed_tensor_bytes(synth, placement.buft, kVocoderPrefix);
    mark_allocated(model.wctx_lm.ctx, placement, placement.weights);
    mark_allocated(synth, placement, placement.weights);
    return weights;
}

Stages measure_stages(const MM3Model & model, const BackendPair & pair, const Workload & workload) {
    Placement placement;
    placement.pair    = pair;
    placement.buft    = ggml_backend_get_default_buffer_type(pair.backend);
    placement.weights = placeholder_buffer(placement.buft, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    placement.state   = placeholder_buffer(placement.buft, GGML_BACKEND_BUFFER_USAGE_ANY);
    const WeightBytes weights = measure_weights(model, placement);
    Stages            stages;
    stages.lm      = measure_lm(model, placement, weights.lm, workload);
    stages.depth   = measure_depth(model, weights.depth, placement);
    stages.cond    = measure_cond(model, weights.cond, workload, placement);
    stages.dit     = measure_dit(model, weights.dit, workload, placement);
    stages.vocoder = measure_vocoder(model, weights.vocoder, workload, placement, stages.vocoder_prep);
    return stages;
}

uint64_t frame_hidden_bytes(const MM3Model & model, const Workload & workload) {
    const uint64_t rows = sat_mul(to_u64(workload.frames), model.lm_cfg.num_codebooks);
    return f32_bytes(sat_mul(rows, model.lm_cfg.embedding_length));
}

uint64_t ar_code_bytes(const MM3Model & model, const Workload & workload) {
    return sat_mul(sat_mul(to_u64(workload.frames) + 1, model.lm_cfg.num_codebooks), sizeof(int32_t));
}

uint64_t lm_staging_bytes(const Workload & workload) {
    const uint64_t columns = sat_mul(to_u64(workload.prompt), MM3_LM_CFG_ROWS);
    const uint64_t ids     = sat_mul(columns, sizeof(int32_t) + sizeof(int32_t) + sizeof(int64_t));
    const uint64_t mask    = sat_mul(sat_mul(to_u64(workload.prefill_pad), to_u64(workload.prompt)), sizeof(uint16_t));
    return sat_add(ids, mask);
}

uint64_t window_latent_bytes(const MM3Model & model, const Workload & workload) {
    uint64_t total = 0;
    for (const int64_t latents : workload.window_latents) {
        total = sat_add(total, f32_bytes(sat_mul(model.synth_cfg.dit.in_channels, to_u64(latents))));
    }
    return total;
}

uint64_t window_transient_bytes(const MM3Model & model, const Workload & workload) {
    const uint64_t channels   = model.synth_cfg.dit.in_channels;
    const uint64_t condition  = model.synth_cfg.cond.out_dim;
    const uint64_t latents    = to_u64(*workload.dit_latents.rbegin());
    const uint64_t per_latent = sat_add(condition, sat_mul(channels, 3));
    const detail::FlowWindowGeometry geometry = detail::flow_window_geometry(
        static_cast<int64_t>(model.synth_cfg.dit.window_latents), static_cast<int64_t>(model.synth_cfg.dit.hop_latents));
    const uint64_t carry = sat_mul(to_u64(geometry.carry_span), sat_add(channels, condition));
    return f32_bytes(sat_add(sat_mul(per_latent, latents), carry));
}

uint64_t waveform_bytes(const MM3Model & model, const Workload & workload) {
    const uint64_t per_latent = sat_mul(model.synth_cfg.voc.channels, model.synth_cfg.voc.total_upsample);
    uint64_t       total      = 0;
    for (const int64_t latents : workload.window_latents) {
        total = sat_add(total, f32_bytes(sat_mul(per_latent, to_u64(latents))));
    }
    return total;
}

uint64_t tile_staging_bytes(const MM3Model & model, const Workload & workload) {
    const uint64_t tile       = to_u64(*workload.vocoder_tiles.rbegin());
    const uint64_t per_latent = sat_add(model.synth_cfg.voc.fold_channels, model.synth_cfg.voc.total_upsample);
    return f32_bytes(sat_mul(per_latent, tile));
}

uint64_t audio_bytes(const MM3Model & model, const Workload & workload) {
    return f32_bytes(sat_mul(model.synth_cfg.voc.channels, to_u64(workload.samples)));
}

uint64_t largest_convt_bytes(const MM3VocWeights & vocoder) {
    uint64_t largest = 0;
    for (const MM3VocBlock & block : vocoder.blk) {
        largest = std::max<uint64_t>(largest, ggml_nbytes(block.convt_w));
    }
    return largest;
}

uint64_t stage_host(const StageBytes & stage) {
    return sat_add(stage.staging, stage.arena);
}

HostPhases measure_host(const MM3Model & model, const Stages & stages, const Workload & workload) {
    const uint64_t hiddens = frame_hidden_bytes(model, workload);
    const uint64_t windows = sat_add(hiddens, window_latent_bytes(model, workload));
    const uint64_t audio   = audio_bytes(model, workload);
    HostPhases     phases;
    phases.persistent      = sat_add(sat_add(stage_host(stages.depth), stage_host(stages.cond)),
                                     sat_add(stage_host(stages.dit), stage_host(stages.vocoder)));
    phases.lm              = sat_add(sat_add(stage_host(stages.lm), hiddens),
                                     sat_add(ar_code_bytes(model, workload), lm_staging_bytes(workload)));
    phases.window          = sat_add(windows, window_transient_bytes(model, workload));
    phases.vocoder_prepare = sat_add(windows, sat_add(stages.vocoder_prep, largest_convt_bytes(model.synth.voc)));
    phases.stitch          = sat_add(sat_add(windows, waveform_bytes(model, workload)),
                                     sat_add(tile_staging_bytes(model, workload), audio));
    phases.interleave      = sat_add(hiddens, sat_mul(audio, kInterleavedCopies));
    return phases;
}

uint64_t stage_total(const StageBytes & stage) {
    return sat_add(sat_add(stage.weights, stage.state), stage.compute);
}

uint64_t resident_device_bytes(const Stages & stages) {
    const uint64_t synth = sat_add(sat_add(stage_total(stages.depth), stage_total(stages.cond)),
                                   sat_add(stage_total(stages.dit), stage_total(stages.vocoder)));
    return sat_add(stages.lm.weights, synth);
}

uint64_t lm_device_bytes(const Stages & stages) {
    return sat_add(stages.lm.state, stages.lm.compute);
}

PoolCharge phase_charge(const std::vector<FitPool> & pools, size_t primary, uint64_t device, uint64_t host) {
    PoolCharge charge(pools.size());
    charge.add_to(primary, device);
    charge.add_host(host);
    return charge;
}

PoolCharge peak_charge(const std::vector<FitPool> & pools, size_t primary, const Stages & stages,
                       const HostPhases & host) {
    const uint64_t resident = resident_device_bytes(stages);
    const uint64_t lm       = sat_add(resident, lm_device_bytes(stages));
    PoolCharge     peak(pools.size());
    peak.max_with(phase_charge(pools, primary, lm, sat_add(host.persistent, host.lm)));
    peak.max_with(phase_charge(pools, primary, resident, sat_add(host.persistent, host.window)));
    peak.max_with(phase_charge(pools, primary, resident, sat_add(host.persistent, host.vocoder_prepare)));
    peak.max_with(phase_charge(pools, primary, resident, sat_add(host.persistent, host.stitch)));
    peak.max_with(phase_charge(pools, primary, resident, sat_add(host.persistent, host.interleave)));
    return peak;
}

std::vector<FitPool> register_pools(const BackendPair & pair, size_t & primary) {
    std::vector<FitPool> pools;
    acestep::fit_pool_register(pools, acestep::fit_pool_for_device(ggml_backend_get_device(pair.cpu_backend)));
    acestep::fit_pool_register(pools, acestep::fit_pool_for_device(ggml_backend_get_device(pair.backend)));
    primary = acestep::fit_pool_of(pools, ggml_backend_get_device(pair.backend));
    return pools;
}

FitStageProjection stage_row(const char * name, const BackendPair & pair, const StageBytes & bytes, uint64_t host) {
    FitStageProjection row;
    ggml_backend_dev_t dev = ggml_backend_get_device(pair.backend);
    row.name          = name;
    row.device_name   = dev ? ggml_backend_dev_name(dev) : "CPU";
    row.on_gpu        = pair.has_gpu;
    row.weights_bytes = bytes.weights;
    row.state_bytes   = bytes.state;
    row.compute_bytes = bytes.compute;
    row.host_bytes    = host;
    return row;
}

std::vector<FitStageProjection> stage_rows(const BackendPair & pair, const Stages & stages, const HostPhases & host) {
    const uint64_t vocoder_phase = std::max({ host.vocoder_prepare, host.stitch, host.interleave });
    return { stage_row(kStageLm, pair, stages.lm, host.lm),
             stage_row(kStageDepth, pair, stages.depth, stage_host(stages.depth)),
             stage_row(kStageCond, pair, stages.cond, stage_host(stages.cond)),
             stage_row(kStageDit, pair, stages.dit, sat_add(stage_host(stages.dit), host.window)),
             stage_row(kStageVocoder, pair, stages.vocoder, sat_add(stage_host(stages.vocoder), vocoder_phase)) };
}

std::string format_mib(uint64_t bytes) {
    std::ostringstream text;
    text << std::fixed << std::setprecision(kMibPrecision) << static_cast<double>(bytes) / kBytesPerMib << " MiB";
    return text.str();
}

void report_device(std::ostringstream & report, const FitResult & result, bool separate_host) {
    report << "device:   " << result.device_name << " (" << (result.device_is_cpu ? "CPU" : "GPU") << "), free "
           << format_mib(result.device_free_bytes) << " / total " << format_mib(result.device_total_bytes)
           << (result.device_shares_host_memory ? " (shares host RAM)" : "") << '\n';
    if (separate_host) {
        report << "host:     free " << format_mib(result.host_free_bytes) << " / total "
               << format_mib(result.host_total_bytes) << '\n';
    }
}

void report_workload(std::ostringstream & report, const MM3Model & model, const Workload & workload) {
    const double seconds = static_cast<double>(workload.frames) / static_cast<double>(model.lm_cfg.frame_rate);
    report << "workload: " << workload.frames << " frames (" << std::fixed << std::setprecision(kMibPrecision)
           << seconds << " s), prompt " << workload.prompt << " tokens -> KV " << workload.kv_positions
           << " positions, " << workload.window_latents.size() << " window(s), DiT L="
           << *workload.dit_latents.rbegin() << ", vocoder tile " << *workload.vocoder_tiles.rbegin() << '\n';
}

void report_stages(std::ostringstream & report, const std::vector<FitStageProjection> & stages) {
    report << "stages (all resident; the LM KV cache and graphs are freed after the AR stage):\n";
    for (const FitStageProjection & stage : stages) {
        report << "  " << std::left << std::setw(8) << stage.name << ' ' << stage.device_name << "  weights "
               << format_mib(stage.weights_bytes) << ", state " << format_mib(stage.state_bytes) << ", compute "
               << format_mib(stage.compute_bytes) << ", host " << format_mib(stage.host_bytes) << '\n';
    }
}

void report_verdict(std::ostringstream & report, const FitResult & result, uint64_t margin, uint64_t headroom) {
    report << "peak:     device " << format_mib(result.peak_device_bytes) << ", host "
           << format_mib(result.peak_host_bytes) << ", margin " << format_mib(margin) << '\n';
    if (result.fits) {
        report << "verdict:  FITS (headroom " << format_mib(headroom) << ")\n";
    } else {
        report << "verdict:  DOES NOT FIT\n";
    }
}

std::string build_report(const FitResult & result, const MM3Model & model, const Workload & workload,
                         bool separate_host, uint64_t margin, uint64_t headroom) {
    std::ostringstream report;
    report << "model:    " << result.model_name << '\n';
    report_device(report, result, separate_host);
    report_workload(report, model, workload);
    report_stages(report, result.stages);
    report_verdict(report, result, margin, headroom);
    return report.str();
}

std::string model_name(const MM3Model & model) {
    return model.lm_file.general_name.empty() ? kDefaultModelName : model.lm_file.general_name;
}

void fill_pools(FitResult & result, const std::vector<FitPool> & pools, size_t primary, const PoolCharge & peak) {
    result.device_shares_host_memory = pools[primary].shares_host_memory;
    result.device_free_bytes         = pools[primary].free_bytes;
    result.device_total_bytes        = pools[primary].total_bytes;
    result.host_free_bytes           = pools[FIT_HOST_POOL].free_bytes;
    result.host_total_bytes          = pools[FIT_HOST_POOL].total_bytes;
    result.peak_device_bytes         = peak.bytes[primary];
    result.peak_host_bytes           = peak.bytes[FIT_HOST_POOL];
}

FitResult project(const EngineOptions & options, const FitWorkload & request) {
    validate_workload(request);
    if (!options.coreml_conditioner_path.empty() || !options.coreml_profile_dir.empty() ||
        !options.coreml_compute_units.empty()) {
        throw FitError(kReasonCoreML, "Core ML resident models and prediction "
                                      "buffers cannot be projected.");
    }
    const std::string   device   = resolve_device(options);
    const EngineOptions resolved = resolve_paths(options);
    FitModel            fit_model;
    MM3Model &          model = fit_model.model;
    probe_metadata(model, resolved);
    load_model_metadata(model);
    validate_runtime_contract(model);
    const Workload workload = derive_workload(model, request);

    FitBackends backends;
    backends.pair = create_backends(device, options.backends_dir);
    const BackendPair & pair = backends.pair;

    const Stages     stages = measure_stages(model, pair, workload);
    const HostPhases host   = measure_host(model, stages, workload);

    size_t                            primary = FIT_HOST_POOL;
    const std::vector<FitPool>        pools   = register_pools(pair, primary);
    const PoolCharge                  peak    = peak_charge(pools, primary, stages, host);
    const std::vector<FitBudgetCheck> checks  = acestep::fit_budget_checks(pools, peak, primary, request.margin_bytes);

    FitResult result;
    result.model_name      = model_name(model);
    result.device_name     = ggml_backend_dev_name(pools[primary].device);
    result.device_is_cpu   = !pair.has_gpu;
    result.stages_resident = true;
    result.stages          = stage_rows(pair, stages, host);
    fill_pools(result, pools, primary, peak);
    result.fits   = acestep::fit_checks_hold(checks);
    result.status = result.fits ? FitStatus::Success : FitStatus::Failure;
    result.reason = result.fits ? kReasonFits : kReasonDoesNotFit;
    const bool separate_host = primary != FIT_HOST_POOL && acestep::fit_shared_budget_pool(pools, primary) == FIT_HOST_POOL;
    result.report = build_report(result, model, workload, separate_host, request.margin_bytes,
                                 acestep::fit_checks_headroom(checks));
    return result;
}

FitResult error_result(const char * reason, const std::string & message) {
    FitResult result;
    result.reason = reason;
    result.report = message;
    return result;
}

}

FitResult fit_params(const EngineOptions & options, const FitWorkload & workload) {
    try {
        return project(options, workload);
    } catch (const FitError & error) {
        return error_result(error.reason, error.what());
    } catch (const std::exception & error) {
        return error_result(kReasonMeasurement, error.what());
    }
}

}
