// Staged parity test for T3 multilingual (Llama-520M).
//
// Each stage runs a narrow slice of the C++ forward pass with the rest
// injected from Python-dumped .npy references.  Walk bottom-up to localise
// any numerical drift to a specific sub-stage:
//
//   cond      : build cond_emb (spkr_enc + perceiver + emotion) only
//   text      : text_emb + learned text_pos_emb
//   inputs    : full input assembly (cond_emb + text + initial_speech + bos)
//   layer0    : one Llama block on top of Python-injected inputs_embeds
//   layerN    : first N Llama blocks on Python-injected inputs_embeds
//   full      : all 30 layers + final RMSNorm + speech_head
//   all       : run each stage in turn and stop at the first failure

#include "chatterbox_t3_internal.h"
#include "t3_mtl.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "npy.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

// Pull the implementation-detail symbols (`chatterbox_model`,
// `build_stage_*_graph`, etc.) into the global scope so each `int main`
// stage can stay compact.  Fine for a test executable; tightly couples
// these tests to detail-namespace renames, which is the right trade-off
// for a parity harness.
using namespace tts_cpp::chatterbox::detail;

namespace {

struct stage_run {
    ggml_gallocr_t allocr = nullptr;
    ggml_backend_t backend = nullptr;
    int n_threads = 4;
};

std::vector<float> run_and_read(stage_run & r, ggml_cgraph * gf, const char * out_name) {
    if (!ggml_gallocr_reserve(r.allocr, gf)) {
        throw std::runtime_error("gallocr_reserve failed");
    }
    ggml_gallocr_alloc_graph(r.allocr, gf);
    if (ggml_backend_is_cpu(r.backend)) {
        ggml_backend_cpu_set_n_threads(r.backend, r.n_threads);
    }
    ggml_backend_graph_compute(r.backend, gf);

    ggml_tensor * out = ggml_graph_get_tensor(gf, out_name);
    if (!out) throw std::runtime_error(std::string("output tensor not found: ") + out_name);
    std::vector<float> vec(ggml_nelements(out));
    ggml_backend_tensor_get(out, vec.data(), 0, ggml_nbytes(out));
    return vec;
}

void set_tensor_if_present(ggml_cgraph * gf, const char * name,
                           const void * data, size_t bytes) {
    ggml_tensor * t = ggml_graph_get_tensor(gf, name);
    if (t) ggml_backend_tensor_set(t, data, 0, bytes);
}

std::vector<int32_t> arange_i32(int n, int start = 0) {
    std::vector<int32_t> v(n);
    for (int i = 0; i < n; ++i) v[i] = start + i;
    return v;
}

std::vector<ggml_fp16_t> causal_mask_f16(int N) {
    std::vector<ggml_fp16_t> m((size_t) N * N);
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            m[(size_t) i * N + j] = (j <= i) ? zero : ninf;
    return m;
}

// Python dumps are numpy (batch, seq, embd).  ggml F32 layout for our 2D
// (n_embd, seq) is the same contiguous `embd`-fastest order as numpy
// (seq, embd) after batch index 0.  Return a pointer into the raw bytes.
const float * row_0(const npy_array & arr, size_t & n) {
    if (arr.dtype != "<f4") throw std::runtime_error("expected float32");
    n = 1;
    for (size_t i = 1; i < arr.shape.size(); ++i) n *= (size_t) arr.shape[i];
    return reinterpret_cast<const float *>(arr.data.data());
}

// Stage: cond_emb only.
bool stage_cond(stage_run & r, const chatterbox_model & model, const std::string & ref_dir) {
    fprintf(stderr, "\n== STAGE cond ==\n");
    ggml_cgraph * gf = build_stage_cond_emb_graph(model);
    if (!ggml_gallocr_reserve(r.allocr, gf)) return false;
    ggml_gallocr_alloc_graph(r.allocr, gf);

    float exag = 0.5f;
    set_tensor_if_present(gf, "exaggeration", &exag, sizeof(exag));
    auto cond_pos = arange_i32(model.hparams.cond_prompt_len);
    set_tensor_if_present(gf, "cond_prompt_pos_ids",
                          cond_pos.data(), cond_pos.size() * sizeof(int32_t));

    if (ggml_backend_is_cpu(r.backend))
        ggml_backend_cpu_set_n_threads(r.backend, r.n_threads);
    ggml_backend_graph_compute(r.backend, gf);

    ggml_tensor * cond = ggml_graph_get_tensor(gf, "cond_emb");
    std::vector<float> got(ggml_nelements(cond));
    ggml_backend_tensor_get(cond, got.data(), 0, ggml_nbytes(cond));

    npy_array ref = npy_load(ref_dir + "/cond_emb.npy");  // (1, 34, n_embd)
    size_t N_ref = 1;
    for (auto d : ref.shape) N_ref *= (size_t) d;
    if (N_ref != got.size()) {
        fprintf(stderr, "stage cond: size mismatch got=%zu ref=%zu\n", got.size(), N_ref);
        return false;
    }
    auto s = compare_f32(got.data(),
                         reinterpret_cast<const float *>(ref.data.data()),
                         got.size());
    print_compare("cond_emb", s);
    // F16-stored speech_emb contributes ~1.5e-4 rel drift in cond_prompt.
    return s.rel_err < 5e-4;
}

bool stage_text(stage_run & r, const chatterbox_model & model, const std::string & ref_dir) {
    fprintf(stderr, "\n== STAGE text ==\n");
    npy_array toks = npy_load(ref_dir + "/text_tokens.npy");
    if (toks.dtype != "<i4") { fprintf(stderr, "text_tokens not i32\n"); return false; }
    const int T_text = (int) toks.n_elements();

    ggml_cgraph * gf = build_stage_text_emb_graph(model, T_text);
    if (!ggml_gallocr_reserve(r.allocr, gf)) return false;
    ggml_gallocr_alloc_graph(r.allocr, gf);

    set_tensor_if_present(gf, "text_tokens", toks.data.data(), toks.data.size());
    auto pos = arange_i32(T_text);
    set_tensor_if_present(gf, "text_pos_ids", pos.data(), pos.size() * sizeof(int32_t));

    if (ggml_backend_is_cpu(r.backend))
        ggml_backend_cpu_set_n_threads(r.backend, r.n_threads);
    ggml_backend_graph_compute(r.backend, gf);

    ggml_tensor * out = ggml_graph_get_tensor(gf, "text_emb_with_pos");
    std::vector<float> got(ggml_nelements(out));
    ggml_backend_tensor_get(out, got.data(), 0, ggml_nbytes(out));

    // Python ref: text_emb_raw (2, T, C) is batched (post-CFG doubling).
    // text_pos_emb_out is (T, C).  Expected cond row = text_emb_raw[0] + text_pos_emb_out.
    npy_array raw = npy_load(ref_dir + "/text_emb_raw.npy");
    npy_array pos_ref = npy_load(ref_dir + "/text_pos_emb_out.npy");
    if (raw.dtype != "<f4" || pos_ref.dtype != "<f4") { fprintf(stderr, "expected f32\n"); return false; }
    if (raw.shape.size() != 3 || raw.shape[0] < 1) { fprintf(stderr, "bad text_emb_raw shape\n"); return false; }
    const int64_t T_ref = raw.shape[1];
    const int64_t C_ref = raw.shape[2];
    if ((int64_t) got.size() != T_ref * C_ref) {
        fprintf(stderr, "size mismatch got=%zu ref=%lld\n", got.size(), (long long)(T_ref * C_ref));
        return false;
    }
    const float * raw_p = reinterpret_cast<const float *>(raw.data.data());
    const float * pos_p = reinterpret_cast<const float *>(pos_ref.data.data());
    std::vector<float> expected((size_t)(T_ref * C_ref));
    for (int64_t t = 0; t < T_ref; ++t)
        for (int64_t c = 0; c < C_ref; ++c)
            expected[(size_t)(t * C_ref + c)] =
                raw_p[0 * T_ref * C_ref + t * C_ref + c] + pos_p[t * C_ref + c];
    auto s = compare_f32(got.data(), expected.data(), got.size());
    print_compare("text_emb + pos (cond row)", s);
    return s.rel_err < 5e-4;
}

bool stage_inputs(stage_run & r, const chatterbox_model & model,
                  const std::string & ref_dir, bool is_uncond) {
    fprintf(stderr, "\n== STAGE inputs (%s) ==\n", is_uncond ? "uncond" : "cond");
    npy_array toks = npy_load(ref_dir + "/text_tokens.npy");
    const int T_text = (int) toks.n_elements();
    ggml_cgraph * gf = build_stage_inputs_graph(model, T_text, is_uncond);
    if (!ggml_gallocr_reserve(r.allocr, gf)) return false;
    ggml_gallocr_alloc_graph(r.allocr, gf);

    set_tensor_if_present(gf, "text_tokens", toks.data.data(), toks.data.size());
    auto tp = arange_i32(T_text);
    set_tensor_if_present(gf, "text_pos_ids", tp.data(), tp.size() * sizeof(int32_t));
    int32_t sbos = model.hparams.start_speech_token;
    set_tensor_if_present(gf, "speech_bos", &sbos, sizeof(sbos));
    int32_t sp0 = 0;
    set_tensor_if_present(gf, "speech_pos0", &sp0, sizeof(sp0));
    float exag = 0.5f;
    set_tensor_if_present(gf, "exaggeration", &exag, sizeof(exag));
    auto cond_pos = arange_i32(model.hparams.cond_prompt_len);
    set_tensor_if_present(gf, "cond_prompt_pos_ids",
                          cond_pos.data(), cond_pos.size() * sizeof(int32_t));

    if (ggml_backend_is_cpu(r.backend))
        ggml_backend_cpu_set_n_threads(r.backend, r.n_threads);
    ggml_backend_graph_compute(r.backend, gf);

    ggml_tensor * out = ggml_graph_get_tensor(gf, "inputs_embeds");
    std::vector<float> got(ggml_nelements(out));
    ggml_backend_tensor_get(out, got.data(), 0, ggml_nbytes(out));

    npy_array ref = npy_load(ref_dir + "/inputs_embeds_initial.npy");  // (2, N, C)
    if (ref.shape.size() != 3) { fprintf(stderr, "bad inputs_embeds shape\n"); return false; }
    const int64_t T = ref.shape[1];
    const int64_t C = ref.shape[2];
    if ((int64_t) got.size() != T * C) {
        fprintf(stderr, "size mismatch got=%zu ref=%lld\n", got.size(), (long long)(T * C));
        return false;
    }
    const float * ref_p = reinterpret_cast<const float *>(ref.data.data());
    const float * row = ref_p + (is_uncond ? T * C : 0);
    auto s = compare_f32(got.data(), row, got.size());
    print_compare("inputs_embeds", s);
    return s.rel_err < 5e-4;
}

bool stage_layers(stage_run & r, const chatterbox_model & model,
                  const std::string & ref_dir, int n_layers, bool is_uncond) {
    fprintf(stderr, "\n== STAGE layers (%d, %s) ==\n", n_layers, is_uncond ? "uncond" : "cond");
    npy_array inp_ref = npy_load(ref_dir + "/inputs_embeds_initial.npy");
    if (inp_ref.shape.size() != 3) { fprintf(stderr, "bad inputs_embeds shape\n"); return false; }
    const int64_t T = inp_ref.shape[1];
    const int64_t C = inp_ref.shape[2];

    ggml_cgraph * gf = build_stage_layers_graph(model, (int) T, n_layers, is_uncond);
    if (!ggml_gallocr_reserve(r.allocr, gf)) return false;
    ggml_gallocr_alloc_graph(r.allocr, gf);

    const float * ref_p = reinterpret_cast<const float *>(inp_ref.data.data());
    const float * row = ref_p + (is_uncond ? T * C : 0);
    ggml_tensor * inp = ggml_graph_get_tensor(gf, "inputs_embeds");
    ggml_backend_tensor_set(inp, row, 0, (size_t)(T * C) * sizeof(float));

    auto pos = arange_i32((int) T);
    set_tensor_if_present(gf, "pos_ids", pos.data(), pos.size() * sizeof(int32_t));

    auto mask = causal_mask_f16((int) T);
    set_tensor_if_present(gf, "kq_mask", mask.data(), mask.size() * sizeof(ggml_fp16_t));

    if (ggml_backend_is_cpu(r.backend))
        ggml_backend_cpu_set_n_threads(r.backend, r.n_threads);
    ggml_backend_graph_compute(r.backend, gf);

    ggml_tensor * out = ggml_graph_get_tensor(gf, "layers_out");
    std::vector<float> got(ggml_nelements(out));
    ggml_backend_tensor_get(out, got.data(), 0, ggml_nbytes(out));

    // Reference: layer{n_layers-1}_out_call0.npy[0] for cond, [1] for uncond.
    int ref_idx = n_layers - 1;
    std::string ref_name;
    if (ref_idx == 0) ref_name = "layer0_out_call0.npy";
    else if (ref_idx == 1) ref_name = "layer1_out_call0.npy";
    else if (ref_idx == 2) ref_name = "layer2_out_call0.npy";
    else if (ref_idx == 14) ref_name = "layer14_out_call0.npy";
    else if (ref_idx == 29) ref_name = "layer29_out_call0.npy";
    else {
        fprintf(stderr, "no reference dumped for layer %d\n", ref_idx);
        return false;
    }
    npy_array layer_ref = npy_load(ref_dir + "/" + ref_name);
    if (layer_ref.shape.size() != 3) return false;
    const int64_t T2 = layer_ref.shape[1];
    const int64_t C2 = layer_ref.shape[2];
    if ((int64_t) got.size() != T2 * C2) {
        fprintf(stderr, "size mismatch got=%zu ref=%lld\n", got.size(), (long long)(T2 * C2));
        return false;
    }
    const float * layer_p = reinterpret_cast<const float *>(layer_ref.data.data());
    const float * layer_row = layer_p + (is_uncond ? T2 * C2 : 0);
    auto s = compare_f32(got.data(), layer_row, got.size());
    print_compare(ref_name.c_str(), s);
    return s.rel_err < 1e-3;
}

bool stage_logits(stage_run & r, const chatterbox_model & model,
                  const std::string & ref_dir, bool is_uncond) {
    fprintf(stderr, "\n== STAGE logits (%s) ==\n", is_uncond ? "uncond" : "cond");
    const auto & hp = model.hparams;
    npy_array inp_ref = npy_load(ref_dir + "/inputs_embeds_initial.npy");
    if (inp_ref.shape.size() != 3) return false;
    const int64_t T = inp_ref.shape[1];
    const int64_t C = inp_ref.shape[2];

    ggml_cgraph * gl = build_stage_layers_graph(model, (int) T, hp.n_layer, is_uncond);
    if (!ggml_gallocr_reserve(r.allocr, gl)) return false;
    ggml_gallocr_alloc_graph(r.allocr, gl);
    const float * ref_p = reinterpret_cast<const float *>(inp_ref.data.data());
    const float * row = ref_p + (is_uncond ? T * C : 0);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gl, "inputs_embeds"), row, 0,
                            (size_t)(T * C) * sizeof(float));
    auto pos = arange_i32((int) T);
    set_tensor_if_present(gl, "pos_ids", pos.data(), pos.size() * sizeof(int32_t));
    auto mask = causal_mask_f16((int) T);
    set_tensor_if_present(gl, "kq_mask", mask.data(), mask.size() * sizeof(ggml_fp16_t));
    if (ggml_backend_is_cpu(r.backend))
        ggml_backend_cpu_set_n_threads(r.backend, r.n_threads);
    ggml_backend_graph_compute(r.backend, gl);
    ggml_tensor * layers_out = ggml_graph_get_tensor(gl, "layers_out");
    std::vector<float> hidden(ggml_nelements(layers_out));
    ggml_backend_tensor_get(layers_out, hidden.data(), 0, ggml_nbytes(layers_out));

    ggml_cgraph * gh = build_stage_head_graph(model, (int) T);
    if (!ggml_gallocr_reserve(r.allocr, gh)) return false;
    ggml_gallocr_alloc_graph(r.allocr, gh);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gh, "inputs_embeds"), hidden.data(), 0,
                            hidden.size() * sizeof(float));
    ggml_backend_graph_compute(r.backend, gh);
    ggml_tensor * logits = ggml_graph_get_tensor(gh, "logits");
    std::vector<float> got(ggml_nelements(logits));
    ggml_backend_tensor_get(logits, got.data(), 0, ggml_nbytes(logits));

    npy_array lref = npy_load(ref_dir + "/speech_logits_call0.npy");
    if (lref.shape.size() != 3) return false;
    const int64_t T2 = lref.shape[1];
    const int64_t V2 = lref.shape[2];
    const float * lp = reinterpret_cast<const float *>(lref.data.data());
    const float * last_row = lp + (is_uncond ? T2 * V2 : 0) + (T2 - 1) * V2;
    if ((int64_t) got.size() != V2) {
        fprintf(stderr, "logits size mismatch got=%zu ref=%lld\n", got.size(), (long long) V2);
        return false;
    }
    auto s = compare_f32(got.data(), last_row, got.size());
    print_compare("speech_logits (last pos)", s);
    // Accumulated drift through 30 F16 Llama layers + F16 speech_head.
    // Logit magnitudes are O(8) so 5e-3 rel is argmax-safe.
    return s.rel_err < 5e-3;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr,
            "usage: %s MODEL.gguf REF_DIR STAGE [--threads N] [--n-gpu-layers N]\n"
            "  STAGE one of: cond text inputs-cond inputs-uncond layer0 layer1 layer2 layer14 layer29 all\n",
            argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    const std::string ref_dir    = argv[2];
    const std::string stage      = argv[3];
    int n_threads = (int) std::thread::hardware_concurrency();
    int n_gpu_layers = 99;
    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--threads" && i + 1 < argc)      n_threads    = atoi(argv[++i]);
        else if (a == "--n-gpu-layers" && i + 1 < argc) n_gpu_layers = atoi(argv[++i]);
    }
    if (n_threads <= 0) n_threads = 4;

    chatterbox_model model;
    if (!load_model_gguf(model_path, model, /*ctx=*/0, n_gpu_layers)) return 1;
    if (model.hparams.variant != CHBX_VARIANT_MTL) {
        fprintf(stderr, "model is not t3_mtl\n"); return 1;
    }

    stage_run r;
    r.backend = model.backend;
    r.allocr  = ggml_gallocr_new(ggml_backend_get_default_buffer_type(r.backend));
    r.n_threads = n_threads;

    bool ok = true;
    auto dispatch = [&](const std::string & s) -> bool {
        if (s == "cond")          return stage_cond  (r, model, ref_dir);
        if (s == "text")          return stage_text  (r, model, ref_dir);
        if (s == "inputs-cond")   return stage_inputs(r, model, ref_dir, false);
        if (s == "inputs-uncond") return stage_inputs(r, model, ref_dir, true);
        if (s == "layer0")        return stage_layers(r, model, ref_dir, 1,  false);
        if (s == "layer1")        return stage_layers(r, model, ref_dir, 2,  false);
        if (s == "layer2")        return stage_layers(r, model, ref_dir, 3,  false);
        if (s == "layer14")       return stage_layers(r, model, ref_dir, 15, false);
        if (s == "layer29")       return stage_layers(r, model, ref_dir, 30, false);
        if (s == "logits-cond")   return stage_logits(r, model, ref_dir, false);
        if (s == "logits-uncond") return stage_logits(r, model, ref_dir, true);
        fprintf(stderr, "unknown stage: %s\n", s.c_str());
        return false;
    };

    if (stage == "all") {
        const std::vector<std::string> all = {
            "cond", "text", "inputs-cond", "inputs-uncond",
            "layer0", "layer1", "layer2", "layer14", "layer29",
            "logits-cond", "logits-uncond",
        };
        for (const auto & s : all) {
            ok = dispatch(s);
            if (!ok) { fprintf(stderr, "=> STOP at stage '%s'\n", s.c_str()); break; }
        }
    } else {
        ok = dispatch(stage);
    }

    fprintf(stderr, "\n%s\n", ok ? "RESULT: PASS" : "RESULT: FAIL");
    return ok ? 0 : 1;
}
