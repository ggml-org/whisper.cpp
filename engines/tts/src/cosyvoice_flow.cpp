// cosyvoice-flow: CosyVoice3 flow (stage 4) parity CLI.
//
// The flow graph itself now lives in cosyvoice_pipeline.cpp (shared with the
// Engine); this CLI is the numerical parity harness around it:
//   --mode dit    : run the DiT estimator once on dit_in_{x,mu,cond,t,spks},
//                   compare to dit_out.npy      (the core 22-layer gate)
//   --mode flow   : full front-end + Euler loop -> compare flow_mel.npy
//
// Weights come from scripts/convert-cosyvoice3-flow-to-gguf.py (flow/* names).
//
// Usage:
//   cosyvoice-flow --flow-gguf FLOW.gguf --in-dir flow-ref --mode dit

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "npy.h"

#include "cosyvoice_pipeline.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static model_ctx load_gguf(const std::string & path) { return cosyvoice_load_gguf(path); }

// ---------- DiT parity gate (--mode dit) ----------
static int run_dit_test(model_ctx & m, const std::string & in_dir) {
    dit_hp hp;
    auto ld = [&](const std::string & n){ return npy_load(in_dir + "/" + n); };
    npy_array x_a = ld("dit_in_x.npy"), mu_a = ld("dit_in_mu.npy"), cond_a = ld("dit_in_cond.npy");
    npy_array t_a = ld("dit_in_t.npy"), spks_a = ld("dit_in_spks.npy"), out_a = ld("dit_out.npy");
    int B = (int)x_a.shape[0], MEL = (int)x_a.shape[1], N = (int)x_a.shape[2];
    fprintf(stderr, "DiT inputs: B=%d MEL=%d N=%d\n", B, MEL, N);

    static size_t buf_size = (size_t)2 * 1024 * 1024 * 1024;
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params gp = { buf_size, buf.data(), true };
    ggml_context * c = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph_custom(c, 262144, false);

    ggml_tensor * x = ggml_new_tensor_3d(c, GGML_TYPE_F32, MEL, N, B); ggml_set_name(x,"x"); ggml_set_input(x);
    ggml_tensor * mu = ggml_new_tensor_3d(c, GGML_TYPE_F32, MEL, N, B); ggml_set_name(mu,"mu"); ggml_set_input(mu);
    ggml_tensor * cnd = ggml_new_tensor_3d(c, GGML_TYPE_F32, MEL, N, B); ggml_set_name(cnd,"cond"); ggml_set_input(cnd);
    ggml_tensor * spks = ggml_new_tensor_2d(c, GGML_TYPE_F32, 80, B); ggml_set_name(spks,"spks"); ggml_set_input(spks);
    ggml_tensor * tsin = ggml_new_tensor_2d(c, GGML_TYPE_F32, 256, B); ggml_set_name(tsin,"tsin"); ggml_set_input(tsin);
    ggml_tensor * pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, N); ggml_set_name(pos,"pos"); ggml_set_input(pos);

    ggml_tensor * out = build_dit(c, m, hp, x, mu, cnd, spks, tsin, pos, N, B);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);

    auto set_seq = [&](ggml_tensor * dst, const npy_array & a){
        const float * s = npy_as_f32(a);
        std::vector<float> tmp((size_t)MEL*N*B);
        for (int b=0;b<B;++b) for (int me=0; me<MEL; ++me) for (int n=0;n<N;++n)
            tmp[((size_t)b*N + n)*MEL + me] = s[((size_t)b*MEL + me)*N + n];
        ggml_backend_tensor_set(dst, tmp.data(), 0, tmp.size()*sizeof(float));
    };
    set_seq(x, x_a); set_seq(mu, mu_a); set_seq(cnd, cond_a);
    {
        const float * s = npy_as_f32(spks_a);
        std::vector<float> tmp((size_t)80*B);
        for (int b=0;b<B;++b) for (int d=0; d<80; ++d) tmp[(size_t)b*80 + d] = s[(size_t)b*80 + d];
        ggml_backend_tensor_set(spks, tmp.data(), 0, tmp.size()*sizeof(float));
    }
    {
        const float * tv = npy_as_f32(t_a);
        std::vector<float> tvec(tv, tv + B);
        std::vector<float> se = sinus_time_emb(tvec, 256);
        ggml_backend_tensor_set(tsin, se.data(), 0, se.size()*sizeof(float));
    }
    {
        std::vector<int32_t> pv(N);
        for (int i=0;i<N;++i) pv[i]=i;
        ggml_backend_tensor_set(pos, pv.data(), 0, pv.size()*sizeof(int32_t));
    }

    ggml_backend_graph_compute(m.backend, gf);

    std::vector<float> got((size_t)MEL*N*B);
    ggml_backend_tensor_get(out, got.data(), 0, got.size()*sizeof(float));
    std::vector<float> got_ref_order((size_t)MEL*N*B);
    for (int b=0;b<B;++b) for (int me=0; me<MEL; ++me) for (int n=0;n<N;++n)
        got_ref_order[((size_t)b*MEL + me)*N + n] = got[((size_t)b*N + n)*MEL + me];

    compare_stats s = compare_f32(got_ref_order.data(), npy_as_f32(out_a), (size_t)MEL*N*B);
    print_compare("dit_out", s);
    const float * e = npy_as_f32(out_a);
    double dot=0, na=0, nb=0;
    for (size_t i=0;i<got_ref_order.size();++i){ const double a=got_ref_order[i], b=e[i]; dot+=a*b; na+=a*a; nb+=b*b; }
    fprintf(stderr, "  cosine=%.6f\n", dot/(std::sqrt(na)*std::sqrt(nb)));

    ggml_gallocr_free(allocr);
    ggml_free(c);
    return 0;
}

// ---------- Full flow parity (--mode flow) ----------
// Thin harness around cosyvoice_flow_run(): load the prompt/token inputs from
// npy, run the shared pipeline, compare the mel to flow_mel.npy, save.
static int run_flow(model_ctx & m, const std::string & in_dir, const std::string & out_npy) {
    const int MEL = 80;
    npy_array ptok_a = npy_load(in_dir + "/prompt_token.npy");
    npy_array stok_a = npy_load(in_dir + "/speech_tokens.npy");
    npy_array pfeat_a = npy_load(in_dir + "/prompt_feat.npy");
    npy_array emb_a = npy_load(in_dir + "/embedding.npy");
    int T_ptok = (int)ptok_a.shape[0], T_stok = (int)stok_a.shape[0];
    int mel_len1 = (int)pfeat_a.shape[0];

    std::vector<int> ptok(T_ptok), stok(T_stok);
    { const int32_t * p = npy_as_i32(ptok_a); for (int i=0;i<T_ptok;++i) ptok[i]=p[i]; }
    { const int32_t * p = npy_as_i32(stok_a); for (int i=0;i<T_stok;++i) stok[i]=p[i]; }
    std::vector<float> pfeat(npy_as_f32(pfeat_a), npy_as_f32(pfeat_a) + (size_t)mel_len1*MEL);
    std::vector<float> emb(npy_as_f32(emb_a), npy_as_f32(emb_a) + (size_t)emb_a.shape[0]);

    int mel_len2 = 0;
    std::vector<float> mel = cosyvoice_flow_run(m, ptok, stok, pfeat, mel_len1, emb, mel_len2);
    fprintf(stderr, "flow: T_tok=%d mel_len1=%d mel_len2=%d\n", T_ptok+T_stok, mel_len1, mel_len2);

    { std::string p = in_dir+"/flow_mel.npy"; FILE*f=fopen(p.c_str(),"rb");
      if (f){ fclose(f); npy_array r=npy_load(p); if ((int)r.shape[0]==MEL){
          compare_stats cs=compare_f32(mel.data(), npy_as_f32(r), mel.size());
          print_compare("flow_mel", cs);
          const float*e=npy_as_f32(r); double dot=0,na=0,nb=0;
          for(size_t i=0;i<mel.size();++i){const double a=mel[i],b=e[i];dot+=a*b;na+=a*a;nb+=b*b;}
          fprintf(stderr,"  flow_mel cosine=%.6f\n", dot/(std::sqrt(na)*std::sqrt(nb))); } } }

    std::vector<int64_t> shape = { MEL, mel_len2 };
    npy_save_f32(out_npy, shape, mel.data());
    fprintf(stderr, "wrote mel -> %s  [%d, %d]\n", out_npy.c_str(), MEL, mel_len2);
    return 0;
}

int main(int argc, char ** argv) {
    std::string gguf, in_dir, mode = "dit", out_npy = "flow_mel_cpp.npy";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--flow-gguf" && i+1 < argc) gguf = argv[++i];
        else if (a == "--in-dir" && i+1 < argc) in_dir = argv[++i];
        else if (a == "--mode" && i+1 < argc) mode = argv[++i];
        else if (a == "--out" && i+1 < argc) out_npy = argv[++i];
    }
    if (gguf.empty() || in_dir.empty()) { fprintf(stderr, "need --flow-gguf and --in-dir\n"); return 1; }
    model_ctx m = load_gguf(gguf);
    fprintf(stderr, "loaded %zu tensors\n", m.tensors.size());
    if (mode == "flow") return run_flow(m, in_dir, out_npy);
    return run_dit_test(m, in_dir);
}
