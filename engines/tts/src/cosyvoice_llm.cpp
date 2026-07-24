// cosyvoice-llm: CosyVoice3 speech LM (stage 5) parity CLI.
//
// The Qwen2 graph + autoregressive decode now live in cosyvoice_pipeline.cpp
// (shared with the Engine); this CLI is the parity / bring-up harness:
//   --mode prefill : feed lm_input.npy [L,896], run the 24-layer prefill,
//                    project the last position, compare to logits.npy[0].
//   --mode text    : tokenize typed text (+prompt) and prompt_stok.npy, run the
//                    shared decode, save the generated speech tokens.
//
// Weights: scripts/convert-cosyvoice3-llm-to-gguf.py (lm/* names).
//
// Usage: cosyvoice-llm --llm-gguf LLM.gguf --in-dir llm-ref --mode prefill

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "npy.h"

#include "cosyvoice_pipeline.h"
#include "qwen_tokenizer.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static model_ctx load_gguf(const std::string & path) { return cosyvoice_load_gguf(path); }

// text mode: build lm_input from typed text (Qwen2 BPE) + prompt tokens, run
// the shared decode, save the generated speech tokens.
static int run_text(model_ctx & m, const qwen_hp & hp, const std::string & tts_text,
                    const std::string & prompt_text, const std::string & prompt_stok_npy,
                    const std::string & vocab, const std::string & merges,
                    const std::string & out_npy, int max_steps, bool greedy, int seed){
    QwenTokenizer tok;
    if(!tok.load(vocab, merges)){ fprintf(stderr,"tokenizer load failed\n"); return 1; }
    std::vector<int> tts_ids = tok.encode(tts_text);
    std::vector<int> text_ids = tok.encode(prompt_text);
    text_ids.insert(text_ids.end(), tts_ids.begin(), tts_ids.end());
    fprintf(stderr,"text_ids=%zu (prompt_text+tts_text)\n", text_ids.size());
    std::vector<int> pstok;
    { npy_array a=npy_load(prompt_stok_npy); const int32_t* p=npy_as_i32(a);
      for(int i=0;i<(int)a.shape[0];++i) pstok.push_back(p[i]); }
    int min_len = 2 * (int)tts_ids.size();
    std::vector<int> tokens = cosyvoice_llm_generate(m, hp, text_ids, pstok, max_steps, greedy, seed, min_len);
    fprintf(stderr,"generated %d speech tokens. first20:", (int)tokens.size());
    for(int i=0;i<(int)tokens.size()&&i<20;++i) fprintf(stderr," %d", tokens[i]); fprintf(stderr,"\n");
    std::vector<int64_t> sh={(int64_t)tokens.size()};
    npy_save_i32(out_npy, sh, tokens.data());
    fprintf(stderr,"wrote %s  (%d tokens)\n", out_npy.c_str(), (int)tokens.size());
    return 0;
}

int main(int argc, char** argv){
    std::string gguf, in_dir, mode="prefill", out="speech_tokens_cpp.npy";
    std::string text, prompt_text="You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。";
    std::string prompt_stok, vocab, merges;
    int max_steps=600, seed=1234; bool greedy=false;
    for(int i=1;i<argc;++i){ std::string a=argv[i];
        if(a=="--llm-gguf"&&i+1<argc) gguf=argv[++i];
        else if(a=="--in-dir"&&i+1<argc) in_dir=argv[++i];
        else if(a=="--mode"&&i+1<argc) mode=argv[++i];
        else if(a=="--out"&&i+1<argc) out=argv[++i];
        else if(a=="--text"&&i+1<argc) text=argv[++i];
        else if(a=="--prompt-text"&&i+1<argc) prompt_text=argv[++i];
        else if(a=="--prompt-stok"&&i+1<argc) prompt_stok=argv[++i];
        else if(a=="--vocab"&&i+1<argc) vocab=argv[++i];
        else if(a=="--merges"&&i+1<argc) merges=argv[++i];
        else if(a=="--max-steps"&&i+1<argc) max_steps=atoi(argv[++i]);
        else if(a=="--seed"&&i+1<argc) seed=atoi(argv[++i]);
        else if(a=="--greedy") greedy=true; }
    if(gguf.empty()){ fprintf(stderr,"need --llm-gguf\n"); return 1; }
    model_ctx m = load_gguf(gguf);
    fprintf(stderr,"loaded %zu tensors\n", m.tensors.size());
    qwen_hp hp;
    if(mode=="text") return run_text(m, hp, text, prompt_text, prompt_stok, vocab, merges, out, max_steps, greedy, seed);

    // --mode prefill: logits parity gate.
    npy_array lmin = npy_load(in_dir+"/lm_input.npy");   // [L, 896]
    npy_array logref = npy_load(in_dir+"/logits.npy");   // [S, 6761]
    int L=(int)lmin.shape[0], D=(int)lmin.shape[1];
    int VS=(int)logref.shape[1];
    fprintf(stderr,"L=%d D=%d vocab=%d\n", L, D, VS);

    size_t bs=(size_t)3*1024*1024*1024;
    std::vector<uint8_t> buf(bs);
    ggml_init_params gp={bs, buf.data(), true};
    ggml_context* c=ggml_init(gp);
    ggml_cgraph* gf=ggml_new_graph_custom(c, 262144, false);

    ggml_tensor* x=ggml_new_tensor_2d(c, GGML_TYPE_F32, D, L); ggml_set_name(x,"x"); ggml_set_input(x);
    ggml_tensor* pos=ggml_new_tensor_1d(c, GGML_TYPE_I32, L); ggml_set_name(pos,"pos"); ggml_set_input(pos);
    ggml_tensor* mask=ggml_new_tensor_2d(c, GGML_TYPE_F32, L, L); ggml_set_name(mask,"mask"); ggml_set_input(mask);

    ggml_tensor* logits = build_qwen(c, m, hp, x, pos, mask, L);
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);
    ggml_tensor* hid = ggml_graph_get_tensor(gf, "hidden");
    if (hid) ggml_build_forward_expand(gf, hid);
    ggml_gallocr_t al=ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_reserve(al, gf); ggml_gallocr_alloc_graph(al, gf);

    ggml_backend_tensor_set(x, npy_as_f32(lmin), 0, (size_t)L*D*4);
    { std::vector<int32_t> pv(L); for(int i=0;i<L;++i) pv[i]=i; ggml_backend_tensor_set(pos, pv.data(),0,pv.size()*4); }
    { std::vector<float> mk((size_t)L*L, 0.f);
      for(int q=0;q<L;++q) for(int k=0;k<L;++k) mk[(size_t)q*L+k] = (k<=q)?0.f:-INFINITY;
      ggml_backend_tensor_set(mask, mk.data(),0,mk.size()*4); }

    ggml_backend_graph_compute(m.backend, gf);

    { std::string p=in_dir+"/hidden0.npy"; FILE*f=fopen(p.c_str(),"rb");
      if(f){ fclose(f); npy_array hr=npy_load(p); ggml_tensor* hid2=ggml_graph_get_tensor(gf,"hidden");
        std::vector<float> hh((size_t)D*L); ggml_backend_tensor_get(hid2, hh.data(),0,hh.size()*4);
        compare_stats cs=compare_f32(hh.data(), npy_as_f32(hr), (size_t)D*L);
        print_compare("hidden", cs);
        const float*e=npy_as_f32(hr); double dot=0,na=0,nb=0;
        for(size_t i=0;i<hh.size();++i){const double a=hh[i],b=e[i];dot+=a*b;na+=a*a;nb+=b*b;}
        fprintf(stderr,"  hidden cosine=%.6f\n", dot/(std::sqrt(na)*std::sqrt(nb))); } }

    std::vector<float> got(VS);
    ggml_backend_tensor_get(logits, got.data(), (size_t)(L-1)*VS*4, (size_t)VS*4);
    std::vector<float> lg(VS); double mx=-1e30; for(float f:got) mx=std::max(mx,(double)f);
    double se=0; for(int i=0;i<VS;++i) se+=std::exp(got[i]-mx);
    double lse=mx+std::log(se);
    for(int i=0;i<VS;++i) lg[i]=(float)(got[i]-lse);

    const float* ref=npy_as_f32(logref);
    compare_stats s=compare_f32(lg.data(), ref, VS);
    print_compare("logits[0] (log-softmax)", s);
    double dot=0,na=0,nb=0; for(int i=0;i<VS;++i){const double a=lg[i],b=ref[i];dot+=a*b;na+=a*a;nb+=b*b;}
    fprintf(stderr,"  cosine=%.6f\n", dot/(std::sqrt(na)*std::sqrt(nb)));
    int am=0,ar=0; for(int i=1;i<VS;++i){ if(lg[i]>lg[am])am=i; if(ref[i]>ref[ar])ar=i; }
    fprintf(stderr,"  argmax mine=%d ref=%d\n", am, ar);

    ggml_gallocr_free(al); ggml_free(c);
    return 0;
}
