// End-to-end greedy parity: text -> tokenizer -> T5 -> decoder loop (delay
// pattern + logits processor) -> un-delay -> DAC -> waveform, compared against
// the HF fixtures. Also smoke-tests the public Engine on the same inputs.
//
// Escape hatch (documented in the plan): the greedy token trace must match
// exactly UNLESS the first divergence happens at a step whose top-2 logit
// margin is < 1e-3 (a numerical near-tie); then the prefix is compared and
// the trace check ends there, but the DAC/wav comparison is skipped.

#include "parler/internal.h"
#include "parler/delay.h"
#include "parler/tokenizer.h"
#include "parler/bpe_tokenizer.h"
#include "tts-cpp/parler/engine.h"
#include "json.hpp"
#include "npy.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace tts_cpp::parler::detail;

// case0's texts come from the fixture dir's meta.json (written by
// dump-reference.py), so the binary serves every model's fixtures.
static bool load_case0_texts(const std::string & ref_dir,
                             std::string & description, std::string & prompt) {
    std::ifstream ifs(ref_dir + "/meta.json");
    if (!ifs) {
        fprintf(stderr, "cannot open %s/meta.json\n", ref_dir.c_str());
        return false;
    }
    nlohmann::json j = nlohmann::json::parse(ifs, nullptr, /*allow_exceptions*/ false);
    if (j.is_discarded() || !j.contains("cases") || !j["cases"].is_array() ||
        j["cases"].empty()) {
        fprintf(stderr, "malformed meta.json in %s\n", ref_dir.c_str());
        return false;
    }
    description = j["cases"][0]["description"].get<std::string>();
    prompt      = j["cases"][0]["prompt"].get<std::string>();
    return true;
}

static std::vector<int32_t> load_ids(const std::string & path) {
    npy_array a = npy_load(path);
    const int64_t * p = reinterpret_cast<const int64_t *>(a.data.data());
    std::vector<int32_t> v(a.n_elements());
    for (size_t i = 0; i < v.size(); ++i) v[i] = (int32_t) p[i];
    return v;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL.gguf REF_DIR [OUT_WAV_DIR]\n", argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    const std::string ref_dir = argv[2];
    const std::string out_dir = argc > 3 ? argv[3] : "";
    const int n_threads = 4;

    std::string kDescription, kPrompt;
    if (!load_case0_texts(ref_dir, kDescription, kPrompt)) {
        return 1;
    }

    parler_model model;
    std::string err;
    const int ngl = std::getenv("PARLER_TEST_GPU") ? 99 : 0;
    if (!parler_load_gguf(model_path, model, ngl, &err)) {
        fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }
    const parler_hparams & hp = model.hparams;
    int rc = 1;
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

    do {
        // tokenizer parity on the case texts
        parler_tokenizer tok;
        if (!tok.load(model.tok_pieces, model.tok_scores, model.tok_charsmap,
                      model.tok_unk_id, model.tok_eos_id, model.tok_add_eos)) {
            fprintf(stderr, "tokenizer load failed\n");
            break;
        }
        parler_bpe_tokenizer ptok;
        if (model.has_prompt_tok &&
            !ptok.load(model.ptok_pieces, model.ptok_merges, model.ptok_unk_id,
                       model.ptok_bos_id, model.ptok_add_bos)) {
            fprintf(stderr, "prompt tokenizer load failed\n");
            break;
        }
        const std::vector<int32_t> desc_ids = tok.encode(kDescription);
        const std::vector<int32_t> prompt_ids = model.has_prompt_tok
            ? ptok.encode(kPrompt) : tok.encode(kPrompt);
        if (desc_ids != load_ids(ref_dir + "/case0_desc_ids.npy") ||
            prompt_ids != load_ids(ref_dir + "/case0_prompt_ids.npy")) {
            fprintf(stderr, "FAIL: tokenized ids differ from fixtures\n");
            break;
        }

        if (!parler_encode_description(model, desc_ids, n_threads, nullptr)) break;

        npy_array greedy = npy_load(ref_dir + "/case0_greedy_delayed.npy"); // [9, L]
        const int64_t * gref = reinterpret_cast<const int64_t *>(greedy.data.data());
        const int n_cb = (int) greedy.shape[0];
        const int L = (int) greedy.shape[1];

        delay_config dcfg;
        dcfg.n_codebooks = n_cb;
        dcfg.bos_id = hp.bos_id;
        dcfg.eos_id = hp.eos_id;
        dcfg.pad_id = hp.pad_id;
        dcfg.max_length = L; // fixture was generated with this cap
        dcfg.min_new_tokens = hp.gen_min_new_tokens;
        delay_state st(dcfg);

        std::vector<float> logits;
        int n_past = 0;
        if (!parler_dec_prefill(model, prompt_ids, st.input_frame(), allocr, n_threads,
                                logits, n_past)) break;

        bool trace_exact = true;
        int steps_done = 0;
        while (true) {
            st.process_logits(logits.data(), hp.dec_vocab);
            // greedy argmax + top-2 margin per codebook
            std::vector<int32_t> frame(n_cb);
            std::vector<float> margin(n_cb, 1e30f);
            for (int k = 0; k < n_cb; ++k) {
                const float * row = logits.data() + (size_t) k * hp.dec_vocab;
                int best = 0, second = -1;
                for (int v = 1; v < hp.dec_vocab; ++v) {
                    if (row[v] > row[best]) { second = best; best = v; }
                    else if (second < 0 || row[v] > row[second]) second = v;
                }
                frame[k] = best;
                if (second >= 0) margin[k] = row[best] - row[second];
            }
            // compare against the recorded column steps_done+1; the escape
            // margin considers only the mismatching codebooks
            const int col = steps_done + 1;
            bool match = true;
            float min_margin = 1e30f;
            for (int k = 0; k < n_cb; ++k) {
                if ((int64_t) frame[k] != gref[(size_t) k * L + col]) {
                    match = false;
                    min_margin = std::min(min_margin, margin[k]);
                }
            }
            if (!match) {
                if (min_margin < 1e-3f) {
                    fprintf(stderr, "note: trace diverged at step %d with top-2 margin "
                            "%.3e < 1e-3 (numerical near-tie); prefix of %d steps matched\n",
                            col, min_margin, steps_done);
                    trace_exact = false;
                    break;
                }
                fprintf(stderr, "FAIL: greedy trace diverged at step %d (margin %.3e)\n",
                        col, min_margin);
                for (int k = 0; k < n_cb; ++k) {
                    fprintf(stderr, "  cb%d: got %d ref %lld\n", k, frame[k],
                            (long long) gref[(size_t) k * L + col]);
                }
                st.append(frame); // keep state consistent for the report
                goto done;
            }
            st.append(frame);
            steps_done++;
            if (st.finished()) break;
            if (!parler_dec_step(model, st.input_frame(), n_past, allocr, n_threads, logits)) {
                goto done;
            }
            n_past++;
        }
        fprintf(stderr, "  greedy trace: %d/%d steps matched%s\n", steps_done, L - 1,
                trace_exact ? "" : " (prefix, near-tie escape)");

        if (trace_exact) {
            // codes parity
            int n_frames = 0;
            std::vector<int32_t> codes = st.undelay(hp.dac_codebook_size, &n_frames);
            npy_array cref = npy_load(ref_dir + "/case0_codes.npy"); // [1, 9, F]
            const int64_t * cr = reinterpret_cast<const int64_t *>(cref.data.data());
            if ((int) cref.shape[2] != n_frames) {
                fprintf(stderr, "FAIL: frame count %d vs ref %lld\n",
                        n_frames, (long long) cref.shape[2]);
                break;
            }
            bool codes_ok = true;
            for (size_t i = 0; i < codes.size(); ++i) {
                if ((int64_t) codes[i] != cr[i]) { codes_ok = false; break; }
            }
            if (!codes_ok) {
                fprintf(stderr, "FAIL: un-delayed codes differ\n");
                break;
            }

            // full waveform parity
            std::vector<float> pcm;
            if (!parler_dac_decode(model, codes.data(), n_frames, n_threads, pcm)) break;
            npy_array wref = npy_load(ref_dir + "/case0_wav_greedy.npy");
            if (wref.n_elements() != pcm.size()) {
                fprintf(stderr, "FAIL: wav length %zu vs ref %zu\n",
                        pcm.size(), wref.n_elements());
                break;
            }
            const float * wr = npy_as_f32(wref);
            double num = 0.0, den = 0.0;
            for (size_t i = 0; i < pcm.size(); ++i) {
                num += (double) wr[i] * wr[i];
                const double d = (double) pcm[i] - wr[i];
                den += d * d;
            }
            const double snr = 10.0 * std::log10(num / (den > 0 ? den : 1e-30));
            fprintf(stderr, "  e2e wav SNR = %.2f dB (%d frames)\n", snr, n_frames);
            if (snr < 60.0) {
                fprintf(stderr, "FAIL: wav SNR below 60 dB\n");
                break;
            }
        }

        // public Engine smoke on the same inputs (short)
        {
            tts_cpp::parler::EngineOptions opts;
            opts.model_gguf_path = model_path;
            opts.max_frames = 60;
            opts.n_threads = n_threads;
            tts_cpp::parler::Engine engine(opts);
            tts_cpp::parler::SynthesisResult res = engine.synthesize(kPrompt, kDescription);
            if (res.pcm.empty() || res.sample_rate != hp.dac_sample_rate) {
                fprintf(stderr, "FAIL: engine produced empty/invalid result\n");
                break;
            }
            for (float v : res.pcm) {
                if (!std::isfinite(v)) {
                    fprintf(stderr, "FAIL: engine produced non-finite samples\n");
                    goto done;
                }
            }
            fprintf(stderr, "  engine smoke: %.2f s @ %d Hz\n", res.duration_s, res.sample_rate);
            if (!out_dir.empty()) {
                // raw f32 dump; by-ear wavs come from parler-cli
                npy_save_f32(out_dir + "/e2e_engine_smoke.npy",
                             { (int64_t) res.pcm.size() }, res.pcm.data());
            }
        }

        rc = 0;
    } while (false);
done:
    ggml_gallocr_free(allocr);
    parler_free_model(model);
    if (rc == 0) fprintf(stderr, "parler e2e: PASS\n");
    return rc;
}
