// Validation harness for the streaming port (PROGRESS.md B1).
//
// Reads Python's streaming dumps produced by
// `scripts/dump-streaming-reference.py`, runs the C++ pipeline once per
// chunk with the matching `finalize` flag and token prefix, and compares
// per-chunk mel output against Python's `chunk_{k}_mels_new.npy`.
//
// Phase 3 additions: carries `hift_cache_source` across chunks, applies
// `apply_trim_fade` only to chunk 0, concatenates the per-chunk wavs into
// `streamed_wav_cpp.npy`, and reports RMS error vs Python's
// `streamed_wav.npy` — HiFT sinegen uses a different RNG than Python's
// torch.randn so bit-exact parity is not expected, but the streamed RMS
// should be close to Python's streamed-vs-batch gap (≈3%).
//
// Usage:
//   ./build/test-streaming  models/chatterbox-s3gen.gguf  /tmp/streaming_ref/

#include "tts-cpp/chatterbox/s3gen_pipeline.h"
#include "npy.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <sys/stat.h>

// Minimal .npy writer (float32 1-D / 2-D only; matches the harness needs).
static void npy_write_f32(const std::string & path, const std::vector<float> & v,
                          const std::vector<int64_t> & shape) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "error: cannot write %s\n", path.c_str()); return; }
    std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) {
        hdr += std::to_string(shape[i]);
        if (shape.size() == 1 || i + 1 < shape.size()) hdr += ", ";
    }
    hdr += "), }";
    // pad so total header length (10 bytes preamble + hdr + newline) is % 64.
    size_t total = 10 + hdr.size() + 1;
    size_t pad = (64 - (total % 64)) % 64;
    hdr.append(pad, ' ');
    hdr += '\n';
    uint16_t hlen = (uint16_t)hdr.size();
    f.write("\x93NUMPY", 6);
    char ver[2] = { 1, 0 };
    f.write(ver, 2);
    f.write((const char*)&hlen, 2);
    f.write(hdr.data(), hdr.size());
    f.write((const char*)v.data(), v.size() * sizeof(float));
}

static bool path_exists(const std::string & p) {
    struct stat st; return ::stat(p.c_str(), &st) == 0;
}

struct cmp_result {
    double max_abs = 0, rms = 0, rel = 0;
    int n = 0;
};

static cmp_result cmp_f32(const float * a, const float * b, size_t n) {
    cmp_result r; r.n = (int)n;
    double ss = 0, max_ref = 0;
    for (size_t i = 0; i < n; ++i) {
        double d = std::fabs((double)a[i] - (double)b[i]);
        if (d > r.max_abs) r.max_abs = d;
        ss += d * d;
        double e = std::fabs((double)b[i]);
        if (e > max_ref) max_ref = e;
    }
    r.rms = std::sqrt(ss / n);
    r.rel = max_ref > 0 ? r.max_abs / max_ref : r.max_abs;
    return r;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s S3GEN.gguf STREAMING_REF_DIR\n", argv[0]);
        return 1;
    }
    const std::string gguf = argv[1];
    const std::string ref  = argv[2];

    npy_array all_tokens = npy_load(ref + "/speech_tokens.npy");
    const int n_speech = (int)all_tokens.n_elements();
    const int32_t * tok_ptr = (const int32_t *)all_tokens.data.data();
    fprintf(stderr, "loaded %d speech tokens\n", n_speech);

    int k = 0;
    int prev_mels_emitted = 0;
    double max_rel = 0;
    double total_rms = 0;
    std::vector<float> hift_cache_source;   // tail of prev chunk's post-SineGen source
    std::vector<float> streamed_wav_cpp;    // accumulated across chunks
    double wav_max_rel = 0, wav_total_rms = 0; int wav_chunks = 0;
    while (true) {
        char kbuf[16];
        std::snprintf(kbuf, sizeof(kbuf), "%02d", k + 1);
        const std::string tok_path = ref + "/chunk_" + kbuf + "_tokens.npy";
        const std::string mel_path = ref + "/chunk_" + kbuf + "_mels_new.npy";
        if (!path_exists(tok_path)) break;

        npy_array toks_chunk = npy_load(tok_path);
        const int32_t * cp = (const int32_t *)toks_chunk.data.data();
        const int n_cum = (int)toks_chunk.n_elements();
        const bool is_last = (n_cum == n_speech);

        fprintf(stderr, "\n=== chunk %d ===  tokens_total=%d  finalize=%s  "
                        "prev_mels_emitted=%d\n",
                k + 1, n_cum, is_last ? "true" : "false", prev_mels_emitted);

        // Sanity-check the cumulative token prefix matches the global sequence.
        for (int i = 0; i < n_cum; ++i) {
            if (cp[i] != tok_ptr[i]) {
                fprintf(stderr, "error: chunk tokens diverge from global sequence at i=%d\n", i);
                return 1;
            }
        }

        std::vector<int32_t> chunk_tokens(cp, cp + n_cum);

        const std::string dump_path = ref + "/cpp_chunk_" + kbuf + "_mels_new.npy";

        s3gen_synthesize_opts opts;
        opts.s3gen_gguf_path           = gguf;
        opts.out_wav_path              = ref + "/cpp_chunk_" + kbuf + ".wav";
        opts.ref_dir                   = ref;
        opts.seed                      = 42;
        // Streaming semantics mirrored from scripts/dump-streaming-reference.py:
        //   * the Python loop appends 3 silence tokens ONCE at the top,
        //     before chunking, so per-chunk calls don't re-append.  We match
        //     that with append_lookahead_silence=false.
        //   * `finalize` only controls the tail 6-frame trim.
        //   * `skip_mel_frames` drops the frames already emitted by earlier
        //     chunks, leaving only what's new this chunk.
        opts.append_lookahead_silence  = false;
        opts.finalize                  = is_last;
        opts.skip_mel_frames           = prev_mels_emitted;
        opts.dump_mel_path             = dump_path;

        // HiFT streaming: carry source tail across chunks, fade-in only on
        // chunk 0 (matches Python dump-streaming-reference.py).
        opts.hift_cache_source         = hift_cache_source;
        opts.apply_trim_fade           = (k == 0);
        std::vector<float> source_tail_out;
        opts.hift_source_tail_out      = &source_tail_out;
        opts.source_tail_samples       = 480;   // 1 mel hop at 24 kHz = 20 ms

        // Inject Python's exact per-chunk CFM noise so the two pipelines
        // become bit-exact comparable (bypasses the torch.randn vs
        // std::mt19937 divergence).
        // Use chunk_KK_step0_x_in.npy (the ACTUAL z passed into estimator.forward
        // at step 0 — includes the `noised_mels` overlay from flow_inference).
        // The older chunk_KK_cfm_z.npy only captured the first torch.randn_like
        // call and missed the meanflow-specific speech-region overwrite, which
        // caused post-prompt divergence.
        const std::string z_path = ref + "/chunk_" + kbuf + "_step0_x_in.npy";
        if (path_exists(z_path)) {
            npy_array z = npy_load(z_path);
            const float * zp = (const float *)z.data.data();
            opts.cfm_z0_override.assign(zp, zp + z.n_elements());
        } else {
            fprintf(stderr, "  (no cfm_z0 dump; using std::mt19937 — expect rel ≈ 0.25)\n");
        }

        int rc = s3gen_synthesize_to_wav(chunk_tokens, opts);
        if (rc != 0) {
            fprintf(stderr, "error: s3gen_synthesize_to_wav returned %d\n", rc);
            return rc;
        }

        // Compare mel against Python.
        npy_array py_mel  = npy_load(mel_path);
        npy_array cpp_mel = npy_load(dump_path);
        const float * py  = (const float *)py_mel.data.data();
        const float * cpp = (const float *)cpp_mel.data.data();
        const size_t n    = std::min(py_mel.n_elements(), cpp_mel.n_elements());
        const size_t py_rows  = py_mel.shape[0];
        const size_t cpp_rows = cpp_mel.shape[0];
        fprintf(stderr, "  mel shapes:   py=(%zu, %lld)  cpp=(%zu, %lld)\n",
                py_rows,  (long long)py_mel.shape[1],
                cpp_rows, (long long)cpp_mel.shape[1]);
        if (py_rows != cpp_rows) {
            fprintf(stderr, "  *** SHAPE MISMATCH — C++ emitted %zu frames, Python %zu\n",
                    cpp_rows, py_rows);
        }
        auto s = cmp_f32(cpp, py, n);
        fprintf(stderr, "  mel diff:     max_abs=%.4e  rms=%.4e  rel=%.4e\n",
                s.max_abs, s.rms, s.rel);
        if (s.rel > max_rel) max_rel = s.rel;
        total_rms += s.rms;
        prev_mels_emitted += (int)cpp_rows;

        // Carry cache_source over to the next chunk.
        hift_cache_source = std::move(source_tail_out);

        // Read the per-chunk C++ wav (just written to disk) and splice into
        // the running streamed wav.  Compare to Python's chunk_KK_wav.npy:
        // C++ SineGen RNG differs from torch.randn so bit-exact parity is
        // not achievable here — we just report RMS as an informational
        // sanity signal.
        const std::string cpp_wav_path = opts.out_wav_path;
        // Minimal WAV reader (mono 16-bit PCM from write_wav()).
        std::vector<float> cpp_wav;
        {
            std::ifstream wf(cpp_wav_path, std::ios::binary);
            if (wf) {
                char hdr[44]; wf.read(hdr, 44);
                // Read 16-bit PCM samples
                std::vector<int16_t> pcm;
                int16_t s16;
                while (wf.read((char*)&s16, 2)) pcm.push_back(s16);
                cpp_wav.resize(pcm.size());
                for (size_t i = 0; i < pcm.size(); ++i) cpp_wav[i] = (float)pcm[i] / 32768.0f;
            }
        }
        const std::string py_wav_path = ref + "/chunk_" + kbuf + "_wav.npy";
        if (path_exists(py_wav_path) && !cpp_wav.empty()) {
            npy_array py_wav_arr = npy_load(py_wav_path);
            const float * pywav = (const float *)py_wav_arr.data.data();
            size_t wn = std::min((size_t)py_wav_arr.n_elements(), cpp_wav.size());
            auto ws = cmp_f32(cpp_wav.data(), pywav, wn);
            fprintf(stderr, "  wav  diff:    max_abs=%.4e  rms=%.4e  rel=%.4e  (informational, RNG mismatch expected)\n",
                    ws.max_abs, ws.rms, ws.rel);
            if (ws.rel > wav_max_rel) wav_max_rel = ws.rel;
            wav_total_rms += ws.rms;
            wav_chunks += 1;
        }
        streamed_wav_cpp.insert(streamed_wav_cpp.end(), cpp_wav.begin(), cpp_wav.end());
        k += 1;
    }

    fprintf(stderr, "\nresult: %d chunks verified, worst mel rel=%.4e, mean mel rms=%.4e\n",
            k, max_rel, total_rms / std::max(k, 1));
    if (wav_chunks > 0) {
        fprintf(stderr, "        per-chunk wav: worst rel=%.4e, mean rms=%.4e\n",
                wav_max_rel, wav_total_rms / wav_chunks);
    }

    // Save streamed_wav_cpp for offline comparison vs Python's streamed_wav.
    if (!streamed_wav_cpp.empty()) {
        npy_write_f32(ref + "/streamed_wav_cpp.npy",
                      streamed_wav_cpp, {(int64_t)streamed_wav_cpp.size()});
        fprintf(stderr, "        streamed wav (%zu samples) → %s/streamed_wav_cpp.npy\n",
                streamed_wav_cpp.size(), ref.c_str());
        const std::string py_streamed = ref + "/streamed_wav.npy";
        if (path_exists(py_streamed)) {
            npy_array py_sw = npy_load(py_streamed);
            const float * py = (const float *)py_sw.data.data();
            size_t n = std::min((size_t)py_sw.n_elements(), streamed_wav_cpp.size());
            auto sw = cmp_f32(streamed_wav_cpp.data(), py, n);
            fprintf(stderr, "        streamed vs Python: rms=%.4e max_abs=%.4e rel=%.4e\n",
                    sw.rms, sw.max_abs, sw.rel);
        }
    }
    return 0;
}
