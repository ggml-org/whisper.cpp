#include "parler/internal.h"
#include <cstdlib>
#include "npy.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

using namespace tts_cpp::parler::detail;

// micro fixture: torch ConvTranspose1d(4, 3, k=16, s=8, p=4) — verifies the
// unpadded conv_transpose + symmetric-trim helper before the full stack.
static bool test_convt_micro(const parler_model & model, const std::string & ref_dir) {
    npy_array w_npy  = npy_load(ref_dir + "/convt_unit_w.npy");   // [4, 3, 16]
    npy_array b_npy  = npy_load(ref_dir + "/convt_unit_b.npy");   // [3]
    npy_array in_npy = npy_load(ref_dir + "/convt_unit_in.npy");  // [1, 4, 11]
    npy_array out_npy = npy_load(ref_dir + "/convt_unit_out.npy");// [1, 3, 88]

    const size_t ctx_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false);
    ggml_init_params ip = { ctx_size, nullptr, /*no_alloc=*/ true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);

    ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 3, 4);
    ggml_set_name(w, "w"); ggml_set_input(w);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
    ggml_set_name(b, "b"); ggml_set_input(b);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 11, 4);
    ggml_set_name(x, "x"); ggml_set_input(x);

    const int stride = 8, padding = 4;
    ggml_tensor * out = ggml_conv_transpose_1d(ctx, w, x, stride, 0, 1);
    ggml_tensor * v = ggml_view_3d(ctx, out, out->ne[0] - 2 * padding, out->ne[1], out->ne[2],
                                   out->nb[1], out->nb[2], (size_t) padding * out->nb[0]);
    out = ggml_cont(ctx, v);
    out = ggml_add(ctx, out, ggml_reshape_2d(ctx, b, 1, 3));
    ggml_set_name(out, "out"); ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    bool use_sched = false;
    bool ok = false;
    do {
        if (!parler_graph_prepare(model, gf, allocr, use_sched, __func__)) break;
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "w"), npy_as_f32(w_npy), 0, w_npy.data.size());
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "b"), npy_as_f32(b_npy), 0, b_npy.data.size());
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x"), npy_as_f32(in_npy), 0, in_npy.data.size());
        if (!parler_graph_compute(model, gf, use_sched, 4, __func__)) break;

        ggml_tensor * out_t = ggml_graph_get_tensor(gf, "out");
        if ((size_t) ggml_nelements(out_t) != out_npy.n_elements()) {
            fprintf(stderr, "convt micro: length mismatch got %lld ref %zu\n",
                    (long long) ggml_nelements(out_t), out_npy.n_elements());
            break;
        }
        std::vector<float> got(ggml_nelements(out_t));
        ggml_backend_tensor_get(out_t, got.data(), 0, ggml_nbytes(out_t));
        compare_stats s = compare_f32(got.data(), npy_as_f32(out_npy), got.size());
        print_compare("convt_micro", s);
        ok = s.max_abs_err <= 1e-5;
        if (!ok) fprintf(stderr, "convt micro: FAIL (max_abs %.3e > 1e-5)\n", s.max_abs_err);
    } while (false);

    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    return ok;
}

static bool load_codes_i32(const std::string & path, std::vector<int32_t> & codes, int & n_frames) {
    npy_array a = npy_load(path);
    if (a.dtype != "<i8" || a.shape.size() != 3 || a.shape[0] != 1) {
        fprintf(stderr, "unexpected codes npy shape/dtype in %s\n", path.c_str());
        return false;
    }
    n_frames = (int) a.shape[2];
    const int64_t * c64 = reinterpret_cast<const int64_t *>(a.data.data());
    codes.resize(a.n_elements());
    for (size_t i = 0; i < codes.size(); ++i) codes[i] = (int32_t) c64[i];
    return true;
}

// returns SNR in dB; also prints compare stats and reports peak error
static double wav_snr(const char * name, const std::vector<float> & got, const float * ref,
                      size_t n, double & max_abs_out) {
    compare_stats s = compare_f32(got.data(), ref, n);
    print_compare(name, s);
    max_abs_out = s.max_abs_err;
    double sig = 0, noise = 0;
    for (size_t i = 0; i < n; ++i) {
        sig += (double) ref[i] * ref[i];
        double d = (double) got[i] - ref[i];
        noise += d * d;
    }
    // A NaN/Inf sample makes noise non-finite; surface it as NaN so the caller rejects it
    // (without this, `noise > 0` is false and snr would take the INFINITY "perfect" branch).
    const double snr = !std::isfinite(noise) ? NAN
                     : (noise > 0 ? 10.0 * log10(sig / noise) : INFINITY);
    fprintf(stderr, "  [%s] SNR = %.2f dB  max_abs = %.3e\n", name, snr, s.max_abs_err);
    return snr;
}

static bool test_wav_case(const parler_model & model, const std::string & ref_dir,
                          const char * codes_name, const char * wav_name,
                          std::vector<float> * latent_out) {
    std::vector<int32_t> codes;
    int n_frames = 0;
    if (!load_codes_i32(ref_dir + "/" + codes_name, codes, n_frames)) return false;

    std::vector<float> pcm;
    if (!parler_dac_decode(model, codes.data(), n_frames, 4, pcm, latent_out)) {
        fprintf(stderr, "%s: parler_dac_decode failed\n", codes_name);
        return false;
    }

    npy_array wav_ref = npy_load(ref_dir + "/" + wav_name);
    if (pcm.size() != wav_ref.n_elements()) {
        fprintf(stderr, "%s: wav length mismatch got %zu ref %zu\n",
                codes_name, pcm.size(), wav_ref.n_elements());
        return false;
    }
    double wav_max_abs = 0;
    const double snr = wav_snr(wav_name, pcm, npy_as_f32(wav_ref), pcm.size(), wav_max_abs);
    if (std::isnan(snr)) {
        fprintf(stderr, "%s: FAIL non-finite audio\n", wav_name);
        return false;
    }
    // GPU FP32 reorder diverges from CPU; accept high whole-signal SNR OR inaudible peak error
    // (quiet clips show low SNR at negligible abs error). CPU keeps the strict bit-parity bar.
    const bool ok = model.on_gpu ? (snr >= 50.0 || wav_max_abs <= 5e-4) : (snr >= 60.0);
    if (!ok) {
        fprintf(stderr, "%s: FAIL (SNR %.2f dB, max_abs %.3e)\n", wav_name, snr, wav_max_abs);
        return false;
    }
    return true;
}

// The DAC is decoded in bounded windows (PARLER_DAC_WINDOW_FRAMES) padded with
// PARLER_DAC_RF_FRAMES of real context. Pins the invariant that makes that legal:
// a partial-range decode is BIT-IDENTICAL to slicing a full decode, for ranges
// that straddle window boundaries, sit at the sequence ends, and are empty.
// This is what streaming relies on, and it is what would silently regress if the
// context/keep arithmetic were ever wrong.
static bool test_range_equivalence(const parler_model & model, const std::string & ref_dir,
                                   const char * codes_name) {
    std::vector<int32_t> codes;
    int n_frames = 0;
    if (!load_codes_i32(ref_dir + "/" + codes_name, codes, n_frames)) return false;
    fprintf(stderr, "  [range-equivalence] %s: %d frames\n", codes_name, n_frames);

    const int hop = model.hparams.dac_hop;
    const int lat = model.hparams.dac_latent;
    std::vector<float> full, full_lat;
    if (!parler_dac_decode(model, codes.data(), n_frames, 4, full, &full_lat)) {
        fprintf(stderr, "range-equivalence: full decode failed\n");
        return false;
    }
    if (full.size() != (size_t) n_frames * hop) {
        fprintf(stderr, "range-equivalence: full length %zu != %d frames * %d hop\n",
                full.size(), n_frames, hop);
        return false;
    }

    const int W = PARLER_DAC_WINDOW_FRAMES;
    const int R = parler_dac_rf_frames(model);
    const std::vector<std::pair<int, int>> ranges = {
        {0, n_frames},                                          // whole sequence
        {0, 1},                                                 // first frame only
        {n_frames - 1, n_frames},                               // last frame only
        {5, 5},                                                 // empty
        {std::min(W - 1, n_frames), std::min(W + 1, n_frames)}, // straddles a window edge
        {std::min(W, n_frames), n_frames},                      // starts on a window edge
        {std::min(R / 2, n_frames), std::min(2 * W + 3, n_frames)}, // spans >1 window
    };

    bool ok = true;
    for (const auto & r : ranges) {
        const int a = r.first, b = r.second;
        if (a < 0 || b > n_frames || a > b) continue;
        std::vector<float> part, part_lat;
        if (!parler_dac_decode(model, codes.data(), n_frames, 4, part, &part_lat, a, b)) {
            fprintf(stderr, "range-equivalence: decode [%d, %d) failed\n", a, b);
            ok = false;
            continue;
        }
        const size_t want = (size_t) (b - a) * hop;
        if (part.size() != want) {
            fprintf(stderr, "range-equivalence: [%d, %d) length %zu, want %zu\n",
                    a, b, part.size(), want);
            ok = false;
            continue;
        }
        if (want != 0 &&
            memcmp(part.data(), full.data() + (size_t) a * hop, want * sizeof(float)) != 0) {
            fprintf(stderr, "range-equivalence: [%d, %d) NOT bit-identical to the full decode\n",
                    a, b);
            ok = false;
            continue;
        }
        // same for the latent, whose per-window offset uses dac_latent rather than dac_hop
        const size_t want_lat = (size_t) (b - a) * lat;
        if (part_lat.size() != want_lat) {
            fprintf(stderr, "range-equivalence: [%d, %d) latent length %zu, want %zu\n",
                    a, b, part_lat.size(), want_lat);
            ok = false;
            continue;
        }
        if (want_lat != 0 &&
            memcmp(part_lat.data(), full_lat.data() + (size_t) a * lat,
                   want_lat * sizeof(float)) != 0) {
            fprintf(stderr, "range-equivalence: [%d, %d) latent NOT bit-identical\n", a, b);
            ok = false;
            continue;
        }
        fprintf(stderr, "  [range %d, %d) bit-identical (%zu samples, %zu latent)\n",
                a, b, want, want_lat);
    }
    return ok;
}

// The point of the windowed decode: peak compute memory must not grow with output
// length. A revert to a whole-sequence graph stays bit-exact and would pass every
// other test here, so this is the only guard against it.
static bool test_bounded_memory(const parler_model & model) {
    const int lo = 512, hi = 8192;   // both well past one window + context
    const size_t s_lo = parler_dac_compute_buffer_size(model, lo);
    const size_t s_hi = parler_dac_compute_buffer_size(model, hi);
    fprintf(stderr, "  [bounded-memory] %d frames = %.2f MiB, %d frames = %.2f MiB\n",
            lo, s_lo / 1024.0 / 1024.0, hi, s_hi / 1024.0 / 1024.0);
    if (s_lo == 0 || s_hi == 0) {
        fprintf(stderr, "bounded-memory: FAIL (could not measure)\n");
        return false;
    }
    if (s_hi != s_lo) {
        fprintf(stderr, "bounded-memory: FAIL DAC buffer grows with output length "
                        "(%zu -> %zu bytes for %dx the frames)\n", s_lo, s_hi, hi / lo);
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL.gguf REF_DIR\n", argv[0]);
        return 2;
    }
    const std::string ref_dir = argv[2];

    parler_model model;
    std::string error;
    const int ngl = std::getenv("PARLER_TEST_GPU") ? 99 : 0;
    if (!parler_load_gguf(argv[1], model, ngl, &error)) {
        fprintf(stderr, "parler_load_gguf failed: %s\n", error.c_str());
        return 1;
    }

    int rc = 0;
    try {
        if (!test_convt_micro(model, ref_dir)) rc = 1;

        std::vector<float> latent;
        if (rc == 0 && !test_wav_case(model, ref_dir, "dacrand_codes.npy", "dacrand_wav.npy", &latent)) rc = 1;

        if (rc == 0) {
            // latent parity: ref npy [1, latent_dim, T] (element (c,t) at c*T + t),
            // ours is ggml [latent_dim, T] (element (c,t) at t*latent_dim + c)
            npy_array lat_ref = npy_load(ref_dir + "/dacrand_latent.npy");
            const int64_t C = lat_ref.shape[1], T = lat_ref.shape[2];
            if ((size_t) (C * T) != latent.size()) {
                fprintf(stderr, "latent size mismatch got %zu ref %lld\n",
                        latent.size(), (long long) (C * T));
                rc = 1;
            } else {
                const float * lr = npy_as_f32(lat_ref);
                double linf = 0;
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t t = 0; t < T; ++t) {
                        double d = std::fabs((double) latent[t * C + c] - lr[c * T + t]);
                        if (d > linf) linf = d;
                    }
                }
                fprintf(stderr, "  [dacrand_latent] L_inf = %.3e\n", linf);
                if (linf > 1e-4) {
                    fprintf(stderr, "latent parity: FAIL (L_inf %.3e > 1e-4)\n", linf);
                    rc = 1;
                }
            }
        }

        if (rc == 0 && !test_wav_case(model, ref_dir, "case0_codes.npy", "case0_wav_greedy.npy", nullptr)) rc = 1;

        // short set exercises n_frames < one window; the long one straddles window edges
        if (rc == 0 && !test_range_equivalence(model, ref_dir, "dacrand_codes.npy")) rc = 1;
        if (rc == 0 && !test_range_equivalence(model, ref_dir, "case0_codes.npy")) rc = 1;

        if (rc == 0) {
            fprintf(stderr, "  [receptive-field] %d frames per side\n", parler_dac_rf_frames(model));
        }
        if (rc == 0 && !test_bounded_memory(model)) rc = 1;
    } catch (const std::exception & e) {
        fprintf(stderr, "test failed: %s\n", e.what());
        rc = 1;
    }

    if (rc == 0) fprintf(stderr, "parler dac: PASS\n");
    parler_free_model(model);
    return rc;
}
