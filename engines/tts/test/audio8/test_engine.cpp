// Audio8 end-to-end parity through the public engine API.
//
// The stage tests pin the language model and the codec separately against
// their own fixtures. This one drives the same path a caller drives -- text in,
// waveform out -- so it also covers what only shows up when the stages are
// joined: the ChatML prompt the tokenizer builds, the frame budget, the
// codebook-major transposition between the two halves, and, on the cloning
// side, the engine encoding a WAV into the speaker history by itself. The last
// case covers the other side of that joining: a set of GGUFs that does not
// agree on the shape of a frame has to be refused rather than indexed.
//
// Greedy decoding throughout, because that is the only trajectory the fixtures
// can pin; the sampler has its own test.

#include "tts-cpp/audio8/engine.h"

#include "ggml.h"
#include "gguf.h"
#include "json.hpp"
#include "npy.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

// The waveform is bounded by a tanh and the codec test already holds it to a
// few times 1e-6; the extra room is for the language model's own reassociation
// reaching the codec through identical codes.
constexpr double WAVEFORM_TOLERANCE = 5e-5;

struct fixture {
    std::string dir;

    npy_array load(const std::string & name) const {
        return npy_load(dir + "/" + name + ".npy");
    }

    nlohmann::json meta() const {
        std::ifstream file(dir + "/meta.json");
        if (!file) throw std::runtime_error("cannot open " + dir + "/meta.json");
        return nlohmann::json::parse(file);
    }
};

struct paths {
    std::string lm;
    std::string decoder;
    std::string encoder;
    std::string reference_wav;
    int threads = 4;
};

const float * as_f32(const npy_array & array) {
    return reinterpret_cast<const float *>(array.data.data());
}

tts_cpp::audio8::EngineOptions options_for(const paths & where, const nlohmann::json & meta) {
    tts_cpp::audio8::EngineOptions opts;
    opts.lm_gguf_path = where.lm;
    opts.codec_decoder_gguf_path = where.decoder;
    opts.codec_encoder_gguf_path = where.encoder;
    opts.n_threads = where.threads;
    opts.greedy = true;
    opts.max_frames = meta.at("max_new_tokens").get<int>();
    return opts;
}

tts_cpp::audio8::VoicePrompt voice_from(const std::string & wav_path,
                                        const nlohmann::json & meta) {
    return tts_cpp::audio8::load_voice_prompt(
        wav_path, meta.at("reference_text").get<std::string>());
}

bool check_waveform(const char * tag, const std::vector<float> & got,
                    const npy_array & want) {
    if (got.size() != want.n_elements()) {
        std::fprintf(stderr, "%s: FAIL %zu samples, reference has %zu\n", tag, got.size(),
                     want.n_elements());
        return false;
    }
    const compare_stats stats = compare_f32(got.data(), as_f32(want), got.size());
    print_compare(tag, stats);
    if (!std::isfinite(stats.max_abs_err)) {
        std::fprintf(stderr, "%s: FAIL non-finite samples\n", tag);
        return false;
    }
    if (stats.max_abs_err > WAVEFORM_TOLERANCE) {
        std::fprintf(stderr, "%s: FAIL max|delta| %.3e > %.1e\n", tag, stats.max_abs_err,
                     WAVEFORM_TOLERANCE);
        return false;
    }
    return true;
}

bool check_frames(const char * tag, int got, const nlohmann::json & meta) {
    const int want = meta.at("generated_frames").get<int>();
    if (got == want) return true;
    std::fprintf(stderr, "%s: FAIL %d frames, reference emitted %d\n", tag, got, want);
    return false;
}

// An encoder from another checkpoint emits fewer rows per frame than the
// prompt reads, which used to be a read past the codes. The engine now
// compares the three files before loading any of them, so the regression is
// a copy of the encoder's metadata with the codebook count moved by one --
// no weights, since it is rejected before they would be read.
constexpr const char * MISMATCHED_ENCODER = "audio8-mismatched-encoder.gguf";
// Both, so the copy stays a coherent nine-codebook encoder rather than one
// that contradicts its own quantizer bank -- the point is the disagreement
// with the other two files, not a malformed file.
const char * const NARROWED_KEYS[] = {"audio8.codec.num_codebooks",
                                      "audio8.codec.residual_codebooks"};

bool narrow_by_one(gguf_context * gguf, const char * key) {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0) {
        std::fprintf(stderr, "mismatch: %s has no %s\n", MISMATCHED_ENCODER, key);
        return false;
    }
    gguf_set_val_u32(gguf, key, gguf_get_val_u32(gguf, id) - 1);
    return true;
}

bool narrow_all(gguf_context * gguf) {
    for (const char * key : NARROWED_KEYS) {
        if (!narrow_by_one(gguf, key)) return false;
    }
    return true;
}

bool write_mismatched_encoder(const std::string & source, const char * out) {
    ggml_context * meta = nullptr;
    gguf_init_params params = {/*no_alloc=*/true, &meta};
    gguf_context * gguf = gguf_init_from_file(source.c_str(), params);
    if (!gguf) {
        std::fprintf(stderr, "mismatch: cannot read %s\n", source.c_str());
        return false;
    }
    const bool written = narrow_all(gguf) && gguf_write_to_file(gguf, out,
                                                                /*only_meta=*/true);
    gguf_free(gguf);
    ggml_free(meta);
    if (!written) std::fprintf(stderr, "mismatch: cannot write %s\n", out);
    return written;
}

bool rejected_for(const char * tag, const std::exception & failure, const char * reason) {
    if (std::strstr(failure.what(), reason)) {
        std::printf("%s: rejected with \"%s\"\n", tag, failure.what());
        return true;
    }
    std::fprintf(stderr, "%s: FAIL rejected for the wrong reason: %s\n", tag,
                 failure.what());
    return false;
}

bool run_mismatch_case(const char * tag, const paths & where) {
    if (!write_mismatched_encoder(where.encoder, MISMATCHED_ENCODER)) return false;
    tts_cpp::audio8::EngineOptions opts;
    opts.lm_gguf_path = where.lm;
    opts.codec_decoder_gguf_path = where.decoder;
    opts.codec_encoder_gguf_path = MISMATCHED_ENCODER;
    opts.n_threads = where.threads;

    bool ok = false;
    try {
        tts_cpp::audio8::Engine engine(opts);
        std::fprintf(stderr, "%s: FAIL accepted an encoder the decoder disagrees with\n",
                     tag);
    } catch (const std::exception & failure) {
        ok = rejected_for(tag, failure, "the codebook count disagrees");
    }
    std::remove(MISMATCHED_ENCODER);
    return ok;
}

bool run_case(const char * tag, const paths & where, const fixture & data, bool cloning) {
    const nlohmann::json meta = data.meta();
    tts_cpp::audio8::Engine engine(options_for(where, meta));
    const std::string text = meta.at("text").get<std::string>();

    const tts_cpp::audio8::SynthesisResult result =
        cloning ? engine.synthesize(text, voice_from(where.reference_wav, meta))
                : engine.synthesize(text);
    std::printf("%s: %d frames, %.2f s at %d Hz\n", tag, result.frames, result.duration_s,
                result.sample_rate);

    bool ok = check_frames(tag, result.frames, meta);
    ok &= check_waveform(tag, result.pcm, data.load("wav"));
    return ok;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 7) {
        std::fprintf(stderr,
                     "usage: %s <lm.gguf> <codec-decoder.gguf> <codec-encoder.gguf> "
                     "<ref-dir> <clone-ref-dir> <reference.wav> [threads]\n",
                     argv[0]);
        return 1;
    }
    paths where;
    where.lm = argv[1];
    where.decoder = argv[2];
    where.encoder = argv[3];
    where.reference_wav = argv[6];
    where.threads = argc > 7 ? std::atoi(argv[7]) : 4;

    try {
        bool ok = run_case("text", where, fixture{argv[4]}, /*cloning=*/false);
        ok &= run_case("clone", where, fixture{argv[5]}, /*cloning=*/true);
        ok &= run_mismatch_case("mismatch", where);
        std::printf("\n%s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception & failure) {
        std::fprintf(stderr, "engine: %s\n", failure.what());
        return 1;
    }
}
