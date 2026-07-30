// Direct-vs-sched dispatch equivalence for the Parler engine: synthesize the
// same seeded input as A (direct), B (TTS_CPP_FORCE_SCHED=1) and A' (direct
// again); all three PCM buffers must be bit-identical.

#include "../test_env_portable.h"
#include "tts-cpp/parler/engine.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<float> synth(const char * model, bool force_sched) {
    if (force_sched) {
        setenv("TTS_CPP_FORCE_SCHED", "1", 1);
    } else {
        unsetenv("TTS_CPP_FORCE_SCHED");
    }
    tts_cpp::parler::EngineOptions opts;
    opts.model_gguf_path = model;
    opts.seed = 42;
    opts.max_frames = 48;
    opts.n_threads = 4;
    tts_cpp::parler::Engine engine(opts);
    return engine.synthesize("Testing dispatch equivalence.",
                             "A calm female voice with studio quality.").pcm;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
        return 2;
    }
    try {
        const std::vector<float> a  = synth(argv[1], false);
        const std::vector<float> b  = synth(argv[1], true);
        const std::vector<float> a2 = synth(argv[1], false);
        unsetenv("TTS_CPP_FORCE_SCHED");
        if (a.empty()) {
            fprintf(stderr, "FAIL: empty PCM\n");
            return 1;
        }
        if (a.size() != b.size() || a.size() != a2.size()) {
            fprintf(stderr, "FAIL: size mismatch A=%zu B=%zu A'=%zu\n",
                    a.size(), b.size(), a2.size());
            return 1;
        }
        if (std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) != 0) {
            fprintf(stderr, "FAIL: direct vs sched PCM differs\n");
            return 1;
        }
        if (std::memcmp(a.data(), a2.data(), a.size() * sizeof(float)) != 0) {
            fprintf(stderr, "FAIL: direct run not reproducible\n");
            return 1;
        }
        fprintf(stderr, "parler sched equivalence: PASS (%zu samples bit-identical)\n",
                a.size());
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
