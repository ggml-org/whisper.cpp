// Engine robustness: construction/argument errors must throw cleanly (no
// crashes, no leaks under ASAN), cancel() must interrupt a synthesis, and
// consecutive synthesize() calls on one engine must work (KV reset,
// description cache reuse and replacement).

#include "tts-cpp/parler/engine.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>

using tts_cpp::parler::Engine;
using tts_cpp::parler::EngineOptions;

static int failures = 0;

static void expect_throw(const char * what, void (*fn)(const char *), const char * arg) {
    try {
        fn(arg);
        fprintf(stderr, "FAIL: %s did not throw\n", what);
        failures++;
    } catch (const std::exception & e) {
        fprintf(stderr, "  ok: %s -> %s\n", what, e.what());
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL.gguf [WRONG_ARCH.gguf]\n", argv[0]);
        return 2;
    }
    const std::string model = argv[1];

    expect_throw("missing model file", [](const char *) {
        EngineOptions o;
        o.model_gguf_path = "/nonexistent/parler.gguf";
        Engine e(o);
    }, nullptr);

    if (argc > 2) {
        // only meaningful when the fixture is actually staged: a missing file
        // also throws, which would pass this check for the wrong reason
        FILE * probe = fopen(argv[2], "rb");
        if (!probe) {
            fprintf(stderr, "  skip: wrong-arch fixture not staged (%s)\n", argv[2]);
        } else {
            fclose(probe);
            try {
                EngineOptions o;
                o.model_gguf_path = argv[2];
                Engine e(o);
                fprintf(stderr, "FAIL: wrong-arch GGUF did not throw\n");
                failures++;
            } catch (const std::exception & e) {
                if (std::string(e.what()).find("parler.arch") == std::string::npos) {
                    fprintf(stderr, "FAIL: wrong-arch GGUF threw, but not the arch check: %s\n",
                            e.what());
                    failures++;
                } else {
                    fprintf(stderr, "  ok: wrong-arch GGUF -> %s\n", e.what());
                }
            }
        }
    }

    EngineOptions opts;
    opts.model_gguf_path = model;
    opts.greedy = true;
    opts.max_frames = 24;
    opts.n_threads = 4;
    Engine engine(opts);

    // argument validation
    try {
        engine.synthesize("hello");
        fprintf(stderr, "FAIL: missing default_description did not throw\n");
        failures++;
    } catch (const std::exception & e) {
        fprintf(stderr, "  ok: no default description -> %s\n", e.what());
    }
    try {
        engine.synthesize("", "a voice");
        fprintf(stderr, "FAIL: empty prompt did not throw\n");
        failures++;
    } catch (const std::exception & e) {
        fprintf(stderr, "  ok: empty prompt -> %s\n", e.what());
    }
    try {
        engine.synthesize("hello", "");
        fprintf(stderr, "FAIL: empty description did not throw\n");
        failures++;
    } catch (const std::exception & e) {
        fprintf(stderr, "  ok: empty description -> %s\n", e.what());
    }
    // max_frames below n_codebooks+1 can never yield a frame; must throw
    // instead of aborting in the delay module (assert) or dying late
    try {
        EngineOptions tiny = opts;
        tiny.max_frames = 1;
        Engine e(tiny);
        e.synthesize("hello", "a voice");
        fprintf(stderr, "FAIL: max_frames=1 did not throw\n");
        failures++;
    } catch (const std::exception & e) {
        fprintf(stderr, "  ok: max_frames=1 -> %s\n", e.what());
    }

    // consecutive syntheses: same description (cross-KV cache reuse), then a
    // different description (cache replacement)
    {
        auto r1 = engine.synthesize("First call.", "A calm female voice.");
        auto r2 = engine.synthesize("Second call, same voice.", "A calm female voice.");
        auto r3 = engine.synthesize("Third call.", "A deep male voice in a large hall.");
        if (r1.pcm.empty() || r2.pcm.empty() || r3.pcm.empty()) {
            fprintf(stderr, "FAIL: consecutive syntheses produced empty PCM\n");
            failures++;
        } else {
            fprintf(stderr, "  ok: 3 consecutive syntheses (%.2fs, %.2fs, %.2fs)\n",
                    r1.duration_s, r2.duration_s, r3.duration_s);
        }
    }

    // cancel: fire from another thread shortly after starting a long synth
    {
        EngineOptions lo = opts;
        lo.max_frames = 0; // full-length budget so there is time to cancel
        Engine slow(lo);
        std::thread killer([&slow]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            slow.cancel();
        });
        bool threw = false;
        try {
            slow.synthesize("This synthesis is going to be cancelled before it finishes, "
                            "so this long sentence should never be fully generated.",
                            "A calm female voice.");
        } catch (const std::exception & e) {
            threw = true;
            fprintf(stderr, "  ok: cancel -> %s\n", e.what());
        }
        killer.join();
        if (!threw) {
            fprintf(stderr, "FAIL: cancel did not interrupt synthesis\n");
            failures++;
        }
    }

    if (failures == 0) {
        fprintf(stderr, "parler engine: PASS\n");
        return 0;
    }
    fprintf(stderr, "parler engine: %d failure(s)\n", failures);
    return 1;
}
