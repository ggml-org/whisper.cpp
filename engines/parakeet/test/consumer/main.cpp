// Minimal downstream consumer of the installed qvac-parakeet package.
//
// Includes the umbrella header (which pulls in <parakeet/log.h> ->
// "ggml.h", so it also proves the package advertises ggml's include dir)
// and forces the linker to resolve a wide slice of the library.
//
// Why it references parakeet_cli_main rather than only calling the cheap
// log setter: a static archive contributes only the object files needed
// to resolve referenced symbols, so a one-symbol consumer links fine even
// when the .pc omits the library's PRIVATE dependencies. Taking the
// address of parakeet_cli_main pulls in the engine/mel closure -- and
// with it mel_preprocess.cpp (OpenMP) and, under PARAKEET_COREML, the
// Core ML wrapper (Foundation/CoreML frameworks). Those are exactly the
// deps Libs.private must advertise, so an incomplete .pc fails to link
// here instead of silently passing.

#include <parakeet/parakeet.h>

#include <cstdio>

int main() {
    // Exported C entry point; restores the default stderr sink.
    parakeet_log_set(nullptr, nullptr);

    // Reference (never invoke) the CLI entry point so the linker must
    // resolve the library's full transitive symbol closure. volatile so
    // the optimizer cannot discard the reference.
    int (*volatile cli)(int, char **) = &parakeet_cli_main;
    if (cli == nullptr) {
        std::fprintf(stderr, "parakeet_cli_main unexpectedly null\n");
        return 1;
    }

    std::printf("qvac-parakeet consumer OK\n");
    return 0;
}
