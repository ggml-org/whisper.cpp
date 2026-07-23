// End-to-end transcription parity: Engine::transcribe() driven by the Core ML
// (Apple Neural Engine) encoder must produce the same text as the ggml encoder.
//
// Each transcription runs in its own forked child so that every Engine is the
// only one in its process (a single process that constructs two GPU-backed
// Engines in sequence is not supported). Child A loads normally (the Core ML
// sidecar drives the encoder when present); child B loads with
// PARAKEET_COREML_DISABLE set (forcing the ggml encoder). The parent compares
// the two transcripts.
//
// Skips (exit 0) when the Core ML sidecar is not active (non-Apple build,
// PARAKEET_COREML off, or no `<model>-encoder.mlmodelc`), so it is a no-op on CI
// without Apple hardware and a real gate on Apple with a sidecar present.

#include "parakeet/engine.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

#ifndef _WIN32
namespace {

std::string transcribe_in_child(const std::string & gguf, const std::string & wav, bool disable) {
    int fds[2];
    if (pipe(fds) != 0) {
        return std::string();
    }
    const pid_t pid = fork();
    if (pid == 0) {
        close(fds[0]);
        if (disable) {
            setenv("PARAKEET_COREML_DISABLE", "1", 1);
        }
        parakeet::EngineOptions opts;
        opts.model_gguf_path = gguf;
        opts.n_gpu_layers    = 999;
        parakeet::Engine engine(opts);
        std::string payload = std::string("coreml=") +
                              (engine.encoder_on_coreml() ? "1" : "0") + "\n" +
                              engine.transcribe(wav).text;
        for (size_t off = 0; off < payload.size();) {
            const ssize_t n = write(fds[1], payload.data() + off, payload.size() - off);
            if (n <= 0) break;
            off += static_cast<size_t>(n);
        }
        close(fds[1]);
        _exit(0);
    }
    close(fds[1]);
    std::string out;
    char chunk[4096];
    ssize_t n;
    while ((n = read(fds[0], chunk, sizeof(chunk))) > 0) {
        out.append(chunk, static_cast<size_t>(n));
    }
    close(fds[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    return out;
}

bool parse_child(const std::string & raw, bool & on_coreml, std::string & text) {
    const std::string prefix = "coreml=";
    if (raw.rfind(prefix, 0) != 0) {
        return false;
    }
    const size_t nl = raw.find('\n');
    if (nl == std::string::npos) {
        return false;
    }
    on_coreml = raw.substr(prefix.size(), nl - prefix.size()) == "1";
    text = raw.substr(nl + 1);
    return true;
}

}  // namespace
#endif

int main(int argc, char ** argv) {
#ifdef _WIN32
    (void) argc;
    (void) argv;
    std::fprintf(stderr, "[transcribe-coreml-parity] SKIP: Core ML is Apple-only.\n");
    return 0;
#else
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <gguf> <wav>\n", argv[0]);
        return 2;
    }
    const std::string gguf = argv[1];
    const std::string wav  = argv[2];

    bool        coreml_on = false;
    std::string coreml_text;
    if (!parse_child(transcribe_in_child(gguf, wav, /*disable=*/false), coreml_on, coreml_text)) {
        std::fprintf(stderr, "[transcribe-coreml-parity] FAIL: no output from Core ML child\n");
        return 1;
    }
    if (!coreml_on) {
        std::fprintf(stderr,
            "[transcribe-coreml-parity] SKIP: Core ML encoder not active "
            "(non-Apple build, PARAKEET_COREML off, or no sidecar).\n");
        return 0;
    }

    bool        ggml_on = true;
    std::string ggml_text;
    if (!parse_child(transcribe_in_child(gguf, wav, /*disable=*/true), ggml_on, ggml_text)) {
        std::fprintf(stderr, "[transcribe-coreml-parity] FAIL: no output from ggml child\n");
        return 1;
    }
    if (ggml_on) {
        std::fprintf(stderr, "[transcribe-coreml-parity] FAIL: PARAKEET_COREML_DISABLE ignored\n");
        return 1;
    }

    std::fprintf(stderr, "[transcribe-coreml-parity] coreml=\"%s\"\n", coreml_text.c_str());
    std::fprintf(stderr, "[transcribe-coreml-parity] ggml  =\"%s\"\n", ggml_text.c_str());
    if (coreml_text != ggml_text) {
        std::fprintf(stderr, "[transcribe-coreml-parity] FAIL: transcripts differ\n");
        return 1;
    }
    std::fprintf(stderr, "[transcribe-coreml-parity] PASS (identical transcript)\n");
    return 0;
#endif
}
