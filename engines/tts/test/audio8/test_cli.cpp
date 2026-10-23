// audio8-cli's argument surface and its code dump, without a model.
//
// The CLI is what the campaign's scripts and the on-device harness drive, so
// its flags are part of the engine's contract: -ngl selects the backend,
// --verbose asks for the stage breakdown and --dump-codes writes the discrete
// trajectory two runs get compared on.

#include "audio8/cli.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using tts_cpp::audio8::cli::options;
using tts_cpp::audio8::cli::parse_args;
using tts_cpp::audio8::cli::write_codes;

namespace {

// Both GGUF paths are required, so every case that expects a parse carries them.
const std::vector<const char *> REQUIRED = {"audio8-cli", "--lm", "lm.gguf",
                                            "--codec-decoder", "dec.gguf"};

bool parse(const std::vector<const char *> & tail, options & opts) {
    std::vector<const char *> argv = REQUIRED;
    argv.insert(argv.end(), tail.begin(), tail.end());
    return parse_args(static_cast<int>(argv.size()), argv.data(), opts);
}

bool fail(const char * tag, const std::string & detail) {
    std::fprintf(stderr, "%s: FAIL %s\n", tag, detail.c_str());
    return false;
}

bool check_required_paths() {
    options opts;
    if (!parse({}, opts)) return fail("required", "both GGUF paths were rejected");
    if (opts.lm != "lm.gguf" || opts.codec_decoder != "dec.gguf") {
        return fail("required", "the GGUF paths did not land in the options");
    }
    const char * without_lm[] = {"audio8-cli", "--codec-decoder", "dec.gguf"};
    options no_lm;
    if (parse_args(3, without_lm, no_lm)) return fail("required", "--lm was not required");
    const char * without_decoder[] = {"audio8-cli", "--lm", "lm.gguf"};
    options no_decoder;
    if (parse_args(3, without_decoder, no_decoder)) {
        return fail("required", "--codec-decoder was not required");
    }
    std::printf("required: PASS\n");
    return true;
}

// -ngl is the spelling the CosyVoice CLIs use and the one the bench scripts pass.
bool check_gpu_layers() {
    options spelled;
    options aliased;
    if (!parse({"--n-gpu-layers", "99"}, spelled) || !parse({"-ngl", "99"}, aliased)) {
        return fail("gpu-layers", "the flag was rejected");
    }
    if (spelled.n_gpu_layers != 99 || aliased.n_gpu_layers != 99) {
        return fail("gpu-layers", "expected 99 from both spellings");
    }
    options defaulted;
    if (!parse({}, defaulted) || defaulted.n_gpu_layers != 0) {
        return fail("gpu-layers", "the default is not CPU");
    }
    std::printf("gpu-layers: PASS\n");
    return true;
}

// A switch that ate the next argument would swallow whatever followed it.
bool check_switches() {
    options opts;
    if (!parse({"--verbose", "--stream", "--greedy", "--seed", "7"}, opts)) {
        return fail("switches", "the switches were rejected");
    }
    if (!opts.stream) return fail("switches", "--stream did not set stream");
    if (!opts.verbose) return fail("switches", "--verbose did not set verbose");
    if (!opts.greedy) return fail("switches", "--greedy did not set greedy");
    if (opts.seed != 7) return fail("switches", "a switch consumed the seed");
    options defaulted;
    if (!parse({}, defaulted) || defaulted.verbose || defaulted.stream) {
        return fail("switches", "verbose is on by default");
    }
    std::printf("switches: PASS\n");
    return true;
}

// --backend lands as a verbatim string in options.backend and defaults to
// empty (= EngineOptions::backend "auto" on the engine side). The flag
// must accept any value: "cpu", "hexagon", "HTP0", "CUDA0" are all legal
// here; validation happens inside init_backend at engine construction.
bool check_backend_flag() {
    for (const char * value : {"cpu", "hexagon", "HTP0", "CUDA0", "auto"}) {
        options opts;
        if (!parse({"--backend", value}, opts)) {
            return fail("backend", std::string("--backend ") + value + " was rejected");
        }
        if (opts.backend != value) {
            return fail("backend",
                        std::string("--backend ") + value + " did not land in options.backend");
        }
    }
    options defaulted;
    if (!parse({}, defaulted) || !defaulted.backend.empty()) {
        return fail("backend", "default options.backend is not empty");
    }
    std::printf("backend: PASS\n");
    return true;
}

bool check_dump_codes_flag() {
    options opts;
    if (!parse({"--dump-codes", "codes.txt"}, opts)) {
        return fail("dump-codes", "the flag was rejected");
    }
    if (opts.codes_out != "codes.txt") return fail("dump-codes", "the path was not kept");
    options defaulted;
    if (!parse({}, defaulted) || !defaulted.codes_out.empty()) {
        return fail("dump-codes", "codes are dumped by default");
    }
    std::printf("dump-codes: PASS\n");
    return true;
}

bool rejects(const std::vector<const char *> & tail) {
    options opts;
    return !parse(tail, opts);
}

bool check_malformed() {
    if (!rejects({"--not-a-flag", "x"})) return fail("malformed", "an unknown flag parsed");
    if (!rejects({"--seed"})) return fail("malformed", "a flag with no value parsed");
    if (!rejects({"--dump-codes"})) {
        return fail("malformed", "--dump-codes with no path parsed");
    }
    std::printf("malformed: PASS\n");
    return true;
}

std::vector<std::string> read_lines(const std::string & path) {
    std::vector<std::string> lines;
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) lines.push_back(line);
    return lines;
}

// Frame-major, num_codebooks per frame: what SynthesisResult::codes holds.
bool check_dump_codes_format(const std::string & path) {
    const std::vector<int> codes = {1, 2, 3, 40, 50, 60};
    if (!write_codes(path, codes, 2)) return fail("codes-format", "the file was not written");
    const std::vector<std::string> lines = read_lines(path);
    if (lines.size() != 2) {
        return fail("codes-format", "expected one line per frame, got " +
                                        std::to_string(lines.size()));
    }
    if (lines[0] != "1,2,3") return fail("codes-format", "frame 0 reads " + lines[0]);
    if (lines[1] != "40,50,60") return fail("codes-format", "frame 1 reads " + lines[1]);
    if (write_codes(path, codes, 0)) return fail("codes-format", "zero frames was written");
    std::printf("codes-format: PASS\n");
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    const std::string codes_path = argc > 1 ? argv[1] : "audio8_test_codes.txt";

    bool ok = check_required_paths();
    ok &= check_gpu_layers();
    ok &= check_switches();
    ok &= check_backend_flag();
    ok &= check_dump_codes_flag();
    ok &= check_malformed();
    ok &= check_dump_codes_format(codes_path);

    std::remove(codes_path.c_str());
    std::printf("\n%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
