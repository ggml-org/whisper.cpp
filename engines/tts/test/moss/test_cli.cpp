#include "gguf_fixtures.h"

#include "moss/cli.h"

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using namespace moss_fixtures;
using tts_cpp::moss::cli::CliArgs;
using tts_cpp::moss::cli::parse_args;

namespace {

int failures = 0;

void check(bool condition, const std::string & label) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", label.c_str());
        failures++;
    }
}

bool parse(const std::vector<const char *> & argv, CliArgs & args) {
    return parse_args((int) argv.size(), argv.data(), args);
}

void test_required_flags() {
    CliArgs full;
    check(parse({"moss-cli", "--backbone", "b.gguf", "--decoder", "d.gguf", "--text", "hi"}, full),
            "backbone + decoder + text parse");
    check(full.options.backbone_path == "b.gguf" && full.options.decoder_path == "d.gguf" &&
            full.text == "hi", "required values land in the options");

    CliArgs no_backbone;
    check(!parse({"moss-cli", "--decoder", "d.gguf", "--text", "hi"}, no_backbone),
            "--backbone is required");
    CliArgs no_text;
    check(!parse({"moss-cli", "--backbone", "b.gguf", "--decoder", "d.gguf"}, no_text),
            "--text is required");
    CliArgs bogus;
    check(!parse({"moss-cli", "--backbone", "b.gguf", "--decoder", "d.gguf", "--text", "hi",
            "--bogus"}, bogus), "unknown flags are rejected");
}

void test_option_mapping() {
    CliArgs args;
    check(parse({"moss-cli", "--backbone", "b.gguf", "--decoder", "d.gguf", "--text", "hi",
            "--encoder", "e.gguf", "--ref-audio", "r.wav", "--language", "en",
            "--max-new-tokens", "7", "--context", "128", "--seed", "42", "--threads", "2",
            "--duration-tokens", "38", "--backends-dir", "/opt/backends", "--backend", "hexagon",
            "--gpu", "--text-temperature", "0.5", "--audio-top-k", "9", "--out", "x.wav"}, args),
            "full flag surface parses");
    check(args.options.encoder_path == "e.gguf" && args.options.reference_audio_path == "r.wav",
            "cloning flags map");
    check(args.options.language == "en" && args.options.max_new_tokens == 7 &&
            args.options.context == 128 && args.options.seed == 42 &&
            args.options.n_threads == 2 && args.options.use_gpu, "runtime flags map");
    check(args.options.text_temperature == 0.5f && args.options.audio_top_k == 9,
            "sampling flags map");
    check(args.options.duration_tokens == 38, "duration flag maps");
    check(args.options.backends_dir == "/opt/backends", "backends directory flag maps");
    check(args.options.backend == "hexagon", "backend flag maps");
    check(args.out_path == "x.wav", "output path maps");
}

void test_backend_modes() {
    CliArgs sfx;
    check(parse({"moss-cli", "--mode", "sfx", "--model", "m.gguf", "--text", "rain",
            "--backend", "hexagon"}, sfx), "--backend is accepted in sfx mode");
    check(sfx.sound_options.backend == "hexagon", "backend flag reaches the sfx engine");
    CliArgs s2s;
    check(!parse({"moss-cli", "--mode", "s2s", "--model", "m.gguf", "--codec", "c.gguf",
            "--audio", "q.wav", "--backend", "hexagon"}, s2s), "--backend is rejected in s2s mode");
}

void test_end_to_end_run() {
    const auto backbone = write_backbone("cli-backbone", [](gguf_context *) {});
    const auto decoder = write_decoder("cli-decoder", N_VQ, 3 * D_MODEL, [](gguf_context *) {});
    const auto out = temp_gguf("cli-out");
    const std::string out_wav = out.string() + ".wav";
    CliArgs args;
    args.options.backbone_path = backbone.string();
    args.options.decoder_path = decoder.string();
    args.options.n_threads = 1;
    args.options.context = 512;
    args.options.max_new_tokens = 24;
    args.text = "hi";
    args.out_path = out_wav;
    const int rc = tts_cpp::moss::cli::run(args);
    check(rc == 0, "cli run succeeds on the fixture models");
    check(std::filesystem::exists(out_wav) && std::filesystem::file_size(out_wav) > 44,
            "cli writes a non-empty WAV");
    args.options.backend = "cpu";
    check(tts_cpp::moss::cli::run(args) == 0, "an explicit cpu backend runs");
    args.options.backend = "no-such-device";
    check(tts_cpp::moss::cli::run(args) != 0, "an unavailable backend fails instead of falling back");
    std::filesystem::remove(backbone);
    std::filesystem::remove(decoder);
    std::filesystem::remove(out_wav);
}

} // namespace

int main() {
    try {
        test_required_flags();
        test_option_mapping();
        test_backend_modes();
        test_end_to_end_run();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    if (failures == 0) {
        std::printf("moss cli: OK\n");
        return 0;
    }
    std::fprintf(stderr, "moss cli: %d failures\n", failures);
    return 1;
}
