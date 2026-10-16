#include "tts-cpp/pocket/fit.h"
#include "test_workload.h"
#include "test_backend.h"
#include "gguf.h"
#include "ggml.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace tts_cpp;
namespace fs = std::filesystem;
void check(bool condition, const char * message) { if (!condition) throw std::runtime_error(message); }
void metadata_copy(const fs::path & src, const fs::path & dst) {
    ggml_context * ctx = nullptr;
    auto * gguf = gguf_init_from_file(src.string().c_str(), {true, &ctx});
    check(gguf && ctx, "cannot open fixture metadata");
    const auto bytes = gguf_get_data_offset(gguf);
    gguf_free(gguf); ggml_free(ctx);
    std::vector<char> data(bytes);
    std::ifstream input(src, std::ios::binary); input.read(data.data(), bytes);
    check(bool(input), "metadata read failed");
    std::ofstream output(dst, std::ios::binary); output.write(data.data(), bytes);
    check(bool(output), "metadata write failed");
}
void reference_wav(const fs::path & path, bool hostile) {
    std::ofstream f(path, std::ios::binary);
    if (hostile) {
        // 82-byte RF64 header claims 2^30 frames (4 GiB float allocation in
        // an unbounded decoder) despite containing only one PCM16 sample.
        const std::string hex = "52463634ffffffff57415645647336341c0000004a000000000000000200"
            "000000000000000000400000000000000000666d74201000000001000100"
            "c05d000080bb00000200100064617461ffffffff0000";
        for (size_t i = 0; i < hex.size(); i += 2) f.put(char(std::stoul(hex.substr(i, 2), nullptr, 16)));
        return;
    }
    auto put = [&](uint64_t value, int n) { for (int i = 0; i < n; ++i) f.put(char(value >> (8*i))); };
    f << "RIFF"; put(36+4800, 4); f << "WAVEfmt "; put(16, 4);
    put(1, 2); put(1, 2); put(24000, 4); put(48000, 4); put(2, 2); put(16, 2);
    f << "data"; put(4800, 4); for (int i = 0; i < 2400; ++i) put(0, 2);
}
void check_tail_budget(pocket::FitOptions opts) {
    opts.text = "Hi.";
    const auto normal = pocket::fit_params(opts);
    opts.engine.frames_after_eos = 100;
    const auto long_tail = pocket::fit_params(opts);
    check(normal.status != FitStatus::Error && long_tail.status != FitStatus::Error, "tail fit failed");
    check(long_tail.host_bytes > normal.host_bytes, "EOS tail not included in host memory estimate");
    opts.engine.context = context_with_tail(opts.engine, opts.text);
    check(pocket::fit_params(opts).status != FitStatus::Error, "exact context boundary rejected");
    --opts.engine.context;
    check(pocket::fit_params(opts).reason == "workload-too-large", "context omitted EOS tail");
    opts.engine.frames_after_eos = -1;
    opts.engine.context = context_with_tail(opts.engine, opts.text);
    check(pocket::fit_params(opts).status != FitStatus::Error, "default-tail context boundary rejected");
    --opts.engine.context;
    check(pocket::fit_params(opts).reason == "workload-too-large", "context omitted default EOS tail");
}
int main(int argc, char ** argv) {
    try {
        check(argc == 2 || argc == 3, "expected Pocket bundle directory [--gpu]");
        pocket::FitOptions opts;
        opts.engine.use_gpu = argc == 3 && std::string(argv[2]) == "--gpu";
        opts.engine.backend = argc == 3 && !opts.engine.use_gpu ? argv[2] : "";
        const fs::path bundle = argv[1];
        auto set_paths = [&](const fs::path & dir) {
            opts.engine.flow_lm_path = (dir/"flow-lm.gguf").string();
            opts.engine.mimi_path = (dir/"mimi.gguf").string();
            opts.engine.frontend_path = (dir/"frontend.json").string();
            opts.engine.voice_path = (dir/"voice.gguf").string();
        };
        set_paths(bundle); check_tail_budget(opts); opts.text = "Hello! This is a Pocket memory preflight.";
        const auto result = pocket::fit_params(opts);
        check(result.status != FitStatus::Error, result.report.c_str());
        check(result.device_is_cpu == (!opts.engine.use_gpu && opts.engine.backend != "opencl"), "fit reported the wrong backend");
        if (opts.engine.backend != "opencl") check_pocket_backend(result.device_name.c_str(), opts.engine.use_gpu);
        check(result.device.weights_bytes > 100*1024*1024, "expanded weights not priced");
        check(result.device.state_bytes && result.device.lm_compute_bytes && result.device.codec_compute_bytes && result.host_bytes, "missing memory components");
        std::cout << result.report;
        const auto temp = fs::temp_directory_path()/(
            "pocket-fit-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directory(temp);
        struct Cleanup { fs::path p; ~Cleanup() { std::error_code e; fs::remove_all(p, e); } } cleanup{temp};
        for (const char * name : {"flow-lm.gguf", "mimi.gguf", "voice.gguf"}) metadata_copy(bundle/name, temp/name);
        fs::copy_file(bundle/"frontend.json", temp/"frontend.json");
        set_paths(temp);
        const auto metadata_only = pocket::fit_params(opts);
        check(metadata_only.status != FitStatus::Error, metadata_only.report.c_str());
        check(metadata_only.device.total_bytes == result.device.total_bytes && metadata_only.host_bytes == result.host_bytes,
              "projection read or depended on tensor payloads");
        opts.engine.voice_path.clear();
        opts.engine.reference_audio_path = (temp/"reference.wav").string();
        reference_wav(opts.engine.reference_audio_path, false);
        check(pocket::fit_params(opts).status != FitStatus::Error, "valid reference header rejected");
        reference_wav(opts.engine.reference_audio_path, true);
        check(pocket::fit_params(opts).reason == "invalid-arguments", "RF64 oversized decoded allocation accepted");
        auto direct = opts.engine;
        direct.flow_lm_path = "missing-model-must-not-be-opened.gguf";
        bool rejected_header = false;
        try { pocket::Engine invalid(direct); }
        catch (const std::invalid_argument & error) {
            rejected_header = std::string(error.what()).find("reference") != std::string::npos;
        }
        check(rejected_header, "direct Engine did not reject malformed reference before model allocation");
        opts.engine.reference_audio_path.clear();
        opts.engine.voice_path = (temp/"voice.gguf").string();
        opts.memory_budget_bytes = 1;
        auto small = pocket::fit_params(opts);
        check(small.status == FitStatus::Failure && !small.fits, "tiny budget incorrectly fits");
        opts.memory_budget_bytes = 0; opts.margin_bytes = std::numeric_limits<uint64_t>::max();
        check(pocket::fit_params(opts).status == FitStatus::Failure, "margin overflow incorrectly fits");
        opts.margin_bytes = 0; opts.text.clear();
        check(pocket::fit_params(opts).reason == "invalid-arguments", "empty text accepted");
        opts.text = "Hello from Pocket"; opts.engine.context = 8;
        check(pocket::fit_params(opts).reason == "workload-too-large", "oversized workload accepted");
        opts.engine.context = 2048; opts.engine.flow_lm_path = (temp/"missing.gguf").string();
        check(pocket::fit_params(opts).status == FitStatus::Error, "missing model accepted");
        std::cout << "Pocket metadata-only fit tests passed\n";
        return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
