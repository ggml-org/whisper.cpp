#include <cstdlib>
#include <cstdio>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#endif
#ifndef WHISPER_STREAM_PCM_PATH
#error "WHISPER_STREAM_PCM_PATH is not defined"
#endif

#ifndef WHISPER_TEST_MODEL_PATH
#error "WHISPER_TEST_MODEL_PATH is not defined"
#endif

static std::filesystem::path temp_pcm_path() {
    std::error_code ec;
    auto dir = std::filesystem::temp_directory_path(ec);
    if (ec) {
        return std::filesystem::path("whisper_stream_pcm_test.raw");
    }
    return dir / "whisper_stream_pcm_test.raw";
}

int main() {
    const int sample_rate = 16000;
    const int seconds = 2;
    const size_t n_samples = (size_t) sample_rate * seconds;

    std::vector<float> zeros(n_samples, 0.0f);

    auto pcm_path = temp_pcm_path();
    pcm_path.make_preferred();
    std::ofstream out(pcm_path, std::ios::binary);
    if (!out.is_open()) {
        fprintf(stderr, "failed to open temp PCM path: %s\n", pcm_path.string().c_str());
        return 1;
    }

    out.write(reinterpret_cast<const char *>(zeros.data()), zeros.size() * sizeof(float));
    out.close();

    const std::string stream_bin = std::filesystem::path(WHISPER_STREAM_PCM_PATH).make_preferred().string();
    const std::string model_path = std::filesystem::path(WHISPER_TEST_MODEL_PATH).make_preferred().string();

    std::vector<std::string> args = {
        stream_bin,
        "-m", model_path,
        "--input", pcm_path.string(),
        "--format", "f32",
        "--sample-rate", "16000",
        "--step", "500",
        "--length", "2000",
        "-t", "1",
        "-ng",
    };

    int rc = 1;
#if defined(_WIN32)
    std::vector<const char *> argv;
    argv.reserve(args.size() + 1);
    for (const auto & arg : args) {
        argv.push_back(arg.c_str());
    }
    argv.push_back(nullptr);

    rc = _spawnv(_P_WAIT, stream_bin.c_str(), argv.data());
    if (rc == -1) {
        fprintf(stderr, "failed to spawn whisper-stream-pcm: %s\n", std::strerror(errno));
        rc = 1;
    }
#else
    std::string cmd;
    cmd.reserve(1024);
    for (const auto & arg : args) {
        if (!cmd.empty()) {
            cmd += " ";
        }
        cmd += "\"";
        cmd += arg;
        cmd += "\"";
    }
    rc = std::system(cmd.c_str());
#endif

    std::error_code ec;
    std::filesystem::remove(pcm_path, ec);

    if (rc != 0) {
        fprintf(stderr, "whisper-stream-pcm exited with code %d\n", rc);
        return 1;
    }

    return 0;
}
