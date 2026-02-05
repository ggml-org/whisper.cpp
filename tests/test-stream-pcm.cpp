#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

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

    const auto pcm_path = temp_pcm_path();
    std::ofstream out(pcm_path, std::ios::binary);
    if (!out.is_open()) {
        fprintf(stderr, "failed to open temp PCM path: %s\n", pcm_path.string().c_str());
        return 1;
    }

    out.write(reinterpret_cast<const char *>(zeros.data()), zeros.size() * sizeof(float));
    out.close();

    const std::string stream_bin = WHISPER_STREAM_PCM_PATH;
    const std::string model_path = WHISPER_TEST_MODEL_PATH;

    std::string cmd;
    cmd.reserve(1024);
    cmd += "\"" + stream_bin + "\"";
    cmd += " -m \"" + model_path + "\"";
    cmd += " --input \"" + pcm_path.string() + "\"";
    cmd += " --format f32 --sample-rate 16000 --step 500 --length 2000 -t 1 -ng";

    const int rc = std::system(cmd.c_str());

    std::error_code ec;
    std::filesystem::remove(pcm_path, ec);

    if (rc != 0) {
        fprintf(stderr, "whisper-stream-pcm exited with code %d\n", rc);
        return 1;
    }

    return 0;
}
