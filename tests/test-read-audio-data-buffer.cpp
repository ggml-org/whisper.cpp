#include "common-whisper.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

static std::vector<char> read_file_bytes(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    assert(input.good());
    return std::vector<char>(
            std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>());
}

static void assert_pcm_nearly_equal(const std::vector<float> & expected, const std::vector<float> & actual) {
    assert(expected.size() == actual.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        assert(std::fabs(expected[i] - actual[i]) < 1e-6f);
    }
}

int main() {
    const std::string sample_path = SAMPLE_PATH;
    const std::vector<char> wav_data = read_file_bytes(sample_path);
    assert(!wav_data.empty());

    std::vector<float> pcm_from_file;
    std::vector<std::vector<float>> stereo_from_file;
    assert(read_audio_data(sample_path, pcm_from_file, stereo_from_file, false));
    assert(!pcm_from_file.empty());
    assert(stereo_from_file.empty());

    std::vector<float> pcm_from_memory;
    std::vector<std::vector<float>> stereo_from_memory;
    assert(read_audio_data(wav_data.data(), wav_data.size(), pcm_from_memory, stereo_from_memory, false));
    assert(!pcm_from_memory.empty());
    assert(stereo_from_memory.empty());

    assert_pcm_nearly_equal(pcm_from_file, pcm_from_memory);

    printf("Decoded %zu bytes from memory into %zu PCM samples\n", wav_data.size(), pcm_from_memory.size());
    return 0;
}
