// Tiny shared helpers for the C++ test binaries. Kept dependency-light
// (just the standard headers below) so any test can include this without
// pulling in the public Engine surface or any project-internal types.
//
// History: previously these helpers lived inline in
// test_sortformer_streaming.cpp and test_sortformer_aosc_speakers.cpp.
// Pulling them up here avoids drift between two near-identical copies.
#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace parakeet_test {

inline bool file_exists(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    return f.good();
}

// Load a 16 kHz / mono / s16le RIFF/WAVE file into [-1, 1) float samples.
// Returns false on any header mismatch (non-PCM, non-mono, non-16bit) or
// missing chunk; on success writes the sample rate via `sample_rate`.
inline bool load_wav_pcm16le_mono(const std::string & path,
                                  std::vector<float> & samples,
                                  int & sample_rate) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char riff[4]; f.read(riff, 4);
    if (std::memcmp(riff, "RIFF", 4) != 0) return false;
    f.ignore(4);
    char wave[4]; f.read(wave, 4);
    if (std::memcmp(wave, "WAVE", 4) != 0) return false;

    bool fmt_ok = false; uint16_t channels = 0; uint16_t bits = 0; uint32_t srate = 0;
    std::vector<char> data;
    while (f) {
        char id[4]; f.read(id, 4);
        if (!f) break;
        uint32_t sz = 0; f.read((char *) &sz, 4);
        if (std::memcmp(id, "fmt ", 4) == 0) {
            std::vector<char> hdr(sz);
            f.read(hdr.data(), sz);
            uint16_t fmt = *(uint16_t *) hdr.data();
            channels    = *(uint16_t *) (hdr.data() + 2);
            srate       = *(uint32_t *) (hdr.data() + 4);
            bits        = *(uint16_t *) (hdr.data() + 14);
            if (fmt != 1 || channels != 1 || bits != 16) return false;
            fmt_ok = true;
        } else if (std::memcmp(id, "data", 4) == 0) {
            data.resize(sz);
            f.read(data.data(), sz);
            break;
        } else {
            f.ignore(sz);
        }
    }
    if (!fmt_ok || data.empty()) return false;
    sample_rate = (int) srate;
    const int n = (int) (data.size() / 2);
    samples.resize(n);
    const int16_t * s16 = reinterpret_cast<const int16_t *>(data.data());
    for (int i = 0; i < n; ++i) samples[i] = (float) s16[i] / 32768.0f;
    return true;
}

}  // namespace parakeet_test
