// Tiny shared helpers for the C++ test binaries. Kept dependency-light
// (just the standard headers below) so any test can include this without
// pulling in the public Engine surface or any project-internal types.
//
// History: previously these helpers lived inline in
// test_sortformer_streaming.cpp and test_sortformer_aosc_speakers.cpp.
// Pulling them up here avoids drift between two near-identical copies.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace parakeet_test {

// Repeats `src` until it covers `secs` seconds. A different gain per
// repetition makes each long-form window see different loudness, so a parity
// check exercises the global-CMVN guarantee (and attention locality) rather
// than feeding identical copies with identical statistics.
inline std::vector<float> tile_to_seconds(const std::vector<float> & src, int sample_rate, int secs) {
    if (src.empty() || secs <= 0) return src;
    static const float gains[] = {1.0f, 0.35f, 0.7f, 0.5f, 0.85f, 0.25f};
    const size_t n_gains = sizeof(gains) / sizeof(gains[0]);
    const size_t target = (size_t) sample_rate * (size_t) secs;
    std::vector<float> out;
    out.reserve(target + src.size());
    for (size_t tile = 0; out.size() < target; ++tile) {
        const float g = gains[tile % n_gains];
        for (float s : src) out.push_back(s * g);
    }
    return out;
}

inline std::vector<std::string> split_words(const std::string & s) {
    std::vector<std::string> words;
    std::istringstream is(s);
    std::string w;
    while (is >> w) words.push_back(w);
    return words;
}

// Word-level Levenshtein similarity in [0, 1]: 1.0 == identical token streams.
inline double word_similarity(const std::string & a, const std::string & b) {
    const std::vector<std::string> wa = split_words(a);
    const std::vector<std::string> wb = split_words(b);
    if (wa.empty() && wb.empty()) return 1.0;
    const size_t n = wa.size();
    const size_t m = wb.size();
    std::vector<size_t> prev(m + 1), cur(m + 1);
    for (size_t j = 0; j <= m; ++j) prev[j] = j;
    for (size_t i = 1; i <= n; ++i) {
        cur[0] = i;
        for (size_t j = 1; j <= m; ++j) {
            const size_t cost = (wa[i - 1] == wb[j - 1]) ? 0 : 1;
            cur[j] = std::min({ prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost });
        }
        std::swap(prev, cur);
    }
    const size_t dist = prev[m];
    const size_t denom = std::max<size_t>(1, std::max(n, m));
    return 1.0 - (double) dist / (double) denom;
}

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
