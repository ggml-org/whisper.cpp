// Invoked by test_coreml_padding_runtime.py with generated, checkpoint-free models.
#include "coreml/parakeet-encoder.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static std::vector<float> read(const std::string & path, size_t count) {
    std::vector<float> data(count);
    std::ifstream file(path, std::ios::binary);
    file.read(reinterpret_cast<char *>(data.data()), count * sizeof(float));
    if (!file || file.peek() != std::char_traits<char>::eof()) return {};
    return data;
}

int main(int argc, char ** argv) {
    if (argc != 4) return 2;
    const std::string mode = argv[2];
    auto * ctx = parakeet_coreml_init(argv[1]);
    if (!ctx) return 3;
    if (parakeet_coreml_masks_mel_padding(ctx) != (mode == "masked")) return 4;
    std::ifstream cases(argv[3]);
    std::string input, reference;
    int frames, valid, features, rows, dim, tested = 0;
    float max_error = 0.0f;
    while (cases >> input >> reference >> frames >> valid >> features >> rows >> dim) {
        auto mel = read(input, size_t(frames) * features);
        auto expected = read(reference, size_t(rows) * dim);
        if (mel.empty() || expected.empty()) return 5;
        std::vector<float> actual(size_t(rows) * dim, NAN);
        int rc;
        if (mode == "bypass") {
            rc = parakeet_coreml_encode_embeddings(ctx, frames, dim, mel.data(), actual.data());
        } else if (mode == "legacy") {
            rc = parakeet_coreml_encode(ctx, frames, frames, features, mel.data(), rows, dim, actual.data());
        } else {
            // Invalid lengths must fail without writing past the caller's output.
            if (parakeet_coreml_encode_valid(ctx, frames, 0, features, mel.data(), rows,
                                             dim, actual.data()) == 0 ||
                parakeet_coreml_encode_valid(ctx, frames, frames + 1, features, mel.data(),
                                             rows, dim, actual.data()) == 0) return 6;
            rc = parakeet_coreml_encode_valid(ctx, frames, valid, features, mel.data(),
                                               rows, dim, actual.data());
            if (rc == 0) {
                // The upstream entry point accepts valid lengths too, with 0
                // meaning the full slab. It must agree with the strict API.
                std::vector<float> compatible(actual.size(), NAN);
                rc = parakeet_coreml_encode(ctx, frames, valid == frames ? 0 : valid,
                                             features, mel.data(), rows, dim, compatible.data());
                for (size_t i = 0; rc == 0 && i < actual.size(); ++i) {
                    if (!std::isfinite(compatible[i]) ||
                        std::abs(compatible[i] - actual[i]) > 0.008f) return 9;
                }
            }
        }
        if (rc != 0) { std::cerr << "encode rc=" << rc << '\n'; return 7; }
        for (size_t i = 0; i < actual.size(); ++i) {
            max_error = std::max(max_error, std::abs(actual[i] - expected[i]));
            if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > 0.008f) {
                std::cerr << mode << " case=" << tested << " index=" << i
                          << " actual=" << actual[i] << " expected=" << expected[i] << '\n';
                return 8;
            }
        }
        ++tested;
    }
    parakeet_coreml_free(ctx);
    if (tested == 0 || !cases.eof()) return 9;
    std::cout << mode << ": " << tested << " native wrapper cases passed, max_abs_error=" << max_error << "\n";
    return 0;
}
