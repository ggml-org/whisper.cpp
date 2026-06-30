#pragma once

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#ifndef TRANSCRIPTION_SIMILARITY_THRESHOLD
#define TRANSCRIPTION_SIMILARITY_THRESHOLD 1.0
#endif

static std::string read_expected_transcription(const char * path) {
    std::ifstream fin(path);
    assert(fin.is_open());

    std::string text(
        (std::istreambuf_iterator<char>(fin)),
         std::istreambuf_iterator<char>());

    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
        text.pop_back();
    }

    return text;
}

static std::vector<std::string> transcription_words(const std::string & text) {
    std::vector<std::string> words;
    std::string word;

    for (unsigned char ch : text) {
        if (std::isalnum(ch)) {
            word.push_back((char) std::tolower(ch));
        } else if (!word.empty()) {
            words.push_back(word);
            word.clear();
        }
    }

    if (!word.empty()) {
        words.push_back(word);
    }

    return words;
}

static double transcription_lcs_similarity(const std::string & expected, const std::string & actual) {
    const std::vector<std::string> expected_words = transcription_words(expected);
    const std::vector<std::string> actual_words   = transcription_words(actual);

    if (expected_words.empty() && actual_words.empty()) {
        return 1.0;
    }

    if (expected_words.empty() || actual_words.empty()) {
        return 0.0;
    }

    std::vector<int> prev(actual_words.size() + 1, 0);
    std::vector<int> cur (actual_words.size() + 1, 0);

    for (size_t i = 0; i < expected_words.size(); ++i) {
        std::fill(cur.begin(), cur.end(), 0);

        for (size_t j = 0; j < actual_words.size(); ++j) {
            if (expected_words[i] == actual_words[j]) {
                cur[j + 1] = prev[j] + 1;
            } else {
                cur[j + 1] = std::max(prev[j + 1], cur[j]);
            }
        }

        prev.swap(cur);
    }

    const int lcs = prev[actual_words.size()];
    return (2.0 * lcs) / (expected_words.size() + actual_words.size());
}

static bool verify_transcription(const std::string & expected, const std::string & actual) {
    const double threshold = TRANSCRIPTION_SIMILARITY_THRESHOLD;

    if (threshold >= 1.0) {
        if (actual == expected) {
            return true;
        }

        fprintf(stderr, "\n\n");
        fprintf(stderr, "[Failed] Transcript mismatched\n");
        fprintf(stderr, "expected:\n%s\n\n", expected.c_str());
        fprintf(stderr, "actual:\n%s\n", actual.c_str());
        return false;
    }

    const double similarity = transcription_lcs_similarity(expected, actual);
    printf("\nTranscript similarity: %.6f (threshold %.6f)\n", similarity, threshold);

    if (similarity >= threshold) {
        return true;
    }

    fprintf(stderr, "\n\nTranscript similarity below threshold: %.6f < %.6f\n", similarity, threshold);
    fprintf(stderr, "Expected:\n%s\n\n", expected.c_str());
    fprintf(stderr, "Actual:\n%s\n", actual.c_str());
    return false;
}
