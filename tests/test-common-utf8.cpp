#include "common-whisper.h"

#include <cstdlib>
#include <cstdio>
#include <string>

static void expect_needed(const std::string & input, int expected) {
    const int actual = utf8_trailing_bytes_needed(input);
    if (actual != expected) {
        fprintf(stderr, "expected %d trailing UTF-8 bytes, got %d\n", expected, actual);
        std::abort();
    }
}

static void expect_sanitized(const std::string & input, const std::string & expected) {
    const std::string actual = utf8_sanitize(input);
    if (actual != expected) {
        fprintf(stderr, "utf8_sanitize: expected %zu bytes, got %zu\n", expected.size(), actual.size());
        std::abort();
    }
}

int main() {
    expect_needed("", 0);
    expect_needed("plain ascii", 0);

    const std::string cjk = "\xE4\xBD\xA0"; // U+4F60
    expect_needed(cjk.substr(0, 1), 2);
    expect_needed(cjk.substr(0, 2), 1);
    expect_needed(cjk, 0);

    const std::string emoji = "\xF0\x9F\x98\x80"; // U+1F600
    expect_needed(emoji.substr(0, 1), 3);
    expect_needed(emoji.substr(0, 2), 2);
    expect_needed(emoji.substr(0, 3), 1);
    expect_needed(emoji, 0);

    expect_needed("\x80\x80", 0);
    expect_needed("\xFF", 0);

    // utf8_sanitize: valid input is preserved byte-for-byte
    expect_sanitized("", "");
    expect_sanitized("plain ascii", "plain ascii");
    expect_sanitized(cjk, cjk);
    expect_sanitized(emoji, emoji);
    expect_sanitized("hi " + cjk + "!", "hi " + cjk + "!");

    // issue #3760: a lone UTF-8 lead byte emitted as a whole segment's text
    expect_sanitized("\xC3", "");
    expect_sanitized("\xC5", "");
    // a valid prefix followed by a trailing incomplete lead byte
    expect_sanitized("A\xC3", "A");
    expect_sanitized(cjk + "\xC3", cjk);
    // orphan continuation byte (the split character's tail in the next segment)
    expect_sanitized("\x80", "");
    expect_sanitized("\xBF next", " next");
    // truncated multi-byte sequences
    expect_sanitized(emoji.substr(0, 3), "");
    expect_sanitized(cjk.substr(0, 2), "");
    // invalid lead bytes and overlong / surrogate encodings
    expect_sanitized("\xFF", "");
    expect_sanitized("\xC0\x80", "");         // overlong NUL
    expect_sanitized("\xC1\xBF", "");         // overlong
    expect_sanitized("\xED\xA0\x80", "");     // UTF-16 surrogate U+D800
    expect_sanitized("\xF4\x90\x80\x80", "");  // > U+10FFFF

    return 0;
}
