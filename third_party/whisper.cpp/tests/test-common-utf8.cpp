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

    return 0;
}
