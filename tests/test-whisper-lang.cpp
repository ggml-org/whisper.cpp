#include "whisper.h"

#include <cassert>
#include <cstring>

int main() {
    assert(std::strcmp(whisper_lang_title(0), "english") == 0);
    assert(std::strcmp(whisper_lang_title(2), "german") == 0);
    assert(std::strcmp(whisper_lang_title(99), "cantonese") == 0);
    assert(whisper_lang_title(-1) == nullptr);
    assert(whisper_lang_title(whisper_lang_max_id() + 1) == nullptr);

    return 0;
}
