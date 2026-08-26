#include "whisper.h"

#include <cstdio>

int main(void) {
    printf("[test-cmake] version: %s, build: %d (%s)\n",
           whisper_version(), WHISPER_BUILD_NUMBER, WHISPER_BUILD_COMMIT);
    return 0;
}
