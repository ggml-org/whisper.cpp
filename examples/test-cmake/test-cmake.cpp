#include "whisper.h"
#include "parakeet.h"

#include <cstdio>

int main(void) {
    printf("[test-cmake] version: %s, build: %d (%s)\n",
           whisper_version(), WHISPER_BUILD_NUMBER, WHISPER_BUILD_COMMIT);

    printf("[test-cmake] parakeet version: %s, build: %d (%s)\n",
           parakeet_version(), WHISPER_BUILD_NUMBER, WHISPER_BUILD_COMMIT);
    return 0;
}
