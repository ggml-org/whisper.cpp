/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/
#include "../SDL_internal.h"

/* Get the name of the audio device we use for output */

#if defined(SDL_AUDIO_DRIVER_NETBSD) || defined(SDL_AUDIO_DRIVER_OSS) || defined(SDL_AUDIO_DRIVER_SUNAUDIO)

#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h> /* For close() */

#include "SDL_stdinc.h"
#include "SDL_audiodev_c.h"

#ifndef _PATH_DEV_DSP
#if defined(__NETBSD__) || defined(__OPENBSD__)
#define _PATH_DEV_DSP "/dev/audio"
#else
#define _PATH_DEV_DSP "/dev/dsp"
#endif
#endif
#ifndef _PATH_DEV_DSP24
#define _PATH_DEV_DSP24 "/dev/sound/dsp"
#endif
#ifndef _PATH_DEV_AUDIO
#define _PATH_DEV_AUDIO "/dev/audio"
#endif

static void test_device(const int iscapture, const char *fname, int flags, int (*test)(int fd))
{
    struct stat sb;
    if ((stat(fname, &sb) == 0) && (S_ISCHR(sb.st_mode))) {
        const int audio_fd = open(fname, flags | O_CLOEXEC, 0);
        if (audio_fd >= 0) {
            const int okay = test(audio_fd);
            close(audio_fd);
            if (okay) {
                static size_t dummyhandle = 0;
                dummyhandle++;
                SDL_assert(dummyhandle != 0);

                /* Note that spec is NULL; while we are opening the device
                 * endpoint here, the endpoint does not provide any mix format
                 * information,  making this information inaccessible at
                 * enumeration time
                 */
                SDL_AddAudioDevice(iscapture, fname, NULL, (void *)(uintptr_t)dummyhandle);
            }
        }
    }
}

static int test_stub(int fd)
{
    return 1;
}

static void SDL_EnumUnixAudioDevices_Internal(const int iscapture, const int classic, int (*test)(int))
{
    const int flags = iscapture ? OPEN_FLAGS_INPUT : OPEN_FLAGS_OUTPUT;
    const char *audiodev;
    char audiopath[1024];

    if (!test) {
        test = test_stub;
    }

    /* Figure out what our audio device is */
    audiodev = SDL_getenv("SDL_PATH_DSP");
    if (!audiodev) {
        audiodev = SDL_getenv("AUDIODEV");
    }
    if (!audiodev) {
        if (classic) {
            audiodev = _PATH_DEV_AUDIO;
        } else {
            struct stat sb;

            /* Added support for /dev/sound/\* in Linux 2.4 */
            if (((stat("/dev/sound", &sb) == 0) && S_ISDIR(sb.st_mode)) && ((stat(_PATH_DEV_DSP24, &sb) == 0) && S_ISCHR(sb.st_mode))) {
                audiodev = _PATH_DEV_DSP24;
            } else {
                audiodev = _PATH_DEV_DSP;
            }
        }
    }
    test_device(iscapture, audiodev, flags, test);

    if (SDL_strlen(audiodev) < (sizeof(audiopath) - 3)) {
        int instance = 0;
        while (instance <= 64) {
            (void)SDL_snprintf(audiopath, SDL_arraysize(audiopath),
                               "%s%d", audiodev, instance);
            instance++;
            test_device(iscapture, audiopath, flags, test);
        }
    }
}

void SDL_EnumUnixAudioDevices(const int classic, int (*test)(int))
{
    SDL_EnumUnixAudioDevices_Internal(SDL_TRUE, classic, test);
    SDL_EnumUnixAudioDevices_Internal(SDL_FALSE, classic, test);
}

#endif /* Audio driver selection */

/* vi: set ts=4 sw=4 expandtab: */
