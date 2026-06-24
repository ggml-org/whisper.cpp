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
#include "../../SDL_internal.h"

#ifdef SDL_AUDIO_DRIVER_DISK

/* Output raw audio data to a file. */

#ifdef HAVE_STDIO_H
#include <stdio.h>
#endif

#include "SDL_rwops.h"
#include "SDL_timer.h"
#include "SDL_audio.h"
#include "../SDL_audio_c.h"
#include "SDL_diskaudio.h"

/* !!! FIXME: these should be SDL hints, not environment variables. */
/* environment variables and defaults. */
#define DISKENVR_OUTFILE    "SDL_DISKAUDIOFILE"
#define DISKDEFAULT_OUTFILE "sdlaudio.raw"
#define DISKENVR_INFILE     "SDL_DISKAUDIOFILEIN"
#define DISKDEFAULT_INFILE  "sdlaudio-in.raw"
#define DISKENVR_IODELAY    "SDL_DISKAUDIODELAY"

/* This function waits until it is possible to write a full sound buffer */
static void DISKAUDIO_WaitDevice(_THIS)
{
    SDL_Delay(_this->hidden->io_delay);
}

static void DISKAUDIO_PlayDevice(_THIS)
{
    const size_t written = SDL_RWwrite(_this->hidden->io,
                                       _this->hidden->mixbuf,
                                       1, _this->spec.size);

    /* If we couldn't write, assume fatal error for now */
    if (written != _this->spec.size) {
        SDL_OpenedAudioDeviceDisconnected(_this);
    }
#ifdef DEBUG_AUDIO
    fprintf(stderr, "Wrote %d bytes of audio data\n", written);
#endif
}

static Uint8 *DISKAUDIO_GetDeviceBuf(_THIS)
{
    return _this->hidden->mixbuf;
}

static int DISKAUDIO_CaptureFromDevice(_THIS, void *buffer, int buflen)
{
    struct SDL_PrivateAudioData *h = _this->hidden;
    const int origbuflen = buflen;

    SDL_Delay(h->io_delay);

    if (h->io) {
        const size_t br = SDL_RWread(h->io, buffer, 1, buflen);
        buflen -= (int)br;
        buffer = ((Uint8 *)buffer) + br;
        if (buflen > 0) { /* EOF (or error, but whatever). */
            SDL_RWclose(h->io);
            h->io = NULL;
        }
    }

    /* if we ran out of file, just write silence. */
    SDL_memset(buffer, _this->spec.silence, buflen);

    return origbuflen;
}

static void DISKAUDIO_FlushCapture(_THIS)
{
    /* no op...we don't advance the file pointer or anything. */
}

static void DISKAUDIO_CloseDevice(_THIS)
{
    if (_this->hidden->io) {
        SDL_RWclose(_this->hidden->io);
    }
    SDL_free(_this->hidden->mixbuf);
    SDL_free(_this->hidden);
}

static const char *get_filename(const SDL_bool iscapture, const char *devname)
{
    if (!devname) {
        devname = SDL_getenv(iscapture ? DISKENVR_INFILE : DISKENVR_OUTFILE);
        if (!devname) {
            devname = iscapture ? DISKDEFAULT_INFILE : DISKDEFAULT_OUTFILE;
        }
    }
    return devname;
}

static int DISKAUDIO_OpenDevice(_THIS, const char *devname)
{
    void *handle = _this->handle;
    /* handle != NULL means "user specified the placeholder name on the fake detected device list" */
    SDL_bool iscapture = _this->iscapture;
    const char *fname = get_filename(iscapture, handle ? NULL : devname);
    const char *envr = SDL_getenv(DISKENVR_IODELAY);

    _this->hidden = (struct SDL_PrivateAudioData *)
        SDL_malloc(sizeof(*_this->hidden));
    if (!_this->hidden) {
        return SDL_OutOfMemory();
    }
    SDL_zerop(_this->hidden);

    if (envr) {
        _this->hidden->io_delay = SDL_atoi(envr);
    } else {
        _this->hidden->io_delay = ((_this->spec.samples * 1000) / _this->spec.freq);
    }

    /* Open the audio device */
    _this->hidden->io = SDL_RWFromFile(fname, iscapture ? "rb" : "wb");
    if (!_this->hidden->io) {
        return -1;
    }

    /* Allocate mixing buffer */
    if (!iscapture) {
        _this->hidden->mixbuf = (Uint8 *)SDL_malloc(_this->spec.size);
        if (!_this->hidden->mixbuf) {
            return SDL_OutOfMemory();
        }
        SDL_memset(_this->hidden->mixbuf, _this->spec.silence, _this->spec.size);
    }

    SDL_LogCritical(SDL_LOG_CATEGORY_AUDIO,
                    "You are using the SDL disk i/o audio driver!\n");
    SDL_LogCritical(SDL_LOG_CATEGORY_AUDIO,
                    " %s file [%s].\n", iscapture ? "Reading from" : "Writing to",
                    fname);

    /* We're ready to rock and roll. :-) */
    return 0;
}

static void DISKAUDIO_DetectDevices(void)
{
    SDL_AddAudioDevice(SDL_FALSE, DEFAULT_OUTPUT_DEVNAME, NULL, (void *)0x1);
    SDL_AddAudioDevice(SDL_TRUE, DEFAULT_INPUT_DEVNAME, NULL, (void *)0x2);
}

static SDL_bool DISKAUDIO_Init(SDL_AudioDriverImpl *impl)
{
    /* Set the function pointers */
    impl->OpenDevice = DISKAUDIO_OpenDevice;
    impl->WaitDevice = DISKAUDIO_WaitDevice;
    impl->PlayDevice = DISKAUDIO_PlayDevice;
    impl->GetDeviceBuf = DISKAUDIO_GetDeviceBuf;
    impl->CaptureFromDevice = DISKAUDIO_CaptureFromDevice;
    impl->FlushCapture = DISKAUDIO_FlushCapture;

    impl->CloseDevice = DISKAUDIO_CloseDevice;
    impl->DetectDevices = DISKAUDIO_DetectDevices;

    impl->AllowsArbitraryDeviceNames = SDL_TRUE;
    impl->HasCaptureSupport = SDL_TRUE;
    impl->SupportsNonPow2Samples = SDL_TRUE;

    return SDL_TRUE; /* this audio target is available. */
}

AudioBootStrap DISKAUDIO_bootstrap = {
    "disk", "direct-to-disk audio", DISKAUDIO_Init, SDL_TRUE
};

#endif /* SDL_AUDIO_DRIVER_DISK */

/* vi: set ts=4 sw=4 expandtab: */
