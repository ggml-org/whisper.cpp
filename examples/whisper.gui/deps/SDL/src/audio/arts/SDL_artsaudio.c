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

#ifdef SDL_AUDIO_DRIVER_ARTS

/* Allow access to a raw mixing buffer */

#ifdef HAVE_SIGNAL_H
#include <signal.h>
#endif
#include <unistd.h>
#include <errno.h>

#include "SDL_timer.h"
#include "SDL_audio.h"
#include "../SDL_audio_c.h"
#include "SDL_artsaudio.h"

#ifdef SDL_AUDIO_DRIVER_ARTS_DYNAMIC
#include "SDL_name.h"
#include "SDL_loadso.h"
#else
#define SDL_NAME(X) X
#endif

#ifdef SDL_AUDIO_DRIVER_ARTS_DYNAMIC

static const char *arts_library = SDL_AUDIO_DRIVER_ARTS_DYNAMIC;
static void *arts_handle = NULL;

/* !!! FIXME: I hate this SDL_NAME clutter...it makes everything so messy! */
static int (*SDL_NAME(arts_init)) (void);
static void (*SDL_NAME(arts_free)) (void);
static arts_stream_t(*SDL_NAME(arts_play_stream)) (int rate, int bits,
                                                   int channels,
                                                   const char *name);
static int (*SDL_NAME(arts_stream_set)) (arts_stream_t s,
                                         arts_parameter_t param, int value);
static int (*SDL_NAME(arts_stream_get)) (arts_stream_t s,
                                         arts_parameter_t param);
static int (*SDL_NAME(arts_write)) (arts_stream_t s, const void *buffer,
                                    int count);
static void (*SDL_NAME(arts_close_stream)) (arts_stream_t s);
static int (*SDL_NAME(arts_suspend))(void);
static int (*SDL_NAME(arts_suspended)) (void);
static const char *(*SDL_NAME(arts_error_text)) (int errorcode);

#define SDL_ARTS_SYM(x) { #x, (void **) (char *) &SDL_NAME(x) }
static struct
{
    const char *name;
    void **func;
} arts_functions[] = {
/* *INDENT-OFF* */
    SDL_ARTS_SYM(arts_init),
    SDL_ARTS_SYM(arts_free),
    SDL_ARTS_SYM(arts_play_stream),
    SDL_ARTS_SYM(arts_stream_set),
    SDL_ARTS_SYM(arts_stream_get),
    SDL_ARTS_SYM(arts_write),
    SDL_ARTS_SYM(arts_close_stream),
    SDL_ARTS_SYM(arts_suspend),
    SDL_ARTS_SYM(arts_suspended),
    SDL_ARTS_SYM(arts_error_text),
/* *INDENT-ON* */
};

#undef SDL_ARTS_SYM

static void UnloadARTSLibrary(void)
{
    if (arts_handle) {
        SDL_UnloadObject(arts_handle);
        arts_handle = NULL;
    }
}

static int LoadARTSLibrary(void)
{
    int i, retval = -1;

    if (!arts_handle) {
        arts_handle = SDL_LoadObject(arts_library);
        if (arts_handle) {
            retval = 0;
            for (i = 0; i < SDL_arraysize(arts_functions); ++i) {
                *arts_functions[i].func =
                    SDL_LoadFunction(arts_handle, arts_functions[i].name);
                if (!*arts_functions[i].func) {
                    retval = -1;
                    UnloadARTSLibrary();
                    break;
                }
            }
        }
    }

    return retval;
}

#else

static void UnloadARTSLibrary(void)
{
    return;
}

static int LoadARTSLibrary(void)
{
    return 0;
}

#endif /* SDL_AUDIO_DRIVER_ARTS_DYNAMIC */

/* This function waits until it is possible to write a full sound buffer */
static void ARTS_WaitDevice(_THIS)
{
    Sint32 ticks;

    /* Check to see if the thread-parent process is still alive */
    {
        static int cnt = 0;
        /* Note that this only works with thread implementations
           that use a different process id for each thread.
         */
        /* Check every 10 loops */
        if (this->hidden->parent && (((++cnt) % 10) == 0)) {
            if (kill(this->hidden->parent, 0) < 0 && errno == ESRCH) {
                SDL_OpenedAudioDeviceDisconnected(this);
            }
        }
    }

    /* Use timer for general audio synchronization */
    ticks =
        ((Sint32) (this->hidden->next_frame - SDL_GetTicks())) - FUDGE_TICKS;
    if (ticks > 0) {
        SDL_Delay(ticks);
    }
}

static void ARTS_PlayDevice(_THIS)
{
    /* Write the audio data */
    int written = SDL_NAME(arts_write) (this->hidden->stream,
                                        this->hidden->mixbuf,
                                        this->hidden->mixlen);

    /* If timer synchronization is enabled, set the next write frame */
    if (this->hidden->frame_ticks) {
        this->hidden->next_frame += this->hidden->frame_ticks;
    }

    /* If we couldn't write, assume fatal error for now */
    if (written < 0) {
        SDL_OpenedAudioDeviceDisconnected(this);
    }
#ifdef DEBUG_AUDIO
    fprintf(stderr, "Wrote %d bytes of audio data\n", written);
#endif
}

static Uint8 *ARTS_GetDeviceBuf(_THIS)
{
    return (this->hidden->mixbuf);
}


static void ARTS_CloseDevice(_THIS)
{
    if (this->hidden->stream) {
        SDL_NAME(arts_close_stream) (this->hidden->stream);
    }
    SDL_NAME(arts_free) ();
    SDL_free(this->hidden->mixbuf);
    SDL_free(this->hidden);
}

static int ARTS_Suspend(void)
{
    const Uint32 abortms = SDL_GetTicks() + 3000; /* give up after 3 secs */
    while ( (!SDL_NAME(arts_suspended)()) && !SDL_TICKS_PASSED(SDL_GetTicks(), abortms) ) {
        if ( SDL_NAME(arts_suspend)() ) {
            break;
        }
    }
    return SDL_NAME(arts_suspended)();
}

static int ARTS_OpenDevice(_THIS, const char *devname)
{
    int rc = 0;
    int bits, frag_spec = 0;
    SDL_AudioFormat test_format = 0;

    /* Initialize all variables that we clean on shutdown */
    this->hidden = (struct SDL_PrivateAudioData *)SDL_malloc(sizeof(*this->hidden));
    if (!this->hidden) {
        return SDL_OutOfMemory();
    }
    SDL_zerop(this->hidden);

    /* Try for a closest match on audio format */
    for (test_format = SDL_FirstAudioFormat(this->spec.format); test_format; test_format = SDL_NextAudioFormat()) {
#ifdef DEBUG_AUDIO
        fprintf(stderr, "Trying format 0x%4.4x\n", test_format);
#endif
        switch (test_format) {
        case AUDIO_U8:
        case AUDIO_S16LSB:
            break;
        default:
            continue;
        }
        break;
    }
    if (!test_format) {
        return SDL_SetError("%s: Unsupported audio format", "arts");
    }
    this->spec.format = test_format;
    bits = SDL_AUDIO_BITSIZE(test_format);

    if ((rc = SDL_NAME(arts_init) ()) != 0) {
        return SDL_SetError("Unable to initialize ARTS: %s",
                            SDL_NAME(arts_error_text) (rc));
    }

    if (!ARTS_Suspend()) {
        return SDL_SetError("ARTS can not open audio device");
    }

    this->hidden->stream = SDL_NAME(arts_play_stream) (this->spec.freq,
                                                       bits,
                                                       this->spec.channels,
                                                       "SDL");

    /* Play nothing so we have at least one write (server bug workaround). */
    SDL_NAME(arts_write) (this->hidden->stream, "", 0);

    /* Calculate the final parameters for this audio specification */
    SDL_CalculateAudioSpec(&this->spec);

    /* Determine the power of two of the fragment size */
    for (frag_spec = 0; (0x01 << frag_spec) < this->spec.size; ++frag_spec);
    if ((0x01 << frag_spec) != this->spec.size) {
        return SDL_SetError("Fragment size must be a power of two");
    }
    frag_spec |= 0x00020000;    /* two fragments, for low latency */

#ifdef ARTS_P_PACKET_SETTINGS
    SDL_NAME(arts_stream_set) (this->hidden->stream,
                               ARTS_P_PACKET_SETTINGS, frag_spec);
#else
    SDL_NAME(arts_stream_set) (this->hidden->stream, ARTS_P_PACKET_SIZE,
                               frag_spec & 0xffff);
    SDL_NAME(arts_stream_set) (this->hidden->stream, ARTS_P_PACKET_COUNT,
                               frag_spec >> 16);
#endif
    this->spec.size = SDL_NAME(arts_stream_get) (this->hidden->stream,
                                                 ARTS_P_PACKET_SIZE);

    /* Allocate mixing buffer */
    this->hidden->mixlen = this->spec.size;
    this->hidden->mixbuf = (Uint8 *) SDL_malloc(this->hidden->mixlen);
    if (!this->hidden->mixbuf) {
        return SDL_OutOfMemory();
    }
    SDL_memset(this->hidden->mixbuf, this->spec.silence, this->spec.size);

    /* Get the parent process id (we're the parent of the audio thread) */
    this->hidden->parent = getpid();

    /* We're ready to rock and roll. :-) */
    return 0;
}


static void ARTS_Deinitialize(void)
{
    UnloadARTSLibrary();
}


static SDL_bool ARTS_Init(SDL_AudioDriverImpl *impl)
{
    if (LoadARTSLibrary() < 0) {
        return SDL_FALSE;
    } else {
        if (SDL_NAME(arts_init) () != 0) {
            UnloadARTSLibrary();
            SDL_SetError("ARTS: arts_init failed (no audio server?)");
            return SDL_FALSE;
        }

        /* Play a stream so aRts doesn't crash */
        if (ARTS_Suspend()) {
            arts_stream_t stream;
            stream = SDL_NAME(arts_play_stream) (44100, 16, 2, "SDL");
            SDL_NAME(arts_write) (stream, "", 0);
            SDL_NAME(arts_close_stream) (stream);
        }

        SDL_NAME(arts_free) ();
    }

    /* Set the function pointers */
    impl->OpenDevice = ARTS_OpenDevice;
    impl->PlayDevice = ARTS_PlayDevice;
    impl->WaitDevice = ARTS_WaitDevice;
    impl->GetDeviceBuf = ARTS_GetDeviceBuf;
    impl->CloseDevice = ARTS_CloseDevice;
    impl->Deinitialize = ARTS_Deinitialize;
    impl->OnlyHasDefaultOutputDevice = SDL_TRUE;

    return SDL_TRUE;   /* this audio target is available. */
}


AudioBootStrap ARTS_bootstrap = {
    "arts", "Analog RealTime Synthesizer", ARTS_Init, SDL_FALSE
};

#endif /* SDL_AUDIO_DRIVER_ARTS */

/* vi: set ts=4 sw=4 expandtab: */
