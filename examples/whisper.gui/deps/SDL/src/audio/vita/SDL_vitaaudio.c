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

#ifdef SDL_AUDIO_DRIVER_VITA

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <malloc.h> /* memalign() */

#include "SDL_audio.h"
#include "SDL_error.h"
#include "SDL_timer.h"
#include "../SDL_audio_c.h"
#include "../SDL_audiodev_c.h"
#include "../SDL_sysaudio.h"
#include "SDL_vitaaudio.h"

#include <psp2/kernel/threadmgr.h>
#include <psp2/audioout.h>
#include <psp2/audioin.h>

#define SCE_AUDIO_SAMPLE_ALIGN(s) (((s) + 63) & ~63)
#define SCE_AUDIO_MAX_VOLUME      0x8000

static int VITAAUD_OpenCaptureDevice(_THIS)
{
    this->spec.freq = 16000;
    this->spec.samples = 512;
    this->spec.channels = 1;

    SDL_CalculateAudioSpec(&this->spec);

    this->hidden->port = sceAudioInOpenPort(SCE_AUDIO_IN_PORT_TYPE_VOICE, 512, 16000, SCE_AUDIO_IN_PARAM_FORMAT_S16_MONO);

    if (this->hidden->port < 0) {
        return SDL_SetError("Couldn't open audio in port: %x", this->hidden->port);
    }

    return 0;
}

static int VITAAUD_OpenDevice(_THIS, const char *devname)
{
    int format, mixlen, i, port = SCE_AUDIO_OUT_PORT_TYPE_MAIN;
    int vols[2] = { SCE_AUDIO_MAX_VOLUME, SCE_AUDIO_MAX_VOLUME };
    SDL_AudioFormat test_format;

    this->hidden = (struct SDL_PrivateAudioData *)
        SDL_malloc(sizeof(*this->hidden));
    if (!this->hidden) {
        return SDL_OutOfMemory();
    }
    SDL_memset(this->hidden, 0, sizeof(*this->hidden));

    for (test_format = SDL_FirstAudioFormat(this->spec.format); test_format; test_format = SDL_NextAudioFormat()) {
        if (test_format == AUDIO_S16LSB) {
            this->spec.format = test_format;
            break;
        }
    }

    if (!test_format) {
        return SDL_SetError("Unsupported audio format");
    }

    if (this->iscapture) {
        return VITAAUD_OpenCaptureDevice(this);
    }

    /* The sample count must be a multiple of 64. */
    this->spec.samples = SCE_AUDIO_SAMPLE_ALIGN(this->spec.samples);

    /* Update the fragment size as size in bytes. */
    SDL_CalculateAudioSpec(&this->spec);

    /* Allocate the mixing buffer.  Its size and starting address must
       be a multiple of 64 bytes.  Our sample count is already a multiple of
       64, so spec->size should be a multiple of 64 as well. */
    mixlen = this->spec.size * NUM_BUFFERS;
    this->hidden->rawbuf = (Uint8 *)memalign(64, mixlen);
    if (!this->hidden->rawbuf) {
        return SDL_SetError("Couldn't allocate mixing buffer");
    }

    /* Setup the hardware channel. */
    if (this->spec.channels == 1) {
        format = SCE_AUDIO_OUT_MODE_MONO;
    } else {
        format = SCE_AUDIO_OUT_MODE_STEREO;
    }

    if (this->spec.freq < 48000) {
        port = SCE_AUDIO_OUT_PORT_TYPE_BGM;
    }

    this->hidden->port = sceAudioOutOpenPort(port, this->spec.samples, this->spec.freq, format);
    if (this->hidden->port < 0) {
        free(this->hidden->rawbuf);
        this->hidden->rawbuf = NULL;
        return SDL_SetError("Couldn't open audio out port: %x", this->hidden->port);
    }

    sceAudioOutSetVolume(this->hidden->port, SCE_AUDIO_VOLUME_FLAG_L_CH | SCE_AUDIO_VOLUME_FLAG_R_CH, vols);

    SDL_memset(this->hidden->rawbuf, 0, mixlen);
    for (i = 0; i < NUM_BUFFERS; i++) {
        this->hidden->mixbufs[i] = &this->hidden->rawbuf[i * this->spec.size];
    }

    this->hidden->next_buffer = 0;
    return 0;
}

static void VITAAUD_PlayDevice(_THIS)
{
    Uint8 *mixbuf = this->hidden->mixbufs[this->hidden->next_buffer];

    sceAudioOutOutput(this->hidden->port, mixbuf);

    this->hidden->next_buffer = (this->hidden->next_buffer + 1) % NUM_BUFFERS;
}

/* This function waits until it is possible to write a full sound buffer */
static void VITAAUD_WaitDevice(_THIS)
{
    /* Because we block when sending audio, there's no need for this function to do anything. */
}

static Uint8 *VITAAUD_GetDeviceBuf(_THIS)
{
    return this->hidden->mixbufs[this->hidden->next_buffer];
}

static void VITAAUD_CloseDevice(_THIS)
{
    if (this->hidden->port >= 0) {
        if (this->iscapture) {
            sceAudioInReleasePort(this->hidden->port);
        } else {
            sceAudioOutReleasePort(this->hidden->port);
        }
        this->hidden->port = -1;
    }

    if (!this->iscapture && this->hidden->rawbuf) {
        free(this->hidden->rawbuf); /* this uses memalign(), not SDL_malloc(). */
        this->hidden->rawbuf = NULL;
    }
}

static int VITAAUD_CaptureFromDevice(_THIS, void *buffer, int buflen)
{
    int ret;
    SDL_assert(buflen == this->spec.size);
    ret = sceAudioInInput(this->hidden->port, buffer);
    if (ret < 0) {
        return SDL_SetError("Failed to capture from device: %x", ret);
    }
    return this->spec.size;
}

static void VITAAUD_ThreadInit(_THIS)
{
    /* Increase the priority of this audio thread by 1 to put it
       ahead of other SDL threads. */
    SceUID thid;
    SceKernelThreadInfo info;
    thid = sceKernelGetThreadId();
    info.size = sizeof(SceKernelThreadInfo);
    if (sceKernelGetThreadInfo(thid, &info) == 0) {
        sceKernelChangeThreadPriority(thid, info.currentPriority - 1);
    }
}

static SDL_bool VITAAUD_Init(SDL_AudioDriverImpl *impl)
{
    /* Set the function pointers */
    impl->OpenDevice = VITAAUD_OpenDevice;
    impl->PlayDevice = VITAAUD_PlayDevice;
    impl->WaitDevice = VITAAUD_WaitDevice;
    impl->GetDeviceBuf = VITAAUD_GetDeviceBuf;
    impl->CloseDevice = VITAAUD_CloseDevice;
    impl->ThreadInit = VITAAUD_ThreadInit;

    impl->CaptureFromDevice = VITAAUD_CaptureFromDevice;

    /* and the capabilities */
    impl->HasCaptureSupport = SDL_TRUE;
    impl->OnlyHasDefaultOutputDevice = SDL_TRUE;
    impl->OnlyHasDefaultCaptureDevice = SDL_TRUE;

    return SDL_TRUE; /* this audio target is available. */
}

AudioBootStrap VITAAUD_bootstrap = {
    "vita", "VITA audio driver", VITAAUD_Init, SDL_FALSE
};

#endif /* SDL_AUDIO_DRIVER_VITA */

/* vi: set ts=4 sw=4 expandtab: */
