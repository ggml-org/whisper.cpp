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

#ifdef SDL_AUDIO_DRIVER_PSP

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
#include "SDL_pspaudio.h"

#include <pspaudio.h>
#include <pspthreadman.h>

/* The tag name used by PSP audio */
#define PSPAUDIO_DRIVER_NAME "psp"

static inline SDL_bool isBasicAudioConfig(const SDL_AudioSpec *spec)
{
    return spec->freq == 44100;
}

static int PSPAUDIO_OpenDevice(_THIS, const char *devname)
{
    int format, mixlen, i;

    this->hidden = (struct SDL_PrivateAudioData *)
        SDL_malloc(sizeof(*this->hidden));
    if (!this->hidden) {
        return SDL_OutOfMemory();
    }
    SDL_zerop(this->hidden);

    /* device only natively supports S16LSB */
    this->spec.format = AUDIO_S16LSB;

    /*  PSP has some limitations with the Audio. It fully supports 44.1KHz (Mono & Stereo),
        however with frequencies differents than 44.1KHz, it just supports Stereo,
        so a resampler must be done for these scenarios */
    if (isBasicAudioConfig(&this->spec)) {
        /* The sample count must be a multiple of 64. */
        this->spec.samples = PSP_AUDIO_SAMPLE_ALIGN(this->spec.samples);
        /* The number of channels (1 or 2). */
        this->spec.channels = this->spec.channels == 1 ? 1 : 2;
        format = this->spec.channels == 1 ? PSP_AUDIO_FORMAT_MONO : PSP_AUDIO_FORMAT_STEREO;
        this->hidden->channel = sceAudioChReserve(PSP_AUDIO_NEXT_CHANNEL, this->spec.samples, format);
    } else {
        /* 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11050, 8000 */
        switch (this->spec.freq) {
        case 8000:
        case 11025:
        case 12000:
        case 16000:
        case 22050:
        case 24000:
        case 32000:
        case 44100:
        case 48000:
            this->spec.freq = this->spec.freq;
            break;
        default:
            this->spec.freq = 48000;
            break;
        }
        /* The number of samples to output in one output call (min 17, max 4111). */
        this->spec.samples = this->spec.samples < 17 ? 17 : (this->spec.samples > 4111 ? 4111 : this->spec.samples);
        this->spec.channels = 2; /* we're forcing the hardware to stereo. */
        this->hidden->channel = sceAudioSRCChReserve(this->spec.samples, this->spec.freq, 2);
    }

    if (this->hidden->channel < 0) {
        free(this->hidden->rawbuf);
        this->hidden->rawbuf = NULL;
        return SDL_SetError("Couldn't reserve hardware channel");
    }

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

    SDL_memset(this->hidden->rawbuf, 0, mixlen);
    for (i = 0; i < NUM_BUFFERS; i++) {
        this->hidden->mixbufs[i] = &this->hidden->rawbuf[i * this->spec.size];
    }

    this->hidden->next_buffer = 0;
    return 0;
}

static void PSPAUDIO_PlayDevice(_THIS)
{
    Uint8 *mixbuf = this->hidden->mixbufs[this->hidden->next_buffer];
    if (!isBasicAudioConfig(&this->spec)) {
        SDL_assert(this->spec.channels == 2);
        sceAudioSRCOutputBlocking(PSP_AUDIO_VOLUME_MAX, mixbuf);
    } else {
        sceAudioOutputPannedBlocking(this->hidden->channel, PSP_AUDIO_VOLUME_MAX, PSP_AUDIO_VOLUME_MAX, mixbuf);
    }

    this->hidden->next_buffer = (this->hidden->next_buffer + 1) % NUM_BUFFERS;
}

/* This function waits until it is possible to write a full sound buffer */
static void PSPAUDIO_WaitDevice(_THIS)
{
    /* Because we block when sending audio, there's no need for this function to do anything. */
}

static Uint8 *PSPAUDIO_GetDeviceBuf(_THIS)
{
    return this->hidden->mixbufs[this->hidden->next_buffer];
}

static void PSPAUDIO_CloseDevice(_THIS)
{
    if (this->hidden->channel >= 0) {
        if (!isBasicAudioConfig(&this->spec)) {
            sceAudioSRCChRelease();
        } else {
            sceAudioChRelease(this->hidden->channel);
        }
        this->hidden->channel = -1;
    }

    if (this->hidden->rawbuf) {
        free(this->hidden->rawbuf);
        this->hidden->rawbuf = NULL;
    }
}

static void PSPAUDIO_ThreadInit(_THIS)
{
    /* Increase the priority of this audio thread by 1 to put it
       ahead of other SDL threads. */
    SceUID thid;
    SceKernelThreadInfo status;
    thid = sceKernelGetThreadId();
    status.size = sizeof(SceKernelThreadInfo);
    if (sceKernelReferThreadStatus(thid, &status) == 0) {
        sceKernelChangeThreadPriority(thid, status.currentPriority - 1);
    }
}

static SDL_bool PSPAUDIO_Init(SDL_AudioDriverImpl *impl)
{
    /* Set the function pointers */
    impl->OpenDevice = PSPAUDIO_OpenDevice;
    impl->PlayDevice = PSPAUDIO_PlayDevice;
    impl->WaitDevice = PSPAUDIO_WaitDevice;
    impl->GetDeviceBuf = PSPAUDIO_GetDeviceBuf;
    impl->CloseDevice = PSPAUDIO_CloseDevice;
    impl->ThreadInit = PSPAUDIO_ThreadInit;

    /* PSP audio device */
    impl->OnlyHasDefaultOutputDevice = SDL_TRUE;
    /*
    impl->HasCaptureSupport = SDL_TRUE;
    impl->OnlyHasDefaultCaptureDevice = SDL_TRUE;
    */
    return SDL_TRUE; /* this audio target is available. */
}

AudioBootStrap PSPAUDIO_bootstrap = {
    "psp", "PSP audio driver", PSPAUDIO_Init, SDL_FALSE
};

#endif /* SDL_AUDIO_DRIVER_PSP */

/* vi: set ts=4 sw=4 expandtab: */
