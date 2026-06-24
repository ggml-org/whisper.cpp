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

/* Output audio to nowhere... */

#include "SDL_timer.h"
#include "SDL_audio.h"
#include "../SDL_audio_c.h"
#include "SDL_ps2audio.h"

#include <kernel.h>
#include <malloc.h>
#include <audsrv.h>
#include <ps2_audio_driver.h>

/* The tag name used by PS2 audio */
#define PS2AUDIO_DRIVER_NAME "ps2"

static int PS2AUDIO_OpenDevice(_THIS, const char *devname)
{
    int i, mixlen;
    struct audsrv_fmt_t format;

    this->hidden = (struct SDL_PrivateAudioData *)
        SDL_malloc(sizeof(*this->hidden));
    if (!this->hidden) {
        return SDL_OutOfMemory();
    }
    SDL_zerop(this->hidden);

    /* These are the native supported audio PS2 configs  */
    switch (this->spec.freq) {
    case 11025:
    case 12000:
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

    this->spec.samples = 512;
    this->spec.channels = this->spec.channels == 1 ? 1 : 2;
    this->spec.format = this->spec.format == AUDIO_S8 ? AUDIO_S8 : AUDIO_S16;

    SDL_CalculateAudioSpec(&this->spec);

    format.bits = this->spec.format == AUDIO_S8 ? 8 : 16;
    format.freq = this->spec.freq;
    format.channels = this->spec.channels;

    this->hidden->channel = audsrv_set_format(&format);
    audsrv_set_volume(MAX_VOLUME);

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

static void PS2AUDIO_PlayDevice(_THIS)
{
    uint8_t *mixbuf = this->hidden->mixbufs[this->hidden->next_buffer];
    audsrv_play_audio((char *)mixbuf, this->spec.size);

    this->hidden->next_buffer = (this->hidden->next_buffer + 1) % NUM_BUFFERS;
}

/* This function waits until it is possible to write a full sound buffer */
static void PS2AUDIO_WaitDevice(_THIS)
{
    audsrv_wait_audio(this->spec.size);
}

static Uint8 *PS2AUDIO_GetDeviceBuf(_THIS)
{
    return this->hidden->mixbufs[this->hidden->next_buffer];
}

static void PS2AUDIO_CloseDevice(_THIS)
{
    if (this->hidden->channel >= 0) {
        audsrv_stop_audio();
        this->hidden->channel = -1;
    }

    if (this->hidden->rawbuf) {
        free(this->hidden->rawbuf);
        this->hidden->rawbuf = NULL;
    }
}

static void PS2AUDIO_ThreadInit(_THIS)
{
    /* Increase the priority of this audio thread by 1 to put it
       ahead of other SDL threads. */
    int32_t thid;
    ee_thread_status_t status;
    thid = GetThreadId();
    if (ReferThreadStatus(GetThreadId(), &status) == 0) {
        ChangeThreadPriority(thid, status.current_priority - 1);
    }
}

static void PS2AUDIO_Deinitialize(void)
{
    deinit_audio_driver();
}

static SDL_bool PS2AUDIO_Init(SDL_AudioDriverImpl *impl)
{
    if (init_audio_driver() < 0) {
        return SDL_FALSE;
    }

    /* Set the function pointers */
    impl->OpenDevice = PS2AUDIO_OpenDevice;
    impl->PlayDevice = PS2AUDIO_PlayDevice;
    impl->WaitDevice = PS2AUDIO_WaitDevice;
    impl->GetDeviceBuf = PS2AUDIO_GetDeviceBuf;
    impl->CloseDevice = PS2AUDIO_CloseDevice;
    impl->ThreadInit = PS2AUDIO_ThreadInit;
    impl->Deinitialize = PS2AUDIO_Deinitialize;
    impl->OnlyHasDefaultOutputDevice = SDL_TRUE;
    return SDL_TRUE; /* this audio target is available. */
}

AudioBootStrap PS2AUDIO_bootstrap = {
    "ps2", "PS2 audio driver", PS2AUDIO_Init, SDL_FALSE
};

/* vi: set ts=4 sw=4 expandtab: */
