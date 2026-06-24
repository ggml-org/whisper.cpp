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

#ifdef SDL_AUDIO_DRIVER_N3DS

#include "SDL_audio.h"

/* N3DS Audio driver */

#include "../SDL_sysaudio.h"
#include "SDL_n3dsaudio.h"
#include "SDL_timer.h"

#define N3DSAUDIO_DRIVER_NAME "n3ds"

static dspHookCookie dsp_hook;
static SDL_AudioDevice *audio_device;

static void FreePrivateData(_THIS);
static int FindAudioFormat(_THIS);

/* fully local functions related to the wavebufs / DSP, not the same as the SDL-wide mixer lock */
static SDL_INLINE void contextLock(_THIS)
{
    LightLock_Lock(&this->hidden->lock);
}

static SDL_INLINE void contextUnlock(_THIS)
{
    LightLock_Unlock(&this->hidden->lock);
}

static void N3DSAUD_DspHook(DSP_HookType hook)
{
    if (hook == DSPHOOK_ONCANCEL) {
        contextLock(audio_device);
        audio_device->hidden->isCancelled = SDL_TRUE;
        SDL_AtomicSet(&audio_device->enabled, SDL_FALSE);
        CondVar_Broadcast(&audio_device->hidden->cv);
        contextUnlock(audio_device);
    }
}

static void AudioFrameFinished(void *device)
{
    bool shouldBroadcast = false;
    unsigned i;
    SDL_AudioDevice *this = (SDL_AudioDevice *)device;

    contextLock(this);

    for (i = 0; i < NUM_BUFFERS; i++) {
        if (this->hidden->waveBuf[i].status == NDSP_WBUF_DONE) {
            this->hidden->waveBuf[i].status = NDSP_WBUF_FREE;
            shouldBroadcast = SDL_TRUE;
        }
    }

    if (shouldBroadcast) {
        CondVar_Broadcast(&this->hidden->cv);
    }

    contextUnlock(this);
}

static int N3DSAUDIO_OpenDevice(_THIS, const char *devname)
{
    Result ndsp_init_res;
    Uint8 *data_vaddr;
    float mix[12];
    this->hidden = (struct SDL_PrivateAudioData *)SDL_calloc(1, sizeof(*this->hidden));

    if (!this->hidden) {
        return SDL_OutOfMemory();
    }

    /* Initialise the DSP service */
    ndsp_init_res = ndspInit();
    if (R_FAILED(ndsp_init_res)) {
        if ((R_SUMMARY(ndsp_init_res) == RS_NOTFOUND) && (R_MODULE(ndsp_init_res) == RM_DSP)) {
            SDL_SetError("DSP init failed: dspfirm.cdc missing!");
        } else {
            SDL_SetError("DSP init failed. Error code: 0x%lX", ndsp_init_res);
        }
        return -1;
    }

    /* Initialise internal state */
    LightLock_Init(&this->hidden->lock);
    CondVar_Init(&this->hidden->cv);

    if (this->spec.channels > 2) {
        this->spec.channels = 2;
    }

    /* Should not happen but better be safe. */
    if (FindAudioFormat(this) < 0) {
        return SDL_SetError("No supported audio format found.");
    }

    /* Update the fragment size as size in bytes */
    SDL_CalculateAudioSpec(&this->spec);

    /* Allocate mixing buffer */
    if (this->spec.size >= SDL_MAX_UINT32 / 2) {
        return SDL_SetError("Mixing buffer is too large.");
    }

    this->hidden->mixlen = this->spec.size;
    this->hidden->mixbuf = (Uint8 *)SDL_malloc(this->spec.size);
    if (!this->hidden->mixbuf) {
        return SDL_OutOfMemory();
    }

    SDL_memset(this->hidden->mixbuf, this->spec.silence, this->spec.size);

    data_vaddr = (Uint8 *)linearAlloc(this->hidden->mixlen * NUM_BUFFERS);
    if (!data_vaddr) {
        return SDL_OutOfMemory();
    }

    SDL_memset(data_vaddr, 0, this->hidden->mixlen * NUM_BUFFERS);
    DSP_FlushDataCache(data_vaddr, this->hidden->mixlen * NUM_BUFFERS);

    this->hidden->nextbuf = 0;
    this->hidden->channels = this->spec.channels;
    this->hidden->samplerate = this->spec.freq;

    ndspChnReset(0);

    ndspChnSetInterp(0, NDSP_INTERP_LINEAR);
    ndspChnSetRate(0, this->spec.freq);
    ndspChnSetFormat(0, this->hidden->format);

    SDL_memset(mix, 0, sizeof(mix));
    mix[0] = 1.0;
    mix[1] = 1.0;
    ndspChnSetMix(0, mix);

    SDL_memset(this->hidden->waveBuf, 0, sizeof(ndspWaveBuf) * NUM_BUFFERS);

    for (unsigned i = 0; i < NUM_BUFFERS; i++) {
        this->hidden->waveBuf[i].data_vaddr = data_vaddr;
        this->hidden->waveBuf[i].nsamples = this->hidden->mixlen / this->hidden->bytePerSample;
        data_vaddr += this->hidden->mixlen;
    }

    /* Setup callback */
    audio_device = this;
    ndspSetCallback(AudioFrameFinished, this);
    dspHook(&dsp_hook, N3DSAUD_DspHook);

    return 0;
}

static int N3DSAUDIO_CaptureFromDevice(_THIS, void *buffer, int buflen)
{
    /* Delay to make this sort of simulate real audio input. */
    SDL_Delay((this->spec.samples * 1000) / this->spec.freq);

    /* always return a full buffer of silence. */
    SDL_memset(buffer, this->spec.silence, buflen);
    return buflen;
}

static void N3DSAUDIO_PlayDevice(_THIS)
{
    size_t nextbuf;
    size_t sampleLen;

    contextLock(this);

    nextbuf = this->hidden->nextbuf;
    sampleLen = this->hidden->mixlen;

    if (this->hidden->isCancelled ||
        this->hidden->waveBuf[nextbuf].status != NDSP_WBUF_FREE) {
        contextUnlock(this);
        return;
    }

    this->hidden->nextbuf = (nextbuf + 1) % NUM_BUFFERS;

    contextUnlock(this);

    memcpy((void *)this->hidden->waveBuf[nextbuf].data_vaddr,
           this->hidden->mixbuf, sampleLen);
    DSP_FlushDataCache(this->hidden->waveBuf[nextbuf].data_vaddr, sampleLen);

    ndspChnWaveBufAdd(0, &this->hidden->waveBuf[nextbuf]);
}

static void N3DSAUDIO_WaitDevice(_THIS)
{
    contextLock(this);
    while (!this->hidden->isCancelled &&
           this->hidden->waveBuf[this->hidden->nextbuf].status != NDSP_WBUF_FREE) {
        CondVar_Wait(&this->hidden->cv, &this->hidden->lock);
    }
    contextUnlock(this);
}

static Uint8 *N3DSAUDIO_GetDeviceBuf(_THIS)
{
    return this->hidden->mixbuf;
}

static void N3DSAUDIO_CloseDevice(_THIS)
{
    contextLock(this);

    dspUnhook(&dsp_hook);
    ndspSetCallback(NULL, NULL);

    if (!this->hidden->isCancelled) {
        ndspChnReset(0);
        memset(this->hidden->waveBuf, 0, sizeof(ndspWaveBuf) * NUM_BUFFERS);
        CondVar_Broadcast(&this->hidden->cv);
    }

    contextUnlock(this);

    ndspExit();

    FreePrivateData(this);
}

static void N3DSAUDIO_ThreadInit(_THIS)
{
    s32 current_priority = 0x30;
    svcGetThreadPriority(&current_priority, CUR_THREAD_HANDLE);
    current_priority--;
    /* 0x18 is reserved for video, 0x30 is the default for main thread */
    current_priority = SDL_clamp(current_priority, 0x19, 0x2F);
    svcSetThreadPriority(CUR_THREAD_HANDLE, current_priority);
}

static SDL_bool N3DSAUDIO_Init(SDL_AudioDriverImpl *impl)
{
    /* Set the function pointers */
    impl->OpenDevice = N3DSAUDIO_OpenDevice;
    impl->PlayDevice = N3DSAUDIO_PlayDevice;
    impl->WaitDevice = N3DSAUDIO_WaitDevice;
    impl->GetDeviceBuf = N3DSAUDIO_GetDeviceBuf;
    impl->CloseDevice = N3DSAUDIO_CloseDevice;
    impl->ThreadInit = N3DSAUDIO_ThreadInit;
    impl->OnlyHasDefaultOutputDevice = SDL_TRUE;

    /* Should be possible, but micInit would fail */
    impl->HasCaptureSupport = SDL_FALSE;
    impl->CaptureFromDevice = N3DSAUDIO_CaptureFromDevice;

    return SDL_TRUE; /* this audio target is available. */
}

AudioBootStrap N3DSAUDIO_bootstrap = {
    N3DSAUDIO_DRIVER_NAME,
    "SDL N3DS audio driver",
    N3DSAUDIO_Init,
    0
};

/**
 * Cleans up all allocated memory, safe to call with null pointers
 */
static void FreePrivateData(_THIS)
{
    if (!this->hidden) {
        return;
    }

    if (this->hidden->waveBuf[0].data_vaddr) {
        linearFree((void *)this->hidden->waveBuf[0].data_vaddr);
    }

    if (this->hidden->mixbuf) {
        SDL_free(this->hidden->mixbuf);
        this->hidden->mixbuf = NULL;
    }

    SDL_free(this->hidden);
    this->hidden = NULL;
}

static int FindAudioFormat(_THIS)
{
    SDL_bool found_valid_format = SDL_FALSE;
    Uint16 test_format = SDL_FirstAudioFormat(this->spec.format);

    while (!found_valid_format && test_format) {
        this->spec.format = test_format;
        switch (test_format) {
        case AUDIO_S8:
            /* Signed 8-bit audio supported */
            this->hidden->format = (this->spec.channels == 2) ? NDSP_FORMAT_STEREO_PCM8 : NDSP_FORMAT_MONO_PCM8;
            this->hidden->isSigned = 1;
            this->hidden->bytePerSample = this->spec.channels;
            found_valid_format = SDL_TRUE;
            break;
        case AUDIO_S16:
            /* Signed 16-bit audio supported */
            this->hidden->format = (this->spec.channels == 2) ? NDSP_FORMAT_STEREO_PCM16 : NDSP_FORMAT_MONO_PCM16;
            this->hidden->isSigned = 1;
            this->hidden->bytePerSample = this->spec.channels * 2;
            found_valid_format = SDL_TRUE;
            break;
        default:
            test_format = SDL_NextAudioFormat();
            break;
        }
    }

    return found_valid_format ? 0 : -1;
}

#endif /* SDL_AUDIO_DRIVER_N3DS */

/* vi: set sts=4 ts=4 sw=4 expandtab: */
