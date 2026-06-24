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

#ifdef SDL_AUDIO_DRIVER_WASAPI

#include "../../core/windows/SDL_windows.h"
#include "../../core/windows/SDL_immdevice.h"
#include "SDL_audio.h"
#include "SDL_timer.h"
#include "../SDL_audio_c.h"
#include "../SDL_sysaudio.h"

#define COBJMACROS
#include <audioclient.h>

#include "SDL_wasapi.h"

/* These constants aren't available in older SDKs */
#ifndef AUDCLNT_STREAMFLAGS_RATEADJUST
#define AUDCLNT_STREAMFLAGS_RATEADJUST 0x00100000
#endif
#ifndef AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY
#define AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY 0x08000000
#endif
#ifndef AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM
#define AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM 0x80000000
#endif

/* Some GUIDs we need to know without linking to libraries that aren't available before Vista. */
static const IID SDL_IID_IAudioRenderClient = { 0xf294acfc, 0x3146, 0x4483, { 0xa7, 0xbf, 0xad, 0xdc, 0xa7, 0xc2, 0x60, 0xe2 } };
static const IID SDL_IID_IAudioCaptureClient = { 0xc8adbd64, 0xe71e, 0x48a0, { 0xa4, 0xde, 0x18, 0x5c, 0x39, 0x5c, 0xd3, 0x17 } };


static void WASAPI_DetectDevices(void)
{
    WASAPI_EnumerateEndpoints();
}

static SDL_INLINE SDL_bool WasapiFailed(_THIS, const HRESULT err)
{
    if (err == S_OK) {
        return SDL_FALSE;
    }

    if (err == AUDCLNT_E_DEVICE_INVALIDATED) {
        this->hidden->device_lost = SDL_TRUE;
    } else if (SDL_AtomicGet(&this->enabled)) {
        IAudioClient_Stop(this->hidden->client);
        SDL_OpenedAudioDeviceDisconnected(this);
        SDL_assert(!SDL_AtomicGet(&this->enabled));
    }

    return SDL_TRUE;
}

static int UpdateAudioStream(_THIS, const SDL_AudioSpec *oldspec)
{
    /* Since WASAPI requires us to handle all audio conversion, and our
       device format might have changed, we might have to add/remove/change
       the audio stream that the higher level uses to convert data, so
       SDL keeps firing the callback as if nothing happened here. */

    if ((this->callbackspec.channels == this->spec.channels) &&
        (this->callbackspec.format == this->spec.format) &&
        (this->callbackspec.freq == this->spec.freq) &&
        (this->callbackspec.samples == this->spec.samples)) {
        /* no need to buffer/convert in an AudioStream! */
        SDL_FreeAudioStream(this->stream);
        this->stream = NULL;
    } else if ((oldspec->channels == this->spec.channels) &&
               (oldspec->format == this->spec.format) &&
               (oldspec->freq == this->spec.freq)) {
        /* The existing audio stream is okay to keep using. */
    } else {
        /* replace the audiostream for new format */
        SDL_FreeAudioStream(this->stream);
        if (this->iscapture) {
            this->stream = SDL_NewAudioStream(this->spec.format,
                                              this->spec.channels, this->spec.freq,
                                              this->callbackspec.format,
                                              this->callbackspec.channels,
                                              this->callbackspec.freq);
        } else {
            this->stream = SDL_NewAudioStream(this->callbackspec.format,
                                              this->callbackspec.channels,
                                              this->callbackspec.freq, this->spec.format,
                                              this->spec.channels, this->spec.freq);
        }

        if (!this->stream) {
            return -1; /* SDL_NewAudioStream should have called SDL_SetError. */
        }
    }

    /* if the device sample size changed, make sure we're asking for enough data. */
    if (this->callbackspec.samples != this->spec.samples) {
        this->callbackspec.samples = this->spec.samples;
        SDL_CalculateAudioSpec(&this->callbackspec);
    }

    /* make sure our scratch buffer can cover the new device spec. */
    if (this->spec.size > this->work_buffer_len) {
        Uint8 *ptr = (Uint8 *)SDL_realloc(this->work_buffer, this->spec.size);
        if (!ptr) {
            return SDL_OutOfMemory();
        }
        this->work_buffer = ptr;
        this->work_buffer_len = this->spec.size;
    }

    return 0;
}

static void ReleaseWasapiDevice(_THIS);

static SDL_bool RecoverWasapiDevice(_THIS)
{
    ReleaseWasapiDevice(this); /* dump the lost device's handles. */

    if (this->hidden->default_device_generation) {
        this->hidden->default_device_generation = SDL_AtomicGet(this->iscapture ? &SDL_IMMDevice_DefaultCaptureGeneration : &SDL_IMMDevice_DefaultPlaybackGeneration);
    }

    /* this can fail for lots of reasons, but the most likely is we had a
       non-default device that was disconnected, so we can't recover. Default
       devices try to reinitialize whatever the new default is, so it's more
       likely to carry on here, but this handles a non-default device that
       simply had its format changed in the Windows Control Panel. */
    if (WASAPI_ActivateDevice(this, SDL_TRUE) == -1) {
        SDL_OpenedAudioDeviceDisconnected(this);
        return SDL_FALSE;
    }

    this->hidden->device_lost = SDL_FALSE;

    return SDL_TRUE; /* okay, carry on with new device details! */
}

static SDL_bool RecoverWasapiIfLost(_THIS)
{
    const int generation = this->hidden->default_device_generation;
    SDL_bool lost = this->hidden->device_lost;

    if (!SDL_AtomicGet(&this->enabled)) {
        return SDL_FALSE; /* already failed. */
    }

    if (!this->hidden->client) {
        return SDL_TRUE; /* still waiting for activation. */
    }

    if (!lost && (generation > 0)) { /* is a default device? */
        const int newgen = SDL_AtomicGet(this->iscapture ? &SDL_IMMDevice_DefaultCaptureGeneration : &SDL_IMMDevice_DefaultPlaybackGeneration);
        if (generation != newgen) { /* the desired default device was changed, jump over to it. */
            lost = SDL_TRUE;
        }
    }

    return lost ? RecoverWasapiDevice(this) : SDL_TRUE;
}

static void WASAPI_WaitDevice(_THIS);

static Uint8 *WASAPI_GetDeviceBuf(_THIS)
{
    /* get an endpoint buffer from WASAPI. */
    BYTE *buffer = NULL;

    while (RecoverWasapiIfLost(this) && this->hidden->render) {
        const HRESULT ret = IAudioRenderClient_GetBuffer(this->hidden->render, this->spec.samples, &buffer);
        if (ret == AUDCLNT_E_BUFFER_TOO_LARGE) {
            WASAPI_WaitDevice(this);  /* see if we can wait on the buffer to drain some more first... */
        } else if (!WasapiFailed(this, ret)) {
            return (Uint8 *)buffer;
        }
        SDL_assert(buffer == NULL);
    }

    return (Uint8 *)buffer;
}

static void WASAPI_PlayDevice(_THIS)
{
    if (this->hidden->render) { /* definitely activated? */
        /* WasapiFailed() will mark the device for reacquisition or removal elsewhere. */
        WasapiFailed(this, IAudioRenderClient_ReleaseBuffer(this->hidden->render, this->spec.samples, 0));
    }
}

static void WASAPI_WaitDevice(_THIS)
{
    while (RecoverWasapiIfLost(this) && this->hidden->client && this->hidden->event) {
        DWORD waitResult = WaitForSingleObjectEx(this->hidden->event, 200, FALSE);
        if (waitResult == WAIT_OBJECT_0) {
            const UINT32 maxpadding = this->spec.samples;
            UINT32 padding = 0;
            if (!WasapiFailed(this, IAudioClient_GetCurrentPadding(this->hidden->client, &padding))) {
                /*SDL_Log("WASAPI EVENT! padding=%u maxpadding=%u", (unsigned int)padding, (unsigned int)maxpadding);*/
                if (this->iscapture) {
                    if (padding > 0) {
                        break;
                    }
                } else {
                    if (padding <= maxpadding) {
                        break;
                    }
                }
            }
        } else if (waitResult != WAIT_TIMEOUT) {
            /*SDL_Log("WASAPI FAILED EVENT!");*/
            IAudioClient_Stop(this->hidden->client);
            SDL_OpenedAudioDeviceDisconnected(this);
        }
    }
}

static int WASAPI_CaptureFromDevice(_THIS, void *buffer, int buflen)
{
    SDL_AudioStream *stream = this->hidden->capturestream;
    const int avail = SDL_AudioStreamAvailable(stream);
    if (avail > 0) {
        const int cpy = SDL_min(buflen, avail);
        SDL_AudioStreamGet(stream, buffer, cpy);
        return cpy;
    }

    while (RecoverWasapiIfLost(this)) {
        HRESULT ret;
        BYTE *ptr = NULL;
        UINT32 frames = 0;
        DWORD flags = 0;

        /* uhoh, client isn't activated yet, just return silence. */
        if (!this->hidden->capture) {
            /* Delay so we run at about the speed that audio would be arriving. */
            SDL_Delay(((this->spec.samples * 1000) / this->spec.freq));
            SDL_memset(buffer, this->spec.silence, buflen);
            return buflen;
        }

        ret = IAudioCaptureClient_GetBuffer(this->hidden->capture, &ptr, &frames, &flags, NULL, NULL);
        if (ret != AUDCLNT_S_BUFFER_EMPTY) {
            WasapiFailed(this, ret); /* mark device lost/failed if necessary. */
        }

        if ((ret == AUDCLNT_S_BUFFER_EMPTY) || !frames) {
            WASAPI_WaitDevice(this);
        } else if (ret == S_OK) {
            const int total = ((int)frames) * this->hidden->framesize;
            const int cpy = SDL_min(buflen, total);
            const int leftover = total - cpy;
            const SDL_bool silent = (flags & AUDCLNT_BUFFERFLAGS_SILENT) ? SDL_TRUE : SDL_FALSE;

            if (silent) {
                SDL_memset(buffer, this->spec.silence, cpy);
            } else {
                SDL_memcpy(buffer, ptr, cpy);
            }

            if (leftover > 0) {
                ptr += cpy;
                if (silent) {
                    SDL_memset(ptr, this->spec.silence, leftover); /* I guess this is safe? */
                }

                if (SDL_AudioStreamPut(stream, ptr, leftover) == -1) {
                    return -1; /* uhoh, out of memory, etc. Kill device.  :( */
                }
            }

            ret = IAudioCaptureClient_ReleaseBuffer(this->hidden->capture, frames);
            WasapiFailed(this, ret); /* mark device lost/failed if necessary. */

            return cpy;
        }
    }

    return -1; /* unrecoverable error. */
}

static void WASAPI_FlushCapture(_THIS)
{
    BYTE *ptr = NULL;
    UINT32 frames = 0;
    DWORD flags = 0;

    if (!this->hidden->capture) {
        return; /* not activated yet? */
    }

    /* just read until we stop getting packets, throwing them away. */
    while (SDL_TRUE) {
        const HRESULT ret = IAudioCaptureClient_GetBuffer(this->hidden->capture, &ptr, &frames, &flags, NULL, NULL);
        if (ret == AUDCLNT_S_BUFFER_EMPTY) {
            break; /* no more buffered data; we're done. */
        } else if (WasapiFailed(this, ret)) {
            break; /* failed for some other reason, abort. */
        } else if (WasapiFailed(this, IAudioCaptureClient_ReleaseBuffer(this->hidden->capture, frames))) {
            break; /* something broke. */
        }
    }
    SDL_AudioStreamClear(this->hidden->capturestream);
}

static void ReleaseWasapiDevice(_THIS)
{
    if (this->hidden->client) {
        IAudioClient_Stop(this->hidden->client);
        IAudioClient_Release(this->hidden->client);
        this->hidden->client = NULL;
    }

    if (this->hidden->render) {
        IAudioRenderClient_Release(this->hidden->render);
        this->hidden->render = NULL;
    }

    if (this->hidden->capture) {
        IAudioCaptureClient_Release(this->hidden->capture);
        this->hidden->capture = NULL;
    }

    if (this->hidden->waveformat) {
        CoTaskMemFree(this->hidden->waveformat);
        this->hidden->waveformat = NULL;
    }

    if (this->hidden->capturestream) {
        SDL_FreeAudioStream(this->hidden->capturestream);
        this->hidden->capturestream = NULL;
    }

    if (this->hidden->activation_handler) {
        WASAPI_PlatformDeleteActivationHandler(this->hidden->activation_handler);
        this->hidden->activation_handler = NULL;
    }

    if (this->hidden->event) {
        CloseHandle(this->hidden->event);
        this->hidden->event = NULL;
    }
}

static void WASAPI_CloseDevice(_THIS)
{
    WASAPI_UnrefDevice(this);
}

void WASAPI_RefDevice(_THIS)
{
    SDL_AtomicIncRef(&this->hidden->refcount);
}

void WASAPI_UnrefDevice(_THIS)
{
    if (!SDL_AtomicDecRef(&this->hidden->refcount)) {
        return;
    }

    /* actual closing happens here. */

    /* don't touch this->hidden->task in here; it has to be reverted from
       our callback thread. We do that in WASAPI_ThreadDeinit().
       (likewise for this->hidden->coinitialized). */
    ReleaseWasapiDevice(this);

    if (SDL_ThreadID() == this->hidden->open_threadid) {
        WIN_CoUninitialize();  /* if you closed from a different thread than you opened, sorry, it's a leak. We can't help you. */
    }

    SDL_free(this->hidden->devid);
    SDL_free(this->hidden);
}

/* This is called once a device is activated, possibly asynchronously. */
int WASAPI_PrepDevice(_THIS, const SDL_bool updatestream)
{
    /* !!! FIXME: we could request an exclusive mode stream, which is lower latency;
       !!!  it will write into the kernel's audio buffer directly instead of
       !!!  shared memory that a user-mode mixer then writes to the kernel with
       !!!  everything else. Doing this means any other sound using this device will
       !!!  stop playing, including the user's MP3 player and system notification
       !!!  sounds. You'd probably need to release the device when the app isn't in
       !!!  the foreground, to be a good citizen of the system. It's doable, but it's
       !!!  more work and causes some annoyances, and I don't know what the latency
       !!!  wins actually look like. Maybe add a hint to force exclusive mode at
       !!!  some point. To be sure, defaulting to shared mode is the right thing to
       !!!  do in any case. */
    const SDL_AudioSpec oldspec = this->spec;
    const AUDCLNT_SHAREMODE sharemode = AUDCLNT_SHAREMODE_SHARED;
    UINT32 bufsize = 0; /* this is in sample frames, not samples, not bytes. */
    REFERENCE_TIME default_period = 0;
    IAudioClient *client = this->hidden->client;
    IAudioRenderClient *render = NULL;
    IAudioCaptureClient *capture = NULL;
    WAVEFORMATEX *waveformat = NULL;
    SDL_AudioFormat test_format;
    SDL_AudioFormat wasapi_format = 0;
    HRESULT ret = S_OK;
    DWORD streamflags = 0;

    SDL_assert(client != NULL);

#if defined(__WINRT__) || defined(__GDK__) /* CreateEventEx() arrived in Vista, so we need an #ifdef for XP. */
    this->hidden->event = CreateEventEx(NULL, NULL, 0, EVENT_ALL_ACCESS);
#else
    this->hidden->event = CreateEventW(NULL, 0, 0, NULL);
#endif

    if (!this->hidden->event) {
        return WIN_SetError("WASAPI can't create an event handle");
    }

    ret = IAudioClient_GetMixFormat(client, &waveformat);
    if (FAILED(ret)) {
        return WIN_SetErrorFromHRESULT("WASAPI can't determine mix format", ret);
    }

    SDL_assert(waveformat != NULL);
    this->hidden->waveformat = waveformat;

    this->spec.channels = (Uint8)waveformat->nChannels;

    /* Make sure we have a valid format that we can convert to whatever WASAPI wants. */
    wasapi_format = WaveFormatToSDLFormat(waveformat);

    for (test_format = SDL_FirstAudioFormat(this->spec.format); test_format; test_format = SDL_NextAudioFormat()) {
        if (test_format == wasapi_format) {
            this->spec.format = test_format;
            break;
        }
    }

    if (!test_format) {
        return SDL_SetError("%s: Unsupported audio format", "wasapi");
    }

    ret = IAudioClient_GetDevicePeriod(client, &default_period, NULL);
    if (FAILED(ret)) {
        return WIN_SetErrorFromHRESULT("WASAPI can't determine minimum device period", ret);
    }

    /* we've gotten reports that WASAPI's resampler introduces distortions, but in the short term
       it fixes some other WASAPI-specific quirks we haven't quite tracked down.
       Refer to bug #6326 for the immediate concern. */
#if 0
    this->spec.freq = waveformat->nSamplesPerSec;  /* force sampling rate so our resampler kicks in, if necessary. */
#else
    /* favor WASAPI's resampler over our own */
    if (this->spec.freq != waveformat->nSamplesPerSec) {
        streamflags |= (AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY);
        waveformat->nSamplesPerSec = this->spec.freq;
        waveformat->nAvgBytesPerSec = waveformat->nSamplesPerSec * waveformat->nChannels * (waveformat->wBitsPerSample / 8);
    }
#endif

    streamflags |= AUDCLNT_STREAMFLAGS_EVENTCALLBACK;
    ret = IAudioClient_Initialize(client, sharemode, streamflags, 0, 0, waveformat, NULL);
    if (FAILED(ret)) {
        return WIN_SetErrorFromHRESULT("WASAPI can't initialize audio client", ret);
    }

    ret = IAudioClient_SetEventHandle(client, this->hidden->event);
    if (FAILED(ret)) {
        return WIN_SetErrorFromHRESULT("WASAPI can't set event handle", ret);
    }

    ret = IAudioClient_GetBufferSize(client, &bufsize);
    if (FAILED(ret)) {
        return WIN_SetErrorFromHRESULT("WASAPI can't determine buffer size", ret);
    }

    /* Match the callback size to the period size to cut down on the number of
       interrupts waited for in each call to WaitDevice */
    {
        const float period_millis = default_period / 10000.0f;
        const float period_frames = period_millis * this->spec.freq / 1000.0f;
        this->spec.samples = (Uint16)SDL_ceilf(period_frames);
    }

    /* regardless of what we calculated for the period size, clamp it to the expected hardware buffer size. */
    if (this->spec.samples > bufsize) {
        this->spec.samples = bufsize;
    }

    /* Update the fragment size as size in bytes */
    SDL_CalculateAudioSpec(&this->spec);

    this->hidden->framesize = (SDL_AUDIO_BITSIZE(this->spec.format) / 8) * this->spec.channels;

    if (this->iscapture) {
        this->hidden->capturestream = SDL_NewAudioStream(this->spec.format, this->spec.channels, this->spec.freq, this->spec.format, this->spec.channels, this->spec.freq);
        if (!this->hidden->capturestream) {
            return -1; /* already set SDL_Error */
        }

        ret = IAudioClient_GetService(client, &SDL_IID_IAudioCaptureClient, (void **)&capture);
        if (FAILED(ret)) {
            return WIN_SetErrorFromHRESULT("WASAPI can't get capture client service", ret);
        }

        SDL_assert(capture != NULL);
        this->hidden->capture = capture;
        ret = IAudioClient_Start(client);
        if (FAILED(ret)) {
            return WIN_SetErrorFromHRESULT("WASAPI can't start capture", ret);
        }

        WASAPI_FlushCapture(this); /* MSDN says you should flush capture endpoint right after startup. */
    } else {
        ret = IAudioClient_GetService(client, &SDL_IID_IAudioRenderClient, (void **)&render);
        if (FAILED(ret)) {
            return WIN_SetErrorFromHRESULT("WASAPI can't get render client service", ret);
        }

        SDL_assert(render != NULL);
        this->hidden->render = render;
        ret = IAudioClient_Start(client);
        if (FAILED(ret)) {
            return WIN_SetErrorFromHRESULT("WASAPI can't start playback", ret);
        }
    }

    if (updatestream) {
        return UpdateAudioStream(this, &oldspec);
    }

    return 0; /* good to go. */
}

static int WASAPI_OpenDevice(_THIS, const char *devname)
{
    LPCWSTR devid = (LPCWSTR)this->handle;

    /* Initialize all variables that we clean on shutdown */
    this->hidden = (struct SDL_PrivateAudioData *) SDL_malloc(sizeof(*this->hidden));
    if (!this->hidden) {
        return SDL_OutOfMemory();
    }
    SDL_zerop(this->hidden);

    WASAPI_RefDevice(this); /* so CloseDevice() will unref to zero. */

    if (FAILED(WIN_CoInitialize())) {  /* WASAPI uses COM, we need to make sure it's initialized. You have to close the device from the same thread!! */
        return SDL_SetError("WIN_CoInitialize failed during WASAPI device open");
    }
    this->hidden->open_threadid = SDL_ThreadID();  /* set this _after_ coinitialize so we don't uninit if device fails at the wrong moment. */

    if (!devid) { /* is default device? */
        this->hidden->default_device_generation = SDL_AtomicGet(this->iscapture ? &SDL_IMMDevice_DefaultCaptureGeneration : &SDL_IMMDevice_DefaultPlaybackGeneration);
    } else {
        this->hidden->devid = SDL_wcsdup(devid);
        if (!this->hidden->devid) {
            return SDL_OutOfMemory();
        }
    }

    if (WASAPI_ActivateDevice(this, SDL_FALSE) == -1) {
        return -1; /* already set error. */
    }

    /* Ready, but waiting for async device activation.
       Until activation is successful, we will report silence from capture
       devices and ignore data on playback devices.
       Also, since we don't know the _actual_ device format until after
       activation, we let the app have whatever it asks for. We set up
       an SDL_AudioStream to convert, if necessary, once the activation
       completes. */

    return 0;
}

static void WASAPI_ThreadInit(_THIS)
{
    WASAPI_PlatformThreadInit(this);
}

static void WASAPI_ThreadDeinit(_THIS)
{
    WASAPI_PlatformThreadDeinit(this);
}

static void WASAPI_Deinitialize(void)
{
    WASAPI_PlatformDeinit();
}

static SDL_bool WASAPI_Init(SDL_AudioDriverImpl *impl)
{
    if (WASAPI_PlatformInit() == -1) {
        return SDL_FALSE;
    }

    /* Set the function pointers */
    impl->DetectDevices = WASAPI_DetectDevices;
    impl->ThreadInit = WASAPI_ThreadInit;
    impl->ThreadDeinit = WASAPI_ThreadDeinit;
    impl->OpenDevice = WASAPI_OpenDevice;
    impl->PlayDevice = WASAPI_PlayDevice;
    impl->WaitDevice = WASAPI_WaitDevice;
    impl->GetDeviceBuf = WASAPI_GetDeviceBuf;
    impl->CaptureFromDevice = WASAPI_CaptureFromDevice;
    impl->FlushCapture = WASAPI_FlushCapture;
    impl->CloseDevice = WASAPI_CloseDevice;
    impl->Deinitialize = WASAPI_Deinitialize;
    impl->GetDefaultAudioInfo = WASAPI_GetDefaultAudioInfo;
    impl->HasCaptureSupport = SDL_TRUE;
    impl->SupportsNonPow2Samples = SDL_TRUE;

    return SDL_TRUE; /* this audio target is available. */
}

AudioBootStrap WASAPI_bootstrap = {
    "wasapi", "WASAPI", WASAPI_Init, SDL_FALSE
};

#endif /* SDL_AUDIO_DRIVER_WASAPI */

/* vi: set ts=4 sw=4 expandtab: */
