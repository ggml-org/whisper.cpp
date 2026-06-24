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

#ifdef SDL_AUDIO_DRIVER_ANDROID

/* Output audio to Android */

#include "SDL_audio.h"
#include "../SDL_audio_c.h"
#include "SDL_androidaudio.h"

#include "../../core/android/SDL_android.h"

#include <android/log.h>

static SDL_AudioDevice *audioDevice = NULL;
static SDL_AudioDevice *captureDevice = NULL;


static int ANDROIDAUDIO_OpenDevice(_THIS, const char *devname)
{
    SDL_AudioFormat test_format;
    SDL_bool iscapture = this->iscapture;

    SDL_assert((captureDevice == NULL) || !iscapture);
    SDL_assert((audioDevice == NULL) || iscapture);

    if (iscapture) {
        captureDevice = this;
    } else {
        audioDevice = this;
    }

    this->hidden = (struct SDL_PrivateAudioData *)SDL_calloc(1, sizeof(*this->hidden));
    if (!this->hidden) {
        return SDL_OutOfMemory();
    }

    for (test_format = SDL_FirstAudioFormat(this->spec.format); test_format; test_format = SDL_NextAudioFormat()) {
        if ((test_format == AUDIO_U8) ||
            (test_format == AUDIO_S16) ||
            (test_format == AUDIO_F32)) {
            this->spec.format = test_format;
            break;
        }
    }

    if (!test_format) {
        /* Didn't find a compatible format :( */
        return SDL_SetError("%s: Unsupported audio format", "android");
    }

    {
        int audio_device_id = 0;
        if (devname) {
            audio_device_id = SDL_atoi(devname);
        }
        if (Android_JNI_OpenAudioDevice(iscapture, audio_device_id, &this->spec) < 0) {
            return -1;
        }
    }

    SDL_CalculateAudioSpec(&this->spec);

    return 0;
}

static void ANDROIDAUDIO_PlayDevice(_THIS)
{
    Android_JNI_WriteAudioBuffer();
}

static Uint8 *ANDROIDAUDIO_GetDeviceBuf(_THIS)
{
    return Android_JNI_GetAudioBuffer();
}

static int ANDROIDAUDIO_CaptureFromDevice(_THIS, void *buffer, int buflen)
{
    return Android_JNI_CaptureAudioBuffer(buffer, buflen);
}

static void ANDROIDAUDIO_FlushCapture(_THIS)
{
    Android_JNI_FlushCapturedAudio();
}

static void ANDROIDAUDIO_CloseDevice(_THIS)
{
    /* At this point SDL_CloseAudioDevice via close_audio_device took care of terminating the audio thread
       so it's safe to terminate the Java side buffer and AudioTrack
     */
    Android_JNI_CloseAudioDevice(this->iscapture);
    if (this->iscapture) {
        SDL_assert(captureDevice == this);
        captureDevice = NULL;
    } else {
        SDL_assert(audioDevice == this);
        audioDevice = NULL;
    }
    SDL_free(this->hidden);
}

static SDL_bool ANDROIDAUDIO_Init(SDL_AudioDriverImpl *impl)
{
    /* Set the function pointers */
    impl->DetectDevices = Android_DetectDevices;
    impl->OpenDevice = ANDROIDAUDIO_OpenDevice;
    impl->PlayDevice = ANDROIDAUDIO_PlayDevice;
    impl->GetDeviceBuf = ANDROIDAUDIO_GetDeviceBuf;
    impl->CloseDevice = ANDROIDAUDIO_CloseDevice;
    impl->CaptureFromDevice = ANDROIDAUDIO_CaptureFromDevice;
    impl->FlushCapture = ANDROIDAUDIO_FlushCapture;
    impl->AllowsArbitraryDeviceNames = SDL_TRUE;

    /* and the capabilities */
    impl->HasCaptureSupport = SDL_TRUE;
    impl->OnlyHasDefaultOutputDevice = SDL_FALSE;
    impl->OnlyHasDefaultCaptureDevice = SDL_FALSE;

    return SDL_TRUE; /* this audio target is available. */
}

AudioBootStrap ANDROIDAUDIO_bootstrap = {
    "android", "SDL Android audio driver", ANDROIDAUDIO_Init, SDL_FALSE
};

/* Pause (block) all non already paused audio devices by taking their mixer lock */
void ANDROIDAUDIO_PauseDevices(void)
{
    /* TODO: Handle multiple devices? */
    if (audioDevice && audioDevice->hidden) {
        SDL_LockMutex(audioDevice->mixer_lock);
    }

    if (captureDevice && captureDevice->hidden) {
        SDL_LockMutex(captureDevice->mixer_lock);
    }
}

/* Resume (unblock) all non already paused audio devices by releasing their mixer lock */
void ANDROIDAUDIO_ResumeDevices(void)
{
    /* TODO: Handle multiple devices? */
    if (audioDevice && audioDevice->hidden) {
        SDL_UnlockMutex(audioDevice->mixer_lock);
    }

    if (captureDevice && captureDevice->hidden) {
        SDL_UnlockMutex(captureDevice->mixer_lock);
    }
}

#else

void ANDROIDAUDIO_ResumeDevices(void) {}
void ANDROIDAUDIO_PauseDevices(void) {}

#endif /* SDL_AUDIO_DRIVER_ANDROID */

/* vi: set ts=4 sw=4 expandtab: */
