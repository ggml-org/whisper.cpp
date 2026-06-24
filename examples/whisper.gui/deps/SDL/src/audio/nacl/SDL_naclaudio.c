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

#ifdef SDL_AUDIO_DRIVER_NACL

#include "SDL_naclaudio.h"

#include "SDL_audio.h"
#include "SDL_mutex.h"
#include "../SDL_audio_c.h"
#include "../SDL_audiodev_c.h"

#include "ppapi/c/pp_errors.h"
#include "ppapi/c/pp_instance.h"
#include "ppapi_simple/ps.h"
#include "ppapi_simple/ps_interface.h"
#include "ppapi_simple/ps_event.h"

/* The tag name used by NACL audio */
#define NACLAUDIO_DRIVER_NAME         "nacl"

#define SAMPLE_FRAME_COUNT 4096

/* Audio driver functions */
static void nacl_audio_callback(void* samples, uint32_t buffer_size, PP_TimeDelta latency, void* data);

/* FIXME: Make use of latency if needed */
static void nacl_audio_callback(void* stream, uint32_t buffer_size, PP_TimeDelta latency, void* data)
{
    const int len = (int) buffer_size;
    SDL_AudioDevice* _this = (SDL_AudioDevice*) data;
    SDL_AudioCallback callback = _this->callbackspec.callback;

    SDL_LockMutex(_this->mixer_lock);

    /* Only do something if audio is enabled */
    if (!SDL_AtomicGet(&_this->enabled) || SDL_AtomicGet(&_this->paused)) {
        if (_this->stream) {
            SDL_AudioStreamClear(_this->stream);
        }
        SDL_memset(stream, _this->spec.silence, len);
    } else {
        SDL_assert(_this->spec.size == len);

        if (!_this->stream) {  /* no conversion necessary. */
            callback(_this->callbackspec.userdata, stream, len);
        } else {  /* streaming/converting */
            const int stream_len = _this->callbackspec.size;
            while (SDL_AudioStreamAvailable(_this->stream) < len) {
                callback(_this->callbackspec.userdata, _this->work_buffer, stream_len);
                if (SDL_AudioStreamPut(_this->stream, _this->work_buffer, stream_len) == -1) {
                    SDL_AudioStreamClear(_this->stream);
                    SDL_AtomicSet(&_this->enabled, 0);
                    break;
                }
            }

            const int got = SDL_AudioStreamGet(_this->stream, stream, len);
            SDL_assert((got < 0) || (got == len));
            if (got != len) {
                SDL_memset(stream, _this->spec.silence, len);
            }
        }
    }

    SDL_UnlockMutex(_this->mixer_lock);
}

static void NACLAUDIO_CloseDevice(SDL_AudioDevice *device)
{
    const PPB_Core *core = PSInterfaceCore();
    const PPB_Audio *ppb_audio = PSInterfaceAudio();
    SDL_PrivateAudioData *hidden = (SDL_PrivateAudioData *) device->hidden;

    ppb_audio->StopPlayback(hidden->audio);
    core->ReleaseResource(hidden->audio);
}

static int NACLAUDIO_OpenDevice(_THIS, const char *devname)
{
    PP_Instance instance = PSGetInstanceId();
    const PPB_Audio *ppb_audio = PSInterfaceAudio();
    const PPB_AudioConfig *ppb_audiocfg = PSInterfaceAudioConfig();

    private = (SDL_PrivateAudioData *)SDL_calloc(1, sizeof(*private));
    if (!private) {
        return SDL_OutOfMemory();
    }

    _this->spec.freq = 44100;
    _this->spec.format = AUDIO_S16LSB;
    _this->spec.channels = 2;
    _this->spec.samples = ppb_audiocfg->RecommendSampleFrameCount(
        instance,
        PP_AUDIOSAMPLERATE_44100,
        SAMPLE_FRAME_COUNT);

    /* Calculate the final parameters for this audio specification */
    SDL_CalculateAudioSpec(&_this->spec);

    private->audio = ppb_audio->Create(
        instance,
        ppb_audiocfg->CreateStereo16Bit(instance, PP_AUDIOSAMPLERATE_44100, _this->spec.samples),
        nacl_audio_callback,
        _this);

    /* Start audio playback while we are still on the main thread. */
    ppb_audio->StartPlayback(private->audio);

    return 0;
}

static SDL_bool NACLAUDIO_Init(SDL_AudioDriverImpl * impl)
{
    if (PSGetInstanceId() == 0) {
        return SDL_FALSE;
    }

    /* Set the function pointers */
    impl->OpenDevice = NACLAUDIO_OpenDevice;
    impl->CloseDevice = NACLAUDIO_CloseDevice;
    impl->OnlyHasDefaultOutputDevice = SDL_TRUE;
    impl->ProvidesOwnCallbackThread = SDL_TRUE;
    /*
     *    impl->WaitDevice = NACLAUDIO_WaitDevice;
     *    impl->GetDeviceBuf = NACLAUDIO_GetDeviceBuf;
     *    impl->PlayDevice = NACLAUDIO_PlayDevice;
     *    impl->Deinitialize = NACLAUDIO_Deinitialize;
     */

    return SDL_TRUE;
}

AudioBootStrap NACLAUDIO_bootstrap = {
    NACLAUDIO_DRIVER_NAME, "SDL NaCl Audio Driver",
    NACLAUDIO_Init, SDL_FALSE
};

#endif /* SDL_AUDIO_DRIVER_NACL */

/* vi: set ts=4 sw=4 expandtab: */
