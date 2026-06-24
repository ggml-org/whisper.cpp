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

#ifndef SDL_sysaudio_h_
#define SDL_sysaudio_h_

#include "SDL_mutex.h"
#include "SDL_thread.h"
#include "../SDL_dataqueue.h"
#include "./SDL_audio_c.h"

/* !!! FIXME: These are wordy and unlocalized... */
#define DEFAULT_OUTPUT_DEVNAME "System audio output device"
#define DEFAULT_INPUT_DEVNAME  "System audio capture device"

/* The SDL audio driver */
typedef struct SDL_AudioDevice SDL_AudioDevice;
#define _THIS SDL_AudioDevice *_this

/* Audio targets should call this as devices are added to the system (such as
   a USB headset being plugged in), and should also be called for
   for every device found during DetectDevices(). */
extern void SDL_AddAudioDevice(const SDL_bool iscapture, const char *name, SDL_AudioSpec *spec, void *handle);

/* Audio targets should call this as devices are removed, so SDL can update
   its list of available devices. */
extern void SDL_RemoveAudioDevice(const SDL_bool iscapture, void *handle);

/* Audio targets should call this if an opened audio device is lost while
   being used. This can happen due to i/o errors, or a device being unplugged,
   etc. If the device is totally gone, please also call SDL_RemoveAudioDevice()
   as appropriate so SDL's list of devices is accurate. */
extern void SDL_OpenedAudioDeviceDisconnected(SDL_AudioDevice *device);

/* This is the size of a packet when using SDL_QueueAudio(). We allocate
   these as necessary and pool them, under the assumption that we'll
   eventually end up with a handful that keep recycling, meeting whatever
   the app needs. We keep packing data tightly as more arrives to avoid
   wasting space, and if we get a giant block of data, we'll split them
   into multiple packets behind the scenes. My expectation is that most
   apps will have 2-3 of these in the pool. 8k should cover most needs, but
   if this is crippling for some embedded system, we can #ifdef this.
   The system preallocates enough packets for 2 callbacks' worth of data. */
#define SDL_AUDIOBUFFERQUEUE_PACKETLEN (8 * 1024)

typedef struct SDL_AudioDriverImpl
{
    void (*DetectDevices)(void);
    int (*OpenDevice)(_THIS, const char *devname);
    void (*ThreadInit)(_THIS);   /* Called by audio thread at start */
    void (*ThreadDeinit)(_THIS); /* Called by audio thread at end */
    void (*WaitDevice)(_THIS);
    void (*PlayDevice)(_THIS);
    Uint8 *(*GetDeviceBuf)(_THIS);
    int (*CaptureFromDevice)(_THIS, void *buffer, int buflen);
    void (*FlushCapture)(_THIS);
    void (*CloseDevice)(_THIS);
    void (*LockDevice)(_THIS);
    void (*UnlockDevice)(_THIS);
    void (*FreeDeviceHandle)(void *handle); /**< SDL is done with handle from SDL_AddAudioDevice() */
    void (*Deinitialize)(void);
    int (*GetDefaultAudioInfo)(char **name, SDL_AudioSpec *spec, int iscapture);

    /* !!! FIXME: add pause(), so we can optimize instead of mixing silence. */

    /* Some flags to push duplicate code into the core and reduce #ifdefs. */
    SDL_bool ProvidesOwnCallbackThread;
    SDL_bool HasCaptureSupport;
    SDL_bool OnlyHasDefaultOutputDevice;
    SDL_bool OnlyHasDefaultCaptureDevice;
    SDL_bool AllowsArbitraryDeviceNames;
    SDL_bool SupportsNonPow2Samples;
} SDL_AudioDriverImpl;

typedef struct SDL_AudioDeviceItem
{
    void *handle;
    char *name;
    char *original_name;
    SDL_AudioSpec spec;
    int dupenum;
    struct SDL_AudioDeviceItem *next;
} SDL_AudioDeviceItem;

typedef struct SDL_AudioDriver
{
    /* * * */
    /* The name of this audio driver */
    const char *name;

    /* * * */
    /* The description of this audio driver */
    const char *desc;

    SDL_AudioDriverImpl impl;

    /* A mutex for device detection */
    SDL_mutex *detectionLock;
    SDL_bool captureDevicesRemoved;
    SDL_bool outputDevicesRemoved;
    int outputDeviceCount;
    int inputDeviceCount;
    SDL_AudioDeviceItem *outputDevices;
    SDL_AudioDeviceItem *inputDevices;
} SDL_AudioDriver;

/* Define the SDL audio driver structure */
struct SDL_AudioDevice
{
    /* * * */
    /* Data common to all devices */
    SDL_AudioDeviceID id;

    /* The device's current audio specification */
    SDL_AudioSpec spec;

    /* The callback's expected audio specification (converted vs device's spec). */
    SDL_AudioSpec callbackspec;

    /* Stream that converts and resamples. NULL if not needed. */
    SDL_AudioStream *stream;

    /* Current state flags */
    SDL_atomic_t shutdown; /* true if we are signaling the play thread to end. */
    SDL_atomic_t enabled;  /* true if device is functioning and connected. */
    SDL_atomic_t paused;
    SDL_bool iscapture;

    /* Scratch buffer used in the bridge between SDL and the user callback. */
    Uint8 *work_buffer;

    /* Size, in bytes, of work_buffer. */
    Uint32 work_buffer_len;

    /* A mutex for locking the mixing buffers */
    SDL_mutex *mixer_lock;

    /* A thread to feed the audio device */
    SDL_Thread *thread;
    SDL_threadID threadid;

    /* Queued buffers (if app not using callback). */
    SDL_DataQueue *buffer_queue;

    /* * * */
    /* Data private to this driver */
    struct SDL_PrivateAudioData *hidden;

    void *handle;
};
#undef _THIS

typedef struct AudioBootStrap
{
    const char *name;
    const char *desc;
    SDL_bool (*init)(SDL_AudioDriverImpl *impl);
    SDL_bool demand_only; /* 1==request explicitly, or it won't be available. */
} AudioBootStrap;

/* Not all of these are available in a given build. Use #ifdefs, etc. */
extern AudioBootStrap PIPEWIRE_bootstrap;
extern AudioBootStrap PULSEAUDIO_bootstrap;
extern AudioBootStrap ALSA_bootstrap;
extern AudioBootStrap JACK_bootstrap;
extern AudioBootStrap SNDIO_bootstrap;
extern AudioBootStrap NETBSDAUDIO_bootstrap;
extern AudioBootStrap DSP_bootstrap;
extern AudioBootStrap QSAAUDIO_bootstrap;
extern AudioBootStrap SUNAUDIO_bootstrap;
extern AudioBootStrap ARTS_bootstrap;
extern AudioBootStrap ESD_bootstrap;
extern AudioBootStrap NACLAUDIO_bootstrap;
extern AudioBootStrap NAS_bootstrap;
extern AudioBootStrap WASAPI_bootstrap;
extern AudioBootStrap DSOUND_bootstrap;
extern AudioBootStrap WINMM_bootstrap;
extern AudioBootStrap PAUDIO_bootstrap;
extern AudioBootStrap HAIKUAUDIO_bootstrap;
extern AudioBootStrap COREAUDIO_bootstrap;
extern AudioBootStrap DISKAUDIO_bootstrap;
extern AudioBootStrap DUMMYAUDIO_bootstrap;
extern AudioBootStrap FUSIONSOUND_bootstrap;
extern AudioBootStrap aaudio_bootstrap;
extern AudioBootStrap openslES_bootstrap;
extern AudioBootStrap ANDROIDAUDIO_bootstrap;
extern AudioBootStrap PS2AUDIO_bootstrap;
extern AudioBootStrap PSPAUDIO_bootstrap;
extern AudioBootStrap VITAAUD_bootstrap;
extern AudioBootStrap N3DSAUDIO_bootstrap;
extern AudioBootStrap EMSCRIPTENAUDIO_bootstrap;
extern AudioBootStrap OS2AUDIO_bootstrap;

#endif /* SDL_sysaudio_h_ */

/* vi: set ts=4 sw=4 expandtab: */
