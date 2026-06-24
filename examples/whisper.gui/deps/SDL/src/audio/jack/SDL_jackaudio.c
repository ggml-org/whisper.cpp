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

#ifdef SDL_AUDIO_DRIVER_JACK

#include "SDL_timer.h"
#include "SDL_audio.h"
#include "../SDL_audio_c.h"
#include "SDL_jackaudio.h"
#include "SDL_loadso.h"
#include "../../thread/SDL_systhread.h"

static jack_client_t *(*JACK_jack_client_open)(const char *, jack_options_t, jack_status_t *, ...);
static int (*JACK_jack_client_close)(jack_client_t *);
static void (*JACK_jack_on_shutdown)(jack_client_t *, JackShutdownCallback, void *);
static int (*JACK_jack_activate)(jack_client_t *);
static int (*JACK_jack_deactivate)(jack_client_t *);
static void *(*JACK_jack_port_get_buffer)(jack_port_t *, jack_nframes_t);
static int (*JACK_jack_port_unregister)(jack_client_t *, jack_port_t *);
static void (*JACK_jack_free)(void *);
static const char **(*JACK_jack_get_ports)(jack_client_t *, const char *, const char *, unsigned long);
static jack_nframes_t (*JACK_jack_get_sample_rate)(jack_client_t *);
static jack_nframes_t (*JACK_jack_get_buffer_size)(jack_client_t *);
static jack_port_t *(*JACK_jack_port_register)(jack_client_t *, const char *, const char *, unsigned long, unsigned long);
static jack_port_t *(*JACK_jack_port_by_name)(jack_client_t *, const char *);
static const char *(*JACK_jack_port_name)(const jack_port_t *);
static const char *(*JACK_jack_port_type)(const jack_port_t *);
static int (*JACK_jack_connect)(jack_client_t *, const char *, const char *);
static int (*JACK_jack_set_process_callback)(jack_client_t *, JackProcessCallback, void *);

static int load_jack_syms(void);

#ifdef SDL_AUDIO_DRIVER_JACK_DYNAMIC

static const char *jack_library = SDL_AUDIO_DRIVER_JACK_DYNAMIC;
static void *jack_handle = NULL;

/* !!! FIXME: this is copy/pasted in several places now */
static int load_jack_sym(const char *fn, void **addr)
{
    *addr = SDL_LoadFunction(jack_handle, fn);
    if (!*addr) {
        /* Don't call SDL_SetError(): SDL_LoadFunction already did. */
        return 0;
    }

    return 1;
}

/* cast funcs to char* first, to please GCC's strict aliasing rules. */
#define SDL_JACK_SYM(x)                                 \
    if (!load_jack_sym(#x, (void **)(char *)&JACK_##x)) \
    return -1

static void UnloadJackLibrary(void)
{
    if (jack_handle) {
        SDL_UnloadObject(jack_handle);
        jack_handle = NULL;
    }
}

static int LoadJackLibrary(void)
{
    int retval = 0;
    if (!jack_handle) {
        jack_handle = SDL_LoadObject(jack_library);
        if (!jack_handle) {
            retval = -1;
            /* Don't call SDL_SetError(): SDL_LoadObject already did. */
        } else {
            retval = load_jack_syms();
            if (retval < 0) {
                UnloadJackLibrary();
            }
        }
    }
    return retval;
}

#else

#define SDL_JACK_SYM(x) JACK_##x = x

static void UnloadJackLibrary(void)
{
}

static int LoadJackLibrary(void)
{
    load_jack_syms();
    return 0;
}

#endif /* SDL_AUDIO_DRIVER_JACK_DYNAMIC */

static int load_jack_syms(void)
{
    SDL_JACK_SYM(jack_client_open);
    SDL_JACK_SYM(jack_client_close);
    SDL_JACK_SYM(jack_on_shutdown);
    SDL_JACK_SYM(jack_activate);
    SDL_JACK_SYM(jack_deactivate);
    SDL_JACK_SYM(jack_port_get_buffer);
    SDL_JACK_SYM(jack_port_unregister);
    SDL_JACK_SYM(jack_free);
    SDL_JACK_SYM(jack_get_ports);
    SDL_JACK_SYM(jack_get_sample_rate);
    SDL_JACK_SYM(jack_get_buffer_size);
    SDL_JACK_SYM(jack_port_register);
    SDL_JACK_SYM(jack_port_by_name);
    SDL_JACK_SYM(jack_port_name);
    SDL_JACK_SYM(jack_port_type);
    SDL_JACK_SYM(jack_connect);
    SDL_JACK_SYM(jack_set_process_callback);

    return 0;
}

static void jackShutdownCallback(void *arg) /* JACK went away; device is lost. */
{
    SDL_AudioDevice *this = (SDL_AudioDevice *)arg;
    SDL_OpenedAudioDeviceDisconnected(this);
    SDL_SemPost(this->hidden->iosem); /* unblock the SDL thread. */
}

// !!! FIXME: implement and register these!
// typedef int(* JackSampleRateCallback)(jack_nframes_t nframes, void *arg)
// typedef int(* JackBufferSizeCallback)(jack_nframes_t nframes, void *arg)

static int jackProcessPlaybackCallback(jack_nframes_t nframes, void *arg)
{
    SDL_AudioDevice *this = (SDL_AudioDevice *)arg;
    jack_port_t **ports = this->hidden->sdlports;
    const int total_channels = this->spec.channels;
    const int total_frames = this->spec.samples;
    int channelsi;

    if (!SDL_AtomicGet(&this->enabled)) {
        /* silence the buffer to avoid repeats and corruption. */
        SDL_memset(this->hidden->iobuffer, '\0', this->spec.size);
    }

    for (channelsi = 0; channelsi < total_channels; channelsi++) {
        float *dst = (float *)JACK_jack_port_get_buffer(ports[channelsi], nframes);
        if (dst) {
            const float *src = this->hidden->iobuffer + channelsi;
            int framesi;
            for (framesi = 0; framesi < total_frames; framesi++) {
                *(dst++) = *src;
                src += total_channels;
            }
        }
    }

    SDL_SemPost(this->hidden->iosem); /* tell SDL thread we're done; refill the buffer. */
    return 0;
}

/* This function waits until it is possible to write a full sound buffer */
static void JACK_WaitDevice(_THIS)
{
    if (SDL_AtomicGet(&this->enabled)) {
        if (SDL_SemWait(this->hidden->iosem) == -1) {
            SDL_OpenedAudioDeviceDisconnected(this);
        }
    }
}

static Uint8 *JACK_GetDeviceBuf(_THIS)
{
    return (Uint8 *)this->hidden->iobuffer;
}

static int jackProcessCaptureCallback(jack_nframes_t nframes, void *arg)
{
    SDL_AudioDevice *this = (SDL_AudioDevice *)arg;
    if (SDL_AtomicGet(&this->enabled)) {
        jack_port_t **ports = this->hidden->sdlports;
        const int total_channels = this->spec.channels;
        const int total_frames = this->spec.samples;
        int channelsi;

        for (channelsi = 0; channelsi < total_channels; channelsi++) {
            const float *src = (const float *)JACK_jack_port_get_buffer(ports[channelsi], nframes);
            if (src) {
                float *dst = this->hidden->iobuffer + channelsi;
                int framesi;
                for (framesi = 0; framesi < total_frames; framesi++) {
                    *dst = *(src++);
                    dst += total_channels;
                }
            }
        }
    }

    SDL_SemPost(this->hidden->iosem); /* tell SDL thread we're done; new buffer is ready! */
    return 0;
}

static int JACK_CaptureFromDevice(_THIS, void *buffer, int buflen)
{
    SDL_assert(buflen == this->spec.size); /* we always fill a full buffer. */

    /* Wait for JACK to fill the iobuffer */
    if (SDL_SemWait(this->hidden->iosem) == -1) {
        return -1;
    }

    SDL_memcpy(buffer, this->hidden->iobuffer, buflen);
    return buflen;
}

static void JACK_FlushCapture(_THIS)
{
    SDL_SemWait(this->hidden->iosem);
}

static void JACK_CloseDevice(_THIS)
{
    if (this->hidden->client) {
        JACK_jack_deactivate(this->hidden->client);

        if (this->hidden->sdlports) {
            const int channels = this->spec.channels;
            int i;
            for (i = 0; i < channels; i++) {
                JACK_jack_port_unregister(this->hidden->client, this->hidden->sdlports[i]);
            }
            SDL_free(this->hidden->sdlports);
        }

        JACK_jack_client_close(this->hidden->client);
    }

    if (this->hidden->iosem) {
        SDL_DestroySemaphore(this->hidden->iosem);
    }

    SDL_free(this->hidden->iobuffer);
    SDL_free(this->hidden);
}

static int JACK_OpenDevice(_THIS, const char *devname)
{
    /* Note that JACK uses "output" for capture devices (they output audio
        data to us) and "input" for playback (we input audio data to them).
        Likewise, SDL's playback port will be "output" (we write data out)
        and capture will be "input" (we read data in). */
    SDL_bool iscapture = this->iscapture;
    const unsigned long sysportflags = iscapture ? JackPortIsOutput : JackPortIsInput;
    const unsigned long sdlportflags = iscapture ? JackPortIsInput : JackPortIsOutput;
    const JackProcessCallback callback = iscapture ? jackProcessCaptureCallback : jackProcessPlaybackCallback;
    const char *sdlportstr = iscapture ? "input" : "output";
    const char **devports = NULL;
    int *audio_ports;
    jack_client_t *client = NULL;
    jack_status_t status;
    int channels = 0;
    int ports = 0;
    int i;

    /* Initialize all variables that we clean on shutdown */
    this->hidden = (struct SDL_PrivateAudioData *)SDL_calloc(1, sizeof(*this->hidden));
    if (this->hidden == NULL) {
        return SDL_OutOfMemory();
    }

    /* !!! FIXME: we _still_ need an API to specify an app name */
    client = JACK_jack_client_open("SDL", JackNoStartServer, &status, NULL);
    this->hidden->client = client;
    if (client == NULL) {
        return SDL_SetError("Can't open JACK client");
    }

    devports = JACK_jack_get_ports(client, NULL, NULL, JackPortIsPhysical | sysportflags);
    if (devports == NULL || !devports[0]) {
        return SDL_SetError("No physical JACK ports available");
    }

    while (devports[++ports]) {
        /* spin to count devports */
    }

    /* Filter out non-audio ports */
    audio_ports = SDL_calloc(ports, sizeof(*audio_ports));
    for (i = 0; i < ports; i++) {
        const jack_port_t *dport = JACK_jack_port_by_name(client, devports[i]);
        const char *type = JACK_jack_port_type(dport);
        const int len = SDL_strlen(type);
        /* See if type ends with "audio" */
        if (len >= 5 && !SDL_memcmp(type + len - 5, "audio", 5)) {
            audio_ports[channels++] = i;
        }
    }
    if (channels == 0) {
        SDL_free(audio_ports);
        return SDL_SetError("No physical JACK ports available");
    }

    /* !!! FIXME: docs say about buffer size: "This size may change, clients that depend on it must register a bufsize_callback so they will be notified if it does." */

    /* Jack pretty much demands what it wants. */
    this->spec.format = AUDIO_F32SYS;
    this->spec.freq = JACK_jack_get_sample_rate(client);
    this->spec.channels = channels;
    this->spec.samples = JACK_jack_get_buffer_size(client);

    SDL_CalculateAudioSpec(&this->spec);

    this->hidden->iosem = SDL_CreateSemaphore(0);
    if (!this->hidden->iosem) {
        SDL_free(audio_ports);
        return -1; /* error was set by SDL_CreateSemaphore */
    }

    this->hidden->iobuffer = (float *)SDL_calloc(1, this->spec.size);
    if (!this->hidden->iobuffer) {
        SDL_free(audio_ports);
        return SDL_OutOfMemory();
    }

    /* Build SDL's ports, which we will connect to the device ports. */
    this->hidden->sdlports = (jack_port_t **)SDL_calloc(channels, sizeof(jack_port_t *));
    if (!this->hidden->sdlports) {
        SDL_free(audio_ports);
        return SDL_OutOfMemory();
    }

    for (i = 0; i < channels; i++) {
        char portname[32];
        (void)SDL_snprintf(portname, sizeof(portname), "sdl_jack_%s_%d", sdlportstr, i);
        this->hidden->sdlports[i] = JACK_jack_port_register(client, portname, JACK_DEFAULT_AUDIO_TYPE, sdlportflags, 0);
        if (this->hidden->sdlports[i] == NULL) {
            SDL_free(audio_ports);
            return SDL_SetError("jack_port_register failed");
        }
    }

    if (JACK_jack_set_process_callback(client, callback, this) != 0) {
        SDL_free(audio_ports);
        return SDL_SetError("JACK: Couldn't set process callback");
    }

    JACK_jack_on_shutdown(client, jackShutdownCallback, this);

    if (JACK_jack_activate(client) != 0) {
        SDL_free(audio_ports);
        return SDL_SetError("Failed to activate JACK client");
    }

    /* once activated, we can connect all the ports. */
    for (i = 0; i < channels; i++) {
        const char *sdlport = JACK_jack_port_name(this->hidden->sdlports[i]);
        const char *srcport = iscapture ? devports[audio_ports[i]] : sdlport;
        const char *dstport = iscapture ? sdlport : devports[audio_ports[i]];
        if (JACK_jack_connect(client, srcport, dstport) != 0) {
            SDL_free(audio_ports);
            return SDL_SetError("Couldn't connect JACK ports: %s => %s", srcport, dstport);
        }
    }

    /* don't need these anymore. */
    JACK_jack_free(devports);
    SDL_free(audio_ports);

    /* We're ready to rock and roll. :-) */
    return 0;
}

static void JACK_Deinitialize(void)
{
    UnloadJackLibrary();
}

static SDL_bool JACK_Init(SDL_AudioDriverImpl *impl)
{
    if (LoadJackLibrary() < 0) {
        return SDL_FALSE;
    } else {
        /* Make sure a JACK server is running and available. */
        jack_status_t status;
        jack_client_t *client = JACK_jack_client_open("SDL", JackNoStartServer, &status, NULL);
        if (!client) {
            UnloadJackLibrary();
            return SDL_FALSE;
        }
        JACK_jack_client_close(client);
    }

    /* Set the function pointers */
    impl->OpenDevice = JACK_OpenDevice;
    impl->WaitDevice = JACK_WaitDevice;
    impl->GetDeviceBuf = JACK_GetDeviceBuf;
    impl->CloseDevice = JACK_CloseDevice;
    impl->Deinitialize = JACK_Deinitialize;
    impl->CaptureFromDevice = JACK_CaptureFromDevice;
    impl->FlushCapture = JACK_FlushCapture;
    impl->OnlyHasDefaultOutputDevice = SDL_TRUE;
    impl->OnlyHasDefaultCaptureDevice = SDL_TRUE;
    impl->HasCaptureSupport = SDL_TRUE;

    return SDL_TRUE; /* this audio target is available. */
}

AudioBootStrap JACK_bootstrap = {
    "jack", "JACK Audio Connection Kit", JACK_Init, SDL_FALSE
};

#endif /* SDL_AUDIO_DRIVER_JACK */

/* vi: set ts=4 sw=4 expandtab: */
