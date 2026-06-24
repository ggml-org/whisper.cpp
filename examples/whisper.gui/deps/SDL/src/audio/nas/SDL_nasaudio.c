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

#ifdef SDL_AUDIO_DRIVER_NAS

/* Allow access to a raw mixing buffer */

#include <signal.h>
#include <unistd.h>

#include "SDL_timer.h"
#include "SDL_audio.h"
#include "SDL_loadso.h"
#include "../SDL_audio_c.h"
#include "SDL_nasaudio.h"

static void (*NAS_AuCloseServer) (AuServer *);
static void (*NAS_AuNextEvent) (AuServer *, AuBool, AuEvent *);
static AuBool(*NAS_AuDispatchEvent) (AuServer *, AuEvent *);
static void (*NAS_AuHandleEvents) (AuServer *);
static AuFlowID(*NAS_AuCreateFlow) (AuServer *, AuStatus *);
static void (*NAS_AuStartFlow) (AuServer *, AuFlowID, AuStatus *);
static void (*NAS_AuSetElements)
  (AuServer *, AuFlowID, AuBool, int, AuElement *, AuStatus *);
static void (*NAS_AuWriteElement)
  (AuServer *, AuFlowID, int, AuUint32, AuPointer, AuBool, AuStatus *);
static AuUint32 (*NAS_AuReadElement)
  (AuServer *, AuFlowID, int, AuUint32, AuPointer, AuStatus *);
static AuServer *(*NAS_AuOpenServer)
  (_AuConst char *, int, _AuConst char *, int, _AuConst char *, char **);
static AuEventHandlerRec *(*NAS_AuRegisterEventHandler)
  (AuServer *, AuMask, int, AuID, AuEventHandlerCallback, AuPointer);


#ifdef SDL_AUDIO_DRIVER_NAS_DYNAMIC

static const char *nas_library = SDL_AUDIO_DRIVER_NAS_DYNAMIC;
static void *nas_handle = NULL;

static int load_nas_sym(const char *fn, void **addr)
{
    *addr = SDL_LoadFunction(nas_handle, fn);
    if (!*addr) {
        return 0;
    }
    return 1;
}

/* cast funcs to char* first, to please GCC's strict aliasing rules. */
#define SDL_NAS_SYM(x) \
    if (!load_nas_sym(#x, (void **) (char *) &NAS_##x)) return -1
#else
#define SDL_NAS_SYM(x) NAS_##x = x
#endif

static int load_nas_syms(void)
{
    SDL_NAS_SYM(AuCloseServer);
    SDL_NAS_SYM(AuNextEvent);
    SDL_NAS_SYM(AuDispatchEvent);
    SDL_NAS_SYM(AuHandleEvents);
    SDL_NAS_SYM(AuCreateFlow);
    SDL_NAS_SYM(AuStartFlow);
    SDL_NAS_SYM(AuSetElements);
    SDL_NAS_SYM(AuWriteElement);
    SDL_NAS_SYM(AuReadElement);
    SDL_NAS_SYM(AuOpenServer);
    SDL_NAS_SYM(AuRegisterEventHandler);
    return 0;
}

#undef SDL_NAS_SYM

#ifdef SDL_AUDIO_DRIVER_NAS_DYNAMIC

static void UnloadNASLibrary(void)
{
    if (nas_handle) {
        SDL_UnloadObject(nas_handle);
        nas_handle = NULL;
    }
}

static int LoadNASLibrary(void)
{
    int retval = 0;
    if (!nas_handle) {
        nas_handle = SDL_LoadObject(nas_library);
        if (!nas_handle) {
            /* Copy error string so we can use it in a new SDL_SetError(). */
            const char *origerr = SDL_GetError();
            const size_t len = SDL_strlen(origerr) + 1;
            char *err = SDL_stack_alloc(char, len);
            SDL_strlcpy(err, origerr, len);
            SDL_SetError("NAS: SDL_LoadObject('%s') failed: %s", nas_library, err);
            SDL_stack_free(err);
            retval = -1;
        } else {
            retval = load_nas_syms();
            if (retval < 0) {
                UnloadNASLibrary();
            }
        }
    }
    return retval;
}

#else

static void UnloadNASLibrary(void)
{
}

static int LoadNASLibrary(void)
{
    load_nas_syms();
    return 0;
}

#endif /* SDL_AUDIO_DRIVER_NAS_DYNAMIC */

/* This function waits until it is possible to write a full sound buffer */
static void NAS_WaitDevice(_THIS)
{
    while (this->hidden->buf_free < this->hidden->mixlen) {
        AuEvent ev;
        NAS_AuNextEvent(this->hidden->aud, AuTrue, &ev);
        NAS_AuDispatchEvent(this->hidden->aud, &ev);
    }
}

static void NAS_PlayDevice(_THIS)
{
    while (this->hidden->mixlen > this->hidden->buf_free) {
        /*
         * We think the buffer is full? Yikes! Ask the server for events,
         *  in the hope that some of them is LowWater events telling us more
         *  of the buffer is free now than what we think.
         */
        AuEvent ev;
        NAS_AuNextEvent(this->hidden->aud, AuTrue, &ev);
        NAS_AuDispatchEvent(this->hidden->aud, &ev);
    }
    this->hidden->buf_free -= this->hidden->mixlen;

    /* Write the audio data */
    NAS_AuWriteElement(this->hidden->aud, this->hidden->flow, 0,
                       this->hidden->mixlen, this->hidden->mixbuf, AuFalse,
                       NULL);

    this->hidden->written += this->hidden->mixlen;

#ifdef DEBUG_AUDIO
    fprintf(stderr, "Wrote %d bytes of audio data\n", this->hidden->mixlen);
#endif
}

static Uint8 *NAS_GetDeviceBuf(_THIS)
{
    return (this->hidden->mixbuf);
}

static int NAS_CaptureFromDevice(_THIS, void *buffer, int buflen)
{
    struct SDL_PrivateAudioData *h = this->hidden;
    int retval;

    while (SDL_TRUE) {
        /* just keep the event queue moving and the server chattering. */
        NAS_AuHandleEvents(h->aud);

        retval = (int) NAS_AuReadElement(h->aud, h->flow, 1, buflen, buffer, NULL);
        /*printf("read %d capture bytes\n", (int) retval);*/
        if (retval == 0) {
            SDL_Delay(10);  /* don't burn the CPU if we're waiting for data. */
        } else {
            break;
        }
    }

    return retval;
}

static void NAS_FlushCapture(_THIS)
{
    struct SDL_PrivateAudioData *h = this->hidden;
    AuUint32 total = 0;
    AuUint32 br;
    Uint8 buf[512];

    do {
        /* just keep the event queue moving and the server chattering. */
        NAS_AuHandleEvents(h->aud);
        br = NAS_AuReadElement(h->aud, h->flow, 1, sizeof(buf), buf, NULL);
        /*printf("flushed %d capture bytes\n", (int) br);*/
        total += br;
    } while ((br == sizeof(buf)) && (total < this->spec.size));
}

static void NAS_CloseDevice(_THIS)
{
    if (this->hidden->aud) {
        NAS_AuCloseServer(this->hidden->aud);
    }
    SDL_free(this->hidden->mixbuf);
    SDL_free(this->hidden);
}

static AuBool event_handler(AuServer * aud, AuEvent * ev, AuEventHandlerRec * hnd)
{
    SDL_AudioDevice *this = (SDL_AudioDevice *) hnd->data;
    struct SDL_PrivateAudioData *h = this->hidden;
    if (this->iscapture) {
        return AuTrue;  /* we don't (currently) care about any of this for capture devices */
    }

    switch (ev->type) {
    case AuEventTypeElementNotify:
        {
            AuElementNotifyEvent *event = (AuElementNotifyEvent *) ev;

            switch (event->kind) {
            case AuElementNotifyKindLowWater:
                if (h->buf_free >= 0) {
                    h->really += event->num_bytes;
                    gettimeofday(&h->last_tv, 0);
                    h->buf_free += event->num_bytes;
                } else {
                    h->buf_free = event->num_bytes;
                }
                break;
            case AuElementNotifyKindState:
                switch (event->cur_state) {
                case AuStatePause:
                    if (event->reason != AuReasonUser) {
                        if (h->buf_free >= 0) {
                            h->really += event->num_bytes;
                            gettimeofday(&h->last_tv, 0);
                            h->buf_free += event->num_bytes;
                        } else {
                            h->buf_free = event->num_bytes;
                        }
                    }
                    break;
                }
            }
        }
    }
    return AuTrue;
}

static AuDeviceID find_device(_THIS)
{
    /* These "Au" things are all macros, not functions... */
    struct SDL_PrivateAudioData *h = this->hidden;
    const unsigned int devicekind = this->iscapture ? AuComponentKindPhysicalInput : AuComponentKindPhysicalOutput;
    const int numdevs = AuServerNumDevices(h->aud);
    const int nch = this->spec.channels;
    int i;

    /* Try to find exact match on channels first... */
    for (i = 0; i < numdevs; i++) {
        const AuDeviceAttributes *dev = AuServerDevice(h->aud, i);
        if ((AuDeviceKind(dev) == devicekind) && (AuDeviceNumTracks(dev) == nch)) {
            return AuDeviceIdentifier(dev);
        }
    }

    /* Take anything, then... */
    for (i = 0; i < numdevs; i++) {
        const AuDeviceAttributes *dev = AuServerDevice(h->aud, i);
        if (AuDeviceKind(dev) == devicekind) {
            this->spec.channels = AuDeviceNumTracks(dev);
            return AuDeviceIdentifier(dev);
        }
    }
    return AuNone;
}

static int NAS_OpenDevice(_THIS, const char *devname)
{
    AuElement elms[3];
    int buffer_size;
    SDL_bool iscapture = this->iscapture;
    SDL_AudioFormat test_format, format = 0;

    /* Initialize all variables that we clean on shutdown */
    this->hidden = (struct SDL_PrivateAudioData *)SDL_malloc(sizeof(*this->hidden));
    if (!this->hidden) {
        return SDL_OutOfMemory();
    }
    SDL_zerop(this->hidden);

    /* Try for a closest match on audio format */
    for (test_format = SDL_FirstAudioFormat(this->spec.format); test_format; test_format = SDL_NextAudioFormat()) {
        switch (test_format) {
        case AUDIO_U8:
            format = AuFormatLinearUnsigned8;
            break;
        case AUDIO_S8:
            format = AuFormatLinearSigned8;
            break;
        case AUDIO_U16LSB:
            format = AuFormatLinearUnsigned16LSB;
            break;
        case AUDIO_U16MSB:
            format = AuFormatLinearUnsigned16MSB;
            break;
        case AUDIO_S16LSB:
            format = AuFormatLinearSigned16LSB;
            break;
        case AUDIO_S16MSB:
            format = AuFormatLinearSigned16MSB;
            break;
        default:
            continue;
        }
        break;
    }
    if (!test_format) {
        return SDL_SetError("%s: Unsupported audio format", "nas");
    }
    this->spec.format = test_format;

    this->hidden->aud = NAS_AuOpenServer("", 0, NULL, 0, NULL, NULL);
    if (this->hidden->aud == 0) {
        return SDL_SetError("NAS: Couldn't open connection to NAS server");
    }

    this->hidden->dev = find_device(this);
    if ((this->hidden->dev == AuNone)
        || (!(this->hidden->flow = NAS_AuCreateFlow(this->hidden->aud, 0)))) {
        return SDL_SetError("NAS: Couldn't find a fitting device on NAS server");
    }

    buffer_size = this->spec.freq;
    if (buffer_size < 4096)
        buffer_size = 4096;

    if (buffer_size > 32768)
        buffer_size = 32768;    /* So that the buffer won't get unmanageably big. */

    /* Calculate the final parameters for this audio specification */
    SDL_CalculateAudioSpec(&this->spec);

    if (iscapture) {
        AuMakeElementImportDevice(elms, this->spec.freq, this->hidden->dev,
                                  AuUnlimitedSamples, 0, NULL);
        AuMakeElementExportClient(elms + 1, 0, this->spec.freq, format,
                                  this->spec.channels, AuTrue, buffer_size,
                                  buffer_size, 0, NULL);
    } else {
        AuMakeElementImportClient(elms, this->spec.freq, format,
                                  this->spec.channels, AuTrue, buffer_size,
                                  buffer_size / 4, 0, NULL);
        AuMakeElementExportDevice(elms + 1, 0, this->hidden->dev, this->spec.freq,
                                  AuUnlimitedSamples, 0, NULL);
    }

    NAS_AuSetElements(this->hidden->aud, this->hidden->flow, AuTrue,
                      2, elms, NULL);

    NAS_AuRegisterEventHandler(this->hidden->aud, AuEventHandlerIDMask, 0,
                               this->hidden->flow, event_handler,
                               (AuPointer) this);

    NAS_AuStartFlow(this->hidden->aud, this->hidden->flow, NULL);

    /* Allocate mixing buffer */
    if (!iscapture) {
        this->hidden->mixlen = this->spec.size;
        this->hidden->mixbuf = (Uint8 *) SDL_malloc(this->hidden->mixlen);
        if (!this->hidden->mixbuf) {
            return SDL_OutOfMemory();
        }
        SDL_memset(this->hidden->mixbuf, this->spec.silence, this->spec.size);
    }

    /* We're ready to rock and roll. :-) */
    return 0;
}

static void NAS_Deinitialize(void)
{
    UnloadNASLibrary();
}

static SDL_bool NAS_Init(SDL_AudioDriverImpl * impl)
{
    if (LoadNASLibrary() < 0) {
        return SDL_FALSE;
    } else {
        AuServer *aud = NAS_AuOpenServer("", 0, NULL, 0, NULL, NULL);
        if (!aud) {
            SDL_SetError("NAS: AuOpenServer() failed (no audio server?)");
            return SDL_FALSE;
        }
        NAS_AuCloseServer(aud);
    }

    /* Set the function pointers */
    impl->OpenDevice = NAS_OpenDevice;
    impl->PlayDevice = NAS_PlayDevice;
    impl->WaitDevice = NAS_WaitDevice;
    impl->GetDeviceBuf = NAS_GetDeviceBuf;
    impl->CaptureFromDevice = NAS_CaptureFromDevice;
    impl->FlushCapture = NAS_FlushCapture;
    impl->CloseDevice = NAS_CloseDevice;
    impl->Deinitialize = NAS_Deinitialize;

    impl->OnlyHasDefaultOutputDevice = SDL_TRUE;
    impl->OnlyHasDefaultCaptureDevice = SDL_TRUE;
    impl->HasCaptureSupport = SDL_TRUE;

    return SDL_TRUE;   /* this audio target is available. */
}

AudioBootStrap NAS_bootstrap = {
    "nas", "Network Audio System", NAS_Init, SDL_FALSE
};

#endif /* SDL_AUDIO_DRIVER_NAS */

/* vi: set ts=4 sw=4 expandtab: */
