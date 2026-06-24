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

#ifdef SDL_AUDIO_DRIVER_FUSIONSOUND

/* !!! FIXME: why is this is SDL_FS_* instead of FUSIONSOUND_*? */

/* Allow access to a raw mixing buffer */

#ifdef HAVE_SIGNAL_H
#include <signal.h>
#endif
#include <unistd.h>

#include "SDL_timer.h"
#include "SDL_audio.h"
#include "../SDL_audio_c.h"
#include "SDL_fsaudio.h"

#include <fusionsound/fusionsound_version.h>

/* #define SDL_AUDIO_DRIVER_FUSIONSOUND_DYNAMIC "libfusionsound.so" */

#ifdef SDL_AUDIO_DRIVER_FUSIONSOUND_DYNAMIC
#include "SDL_name.h"
#include "SDL_loadso.h"
#else
#define SDL_NAME(X) X
#endif

#if (FUSIONSOUND_MAJOR_VERSION == 1) && (FUSIONSOUND_MINOR_VERSION < 1)
typedef DFBResult DirectResult;
#endif

/* Buffers to use - more than 2 gives a lot of latency */
#define FUSION_BUFFERS              (2)

#ifdef SDL_AUDIO_DRIVER_FUSIONSOUND_DYNAMIC

static const char *fs_library = SDL_AUDIO_DRIVER_FUSIONSOUND_DYNAMIC;
static void *fs_handle = NULL;

static DirectResult (*SDL_NAME(FusionSoundInit)) (int *argc, char *(*argv[]));
static DirectResult (*SDL_NAME(FusionSoundCreate)) (IFusionSound **
                                                   ret_interface);

#define SDL_FS_SYM(x) { #x, (void **) (char *) &SDL_NAME(x) }
static struct
{
    const char *name;
    void **func;
} fs_functions[] = {
/* *INDENT-OFF* */
    SDL_FS_SYM(FusionSoundInit),
    SDL_FS_SYM(FusionSoundCreate),
/* *INDENT-ON* */
};

#undef SDL_FS_SYM

static void UnloadFusionSoundLibrary(void)
{
    if (fs_handle) {
        SDL_UnloadObject(fs_handle);
        fs_handle = NULL;
    }
}

static int LoadFusionSoundLibrary(void)
{
    int i, retval = -1;

    if (!fs_handle) {
        fs_handle = SDL_LoadObject(fs_library);
        if (fs_handle) {
            retval = 0;
            for (i = 0; i < SDL_arraysize(fs_functions); ++i) {
                *fs_functions[i].func =
                    SDL_LoadFunction(fs_handle, fs_functions[i].name);
                if (!*fs_functions[i].func) {
                    retval = -1;
                    UnloadFusionSoundLibrary();
                    break;
                }
            }
        }
    }

    return retval;
}

#else

static void UnloadFusionSoundLibrary(void)
{
    return;
}

static int LoadFusionSoundLibrary(void)
{
    return 0;
}

#endif /* SDL_AUDIO_DRIVER_FUSIONSOUND_DYNAMIC */

/* This function waits until it is possible to write a full sound buffer */
static void SDL_FS_WaitDevice(_THIS)
{
    this->hidden->stream->Wait(this->hidden->stream,
                               this->hidden->mixsamples);
}

static void SDL_FS_PlayDevice(_THIS)
{
    DirectResult ret;

    ret = this->hidden->stream->Write(this->hidden->stream,
                                      this->hidden->mixbuf,
                                      this->hidden->mixsamples);
    /* If we couldn't write, assume fatal error for now */
    if (ret) {
        SDL_OpenedAudioDeviceDisconnected(this);
    }
#ifdef DEBUG_AUDIO
    fprintf(stderr, "Wrote %d bytes of audio data\n", this->hidden->mixlen);
#endif
}


static Uint8 *SDL_FS_GetDeviceBuf(_THIS)
{
    return (this->hidden->mixbuf);
}


static void SDL_FS_CloseDevice(_THIS)
{
    if (this->hidden->stream) {
        this->hidden->stream->Release(this->hidden->stream);
    }
    if (this->hidden->fs) {
        this->hidden->fs->Release(this->hidden->fs);
    }
    SDL_free(this->hidden->mixbuf);
    SDL_free(this->hidden);
}


static int SDL_FS_OpenDevice(_THIS, const char *devname)
{
    int bytes;
    SDL_AudioFormat test_format;
    FSSampleFormat fs_format;
    FSStreamDescription desc;
    DirectResult ret;

    /* Initialize all variables that we clean on shutdown */
    this->hidden = (struct SDL_PrivateAudioData *)SDL_malloc(sizeof(*this->hidden));
    if (!this->hidden) {
        return SDL_OutOfMemory();
    }
    SDL_zerop(this->hidden);

    /* Try for a closest match on audio format */
    for (test_format = SDL_FirstAudioFormat(this->spec.format); test_format; test_format = SDL_NextAudioFormat()) {
#ifdef DEBUG_AUDIO
        fprintf(stderr, "Trying format 0x%4.4x\n", test_format);
#endif
        switch (test_format) {
        case AUDIO_U8:
            fs_format = FSSF_U8;
            break;
        case AUDIO_S16SYS:
            fs_format = FSSF_S16;
            break;
        case AUDIO_S32SYS:
            fs_format = FSSF_S32;
            break;
        case AUDIO_F32SYS:
            fs_format = FSSF_FLOAT;
            break;
        default:
            continue;
        }
        break;
    }

    if (!test_format) {
        return SDL_SetError("%s: Unsupported audio format", "fusionsound");
    }
    this->spec.format = test_format;
    bytes = SDL_AUDIO_BITSIZE(test_format) / 8;

    /* Retrieve the main sound interface. */
    ret = SDL_NAME(FusionSoundCreate) (&this->hidden->fs);
    if (ret) {
        return SDL_SetError("Unable to initialize FusionSound: %d", ret);
    }

    this->hidden->mixsamples = this->spec.size / bytes / this->spec.channels;

    /* Fill stream description. */
    desc.flags = FSSDF_SAMPLERATE | FSSDF_BUFFERSIZE |
        FSSDF_CHANNELS | FSSDF_SAMPLEFORMAT | FSSDF_PREBUFFER;
    desc.samplerate = this->spec.freq;
    desc.buffersize = this->spec.size * FUSION_BUFFERS;
    desc.channels = this->spec.channels;
    desc.prebuffer = 10;
    desc.sampleformat = fs_format;

    ret =
        this->hidden->fs->CreateStream(this->hidden->fs, &desc,
                                       &this->hidden->stream);
    if (ret) {
        return SDL_SetError("Unable to create FusionSoundStream: %d", ret);
    }

    /* See what we got */
    desc.flags = FSSDF_SAMPLERATE | FSSDF_BUFFERSIZE |
        FSSDF_CHANNELS | FSSDF_SAMPLEFORMAT;
    ret = this->hidden->stream->GetDescription(this->hidden->stream, &desc);

    this->spec.freq = desc.samplerate;
    this->spec.size =
        desc.buffersize / FUSION_BUFFERS * bytes * desc.channels;
    this->spec.channels = desc.channels;

    /* Calculate the final parameters for this audio specification */
    SDL_CalculateAudioSpec(&this->spec);

    /* Allocate mixing buffer */
    this->hidden->mixlen = this->spec.size;
    this->hidden->mixbuf = (Uint8 *) SDL_malloc(this->hidden->mixlen);
    if (!this->hidden->mixbuf) {
        return SDL_OutOfMemory();
    }
    SDL_memset(this->hidden->mixbuf, this->spec.silence, this->spec.size);

    /* We're ready to rock and roll. :-) */
    return 0;
}


static void SDL_FS_Deinitialize(void)
{
    UnloadFusionSoundLibrary();
}


static SDL_bool SDL_FS_Init(SDL_AudioDriverImpl * impl)
{
    if (LoadFusionSoundLibrary() < 0) {
        return SDL_FALSE;
    } else {
        DirectResult ret;

        ret = SDL_NAME(FusionSoundInit) (NULL, NULL);
        if (ret) {
            UnloadFusionSoundLibrary();
            SDL_SetError
                ("FusionSound: SDL_FS_init failed (FusionSoundInit: %d)",
                 ret);
            return SDL_FALSE;
        }
    }

    /* Set the function pointers */
    impl->OpenDevice = SDL_FS_OpenDevice;
    impl->PlayDevice = SDL_FS_PlayDevice;
    impl->WaitDevice = SDL_FS_WaitDevice;
    impl->GetDeviceBuf = SDL_FS_GetDeviceBuf;
    impl->CloseDevice = SDL_FS_CloseDevice;
    impl->Deinitialize = SDL_FS_Deinitialize;
    impl->OnlyHasDefaultOutputDevice = SDL_TRUE;

    return SDL_TRUE;   /* this audio target is available. */
}


AudioBootStrap FUSIONSOUND_bootstrap = {
    "fusionsound", "FusionSound", SDL_FS_Init, SDL_FALSE
};

#endif /* SDL_AUDIO_DRIVER_FUSIONSOUND */

/* vi: set ts=4 sw=4 expandtab: */
