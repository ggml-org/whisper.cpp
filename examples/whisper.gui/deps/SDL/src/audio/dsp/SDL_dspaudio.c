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

#ifdef SDL_AUDIO_DRIVER_OSS

/* Allow access to a raw mixing buffer */

#include <stdio.h>  /* For perror() */
#include <string.h> /* For strerror() */
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <sys/stat.h>

#include <sys/soundcard.h>

#include "SDL_timer.h"
#include "SDL_audio.h"
#include "../SDL_audio_c.h"
#include "../SDL_audiodev_c.h"
#include "SDL_dspaudio.h"

static void DSP_DetectDevices(void)
{
    SDL_EnumUnixAudioDevices(0, NULL);
}

static void DSP_CloseDevice(_THIS)
{
    if (this->hidden->audio_fd >= 0) {
        close(this->hidden->audio_fd);
    }
    SDL_free(this->hidden->mixbuf);
    SDL_free(this->hidden);
}

static int DSP_OpenDevice(_THIS, const char *devname)
{
    SDL_bool iscapture = this->iscapture;
    const int flags = ((iscapture) ? OPEN_FLAGS_INPUT : OPEN_FLAGS_OUTPUT);
    int format;
    int value;
    int frag_spec;
    SDL_AudioFormat test_format;

    /* We don't care what the devname is...we'll try to open anything. */
    /*  ...but default to first name in the list... */
    if (!devname) {
        devname = SDL_GetAudioDeviceName(0, iscapture);
        if (!devname) {
            return SDL_SetError("No such audio device");
        }
    }

    /* Make sure fragment size stays a power of 2, or OSS fails. */
    /* I don't know which of these are actually legal values, though... */
    if (this->spec.channels > 8) {
        this->spec.channels = 8;
    } else if (this->spec.channels > 4) {
        this->spec.channels = 4;
    } else if (this->spec.channels > 2) {
        this->spec.channels = 2;
    }

    /* Initialize all variables that we clean on shutdown */
    this->hidden = (struct SDL_PrivateAudioData *) SDL_malloc(sizeof(*this->hidden));
    if (!this->hidden) {
        return SDL_OutOfMemory();
    }
    SDL_zerop(this->hidden);

    /* Open the audio device */
    this->hidden->audio_fd = open(devname, flags | O_CLOEXEC, 0);
    if (this->hidden->audio_fd < 0) {
        return SDL_SetError("Couldn't open %s: %s", devname, strerror(errno));
    }

    /* Make the file descriptor use blocking i/o with fcntl() */
    {
        long ctlflags;
        ctlflags = fcntl(this->hidden->audio_fd, F_GETFL);
        ctlflags &= ~O_NONBLOCK;
        if (fcntl(this->hidden->audio_fd, F_SETFL, ctlflags) < 0) {
            return SDL_SetError("Couldn't set audio blocking mode");
        }
    }

    /* Get a list of supported hardware formats */
    if (ioctl(this->hidden->audio_fd, SNDCTL_DSP_GETFMTS, &value) < 0) {
        perror("SNDCTL_DSP_GETFMTS");
        return SDL_SetError("Couldn't get audio format list");
    }

    /* Try for a closest match on audio format */
    format = 0;
    for (test_format = SDL_FirstAudioFormat(this->spec.format);
         !format && test_format;) {
#ifdef DEBUG_AUDIO
        fprintf(stderr, "Trying format 0x%4.4x\n", test_format);
#endif
        switch (test_format) {
        case AUDIO_U8:
            if (value & AFMT_U8) {
                format = AFMT_U8;
            }
            break;
        case AUDIO_S16LSB:
            if (value & AFMT_S16_LE) {
                format = AFMT_S16_LE;
            }
            break;
        case AUDIO_S16MSB:
            if (value & AFMT_S16_BE) {
                format = AFMT_S16_BE;
            }
            break;
#if 0
/*
 * These formats are not used by any real life systems so they are not
 * needed here.
 */
        case AUDIO_S8:
            if (value & AFMT_S8) {
                format = AFMT_S8;
            }
            break;
        case AUDIO_U16LSB:
            if (value & AFMT_U16_LE) {
                format = AFMT_U16_LE;
            }
            break;
        case AUDIO_U16MSB:
            if (value & AFMT_U16_BE) {
                format = AFMT_U16_BE;
            }
            break;
#endif
        default:
            format = 0;
            break;
        }
        if (!format) {
            test_format = SDL_NextAudioFormat();
        }
    }
    if (format == 0) {
        return SDL_SetError("Couldn't find any hardware audio formats");
    }
    this->spec.format = test_format;

    /* Set the audio format */
    value = format;
    if ((ioctl(this->hidden->audio_fd, SNDCTL_DSP_SETFMT, &value) < 0) ||
        (value != format)) {
        perror("SNDCTL_DSP_SETFMT");
        return SDL_SetError("Couldn't set audio format");
    }

    /* Set the number of channels of output */
    value = this->spec.channels;
    if (ioctl(this->hidden->audio_fd, SNDCTL_DSP_CHANNELS, &value) < 0) {
        perror("SNDCTL_DSP_CHANNELS");
        return SDL_SetError("Cannot set the number of channels");
    }
    this->spec.channels = value;

    /* Set the DSP frequency */
    value = this->spec.freq;
    if (ioctl(this->hidden->audio_fd, SNDCTL_DSP_SPEED, &value) < 0) {
        perror("SNDCTL_DSP_SPEED");
        return SDL_SetError("Couldn't set audio frequency");
    }
    this->spec.freq = value;

    /* Calculate the final parameters for this audio specification */
    SDL_CalculateAudioSpec(&this->spec);

    /* Determine the power of two of the fragment size */
    for (frag_spec = 0; (0x01U << frag_spec) < this->spec.size; ++frag_spec) {
    }
    if ((0x01U << frag_spec) != this->spec.size) {
        return SDL_SetError("Fragment size must be a power of two");
    }
    frag_spec |= 0x00020000; /* two fragments, for low latency */

    /* Set the audio buffering parameters */
#ifdef DEBUG_AUDIO
    fprintf(stderr, "Requesting %d fragments of size %d\n",
            (frag_spec >> 16), 1 << (frag_spec & 0xFFFF));
#endif
    if (ioctl(this->hidden->audio_fd, SNDCTL_DSP_SETFRAGMENT, &frag_spec) < 0) {
        perror("SNDCTL_DSP_SETFRAGMENT");
    }
#ifdef DEBUG_AUDIO
    {
        audio_buf_info info;
        ioctl(this->hidden->audio_fd, SNDCTL_DSP_GETOSPACE, &info);
        fprintf(stderr, "fragments = %d\n", info.fragments);
        fprintf(stderr, "fragstotal = %d\n", info.fragstotal);
        fprintf(stderr, "fragsize = %d\n", info.fragsize);
        fprintf(stderr, "bytes = %d\n", info.bytes);
    }
#endif

    /* Allocate mixing buffer */
    if (!iscapture) {
        this->hidden->mixlen = this->spec.size;
        this->hidden->mixbuf = (Uint8 *)SDL_malloc(this->hidden->mixlen);
        if (!this->hidden->mixbuf) {
            return SDL_OutOfMemory();
        }
        SDL_memset(this->hidden->mixbuf, this->spec.silence, this->spec.size);
    }

    /* We're ready to rock and roll. :-) */
    return 0;
}

static void DSP_PlayDevice(_THIS)
{
    struct SDL_PrivateAudioData *h = this->hidden;
    if (write(h->audio_fd, h->mixbuf, h->mixlen) == -1) {
        perror("Audio write");
        SDL_OpenedAudioDeviceDisconnected(this);
    }
#ifdef DEBUG_AUDIO
    fprintf(stderr, "Wrote %d bytes of audio data\n", h->mixlen);
#endif
}

static Uint8 *DSP_GetDeviceBuf(_THIS)
{
    return this->hidden->mixbuf;
}

static int DSP_CaptureFromDevice(_THIS, void *buffer, int buflen)
{
    return (int)read(this->hidden->audio_fd, buffer, buflen);
}

static void DSP_FlushCapture(_THIS)
{
    struct SDL_PrivateAudioData *h = this->hidden;
    audio_buf_info info;
    if (ioctl(h->audio_fd, SNDCTL_DSP_GETISPACE, &info) == 0) {
        while (info.bytes > 0) {
            char buf[512];
            const size_t len = SDL_min(sizeof(buf), info.bytes);
            const ssize_t br = read(h->audio_fd, buf, len);
            if (br <= 0) {
                break;
            }
            info.bytes -= br;
        }
    }
}

static SDL_bool InitTimeDevicesExist = SDL_FALSE;
static int look_for_devices_test(int fd)
{
    InitTimeDevicesExist = SDL_TRUE; /* note that _something_ exists. */
    /* Don't add to the device list, we're just seeing if any devices exist. */
    return 0;
}

static SDL_bool DSP_Init(SDL_AudioDriverImpl *impl)
{
    InitTimeDevicesExist = SDL_FALSE;
    SDL_EnumUnixAudioDevices(0, look_for_devices_test);
    if (!InitTimeDevicesExist) {
        SDL_SetError("dsp: No such audio device");
        return SDL_FALSE; /* maybe try a different backend. */
    }

    /* Set the function pointers */
    impl->DetectDevices = DSP_DetectDevices;
    impl->OpenDevice = DSP_OpenDevice;
    impl->PlayDevice = DSP_PlayDevice;
    impl->GetDeviceBuf = DSP_GetDeviceBuf;
    impl->CloseDevice = DSP_CloseDevice;
    impl->CaptureFromDevice = DSP_CaptureFromDevice;
    impl->FlushCapture = DSP_FlushCapture;

    impl->AllowsArbitraryDeviceNames = SDL_TRUE;
    impl->HasCaptureSupport = SDL_TRUE;

    return SDL_TRUE; /* this audio target is available. */
}

AudioBootStrap DSP_bootstrap = {
    "dsp", "OSS /dev/dsp standard audio", DSP_Init, SDL_FALSE
};

#endif /* SDL_AUDIO_DRIVER_OSS */

/* vi: set ts=4 sw=4 expandtab: */
