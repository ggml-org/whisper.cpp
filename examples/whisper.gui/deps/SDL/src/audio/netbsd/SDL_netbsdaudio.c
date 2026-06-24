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

#ifdef SDL_AUDIO_DRIVER_NETBSD

/*
 * Driver for native NetBSD audio(4).
 * nia@NetBSD.org
 */

#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/audioio.h>

#include "SDL_timer.h"
#include "SDL_audio.h"
#include "../../core/unix/SDL_poll.h"
#include "../SDL_audio_c.h"
#include "../SDL_audiodev_c.h"
#include "SDL_netbsdaudio.h"

/* #define DEBUG_AUDIO */

static void NETBSDAUDIO_DetectDevices(void)
{
    SDL_EnumUnixAudioDevices(0, NULL);
}

static void NETBSDAUDIO_Status(_THIS)
{
#ifdef DEBUG_AUDIO
    /* *INDENT-OFF* */ /* clang-format off */
    audio_info_t info;
    const struct audio_prinfo *prinfo;

    if (ioctl(this->hidden->audio_fd, AUDIO_GETINFO, &info) < 0) {
        fprintf(stderr, "AUDIO_GETINFO failed.\n");
        return;
    }

    prinfo = this->iscapture ? &info.record : &info.play;

    fprintf(stderr, "\n"
            "[%s info]\n"
            "buffer size	:   %d bytes\n"
            "sample rate	:   %i Hz\n"
            "channels	:   %i\n"
            "precision	:   %i-bit\n"
            "encoding	:   0x%x\n"
            "seek		:   %i\n"
            "sample count	:   %i\n"
            "EOF count	:   %i\n"
            "paused		:   %s\n"
            "error occured	:   %s\n"
            "waiting		:   %s\n"
            "active		:   %s\n"
            "",
            this->iscapture ? "record" : "play",
            prinfo->buffer_size,
            prinfo->sample_rate,
            prinfo->channels,
            prinfo->precision,
            prinfo->encoding,
            prinfo->seek,
            prinfo->samples,
            prinfo->eof,
            prinfo->pause ? "yes" : "no",
            prinfo->error ? "yes" : "no",
            prinfo->waiting ? "yes" : "no",
            prinfo->active ? "yes" : "no");

    fprintf(stderr, "\n"
            "[audio info]\n"
            "monitor_gain	:   %i\n"
            "hw block size	:   %d bytes\n"
            "hi watermark	:   %i\n"
            "lo watermark	:   %i\n"
            "audio mode	:   %s\n"
            "",
            info.monitor_gain,
            info.blocksize,
            info.hiwat, info.lowat,
            (info.mode == AUMODE_PLAY) ? "PLAY"
            : (info.mode = AUMODE_RECORD) ? "RECORD"
            : (info.mode == AUMODE_PLAY_ALL ? "PLAY_ALL" : "?"));

    fprintf(stderr, "\n"
            "[audio spec]\n"
            "format		:   0x%x\n"
            "size		:   %u\n"
            "",
            this->spec.format,
            this->spec.size);
    /* *INDENT-ON* */ /* clang-format on */

#endif /* DEBUG_AUDIO */
}

static void NETBSDAUDIO_PlayDevice(_THIS)
{
    struct SDL_PrivateAudioData *h = this->hidden;
    int written;

    /* Write the audio data */
    written = write(h->audio_fd, h->mixbuf, h->mixlen);
    if (written == -1) {
        /* Non recoverable error has occurred. It should be reported!!! */
        SDL_OpenedAudioDeviceDisconnected(this);
        perror("audio");
        return;
    }

#ifdef DEBUG_AUDIO
    fprintf(stderr, "Wrote %d bytes of audio data\n", written);
#endif
}

static Uint8 *NETBSDAUDIO_GetDeviceBuf(_THIS)
{
    return this->hidden->mixbuf;
}

static int NETBSDAUDIO_CaptureFromDevice(_THIS, void *_buffer, int buflen)
{
    Uint8 *buffer = (Uint8 *)_buffer;
    int br;

    br = read(this->hidden->audio_fd, buffer, buflen);
    if (br == -1) {
        /* Non recoverable error has occurred. It should be reported!!! */
        perror("audio");
        return -1;
    }

#ifdef DEBUG_AUDIO
    fprintf(stderr, "Captured %d bytes of audio data\n", br);
#endif
    return 0;
}

static void NETBSDAUDIO_FlushCapture(_THIS)
{
    audio_info_t info;
    size_t remain;
    Uint8 buf[512];

    if (ioctl(this->hidden->audio_fd, AUDIO_GETINFO, &info) < 0) {
        return; /* oh well. */
    }

    remain = (size_t)(info.record.samples * (SDL_AUDIO_BITSIZE(this->spec.format) / 8));
    while (remain > 0) {
        const size_t len = SDL_min(sizeof(buf), remain);
        const int br = read(this->hidden->audio_fd, buf, len);
        if (br <= 0) {
            return; /* oh well. */
        }
        remain -= br;
    }
}

static void NETBSDAUDIO_CloseDevice(_THIS)
{
    if (this->hidden->audio_fd >= 0) {
        close(this->hidden->audio_fd);
    }
    SDL_free(this->hidden->mixbuf);
    SDL_free(this->hidden);
}

static int NETBSDAUDIO_OpenDevice(_THIS, const char *devname)
{
    SDL_bool iscapture = this->iscapture;
    SDL_AudioFormat test_format;
    int encoding = AUDIO_ENCODING_NONE;
    audio_info_t info, hwinfo;
    struct audio_prinfo *prinfo = iscapture ? &info.record : &info.play;

    /* We don't care what the devname is...we'll try to open anything. */
    /*  ...but default to first name in the list... */
    if (!devname) {
        devname = SDL_GetAudioDeviceName(0, iscapture);
        if (!devname) {
            return SDL_SetError("No such audio device");
        }
    }

    /* Initialize all variables that we clean on shutdown */
    this->hidden = (struct SDL_PrivateAudioData *) SDL_malloc(sizeof(*this->hidden));
    if (!this->hidden) {
        return SDL_OutOfMemory();
    }
    SDL_zerop(this->hidden);

    /* Open the audio device */
    this->hidden->audio_fd = open(devname, (iscapture ? O_RDONLY : O_WRONLY) | O_CLOEXEC);
    if (this->hidden->audio_fd < 0) {
        return SDL_SetError("Couldn't open %s: %s", devname, strerror(errno));
    }

    AUDIO_INITINFO(&info);

#ifdef AUDIO_GETFORMAT /* Introduced in NetBSD 9.0 */
    if (ioctl(this->hidden->audio_fd, AUDIO_GETFORMAT, &hwinfo) != -1) {
        /*
         * Use the device's native sample rate so the kernel doesn't have to
         * resample.
         */
        this->spec.freq = iscapture ? hwinfo.record.sample_rate : hwinfo.play.sample_rate;
    }
#endif

    prinfo->sample_rate = this->spec.freq;
    prinfo->channels = this->spec.channels;

    for (test_format = SDL_FirstAudioFormat(this->spec.format); test_format; test_format = SDL_NextAudioFormat()) {
        switch (test_format) {
        case AUDIO_U8:
            encoding = AUDIO_ENCODING_ULINEAR;
            break;
        case AUDIO_S8:
            encoding = AUDIO_ENCODING_SLINEAR;
            break;
        case AUDIO_S16LSB:
            encoding = AUDIO_ENCODING_SLINEAR_LE;
            break;
        case AUDIO_S16MSB:
            encoding = AUDIO_ENCODING_SLINEAR_BE;
            break;
        case AUDIO_U16LSB:
            encoding = AUDIO_ENCODING_ULINEAR_LE;
            break;
        case AUDIO_U16MSB:
            encoding = AUDIO_ENCODING_ULINEAR_BE;
            break;
        case AUDIO_S32LSB:
            encoding = AUDIO_ENCODING_SLINEAR_LE;
            break;
        case AUDIO_S32MSB:
            encoding = AUDIO_ENCODING_SLINEAR_BE;
            break;
        default:
            continue;
        }
        break;
    }

    if (!test_format) {
        return SDL_SetError("%s: Unsupported audio format", "netbsd");
    }
    prinfo->encoding = encoding;
    prinfo->precision = SDL_AUDIO_BITSIZE(test_format);

    info.hiwat = 5;
    info.lowat = 3;
    if (ioctl(this->hidden->audio_fd, AUDIO_SETINFO, &info) < 0) {
        return SDL_SetError("AUDIO_SETINFO failed for %s: %s", devname, strerror(errno));
    }

    if (ioctl(this->hidden->audio_fd, AUDIO_GETINFO, &info) < 0) {
        return SDL_SetError("AUDIO_GETINFO failed for %s: %s", devname, strerror(errno));
    }

    /* Final spec used for the device. */
    this->spec.format = test_format;
    this->spec.freq = prinfo->sample_rate;
    this->spec.channels = prinfo->channels;

    SDL_CalculateAudioSpec(&this->spec);

    if (!iscapture) {
        /* Allocate mixing buffer */
        this->hidden->mixlen = this->spec.size;
        this->hidden->mixbuf = (Uint8 *)SDL_malloc(this->hidden->mixlen);
        if (!this->hidden->mixbuf) {
            return SDL_OutOfMemory();
        }
        SDL_memset(this->hidden->mixbuf, this->spec.silence, this->spec.size);
    }

    NETBSDAUDIO_Status(this);

    /* We're ready to rock and roll. :-) */
    return 0;
}

static SDL_bool NETBSDAUDIO_Init(SDL_AudioDriverImpl *impl)
{
    /* Set the function pointers */
    impl->DetectDevices = NETBSDAUDIO_DetectDevices;
    impl->OpenDevice = NETBSDAUDIO_OpenDevice;
    impl->PlayDevice = NETBSDAUDIO_PlayDevice;
    impl->GetDeviceBuf = NETBSDAUDIO_GetDeviceBuf;
    impl->CloseDevice = NETBSDAUDIO_CloseDevice;
    impl->CaptureFromDevice = NETBSDAUDIO_CaptureFromDevice;
    impl->FlushCapture = NETBSDAUDIO_FlushCapture;

    impl->HasCaptureSupport = SDL_TRUE;
    impl->AllowsArbitraryDeviceNames = SDL_TRUE;

    return SDL_TRUE; /* this audio target is available. */
}

AudioBootStrap NETBSDAUDIO_bootstrap = {
    "netbsd", "NetBSD audio", NETBSDAUDIO_Init, SDL_FALSE
};

#endif /* SDL_AUDIO_DRIVER_NETBSD */

/* vi: set ts=4 sw=4 expandtab: */
