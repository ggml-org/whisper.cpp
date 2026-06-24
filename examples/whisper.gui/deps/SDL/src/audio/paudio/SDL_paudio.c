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

#ifdef SDL_AUDIO_DRIVER_PAUDIO

/* Allow access to a raw mixing buffer */

#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "SDL_timer.h"
#include "SDL_audio.h"
#include "SDL_stdinc.h"
#include "../SDL_audio_c.h"
#include "../../core/unix/SDL_poll.h"
#include "SDL_paudio.h"

/* #define DEBUG_AUDIO */

/* A conflict within AIX 4.3.3 <sys/> headers and probably others as well.
 * I guess nobody ever uses audio... Shame over AIX header files.  */
#include <sys/machine.h>
#undef BIG_ENDIAN
#include <sys/audio.h>

/* Open the audio device for playback, and don't block if busy */
/* #define OPEN_FLAGS   (O_WRONLY|O_NONBLOCK) */
#define OPEN_FLAGS  O_WRONLY

/* Get the name of the audio device we use for output */

#ifndef _PATH_DEV_DSP
#define _PATH_DEV_DSP   "/dev/%caud%c/%c"
#endif

static char devsettings[][3] = {
    {'p', '0', '1'}, {'p', '0', '2'}, {'p', '0', '3'}, {'p', '0', '4'},
    {'p', '1', '1'}, {'p', '1', '2'}, {'p', '1', '3'}, {'p', '1', '4'},
    {'p', '2', '1'}, {'p', '2', '2'}, {'p', '2', '3'}, {'p', '2', '4'},
    {'p', '3', '1'}, {'p', '3', '2'}, {'p', '3', '3'}, {'p', '3', '4'},
    {'b', '0', '1'}, {'b', '0', '2'}, {'b', '0', '3'}, {'b', '0', '4'},
    {'b', '1', '1'}, {'b', '1', '2'}, {'b', '1', '3'}, {'b', '1', '4'},
    {'b', '2', '1'}, {'b', '2', '2'}, {'b', '2', '3'}, {'b', '2', '4'},
    {'b', '3', '1'}, {'b', '3', '2'}, {'b', '3', '3'}, {'b', '3', '4'},
    {'\0', '\0', '\0'}
};

static int OpenUserDefinedDevice(char *path, int maxlen, int flags)
{
    const char *audiodev;
    int fd;

    /* Figure out what our audio device is */
    if ((audiodev = SDL_getenv("SDL_PATH_DSP")) == NULL) {
        audiodev = SDL_getenv("AUDIODEV");
    }
    if (!audiodev) {
        return -1;
    }
    fd = open(audiodev, flags, 0);
    if (path) {
        SDL_strlcpy(path, audiodev, maxlen);
        path[maxlen - 1] = '\0';
    }
    return fd;
}

static int OpenAudioPath(char *path, int maxlen, int flags, int classic)
{
    struct stat sb;
    int cycle = 0;
    int fd = OpenUserDefinedDevice(path, maxlen, flags);

    if (fd != -1) {
        return fd;
    }

    /* !!! FIXME: do we really need a table here? */
    while (devsettings[cycle][0] != '\0') {
        char audiopath[1024];
        SDL_snprintf(audiopath, SDL_arraysize(audiopath),
                     _PATH_DEV_DSP,
                     devsettings[cycle][0],
                     devsettings[cycle][1], devsettings[cycle][2]);

        if (stat(audiopath, &sb) == 0) {
            fd = open(audiopath, flags, 0);
            if (fd >= 0) {
                if (path) {
                    SDL_strlcpy(path, audiopath, maxlen);
                }
                return fd;
            }
        }
    }
    return -1;
}

/* This function waits until it is possible to write a full sound buffer */
static void PAUDIO_WaitDevice(_THIS)
{
    fd_set fdset;

    /* See if we need to use timed audio synchronization */
    if (this->hidden->frame_ticks) {
        /* Use timer for general audio synchronization */
        Sint32 ticks;

        ticks = ((Sint32) (this->hidden->next_frame - SDL_GetTicks())) - FUDGE_TICKS;
        if (ticks > 0) {
            SDL_Delay(ticks);
        }
    } else {
        int timeoutMS;
        audio_buffer paud_bufinfo;

        if (ioctl(this->hidden->audio_fd, AUDIO_BUFFER, &paud_bufinfo) < 0) {
#ifdef DEBUG_AUDIO
            fprintf(stderr, "Couldn't get audio buffer information\n");
#endif
            timeoutMS = 10 * 1000;
        } else {
            timeoutMS = paud_bufinfo.write_buf_time;
#ifdef DEBUG_AUDIO
            fprintf(stderr, "Waiting for write_buf_time=%d ms\n", timeoutMS);
#endif
        }

#ifdef DEBUG_AUDIO
        fprintf(stderr, "Waiting for audio to get ready\n");
#endif
        if (SDL_IOReady(this->hidden->audio_fd, SDL_IOR_WRITE, timeoutMS) <= 0) {
            /*
             * In general we should never print to the screen,
             * but in this case we have no other way of letting
             * the user know what happened.
             */
            fprintf(stderr, "SDL: %s - Audio timeout - buggy audio driver? (disabled)\n", strerror(errno));
            SDL_OpenedAudioDeviceDisconnected(this);
            /* Don't try to close - may hang */
            this->hidden->audio_fd = -1;
#ifdef DEBUG_AUDIO
            fprintf(stderr, "Done disabling audio\n");
#endif
        }
#ifdef DEBUG_AUDIO
        fprintf(stderr, "Ready!\n");
#endif
    }
}

static void PAUDIO_PlayDevice(_THIS)
{
    int written = 0;
    const Uint8 *mixbuf = this->hidden->mixbuf;
    const size_t mixlen = this->hidden->mixlen;

    /* Write the audio data, checking for EAGAIN on broken audio drivers */
    do {
        written = write(this->hidden->audio_fd, mixbuf, mixlen);
        if ((written < 0) && ((errno == 0) || (errno == EAGAIN))) {
            SDL_Delay(1);       /* Let a little CPU time go by */
        }
    } while ((written < 0) &&
             ((errno == 0) || (errno == EAGAIN) || (errno == EINTR)));

    /* If timer synchronization is enabled, set the next write frame */
    if (this->hidden->frame_ticks) {
        this->hidden->next_frame += this->hidden->frame_ticks;
    }

    /* If we couldn't write, assume fatal error for now */
    if (written < 0) {
        SDL_OpenedAudioDeviceDisconnected(this);
    }
#ifdef DEBUG_AUDIO
    fprintf(stderr, "Wrote %d bytes of audio data\n", written);
#endif
}

static Uint8 *PAUDIO_GetDeviceBuf(_THIS)
{
    return this->hidden->mixbuf;
}

static void PAUDIO_CloseDevice(_THIS)
{
    if (this->hidden->audio_fd >= 0) {
        close(this->hidden->audio_fd);
    }
    SDL_free(this->hidden->mixbuf);
    SDL_free(this->hidden);
}

static int PAUDIO_OpenDevice(_THIS, const char *devname)
{
    const char *workaround = SDL_getenv("SDL_DSP_NOSELECT");
    char audiodev[1024];
    const char *err = NULL;
    int flags;
    int bytes_per_sample;
    SDL_AudioFormat test_format;
    audio_init paud_init;
    audio_buffer paud_bufinfo;
    audio_control paud_control;
    audio_change paud_change;
    int fd = -1;

    /* Initialize all variables that we clean on shutdown */
    this->hidden = (struct SDL_PrivateAudioData *)SDL_malloc(sizeof(*this->hidden));
    if (!this->hidden) {
        return SDL_OutOfMemory();
    }
    SDL_zerop(this->hidden);

    /* Open the audio device */
    fd = OpenAudioPath(audiodev, sizeof(audiodev), OPEN_FLAGS, 0);
    this->hidden->audio_fd = fd;
    if (fd < 0) {
        return SDL_SetError("Couldn't open %s: %s", audiodev, strerror(errno));
    }

    /*
     * We can't set the buffer size - just ask the device for the maximum
     * that we can have.
     */
    if (ioctl(fd, AUDIO_BUFFER, &paud_bufinfo) < 0) {
        return SDL_SetError("Couldn't get audio buffer information");
    }

    if (this->spec.channels > 1)
        this->spec.channels = 2;
    else
        this->spec.channels = 1;

    /*
     * Fields in the audio_init structure:
     *
     * Ignored by us:
     *
     * paud.loadpath[LOAD_PATH]; * DSP code to load, MWave chip only?
     * paud.slot_number;         * slot number of the adapter
     * paud.device_id;           * adapter identification number
     *
     * Input:
     *
     * paud.srate;           * the sampling rate in Hz
     * paud.bits_per_sample; * 8, 16, 32, ...
     * paud.bsize;           * block size for this rate
     * paud.mode;            * ADPCM, PCM, MU_LAW, A_LAW, SOURCE_MIX
     * paud.channels;        * 1=mono, 2=stereo
     * paud.flags;           * FIXED - fixed length data
     *                       * LEFT_ALIGNED, RIGHT_ALIGNED (var len only)
     *                       * TWOS_COMPLEMENT - 2's complement data
     *                       * SIGNED - signed? comment seems wrong in sys/audio.h
     *                       * BIG_ENDIAN
     * paud.operation;       * PLAY, RECORD
     *
     * Output:
     *
     * paud.flags;           * PITCH            - pitch is supported
     *                       * INPUT            - input is supported
     *                       * OUTPUT           - output is supported
     *                       * MONITOR          - monitor is supported
     *                       * VOLUME           - volume is supported
     *                       * VOLUME_DELAY     - volume delay is supported
     *                       * BALANCE          - balance is supported
     *                       * BALANCE_DELAY    - balance delay is supported
     *                       * TREBLE           - treble control is supported
     *                       * BASS             - bass control is supported
     *                       * BESTFIT_PROVIDED - best fit returned
     *                       * LOAD_CODE        - DSP load needed
     * paud.rc;              * NO_PLAY         - DSP code can't do play requests
     *                       * NO_RECORD       - DSP code can't do record requests
     *                       * INVALID_REQUEST - request was invalid
     *                       * CONFLICT        - conflict with open's flags
     *                       * OVERLOADED      - out of DSP MIPS or memory
     * paud.position_resolution; * smallest increment for position
     */

    paud_init.srate = this->spec.freq;
    paud_init.mode = PCM;
    paud_init.operation = PLAY;
    paud_init.channels = this->spec.channels;

    /* Try for a closest match on audio format */
    for (test_format = SDL_FirstAudioFormat(this->spec.format); test_format; test_format = SDL_NextAudioFormat()) {
#ifdef DEBUG_AUDIO
        fprintf(stderr, "Trying format 0x%4.4x\n", test_format);
#endif
        switch (test_format) {
        case AUDIO_U8:
            flags = TWOS_COMPLEMENT | FIXED;
            break;
        case AUDIO_S8:
            flags = SIGNED | TWOS_COMPLEMENT | FIXED;
            break;
        case AUDIO_S16LSB:
            flags = SIGNED | TWOS_COMPLEMENT | FIXED;
            break;
        case AUDIO_S16MSB:
            flags = BIG_ENDIAN | SIGNED | TWOS_COMPLEMENT | FIXED;
            break;
        case AUDIO_U16LSB:
            flags = TWOS_COMPLEMENT | FIXED;
            break;
        case AUDIO_U16MSB:
            flags = BIG_ENDIAN | TWOS_COMPLEMENT | FIXED;
            break;
        default:
            continue;
        }
        break;
    }
    if (!test_format) {
#ifdef DEBUG_AUDIO
        fprintf(stderr, "Couldn't find any hardware audio formats\n");
#endif
        return SDL_SetError("%s: Unsupported audio format", "paud");
    }
    this->spec.format = test_format;
    paud_init.bits_per_sample = SDL_AUDIO_BITSIZE(test_format);
    bytes_per_sample = SDL_AUDIO_BITSIZE(test_format) / 8;
    paud_init.flags = flags;

    /*
     * We know the buffer size and the max number of subsequent writes
     *  that can be pending. If more than one can pend, allow the application
     *  to do something like double buffering between our write buffer and
     *  the device's own buffer that we are filling with write() anyway.
     *
     * We calculate this->spec.samples like this because
     *  SDL_CalculateAudioSpec() will give put paud_bufinfo.write_buf_cap
     *  (or paud_bufinfo.write_buf_cap/2) into this->spec.size in return.
     */
    if (paud_bufinfo.request_buf_cap == 1) {
        this->spec.samples = paud_bufinfo.write_buf_cap
            / bytes_per_sample / this->spec.channels;
    } else {
        this->spec.samples = paud_bufinfo.write_buf_cap
            / bytes_per_sample / this->spec.channels / 2;
    }
    paud_init.bsize = bytes_per_sample * this->spec.channels;

    SDL_CalculateAudioSpec(&this->spec);

    /*
     * The AIX paud device init can't modify the values of the audio_init
     * structure that we pass to it. So we don't need any recalculation
     * of this stuff and no reinit call as in linux dsp code.
     *
     * /dev/paud supports all of the encoding formats, so we don't need
     * to do anything like reopening the device, either.
     */
    if (ioctl(fd, AUDIO_INIT, &paud_init) < 0) {
        switch (paud_init.rc) {
        case 1:
            err = "DSP can't do play requests";
            break;
        case 2:
            err = "DSP can't do record requests";
            break;
        case 4:
            err = "request was invalid";
            break;
        case 5:
            err = "conflict with open's flags";
            break;
        case 6:
            err = "out of DSP MIPS or memory";
            break;
        default:
            err = "not documented in sys/audio.h";
            break;
        }
        return SDL_SetError("paud: Couldn't set audio format (%s)", err);
    }

    /* Allocate mixing buffer */
    this->hidden->mixlen = this->spec.size;
    this->hidden->mixbuf = (Uint8 *) SDL_malloc(this->hidden->mixlen);
    if (!this->hidden->mixbuf) {
        return SDL_OutOfMemory();
    }
    SDL_memset(this->hidden->mixbuf, this->spec.silence, this->spec.size);

    /*
     * Set some paramters: full volume, first speaker that we can find.
     * Ignore the other settings for now.
     */
    paud_change.input = AUDIO_IGNORE;   /* the new input source */
    paud_change.output = OUTPUT_1;      /* EXTERNAL_SPEAKER,INTERNAL_SPEAKER,OUTPUT_1 */
    paud_change.monitor = AUDIO_IGNORE; /* the new monitor state */
    paud_change.volume = 0x7fffffff;    /* volume level [0-0x7fffffff] */
    paud_change.volume_delay = AUDIO_IGNORE;    /* the new volume delay */
    paud_change.balance = 0x3fffffff;   /* the new balance */
    paud_change.balance_delay = AUDIO_IGNORE;   /* the new balance delay */
    paud_change.treble = AUDIO_IGNORE;  /* the new treble state */
    paud_change.bass = AUDIO_IGNORE;    /* the new bass state */
    paud_change.pitch = AUDIO_IGNORE;   /* the new pitch state */

    paud_control.ioctl_request = AUDIO_CHANGE;
    paud_control.request_info = (char *) &paud_change;
    if (ioctl(fd, AUDIO_CONTROL, &paud_control) < 0) {
#ifdef DEBUG_AUDIO
        fprintf(stderr, "Can't change audio display settings\n");
#endif
    }

    /*
     * Tell the device to expect data. Actual start will wait for
     * the first write() call.
     */
    paud_control.ioctl_request = AUDIO_START;
    paud_control.position = 0;
    if (ioctl(fd, AUDIO_CONTROL, &paud_control) < 0) {
#ifdef DEBUG_AUDIO
        fprintf(stderr, "Can't start audio play\n");
#endif
        return SDL_SetError("Can't start audio play");
    }

    /* Check to see if we need to use SDL_IOReady() workaround */
    if (workaround) {
        this->hidden->frame_ticks = (float) (this->spec.samples * 1000) /
            this->spec.freq;
        this->hidden->next_frame = SDL_GetTicks() + this->hidden->frame_ticks;
    }

    /* We're ready to rock and roll. :-) */
    return 0;
}

static SDL_bool PAUDIO_Init(SDL_AudioDriverImpl * impl)
{
    /* !!! FIXME: not right for device enum? */
    int fd = OpenAudioPath(NULL, 0, OPEN_FLAGS, 0);
    if (fd < 0) {
        SDL_SetError("PAUDIO: Couldn't open audio device");
        return SDL_FALSE;
    }
    close(fd);

    /* Set the function pointers */
    impl->OpenDevice = PAUDIO_OpenDevice;
    impl->PlayDevice = PAUDIO_PlayDevice;
    impl->WaitDevice = PAUDIO_WaitDevice;
    impl->GetDeviceBuf = PAUDIO_GetDeviceBuf;
    impl->CloseDevice = PAUDIO_CloseDevice;
    impl->OnlyHasDefaultOutputDevice = SDL_TRUE;       /* !!! FIXME: add device enum! */

    return SDL_TRUE;   /* this audio target is available. */
}

AudioBootStrap PAUDIO_bootstrap = {
    "paud", "AIX Paudio", PAUDIO_Init, SDL_FALSE
};

#endif /* SDL_AUDIO_DRIVER_PAUDIO */

/* vi: set ts=4 sw=4 expandtab: */
