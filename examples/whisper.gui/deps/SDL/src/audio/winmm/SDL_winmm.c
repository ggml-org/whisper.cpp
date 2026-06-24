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

#ifdef SDL_AUDIO_DRIVER_WINMM

/* Allow access to a raw mixing buffer */

#include "../../core/windows/SDL_windows.h"
#include <mmsystem.h>

#include "SDL_timer.h"
#include "SDL_audio.h"
#include "../SDL_audio_c.h"
#include "SDL_winmm.h"

/* MinGW32 mmsystem.h doesn't include these structures */
#if defined(__MINGW32__) && defined(_MMSYSTEM_H)

typedef struct tagWAVEINCAPS2W
{
    WORD wMid;
    WORD wPid;
    MMVERSION vDriverVersion;
    WCHAR szPname[MAXPNAMELEN];
    DWORD dwFormats;
    WORD wChannels;
    WORD wReserved1;
    GUID ManufacturerGuid;
    GUID ProductGuid;
    GUID NameGuid;
} WAVEINCAPS2W,*PWAVEINCAPS2W,*NPWAVEINCAPS2W,*LPWAVEINCAPS2W;

typedef struct tagWAVEOUTCAPS2W
{
    WORD wMid;
    WORD wPid;
    MMVERSION vDriverVersion;
    WCHAR szPname[MAXPNAMELEN];
    DWORD dwFormats;
    WORD wChannels;
    WORD wReserved1;
    DWORD dwSupport;
    GUID ManufacturerGuid;
    GUID ProductGuid;
    GUID NameGuid;
} WAVEOUTCAPS2W,*PWAVEOUTCAPS2W,*NPWAVEOUTCAPS2W,*LPWAVEOUTCAPS2W;

#endif /* defined(__MINGW32__) && defined(_MMSYSTEM_H) */

#ifndef WAVE_FORMAT_IEEE_FLOAT
#define WAVE_FORMAT_IEEE_FLOAT 0x0003
#endif

#define DETECT_DEV_IMPL(iscap, typ, capstyp) \
static void DetectWave##typ##Devs(void) { \
    const UINT iscapture = iscap ? 1 : 0; \
    const UINT devcount = wave##typ##GetNumDevs(); \
    capstyp##2W caps; \
    SDL_AudioSpec spec; \
    UINT i; \
    SDL_zero(spec); \
    for (i = 0; i < devcount; i++) { \
        if (wave##typ##GetDevCaps(i,(LP##capstyp##W)&caps,sizeof(caps))==MMSYSERR_NOERROR) { \
            char *name = WIN_LookupAudioDeviceName(caps.szPname,&caps.NameGuid); \
            if (name != NULL) { \
                /* Note that freq/format are not filled in, as this information \
                 * is not provided by the caps struct! At best, we get possible \
                 * sample formats, but not an _active_ format. \
                 */ \
                spec.channels = (Uint8)caps.wChannels; \
                SDL_AddAudioDevice((int) iscapture, name, &spec, (void *) ((size_t) i+1)); \
                SDL_free(name); \
            } \
        } \
    } \
}

DETECT_DEV_IMPL(SDL_FALSE, Out, WAVEOUTCAPS)
DETECT_DEV_IMPL(SDL_TRUE, In, WAVEINCAPS)

static void WINMM_DetectDevices(void)
{
    DetectWaveInDevs();
    DetectWaveOutDevs();
}

static void CALLBACK CaptureSound(HWAVEIN hwi, UINT uMsg, DWORD_PTR dwInstance, DWORD_PTR dwParam1, DWORD_PTR dwParam2)
{
    SDL_AudioDevice *this = (SDL_AudioDevice *) dwInstance;

    /* Only service "buffer is filled" messages */
    if (uMsg != WIM_DATA)
        return;

    /* Signal that we have a new buffer of data */
    ReleaseSemaphore(this->hidden->audio_sem, 1, NULL);
}


/* The Win32 callback for filling the WAVE device */
static void CALLBACK FillSound(HWAVEOUT hwo, UINT uMsg, DWORD_PTR dwInstance, DWORD_PTR dwParam1, DWORD_PTR dwParam2)
{
    SDL_AudioDevice *this = (SDL_AudioDevice *) dwInstance;

    /* Only service "buffer done playing" messages */
    if (uMsg != WOM_DONE)
        return;

    /* Signal that we are done playing a buffer */
    ReleaseSemaphore(this->hidden->audio_sem, 1, NULL);
}

static int SetMMerror(const char *function, MMRESULT code)
{
    int len;
    char errbuf[MAXERRORLENGTH];
    wchar_t werrbuf[MAXERRORLENGTH];

    SDL_snprintf(errbuf, SDL_arraysize(errbuf), "%s: ", function);
    len = SDL_static_cast(int, SDL_strlen(errbuf));

    waveOutGetErrorText(code, werrbuf, MAXERRORLENGTH - len);
    WideCharToMultiByte(CP_ACP, 0, werrbuf, -1, errbuf + len,
                        MAXERRORLENGTH - len, NULL, NULL);

    return SDL_SetError("%s", errbuf);
}

static void WINMM_WaitDevice(_THIS)
{
    /* Wait for an audio chunk to finish */
    WaitForSingleObject(this->hidden->audio_sem, INFINITE);
}

static Uint8 *WINMM_GetDeviceBuf(_THIS)
{
    return (Uint8 *) (this->hidden->
                      wavebuf[this->hidden->next_buffer].lpData);
}

static void WINMM_PlayDevice(_THIS)
{
    /* Queue it up */
    waveOutWrite(this->hidden->hout,
                 &this->hidden->wavebuf[this->hidden->next_buffer],
                 sizeof(this->hidden->wavebuf[0]));
    this->hidden->next_buffer = (this->hidden->next_buffer + 1) % NUM_BUFFERS;
}

static int WINMM_CaptureFromDevice(_THIS, void *buffer, int buflen)
{
    const int nextbuf = this->hidden->next_buffer;
    MMRESULT result;

    SDL_assert(buflen == this->spec.size);

    /* Wait for an audio chunk to finish */
    WaitForSingleObject(this->hidden->audio_sem, INFINITE);

    /* Copy it to caller's buffer... */
    SDL_memcpy(buffer, this->hidden->wavebuf[nextbuf].lpData, this->spec.size);

    /* requeue the buffer that just finished. */
    result = waveInAddBuffer(this->hidden->hin,
                             &this->hidden->wavebuf[nextbuf],
                             sizeof(this->hidden->wavebuf[nextbuf]));
    if (result != MMSYSERR_NOERROR) {
        return -1;  /* uhoh! Disable the device. */
    }

    /* queue the next buffer in sequence, next time. */
    this->hidden->next_buffer = (nextbuf + 1) % NUM_BUFFERS;
    return this->spec.size;
}

static void WINMM_FlushCapture(_THIS)
{
    /* Wait for an audio chunk to finish */
    if (WaitForSingleObject(this->hidden->audio_sem, 0) == WAIT_OBJECT_0) {
        const int nextbuf = this->hidden->next_buffer;
        /* requeue the buffer that just finished without reading from it. */
        waveInAddBuffer(this->hidden->hin,
                        &this->hidden->wavebuf[nextbuf],
                        sizeof(this->hidden->wavebuf[nextbuf]));
        this->hidden->next_buffer = (nextbuf + 1) % NUM_BUFFERS;
    }
}

static void WINMM_CloseDevice(_THIS)
{
    int i;

    if (this->hidden->hout) {
        waveOutReset(this->hidden->hout);

        /* Clean up mixing buffers */
        for (i = 0; i < NUM_BUFFERS; ++i) {
            if (this->hidden->wavebuf[i].dwUser != 0xFFFF) {
                waveOutUnprepareHeader(this->hidden->hout,
                                       &this->hidden->wavebuf[i],
                                       sizeof(this->hidden->wavebuf[i]));
            }
        }

        waveOutClose(this->hidden->hout);
    }

    if (this->hidden->hin) {
        waveInReset(this->hidden->hin);

        /* Clean up mixing buffers */
        for (i = 0; i < NUM_BUFFERS; ++i) {
            if (this->hidden->wavebuf[i].dwUser != 0xFFFF) {
                waveInUnprepareHeader(this->hidden->hin,
                                       &this->hidden->wavebuf[i],
                                       sizeof(this->hidden->wavebuf[i]));
            }
        }
        waveInClose(this->hidden->hin);
    }

    if (this->hidden->audio_sem) {
        CloseHandle(this->hidden->audio_sem);
    }

    SDL_free(this->hidden->mixbuf);
    SDL_free(this->hidden);
}

static SDL_bool PrepWaveFormat(_THIS, UINT devId, WAVEFORMATEX *pfmt, const int iscapture)
{
    SDL_zerop(pfmt);

    if (SDL_AUDIO_ISFLOAT(this->spec.format)) {
        pfmt->wFormatTag = WAVE_FORMAT_IEEE_FLOAT;
    } else {
        pfmt->wFormatTag = WAVE_FORMAT_PCM;
    }
    pfmt->wBitsPerSample = SDL_AUDIO_BITSIZE(this->spec.format);

    pfmt->nChannels = this->spec.channels;
    pfmt->nSamplesPerSec = this->spec.freq;
    pfmt->nBlockAlign = pfmt->nChannels * (pfmt->wBitsPerSample / 8);
    pfmt->nAvgBytesPerSec = pfmt->nSamplesPerSec * pfmt->nBlockAlign;

    if (iscapture) {
        return (waveInOpen(0, devId, pfmt, 0, 0, WAVE_FORMAT_QUERY) == 0);
    } else {
        return (waveOutOpen(0, devId, pfmt, 0, 0, WAVE_FORMAT_QUERY) == 0);
    }
}

static int WINMM_OpenDevice(_THIS, const char *devname)
{
    SDL_AudioFormat test_format;
    SDL_bool iscapture = this->iscapture;
    void *handle = this->handle;
    MMRESULT result;
    WAVEFORMATEX waveformat;
    UINT devId = WAVE_MAPPER;  /* WAVE_MAPPER == choose system's default */
    UINT i;

    if (handle) {  /* specific device requested? */
        /* -1 because we increment the original value to avoid NULL. */
        const size_t val = ((size_t) handle) - 1;
        devId = (UINT) val;
    }

    /* Initialize all variables that we clean on shutdown */
    this->hidden = (struct SDL_PrivateAudioData *)SDL_malloc(sizeof(*this->hidden));
    if (!this->hidden) {
        return SDL_OutOfMemory();
    }
    SDL_zerop(this->hidden);

    /* Initialize the wavebuf structures for closing */
    for (i = 0; i < NUM_BUFFERS; ++i)
        this->hidden->wavebuf[i].dwUser = 0xFFFF;

    if (this->spec.channels > 2)
        this->spec.channels = 2;        /* !!! FIXME: is this right? */

    for (test_format = SDL_FirstAudioFormat(this->spec.format); test_format; test_format = SDL_NextAudioFormat()) {
        switch (test_format) {
        case AUDIO_U8:
        case AUDIO_S16:
        case AUDIO_S32:
        case AUDIO_F32:
            this->spec.format = test_format;
            if (PrepWaveFormat(this, devId, &waveformat, iscapture)) {
                break;
            }
            continue;
        default:
            continue;
        }
        break;
    }

    if (!test_format) {
        return SDL_SetError("%s: Unsupported audio format", "winmm");
    }

    /* Update the fragment size as size in bytes */
    SDL_CalculateAudioSpec(&this->spec);

    /* Open the audio device */
    if (iscapture) {
        result = waveInOpen(&this->hidden->hin, devId, &waveformat,
                             (DWORD_PTR) CaptureSound, (DWORD_PTR) this,
                             CALLBACK_FUNCTION);
        if (result != MMSYSERR_NOERROR) {
            return SetMMerror("waveInOpen()", result);
        }
    } else {
        result = waveOutOpen(&this->hidden->hout, devId, &waveformat,
                             (DWORD_PTR) FillSound, (DWORD_PTR) this,
                             CALLBACK_FUNCTION);
        if (result != MMSYSERR_NOERROR) {
            return SetMMerror("waveOutOpen()", result);
        }
    }

#ifdef SOUND_DEBUG
    /* Check the sound device we retrieved */
    {
        if (iscapture) {
            WAVEINCAPS caps;
            result = waveInGetDevCaps((UINT) this->hidden->hout,
                                      &caps, sizeof(caps));
            if (result != MMSYSERR_NOERROR) {
                return SetMMerror("waveInGetDevCaps()", result);
            }
            printf("Audio device: %s\n", caps.szPname);
        } else {
            WAVEOUTCAPS caps;
            result = waveOutGetDevCaps((UINT) this->hidden->hout,
                                       &caps, sizeof(caps));
            if (result != MMSYSERR_NOERROR) {
                return SetMMerror("waveOutGetDevCaps()", result);
            }
            printf("Audio device: %s\n", caps.szPname);
        }
    }
#endif

    /* Create the audio buffer semaphore */
    this->hidden->audio_sem = CreateSemaphore(NULL, iscapture ? 0 : NUM_BUFFERS - 1, NUM_BUFFERS, NULL);
    if (!this->hidden->audio_sem) {
        return SDL_SetError("Couldn't create semaphore");
    }

    /* Create the sound buffers */
    this->hidden->mixbuf =
        (Uint8 *) SDL_malloc(NUM_BUFFERS * this->spec.size);
    if (!this->hidden->mixbuf) {
        return SDL_OutOfMemory();
    }

    SDL_zeroa(this->hidden->wavebuf);
    for (i = 0; i < NUM_BUFFERS; ++i) {
        this->hidden->wavebuf[i].dwBufferLength = this->spec.size;
        this->hidden->wavebuf[i].dwFlags = WHDR_DONE;
        this->hidden->wavebuf[i].lpData =
            (LPSTR) & this->hidden->mixbuf[i * this->spec.size];

        if (iscapture) {
            result = waveInPrepareHeader(this->hidden->hin,
                                          &this->hidden->wavebuf[i],
                                          sizeof(this->hidden->wavebuf[i]));
            if (result != MMSYSERR_NOERROR) {
                return SetMMerror("waveInPrepareHeader()", result);
            }

            result = waveInAddBuffer(this->hidden->hin,
                                     &this->hidden->wavebuf[i],
                                     sizeof(this->hidden->wavebuf[i]));
            if (result != MMSYSERR_NOERROR) {
                return SetMMerror("waveInAddBuffer()", result);
            }
        } else {
            result = waveOutPrepareHeader(this->hidden->hout,
                                          &this->hidden->wavebuf[i],
                                          sizeof(this->hidden->wavebuf[i]));
            if (result != MMSYSERR_NOERROR) {
                return SetMMerror("waveOutPrepareHeader()", result);
            }
        }
    }

    if (iscapture) {
        result = waveInStart(this->hidden->hin);
        if (result != MMSYSERR_NOERROR) {
            return SetMMerror("waveInStart()", result);
        }
    }

    return 0;                   /* Ready to go! */
}

static SDL_bool WINMM_Init(SDL_AudioDriverImpl * impl)
{
    /* Set the function pointers */
    impl->DetectDevices = WINMM_DetectDevices;
    impl->OpenDevice = WINMM_OpenDevice;
    impl->PlayDevice = WINMM_PlayDevice;
    impl->WaitDevice = WINMM_WaitDevice;
    impl->GetDeviceBuf = WINMM_GetDeviceBuf;
    impl->CaptureFromDevice = WINMM_CaptureFromDevice;
    impl->FlushCapture = WINMM_FlushCapture;
    impl->CloseDevice = WINMM_CloseDevice;

    impl->HasCaptureSupport = SDL_TRUE;

    return SDL_TRUE;   /* this audio target is available. */
}

AudioBootStrap WINMM_bootstrap = {
    "winmm", "Windows Waveform Audio", WINMM_Init, SDL_FALSE
};

#endif /* SDL_AUDIO_DRIVER_WINMM */

/* vi: set ts=4 sw=4 expandtab: */
