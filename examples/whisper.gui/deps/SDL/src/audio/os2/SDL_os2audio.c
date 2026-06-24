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

#ifdef SDL_AUDIO_DRIVER_OS2

/* Allow access to a raw mixing buffer */

#include "../../core/os2/SDL_os2.h"

#include "SDL_audio.h"
#include "../SDL_audio_c.h"
#include "SDL_os2audio.h"

static PMCI_MIX_BUFFER _getNextBuffer(SDL_PrivateAudioData *pAData, PMCI_MIX_BUFFER pBuffer)
{
    PMCI_MIX_BUFFER pFirstBuffer = &pAData->aMixBuffers[0];
    PMCI_MIX_BUFFER pLastBuffer = &pAData->aMixBuffers[pAData->cMixBuffers -1];
    return (pBuffer == pLastBuffer ? pFirstBuffer : pBuffer+1);
}

static ULONG _getEnvULong(const char *name, ULONG ulMax, ULONG ulDefault)
{
    ULONG   ulValue;
    char*   end;
    char*   envval = SDL_getenv(name);

    if (!envval)
        return ulDefault;

    ulValue = SDL_strtoul(envval, &end, 10);
    return (end == envval) || (ulValue > ulMax)? ulDefault : ulMax;
}

static int _MCIError(const char *func, ULONG ulResult)
{
    CHAR    acBuf[128];
    mciGetErrorString(ulResult, acBuf, sizeof(acBuf));
    return SDL_SetError("[%s] %s", func, acBuf);
}

static void _mixIOError(const char *function, ULONG ulRC)
{
    debug_os2("%s() - failed, rc = 0x%lX (%s)",
              function, ulRC,
              (ulRC == MCIERR_INVALID_MODE)   ? "Mixer mode does not match request" :
              (ulRC == MCIERR_INVALID_BUFFER) ? "Caller sent an invalid buffer"     : "unknown");
}

static LONG APIENTRY cbAudioWriteEvent(ULONG ulStatus, PMCI_MIX_BUFFER pBuffer,
                                       ULONG ulFlags)
{
    SDL_AudioDevice      *_this = (SDL_AudioDevice *)pBuffer->ulUserParm;
    SDL_PrivateAudioData *pAData = (SDL_PrivateAudioData *)_this->hidden;
    ULONG   ulRC;

    debug_os2("cbAudioWriteEvent: ulStatus = %lu, pBuffer = %p, ulFlags = %#lX",ulStatus,pBuffer,ulFlags);

    if (pAData->ulState == 2)
    {
        return 0;
    }

    if (ulFlags != MIX_WRITE_COMPLETE) {
        debug_os2("flags = 0x%lX", ulFlags);
        return 0;
    }

    pAData->pDrainBuffer = pBuffer;
    ulRC = pAData->stMCIMixSetup.pmixWrite(pAData->stMCIMixSetup.ulMixHandle,
                                           pAData->pDrainBuffer, 1);
    if (ulRC != MCIERR_SUCCESS) {
        _mixIOError("pmixWrite", ulRC);
        return 0;
    }

    ulRC = DosPostEventSem(pAData->hevBuf);
    if (ulRC != NO_ERROR && ulRC != ERROR_ALREADY_POSTED) {
        debug_os2("DosPostEventSem(), rc = %lu", ulRC);
    }

    return 1; /* return value doesn't seem to matter. */
}

static LONG APIENTRY cbAudioReadEvent(ULONG ulStatus, PMCI_MIX_BUFFER pBuffer,
                                      ULONG ulFlags)
{
    SDL_AudioDevice      *_this = (SDL_AudioDevice *)pBuffer->ulUserParm;
    SDL_PrivateAudioData *pAData = (SDL_PrivateAudioData *)_this->hidden;
    ULONG   ulRC;

    debug_os2("cbAudioReadEvent: ulStatus = %lu, pBuffer = %p, ulFlags = %#lX",ulStatus,pBuffer,ulFlags);

    if (pAData->ulState == 2)
    {
        return 0;
    }

    if (ulFlags != MIX_READ_COMPLETE) {
        debug_os2("flags = 0x%lX", ulFlags);
        return 0;
    }

    pAData->pFillBuffer = pBuffer;
    if (pAData->pFillBuffer == pAData->aMixBuffers)
    {
       ulRC = pAData->stMCIMixSetup.pmixRead(pAData->stMCIMixSetup.ulMixHandle,
                                             pAData->pFillBuffer, pAData->cMixBuffers);
       if (ulRC != MCIERR_SUCCESS) {
           _mixIOError("pmixRead", ulRC);
           return 0;
       }
    }

    ulRC = DosPostEventSem(pAData->hevBuf);
    if (ulRC != NO_ERROR && ulRC != ERROR_ALREADY_POSTED) {
        debug_os2("DosPostEventSem(), rc = %lu", ulRC);
    }

    return 1;
}


static void OS2_DetectDevices(void)
{
    MCI_SYSINFO_PARMS       stMCISysInfo;
    CHAR                    acBuf[256];
    ULONG                   ulDevicesNum;
    MCI_SYSINFO_LOGDEVICE   stLogDevice;
    MCI_SYSINFO_PARMS       stSysInfoParams;
    ULONG                   ulRC;
    ULONG                   ulNumber;
    MCI_GETDEVCAPS_PARMS    stDevCapsParams;
    MCI_OPEN_PARMS          stMCIOpen;
    MCI_GENERIC_PARMS       stMCIGenericParams;

    SDL_memset(&stMCISysInfo, 0, sizeof(stMCISysInfo));
    acBuf[0] = '\0';
    stMCISysInfo.pszReturn    = acBuf;
    stMCISysInfo.ulRetSize    = sizeof(acBuf);
    stMCISysInfo.usDeviceType = MCI_DEVTYPE_AUDIO_AMPMIX;
    ulRC = mciSendCommand(0, MCI_SYSINFO, MCI_WAIT | MCI_SYSINFO_QUANTITY,
                          &stMCISysInfo, 0);
    if (LOUSHORT(ulRC) != MCIERR_SUCCESS) {
        debug_os2("MCI_SYSINFO, MCI_SYSINFO_QUANTITY - failed, rc = 0x%hX", LOUSHORT(ulRC));
        return;
    }

    ulDevicesNum = SDL_strtoul(stMCISysInfo.pszReturn, NULL, 10);

    for (ulNumber = 1; ulNumber <= ulDevicesNum;
         ulNumber++) {
        /* Get device install name. */
        stSysInfoParams.ulNumber     = ulNumber;
        stSysInfoParams.pszReturn    = acBuf;
        stSysInfoParams.ulRetSize    = sizeof(acBuf);
        stSysInfoParams.usDeviceType = MCI_DEVTYPE_AUDIO_AMPMIX;
        ulRC = mciSendCommand(0, MCI_SYSINFO, MCI_WAIT | MCI_SYSINFO_INSTALLNAME,
                              &stSysInfoParams, 0);
        if (LOUSHORT(ulRC) != MCIERR_SUCCESS) {
            debug_os2("MCI_SYSINFO, MCI_SYSINFO_INSTALLNAME - failed, rc = 0x%hX", LOUSHORT(ulRC));
            continue;
        }

        /* Get textual product description. */
        stSysInfoParams.ulItem = MCI_SYSINFO_QUERY_DRIVER;
        stSysInfoParams.pSysInfoParm = &stLogDevice;
        SDL_strlcpy(stLogDevice.szInstallName, stSysInfoParams.pszReturn, MAX_DEVICE_NAME);
        ulRC = mciSendCommand(0, MCI_SYSINFO, MCI_WAIT | MCI_SYSINFO_ITEM,
                              &stSysInfoParams, 0);
        if (LOUSHORT(ulRC) != MCIERR_SUCCESS) {
            debug_os2("MCI_SYSINFO, MCI_SYSINFO_ITEM - failed, rc = 0x%hX", LOUSHORT(ulRC));
            continue;
        }

        SDL_AddAudioDevice(0, stLogDevice.szProductInfo, NULL, (void *)ulNumber);

        /* Open audio device for querying its capabilities */
        /* at this point we HAVE TO OPEN the waveaudio device and not the ampmix device */
        /* because only the waveaudio device (tied to the ampmix device) supports querying for playback/record capability */
        SDL_memset(&stMCIOpen, 0, sizeof(stMCIOpen));
        stMCIOpen.pszDeviceType = (PSZ)MAKEULONG(MCI_DEVTYPE_WAVEFORM_AUDIO,LOUSHORT(ulNumber));
        ulRC = mciSendCommand(0, MCI_OPEN,MCI_WAIT | MCI_OPEN_TYPE_ID | MCI_OPEN_SHAREABLE,&stMCIOpen,  0);
        if (LOUSHORT(ulRC) != MCIERR_SUCCESS) {
            debug_os2("MCI_OPEN (getDevCaps) - failed");
            continue;
        }

        /* check for recording capability */
        SDL_memset(&stDevCapsParams, 0, sizeof(stDevCapsParams));
        stDevCapsParams.ulItem = MCI_GETDEVCAPS_CAN_RECORD;
        ulRC = mciSendCommand(stMCIOpen.usDeviceID, MCI_GETDEVCAPS, MCI_WAIT | MCI_GETDEVCAPS_ITEM,
                              &stDevCapsParams, 0);
        if (LOUSHORT(ulRC) != MCIERR_SUCCESS) {
            debug_os2("MCI_GETDEVCAPS, MCI_GETDEVCAPS_ITEM - failed, rc = 0x%hX", LOUSHORT(ulRC));
        }
        else {
            if (stDevCapsParams.ulReturn) {
                SDL_AddAudioDevice(1, stLogDevice.szProductInfo, NULL, (void *)(ulNumber | 0x80000000));
            }
        }

        /* close the audio device, we are done querying its capabilities */
        SDL_memset(&stMCIGenericParams, 0, sizeof(stMCIGenericParams));
        ulRC = mciSendCommand(stMCIOpen.usDeviceID, MCI_CLOSE, MCI_WAIT,
                              &stMCIGenericParams, 0);
        if (LOUSHORT(ulRC) != MCIERR_SUCCESS) {
            debug_os2("MCI_CLOSE (getDevCaps) - failed");
        }
    }
}

static void OS2_WaitDevice(_THIS)
{
    SDL_PrivateAudioData *pAData = (SDL_PrivateAudioData *)_this->hidden;
    ULONG   ulRC;

    debug_os2("Enter");

    /* Wait for an audio chunk to finish */
    ulRC = DosWaitEventSem(pAData->hevBuf, 5000);
    if (ulRC != NO_ERROR) {
        debug_os2("DosWaitEventSem(), rc = %lu", ulRC);
    }
}

static Uint8 *OS2_GetDeviceBuf(_THIS)
{
    SDL_PrivateAudioData *pAData = (SDL_PrivateAudioData *)_this->hidden;

    debug_os2("Enter");

    return (Uint8 *) pAData->pFillBuffer->pBuffer;
}

static void OS2_PlayDevice(_THIS)
{
    SDL_PrivateAudioData *pAData = (SDL_PrivateAudioData *)_this->hidden;
    ULONG                 ulRC;
    PMCI_MIX_BUFFER       pMixBuffer = NULL;

    debug_os2("Enter");

    pMixBuffer  = pAData->pDrainBuffer;
    pAData->pFillBuffer = _getNextBuffer(pAData, pAData->pFillBuffer);
    if (!pAData->ulState && pAData->pFillBuffer != pMixBuffer)
    {
        /*
         * this buffer was filled but we have not yet filled all buffers
         * so just signal event sem so that OS2_WaitDevice does not need
         * to block
         */
        ulRC = DosPostEventSem(pAData->hevBuf);
    }

    if (!pAData->ulState && (pAData->pFillBuffer == pMixBuffer) )
    {
        debug_os2("!hasStarted");
        pAData->ulState = 1;

        /* Write buffers to kick off the amp mixer */
        ulRC = pAData->stMCIMixSetup.pmixWrite(pAData->stMCIMixSetup.ulMixHandle,
                                               pMixBuffer, pAData->cMixBuffers);

        if (ulRC != MCIERR_SUCCESS) {
            _mixIOError("pmixWrite", ulRC);
        }
    }
}

static int OS2_CaptureFromDevice(_THIS,void *buffer,int buflen)
{
    SDL_PrivateAudioData *pAData = (SDL_PrivateAudioData *)_this->hidden;
    ULONG                 ulRC;
    PMCI_MIX_BUFFER       pMixBuffer = NULL;
    int                   len = 0;

    if (!pAData->ulState)
    {
        pAData->ulState = 1;
        ulRC = pAData->stMCIMixSetup.pmixRead(pAData->stMCIMixSetup.ulMixHandle,
                                              pAData->aMixBuffers, pAData->cMixBuffers);
        if (ulRC != MCIERR_SUCCESS) {
            _mixIOError("pmixRead", ulRC);
            return -1;
        }
    }

    /* Wait for an audio chunk to finish */
    ulRC = DosWaitEventSem(pAData->hevBuf, 5000);
    if (ulRC != NO_ERROR)
    {
        debug_os2("DosWaitEventSem(), rc = %lu", ulRC);
        return -1;
    }

    pMixBuffer = pAData->pDrainBuffer;
    len = SDL_min((int)pMixBuffer->ulBufferLength, buflen);
    SDL_memcpy(buffer,pMixBuffer->pBuffer, len);
    pAData->pDrainBuffer = _getNextBuffer(pAData, pMixBuffer);

    debug_os2("buflen = %u, ulBufferLength = %lu",buflen,pMixBuffer->ulBufferLength);

    return len;
}

static void OS2_FlushCapture(_THIS)
{
    SDL_PrivateAudioData *pAData = (SDL_PrivateAudioData *)_this->hidden;
    ULONG                 ulIdx;

    debug_os2("Enter");

    /* Fill all device buffers with data */
    for (ulIdx = 0; ulIdx < pAData->cMixBuffers; ulIdx++) {
        pAData->aMixBuffers[ulIdx].ulFlags        = 0;
        pAData->aMixBuffers[ulIdx].ulBufferLength = _this->spec.size;
        pAData->aMixBuffers[ulIdx].ulUserParm     = (ULONG)_this;

        SDL_memset(((PMCI_MIX_BUFFER)pAData->aMixBuffers)[ulIdx].pBuffer,
                   _this->spec.silence, _this->spec.size);
    }
    pAData->pFillBuffer  = pAData->aMixBuffers;
    pAData->pDrainBuffer = pAData->aMixBuffers;
}


static void OS2_CloseDevice(_THIS)
{
    SDL_PrivateAudioData *pAData = (SDL_PrivateAudioData *)_this->hidden;
    MCI_GENERIC_PARMS     sMCIGenericParms;
    ULONG                 ulRC;

    debug_os2("Enter");

    if (!pAData)
        return;

    pAData->ulState = 2;

    /* Close up audio */
    if (pAData->usDeviceId != (USHORT)~0) { /* Device is open. */
            SDL_zero(sMCIGenericParms);

            ulRC = mciSendCommand(pAData->usDeviceId, MCI_STOP,
                                  MCI_WAIT,
                                  &sMCIGenericParms, 0);
            if (LOUSHORT(ulRC) != MCIERR_SUCCESS) {
                debug_os2("MCI_STOP - failed" );
            }

        if (pAData->stMCIMixSetup.ulBitsPerSample != 0) { /* Mixer was initialized. */
            ulRC = mciSendCommand(pAData->usDeviceId, MCI_MIXSETUP,
                                  MCI_WAIT | MCI_MIXSETUP_DEINIT,
                                  &pAData->stMCIMixSetup, 0);
            if (LOUSHORT(ulRC) != MCIERR_SUCCESS) {
                debug_os2("MCI_MIXSETUP, MCI_MIXSETUP_DEINIT - failed");
            }
        }

        if (pAData->cMixBuffers != 0) { /* Buffers was allocated. */
            MCI_BUFFER_PARMS    stMCIBuffer;

            stMCIBuffer.ulBufferSize = pAData->aMixBuffers[0].ulBufferLength;
            stMCIBuffer.ulNumBuffers = pAData->cMixBuffers;
            stMCIBuffer.pBufList = pAData->aMixBuffers;

            ulRC = mciSendCommand(pAData->usDeviceId, MCI_BUFFER,
                                  MCI_WAIT | MCI_DEALLOCATE_MEMORY, &stMCIBuffer, 0);
            if (LOUSHORT(ulRC) != MCIERR_SUCCESS) {
                debug_os2("MCI_BUFFER, MCI_DEALLOCATE_MEMORY - failed");
            }
        }

        ulRC = mciSendCommand(pAData->usDeviceId, MCI_CLOSE, MCI_WAIT,
                              &sMCIGenericParms, 0);
        if (LOUSHORT(ulRC) != MCIERR_SUCCESS) {
            debug_os2("MCI_CLOSE - failed");
        }
    }

    if (pAData->hevBuf != NULLHANDLE)
        DosCloseEventSem(pAData->hevBuf);

    SDL_free(pAData);
}

static int OS2_OpenDevice(_THIS, const char *devname)
{
    SDL_PrivateAudioData *pAData;
    SDL_AudioFormat       test_format;
    MCI_AMP_OPEN_PARMS    stMCIAmpOpen;
    MCI_BUFFER_PARMS      stMCIBuffer;
    ULONG                 ulRC;
    ULONG                 ulIdx;
    BOOL                  new_freq;
    ULONG                 ulHandle = (ULONG)_this->handle;
    SDL_bool              iscapture = _this->iscapture;

    new_freq = FALSE;
    SDL_zero(stMCIAmpOpen);
    SDL_zero(stMCIBuffer);

    for (test_format = SDL_FirstAudioFormat(_this->spec.format); test_format; test_format = SDL_NextAudioFormat()) {
        if (test_format == AUDIO_U8 || test_format == AUDIO_S16)
            break;
    }
    if (!test_format) {
        debug_os2("Unsupported audio format, AUDIO_S16 used");
        test_format = AUDIO_S16;
    }

    pAData = (SDL_PrivateAudioData *) SDL_calloc(1, sizeof(struct SDL_PrivateAudioData));
    if (!pAData)
        return SDL_OutOfMemory();
    _this->hidden = pAData;

    ulRC = DosCreateEventSem(NULL, &pAData->hevBuf, DCE_AUTORESET, TRUE);
    if (ulRC != NO_ERROR) {
        debug_os2("DosCreateEventSem() failed, rc = %lu", ulRC);
        return -1;
    }

    /* Open audio device */
    stMCIAmpOpen.usDeviceID = 0;
    stMCIAmpOpen.pszDeviceType = (PSZ)MAKEULONG(MCI_DEVTYPE_AUDIO_AMPMIX,LOUSHORT(ulHandle));
    ulRC = mciSendCommand(0, MCI_OPEN,
                          (_getEnvULong("SDL_AUDIO_SHARE", 1, 0) != 0)?
                           MCI_WAIT | MCI_OPEN_TYPE_ID | MCI_OPEN_SHAREABLE :
                           MCI_WAIT | MCI_OPEN_TYPE_ID,
                          &stMCIAmpOpen,  0);
    if (LOUSHORT(ulRC) != MCIERR_SUCCESS) {
        DosCloseEventSem(pAData->hevBuf);
        pAData->usDeviceId = (USHORT)~0;
        return _MCIError("MCI_OPEN", ulRC);
    }
    pAData->usDeviceId = stMCIAmpOpen.usDeviceID;

    if (iscapture) {
        MCI_CONNECTOR_PARMS stMCIConnector;
        MCI_AMP_SET_PARMS   stMCIAmpSet;
        BOOL                fLineIn = _getEnvULong("SDL_AUDIO_LINEIN", 1, 0);

        /* Set particular connector. */
        SDL_zero(stMCIConnector);
        stMCIConnector.ulConnectorType = (fLineIn)? MCI_LINE_IN_CONNECTOR :
                                                    MCI_MICROPHONE_CONNECTOR;
        mciSendCommand(stMCIAmpOpen.usDeviceID, MCI_CONNECTOR,
                       MCI_WAIT | MCI_ENABLE_CONNECTOR |
                       MCI_CONNECTOR_TYPE, &stMCIConnector, 0);

        /* Disable monitor. */
        SDL_zero(stMCIAmpSet);
        stMCIAmpSet.ulItem = MCI_AMP_SET_MONITOR;
        mciSendCommand(stMCIAmpOpen.usDeviceID, MCI_SET,
                       MCI_WAIT | MCI_SET_OFF | MCI_SET_ITEM,
                       &stMCIAmpSet, 0);

        /* Set record volume. */
        stMCIAmpSet.ulLevel = _getEnvULong("SDL_AUDIO_RECVOL", 100, 90);
        stMCIAmpSet.ulItem  = MCI_AMP_SET_AUDIO;
        stMCIAmpSet.ulAudio = MCI_SET_AUDIO_ALL; /* Both cnannels. */
        stMCIAmpSet.ulValue = (fLineIn) ? MCI_LINE_IN_CONNECTOR :
                                          MCI_MICROPHONE_CONNECTOR ;

        mciSendCommand(stMCIAmpOpen.usDeviceID, MCI_SET,
                       MCI_WAIT | MCI_SET_AUDIO | MCI_AMP_SET_GAIN,
                       &stMCIAmpSet, 0);
    }

    _this->spec.format = test_format;
    _this->spec.channels = _this->spec.channels > 1 ? 2 : 1;
    if (_this->spec.freq < 8000) {
        _this->spec.freq = 8000;
        new_freq = TRUE;
    } else if (_this->spec.freq > 48000) {
        _this->spec.freq = 48000;
        new_freq = TRUE;
    }

    /* Setup mixer. */
    pAData->stMCIMixSetup.ulFormatTag     = MCI_WAVE_FORMAT_PCM;
    pAData->stMCIMixSetup.ulBitsPerSample = SDL_AUDIO_BITSIZE(test_format);
    pAData->stMCIMixSetup.ulSamplesPerSec = _this->spec.freq;
    pAData->stMCIMixSetup.ulChannels      = _this->spec.channels;
    pAData->stMCIMixSetup.ulDeviceType    = MCI_DEVTYPE_WAVEFORM_AUDIO;
    if (!iscapture) {
        pAData->stMCIMixSetup.ulFormatMode= MCI_PLAY;
        pAData->stMCIMixSetup.pmixEvent   = cbAudioWriteEvent;
    } else {
        pAData->stMCIMixSetup.ulFormatMode= MCI_RECORD;
        pAData->stMCIMixSetup.pmixEvent   = cbAudioReadEvent;
    }

    ulRC = mciSendCommand(pAData->usDeviceId, MCI_MIXSETUP,
                          MCI_WAIT | MCI_MIXSETUP_INIT, &pAData->stMCIMixSetup, 0);
    if (LOUSHORT(ulRC) != MCIERR_SUCCESS && _this->spec.freq > 44100) {
        new_freq = TRUE;
        pAData->stMCIMixSetup.ulSamplesPerSec = 44100;
        _this->spec.freq = 44100;
        ulRC = mciSendCommand(pAData->usDeviceId, MCI_MIXSETUP,
                              MCI_WAIT | MCI_MIXSETUP_INIT, &pAData->stMCIMixSetup, 0);
    }

    debug_os2("Setup mixer [BPS: %lu, Freq.: %lu, Channels: %lu]: %s",
              pAData->stMCIMixSetup.ulBitsPerSample,
              pAData->stMCIMixSetup.ulSamplesPerSec,
              pAData->stMCIMixSetup.ulChannels,
              (ulRC == MCIERR_SUCCESS)? "SUCCESS" : "FAIL");

    if (ulRC != MCIERR_SUCCESS) {
        pAData->stMCIMixSetup.ulBitsPerSample = 0;
        return _MCIError("MCI_MIXSETUP", ulRC);
    }

    if (_this->spec.samples == 0 || new_freq == TRUE) {
    /* also see SDL_audio.c:prepare_audiospec() */
    /* Pick a default of ~46 ms at desired frequency */
        Uint32 samples = (_this->spec.freq / 1000) * 46;
        Uint32 power2 = 1;
        while (power2 < samples) {
            power2 <<= 1;
        }
        _this->spec.samples = power2;
    }
    /* Update the fragment size as size in bytes */
    SDL_CalculateAudioSpec(&_this->spec);

    /* Allocate memory buffers */
    stMCIBuffer.ulBufferSize = _this->spec.size;/* (_this->spec.freq / 1000) * 100 */
    stMCIBuffer.ulNumBuffers = NUM_BUFFERS;
    stMCIBuffer.pBufList     = pAData->aMixBuffers;

    ulRC = mciSendCommand(pAData->usDeviceId, MCI_BUFFER,
                          MCI_WAIT | MCI_ALLOCATE_MEMORY, &stMCIBuffer, 0);
    if (LOUSHORT(ulRC) != MCIERR_SUCCESS) {
        return _MCIError("MCI_BUFFER", ulRC);
    }
    pAData->cMixBuffers = stMCIBuffer.ulNumBuffers;
    _this->spec.size = stMCIBuffer.ulBufferSize;

    debug_os2("%s, number of mix buffers: %lu",iscapture ? "capture": "play",pAData->cMixBuffers);

    /* Fill all device buffers with data */
    for (ulIdx = 0; ulIdx < stMCIBuffer.ulNumBuffers; ulIdx++) {
        pAData->aMixBuffers[ulIdx].ulFlags        = 0;
        pAData->aMixBuffers[ulIdx].ulBufferLength = stMCIBuffer.ulBufferSize;
        pAData->aMixBuffers[ulIdx].ulUserParm     = (ULONG)_this;

        SDL_memset(((PMCI_MIX_BUFFER)stMCIBuffer.pBufList)[ulIdx].pBuffer,
                   _this->spec.silence, stMCIBuffer.ulBufferSize);
    }
    pAData->pFillBuffer  = pAData->aMixBuffers;
    pAData->pDrainBuffer = pAData->aMixBuffers;

    return 0;
}


static SDL_bool OS2_Init(SDL_AudioDriverImpl * impl)
{
    /* Set the function pointers */
    impl->DetectDevices = OS2_DetectDevices;
    impl->OpenDevice    = OS2_OpenDevice;
    impl->PlayDevice    = OS2_PlayDevice;
    impl->WaitDevice    = OS2_WaitDevice;
    impl->GetDeviceBuf  = OS2_GetDeviceBuf;
    impl->CloseDevice   = OS2_CloseDevice;
    impl->CaptureFromDevice = OS2_CaptureFromDevice ;
    impl->FlushCapture = OS2_FlushCapture;
    impl->HasCaptureSupport = SDL_TRUE;

    return SDL_TRUE; /* this audio target is available. */
}


AudioBootStrap OS2AUDIO_bootstrap = {
    "DART", "OS/2 DART", OS2_Init, SDL_FALSE
};

#endif /* SDL_AUDIO_DRIVER_OS2 */

/* vi: set ts=4 sw=4 expandtab: */
