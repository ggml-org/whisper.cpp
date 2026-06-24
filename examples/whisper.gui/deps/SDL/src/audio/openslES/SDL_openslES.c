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

#ifdef SDL_AUDIO_DRIVER_OPENSLES

/* For more discussion of low latency audio on Android, see this:
   https://googlesamples.github.io/android-audio-high-performance/guides/opensl_es.html
*/

#include "SDL_audio.h"
#include "../SDL_audio_c.h"
#include "../../core/android/SDL_android.h"
#include "SDL_openslES.h"

/* for native audio */
#include <SLES/OpenSLES.h>
#include <SLES/OpenSLES_Android.h>

#include <android/log.h>

#if 0
#define LOG_TAG   "SDL_openslES"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
//#define LOGV(...)  __android_log_print(ANDROID_LOG_VERBOSE,LOG_TAG,__VA_ARGS__)
#define LOGV(...)
#else
#define LOGE(...)
#define LOGI(...)
#define LOGV(...)
#endif

/*
#define SL_SPEAKER_FRONT_LEFT            ((SLuint32) 0x00000001)
#define SL_SPEAKER_FRONT_RIGHT           ((SLuint32) 0x00000002)
#define SL_SPEAKER_FRONT_CENTER          ((SLuint32) 0x00000004)
#define SL_SPEAKER_LOW_FREQUENCY         ((SLuint32) 0x00000008)
#define SL_SPEAKER_BACK_LEFT             ((SLuint32) 0x00000010)
#define SL_SPEAKER_BACK_RIGHT            ((SLuint32) 0x00000020)
#define SL_SPEAKER_FRONT_LEFT_OF_CENTER  ((SLuint32) 0x00000040)
#define SL_SPEAKER_FRONT_RIGHT_OF_CENTER ((SLuint32) 0x00000080)
#define SL_SPEAKER_BACK_CENTER           ((SLuint32) 0x00000100)
#define SL_SPEAKER_SIDE_LEFT             ((SLuint32) 0x00000200)
#define SL_SPEAKER_SIDE_RIGHT            ((SLuint32) 0x00000400)
#define SL_SPEAKER_TOP_CENTER            ((SLuint32) 0x00000800)
#define SL_SPEAKER_TOP_FRONT_LEFT        ((SLuint32) 0x00001000)
#define SL_SPEAKER_TOP_FRONT_CENTER      ((SLuint32) 0x00002000)
#define SL_SPEAKER_TOP_FRONT_RIGHT       ((SLuint32) 0x00004000)
#define SL_SPEAKER_TOP_BACK_LEFT         ((SLuint32) 0x00008000)
#define SL_SPEAKER_TOP_BACK_CENTER       ((SLuint32) 0x00010000)
#define SL_SPEAKER_TOP_BACK_RIGHT        ((SLuint32) 0x00020000)
*/
#define SL_ANDROID_SPEAKER_STEREO (SL_SPEAKER_FRONT_LEFT | SL_SPEAKER_FRONT_RIGHT)
#define SL_ANDROID_SPEAKER_QUAD   (SL_ANDROID_SPEAKER_STEREO | SL_SPEAKER_BACK_LEFT | SL_SPEAKER_BACK_RIGHT)
#define SL_ANDROID_SPEAKER_5DOT1  (SL_ANDROID_SPEAKER_QUAD | SL_SPEAKER_FRONT_CENTER | SL_SPEAKER_LOW_FREQUENCY)
#define SL_ANDROID_SPEAKER_7DOT1  (SL_ANDROID_SPEAKER_5DOT1 | SL_SPEAKER_SIDE_LEFT | SL_SPEAKER_SIDE_RIGHT)

/* engine interfaces */
static SLObjectItf engineObject = NULL;
static SLEngineItf engineEngine = NULL;

/* output mix interfaces */
static SLObjectItf outputMixObject = NULL;

/* buffer queue player interfaces */
static SLObjectItf bqPlayerObject = NULL;
static SLPlayItf bqPlayerPlay = NULL;
static SLAndroidSimpleBufferQueueItf bqPlayerBufferQueue = NULL;
#if 0
static SLVolumeItf bqPlayerVolume;
#endif

/* recorder interfaces */
static SLObjectItf recorderObject = NULL;
static SLRecordItf recorderRecord = NULL;
static SLAndroidSimpleBufferQueueItf recorderBufferQueue = NULL;

#if 0
static const char *sldevaudiorecorderstr = "SLES Audio Recorder";
static const char *sldevaudioplayerstr   = "SLES Audio Player";

#define SLES_DEV_AUDIO_RECORDER sldevaudiorecorderstr
#define SLES_DEV_AUDIO_PLAYER   sldevaudioplayerstr
static void openslES_DetectDevices( int iscapture )
{
    LOGI( "openSLES_DetectDevices()" );
    if ( iscapture )
            addfn( SLES_DEV_AUDIO_RECORDER );
    else
            addfn( SLES_DEV_AUDIO_PLAYER );
}
#endif

static void openslES_DestroyEngine(void)
{
    LOGI("openslES_DestroyEngine()");

    /* destroy output mix object, and invalidate all associated interfaces */
    if (outputMixObject != NULL) {
        (*outputMixObject)->Destroy(outputMixObject);
        outputMixObject = NULL;
    }

    /* destroy engine object, and invalidate all associated interfaces */
    if (engineObject != NULL) {
        (*engineObject)->Destroy(engineObject);
        engineObject = NULL;
        engineEngine = NULL;
    }
}

static int openslES_CreateEngine(void)
{
    const SLInterfaceID ids[1] = { SL_IID_VOLUME };
    const SLboolean req[1] = { SL_BOOLEAN_FALSE };
    SLresult result;

    LOGI("openSLES_CreateEngine()");

    /* create engine */
    result = slCreateEngine(&engineObject, 0, NULL, 0, NULL, NULL);
    if (SL_RESULT_SUCCESS != result) {
        LOGE("slCreateEngine failed: %d", result);
        goto error;
    }
    LOGI("slCreateEngine OK");

    /* realize the engine */
    result = (*engineObject)->Realize(engineObject, SL_BOOLEAN_FALSE);
    if (SL_RESULT_SUCCESS != result) {
        LOGE("RealizeEngine failed: %d", result);
        goto error;
    }
    LOGI("RealizeEngine OK");

    /* get the engine interface, which is needed in order to create other objects */
    result = (*engineObject)->GetInterface(engineObject, SL_IID_ENGINE, &engineEngine);
    if (SL_RESULT_SUCCESS != result) {
        LOGE("EngineGetInterface failed: %d", result);
        goto error;
    }
    LOGI("EngineGetInterface OK");

    /* create output mix */
    result = (*engineEngine)->CreateOutputMix(engineEngine, &outputMixObject, 1, ids, req);
    if (SL_RESULT_SUCCESS != result) {
        LOGE("CreateOutputMix failed: %d", result);
        goto error;
    }
    LOGI("CreateOutputMix OK");

    /* realize the output mix */
    result = (*outputMixObject)->Realize(outputMixObject, SL_BOOLEAN_FALSE);
    if (SL_RESULT_SUCCESS != result) {
        LOGE("RealizeOutputMix failed: %d", result);
        goto error;
    }
    return 1;

error:
    openslES_DestroyEngine();
    return 0;
}

/* this callback handler is called every time a buffer finishes recording */
static void bqRecorderCallback(SLAndroidSimpleBufferQueueItf bq, void *context)
{
    struct SDL_PrivateAudioData *audiodata = (struct SDL_PrivateAudioData *)context;

    LOGV("SLES: Recording Callback");
    SDL_SemPost(audiodata->playsem);
}

static void openslES_DestroyPCMRecorder(_THIS)
{
    struct SDL_PrivateAudioData *audiodata = this->hidden;
    SLresult result;

    /* stop recording */
    if (recorderRecord != NULL) {
        result = (*recorderRecord)->SetRecordState(recorderRecord, SL_RECORDSTATE_STOPPED);
        if (SL_RESULT_SUCCESS != result) {
            LOGE("SetRecordState stopped: %d", result);
        }
    }

    /* destroy audio recorder object, and invalidate all associated interfaces */
    if (recorderObject != NULL) {
        (*recorderObject)->Destroy(recorderObject);
        recorderObject = NULL;
        recorderRecord = NULL;
        recorderBufferQueue = NULL;
    }

    if (audiodata->playsem) {
        SDL_DestroySemaphore(audiodata->playsem);
        audiodata->playsem = NULL;
    }

    if (audiodata->mixbuff) {
        SDL_free(audiodata->mixbuff);
    }
}

static int openslES_CreatePCMRecorder(_THIS)
{
    struct SDL_PrivateAudioData *audiodata = this->hidden;
    SLDataFormat_PCM format_pcm;
    SLDataLocator_AndroidSimpleBufferQueue loc_bufq;
    SLDataSink audioSnk;
    SLDataLocator_IODevice loc_dev;
    SLDataSource audioSrc;
    const SLInterfaceID ids[1] = { SL_IID_ANDROIDSIMPLEBUFFERQUEUE };
    const SLboolean req[1] = { SL_BOOLEAN_TRUE };
    SLresult result;
    int i;

    if (!Android_JNI_RequestPermission("android.permission.RECORD_AUDIO")) {
        LOGE("This app doesn't have RECORD_AUDIO permission");
        return SDL_SetError("This app doesn't have RECORD_AUDIO permission");
    }

    /* Just go with signed 16-bit audio as it's the most compatible */
    this->spec.format = AUDIO_S16SYS;
    this->spec.channels = 1;
    /*this->spec.freq = SL_SAMPLINGRATE_16 / 1000;*/

    /* Update the fragment size as size in bytes */
    SDL_CalculateAudioSpec(&this->spec);

    LOGI("Try to open %u hz %u bit chan %u %s samples %u",
         this->spec.freq, SDL_AUDIO_BITSIZE(this->spec.format),
         this->spec.channels, (this->spec.format & 0x1000) ? "BE" : "LE", this->spec.samples);

    /* configure audio source */
    loc_dev.locatorType = SL_DATALOCATOR_IODEVICE;
    loc_dev.deviceType = SL_IODEVICE_AUDIOINPUT;
    loc_dev.deviceID = SL_DEFAULTDEVICEID_AUDIOINPUT;
    loc_dev.device = NULL;
    audioSrc.pLocator = &loc_dev;
    audioSrc.pFormat = NULL;

    /* configure audio sink */
    loc_bufq.locatorType = SL_DATALOCATOR_ANDROIDSIMPLEBUFFERQUEUE;
    loc_bufq.numBuffers = NUM_BUFFERS;

    format_pcm.formatType = SL_DATAFORMAT_PCM;
    format_pcm.numChannels = this->spec.channels;
    format_pcm.samplesPerSec = this->spec.freq * 1000; /* / kilo Hz to milli Hz */
    format_pcm.bitsPerSample = SDL_AUDIO_BITSIZE(this->spec.format);
    format_pcm.containerSize = SDL_AUDIO_BITSIZE(this->spec.format);
    format_pcm.endianness = SL_BYTEORDER_LITTLEENDIAN;
    format_pcm.channelMask = SL_SPEAKER_FRONT_CENTER;

    audioSnk.pLocator = &loc_bufq;
    audioSnk.pFormat = &format_pcm;

    /* create audio recorder */
    /* (requires the RECORD_AUDIO permission) */
    result = (*engineEngine)->CreateAudioRecorder(engineEngine, &recorderObject, &audioSrc, &audioSnk, 1, ids, req);
    if (SL_RESULT_SUCCESS != result) {
        LOGE("CreateAudioRecorder failed: %d", result);
        goto failed;
    }

    /* realize the recorder */
    result = (*recorderObject)->Realize(recorderObject, SL_BOOLEAN_FALSE);
    if (SL_RESULT_SUCCESS != result) {
        LOGE("RealizeAudioPlayer failed: %d", result);
        goto failed;
    }

    /* get the record interface */
    result = (*recorderObject)->GetInterface(recorderObject, SL_IID_RECORD, &recorderRecord);
    if (SL_RESULT_SUCCESS != result) {
        LOGE("SL_IID_RECORD interface get failed: %d", result);
        goto failed;
    }

    /* get the buffer queue interface */
    result = (*recorderObject)->GetInterface(recorderObject, SL_IID_ANDROIDSIMPLEBUFFERQUEUE, &recorderBufferQueue);
    if (SL_RESULT_SUCCESS != result) {
        LOGE("SL_IID_BUFFERQUEUE interface get failed: %d", result);
        goto failed;
    }

    /* register callback on the buffer queue */
    /* context is '(SDL_PrivateAudioData *)this->hidden' */
    result = (*recorderBufferQueue)->RegisterCallback(recorderBufferQueue, bqRecorderCallback, this->hidden);
    if (SL_RESULT_SUCCESS != result) {
        LOGE("RegisterCallback failed: %d", result);
        goto failed;
    }

    /* Create the audio buffer semaphore */
    audiodata->playsem = SDL_CreateSemaphore(0);
    if (!audiodata->playsem) {
        LOGE("cannot create Semaphore!");
        goto failed;
    }

    /* Create the sound buffers */
    audiodata->mixbuff = (Uint8 *)SDL_malloc(NUM_BUFFERS * this->spec.size);
    if (!audiodata->mixbuff) {
        LOGE("mixbuffer allocate - out of memory");
        goto failed;
    }

    for (i = 0; i < NUM_BUFFERS; i++) {
        audiodata->pmixbuff[i] = audiodata->mixbuff + i * this->spec.size;
    }

    /* in case already recording, stop recording and clear buffer queue */
    result = (*recorderRecord)->SetRecordState(recorderRecord, SL_RECORDSTATE_STOPPED);
    if (SL_RESULT_SUCCESS != result) {
        LOGE("Record set state failed: %d", result);
        goto failed;
    }

    /* enqueue empty buffers to be filled by the recorder */
    for (i = 0; i < NUM_BUFFERS; i++) {
        result = (*recorderBufferQueue)->Enqueue(recorderBufferQueue, audiodata->pmixbuff[i], this->spec.size);
        if (SL_RESULT_SUCCESS != result) {
            LOGE("Record enqueue buffers failed: %d", result);
            goto failed;
        }
    }

    /* start recording */
    result = (*recorderRecord)->SetRecordState(recorderRecord, SL_RECORDSTATE_RECORDING);
    if (SL_RESULT_SUCCESS != result) {
        LOGE("Record set state failed: %d", result);
        goto failed;
    }

    return 0;

failed:
    return SDL_SetError("Open device failed!");
}

/* this callback handler is called every time a buffer finishes playing */
static void bqPlayerCallback(SLAndroidSimpleBufferQueueItf bq, void *context)
{
    struct SDL_PrivateAudioData *audiodata = (struct SDL_PrivateAudioData *)context;

    LOGV("SLES: Playback Callback");
    SDL_SemPost(audiodata->playsem);
}

static void openslES_DestroyPCMPlayer(_THIS)
{
    struct SDL_PrivateAudioData *audiodata = this->hidden;
    SLresult result;

    /* set the player's state to 'stopped' */
    if (bqPlayerPlay != NULL) {
        result = (*bqPlayerPlay)->SetPlayState(bqPlayerPlay, SL_PLAYSTATE_STOPPED);
        if (SL_RESULT_SUCCESS != result) {
            LOGE("SetPlayState stopped failed: %d", result);
        }
    }

    /* destroy buffer queue audio player object, and invalidate all associated interfaces */
    if (bqPlayerObject != NULL) {
        (*bqPlayerObject)->Destroy(bqPlayerObject);

        bqPlayerObject = NULL;
        bqPlayerPlay = NULL;
        bqPlayerBufferQueue = NULL;
    }

    if (audiodata->playsem) {
        SDL_DestroySemaphore(audiodata->playsem);
        audiodata->playsem = NULL;
    }

    if (audiodata->mixbuff) {
        SDL_free(audiodata->mixbuff);
    }
}

static int openslES_CreatePCMPlayer(_THIS)
{
    struct SDL_PrivateAudioData *audiodata = this->hidden;
    SLDataLocator_AndroidSimpleBufferQueue loc_bufq;
    SLDataFormat_PCM format_pcm;
    SLAndroidDataFormat_PCM_EX format_pcm_ex;
    SLDataSource audioSrc;
    SLDataSink audioSnk;
    SLDataLocator_OutputMix loc_outmix;
    const SLInterfaceID ids[2] = { SL_IID_ANDROIDSIMPLEBUFFERQUEUE, SL_IID_VOLUME };
    const SLboolean req[2] = { SL_BOOLEAN_TRUE, SL_BOOLEAN_FALSE };
    SLresult result;
    int i;

    /* If we want to add floating point audio support (requires API level 21)
       it can be done as described here:
        https://developer.android.com/ndk/guides/audio/opensl/android-extensions.html#floating-point
    */
    if (SDL_GetAndroidSDKVersion() >= 21) {
        SDL_AudioFormat test_format;
        for (test_format = SDL_FirstAudioFormat(this->spec.format); test_format; test_format = SDL_NextAudioFormat()) {
            if (SDL_AUDIO_ISSIGNED(test_format)) {
                break;
            }
        }

        if (!test_format) {
            /* Didn't find a compatible format : */
            LOGI("No compatible audio format, using signed 16-bit audio");
            test_format = AUDIO_S16SYS;
        }
        this->spec.format = test_format;
    } else {
        /* Just go with signed 16-bit audio as it's the most compatible */
        this->spec.format = AUDIO_S16SYS;
    }

    /* Update the fragment size as size in bytes */
    SDL_CalculateAudioSpec(&this->spec);

    LOGI("Try to open %u hz %s %u bit chan %u %s samples %u",
         this->spec.freq, SDL_AUDIO_ISFLOAT(this->spec.format) ? "float" : "pcm", SDL_AUDIO_BITSIZE(this->spec.format),
         this->spec.channels, (this->spec.format & 0x1000) ? "BE" : "LE", this->spec.samples);

    /* configure audio source */
    loc_bufq.locatorType = SL_DATALOCATOR_ANDROIDSIMPLEBUFFERQUEUE;
    loc_bufq.numBuffers = NUM_BUFFERS;

    format_pcm.formatType = SL_DATAFORMAT_PCM;
    format_pcm.numChannels = this->spec.channels;
    format_pcm.samplesPerSec = this->spec.freq * 1000; /* / kilo Hz to milli Hz */
    format_pcm.bitsPerSample = SDL_AUDIO_BITSIZE(this->spec.format);
    format_pcm.containerSize = SDL_AUDIO_BITSIZE(this->spec.format);

    if (SDL_AUDIO_ISBIGENDIAN(this->spec.format)) {
        format_pcm.endianness = SL_BYTEORDER_BIGENDIAN;
    } else {
        format_pcm.endianness = SL_BYTEORDER_LITTLEENDIAN;
    }

    switch (this->spec.channels) {
    case 1:
        format_pcm.channelMask = SL_SPEAKER_FRONT_LEFT;
        break;
    case 2:
        format_pcm.channelMask = SL_ANDROID_SPEAKER_STEREO;
        break;
    case 3:
        format_pcm.channelMask = SL_ANDROID_SPEAKER_STEREO | SL_SPEAKER_FRONT_CENTER;
        break;
    case 4:
        format_pcm.channelMask = SL_ANDROID_SPEAKER_QUAD;
        break;
    case 5:
        format_pcm.channelMask = SL_ANDROID_SPEAKER_QUAD | SL_SPEAKER_FRONT_CENTER;
        break;
    case 6:
        format_pcm.channelMask = SL_ANDROID_SPEAKER_5DOT1;
        break;
    case 7:
        format_pcm.channelMask = SL_ANDROID_SPEAKER_5DOT1 | SL_SPEAKER_BACK_CENTER;
        break;
    case 8:
        format_pcm.channelMask = SL_ANDROID_SPEAKER_7DOT1;
        break;
    default:
        /* Unknown number of channels, fall back to stereo */
        this->spec.channels = 2;
        format_pcm.channelMask = SL_SPEAKER_FRONT_LEFT | SL_SPEAKER_FRONT_RIGHT;
        break;
    }

    if (SDL_AUDIO_ISFLOAT(this->spec.format)) {
        /* Copy all setup into PCM EX structure */
        format_pcm_ex.formatType = SL_ANDROID_DATAFORMAT_PCM_EX;
        format_pcm_ex.endianness = format_pcm.endianness;
        format_pcm_ex.channelMask = format_pcm.channelMask;
        format_pcm_ex.numChannels = format_pcm.numChannels;
        format_pcm_ex.sampleRate = format_pcm.samplesPerSec;
        format_pcm_ex.bitsPerSample = format_pcm.bitsPerSample;
        format_pcm_ex.containerSize = format_pcm.containerSize;
        format_pcm_ex.representation = SL_ANDROID_PCM_REPRESENTATION_FLOAT;
    }

    audioSrc.pLocator = &loc_bufq;
    audioSrc.pFormat = SDL_AUDIO_ISFLOAT(this->spec.format) ? (void *)&format_pcm_ex : (void *)&format_pcm;

    /* configure audio sink */
    loc_outmix.locatorType = SL_DATALOCATOR_OUTPUTMIX;
    loc_outmix.outputMix = outputMixObject;
    audioSnk.pLocator = &loc_outmix;
    audioSnk.pFormat = NULL;

    /* create audio player */
    result = (*engineEngine)->CreateAudioPlayer(engineEngine, &bqPlayerObject, &audioSrc, &audioSnk, 2, ids, req);
    if (SL_RESULT_SUCCESS != result) {
        LOGE("CreateAudioPlayer failed: %d", result);
        goto failed;
    }

    /* realize the player */
    result = (*bqPlayerObject)->Realize(bqPlayerObject, SL_BOOLEAN_FALSE);
    if (SL_RESULT_SUCCESS != result) {
        LOGE("RealizeAudioPlayer failed: %d", result);
        goto failed;
    }

    /* get the play interface */
    result = (*bqPlayerObject)->GetInterface(bqPlayerObject, SL_IID_PLAY, &bqPlayerPlay);
    if (SL_RESULT_SUCCESS != result) {
        LOGE("SL_IID_PLAY interface get failed: %d", result);
        goto failed;
    }

    /* get the buffer queue interface */
    result = (*bqPlayerObject)->GetInterface(bqPlayerObject, SL_IID_ANDROIDSIMPLEBUFFERQUEUE, &bqPlayerBufferQueue);
    if (SL_RESULT_SUCCESS != result) {
        LOGE("SL_IID_BUFFERQUEUE interface get failed: %d", result);
        goto failed;
    }

    /* register callback on the buffer queue */
    /* context is '(SDL_PrivateAudioData *)this->hidden' */
    result = (*bqPlayerBufferQueue)->RegisterCallback(bqPlayerBufferQueue, bqPlayerCallback, this->hidden);
    if (SL_RESULT_SUCCESS != result) {
        LOGE("RegisterCallback failed: %d", result);
        goto failed;
    }

#if 0
    /* get the volume interface */
    result = (*bqPlayerObject)->GetInterface(bqPlayerObject, SL_IID_VOLUME, &bqPlayerVolume);
    if (SL_RESULT_SUCCESS != result) {
        LOGE("SL_IID_VOLUME interface get failed: %d", result);
        /* goto failed; */
    }
#endif

    /* Create the audio buffer semaphore */
    audiodata->playsem = SDL_CreateSemaphore(NUM_BUFFERS - 1);
    if (!audiodata->playsem) {
        LOGE("cannot create Semaphore!");
        goto failed;
    }

    /* Create the sound buffers */
    audiodata->mixbuff = (Uint8 *)SDL_malloc(NUM_BUFFERS * this->spec.size);
    if (!audiodata->mixbuff) {
        LOGE("mixbuffer allocate - out of memory");
        goto failed;
    }

    for (i = 0; i < NUM_BUFFERS; i++) {
        audiodata->pmixbuff[i] = audiodata->mixbuff + i * this->spec.size;
    }

    /* set the player's state to playing */
    result = (*bqPlayerPlay)->SetPlayState(bqPlayerPlay, SL_PLAYSTATE_PLAYING);
    if (SL_RESULT_SUCCESS != result) {
        LOGE("Play set state failed: %d", result);
        goto failed;
    }

    return 0;

failed:
    return -1;
}

static int openslES_OpenDevice(_THIS, const char *devname)
{
    this->hidden = (struct SDL_PrivateAudioData *)SDL_calloc(1, sizeof(*this->hidden));
    if (!this->hidden) {
        return SDL_OutOfMemory();
    }

    if (this->iscapture) {
        LOGI("openslES_OpenDevice() %s for capture", devname);
        return openslES_CreatePCMRecorder(this);
    } else {
        int ret;
        LOGI("openslES_OpenDevice() %s for playing", devname);
        ret = openslES_CreatePCMPlayer(this);
        if (ret < 0) {
            /* Another attempt to open the device with a lower frequency */
            if (this->spec.freq > 48000) {
                openslES_DestroyPCMPlayer(this);
                this->spec.freq = 48000;
                ret = openslES_CreatePCMPlayer(this);
            }
        }

        if (ret == 0) {
            return 0;
        } else {
            return SDL_SetError("Open device failed!");
        }
    }
}

static void openslES_WaitDevice(_THIS)
{
    struct SDL_PrivateAudioData *audiodata = this->hidden;

    LOGV("openslES_WaitDevice()");

    /* Wait for an audio chunk to finish */
    SDL_SemWait(audiodata->playsem);
}

static void openslES_PlayDevice(_THIS)
{
    struct SDL_PrivateAudioData *audiodata = this->hidden;
    SLresult result;

    LOGV("======openslES_PlayDevice()======");

    /* Queue it up */
    result = (*bqPlayerBufferQueue)->Enqueue(bqPlayerBufferQueue, audiodata->pmixbuff[audiodata->next_buffer], this->spec.size);

    audiodata->next_buffer++;
    if (audiodata->next_buffer >= NUM_BUFFERS) {
        audiodata->next_buffer = 0;
    }

    /* If Enqueue fails, callback won't be called.
     * Post the semphore, not to run out of buffer */
    if (SL_RESULT_SUCCESS != result) {
        SDL_SemPost(audiodata->playsem);
    }
}

/*/           n   playn sem */
/* getbuf     0   -     1 */
/* fill buff  0   -     1 */
/* play       0 - 0     1 */
/* wait       1   0     0 */
/* getbuf     1   0     0 */
/* fill buff  1   0     0 */
/* play       0   0     0 */
/* wait */
/* */
/* okay.. */

static Uint8 *openslES_GetDeviceBuf(_THIS)
{
    struct SDL_PrivateAudioData *audiodata = this->hidden;

    LOGV("openslES_GetDeviceBuf()");
    return audiodata->pmixbuff[audiodata->next_buffer];
}

static int openslES_CaptureFromDevice(_THIS, void *buffer, int buflen)
{
    struct SDL_PrivateAudioData *audiodata = this->hidden;
    SLresult result;

    /* Wait for new recorded data */
    SDL_SemWait(audiodata->playsem);

    /* Copy it to the output buffer */
    SDL_assert(buflen == this->spec.size);
    SDL_memcpy(buffer, audiodata->pmixbuff[audiodata->next_buffer], this->spec.size);

    /* Re-enqueue the buffer */
    result = (*recorderBufferQueue)->Enqueue(recorderBufferQueue, audiodata->pmixbuff[audiodata->next_buffer], this->spec.size);
    if (SL_RESULT_SUCCESS != result) {
        LOGE("Record enqueue buffers failed: %d", result);
        return -1;
    }

    audiodata->next_buffer++;
    if (audiodata->next_buffer >= NUM_BUFFERS) {
        audiodata->next_buffer = 0;
    }

    return this->spec.size;
}

static void openslES_CloseDevice(_THIS)
{
    /* struct SDL_PrivateAudioData *audiodata = this->hidden; */

    if (this->iscapture) {
        LOGI("openslES_CloseDevice() for capture");
        openslES_DestroyPCMRecorder(this);
    } else {
        LOGI("openslES_CloseDevice() for playing");
        openslES_DestroyPCMPlayer(this);
    }

    SDL_free(this->hidden);
}

static SDL_bool openslES_Init(SDL_AudioDriverImpl *impl)
{
    LOGI("openslES_Init() called");

    if (!openslES_CreateEngine()) {
        return SDL_FALSE;
    }

    LOGI("openslES_Init() - set pointers");

    /* Set the function pointers */
    /* impl->DetectDevices = openslES_DetectDevices; */
    impl->OpenDevice = openslES_OpenDevice;
    impl->WaitDevice = openslES_WaitDevice;
    impl->PlayDevice = openslES_PlayDevice;
    impl->GetDeviceBuf = openslES_GetDeviceBuf;
    impl->CaptureFromDevice = openslES_CaptureFromDevice;
    impl->CloseDevice = openslES_CloseDevice;
    impl->Deinitialize = openslES_DestroyEngine;

    /* and the capabilities */
    impl->HasCaptureSupport = SDL_TRUE;
    impl->OnlyHasDefaultOutputDevice = SDL_TRUE;
    impl->OnlyHasDefaultCaptureDevice = SDL_TRUE;

    LOGI("openslES_Init() - success");

    /* this audio target is available. */
    return SDL_TRUE;
}

AudioBootStrap openslES_bootstrap = {
    "openslES", "opensl ES audio driver", openslES_Init, SDL_FALSE
};

void openslES_ResumeDevices(void)
{
    if (bqPlayerPlay != NULL) {
        /* set the player's state to 'playing' */
        SLresult result = (*bqPlayerPlay)->SetPlayState(bqPlayerPlay, SL_PLAYSTATE_PLAYING);
        if (SL_RESULT_SUCCESS != result) {
            LOGE("openslES_ResumeDevices failed: %d", result);
        }
    }
}

void openslES_PauseDevices(void)
{
    if (bqPlayerPlay != NULL) {
        /* set the player's state to 'paused' */
        SLresult result = (*bqPlayerPlay)->SetPlayState(bqPlayerPlay, SL_PLAYSTATE_PAUSED);
        if (SL_RESULT_SUCCESS != result) {
            LOGE("openslES_PauseDevices failed: %d", result);
        }
    }
}

#endif /* SDL_AUDIO_DRIVER_OPENSLES */

/* vi: set ts=4 sw=4 expandtab: */
