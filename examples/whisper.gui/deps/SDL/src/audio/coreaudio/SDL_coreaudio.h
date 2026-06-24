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

#ifndef SDL_coreaudio_h_
#define SDL_coreaudio_h_

#include "../SDL_sysaudio.h"

#if !defined(__IPHONEOS__)
#define MACOSX_COREAUDIO
#endif

#ifdef MACOSX_COREAUDIO
#include <CoreAudio/CoreAudio.h>
#else
#import <AVFoundation/AVFoundation.h>
#import <UIKit/UIApplication.h>
#endif

#include <AudioToolbox/AudioToolbox.h>
#include <AudioUnit/AudioUnit.h>

/* Things named "Master" were renamed to "Main" in macOS 12.0's SDK. */
#ifdef MACOSX_COREAUDIO
#include <AvailabilityMacros.h>
#ifndef MAC_OS_VERSION_12_0
#define kAudioObjectPropertyElementMain kAudioObjectPropertyElementMaster
#endif
#endif

/* Hidden "this" pointer for the audio functions */
#define _THIS SDL_AudioDevice *this

struct SDL_PrivateAudioData
{
    SDL_Thread *thread;
    AudioQueueRef audioQueue;
    int numAudioBuffers;
    AudioQueueBufferRef *audioBuffer;
    void *buffer;
    UInt32 bufferOffset;
    UInt32 bufferSize;
    AudioStreamBasicDescription strdesc;
    SDL_sem *ready_semaphore;
    char *thread_error;
#ifdef MACOSX_COREAUDIO
    AudioDeviceID deviceID;
    SDL_atomic_t device_change_flag;
#else
    SDL_bool interrupted;
    CFTypeRef interruption_listener;
#endif
};

#endif /* SDL_coreaudio_h_ */

/* vi: set ts=4 sw=4 expandtab: */
