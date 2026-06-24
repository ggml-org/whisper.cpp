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

#ifdef SDL_VIDEO_DRIVER_WINRT

/*
 * Windows includes:
 */
#include <Windows.h>
using namespace Windows::UI::Core;
using Windows::UI::Core::CoreCursor;

/*
 * SDL includes:
 */
#include "SDL_winrtevents_c.h"
#include "../../core/winrt/SDL_winrtapp_common.h"
#include "../../core/winrt/SDL_winrtapp_direct3d.h"
#include "../../core/winrt/SDL_winrtapp_xaml.h"
#include "SDL_system.h"

extern "C" {
#include "../../thread/SDL_systhread.h"
#include "../SDL_sysvideo.h"
#include "../../events/SDL_events_c.h"
}

/* Forward declarations */
static void WINRT_YieldXAMLThread();

/* Global event management */

void WINRT_PumpEvents(_THIS)
{
    if (SDL_WinRTGlobalApp) {
        SDL_WinRTGlobalApp->PumpEvents();
    } else if (WINRT_XAMLWasEnabled) {
        WINRT_YieldXAMLThread();
    }
}

/* XAML Thread management */

enum SDL_XAMLAppThreadState
{
    ThreadState_NotLaunched = 0,
    ThreadState_Running,
    ThreadState_Yielding
};

static SDL_XAMLAppThreadState _threadState = ThreadState_NotLaunched;
static SDL_Thread *_XAMLThread = nullptr;
static SDL_mutex *_mutex = nullptr;
static SDL_cond *_cond = nullptr;

static void WINRT_YieldXAMLThread()
{
    SDL_LockMutex(_mutex);
    SDL_assert(_threadState == ThreadState_Running);
    _threadState = ThreadState_Yielding;
    SDL_UnlockMutex(_mutex);

    SDL_CondSignal(_cond);

    SDL_LockMutex(_mutex);
    while (_threadState != ThreadState_Running) {
        SDL_CondWait(_cond, _mutex);
    }
    SDL_UnlockMutex(_mutex);
}

static int WINRT_XAMLThreadMain(void *userdata)
{
    // TODO, WinRT: pass the C-style main() a reasonably realistic
    // representation of command line arguments.
    int argc = 0;
    char **argv = NULL;
    return WINRT_SDLAppEntryPoint(argc, argv);
}

void WINRT_CycleXAMLThread(void)
{
    switch (_threadState) {
    case ThreadState_NotLaunched:
    {
        _cond = SDL_CreateCond();

        _mutex = SDL_CreateMutex();
        _threadState = ThreadState_Running;
        _XAMLThread = SDL_CreateThreadInternal(WINRT_XAMLThreadMain, "SDL/XAML App Thread", 0, nullptr);

        SDL_LockMutex(_mutex);
        while (_threadState != ThreadState_Yielding) {
            SDL_CondWait(_cond, _mutex);
        }
        SDL_UnlockMutex(_mutex);

        break;
    }

    case ThreadState_Running:
    {
        SDL_assert(false);
        break;
    }

    case ThreadState_Yielding:
    {
        SDL_LockMutex(_mutex);
        SDL_assert(_threadState == ThreadState_Yielding);
        _threadState = ThreadState_Running;
        SDL_UnlockMutex(_mutex);

        SDL_CondSignal(_cond);

        SDL_LockMutex(_mutex);
        while (_threadState != ThreadState_Yielding) {
            SDL_CondWait(_cond, _mutex);
        }
        SDL_UnlockMutex(_mutex);
    }
    }
}

#endif /* SDL_VIDEO_DRIVER_WINRT */

/* vi: set ts=4 sw=4 expandtab: */
