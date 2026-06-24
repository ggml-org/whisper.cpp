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

#include "SDL_main.h"
#include "SDL_system.h"
#include "SDL_winrtapp_direct3d.h"
#include "SDL_winrtapp_xaml.h"

#include <wrl.h>

int (*WINRT_SDLAppEntryPoint)(int, char **) = NULL;

extern "C" DECLSPEC int
SDL_WinRTRunApp(SDL_main_func mainFunction, void *xamlBackgroundPanel)
{
    if (xamlBackgroundPanel) {
        return SDL_WinRTInitXAMLApp(mainFunction, xamlBackgroundPanel);
    } else {
        if (FAILED(Windows::Foundation::Initialize(RO_INIT_MULTITHREADED))) {
            return 1;
        }
        return SDL_WinRTInitNonXAMLApp(mainFunction);
    }
}

extern "C" DECLSPEC SDL_WinRT_DeviceFamily
SDL_WinRTGetDeviceFamily()
{
#if NTDDI_VERSION >= NTDDI_WIN10 /* !!! FIXME: I have no idea if this is the right test. This is a UWP API, I think. Older windows should...just return "mobile"? I don't know. --ryan. */
    Platform::String ^ deviceFamily = Windows::System::Profile::AnalyticsInfo::VersionInfo->DeviceFamily;

    if (deviceFamily->Equals("Windows.Desktop")) {
        return SDL_WINRT_DEVICEFAMILY_DESKTOP;
    } else if (deviceFamily->Equals("Windows.Mobile")) {
        return SDL_WINRT_DEVICEFAMILY_MOBILE;
    } else if (deviceFamily->Equals("Windows.Xbox")) {
        return SDL_WINRT_DEVICEFAMILY_XBOX;
    }
#endif

    return SDL_WINRT_DEVICEFAMILY_UNKNOWN;
}
