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

#if SDL_VIDEO_RENDER_D3D11

#include "SDL_syswm.h"
#include "../../video/winrt/SDL_winrtvideo_cpp.h"
extern "C" {
#include "../SDL_sysrender.h"
}

#include <windows.ui.core.h>
#include <windows.graphics.display.h>

#if WINAPI_FAMILY == WINAPI_FAMILY_APP
#include <windows.ui.xaml.media.dxinterop.h>
#endif

using namespace Windows::UI::Core;
using namespace Windows::Graphics::Display;

#include <DXGI.h>

#include "SDL_render_winrt.h"

extern "C" void *
D3D11_GetCoreWindowFromSDLRenderer(SDL_Renderer *renderer)
{
    SDL_Window *sdlWindow = renderer->window;
    if (!renderer->window) {
        return NULL;
    }

    SDL_SysWMinfo sdlWindowInfo;
    SDL_VERSION(&sdlWindowInfo.version);
    if ( ! SDL_GetWindowWMInfo(sdlWindow, &sdlWindowInfo) ) {
        return NULL;
    }

    if (sdlWindowInfo.subsystem != SDL_SYSWM_WINRT) {
        return NULL;
    }

    if (!sdlWindowInfo.info.winrt.window) {
        return NULL;
    }

    ABI::Windows::UI::Core::ICoreWindow *coreWindow = NULL;
    if (FAILED(sdlWindowInfo.info.winrt.window->QueryInterface(&coreWindow))) {
        return NULL;
    }

    IUnknown *coreWindowAsIUnknown = NULL;
    coreWindow->QueryInterface(&coreWindowAsIUnknown);
    coreWindow->Release();

    return coreWindowAsIUnknown;
}

extern "C" DXGI_MODE_ROTATION
D3D11_GetCurrentRotation()
{
    const DisplayOrientations currentOrientation = WINRT_DISPLAY_PROPERTY(CurrentOrientation);

    switch (currentOrientation) {

#if SDL_WINAPI_FAMILY_PHONE
    /* Windows Phone rotations */
    case DisplayOrientations::Landscape:
        return DXGI_MODE_ROTATION_ROTATE90;
    case DisplayOrientations::Portrait:
        return DXGI_MODE_ROTATION_IDENTITY;
    case DisplayOrientations::LandscapeFlipped:
        return DXGI_MODE_ROTATION_ROTATE270;
    case DisplayOrientations::PortraitFlipped:
        return DXGI_MODE_ROTATION_ROTATE180;
#else
    /* Non-Windows-Phone rotations (ex: Windows 8, Windows RT) */
    case DisplayOrientations::Landscape:
        return DXGI_MODE_ROTATION_IDENTITY;
    case DisplayOrientations::Portrait:
        return DXGI_MODE_ROTATION_ROTATE270;
    case DisplayOrientations::LandscapeFlipped:
        return DXGI_MODE_ROTATION_ROTATE180;
    case DisplayOrientations::PortraitFlipped:
        return DXGI_MODE_ROTATION_ROTATE90;
#endif /* SDL_WINAPI_FAMILY_PHONE */
    }

    return DXGI_MODE_ROTATION_IDENTITY;
}

#endif /* SDL_VIDEO_RENDER_D3D11 */

/* vi: set ts=4 sw=4 expandtab: */
