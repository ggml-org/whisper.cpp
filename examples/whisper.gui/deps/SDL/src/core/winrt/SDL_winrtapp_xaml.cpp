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

/* Windows includes */
#include <agile.h>
#include <Windows.h>

#if WINAPI_FAMILY == WINAPI_FAMILY_APP
#include <windows.ui.xaml.media.dxinterop.h>
#endif

/* SDL includes */
#include "../../SDL_internal.h"
#include "SDL.h"
#include "../../video/winrt/SDL_winrtevents_c.h"
#include "../../video/winrt/SDL_winrtvideo_cpp.h"
#include "SDL_winrtapp_common.h"
#include "SDL_winrtapp_xaml.h"

/* SDL-internal globals: */
SDL_bool WINRT_XAMLWasEnabled = SDL_FALSE;

#if WINAPI_FAMILY == WINAPI_FAMILY_APP
extern "C" ISwapChainBackgroundPanelNative *WINRT_GlobalSwapChainBackgroundPanelNative = NULL;
static Windows::Foundation::EventRegistrationToken WINRT_XAMLAppEventToken;
#endif

/*
 * Input event handlers (XAML)
 */
#if WINAPI_FAMILY == WINAPI_FAMILY_APP

static void WINRT_OnPointerPressedViaXAML(Platform::Object ^ sender, Windows::UI::Xaml::Input::PointerRoutedEventArgs ^ args)
{
    WINRT_ProcessPointerPressedEvent(WINRT_GlobalSDLWindow, args->GetCurrentPoint(nullptr));
}

static void WINRT_OnPointerMovedViaXAML(Platform::Object ^ sender, Windows::UI::Xaml::Input::PointerRoutedEventArgs ^ args)
{
    WINRT_ProcessPointerMovedEvent(WINRT_GlobalSDLWindow, args->GetCurrentPoint(nullptr));
}

static void WINRT_OnPointerReleasedViaXAML(Platform::Object ^ sender, Windows::UI::Xaml::Input::PointerRoutedEventArgs ^ args)
{
    WINRT_ProcessPointerReleasedEvent(WINRT_GlobalSDLWindow, args->GetCurrentPoint(nullptr));
}

static void WINRT_OnPointerWheelChangedViaXAML(Platform::Object ^ sender, Windows::UI::Xaml::Input::PointerRoutedEventArgs ^ args)
{
    WINRT_ProcessPointerWheelChangedEvent(WINRT_GlobalSDLWindow, args->GetCurrentPoint(nullptr));
}

#endif // WINAPI_FAMILY == WINAPI_FAMILY_APP

/*
 * XAML-to-SDL Rendering Callback
 */
#if WINAPI_FAMILY == WINAPI_FAMILY_APP

static void WINRT_OnRenderViaXAML(_In_ Platform::Object ^ sender, _In_ Platform::Object ^ args)
{
    WINRT_CycleXAMLThread();
}

#endif // WINAPI_FAMILY == WINAPI_FAMILY_APP

/*
 * SDL + XAML Initialization
 */

int SDL_WinRTInitXAMLApp(int (*mainFunction)(int, char **), void *backgroundPanelAsIInspectable)
{
#if SDL_WINAPI_FAMILY_PHONE
    return SDL_SetError("XAML support is not yet available in Windows Phone.");
#else
    // Declare C++/CX namespaces:
    using namespace Platform;
    using namespace Windows::Foundation;
    using namespace Windows::UI::Core;
    using namespace Windows::UI::Xaml;
    using namespace Windows::UI::Xaml::Controls;
    using namespace Windows::UI::Xaml::Input;
    using namespace Windows::UI::Xaml::Media;

    // Make sure we have a valid XAML element (to draw onto):
    if (!backgroundPanelAsIInspectable) {
        return SDL_InvalidParamError("backgroundPanelAsIInspectable");
    }

    Platform::Object ^ backgroundPanel = reinterpret_cast<Object ^>((IInspectable *)backgroundPanelAsIInspectable);
    SwapChainBackgroundPanel ^ swapChainBackgroundPanel = dynamic_cast<SwapChainBackgroundPanel ^>(backgroundPanel);
    if (!swapChainBackgroundPanel) {
        return SDL_SetError("An unknown or unsupported type of XAML control was specified.");
    }

    // Setup event handlers:
    swapChainBackgroundPanel->PointerPressed += ref new PointerEventHandler(WINRT_OnPointerPressedViaXAML);
    swapChainBackgroundPanel->PointerReleased += ref new PointerEventHandler(WINRT_OnPointerReleasedViaXAML);
    swapChainBackgroundPanel->PointerWheelChanged += ref new PointerEventHandler(WINRT_OnPointerWheelChangedViaXAML);
    swapChainBackgroundPanel->PointerMoved += ref new PointerEventHandler(WINRT_OnPointerMovedViaXAML);

    // Setup for rendering:
    IInspectable *panelInspectable = (IInspectable *)reinterpret_cast<IInspectable *>(swapChainBackgroundPanel);
    panelInspectable->QueryInterface(__uuidof(ISwapChainBackgroundPanelNative), (void **)&WINRT_GlobalSwapChainBackgroundPanelNative);

    WINRT_XAMLAppEventToken = CompositionTarget::Rendering::add(ref new EventHandler<Object ^>(WINRT_OnRenderViaXAML));

    // Make sure the app is ready to call the SDL-centric main() function:
    WINRT_SDLAppEntryPoint = mainFunction;
    SDL_SetMainReady();

    // Make sure video-init knows that we're initializing XAML:
    SDL_bool oldXAMLWasEnabledValue = WINRT_XAMLWasEnabled;
    WINRT_XAMLWasEnabled = SDL_TRUE;

    // Make sure video modes are detected now, while we still have access to the WinRT
    // CoreWindow.  WinRT will not allow the app's CoreWindow to be accessed via the
    // SDL/WinRT thread.
    if (SDL_InitSubSystem(SDL_INIT_VIDEO) < 0) {
        // SDL_InitSubSystem will, on error, set the SDL error.  Let that propogate to
        // the caller to here:
        WINRT_XAMLWasEnabled = oldXAMLWasEnabledValue;
        return -1;
    }

    // All done, for now.
    return 0;
#endif // SDL_WINAPI_FAMILY_PHONE  /  else
}
