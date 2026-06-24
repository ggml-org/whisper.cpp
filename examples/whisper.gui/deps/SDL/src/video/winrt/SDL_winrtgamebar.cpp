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

/* Windows includes */
#include <roapi.h>
#include <windows.foundation.h>
#include <windows.system.h>

/* SDL includes */
extern "C" {
#include "SDL_mouse.h"
#include "../SDL_sysvideo.h"
}
#include "SDL_winrtvideo_cpp.h"

/* Game Bar events can come in off the main thread.  Use the following
   WinRT CoreDispatcher to deal with them on SDL's thread.
*/
static Platform::WeakReference WINRT_MainThreadDispatcher;

/* Win10's initial SDK (the 10.0.10240.0 release) does not include references
   to Game Bar APIs, as the Game Bar was released via Win10 10.0.10586.0.

   Declare its WinRT/COM interface here, to allow compilation with earlier
   Windows SDKs.
*/
MIDL_INTERFACE("1DB9A292-CC78-4173-BE45-B61E67283EA7")
IGameBarStatics_ : public IInspectable
{
  public:
    virtual HRESULT STDMETHODCALLTYPE add_VisibilityChanged(
        __FIEventHandler_1_IInspectable * handler,
        Windows::Foundation::EventRegistrationToken * token) = 0;

    virtual HRESULT STDMETHODCALLTYPE remove_VisibilityChanged(
        Windows::Foundation::EventRegistrationToken token) = 0;

    virtual HRESULT STDMETHODCALLTYPE add_IsInputRedirectedChanged(
        __FIEventHandler_1_IInspectable * handler,
        Windows::Foundation::EventRegistrationToken * token) = 0;

    virtual HRESULT STDMETHODCALLTYPE remove_IsInputRedirectedChanged(
        Windows::Foundation::EventRegistrationToken token) = 0;

    virtual HRESULT STDMETHODCALLTYPE get_Visible(
        boolean * value) = 0;

    virtual HRESULT STDMETHODCALLTYPE get_IsInputRedirected(
        boolean * value) = 0;
};

/* Declare the game bar's COM GUID */
static GUID IID_IGameBarStatics_ = { MAKELONG(0xA292, 0x1DB9), 0xCC78, 0x4173, { 0xBE, 0x45, 0xB6, 0x1E, 0x67, 0x28, 0x3E, 0xA7 } };

/* Retrieves a pointer to the game bar, or NULL if it is not available.
   If a pointer is returned, it's ->Release() method must be called
   after the caller has finished using it.
*/
static IGameBarStatics_ *WINRT_GetGameBar()
{
    wchar_t *wClassName = L"Windows.Gaming.UI.GameBar";
    HSTRING hClassName;
    IActivationFactory *pActivationFactory = NULL;
    IGameBarStatics_ *pGameBar = NULL;
    HRESULT hr;

    hr = ::WindowsCreateString(wClassName, (UINT32)SDL_wcslen(wClassName), &hClassName);
    if (FAILED(hr)) {
        goto done;
    }

    hr = Windows::Foundation::GetActivationFactory(hClassName, &pActivationFactory);
    if (FAILED(hr)) {
        goto done;
    }

    pActivationFactory->QueryInterface(IID_IGameBarStatics_, (void **)&pGameBar);

done:
    if (pActivationFactory) {
        pActivationFactory->Release();
    }
    if (hClassName) {
        ::WindowsDeleteString(hClassName);
    }
    return pGameBar;
}

static void WINRT_HandleGameBarIsInputRedirected_MainThread()
{
    IGameBarStatics_ *gameBar;
    boolean isInputRedirected = 0;
    if (!WINRT_MainThreadDispatcher) {
        /* The game bar event handler has been deregistered! */
        return;
    }
    gameBar = WINRT_GetGameBar();
    if (!gameBar) {
        /* Shouldn't happen, but just in case... */
        return;
    }
    if (SUCCEEDED(gameBar->get_IsInputRedirected(&isInputRedirected))) {
        if (!isInputRedirected) {
            /* Input-control is now back to the SDL app. Restore the cursor,
               in case Windows does not (it does not in either Win10
               10.0.10240.0 or 10.0.10586.0, maybe later version(s) too.
            */
            SDL_Cursor *cursor = SDL_GetCursor();
            SDL_SetCursor(cursor);
        }
    }
    gameBar->Release();
}

static void WINRT_HandleGameBarIsInputRedirected_NonMainThread(Platform::Object ^ o1, Platform::Object ^ o2)
{
    Windows::UI::Core::CoreDispatcher ^ dispatcher = WINRT_MainThreadDispatcher.Resolve<Windows::UI::Core::CoreDispatcher>();
    if (dispatcher) {
        dispatcher->RunAsync(
            Windows::UI::Core::CoreDispatcherPriority::Normal,
            ref new Windows::UI::Core::DispatchedHandler(&WINRT_HandleGameBarIsInputRedirected_MainThread));
    }
}

void WINRT_InitGameBar(_THIS)
{
    SDL_VideoData *driverdata = (SDL_VideoData *)_this->driverdata;
    IGameBarStatics_ *gameBar = WINRT_GetGameBar();
    if (gameBar) {
        /* GameBar.IsInputRedirected events can come in via something other than
           the main/SDL thread.

           Get a WinRT 'CoreDispatcher' that can be used to call back into the
           SDL thread.
        */
        WINRT_MainThreadDispatcher = Windows::UI::Core::CoreWindow::GetForCurrentThread()->Dispatcher;
        Windows::Foundation::EventHandler<Platform::Object ^> ^ handler =
            ref new Windows::Foundation::EventHandler<Platform::Object ^>(&WINRT_HandleGameBarIsInputRedirected_NonMainThread);
        __FIEventHandler_1_IInspectable *pHandler = reinterpret_cast<__FIEventHandler_1_IInspectable *>(handler);
        gameBar->add_IsInputRedirectedChanged(pHandler, &driverdata->gameBarIsInputRedirectedToken);
        gameBar->Release();
    }
}

void WINRT_QuitGameBar(_THIS)
{
    SDL_VideoData *driverdata;
    IGameBarStatics_ *gameBar;
    if (!_this || !_this->driverdata) {
        return;
    }
    gameBar = WINRT_GetGameBar();
    if (!gameBar) {
        return;
    }
    driverdata = (SDL_VideoData *)_this->driverdata;
    if (driverdata->gameBarIsInputRedirectedToken.Value) {
        gameBar->remove_IsInputRedirectedChanged(driverdata->gameBarIsInputRedirectedToken);
        driverdata->gameBarIsInputRedirectedToken.Value = 0;
    }
    WINRT_MainThreadDispatcher = nullptr;
    gameBar->Release();
}

#endif /* SDL_VIDEO_DRIVER_WINRT */

/* vi: set ts=4 sw=4 expandtab: */
