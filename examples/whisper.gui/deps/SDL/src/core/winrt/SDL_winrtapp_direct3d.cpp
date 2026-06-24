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

/* Windows includes */
#include "ppltasks.h"
using namespace concurrency;
using namespace Windows::ApplicationModel;
using namespace Windows::ApplicationModel::Core;
using namespace Windows::ApplicationModel::Activation;
using namespace Windows::Devices::Input;
using namespace Windows::Graphics::Display;
using namespace Windows::Foundation;
using namespace Windows::System;
using namespace Windows::UI::Core;
using namespace Windows::UI::Input;

#if SDL_WINAPI_FAMILY_PHONE
using namespace Windows::Phone::UI::Input;
#endif

/* SDL includes */
extern "C" {
#include "SDL_events.h"
#include "SDL_hints.h"
#include "SDL_main.h"
#include "SDL_stdinc.h"
#include "SDL_render.h"
#include "../../video/SDL_sysvideo.h"
#include "../../events/SDL_events_c.h"
#include "../../events/SDL_keyboard_c.h"
#include "../../events/SDL_mouse_c.h"
#include "../../events/SDL_windowevents_c.h"
#include "../../render/SDL_sysrender.h"
#include "../windows/SDL_windows.h"
}

#include "../../video/winrt/SDL_winrtevents_c.h"
#include "../../video/winrt/SDL_winrtvideo_cpp.h"
#include "SDL_winrtapp_common.h"
#include "SDL_winrtapp_direct3d.h"

#if SDL_VIDEO_RENDER_D3D11
/* Calling IDXGIDevice3::Trim on the active Direct3D 11.x device is necessary
 * when Windows 8.1 apps are about to get suspended.
 */
extern "C" void D3D11_Trim(SDL_Renderer *);
#endif

// Compile-time debugging options:
// To enable, uncomment; to disable, comment them out.
// #define LOG_POINTER_EVENTS 1
// #define LOG_WINDOW_EVENTS 1
// #define LOG_ORIENTATION_EVENTS 1

// HACK, DLudwig: record a reference to the global, WinRT 'app'/view.
// SDL/WinRT will use this throughout its code.
//
// TODO, WinRT: consider replacing SDL_WinRTGlobalApp with something
// non-global, such as something created inside
// SDL_InitSubSystem(SDL_INIT_VIDEO), or something inside
// SDL_CreateWindow().
SDL_WinRTApp ^ SDL_WinRTGlobalApp = nullptr;

ref class SDLApplicationSource sealed : Windows::ApplicationModel::Core::IFrameworkViewSource
{
  public:
    virtual Windows::ApplicationModel::Core::IFrameworkView ^ CreateView();
};

IFrameworkView ^ SDLApplicationSource::CreateView()
{
    // TODO, WinRT: see if this function (CreateView) can ever get called
    // more than once.  For now, just prevent it from ever assigning
    // SDL_WinRTGlobalApp more than once.
    SDL_assert(!SDL_WinRTGlobalApp);
    SDL_WinRTApp ^ app = ref new SDL_WinRTApp();
    if (!SDL_WinRTGlobalApp) {
        SDL_WinRTGlobalApp = app;
    }
    return app;
}

int SDL_WinRTInitNonXAMLApp(int (*mainFunction)(int, char **))
{
    WINRT_SDLAppEntryPoint = mainFunction;
    auto direct3DApplicationSource = ref new SDLApplicationSource();
    CoreApplication::Run(direct3DApplicationSource);
    return 0;
}

static void WINRT_ProcessWindowSizeChange() // TODO: Pass an SDL_Window-identifying thing into WINRT_ProcessWindowSizeChange()
{
    CoreWindow ^ coreWindow = CoreWindow::GetForCurrentThread();
    if (coreWindow) {
        if (WINRT_GlobalSDLWindow) {
            SDL_Window *window = WINRT_GlobalSDLWindow;
            SDL_WindowData *data = (SDL_WindowData *)window->driverdata;

            int x = WINRT_DIPS_TO_PHYSICAL_PIXELS(data->coreWindow->Bounds.Left);
            int y = WINRT_DIPS_TO_PHYSICAL_PIXELS(data->coreWindow->Bounds.Top);
            int w = WINRT_DIPS_TO_PHYSICAL_PIXELS(data->coreWindow->Bounds.Width);
            int h = WINRT_DIPS_TO_PHYSICAL_PIXELS(data->coreWindow->Bounds.Height);

#if SDL_WINAPI_FAMILY_PHONE && (NTDDI_VERSION == NTDDI_WIN8)
            /* WinPhone 8.0 always keeps its native window size in portrait,
               regardless of orientation.  This changes in WinPhone 8.1,
               in which the native window's size changes along with
               orientation.

               Attempt to emulate WinPhone 8.1's behavior on WinPhone 8.0, with
               regards to window size.  This fixes a rendering bug that occurs
               when a WinPhone 8.0 app is rotated to either 90 or 270 degrees.
            */
            const DisplayOrientations currentOrientation = WINRT_DISPLAY_PROPERTY(CurrentOrientation);
            switch (currentOrientation) {
            case DisplayOrientations::Landscape:
            case DisplayOrientations::LandscapeFlipped:
            {
                int tmp = w;
                w = h;
                h = tmp;
            } break;
            }
#endif

            const Uint32 latestFlags = WINRT_DetectWindowFlags(window);
            if (latestFlags & SDL_WINDOW_MAXIMIZED) {
                SDL_SendWindowEvent(window, SDL_WINDOWEVENT_MAXIMIZED, 0, 0);
            } else {
                SDL_SendWindowEvent(window, SDL_WINDOWEVENT_RESTORED, 0, 0);
            }

            WINRT_UpdateWindowFlags(window, SDL_WINDOW_FULLSCREEN_DESKTOP);

            /* The window can move during a resize event, such as when maximizing
               or resizing from a corner */
            SDL_SendWindowEvent(window, SDL_WINDOWEVENT_MOVED, x, y);
            SDL_SendWindowEvent(window, SDL_WINDOWEVENT_RESIZED, w, h);
        }
    }
}

SDL_WinRTApp::SDL_WinRTApp() : m_windowClosed(false),
                               m_windowVisible(true)
{
}

void SDL_WinRTApp::Initialize(CoreApplicationView ^ applicationView)
{
    applicationView->Activated +=
        ref new TypedEventHandler<CoreApplicationView ^, IActivatedEventArgs ^>(this, &SDL_WinRTApp::OnAppActivated);

    CoreApplication::Suspending +=
        ref new EventHandler<SuspendingEventArgs ^>(this, &SDL_WinRTApp::OnSuspending);

    CoreApplication::Resuming +=
        ref new EventHandler<Platform::Object ^>(this, &SDL_WinRTApp::OnResuming);

    CoreApplication::Exiting +=
        ref new EventHandler<Platform::Object ^>(this, &SDL_WinRTApp::OnExiting);

#if NTDDI_VERSION >= NTDDI_WIN10
    /* HACK ALERT!  Xbox One doesn't seem to detect gamepads unless something
       gets registered to receive Win10's Windows.Gaming.Input.Gamepad.GamepadAdded
       events.  We'll register an event handler for these events here, to make
       sure that gamepad detection works later on, if requested.
    */
    Windows::Gaming::Input::Gamepad::GamepadAdded +=
        ref new Windows::Foundation::EventHandler<Windows::Gaming::Input::Gamepad ^>(
            this, &SDL_WinRTApp::OnGamepadAdded);
#endif
}

#if NTDDI_VERSION > NTDDI_WIN8
void SDL_WinRTApp::OnOrientationChanged(DisplayInformation ^ sender, Object ^ args)
#else
void SDL_WinRTApp::OnOrientationChanged(Object ^ sender)
#endif
{
#if LOG_ORIENTATION_EVENTS == 1
    {
        CoreWindow ^ window = CoreWindow::GetForCurrentThread();
        if (window) {
            SDL_Log("%s, current orientation=%d, native orientation=%d, auto rot. pref=%d, CoreWindow Bounds={%f,%f,%f,%f}\n",
                    __FUNCTION__,
                    WINRT_DISPLAY_PROPERTY(CurrentOrientation),
                    WINRT_DISPLAY_PROPERTY(NativeOrientation),
                    WINRT_DISPLAY_PROPERTY(AutoRotationPreferences),
                    window->Bounds.X,
                    window->Bounds.Y,
                    window->Bounds.Width,
                    window->Bounds.Height);
        } else {
            SDL_Log("%s, current orientation=%d, native orientation=%d, auto rot. pref=%d\n",
                    __FUNCTION__,
                    WINRT_DISPLAY_PROPERTY(CurrentOrientation),
                    WINRT_DISPLAY_PROPERTY(NativeOrientation),
                    WINRT_DISPLAY_PROPERTY(AutoRotationPreferences));
        }
    }
#endif

    WINRT_ProcessWindowSizeChange();

#if SDL_WINAPI_FAMILY_PHONE
    // HACK: Make sure that orientation changes
    // lead to the Direct3D renderer's viewport getting updated:
    //
    // For some reason, this doesn't seem to need to be done on Windows 8.x,
    // even when going from Landscape to LandscapeFlipped.  It only seems to
    // be needed on Windows Phone, at least when I tested on my devices.
    // I'm not currently sure why this is, but it seems to work fine. -- David L.
    //
    // TODO, WinRT: do more extensive research into why orientation changes on Win 8.x don't need D3D changes, or if they might, in some cases
    SDL_Window *window = WINRT_GlobalSDLWindow;
    if (window) {
        SDL_WindowData *data = (SDL_WindowData *)window->driverdata;
        int w = WINRT_DIPS_TO_PHYSICAL_PIXELS(data->coreWindow->Bounds.Width);
        int h = WINRT_DIPS_TO_PHYSICAL_PIXELS(data->coreWindow->Bounds.Height);
        SDL_SendWindowEvent(WINRT_GlobalSDLWindow, SDL_WINDOWEVENT_SIZE_CHANGED, w, h);
    }
#endif
}

void SDL_WinRTApp::SetWindow(CoreWindow ^ window)
{
#if LOG_WINDOW_EVENTS == 1
    SDL_Log("%s, current orientation=%d, native orientation=%d, auto rot. pref=%d, window bounds={%f, %f, %f,%f}\n",
            __FUNCTION__,
            WINRT_DISPLAY_PROPERTY(CurrentOrientation),
            WINRT_DISPLAY_PROPERTY(NativeOrientation),
            WINRT_DISPLAY_PROPERTY(AutoRotationPreferences),
            window->Bounds.X,
            window->Bounds.Y,
            window->Bounds.Width,
            window->Bounds.Height);
#endif

    window->SizeChanged +=
        ref new TypedEventHandler<CoreWindow ^, WindowSizeChangedEventArgs ^>(this, &SDL_WinRTApp::OnWindowSizeChanged);

    window->VisibilityChanged +=
        ref new TypedEventHandler<CoreWindow ^, VisibilityChangedEventArgs ^>(this, &SDL_WinRTApp::OnVisibilityChanged);

    window->Activated +=
        ref new TypedEventHandler<CoreWindow ^, WindowActivatedEventArgs ^>(this, &SDL_WinRTApp::OnWindowActivated);

    window->Closed +=
        ref new TypedEventHandler<CoreWindow ^, CoreWindowEventArgs ^>(this, &SDL_WinRTApp::OnWindowClosed);

#if !SDL_WINAPI_FAMILY_PHONE
    window->PointerCursor = ref new CoreCursor(CoreCursorType::Arrow, 0);
#endif

    window->PointerPressed +=
        ref new TypedEventHandler<CoreWindow ^, PointerEventArgs ^>(this, &SDL_WinRTApp::OnPointerPressed);

    window->PointerMoved +=
        ref new TypedEventHandler<CoreWindow ^, PointerEventArgs ^>(this, &SDL_WinRTApp::OnPointerMoved);

    window->PointerReleased +=
        ref new TypedEventHandler<CoreWindow ^, PointerEventArgs ^>(this, &SDL_WinRTApp::OnPointerReleased);

    window->PointerEntered +=
        ref new TypedEventHandler<CoreWindow ^, PointerEventArgs ^>(this, &SDL_WinRTApp::OnPointerEntered);

    window->PointerExited +=
        ref new TypedEventHandler<CoreWindow ^, PointerEventArgs ^>(this, &SDL_WinRTApp::OnPointerExited);

    window->PointerWheelChanged +=
        ref new TypedEventHandler<CoreWindow ^, PointerEventArgs ^>(this, &SDL_WinRTApp::OnPointerWheelChanged);

#if !SDL_WINAPI_FAMILY_PHONE
    // Retrieves relative-only mouse movements:
    Windows::Devices::Input::MouseDevice::GetForCurrentView()->MouseMoved +=
        ref new TypedEventHandler<MouseDevice ^, MouseEventArgs ^>(this, &SDL_WinRTApp::OnMouseMoved);
#endif

    window->KeyDown +=
        ref new TypedEventHandler<CoreWindow ^, KeyEventArgs ^>(this, &SDL_WinRTApp::OnKeyDown);

    window->KeyUp +=
        ref new TypedEventHandler<CoreWindow ^, KeyEventArgs ^>(this, &SDL_WinRTApp::OnKeyUp);

    window->CharacterReceived +=
        ref new TypedEventHandler<CoreWindow ^, CharacterReceivedEventArgs ^>(this, &SDL_WinRTApp::OnCharacterReceived);

#if NTDDI_VERSION >= NTDDI_WIN10
    Windows::UI::Core::SystemNavigationManager::GetForCurrentView()->BackRequested +=
        ref new EventHandler<BackRequestedEventArgs ^>(this, &SDL_WinRTApp::OnBackButtonPressed);
#elif SDL_WINAPI_FAMILY_PHONE
    HardwareButtons::BackPressed +=
        ref new EventHandler<BackPressedEventArgs ^>(this, &SDL_WinRTApp::OnBackButtonPressed);
#endif

#if NTDDI_VERSION > NTDDI_WIN8
    DisplayInformation::GetForCurrentView()->OrientationChanged +=
        ref new TypedEventHandler<Windows::Graphics::Display::DisplayInformation ^, Object ^>(this, &SDL_WinRTApp::OnOrientationChanged);
#else
    DisplayProperties::OrientationChanged +=
        ref new DisplayPropertiesEventHandler(this, &SDL_WinRTApp::OnOrientationChanged);
#endif

#if (WINAPI_FAMILY == WINAPI_FAMILY_APP) && (NTDDI_VERSION < NTDDI_WIN10) // for Windows 8/8.1/RT apps... (and not Phone apps)
    // Make sure we know when a user has opened the app's settings pane.
    // This is needed in order to display a privacy policy, which needs
    // to be done for network-enabled apps, as per Windows Store requirements.
    using namespace Windows::UI::ApplicationSettings;
    SettingsPane::GetForCurrentView()->CommandsRequested +=
        ref new TypedEventHandler<SettingsPane ^, SettingsPaneCommandsRequestedEventArgs ^>(this, &SDL_WinRTApp::OnSettingsPaneCommandsRequested);
#endif
}

void SDL_WinRTApp::Load(Platform::String ^ entryPoint)
{
}

void SDL_WinRTApp::Run()
{
    SDL_SetMainReady();
    if (WINRT_SDLAppEntryPoint) {
        // TODO, WinRT: pass the C-style main() a reasonably realistic
        // representation of command line arguments.
        int argc = 1;
        char **argv = (char **)SDL_malloc(2 * sizeof(*argv));
        if (!argv) {
            return;
        }
        argv[0] = SDL_strdup("WinRTApp");
        argv[1] = NULL;
        WINRT_SDLAppEntryPoint(argc, argv);
        SDL_free(argv[0]);
        SDL_free(argv);
    }
}

static bool IsSDLWindowEventPending(SDL_WindowEventID windowEventID)
{
    SDL_Event events[128];
    const int count = SDL_PeepEvents(events, sizeof(events) / sizeof(SDL_Event), SDL_PEEKEVENT, SDL_WINDOWEVENT, SDL_WINDOWEVENT);
    for (int i = 0; i < count; ++i) {
        if (events[i].window.event == windowEventID) {
            return true;
        }
    }
    return false;
}

bool SDL_WinRTApp::ShouldWaitForAppResumeEvents()
{
    /* Don't wait if the app is visible: */
    if (m_windowVisible) {
        return false;
    }

    /* Don't wait until the window-hide events finish processing.
     * Do note that if an app-suspend event is sent (as indicated
     * by SDL_APP_WILLENTERBACKGROUND and SDL_APP_DIDENTERBACKGROUND
     * events), then this code may be a moot point, as WinRT's
     * own event pump (aka ProcessEvents()) will pause regardless
     * of what we do here.  This happens on Windows Phone 8, to note.
     * Windows 8.x apps, on the other hand, may get a chance to run
     * these.
     */
    if (IsSDLWindowEventPending(SDL_WINDOWEVENT_HIDDEN)) {
        return false;
    } else if (IsSDLWindowEventPending(SDL_WINDOWEVENT_FOCUS_LOST)) {
        return false;
    } else if (IsSDLWindowEventPending(SDL_WINDOWEVENT_MINIMIZED)) {
        return false;
    }

    return true;
}

void SDL_WinRTApp::PumpEvents()
{
    if (!m_windowClosed) {
        if (!ShouldWaitForAppResumeEvents()) {
            /* This is the normal way in which events should be pumped.
             * 'ProcessAllIfPresent' will make ProcessEvents() process anywhere
             * from zero to N events, and will then return.
             */
            CoreWindow::GetForCurrentThread()->Dispatcher->ProcessEvents(CoreProcessEventsOption::ProcessAllIfPresent);
        } else {
            /* This style of event-pumping, with 'ProcessOneAndAllPending',
             * will cause anywhere from one to N events to be processed.  If
             * at least one event is processed, the call will return.  If
             * no events are pending, then the call will wait until one is
             * available, and will not return (to the caller) until this
             * happens!  This should only occur when the app is hidden.
             */
            CoreWindow::GetForCurrentThread()->Dispatcher->ProcessEvents(CoreProcessEventsOption::ProcessOneAndAllPending);
        }
    }
}

void SDL_WinRTApp::Uninitialize()
{
}

#if (WINAPI_FAMILY == WINAPI_FAMILY_APP) && (NTDDI_VERSION < NTDDI_WIN10)
void SDL_WinRTApp::OnSettingsPaneCommandsRequested(
    Windows::UI::ApplicationSettings::SettingsPane ^ p,
    Windows::UI::ApplicationSettings::SettingsPaneCommandsRequestedEventArgs ^ args)
{
    using namespace Platform;
    using namespace Windows::UI::ApplicationSettings;
    using namespace Windows::UI::Popups;

    String ^ privacyPolicyURL = nullptr;   // a URL to an app's Privacy Policy
    String ^ privacyPolicyLabel = nullptr; // label/link text
    const char *tmpHintValue = NULL;       // SDL_GetHint-retrieved value, used immediately
    wchar_t *tmpStr = NULL;                // used for UTF8 to UCS2 conversion

    // Setup a 'Privacy Policy' link, if one is available (via SDL_GetHint):
    tmpHintValue = SDL_GetHint(SDL_HINT_WINRT_PRIVACY_POLICY_URL);
    if (tmpHintValue && tmpHintValue[0] != '\0') {
        // Convert the privacy policy's URL to UCS2:
        tmpStr = WIN_UTF8ToString(tmpHintValue);
        privacyPolicyURL = ref new String(tmpStr);
        SDL_free(tmpStr);

        // Optionally retrieve custom label-text for the link.  If this isn't
        // available, a default value will be used instead.
        tmpHintValue = SDL_GetHint(SDL_HINT_WINRT_PRIVACY_POLICY_LABEL);
        if (tmpHintValue && tmpHintValue[0] != '\0') {
            tmpStr = WIN_UTF8ToString(tmpHintValue);
            privacyPolicyLabel = ref new String(tmpStr);
            SDL_free(tmpStr);
        } else {
            privacyPolicyLabel = ref new String(L"Privacy Policy");
        }

        // Register the link, along with a handler to be called if and when it is
        // clicked:
        auto cmd = ref new SettingsCommand(L"privacyPolicy", privacyPolicyLabel,
                                           ref new UICommandInvokedHandler([=](IUICommand ^) {
                                               Windows::System::Launcher::LaunchUriAsync(ref new Uri(privacyPolicyURL));
                                           }));
        args->Request->ApplicationCommands->Append(cmd);
    }
}
#endif // if (WINAPI_FAMILY == WINAPI_FAMILY_APP) && (NTDDI_VERSION < NTDDI_WIN10)

void SDL_WinRTApp::OnWindowSizeChanged(CoreWindow ^ sender, WindowSizeChangedEventArgs ^ args)
{
#if LOG_WINDOW_EVENTS == 1
    SDL_Log("%s, size={%f,%f}, bounds={%f,%f,%f,%f}, current orientation=%d, native orientation=%d, auto rot. pref=%d, WINRT_GlobalSDLWindow?=%s\n",
            __FUNCTION__,
            args->Size.Width, args->Size.Height,
            sender->Bounds.X, sender->Bounds.Y, sender->Bounds.Width, sender->Bounds.Height,
            WINRT_DISPLAY_PROPERTY(CurrentOrientation),
            WINRT_DISPLAY_PROPERTY(NativeOrientation),
            WINRT_DISPLAY_PROPERTY(AutoRotationPreferences),
            (WINRT_GlobalSDLWindow ? "yes" : "no"));
#endif

    WINRT_ProcessWindowSizeChange();
}

void SDL_WinRTApp::OnVisibilityChanged(CoreWindow ^ sender, VisibilityChangedEventArgs ^ args)
{
#if LOG_WINDOW_EVENTS == 1
    SDL_Log("%s, visible?=%s, bounds={%f,%f,%f,%f}, WINRT_GlobalSDLWindow?=%s\n",
            __FUNCTION__,
            (args->Visible ? "yes" : "no"),
            sender->Bounds.X, sender->Bounds.Y,
            sender->Bounds.Width, sender->Bounds.Height,
            (WINRT_GlobalSDLWindow ? "yes" : "no"));
#endif

    m_windowVisible = args->Visible;
    if (WINRT_GlobalSDLWindow) {
        SDL_bool wasSDLWindowSurfaceValid = WINRT_GlobalSDLWindow->surface_valid;
        Uint32 latestWindowFlags = WINRT_DetectWindowFlags(WINRT_GlobalSDLWindow);
        if (args->Visible) {
            SDL_SendWindowEvent(WINRT_GlobalSDLWindow, SDL_WINDOWEVENT_SHOWN, 0, 0);
            SDL_SendWindowEvent(WINRT_GlobalSDLWindow, SDL_WINDOWEVENT_FOCUS_GAINED, 0, 0);
            if (latestWindowFlags & SDL_WINDOW_MAXIMIZED) {
                SDL_SendWindowEvent(WINRT_GlobalSDLWindow, SDL_WINDOWEVENT_MAXIMIZED, 0, 0);
            } else {
                SDL_SendWindowEvent(WINRT_GlobalSDLWindow, SDL_WINDOWEVENT_RESTORED, 0, 0);
            }
        } else {
            SDL_SendWindowEvent(WINRT_GlobalSDLWindow, SDL_WINDOWEVENT_HIDDEN, 0, 0);
            SDL_SendWindowEvent(WINRT_GlobalSDLWindow, SDL_WINDOWEVENT_FOCUS_LOST, 0, 0);
            SDL_SendWindowEvent(WINRT_GlobalSDLWindow, SDL_WINDOWEVENT_MINIMIZED, 0, 0);
        }

        // HACK: Prevent SDL's window-hide handling code, which currently
        // triggers a fake window resize (possibly erronously), from
        // marking the SDL window's surface as invalid.
        //
        // A better solution to this probably involves figuring out if the
        // fake window resize can be prevented.
        WINRT_GlobalSDLWindow->surface_valid = wasSDLWindowSurfaceValid;
    }
}

void SDL_WinRTApp::OnWindowActivated(CoreWindow ^ sender, WindowActivatedEventArgs ^ args)
{
#if LOG_WINDOW_EVENTS == 1
    SDL_Log("%s, WINRT_GlobalSDLWindow?=%s\n\n",
            __FUNCTION__,
            (WINRT_GlobalSDLWindow ? "yes" : "no"));
#endif

    /* There's no property in Win 8.x to tell whether a window is active or
       not.  [De]activation events are, however, sent to the app.  We'll just
       record those, in case the CoreWindow gets wrapped by an SDL_Window at
       some future time.
    */
    sender->CustomProperties->Insert("SDLHelperWindowActivationState", args->WindowActivationState);

    SDL_Window *window = WINRT_GlobalSDLWindow;
    if (window) {
        if (args->WindowActivationState != CoreWindowActivationState::Deactivated) {
            SDL_SendWindowEvent(window, SDL_WINDOWEVENT_SHOWN, 0, 0);
            if (SDL_GetKeyboardFocus() != window) {
                SDL_SetKeyboardFocus(window);
            }

            /* Send a mouse-motion event as appropriate.
               This doesn't work when called from OnPointerEntered, at least
               not in WinRT CoreWindow apps (as OnPointerEntered doesn't
               appear to be called after window-reactivation, at least not
               in Windows 10, Build 10586.3 (November 2015 update, non-beta).

               Don't do it on WinPhone 8.0 though, as CoreWindow's 'PointerPosition'
               property isn't available.
             */
#if !SDL_WINAPI_FAMILY_PHONE || (NTDDI_VERSION >= NTDDI_WINBLUE)
            Point cursorPos = WINRT_TransformCursorPosition(window, sender->PointerPosition, TransformToSDLWindowSize);
            SDL_SendMouseMotion(window, 0, 0, (int)cursorPos.X, (int)cursorPos.Y);
#endif

            /* TODO, WinRT: see if the Win32 bugfix from https://hg.libsdl.org/SDL/rev/d278747da408 needs to be applied (on window activation) */
            // WIN_CheckAsyncMouseRelease(data);

            /* TODO, WinRT: implement clipboard support, if possible */
            ///*
            // * FIXME: Update keyboard state
            // */
            // WIN_CheckClipboardUpdate(data->videodata);

            // HACK: Resetting the mouse-cursor here seems to fix
            // https://bugzilla.libsdl.org/show_bug.cgi?id=3217, whereby a
            // WinRT app's mouse cursor may switch to Windows' 'wait' cursor,
            // after a user alt-tabs back into a full-screened SDL app.
            // This bug does not appear to reproduce 100% of the time.
            // It may be a bug in Windows itself (v.10.0.586.36, as tested,
            // and the most-recent as of this writing).
            SDL_SetCursor(NULL);
        } else {
            if (SDL_GetKeyboardFocus() == window) {
                SDL_SetKeyboardFocus(NULL);
            }
        }
    }
}

void SDL_WinRTApp::OnWindowClosed(CoreWindow ^ sender, CoreWindowEventArgs ^ args)
{
#if LOG_WINDOW_EVENTS == 1
    SDL_Log("%s\n", __FUNCTION__);
#endif
    m_windowClosed = true;
}

void SDL_WinRTApp::OnAppActivated(CoreApplicationView ^ applicationView, IActivatedEventArgs ^ args)
{
    CoreWindow::GetForCurrentThread()->Activate();
}

void SDL_WinRTApp::OnSuspending(Platform::Object ^ sender, SuspendingEventArgs ^ args)
{
    // Save app state asynchronously after requesting a deferral. Holding a deferral
    // indicates that the application is busy performing suspending operations. Be
    // aware that a deferral may not be held indefinitely. After about five seconds,
    // the app will be forced to exit.

    // ... but first, let the app know it's about to go to the background.
    // The separation of events may be important, given that the deferral
    // runs in a separate thread.  This'll make SDL_APP_WILLENTERBACKGROUND
    // the only event among the two that runs in the main thread.  Given
    // that a few WinRT operations can only be done from the main thread
    // (things that access the WinRT CoreWindow are one example of this),
    // this could be important.
    SDL_SendAppEvent(SDL_APP_WILLENTERBACKGROUND);

    SuspendingDeferral ^ deferral = args->SuspendingOperation->GetDeferral();
    create_task([this, deferral]() {
        // Send an app did-enter-background event immediately to observers.
        // CoreDispatcher::ProcessEvents, which is the backbone on which
        // SDL_WinRTApp::PumpEvents is built, will not return to its caller
        // once it sends out a suspend event.  Any events posted to SDL's
        // event queue won't get received until the WinRT app is resumed.
        // SDL_AddEventWatch() may be used to receive app-suspend events on
        // WinRT.
        SDL_SendAppEvent(SDL_APP_DIDENTERBACKGROUND);

        // Let the Direct3D 11 renderer prepare for the app to be backgrounded.
        // This is necessary for Windows 8.1, possibly elsewhere in the future.
        // More details at: http://msdn.microsoft.com/en-us/library/windows/apps/Hh994929.aspx
#if SDL_VIDEO_RENDER_D3D11
        if (WINRT_GlobalSDLWindow) {
            SDL_Renderer *renderer = SDL_GetRenderer(WINRT_GlobalSDLWindow);
            if (renderer && (SDL_strcmp(renderer->info.name, "direct3d11") == 0)) {
                D3D11_Trim(renderer);
            }
        }
#endif

        deferral->Complete();
    });
}

void SDL_WinRTApp::OnResuming(Platform::Object ^ sender, Platform::Object ^ args)
{
    // Restore any data or state that was unloaded on suspend. By default, data
    // and state are persisted when resuming from suspend. Note that these events
    // do not occur if the app was previously terminated.
    SDL_SendAppEvent(SDL_APP_WILLENTERFOREGROUND);
    SDL_SendAppEvent(SDL_APP_DIDENTERFOREGROUND);
}

void SDL_WinRTApp::OnExiting(Platform::Object ^ sender, Platform::Object ^ args)
{
    SDL_SendAppEvent(SDL_APP_TERMINATING);
}

static void WINRT_LogPointerEvent(const char *header, Windows::UI::Core::PointerEventArgs ^ args, Windows::Foundation::Point transformedPoint)
{
    Uint8 button, pressed;
    Windows::UI::Input::PointerPoint ^ pt = args->CurrentPoint;
    WINRT_GetSDLButtonForPointerPoint(pt, &button, &pressed);
    SDL_Log("%s: Position={%f,%f}, Transformed Pos={%f, %f}, MouseWheelDelta=%d, FrameId=%d, PointerId=%d, SDL button=%d pressed=%d\n",
            header,
            pt->Position.X, pt->Position.Y,
            transformedPoint.X, transformedPoint.Y,
            pt->Properties->MouseWheelDelta,
            pt->FrameId,
            pt->PointerId,
            button,
            pressed);
}

void SDL_WinRTApp::OnPointerPressed(CoreWindow ^ sender, PointerEventArgs ^ args)
{
#if LOG_POINTER_EVENTS
    WINRT_LogPointerEvent("pointer pressed", args, WINRT_TransformCursorPosition(WINRT_GlobalSDLWindow, args->CurrentPoint->Position, TransformToSDLWindowSize));
#endif

    WINRT_ProcessPointerPressedEvent(WINRT_GlobalSDLWindow, args->CurrentPoint);
}

void SDL_WinRTApp::OnPointerMoved(CoreWindow ^ sender, PointerEventArgs ^ args)
{
#if LOG_POINTER_EVENTS
    WINRT_LogPointerEvent("pointer moved", args, WINRT_TransformCursorPosition(WINRT_GlobalSDLWindow, args->CurrentPoint->Position, TransformToSDLWindowSize));
#endif

    WINRT_ProcessPointerMovedEvent(WINRT_GlobalSDLWindow, args->CurrentPoint);
}

void SDL_WinRTApp::OnPointerReleased(CoreWindow ^ sender, PointerEventArgs ^ args)
{
#if LOG_POINTER_EVENTS
    WINRT_LogPointerEvent("pointer released", args, WINRT_TransformCursorPosition(WINRT_GlobalSDLWindow, args->CurrentPoint->Position, TransformToSDLWindowSize));
#endif

    WINRT_ProcessPointerReleasedEvent(WINRT_GlobalSDLWindow, args->CurrentPoint);
}

void SDL_WinRTApp::OnPointerEntered(CoreWindow ^ sender, PointerEventArgs ^ args)
{
#if LOG_POINTER_EVENTS
    WINRT_LogPointerEvent("pointer entered", args, WINRT_TransformCursorPosition(WINRT_GlobalSDLWindow, args->CurrentPoint->Position, TransformToSDLWindowSize));
#endif

    WINRT_ProcessPointerEnteredEvent(WINRT_GlobalSDLWindow, args->CurrentPoint);
}

void SDL_WinRTApp::OnPointerExited(CoreWindow ^ sender, PointerEventArgs ^ args)
{
#if LOG_POINTER_EVENTS
    WINRT_LogPointerEvent("pointer exited", args, WINRT_TransformCursorPosition(WINRT_GlobalSDLWindow, args->CurrentPoint->Position, TransformToSDLWindowSize));
#endif

    WINRT_ProcessPointerExitedEvent(WINRT_GlobalSDLWindow, args->CurrentPoint);
}

void SDL_WinRTApp::OnPointerWheelChanged(CoreWindow ^ sender, PointerEventArgs ^ args)
{
#if LOG_POINTER_EVENTS
    WINRT_LogPointerEvent("pointer wheel changed", args, WINRT_TransformCursorPosition(WINRT_GlobalSDLWindow, args->CurrentPoint->Position, TransformToSDLWindowSize));
#endif

    WINRT_ProcessPointerWheelChangedEvent(WINRT_GlobalSDLWindow, args->CurrentPoint);
}

void SDL_WinRTApp::OnMouseMoved(MouseDevice ^ mouseDevice, MouseEventArgs ^ args)
{
    WINRT_ProcessMouseMovedEvent(WINRT_GlobalSDLWindow, args);
}

void SDL_WinRTApp::OnKeyDown(Windows::UI::Core::CoreWindow ^ sender, Windows::UI::Core::KeyEventArgs ^ args)
{
    WINRT_ProcessKeyDownEvent(args);
}

void SDL_WinRTApp::OnKeyUp(Windows::UI::Core::CoreWindow ^ sender, Windows::UI::Core::KeyEventArgs ^ args)
{
    WINRT_ProcessKeyUpEvent(args);
}

void SDL_WinRTApp::OnCharacterReceived(Windows::UI::Core::CoreWindow ^ sender, Windows::UI::Core::CharacterReceivedEventArgs ^ args)
{
    WINRT_ProcessCharacterReceivedEvent(args);
}

template <typename BackButtonEventArgs>
static void WINRT_OnBackButtonPressed(BackButtonEventArgs ^ args)
{
    SDL_SendKeyboardKey(SDL_PRESSED, SDL_SCANCODE_AC_BACK);
    SDL_SendKeyboardKey(SDL_RELEASED, SDL_SCANCODE_AC_BACK);

    if (SDL_GetHintBoolean(SDL_HINT_WINRT_HANDLE_BACK_BUTTON, SDL_FALSE)) {
        args->Handled = true;
    }
}

#if NTDDI_VERSION >= NTDDI_WIN10
void SDL_WinRTApp::OnBackButtonPressed(Platform::Object ^ sender, Windows::UI::Core::BackRequestedEventArgs ^ args)

{
    WINRT_OnBackButtonPressed(args);
}
#elif SDL_WINAPI_FAMILY_PHONE
void SDL_WinRTApp::OnBackButtonPressed(Platform::Object ^ sender, Windows::Phone::UI::Input::BackPressedEventArgs ^ args)

{
    WINRT_OnBackButtonPressed(args);
}
#endif

#if NTDDI_VERSION >= NTDDI_WIN10
void SDL_WinRTApp::OnGamepadAdded(Platform::Object ^ sender, Windows::Gaming::Input::Gamepad ^ gamepad)
{
    /* HACK ALERT: Nothing needs to be done here, as this method currently
       only exists to allow something to be registered with Win10's
       GamepadAdded event, an operation that seems to be necessary to get
       Xinput-based detection to work on Xbox One.
    */
}
#endif

/* vi: set ts=4 sw=4 expandtab: */
