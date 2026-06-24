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

/* SDL includes */
#include "SDL_winrtevents_c.h"
#include "SDL_winrtmouse_c.h"
#include "SDL_winrtvideo_cpp.h"
#include "SDL_system.h"

extern "C" {
#include "../SDL_sysvideo.h"
#include "../../events/SDL_events_c.h"
#include "../../events/SDL_mouse_c.h"
#include "../../events/SDL_touch_c.h"
}

/* File-specific globals: */
static SDL_TouchID WINRT_TouchID = 1;

void WINRT_InitTouch(_THIS)
{
    SDL_AddTouch(WINRT_TouchID, SDL_TOUCH_DEVICE_DIRECT, "");
}

//
// Applies necessary geometric transformations to raw cursor positions:
//
Windows::Foundation::Point
WINRT_TransformCursorPosition(SDL_Window *window,
                              Windows::Foundation::Point rawPosition,
                              WINRT_CursorNormalizationType normalization)
{
    using namespace Windows::UI::Core;
    using namespace Windows::Graphics::Display;

    if (!window) {
        return rawPosition;
    }

    SDL_WindowData *windowData = (SDL_WindowData *)window->driverdata;
    if (windowData->coreWindow == nullptr) {
        // For some reason, the window isn't associated with a CoreWindow.
        // This might end up being the case as XAML support is extended.
        // For now, if there's no CoreWindow attached to the SDL_Window,
        // don't do any transforms.

        // TODO, WinRT: make sure touch input coordinate ranges are correct when using XAML support
        return rawPosition;
    }

    // The CoreWindow can only be accessed on certain thread(s).
    SDL_assert(CoreWindow::GetForCurrentThread() != nullptr);

    CoreWindow ^ nativeWindow = windowData->coreWindow.Get();
    Windows::Foundation::Point outputPosition;

    // Compute coordinates normalized from 0..1.
    // If the coordinates need to be sized to the SDL window,
    // we'll do that after.
#if !SDL_WINAPI_FAMILY_PHONE || (NTDDI_VERSION > NTDDI_WIN8)
    outputPosition.X = rawPosition.X / nativeWindow->Bounds.Width;
    outputPosition.Y = rawPosition.Y / nativeWindow->Bounds.Height;
#else
    switch (WINRT_DISPLAY_PROPERTY(CurrentOrientation)) {
    case DisplayOrientations::Portrait:
        outputPosition.X = rawPosition.X / nativeWindow->Bounds.Width;
        outputPosition.Y = rawPosition.Y / nativeWindow->Bounds.Height;
        break;
    case DisplayOrientations::PortraitFlipped:
        outputPosition.X = 1.0f - (rawPosition.X / nativeWindow->Bounds.Width);
        outputPosition.Y = 1.0f - (rawPosition.Y / nativeWindow->Bounds.Height);
        break;
    case DisplayOrientations::Landscape:
        outputPosition.X = rawPosition.Y / nativeWindow->Bounds.Height;
        outputPosition.Y = 1.0f - (rawPosition.X / nativeWindow->Bounds.Width);
        break;
    case DisplayOrientations::LandscapeFlipped:
        outputPosition.X = 1.0f - (rawPosition.Y / nativeWindow->Bounds.Height);
        outputPosition.Y = rawPosition.X / nativeWindow->Bounds.Width;
        break;
    default:
        break;
    }
#endif

    if (normalization == TransformToSDLWindowSize) {
        outputPosition.X *= ((float32)window->w);
        outputPosition.Y *= ((float32)window->h);
    }

    return outputPosition;
}

SDL_bool WINRT_GetSDLButtonForPointerPoint(Windows::UI::Input::PointerPoint ^ pt, Uint8 *button, Uint8 *pressed)
{
    using namespace Windows::UI::Input;

#if SDL_WINAPI_FAMILY_PHONE
    *button = SDL_BUTTON_LEFT;
    return SDL_TRUE;
#else
    switch (pt->Properties->PointerUpdateKind) {
    case PointerUpdateKind::LeftButtonPressed:
    case PointerUpdateKind::LeftButtonReleased:
        *button = SDL_BUTTON_LEFT;
        *pressed = (pt->Properties->PointerUpdateKind == PointerUpdateKind::LeftButtonPressed);
        return SDL_TRUE;

    case PointerUpdateKind::RightButtonPressed:
    case PointerUpdateKind::RightButtonReleased:
        *button = SDL_BUTTON_RIGHT;
        *pressed = (pt->Properties->PointerUpdateKind == PointerUpdateKind::RightButtonPressed);
        return SDL_TRUE;

    case PointerUpdateKind::MiddleButtonPressed:
    case PointerUpdateKind::MiddleButtonReleased:
        *button = SDL_BUTTON_MIDDLE;
        *pressed = (pt->Properties->PointerUpdateKind == PointerUpdateKind::MiddleButtonPressed);
        return SDL_TRUE;

    case PointerUpdateKind::XButton1Pressed:
    case PointerUpdateKind::XButton1Released:
        *button = SDL_BUTTON_X1;
        *pressed = (pt->Properties->PointerUpdateKind == PointerUpdateKind::XButton1Pressed);
        return SDL_TRUE;

    case PointerUpdateKind::XButton2Pressed:
    case PointerUpdateKind::XButton2Released:
        *button = SDL_BUTTON_X2;
        *pressed = (pt->Properties->PointerUpdateKind == PointerUpdateKind::XButton2Pressed);
        return SDL_TRUE;

    default:
        break;
    }
#endif

    *button = 0;
    *pressed = 0;
    return SDL_FALSE;
}

// const char *
// WINRT_ConvertPointerUpdateKindToString(Windows::UI::Input::PointerUpdateKind kind)
//{
//     using namespace Windows::UI::Input;
//
//     switch (kind)
//     {
//         case PointerUpdateKind::Other:
//             return "Other";
//         case PointerUpdateKind::LeftButtonPressed:
//             return "LeftButtonPressed";
//         case PointerUpdateKind::LeftButtonReleased:
//             return "LeftButtonReleased";
//         case PointerUpdateKind::RightButtonPressed:
//             return "RightButtonPressed";
//         case PointerUpdateKind::RightButtonReleased:
//             return "RightButtonReleased";
//         case PointerUpdateKind::MiddleButtonPressed:
//             return "MiddleButtonPressed";
//         case PointerUpdateKind::MiddleButtonReleased:
//             return "MiddleButtonReleased";
//         case PointerUpdateKind::XButton1Pressed:
//             return "XButton1Pressed";
//         case PointerUpdateKind::XButton1Released:
//             return "XButton1Released";
//         case PointerUpdateKind::XButton2Pressed:
//             return "XButton2Pressed";
//         case PointerUpdateKind::XButton2Released:
//             return "XButton2Released";
//     }
//
//     return "";
// }

static bool WINRT_IsTouchEvent(Windows::UI::Input::PointerPoint ^ pointerPoint)
{
#if SDL_WINAPI_FAMILY_PHONE
    return true;
#else
    using namespace Windows::Devices::Input;
    switch (pointerPoint->PointerDevice->PointerDeviceType) {
    case PointerDeviceType::Touch:
    case PointerDeviceType::Pen:
        return true;
    default:
        return false;
    }
#endif
}

void WINRT_ProcessPointerPressedEvent(SDL_Window *window, Windows::UI::Input::PointerPoint ^ pointerPoint)
{
    if (!window) {
        return;
    }

    if (!WINRT_IsTouchEvent(pointerPoint)) {
        Uint8 button, pressed;
        WINRT_GetSDLButtonForPointerPoint(pointerPoint, &button, &pressed);
        SDL_assert(pressed == 1);
        SDL_SendMouseButton(window, 0, SDL_PRESSED, button);
    } else {
        Windows::Foundation::Point normalizedPoint = WINRT_TransformCursorPosition(window, pointerPoint->Position, NormalizeZeroToOne);
        Windows::Foundation::Point windowPoint = WINRT_TransformCursorPosition(window, pointerPoint->Position, TransformToSDLWindowSize);

        SDL_SendTouch(
            WINRT_TouchID,
            (SDL_FingerID)pointerPoint->PointerId,
            window,
            SDL_TRUE,
            normalizedPoint.X,
            normalizedPoint.Y,
            pointerPoint->Properties->Pressure);
    }
}

void WINRT_ProcessPointerMovedEvent(SDL_Window *window, Windows::UI::Input::PointerPoint ^ pointerPoint)
{
    if (!window || WINRT_UsingRelativeMouseMode) {
        return;
    }

    Windows::Foundation::Point normalizedPoint = WINRT_TransformCursorPosition(window, pointerPoint->Position, NormalizeZeroToOne);
    Windows::Foundation::Point windowPoint = WINRT_TransformCursorPosition(window, pointerPoint->Position, TransformToSDLWindowSize);

    if (!WINRT_IsTouchEvent(pointerPoint)) {
        /* For some odd reason Moved events are used for multiple mouse buttons */
        Uint8 button, pressed;
        if (WINRT_GetSDLButtonForPointerPoint(pointerPoint, &button, &pressed)) {
            SDL_SendMouseButton(window, 0, pressed, button);
        }

        SDL_SendMouseMotion(window, 0, 0, (int)windowPoint.X, (int)windowPoint.Y);
    } else {
        SDL_SendTouchMotion(
            WINRT_TouchID,
            (SDL_FingerID)pointerPoint->PointerId,
            window,
            normalizedPoint.X,
            normalizedPoint.Y,
            pointerPoint->Properties->Pressure);
    }
}

void WINRT_ProcessPointerReleasedEvent(SDL_Window *window, Windows::UI::Input::PointerPoint ^ pointerPoint)
{
    if (!window) {
        return;
    }

    if (!WINRT_IsTouchEvent(pointerPoint)) {
        Uint8 button, pressed;
        WINRT_GetSDLButtonForPointerPoint(pointerPoint, &button, &pressed);
        SDL_assert(pressed == 0);
        SDL_SendMouseButton(window, 0, SDL_RELEASED, button);
    } else {
        Windows::Foundation::Point normalizedPoint = WINRT_TransformCursorPosition(window, pointerPoint->Position, NormalizeZeroToOne);

        SDL_SendTouch(
            WINRT_TouchID,
            (SDL_FingerID)pointerPoint->PointerId,
            window,
            SDL_FALSE,
            normalizedPoint.X,
            normalizedPoint.Y,
            pointerPoint->Properties->Pressure);
    }
}

void WINRT_ProcessPointerEnteredEvent(SDL_Window *window, Windows::UI::Input::PointerPoint ^ pointerPoint)
{
    if (!window) {
        return;
    }

    if (!WINRT_IsTouchEvent(pointerPoint)) {
        SDL_SetMouseFocus(window);
    }
}

void WINRT_ProcessPointerExitedEvent(SDL_Window *window, Windows::UI::Input::PointerPoint ^ pointerPoint)
{
    if (!window) {
        return;
    }

    if (!WINRT_IsTouchEvent(pointerPoint)) {
        SDL_SetMouseFocus(NULL);
    }
}

void WINRT_ProcessPointerWheelChangedEvent(SDL_Window *window, Windows::UI::Input::PointerPoint ^ pointerPoint)
{
    if (!window) {
        return;
    }

    float motion = (float)pointerPoint->Properties->MouseWheelDelta / WHEEL_DELTA;
    SDL_SendMouseWheel(window, 0, 0, (float)motion, SDL_MOUSEWHEEL_NORMAL);
}

void WINRT_ProcessMouseMovedEvent(SDL_Window *window, Windows::Devices::Input::MouseEventArgs ^ args)
{
    if (!window || !WINRT_UsingRelativeMouseMode) {
        return;
    }

    // DLudwig, 2012-12-28: On some systems, namely Visual Studio's Windows
    // Simulator, as well as Windows 8 in a Parallels 8 VM, MouseEventArgs'
    // MouseDelta field often reports very large values.  More information
    // on this can be found at the following pages on MSDN:
    //  - http://social.msdn.microsoft.com/Forums/en-US/winappswithnativecode/thread/a3c789fa-f1c5-49c4-9c0a-7db88d0f90f8
    //  - https://connect.microsoft.com/VisualStudio/Feedback/details/756515
    //
    // The values do not appear to be as large when running on some systems,
    // most notably a Surface RT.  Furthermore, the values returned by
    // CoreWindow's PointerMoved event, and sent to this class' OnPointerMoved
    // method, do not ever appear to be large, even when MouseEventArgs'
    // MouseDelta is reporting to the contrary.
    //
    // On systems with the large-values behavior, it appears that the values
    // get reported as if the screen's size is 65536 units in both the X and Y
    // dimensions.  This can be viewed by using Windows' now-private, "Raw Input"
    // APIs.  (GetRawInputData, RegisterRawInputDevices, WM_INPUT, etc.)
    //
    // MSDN's documentation on MouseEventArgs' MouseDelta field (at
    // http://msdn.microsoft.com/en-us/library/windows/apps/windows.devices.input.mouseeventargs.mousedelta ),
    // does not seem to indicate (to me) that its values should be so large.  It
    // says that its values should be a "change in screen location".  I could
    // be misinterpreting this, however a post on MSDN from a Microsoft engineer (see:
    // http://social.msdn.microsoft.com/Forums/en-US/winappswithnativecode/thread/09a9868e-95bb-4858-ba1a-cb4d2c298d62 ),
    // indicates that these values are in DIPs, which is the same unit used
    // by CoreWindow's PointerMoved events (via the Position field in its CurrentPoint
    // property.  See http://msdn.microsoft.com/en-us/library/windows/apps/windows.ui.input.pointerpoint.position.aspx
    // for details.)
    //
    // To note, PointerMoved events are sent a 'RawPosition' value (via the
    // CurrentPoint property in MouseEventArgs), however these do not seem
    // to exhibit the same large-value behavior.
    //
    // The values passed via PointerMoved events can't always be used for relative
    // mouse motion, unfortunately.  Its values are bound to the cursor's position,
    // which stops when it hits one of the screen's edges.  This can be a problem in
    // first person shooters, whereby it is normal for mouse motion to travel far
    // along any one axis for a period of time.  MouseMoved events do not have the
    // screen-bounding limitation, and can be used regardless of where the system's
    // cursor is.
    //
    // One possible workaround would be to programmatically set the cursor's
    // position to the screen's center (when SDL's relative mouse mode is enabled),
    // however WinRT does not yet seem to have the ability to set the cursor's
    // position via a public API.  Win32 did this via an API call, SetCursorPos,
    // however WinRT makes this function be private.  Apps that use it won't get
    // approved for distribution in the Windows Store.  I've yet to be able to find
    // a suitable, store-friendly counterpart for WinRT.
    //
    // There may be some room for a workaround whereby OnPointerMoved's values
    // are compared to the values from OnMouseMoved in order to detect
    // when this bug is active.  A suitable transformation could then be made to
    // OnMouseMoved's values.  For now, however, the system-reported values are sent
    // to SDL with minimal transformation: from native screen coordinates (in DIPs)
    // to SDL window coordinates.
    //
    const Windows::Foundation::Point mouseDeltaInDIPs((float)args->MouseDelta.X, (float)args->MouseDelta.Y);
    const Windows::Foundation::Point mouseDeltaInSDLWindowCoords = WINRT_TransformCursorPosition(window, mouseDeltaInDIPs, TransformToSDLWindowSize);
    SDL_SendMouseMotion(
        window,
        0,
        1,
        SDL_lroundf(mouseDeltaInSDLWindowCoords.X),
        SDL_lroundf(mouseDeltaInSDLWindowCoords.Y));
}

#endif // SDL_VIDEO_DRIVER_WINRT
