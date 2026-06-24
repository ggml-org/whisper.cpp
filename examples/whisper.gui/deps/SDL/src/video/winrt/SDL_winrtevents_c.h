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
#include "SDL_config.h"

extern "C" {
#include "../SDL_sysvideo.h"
}

/*
 * Internal-use, C-style functions:
 */

#ifdef __cplusplus
extern "C" {
#endif

extern void WINRT_InitTouch(_THIS);
extern void WINRT_PumpEvents(_THIS);

#ifdef __cplusplus
}
#endif

/*
 * Internal-use, C++/CX functions:
 */
#ifdef __cplusplus_winrt

/* Pointers (Mice, Touch, etc.) */
typedef enum
{
    NormalizeZeroToOne,
    TransformToSDLWindowSize
} WINRT_CursorNormalizationType;
extern Windows::Foundation::Point WINRT_TransformCursorPosition(SDL_Window *window,
                                                                Windows::Foundation::Point rawPosition,
                                                                WINRT_CursorNormalizationType normalization);
extern SDL_bool WINRT_GetSDLButtonForPointerPoint(Windows::UI::Input::PointerPoint ^ pt, Uint8 *button, Uint8 *pressed);
extern void WINRT_ProcessPointerPressedEvent(SDL_Window *window, Windows::UI::Input::PointerPoint ^ pointerPoint);
extern void WINRT_ProcessPointerMovedEvent(SDL_Window *window, Windows::UI::Input::PointerPoint ^ pointerPoint);
extern void WINRT_ProcessPointerReleasedEvent(SDL_Window *window, Windows::UI::Input::PointerPoint ^ pointerPoint);
extern void WINRT_ProcessPointerEnteredEvent(SDL_Window *window, Windows::UI::Input::PointerPoint ^ pointerPoint);
extern void WINRT_ProcessPointerExitedEvent(SDL_Window *window, Windows::UI::Input::PointerPoint ^ pointerPoint);
extern void WINRT_ProcessPointerWheelChangedEvent(SDL_Window *window, Windows::UI::Input::PointerPoint ^ pointerPoint);
extern void WINRT_ProcessMouseMovedEvent(SDL_Window *window, Windows::Devices::Input::MouseEventArgs ^ args);

/* Keyboard */
extern void WINRT_ProcessKeyDownEvent(Windows::UI::Core::KeyEventArgs ^ args);
extern void WINRT_ProcessKeyUpEvent(Windows::UI::Core::KeyEventArgs ^ args);
extern void WINRT_ProcessCharacterReceivedEvent(Windows::UI::Core::CharacterReceivedEventArgs ^ args);

#if NTDDI_VERSION >= NTDDI_WIN10
extern void WINTRT_InitialiseInputPaneEvents(_THIS);
extern SDL_bool WINRT_HasScreenKeyboardSupport(_THIS);
extern void WINRT_ShowScreenKeyboard(_THIS, SDL_Window *window);
extern void WINRT_HideScreenKeyboard(_THIS, SDL_Window *window);
extern SDL_bool WINRT_IsScreenKeyboardShown(_THIS, SDL_Window *window);
#endif // NTDDI_VERSION >= ...

/* XAML Thread Management */
extern void WINRT_CycleXAMLThread(void);

#endif // ifdef __cplusplus_winrt

/* vi: set ts=4 sw=4 expandtab: */
