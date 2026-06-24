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

/* Windows includes: */
#include <windows.h>
#ifdef __cplusplus_winrt
#include <agile.h>
#endif

/* SDL includes: */
#include "SDL_video.h"
#include "SDL_events.h"

#if NTDDI_VERSION >= NTDDI_WINBLUE /* ApplicationView's functionality only becomes \
                                      useful for SDL in Win[Phone] 8.1 and up.     \
                                      Plus, it is not available at all in WinPhone 8.0. */
#define SDL_WINRT_USE_APPLICATIONVIEW 1
#endif

extern "C" {
#include "../SDL_sysvideo.h"
#include "../SDL_egl_c.h"
}

/* Private display data */
typedef struct SDL_VideoData
{
    /* An object created by ANGLE/WinRT (OpenGL ES 2 for WinRT) that gets
     * passed to eglGetDisplay and eglCreateWindowSurface:
     */
    IUnknown *winrtEglWindow;

    /* Event token(s), for unregistering WinRT event handler(s).
       These are just a struct with a 64-bit integer inside them
    */
    Windows::Foundation::EventRegistrationToken gameBarIsInputRedirectedToken;

    /* A WinRT DisplayRequest, used for implementing SDL_*ScreenSaver() functions.
     * This is really a pointer to a 'ABI::Windows::System::Display::IDisplayRequest *',
     * It's casted to 'IUnknown *', to help with building SDL.
     */
    IUnknown *displayRequest;
} SDL_VideoData;

/* The global, WinRT, SDL Window.
   For now, SDL/WinRT only supports one window (due to platform limitations of
   WinRT.
*/
extern SDL_Window *WINRT_GlobalSDLWindow;

/* Updates one or more SDL_Window flags, by querying the OS' native windowing APIs.
   SDL_Window flags that can be updated should be specified in 'mask'.
*/
extern void WINRT_UpdateWindowFlags(SDL_Window *window, Uint32 mask);
extern "C" Uint32 WINRT_DetectWindowFlags(SDL_Window *window); /* detects flags w/o applying them */

/* Display mode internals */
// typedef struct
//{
//     Windows::Graphics::Display::DisplayOrientations currentOrientation;
// } SDL_DisplayModeData;

#ifdef __cplusplus_winrt

/* A convenience macro to get a WinRT display property */
#if NTDDI_VERSION > NTDDI_WIN8
#define WINRT_DISPLAY_PROPERTY(NAME) (Windows::Graphics::Display::DisplayInformation::GetForCurrentView()->NAME)
#else
#define WINRT_DISPLAY_PROPERTY(NAME) (Windows::Graphics::Display::DisplayProperties::NAME)
#endif

/* Converts DIPS to/from physical pixels */
#define WINRT_DIPS_TO_PHYSICAL_PIXELS(DIPS)    ((int)(0.5f + (((float)(DIPS) * (float)WINRT_DISPLAY_PROPERTY(LogicalDpi)) / 96.f)))
#define WINRT_PHYSICAL_PIXELS_TO_DIPS(PHYSPIX) (((float)(PHYSPIX)*96.f) / WINRT_DISPLAY_PROPERTY(LogicalDpi))

/* Internal window data */
struct SDL_WindowData
{
    SDL_Window *sdlWindow;
    Platform::Agile<Windows::UI::Core::CoreWindow> coreWindow;
#ifdef SDL_VIDEO_OPENGL_EGL
    EGLSurface egl_surface;
#endif
#ifdef SDL_WINRT_USE_APPLICATIONVIEW
    Windows::UI::ViewManagement::ApplicationView ^ appView;
#endif
};

#endif // ifdef __cplusplus_winrt
