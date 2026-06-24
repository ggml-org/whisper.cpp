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

#ifdef SDL_VIDEO_DRIVER_PANDORA

/* SDL internals */
#include "../SDL_sysvideo.h"
#include "SDL_version.h"
#include "SDL_syswm.h"
#include "SDL_loadso.h"
#include "SDL_events.h"
#include "../../events/SDL_mouse_c.h"
#include "../../events/SDL_keyboard_c.h"

/* PND declarations */
#include "SDL_pandora.h"
#include "SDL_pandora_events.h"

/* WIZ declarations */
#include "GLES/gl.h"
#ifdef WIZ_GLES_LITE
static NativeWindowType hNativeWnd = 0; /* A handle to the window we will create. */
#endif

static void PND_destroy(SDL_VideoDevice * device)
{
    if (device->driverdata) {
        SDL_free(device->driverdata);
        device->driverdata = NULL;
    }
    SDL_free(device);
}

static SDL_VideoDevice *PND_create(void)
{
    SDL_VideoDevice *device;
    SDL_VideoData *phdata;

    /* Initialize SDL_VideoDevice structure */
    device = (SDL_VideoDevice *) SDL_calloc(1, sizeof(SDL_VideoDevice));
    if (!device) {
        SDL_OutOfMemory();
        return NULL;
    }

    /* Initialize internal Pandora specific data */
    phdata = (SDL_VideoData *) SDL_calloc(1, sizeof(SDL_VideoData));
    if (!phdata) {
        SDL_OutOfMemory();
        SDL_free(device);
        return NULL;
    }

    device->driverdata = phdata;

    phdata->egl_initialized = SDL_TRUE;


    /* Setup amount of available displays */
    device->num_displays = 0;

    /* Set device free function */
    device->free = PND_destroy;

    /* Setup all functions which we can handle */
    device->VideoInit = PND_videoinit;
    device->VideoQuit = PND_videoquit;
    device->GetDisplayModes = PND_getdisplaymodes;
    device->SetDisplayMode = PND_setdisplaymode;
    device->CreateSDLWindow = PND_createwindow;
    device->CreateSDLWindowFrom = PND_createwindowfrom;
    device->SetWindowTitle = PND_setwindowtitle;
    device->SetWindowIcon = PND_setwindowicon;
    device->SetWindowPosition = PND_setwindowposition;
    device->SetWindowSize = PND_setwindowsize;
    device->ShowWindow = PND_showwindow;
    device->HideWindow = PND_hidewindow;
    device->RaiseWindow = PND_raisewindow;
    device->MaximizeWindow = PND_maximizewindow;
    device->MinimizeWindow = PND_minimizewindow;
    device->RestoreWindow = PND_restorewindow;
    device->DestroyWindow = PND_destroywindow;
#if 0
    device->GetWindowWMInfo = PND_getwindowwminfo;
#endif
    device->GL_LoadLibrary = PND_gl_loadlibrary;
    device->GL_GetProcAddress = PND_gl_getprocaddres;
    device->GL_UnloadLibrary = PND_gl_unloadlibrary;
    device->GL_CreateContext = PND_gl_createcontext;
    device->GL_MakeCurrent = PND_gl_makecurrent;
    device->GL_SetSwapInterval = PND_gl_setswapinterval;
    device->GL_GetSwapInterval = PND_gl_getswapinterval;
    device->GL_SwapWindow = PND_gl_swapwindow;
    device->GL_DeleteContext = PND_gl_deletecontext;
    device->PumpEvents = PND_PumpEvents;

    /* !!! FIXME: implement SetWindowBordered */

    return device;
}

VideoBootStrap PND_bootstrap = {
#ifdef WIZ_GLES_LITE
    "wiz",
    "SDL Wiz Video Driver",
#else
    "pandora",
    "SDL Pandora Video Driver",
#endif
    PND_create,
    NULL /* no ShowMessageBox implementation */
};

/*****************************************************************************/
/* SDL Video and Display initialization/handling functions                   */
/*****************************************************************************/
int PND_videoinit(_THIS)
{
    SDL_VideoDisplay display;
    SDL_DisplayMode current_mode;

    SDL_zero(current_mode);
#ifdef WIZ_GLES_LITE
    current_mode.w = 320;
    current_mode.h = 240;
#else
    current_mode.w = 800;
    current_mode.h = 480;
#endif
    current_mode.refresh_rate = 60;
    current_mode.format = SDL_PIXELFORMAT_RGB565;
    current_mode.driverdata = NULL;

    SDL_zero(display);
    display.desktop_mode = current_mode;
    display.current_mode = current_mode;
    display.driverdata = NULL;

    SDL_AddVideoDisplay(&display, SDL_FALSE);

    return 1;
}

void PND_videoquit(_THIS)
{

}

void PND_getdisplaymodes(_THIS, SDL_VideoDisplay * display)
{

}

int PND_setdisplaymode(_THIS, SDL_VideoDisplay * display, SDL_DisplayMode * mode)
{
    return 0;
}

int PND_createwindow(_THIS, SDL_Window * window)
{
    SDL_VideoData *phdata = (SDL_VideoData *) _this->driverdata;

    SDL_WindowData *wdata;

    /* Allocate window internal data */
    wdata = (SDL_WindowData *) SDL_calloc(1, sizeof(SDL_WindowData));
    if (!wdata) {
        return SDL_OutOfMemory();
    }

    /* Setup driver data for this window */
    window->driverdata = wdata;

    /* Check if window must support OpenGL ES rendering */
    if ((window->flags & SDL_WINDOW_OPENGL) == SDL_WINDOW_OPENGL) {

        EGLBoolean initstatus;

        /* Mark this window as OpenGL ES compatible */
        wdata->uses_gles = SDL_TRUE;

        /* Create connection to OpenGL ES */
        if (phdata->egl_display == EGL_NO_DISPLAY) {
            phdata->egl_display = eglGetDisplay((NativeDisplayType) 0);
            if (phdata->egl_display == EGL_NO_DISPLAY) {
                return SDL_SetError("PND: Can't get connection to OpenGL ES");
            }

            initstatus = eglInitialize(phdata->egl_display, NULL, NULL);
            if (initstatus != EGL_TRUE) {
                return SDL_SetError("PND: Can't init OpenGL ES library");
            }
        }

        phdata->egl_refcount++;
    }

    /* Window has been successfully created */
    return 0;
}

int PND_createwindowfrom(_THIS, SDL_Window * window, const void *data)
{
    return -1;
}

void PND_setwindowtitle(_THIS, SDL_Window * window)
{
}
void PND_setwindowicon(_THIS, SDL_Window * window, SDL_Surface * icon)
{
}
void PND_setwindowposition(_THIS, SDL_Window * window)
{
}
void PND_setwindowsize(_THIS, SDL_Window * window)
{
}
void PND_showwindow(_THIS, SDL_Window * window)
{
}
void PND_hidewindow(_THIS, SDL_Window * window)
{
}
void PND_raisewindow(_THIS, SDL_Window * window)
{
}
void PND_maximizewindow(_THIS, SDL_Window * window)
{
}
void PND_minimizewindow(_THIS, SDL_Window * window)
{
}
void PND_restorewindow(_THIS, SDL_Window * window)
{
}
void PND_destroywindow(_THIS, SDL_Window * window)
{
    SDL_VideoData *phdata = (SDL_VideoData *) _this->driverdata;
    eglTerminate(phdata->egl_display);
}

/*****************************************************************************/
/* SDL Window Manager function                                               */
/*****************************************************************************/
#if 0
SDL_bool PND_getwindowwminfo(_THIS, SDL_Window * window, struct SDL_SysWMinfo *info)
{
    if (info->version.major <= SDL_MAJOR_VERSION) {
        return SDL_TRUE;
    } else {
        SDL_SetError("application not compiled with SDL %d",
                     SDL_MAJOR_VERSION);
        return SDL_FALSE;
    }

    /* Failed to get window manager information */
    return SDL_FALSE;
}
#endif

/*****************************************************************************/
/* SDL OpenGL/OpenGL ES functions                                            */
/*****************************************************************************/
int PND_gl_loadlibrary(_THIS, const char *path)
{
    /* Check if OpenGL ES library is specified for GF driver */
    if (!path) {
        path = SDL_getenv("SDL_OPENGL_LIBRARY");
        if (!path) {
            path = SDL_getenv("SDL_OPENGLES_LIBRARY");
        }
    }

    /* Check if default library loading requested */
    if (!path) {
        /* Already linked with GF library which provides egl* subset of  */
        /* functions, use Common profile of OpenGL ES library by default */
#ifdef WIZ_GLES_LITE
    path = "/lib/libopengles_lite.so";
#else
        path = "/usr/lib/libGLES_CM.so";
#endif
    }

    /* Load dynamic library */
    _this->gl_config.dll_handle = SDL_LoadObject(path);
    if (!_this->gl_config.dll_handle) {
        /* Failed to load new GL ES library */
        return SDL_SetError("PND: Failed to locate OpenGL ES library");
    }

    /* Store OpenGL ES library path and name */
    SDL_strlcpy(_this->gl_config.driver_path, path,
                SDL_arraysize(_this->gl_config.driver_path));

    /* New OpenGL ES library is loaded */
    return 0;
}

void *PND_gl_getprocaddres(_THIS, const char *proc)
{
    void *function_address;

    /* Try to get function address through the egl interface */
    function_address = eglGetProcAddress(proc);
    if (function_address) {
        return function_address;
    }

    /* Then try to get function in the OpenGL ES library */
    if (_this->gl_config.dll_handle) {
        function_address =
            SDL_LoadFunction(_this->gl_config.dll_handle, proc);
        if (function_address) {
            return function_address;
        }
    }

    /* Failed to get GL ES function address pointer */
    SDL_SetError("PND: Cannot locate OpenGL ES function name");
    return NULL;
}

void PND_gl_unloadlibrary(_THIS)
{
    SDL_VideoData *phdata = (SDL_VideoData *) _this->driverdata;

    if (phdata->egl_initialized == SDL_TRUE) {
        /* Unload OpenGL ES library */
        if (_this->gl_config.dll_handle) {
            SDL_UnloadObject(_this->gl_config.dll_handle);
            _this->gl_config.dll_handle = NULL;
        }
    } else {
        SDL_SetError("PND: GF initialization failed, no OpenGL ES support");
    }
}

SDL_GLContext PND_gl_createcontext(_THIS, SDL_Window * window)
{
    SDL_VideoData *phdata = (SDL_VideoData *) _this->driverdata;
    SDL_WindowData *wdata = (SDL_WindowData *) window->driverdata;
    EGLBoolean status;
    EGLint configs;
    uint32_t attr_pos;
    EGLint attr_value;
    EGLint cit;

    /* Check if EGL was initialized */
    if (phdata->egl_initialized != SDL_TRUE) {
        SDL_SetError("PND: EGL initialization failed, no OpenGL ES support");
        return NULL;
    }

    /* Prepare attributes list to pass them to OpenGL ES */
    attr_pos = 0;
    wdata->gles_attributes[attr_pos++] = EGL_SURFACE_TYPE;
    wdata->gles_attributes[attr_pos++] = EGL_WINDOW_BIT;
    wdata->gles_attributes[attr_pos++] = EGL_RED_SIZE;
    wdata->gles_attributes[attr_pos++] = _this->gl_config.red_size;
    wdata->gles_attributes[attr_pos++] = EGL_GREEN_SIZE;
    wdata->gles_attributes[attr_pos++] = _this->gl_config.green_size;
    wdata->gles_attributes[attr_pos++] = EGL_BLUE_SIZE;
    wdata->gles_attributes[attr_pos++] = _this->gl_config.blue_size;
    wdata->gles_attributes[attr_pos++] = EGL_ALPHA_SIZE;

    /* Setup alpha size in bits */
    if (_this->gl_config.alpha_size) {
        wdata->gles_attributes[attr_pos++] = _this->gl_config.alpha_size;
    } else {
        wdata->gles_attributes[attr_pos++] = EGL_DONT_CARE;
    }

    /* Setup color buffer size */
    if (_this->gl_config.buffer_size) {
        wdata->gles_attributes[attr_pos++] = EGL_BUFFER_SIZE;
        wdata->gles_attributes[attr_pos++] = _this->gl_config.buffer_size;
    } else {
        wdata->gles_attributes[attr_pos++] = EGL_BUFFER_SIZE;
        wdata->gles_attributes[attr_pos++] = EGL_DONT_CARE;
    }

    /* Setup depth buffer bits */
    wdata->gles_attributes[attr_pos++] = EGL_DEPTH_SIZE;
    wdata->gles_attributes[attr_pos++] = _this->gl_config.depth_size;

    /* Setup stencil bits */
    if (_this->gl_config.stencil_size) {
        wdata->gles_attributes[attr_pos++] = EGL_STENCIL_SIZE;
        wdata->gles_attributes[attr_pos++] = _this->gl_config.buffer_size;
    } else {
        wdata->gles_attributes[attr_pos++] = EGL_STENCIL_SIZE;
        wdata->gles_attributes[attr_pos++] = EGL_DONT_CARE;
    }

    /* Set number of samples in multisampling */
    if (_this->gl_config.multisamplesamples) {
        wdata->gles_attributes[attr_pos++] = EGL_SAMPLES;
        wdata->gles_attributes[attr_pos++] =
            _this->gl_config.multisamplesamples;
    }

    /* Multisample buffers, OpenGL ES 1.0 spec defines 0 or 1 buffer */
    if (_this->gl_config.multisamplebuffers) {
        wdata->gles_attributes[attr_pos++] = EGL_SAMPLE_BUFFERS;
        wdata->gles_attributes[attr_pos++] =
            _this->gl_config.multisamplebuffers;
    }

    /* Finish attributes list */
    wdata->gles_attributes[attr_pos] = EGL_NONE;

    /* Request first suitable framebuffer configuration */
    status = eglChooseConfig(phdata->egl_display, wdata->gles_attributes,
                             wdata->gles_configs, 1, &configs);
    if (status != EGL_TRUE) {
        SDL_SetError("PND: Can't find closest configuration for OpenGL ES");
        return NULL;
    }

    /* Check if nothing has been found, try "don't care" settings */
    if (configs == 0) {
        int32_t it;
        int32_t jt;
        GLint depthbits[4] = { 32, 24, 16, EGL_DONT_CARE };

        for (it = 0; it < 4; it++) {
            for (jt = 16; jt >= 0; jt--) {
                /* Don't care about color buffer bits, use what exist */
                /* Replace previous set data with EGL_DONT_CARE       */
                attr_pos = 0;
                wdata->gles_attributes[attr_pos++] = EGL_SURFACE_TYPE;
                wdata->gles_attributes[attr_pos++] = EGL_WINDOW_BIT;
                wdata->gles_attributes[attr_pos++] = EGL_RED_SIZE;
                wdata->gles_attributes[attr_pos++] = EGL_DONT_CARE;
                wdata->gles_attributes[attr_pos++] = EGL_GREEN_SIZE;
                wdata->gles_attributes[attr_pos++] = EGL_DONT_CARE;
                wdata->gles_attributes[attr_pos++] = EGL_BLUE_SIZE;
                wdata->gles_attributes[attr_pos++] = EGL_DONT_CARE;
                wdata->gles_attributes[attr_pos++] = EGL_ALPHA_SIZE;
                wdata->gles_attributes[attr_pos++] = EGL_DONT_CARE;
                wdata->gles_attributes[attr_pos++] = EGL_BUFFER_SIZE;
                wdata->gles_attributes[attr_pos++] = EGL_DONT_CARE;

                /* Try to find requested or smallest depth */
                if (_this->gl_config.depth_size) {
                    wdata->gles_attributes[attr_pos++] = EGL_DEPTH_SIZE;
                    wdata->gles_attributes[attr_pos++] = depthbits[it];
                } else {
                    wdata->gles_attributes[attr_pos++] = EGL_DEPTH_SIZE;
                    wdata->gles_attributes[attr_pos++] = EGL_DONT_CARE;
                }

                if (_this->gl_config.stencil_size) {
                    wdata->gles_attributes[attr_pos++] = EGL_STENCIL_SIZE;
                    wdata->gles_attributes[attr_pos++] = jt;
                } else {
                    wdata->gles_attributes[attr_pos++] = EGL_STENCIL_SIZE;
                    wdata->gles_attributes[attr_pos++] = EGL_DONT_CARE;
                }

                wdata->gles_attributes[attr_pos++] = EGL_SAMPLES;
                wdata->gles_attributes[attr_pos++] = EGL_DONT_CARE;
                wdata->gles_attributes[attr_pos++] = EGL_SAMPLE_BUFFERS;
                wdata->gles_attributes[attr_pos++] = EGL_DONT_CARE;
                wdata->gles_attributes[attr_pos] = EGL_NONE;

                /* Request first suitable framebuffer configuration */
                status =
                    eglChooseConfig(phdata->egl_display,
                                    wdata->gles_attributes,
                                    wdata->gles_configs, 1, &configs);

                if (status != EGL_TRUE) {
                    SDL_SetError
                        ("PND: Can't find closest configuration for OpenGL ES");
                    return NULL;
                }
                if (configs != 0) {
                    break;
                }
            }
            if (configs != 0) {
                break;
            }
        }

        /* No available configs */
        if (configs == 0) {
            SDL_SetError("PND: Can't find any configuration for OpenGL ES");
            return NULL;
        }
    }

    /* Initialize config index */
    wdata->gles_config = 0;

    /* Now check each configuration to find out the best */
    for (cit = 0; cit < configs; cit++) {
        uint32_t stencil_found;
        uint32_t depth_found;

        stencil_found = 0;
        depth_found = 0;

        if (_this->gl_config.stencil_size) {
            status =
                eglGetConfigAttrib(phdata->egl_display,
                                   wdata->gles_configs[cit], EGL_STENCIL_SIZE,
                                   &attr_value);
            if (status == EGL_TRUE) {
                if (attr_value != 0) {
                    stencil_found = 1;
                }
            }
        } else {
            stencil_found = 1;
        }

        if (_this->gl_config.depth_size) {
            status =
                eglGetConfigAttrib(phdata->egl_display,
                                   wdata->gles_configs[cit], EGL_DEPTH_SIZE,
                                   &attr_value);
            if (status == EGL_TRUE) {
                if (attr_value != 0) {
                    depth_found = 1;
                }
            }
        } else {
            depth_found = 1;
        }

        /* Exit from loop if found appropriate configuration */
        if ((depth_found != 0) && (stencil_found != 0)) {
            break;
        }
    }

    /* If best could not be found, use first */
    if (cit == configs) {
        cit = 0;
    }
    wdata->gles_config = cit;

    /* Create OpenGL ES context */
    wdata->gles_context =
        eglCreateContext(phdata->egl_display,
                         wdata->gles_configs[wdata->gles_config], NULL, NULL);
    if (wdata->gles_context == EGL_NO_CONTEXT) {
        SDL_SetError("PND: OpenGL ES context creation has been failed");
        return NULL;
    }

#ifdef WIZ_GLES_LITE
    if (!hNativeWnd) {
        hNativeWnd = (NativeWindowType)SDL_malloc(16*1024);
        if (!hNativeWnd) {
            printf("Error: Wiz framebuffer allocatation failed\n");
        } else {
            printf("SDL: Wiz framebuffer allocated: %X\n", hNativeWnd);
        }
    }
    else {
        printf("SDL: Wiz framebuffer already allocated: %X\n", hNativeWnd);
    }

    wdata->gles_surface =
    eglCreateWindowSurface(phdata->egl_display,
                   wdata->gles_configs[wdata->gles_config],
                   hNativeWnd, NULL );
#else
    wdata->gles_surface =
        eglCreateWindowSurface(phdata->egl_display,
                               wdata->gles_configs[wdata->gles_config],
                               (NativeWindowType) 0, NULL);
#endif


    if (wdata->gles_surface == 0) {
        SDL_SetError("Error : eglCreateWindowSurface failed;");
        return NULL;
    }

    /* Make just created context current */
    status =
        eglMakeCurrent(phdata->egl_display, wdata->gles_surface,
                       wdata->gles_surface, wdata->gles_context);
    if (status != EGL_TRUE) {
        /* Destroy OpenGL ES surface */
        eglDestroySurface(phdata->egl_display, wdata->gles_surface);
        eglDestroyContext(phdata->egl_display, wdata->gles_context);
        wdata->gles_context = EGL_NO_CONTEXT;
        SDL_SetError("PND: Can't set OpenGL ES context on creation");
        return NULL;
    }

    _this->gl_config.accelerated = 1;

    /* Always clear stereo enable, since OpenGL ES do not supports stereo */
    _this->gl_config.stereo = 0;

    /* Get back samples and samplebuffers configurations. Rest framebuffer */
    /* parameters could be obtained through the OpenGL ES API              */
    status =
        eglGetConfigAttrib(phdata->egl_display,
                           wdata->gles_configs[wdata->gles_config],
                           EGL_SAMPLES, &attr_value);
    if (status == EGL_TRUE) {
        _this->gl_config.multisamplesamples = attr_value;
    }
    status =
        eglGetConfigAttrib(phdata->egl_display,
                           wdata->gles_configs[wdata->gles_config],
                           EGL_SAMPLE_BUFFERS, &attr_value);
    if (status == EGL_TRUE) {
        _this->gl_config.multisamplebuffers = attr_value;
    }

    /* Get back stencil and depth buffer sizes */
    status =
        eglGetConfigAttrib(phdata->egl_display,
                           wdata->gles_configs[wdata->gles_config],
                           EGL_DEPTH_SIZE, &attr_value);
    if (status == EGL_TRUE) {
        _this->gl_config.depth_size = attr_value;
    }
    status =
        eglGetConfigAttrib(phdata->egl_display,
                           wdata->gles_configs[wdata->gles_config],
                           EGL_STENCIL_SIZE, &attr_value);
    if (status == EGL_TRUE) {
        _this->gl_config.stencil_size = attr_value;
    }

    /* Under PND OpenGL ES output can't be double buffered */
    _this->gl_config.double_buffer = 0;

    /* GL ES context was successfully created */
    return wdata->gles_context;
}

int PND_gl_makecurrent(_THIS, SDL_Window * window, SDL_GLContext context)
{
    SDL_VideoData *phdata = (SDL_VideoData *) _this->driverdata;
    SDL_WindowData *wdata;
    EGLBoolean status;

    if (phdata->egl_initialized != SDL_TRUE) {
        return SDL_SetError("PND: GF initialization failed, no OpenGL ES support");
    }

    if ((!window) && (!context)) {
        status =
            eglMakeCurrent(phdata->egl_display, EGL_NO_SURFACE,
                           EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (status != EGL_TRUE) {
            /* Failed to set current GL ES context */
            return SDL_SetError("PND: Can't set OpenGL ES context");
        }
    } else {
        wdata = (SDL_WindowData *) window->driverdata;
        if (wdata->gles_surface == EGL_NO_SURFACE) {
            return SDL_SetError
                ("PND: OpenGL ES surface is not initialized for this window");
        }
        if (wdata->gles_context == EGL_NO_CONTEXT) {
            return SDL_SetError
                ("PND: OpenGL ES context is not initialized for this window");
        }
        if (wdata->gles_context != context) {
            return SDL_SetError
                ("PND: OpenGL ES context is not belong to this window");
        }
        status =
            eglMakeCurrent(phdata->egl_display, wdata->gles_surface,
                           wdata->gles_surface, wdata->gles_context);
        if (status != EGL_TRUE) {
            /* Failed to set current GL ES context */
            return SDL_SetError("PND: Can't set OpenGL ES context");
        }
    }
    return 0;
}

int PND_gl_setswapinterval(_THIS, int interval)
{
    SDL_VideoData *phdata = (SDL_VideoData *) _this->driverdata;
    EGLBoolean status;

    if (phdata->egl_initialized != SDL_TRUE) {
        return SDL_SetError("PND: EGL initialization failed, no OpenGL ES support");
    }

    /* Check if OpenGL ES connection has been initialized */
    if (phdata->egl_display != EGL_NO_DISPLAY) {
        /* Set swap OpenGL ES interval */
        status = eglSwapInterval(phdata->egl_display, interval);
        if (status == EGL_TRUE) {
            /* Return success to upper level */
            phdata->swapinterval = interval;
            return 0;
        }
    }

    /* Failed to set swap interval */
    return SDL_SetError("PND: Cannot set swap interval");
}

int PND_gl_getswapinterval(_THIS)
{
    return ((SDL_VideoData *) _this->driverdata)->swapinterval;
}

int PND_gl_swapwindow(_THIS, SDL_Window * window)
{
    SDL_VideoData *phdata = (SDL_VideoData *) _this->driverdata;
    SDL_WindowData *wdata = (SDL_WindowData *) window->driverdata;

    if (phdata->egl_initialized != SDL_TRUE) {
        return SDL_SetError("PND: GLES initialization failed, no OpenGL ES support");
    }

    /* Many applications do not uses glFinish(), so we call it for them */
    glFinish();

    /* Wait until OpenGL ES rendering is completed */
    eglWaitGL();

    eglSwapBuffers(phdata->egl_display, wdata->gles_surface);
    return 0;
}

void PND_gl_deletecontext(_THIS, SDL_GLContext context)
{
    SDL_VideoData *phdata = (SDL_VideoData *) _this->driverdata;
    EGLBoolean status;

    if (phdata->egl_initialized != SDL_TRUE) {
        SDL_SetError("PND: GLES initialization failed, no OpenGL ES support");
        return;
    }

    /* Check if OpenGL ES connection has been initialized */
    if (phdata->egl_display != EGL_NO_DISPLAY) {
        if (context != EGL_NO_CONTEXT) {
            status = eglDestroyContext(phdata->egl_display, context);
            if (status != EGL_TRUE) {
                /* Error during OpenGL ES context destroying */
                SDL_SetError("PND: OpenGL ES context destroy error");
                return;
            }
        }
    }

#ifdef WIZ_GLES_LITE
    if (hNativeWnd != 0) {
      SDL_free(hNativeWnd);
      hNativeWnd = 0;
      printf("SDL: Wiz framebuffer released\n");
    }
#endif

    return;
}

#endif /* SDL_VIDEO_DRIVER_PANDORA */

/* vi: set ts=4 sw=4 expandtab: */
