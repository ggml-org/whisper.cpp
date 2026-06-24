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

#ifdef SDL_VIDEO_DRIVER_PSP

#include <stdlib.h>
#include <string.h>

#include "SDL_error.h"
#include "SDL_pspvideo.h"
#include "SDL_pspgl_c.h"

/*****************************************************************************/
/* SDL OpenGL/OpenGL ES functions                                            */
/*****************************************************************************/
#define EGLCHK(stmt)                           \
    do {                                       \
        EGLint err;                            \
                                               \
        stmt;                                  \
        err = eglGetError();                   \
        if (err != EGL_SUCCESS) {              \
            SDL_SetError("EGL error %d", err); \
            return 0;                          \
        }                                      \
    } while (0)

int PSP_GL_LoadLibrary(_THIS, const char *path)
{
    return 0;
}

/* pspgl doesn't provide this call, so stub it out since SDL requires it.
#define GLSTUB(func,params) void func params {}

GLSTUB(glOrtho,(GLdouble left, GLdouble right, GLdouble bottom, GLdouble top,
                    GLdouble zNear, GLdouble zFar))
*/
void *PSP_GL_GetProcAddress(_THIS, const char *proc)
{
    return eglGetProcAddress(proc);
}

void PSP_GL_UnloadLibrary(_THIS)
{
    eglTerminate(_this->gl_data->display);
}

static EGLint width = 480;
static EGLint height = 272;

SDL_GLContext PSP_GL_CreateContext(_THIS, SDL_Window *window)
{

    SDL_WindowData *wdata = (SDL_WindowData *)window->driverdata;

    EGLint attribs[32];
    EGLDisplay display;
    EGLContext context;
    EGLSurface surface;
    EGLConfig config;
    EGLint num_configs;
    int i;

    /* EGL init taken from glutCreateWindow() in PSPGL's glut.c. */
    EGLCHK(display = eglGetDisplay(0));
    EGLCHK(eglInitialize(display, NULL, NULL));
    wdata->uses_gles = SDL_TRUE;
    window->flags |= SDL_WINDOW_FULLSCREEN;

    /* Setup the config based on SDL's current values. */
    i = 0;
    attribs[i++] = EGL_RED_SIZE;
    attribs[i++] = _this->gl_config.red_size;
    attribs[i++] = EGL_GREEN_SIZE;
    attribs[i++] = _this->gl_config.green_size;
    attribs[i++] = EGL_BLUE_SIZE;
    attribs[i++] = _this->gl_config.blue_size;
    attribs[i++] = EGL_DEPTH_SIZE;
    attribs[i++] = _this->gl_config.depth_size;

    if (_this->gl_config.alpha_size) {
        attribs[i++] = EGL_ALPHA_SIZE;
        attribs[i++] = _this->gl_config.alpha_size;
    }
    if (_this->gl_config.stencil_size) {
        attribs[i++] = EGL_STENCIL_SIZE;
        attribs[i++] = _this->gl_config.stencil_size;
    }

    attribs[i++] = EGL_NONE;

    EGLCHK(eglChooseConfig(display, attribs, &config, 1, &num_configs));

    if (num_configs == 0) {
        SDL_SetError("No valid EGL configs for requested mode");
        return 0;
    }

    EGLCHK(eglGetConfigAttrib(display, config, EGL_WIDTH, &width));
    EGLCHK(eglGetConfigAttrib(display, config, EGL_HEIGHT, &height));

    EGLCHK(context = eglCreateContext(display, config, NULL, NULL));
    EGLCHK(surface = eglCreateWindowSurface(display, config, 0, NULL));
    EGLCHK(eglMakeCurrent(display, surface, surface, context));

    _this->gl_data->display = display;
    _this->gl_data->context = context;
    _this->gl_data->surface = surface;

    return context;
}

int PSP_GL_MakeCurrent(_THIS, SDL_Window *window, SDL_GLContext context)
{
    if (!eglMakeCurrent(_this->gl_data->display, _this->gl_data->surface,
                        _this->gl_data->surface, _this->gl_data->context)) {
        return SDL_SetError("Unable to make EGL context current");
    }
    return 0;
}

int PSP_GL_SetSwapInterval(_THIS, int interval)
{
    EGLBoolean status;
    status = eglSwapInterval(_this->gl_data->display, interval);
    if (status == EGL_TRUE) {
        /* Return success to upper level */
        _this->gl_data->swapinterval = interval;
        return 0;
    }
    /* Failed to set swap interval */
    return SDL_SetError("Unable to set the EGL swap interval");
}

int PSP_GL_GetSwapInterval(_THIS)
{
    return _this->gl_data->swapinterval;
}

int PSP_GL_SwapWindow(_THIS, SDL_Window *window)
{
    if (!eglSwapBuffers(_this->gl_data->display, _this->gl_data->surface)) {
        return SDL_SetError("eglSwapBuffers() failed");
    }
    return 0;
}

void PSP_GL_DeleteContext(_THIS, SDL_GLContext context)
{
    SDL_VideoData *phdata = (SDL_VideoData *)_this->driverdata;
    EGLBoolean status;

    if (phdata->egl_initialized != SDL_TRUE) {
        SDL_SetError("PSP: GLES initialization failed, no OpenGL ES support");
        return;
    }

    /* Check if OpenGL ES connection has been initialized */
    if (_this->gl_data->display != EGL_NO_DISPLAY) {
        if (context != EGL_NO_CONTEXT) {
            status = eglDestroyContext(_this->gl_data->display, context);
            if (status != EGL_TRUE) {
                /* Error during OpenGL ES context destroying */
                SDL_SetError("PSP: OpenGL ES context destroy error");
                return;
            }
        }
    }

    return;
}

#endif /* SDL_VIDEO_DRIVER_PSP */

/* vi: set ts=4 sw=4 expandtab: */
