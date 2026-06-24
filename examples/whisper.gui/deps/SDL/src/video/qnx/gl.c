/*
  Simple DirectMedia Layer
  Copyright (C) 2017 BlackBerry Limited

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
#include "sdl_qnx.h"

static EGLDisplay   egl_disp;

/**
 * Detertmines the pixel format to use based on the current display and EGL
 * configuration.
 * @param   egl_conf    EGL configuration to use
 * @return  A SCREEN_FORMAT* constant for the pixel format to use
 */
static int chooseFormat(EGLConfig egl_conf)
{
    EGLint buffer_bit_depth;
    EGLint alpha_bit_depth;

    eglGetConfigAttrib(egl_disp, egl_conf, EGL_BUFFER_SIZE, &buffer_bit_depth);
    eglGetConfigAttrib(egl_disp, egl_conf, EGL_ALPHA_SIZE, &alpha_bit_depth);

    switch (buffer_bit_depth) {
        case 32:
            return SCREEN_FORMAT_RGBX8888;
        case 24:
            return SCREEN_FORMAT_RGB888;
        case 16:
            switch (alpha_bit_depth) {
                case 4:
                    return SCREEN_FORMAT_RGBX4444;
                case 1:
                    return SCREEN_FORMAT_RGBA5551;
                default:
                    return SCREEN_FORMAT_RGB565;
            }
        default:
            return 0;
    }
}

/**
 * Enumerates the supported EGL configurations and chooses a suitable one.
 * @param[out]  pconf   The chosen configuration
 * @param[out]  pformat The chosen pixel format
 * @return 0 if successful, -1 on error
 */
int glGetConfig(EGLConfig *pconf, int *pformat)
{
    EGLConfig egl_conf = (EGLConfig)0;
    EGLConfig *egl_configs;
    EGLint egl_num_configs;
    EGLint val;
    EGLBoolean rc;
    EGLint i;

    // Determine the numbfer of configurations.
    rc = eglGetConfigs(egl_disp, NULL, 0, &egl_num_configs);
    if (rc != EGL_TRUE) {
        return -1;
    }

    if (egl_num_configs == 0) {
        return -1;
    }

    // Allocate enough memory for all configurations.
    egl_configs = SDL_malloc(egl_num_configs * sizeof(*egl_configs));
    if (!egl_configs) {
        return -1;
    }

    // Get the list of configurations.
    rc = eglGetConfigs(egl_disp, egl_configs, egl_num_configs,
                       &egl_num_configs);
    if (rc != EGL_TRUE) {
        SDL_free(egl_configs);
        return -1;
    }

    // Find a good configuration.
    for (i = 0; i < egl_num_configs; i++) {
        eglGetConfigAttrib(egl_disp, egl_configs[i], EGL_SURFACE_TYPE, &val);
        if (!(val & EGL_WINDOW_BIT)) {
            continue;
        }

        eglGetConfigAttrib(egl_disp, egl_configs[i], EGL_RENDERABLE_TYPE, &val);
        if (!(val & EGL_OPENGL_ES2_BIT)) {
            continue;
        }

        eglGetConfigAttrib(egl_disp, egl_configs[i], EGL_DEPTH_SIZE, &val);
        if (val == 0) {
            continue;
        }

        egl_conf = egl_configs[i];
        break;
    }

    SDL_free(egl_configs);
    *pconf = egl_conf;
    *pformat = chooseFormat(egl_conf);

    return 0;
}

/**
 * Initializes the EGL library.
 * @param   _THIS
 * @param   name    unused
 * @return  0 if successful, -1 on error
 */
int glLoadLibrary(_THIS, const char *name)
{
    EGLNativeDisplayType    disp_id = EGL_DEFAULT_DISPLAY;

    egl_disp = eglGetDisplay(disp_id);
    if (egl_disp == EGL_NO_DISPLAY) {
        return -1;
    }

    if (eglInitialize(egl_disp, NULL, NULL) == EGL_FALSE) {
        return -1;
    }

    return 0;
}

/**
 * Finds the address of an EGL extension function.
 * @param   proc    Function name
 * @return  Function address
 */
void *glGetProcAddress(_THIS, const char *proc)
{
    return eglGetProcAddress(proc);
}

/**
 * Associates the given window with the necessary EGL structures for drawing and
 * displaying content.
 * @param   _THIS
 * @param   window  The SDL window to create the context for
 * @return  A pointer to the created context, if successful, NULL on error
 */
SDL_GLContext glCreateContext(_THIS, SDL_Window *window)
{
    window_impl_t   *impl = (window_impl_t *)window->driverdata;
    EGLContext      context;
    EGLSurface      surface;

    struct {
        EGLint client_version[2];
        EGLint none;
    } egl_ctx_attr = {
        .client_version = { EGL_CONTEXT_CLIENT_VERSION, 2 },
        .none = EGL_NONE
    };

    struct {
        EGLint render_buffer[2];
        EGLint none;
    } egl_surf_attr = {
        .render_buffer = { EGL_RENDER_BUFFER, EGL_BACK_BUFFER },
        .none = EGL_NONE
    };

    context = eglCreateContext(egl_disp, impl->conf, EGL_NO_CONTEXT,
                               (EGLint *)&egl_ctx_attr);
    if (context == EGL_NO_CONTEXT) {
        return NULL;
    }

    surface = eglCreateWindowSurface(egl_disp, impl->conf, impl->window,
                                     (EGLint *)&egl_surf_attr);
    if (surface == EGL_NO_SURFACE) {
        return NULL;
    }

    eglMakeCurrent(egl_disp, surface, surface, context);

    impl->surface = surface;
    return context;
}

/**
 * Sets a new value for the number of frames to display before swapping buffers.
 * @param   _THIS
 * @param   interval    New interval value
 * @return  0 if successful, -1 on error
 */
int glSetSwapInterval(_THIS, int interval)
{
    if (eglSwapInterval(egl_disp, interval) != EGL_TRUE) {
        return -1;
    }

    return 0;
}

/**
 * Swaps the EGL buffers associated with the given window
 * @param   _THIS
 * @param   window  Window to swap buffers for
 * @return  0 if successful, -1 on error
 */
int glSwapWindow(_THIS, SDL_Window *window)
{
    /* !!! FIXME: should we migrate this all over to use SDL_egl.c? */
    window_impl_t   *impl = (window_impl_t *)window->driverdata;
    return eglSwapBuffers(egl_disp, impl->surface) == EGL_TRUE ? 0 : -1;
}

/**
 * Makes the given context the current one for drawing operations.
 * @param   _THIS
 * @param   window  SDL window associated with the context (maybe NULL)
 * @param   context The context to activate
 * @return  0 if successful, -1 on error
 */
int glMakeCurrent(_THIS, SDL_Window *window, SDL_GLContext context)
{
    window_impl_t   *impl;
    EGLSurface      surface = NULL;

    if (window) {
        impl = (window_impl_t *)window->driverdata;
        surface = impl->surface;
    }

    if (eglMakeCurrent(egl_disp, surface, surface, context) != EGL_TRUE) {
        return -1;
    }

    return 0;
}

/**
 * Destroys a context.
 * @param   _THIS
 * @param   context The context to destroy
 */
void glDeleteContext(_THIS, SDL_GLContext context)
{
    eglDestroyContext(egl_disp, context);
}

/**
 * Terminates access to the EGL library.
 * @param   _THIS
 */
void glUnloadLibrary(_THIS)
{
    eglTerminate(egl_disp);
}
