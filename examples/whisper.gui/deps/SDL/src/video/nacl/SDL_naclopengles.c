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

#ifdef SDL_VIDEO_DRIVER_NACL

/* NaCl SDL video GLES 2 driver implementation */

#include "SDL_video.h"
#include "SDL_naclvideo.h"

#ifdef HAVE_DLOPEN
#include "dlfcn.h"
#endif

#include "ppapi/gles2/gl2ext_ppapi.h"
#include "ppapi_simple/ps.h"

/* GL functions */
int NACL_GLES_LoadLibrary(_THIS, const char *path)
{
    /* FIXME: Support dynamic linking when PNACL supports it */
    return glInitializePPAPI(PSGetInterface) == 0;
}

void *NACL_GLES_GetProcAddress(_THIS, const char *proc)
{
#ifdef HAVE_DLOPEN
    return dlsym( 0 /* RTLD_DEFAULT */, proc);
#else
    return NULL;
#endif
}

void NACL_GLES_UnloadLibrary(_THIS)
{
    /* FIXME: Support dynamic linking when PNACL supports it */
    glTerminatePPAPI();
}

int NACL_GLES_MakeCurrent(_THIS, SDL_Window * window, SDL_GLContext sdl_context)
{
    SDL_VideoData *driverdata = (SDL_VideoData *) _this->driverdata;
    /* FIXME: Check threading issues...otherwise use a hardcoded _this->context across all threads */
    driverdata->ppb_instance->BindGraphics(driverdata->instance, (PP_Resource) sdl_context);
    glSetCurrentContextPPAPI((PP_Resource) sdl_context);
    return 0;
}

SDL_GLContext NACL_GLES_CreateContext(_THIS, SDL_Window * window)
{
    SDL_VideoData *driverdata = (SDL_VideoData *) _this->driverdata;
    PP_Resource context, share_context = 0;
    /* 64 seems nice. */
    Sint32 attribs[64];
    int i = 0;

    if (_this->gl_config.share_with_current_context) {
        share_context = (PP_Resource) SDL_GL_GetCurrentContext();
    }

    /* FIXME: Some ATTRIBS from PP_Graphics3DAttrib are not set here */

    attribs[i++] = PP_GRAPHICS3DATTRIB_WIDTH;
    attribs[i++] = window->w;
    attribs[i++] = PP_GRAPHICS3DATTRIB_HEIGHT;
    attribs[i++] = window->h;
    attribs[i++] = PP_GRAPHICS3DATTRIB_RED_SIZE;
    attribs[i++] = _this->gl_config.red_size;
    attribs[i++] = PP_GRAPHICS3DATTRIB_GREEN_SIZE;
    attribs[i++] = _this->gl_config.green_size;
    attribs[i++] = PP_GRAPHICS3DATTRIB_BLUE_SIZE;
    attribs[i++] = _this->gl_config.blue_size;

    if (_this->gl_config.alpha_size) {
        attribs[i++] = PP_GRAPHICS3DATTRIB_ALPHA_SIZE;
        attribs[i++] = _this->gl_config.alpha_size;
    }

    /*if (_this->gl_config.buffer_size) {
        attribs[i++] = EGL_BUFFER_SIZE;
        attribs[i++] = _this->gl_config.buffer_size;
    }*/

    attribs[i++] = PP_GRAPHICS3DATTRIB_DEPTH_SIZE;
    attribs[i++] = _this->gl_config.depth_size;

    if (_this->gl_config.stencil_size) {
        attribs[i++] = PP_GRAPHICS3DATTRIB_STENCIL_SIZE;
        attribs[i++] = _this->gl_config.stencil_size;
    }

    if (_this->gl_config.multisamplebuffers) {
        attribs[i++] = PP_GRAPHICS3DATTRIB_SAMPLE_BUFFERS;
        attribs[i++] = _this->gl_config.multisamplebuffers;
    }

    if (_this->gl_config.multisamplesamples) {
        attribs[i++] = PP_GRAPHICS3DATTRIB_SAMPLES;
        attribs[i++] = _this->gl_config.multisamplesamples;
    }

    attribs[i++] = PP_GRAPHICS3DATTRIB_NONE;

    context = driverdata->ppb_graphics->Create(driverdata->instance, share_context, attribs);

    if (context) {
        /* We need to make the context current, otherwise nothing works */
        SDL_GL_MakeCurrent(window, (SDL_GLContext) context);
    }

    return (SDL_GLContext) context;
}



int NACL_GLES_SetSwapInterval(_THIS, int interval)
{
    /* STUB */
    return SDL_Unsupported();
}

int NACL_GLES_GetSwapInterval(_THIS)
{
    /* STUB */
    return 0;
}

int NACL_GLES_SwapWindow(_THIS, SDL_Window * window)
{
    SDL_VideoData *driverdata = (SDL_VideoData *) _this->driverdata;
    struct PP_CompletionCallback callback = { NULL, 0, PP_COMPLETIONCALLBACK_FLAG_NONE };
    if (driverdata->ppb_graphics->SwapBuffers((PP_Resource) SDL_GL_GetCurrentContext(), callback ) != 0) {
        return SDL_SetError("SwapBuffers failed");
    }
    return 0;
}

void NACL_GLES_DeleteContext(_THIS, SDL_GLContext context)
{
    SDL_VideoData *driverdata = (SDL_VideoData *) _this->driverdata;
    driverdata->ppb_core->ReleaseResource((PP_Resource) context);
}

#endif /* SDL_VIDEO_DRIVER_NACL */

/* vi: set ts=4 sw=4 expandtab: */
