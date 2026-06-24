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

#ifdef SDL_VIDEO_DRIVER_DIRECTFB

#include "SDL_DirectFB_video.h"
#include "SDL_DirectFB_shape.h"
#include "SDL_DirectFB_window.h"

#include "../SDL_shape_internals.h"

SDL_WindowShaper *DirectFB_CreateShaper(SDL_Window* window)
{
    SDL_WindowShaper* result = NULL;
    SDL_ShapeData* data;
    int resized_properly;

    result = SDL_malloc(sizeof(SDL_WindowShaper));
    if (!result) {
        SDL_OutOfMemory();
        return NULL;
    }
    result->window = window;
    result->mode.mode = ShapeModeDefault;
    result->mode.parameters.binarizationCutoff = 1;
    result->userx = result->usery = 0;
    data = SDL_malloc(sizeof(SDL_ShapeData));
    if (!data) {
        SDL_free(result);
        SDL_OutOfMemory();
        return NULL;
    }
    result->driverdata = data;
    data->surface = NULL;
    window->shaper = result;
    resized_properly = DirectFB_ResizeWindowShape(window);
    SDL_assert(resized_properly == 0);

    return result;
}

int DirectFB_ResizeWindowShape(SDL_Window* window)
{
    SDL_ShapeData* data = window->shaper->driverdata;
    SDL_assert(data != NULL);

    if (window->x != -1000)
    {
        window->shaper->userx = window->x;
        window->shaper->usery = window->y;
    }
    SDL_SetWindowPosition(window,-1000,-1000);

    return 0;
}

int DirectFB_SetWindowShape(SDL_WindowShaper *shaper,SDL_Surface *shape,SDL_WindowShapeMode *shape_mode)
{

    if(!shaper || !shape || !shaper->driverdata)
        return -1;
    if(shape->format->Amask == 0 && SDL_SHAPEMODEALPHA(shape_mode->mode))
        return -2;
    if(shape->w != shaper->window->w || shape->h != shaper->window->h)
        return -3;

    {
        SDL_VideoDisplay *display = SDL_GetDisplayForWindow(shaper->window);
        SDL_DFB_DEVICEDATA(display->device);
        Uint32 *pixels;
        Sint32 pitch;
        Uint32 h,w;
        Uint8  *src, *bitmap;
        DFBSurfaceDescription dsc;

        SDL_ShapeData *data = shaper->driverdata;

        SDL_DFB_RELEASE(data->surface);

        dsc.flags = DSDESC_WIDTH | DSDESC_HEIGHT | DSDESC_PIXELFORMAT | DSDESC_CAPS;
        dsc.width = shape->w;
        dsc.height = shape->h;
        dsc.caps = DSCAPS_PREMULTIPLIED;
        dsc.pixelformat = DSPF_ARGB;

        SDL_DFB_CHECKERR(devdata->dfb->CreateSurface(devdata->dfb, &dsc, &data->surface));
        SDL_DFB_CALLOC(bitmap, shape->w * shape->h, 1);
        /* Assume that shaper->alphacutoff already has a value, because SDL_SetWindowShape() should have given it one. */
        SDL_CalculateShapeBitmap(shaper->mode, shape, bitmap, 1);

        src = bitmap;

        SDL_DFB_CHECK(data->surface->Lock(data->surface, DSLF_WRITE | DSLF_READ, (void **) &pixels, &pitch));

        h = shaper->window->h;
        while (h--) {
            for (w = 0; w < shaper->window->w; w++) {
                if (*src)
                    pixels[w] = 0xFFFFFFFF;
                else
                    pixels[w] = 0;
                src++;

            }
            pixels += (pitch >> 2);
        }
        SDL_DFB_CHECK(data->surface->Unlock(data->surface));
        SDL_DFB_FREE(bitmap);

        /* FIXME: Need to call this here - Big ?? */
        DirectFB_WM_RedrawLayout(SDL_GetDisplayForWindow(shaper->window)->device, shaper->window);
    }

    return 0;
error:
    return -1;
}

#endif /* SDL_VIDEO_DRIVER_DIRECTFB */
