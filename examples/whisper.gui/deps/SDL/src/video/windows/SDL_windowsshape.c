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

#if defined(SDL_VIDEO_DRIVER_WINDOWS) && !defined(__XBOXONE__) && !defined(__XBOXSERIES__)

#include "SDL_windowsshape.h"
#include "SDL_windowsvideo.h"

SDL_WindowShaper *Win32_CreateShaper(SDL_Window *window)
{
    int resized_properly;
    SDL_WindowShaper *result = (SDL_WindowShaper *)SDL_malloc(sizeof(SDL_WindowShaper));
    if (!result) {
        SDL_OutOfMemory();
        return NULL;
    }
    result->window = window;
    result->mode.mode = ShapeModeDefault;
    result->mode.parameters.binarizationCutoff = 1;
    result->userx = result->usery = 0;
    result->hasshape = SDL_FALSE;
    result->driverdata = (SDL_ShapeData *)SDL_calloc(1, sizeof(SDL_ShapeData));
    if (!result->driverdata) {
        SDL_free(result);
        SDL_OutOfMemory();
        return NULL;
    }
    window->shaper = result;
    /* Put some driver-data here. */
    resized_properly = Win32_ResizeWindowShape(window);
    if (resized_properly != 0) {
        SDL_free(result->driverdata);
        SDL_free(result);
        window->shaper = NULL;
        return NULL;
    }

    return result;
}

static void CombineRectRegions(SDL_ShapeTree *node, void *closure)
{
    HRGN mask_region = *((HRGN *)closure), temp_region = NULL;
    if (node->kind == OpaqueShape) {
        /* Win32 API regions exclude their outline, so we widen the region by one pixel in each direction to include the real outline. */
        temp_region = CreateRectRgn(node->data.shape.x, node->data.shape.y, node->data.shape.x + node->data.shape.w + 1, node->data.shape.y + node->data.shape.h + 1);
        if (mask_region != NULL) {
            CombineRgn(mask_region, mask_region, temp_region, RGN_OR);
            DeleteObject(temp_region);
        } else {
            *((HRGN *)closure) = temp_region;
        }
    }
}

int Win32_SetWindowShape(SDL_WindowShaper *shaper, SDL_Surface *shape, SDL_WindowShapeMode *shape_mode)
{
    SDL_ShapeData *data;
    HRGN mask_region = NULL;

    if ((!shaper) ||
        (!shape) ||
        ((shape->format->Amask == 0) && (shape_mode->mode != ShapeModeColorKey)) ||
        (shape->w != shaper->window->w) ||
        (shape->h != shaper->window->h)) {
        return SDL_INVALID_SHAPE_ARGUMENT;
    }

    data = (SDL_ShapeData *)shaper->driverdata;
    if (data->mask_tree) {
        SDL_FreeShapeTree(&data->mask_tree);
    }
    data->mask_tree = SDL_CalculateShapeTree(*shape_mode, shape);

    SDL_TraverseShapeTree(data->mask_tree, &CombineRectRegions, &mask_region);
    SDL_assert(mask_region != NULL);

    SetWindowRgn(((SDL_WindowData *)(shaper->window->driverdata))->hwnd, mask_region, TRUE);

    return 0;
}

int Win32_ResizeWindowShape(SDL_Window *window)
{
    SDL_ShapeData *data;

    if (!window) {
        return -1;
    }
    data = (SDL_ShapeData *)window->shaper->driverdata;
    if (!data) {
        return -1;
    }

    if (data->mask_tree) {
        SDL_FreeShapeTree(&data->mask_tree);
    }
    if (window->shaper->hasshape == SDL_TRUE) {
        window->shaper->userx = window->x;
        window->shaper->usery = window->y;
        SDL_SetWindowPosition(window, -1000, -1000);
    }

    return 0;
}

#endif /* SDL_VIDEO_DRIVER_WINDOWS */
