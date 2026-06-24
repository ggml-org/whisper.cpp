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
#include "../SDL_internal.h"

#include "SDL.h"
#include "SDL_video.h"
#include "SDL_sysvideo.h"
#include "SDL_pixels.h"
#include "SDL_surface.h"
#include "SDL_shape.h"
#include "SDL_shape_internals.h"

SDL_Window *SDL_CreateShapedWindow(const char *title, unsigned int x, unsigned int y, unsigned int w, unsigned int h, Uint32 flags)
{
    SDL_Window *result = NULL;
    result = SDL_CreateWindow(title, -1000, -1000, w, h, (flags | SDL_WINDOW_BORDERLESS) & (~SDL_WINDOW_FULLSCREEN) & (~SDL_WINDOW_RESIZABLE) /* & (~SDL_WINDOW_SHOWN) */);
    if (result) {
        if (SDL_GetVideoDevice()->shape_driver.CreateShaper == NULL) {
            SDL_DestroyWindow(result);
            return NULL;
        }
        result->shaper = SDL_GetVideoDevice()->shape_driver.CreateShaper(result);
        if (result->shaper) {
            result->shaper->userx = x;
            result->shaper->usery = y;
            result->shaper->mode.mode = ShapeModeDefault;
            result->shaper->mode.parameters.binarizationCutoff = 1;
            result->shaper->hasshape = SDL_FALSE;
            return result;
        } else {
            SDL_DestroyWindow(result);
            return NULL;
        }
    }
    return NULL;
}

SDL_bool SDL_IsShapedWindow(const SDL_Window *window)
{
    if (window == NULL) {
        return SDL_FALSE;
    }
    return (SDL_bool)(window->shaper != NULL);
}

/* REQUIRES that bitmap point to a w-by-h bitmap with ppb pixels-per-byte. */
void SDL_CalculateShapeBitmap(SDL_WindowShapeMode mode, SDL_Surface *shape, Uint8 *bitmap, Uint8 ppb)
{
    int x = 0;
    int y = 0;
    Uint8 r = 0, g = 0, b = 0, alpha = 0;
    Uint8 *pixel = NULL;
    Uint32 pixel_value = 0, mask_value = 0;
    size_t bytes_per_scanline = (size_t)(shape->w + (ppb - 1)) / ppb;
    Uint8 *bitmap_scanline;
    SDL_Color key;

    if (SDL_MUSTLOCK(shape)) {
        SDL_LockSurface(shape);
    }

    SDL_memset(bitmap, 0, shape->h * bytes_per_scanline);

    for (y = 0; y < shape->h; y++) {
        bitmap_scanline = bitmap + y * bytes_per_scanline;
        for (x = 0; x < shape->w; x++) {
            alpha = 0;
            pixel_value = 0;
            pixel = (Uint8 *)(shape->pixels) + (y * shape->pitch) + (x * shape->format->BytesPerPixel);
            switch (shape->format->BytesPerPixel) {
            case (1):
                pixel_value = *pixel;
                break;
            case (2):
                pixel_value = *(Uint16 *)pixel;
                break;
            case (3):
                pixel_value = *(Uint32 *)pixel & (~shape->format->Amask);
                break;
            case (4):
                pixel_value = *(Uint32 *)pixel;
                break;
            }
            SDL_GetRGBA(pixel_value, shape->format, &r, &g, &b, &alpha);
            switch (mode.mode) {
            case (ShapeModeDefault):
                mask_value = (alpha >= 1 ? 1 : 0);
                break;
            case (ShapeModeBinarizeAlpha):
                mask_value = (alpha >= mode.parameters.binarizationCutoff ? 1 : 0);
                break;
            case (ShapeModeReverseBinarizeAlpha):
                mask_value = (alpha <= mode.parameters.binarizationCutoff ? 1 : 0);
                break;
            case (ShapeModeColorKey):
                key = mode.parameters.colorKey;
                mask_value = ((key.r != r || key.g != g || key.b != b) ? 1 : 0);
                break;
            }
            bitmap_scanline[x / ppb] |= mask_value << (x % ppb);
        }
    }

    if (SDL_MUSTLOCK(shape)) {
        SDL_UnlockSurface(shape);
    }
}

static SDL_ShapeTree *RecursivelyCalculateShapeTree(SDL_WindowShapeMode mode, SDL_Surface *mask, SDL_Rect dimensions)
{
    int x = 0, y = 0;
    Uint8 *pixel = NULL;
    Uint32 pixel_value = 0;
    Uint8 r = 0, g = 0, b = 0, a = 0;
    SDL_bool pixel_opaque = SDL_FALSE;
    int last_opaque = -1;
    SDL_Color key;
    SDL_ShapeTree *result = (SDL_ShapeTree *)SDL_malloc(sizeof(SDL_ShapeTree));
    SDL_Rect next = { 0, 0, 0, 0 };

    if (!result) {
        SDL_OutOfMemory();
        return NULL;
    }

    for (y = dimensions.y; y < dimensions.y + dimensions.h; y++) {
        for (x = dimensions.x; x < dimensions.x + dimensions.w; x++) {
            pixel_value = 0;
            pixel = (Uint8 *)(mask->pixels) + (y * mask->pitch) + (x * mask->format->BytesPerPixel);
            switch (mask->format->BytesPerPixel) {
            case (1):
                pixel_value = *pixel;
                break;
            case (2):
                pixel_value = *(Uint16 *)pixel;
                break;
            case (3):
                pixel_value = *(Uint32 *)pixel & (~mask->format->Amask);
                break;
            case (4):
                pixel_value = *(Uint32 *)pixel;
                break;
            }
            SDL_GetRGBA(pixel_value, mask->format, &r, &g, &b, &a);
            switch (mode.mode) {
            case (ShapeModeDefault):
                pixel_opaque = (a >= 1 ? SDL_TRUE : SDL_FALSE);
                break;
            case (ShapeModeBinarizeAlpha):
                pixel_opaque = (a >= mode.parameters.binarizationCutoff ? SDL_TRUE : SDL_FALSE);
                break;
            case (ShapeModeReverseBinarizeAlpha):
                pixel_opaque = (a <= mode.parameters.binarizationCutoff ? SDL_TRUE : SDL_FALSE);
                break;
            case (ShapeModeColorKey):
                key = mode.parameters.colorKey;
                pixel_opaque = ((key.r != r || key.g != g || key.b != b) ? SDL_TRUE : SDL_FALSE);
                break;
            }
            if (last_opaque == -1) {
                last_opaque = pixel_opaque;
            }
            if (last_opaque != pixel_opaque) {
                const int halfwidth = dimensions.w / 2;
                const int halfheight = dimensions.h / 2;

                result->kind = QuadShape;

                next.x = dimensions.x;
                next.y = dimensions.y;
                next.w = halfwidth;
                next.h = halfheight;
                result->data.children.upleft = (struct SDL_ShapeTree *)RecursivelyCalculateShapeTree(mode, mask, next);

                next.x = dimensions.x + halfwidth;
                next.w = dimensions.w - halfwidth;
                result->data.children.upright = (struct SDL_ShapeTree *)RecursivelyCalculateShapeTree(mode, mask, next);

                next.x = dimensions.x;
                next.w = halfwidth;
                next.y = dimensions.y + halfheight;
                next.h = dimensions.h - halfheight;
                result->data.children.downleft = (struct SDL_ShapeTree *)RecursivelyCalculateShapeTree(mode, mask, next);

                next.x = dimensions.x + halfwidth;
                next.w = dimensions.w - halfwidth;
                result->data.children.downright = (struct SDL_ShapeTree *)RecursivelyCalculateShapeTree(mode, mask, next);

                return result;
            }
        }
    }

    /* If we never recursed, all the pixels in this quadrant have the same "value". */
    result->kind = (last_opaque == SDL_TRUE ? OpaqueShape : TransparentShape);
    result->data.shape = dimensions;
    return result;
}

SDL_ShapeTree *SDL_CalculateShapeTree(SDL_WindowShapeMode mode, SDL_Surface *shape)
{
    SDL_Rect dimensions;
    SDL_ShapeTree *result = NULL;

    dimensions.x = 0;
    dimensions.y = 0;
    dimensions.w = shape->w;
    dimensions.h = shape->h;

    if (SDL_MUSTLOCK(shape)) {
        SDL_LockSurface(shape);
    }
    result = RecursivelyCalculateShapeTree(mode, shape, dimensions);
    if (SDL_MUSTLOCK(shape)) {
        SDL_UnlockSurface(shape);
    }
    return result;
}

void SDL_TraverseShapeTree(SDL_ShapeTree *tree, SDL_TraversalFunction function, void *closure)
{
    SDL_assert(tree != NULL);
    if (tree->kind == QuadShape) {
        SDL_TraverseShapeTree((SDL_ShapeTree *)tree->data.children.upleft, function, closure);
        SDL_TraverseShapeTree((SDL_ShapeTree *)tree->data.children.upright, function, closure);
        SDL_TraverseShapeTree((SDL_ShapeTree *)tree->data.children.downleft, function, closure);
        SDL_TraverseShapeTree((SDL_ShapeTree *)tree->data.children.downright, function, closure);
    } else {
        function(tree, closure);
    }
}

void SDL_FreeShapeTree(SDL_ShapeTree **shape_tree)
{
    if ((*shape_tree)->kind == QuadShape) {
        SDL_FreeShapeTree((SDL_ShapeTree **)(char *)&(*shape_tree)->data.children.upleft);
        SDL_FreeShapeTree((SDL_ShapeTree **)(char *)&(*shape_tree)->data.children.upright);
        SDL_FreeShapeTree((SDL_ShapeTree **)(char *)&(*shape_tree)->data.children.downleft);
        SDL_FreeShapeTree((SDL_ShapeTree **)(char *)&(*shape_tree)->data.children.downright);
    }
    SDL_free(*shape_tree);
    *shape_tree = NULL;
}

int SDL_SetWindowShape(SDL_Window *window, SDL_Surface *shape, SDL_WindowShapeMode *shape_mode)
{
    SDL_VideoDevice *_this = SDL_GetVideoDevice();
    int result;

    if (!window || !SDL_IsShapedWindow(window)) {
        /* The window given was not a shapeable window. */
        return SDL_NONSHAPEABLE_WINDOW;
    }
    if (!shape) {
        /* Invalid shape argument. */
        return SDL_INVALID_SHAPE_ARGUMENT;
    }

    if (shape_mode) {
        window->shaper->mode = *shape_mode;
    }
    result = _this->shape_driver.SetWindowShape(window->shaper, shape, shape_mode);
    window->shaper->hasshape = SDL_TRUE;
    if (window->shaper->userx != 0 && window->shaper->usery != 0) {
        int x = window->shaper->userx;
        int y = window->shaper->usery;

        if (SDL_WINDOWPOS_ISUNDEFINED(x) || SDL_WINDOWPOS_ISUNDEFINED(y) ||
            SDL_WINDOWPOS_ISCENTERED(x) || SDL_WINDOWPOS_ISCENTERED(y)) {
            int displayIndex;
            SDL_Rect bounds;

            if (SDL_WINDOWPOS_ISUNDEFINED(x) || SDL_WINDOWPOS_ISCENTERED(x)) {
                displayIndex = (x & 0xFFFF);
                if (displayIndex >= _this->num_displays) {
                    displayIndex = 0;
                }
            } else {
                displayIndex = (y & 0xFFFF);
                if (displayIndex >= _this->num_displays) {
                    displayIndex = 0;
                }
            }

            SDL_GetDisplayBounds(displayIndex, &bounds);
            if (SDL_WINDOWPOS_ISUNDEFINED(x) || SDL_WINDOWPOS_ISCENTERED(x)) {
                window->x = bounds.x + (bounds.w - window->w) / 2;
            }
            if (SDL_WINDOWPOS_ISUNDEFINED(y) || SDL_WINDOWPOS_ISCENTERED(y)) {
                window->y = bounds.y + (bounds.h - window->h) / 2;
            }
        }
        SDL_SetWindowPosition(window, x, y);
        window->shaper->userx = 0;
        window->shaper->usery = 0;
    }
    return result;
}

static SDL_bool SDL_WindowHasAShape(SDL_Window *window)
{
    if (!window || !SDL_IsShapedWindow(window)) {
        return SDL_FALSE;
    }
    return window->shaper->hasshape;
}

int SDL_GetShapedWindowMode(SDL_Window *window, SDL_WindowShapeMode *shape_mode)
{
    if (window && SDL_IsShapedWindow(window)) {
        if (!shape_mode) {
            if (SDL_WindowHasAShape(window)) {
                return 0; /* The window given has a shape. */
            } else {
                return SDL_WINDOW_LACKS_SHAPE; /* The window given is shapeable but lacks a shape. */
            }
        } else {
            *shape_mode = window->shaper->mode;
            return 0;
        }
    }
    return SDL_NONSHAPEABLE_WINDOW; /* The window given is not a valid shapeable window. */
}
