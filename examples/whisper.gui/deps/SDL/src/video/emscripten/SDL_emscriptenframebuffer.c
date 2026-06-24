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

#ifdef SDL_VIDEO_DRIVER_EMSCRIPTEN

#include "SDL_emscriptenvideo.h"
#include "SDL_emscriptenframebuffer.h"
#include "SDL_hints.h"

#include <emscripten/threading.h>

int Emscripten_CreateWindowFramebuffer(_THIS, SDL_Window *window, Uint32 *format, void **pixels, int *pitch)
{
    SDL_Surface *surface;
    const Uint32 surface_format = SDL_PIXELFORMAT_BGR888;
    int w, h;
    int bpp;
    Uint32 Rmask, Gmask, Bmask, Amask;

    /* Free the old framebuffer surface */
    SDL_WindowData *data = (SDL_WindowData *)window->driverdata;
    surface = data->surface;
    SDL_FreeSurface(surface);

    /* Create a new one */
    SDL_PixelFormatEnumToMasks(surface_format, &bpp, &Rmask, &Gmask, &Bmask, &Amask);
    SDL_GetWindowSizeInPixels(window, &w, &h);

    surface = SDL_CreateRGBSurface(0, w, h, bpp, Rmask, Gmask, Bmask, Amask);
    if (!surface) {
        return -1;
    }

    /* Save the info and return! */
    data->surface = surface;
    *format = surface_format;
    *pixels = surface->pixels;
    *pitch = surface->pitch;
    return 0;
}

int Emscripten_UpdateWindowFramebuffer(_THIS, SDL_Window *window, const SDL_Rect *rects, int numrects)
{
    SDL_Surface *surface;

    SDL_WindowData *data = (SDL_WindowData *)window->driverdata;
    surface = data->surface;
    if (!surface) {
        return SDL_SetError("Couldn't find framebuffer surface for window");
    }

    /* Send the data to the display */

    /* *INDENT-OFF* */ /* clang-format off */
    MAIN_THREAD_EM_ASM({
        var w = $0;
        var h = $1;
        var pixels = $2;

        if (!Module['SDL2']) Module['SDL2'] = {};
        var SDL2 = Module['SDL2'];
        if (SDL2.ctxCanvas !== Module['canvas']) {
            SDL2.ctx = Module['createContext'](Module['canvas'], false, true);
            SDL2.ctxCanvas = Module['canvas'];
        }
        if (SDL2.w !== w || SDL2.h !== h || SDL2.imageCtx !== SDL2.ctx) {
            SDL2.image = SDL2.ctx.createImageData(w, h);
            SDL2.w = w;
            SDL2.h = h;
            SDL2.imageCtx = SDL2.ctx;
        }
        var data = SDL2.image.data;
        var src = pixels / 4;
        var dst = 0;
        var num;
        if (typeof CanvasPixelArray !== 'undefined' && data instanceof CanvasPixelArray) {
            // IE10/IE11: ImageData objects are backed by the deprecated CanvasPixelArray,
            // not UInt8ClampedArray. These don't have buffers, so we need to revert
            // to copying a byte at a time. We do the undefined check because modern
            // browsers do not define CanvasPixelArray anymore.
            num = data.length;
            while (dst < num) {
                var val = HEAP32[src]; // This is optimized. Instead, we could do {{{ makeGetValue('buffer', 'dst', 'i32') }}};
                data[dst  ] = val & 0xff;
                data[dst+1] = (val >> 8) & 0xff;
                data[dst+2] = (val >> 16) & 0xff;
                data[dst+3] = 0xff;
                src++;
                dst += 4;
            }
        } else {
            if (SDL2.data32Data !== data) {
                SDL2.data32 = new Int32Array(data.buffer);
                SDL2.data8 = new Uint8Array(data.buffer);
                SDL2.data32Data = data;
            }
            var data32 = SDL2.data32;
            num = data32.length;
            // logically we need to do
            //      while (dst < num) {
            //          data32[dst++] = HEAP32[src++] | 0xff000000
            //      }
            // the following code is faster though, because
            // .set() is almost free - easily 10x faster due to
            // native SDL_memcpy efficiencies, and the remaining loop
            // just stores, not load + store, so it is faster
            data32.set(HEAP32.subarray(src, src + num));
            var data8 = SDL2.data8;
            var i = 3;
            var j = i + 4*num;
            if (num % 8 == 0) {
                // unrolling gives big speedups
                while (i < j) {
                  data8[i] = 0xff;
                  i = i + 4 | 0;
                  data8[i] = 0xff;
                  i = i + 4 | 0;
                  data8[i] = 0xff;
                  i = i + 4 | 0;
                  data8[i] = 0xff;
                  i = i + 4 | 0;
                  data8[i] = 0xff;
                  i = i + 4 | 0;
                  data8[i] = 0xff;
                  i = i + 4 | 0;
                  data8[i] = 0xff;
                  i = i + 4 | 0;
                  data8[i] = 0xff;
                  i = i + 4 | 0;
                }
             } else {
                while (i < j) {
                  data8[i] = 0xff;
                  i = i + 4 | 0;
                }
            }
        }

        SDL2.ctx.putImageData(SDL2.image, 0, 0);
    }, surface->w, surface->h, surface->pixels);

    if (emscripten_has_asyncify() && SDL_GetHintBoolean(SDL_HINT_EMSCRIPTEN_ASYNCIFY, SDL_TRUE)) {
        /* give back control to browser for screen refresh */
        emscripten_sleep(0);
    }

    return 0;
}

void Emscripten_DestroyWindowFramebuffer(_THIS, SDL_Window *window)
{
    SDL_WindowData *data = (SDL_WindowData *)window->driverdata;

    SDL_FreeSurface(data->surface);
    data->surface = NULL;
}

#endif /* SDL_VIDEO_DRIVER_EMSCRIPTEN */

/* vi: set ts=4 sw=4 expandtab: */
