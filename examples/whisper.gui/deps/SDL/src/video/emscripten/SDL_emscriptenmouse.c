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

#include <emscripten/emscripten.h>
#include <emscripten/html5.h>
#include <emscripten/threading.h>

#include "SDL_emscriptenmouse.h"
#include "SDL_emscriptenvideo.h"

#include "../../events/SDL_mouse_c.h"

/* older Emscriptens don't have this, but we need to for wasm64 compatibility. */
#ifndef MAIN_THREAD_EM_ASM_PTR
    #ifdef __wasm64__
        #error You need to upgrade your Emscripten compiler to support wasm64
    #else
        #define MAIN_THREAD_EM_ASM_PTR MAIN_THREAD_EM_ASM_INT
    #endif
#endif

static SDL_Cursor *Emscripten_CreateCursorFromString(const char *cursor_str, SDL_bool is_custom)
{
    SDL_Cursor *cursor;
    Emscripten_CursorData *curdata;

    cursor = SDL_calloc(1, sizeof(SDL_Cursor));
    if (cursor) {
        curdata = (Emscripten_CursorData *)SDL_calloc(1, sizeof(*curdata));
        if (!curdata) {
            SDL_OutOfMemory();
            SDL_free(cursor);
            return NULL;
        }

        curdata->system_cursor = cursor_str;
        curdata->is_custom = is_custom;
        cursor->driverdata = curdata;
    } else {
        SDL_OutOfMemory();
    }

    return cursor;
}

static SDL_Cursor *Emscripten_CreateDefaultCursor()
{
    return Emscripten_CreateCursorFromString("default", SDL_FALSE);
}

EM_JS_DEPS(sdlmouse, "$stringToUTF8,$UTF8ToString");

static SDL_Cursor *Emscripten_CreateCursor(SDL_Surface *surface, int hot_x, int hot_y)
{
    const char *cursor_url = NULL;
    SDL_Surface *conv_surf;

    conv_surf = SDL_ConvertSurfaceFormat(surface, SDL_PIXELFORMAT_ABGR8888, 0);

    if (!conv_surf) {
        return NULL;
    }

    /* *INDENT-OFF* */ /* clang-format off */
    cursor_url = (const char *)MAIN_THREAD_EM_ASM_PTR({
        var w = $0;
        var h = $1;
        var hot_x = $2;
        var hot_y = $3;
        var pixels = $4;

        var canvas = document.createElement("canvas");
        canvas.width = w;
        canvas.height = h;

        var ctx = canvas.getContext("2d");

        var image = ctx.createImageData(w, h);
        var data = image.data;
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
                data[dst+3] = (val >> 24) & 0xff;
                src++;
                dst += 4;
            }
        } else {
            var data32 = new Int32Array(data.buffer);
            num = data32.length;
            data32.set(HEAP32.subarray(src, src + num));
        }

        ctx.putImageData(image, 0, 0);
        var url = hot_x === 0 && hot_y === 0
            ? "url(" + canvas.toDataURL() + "), auto"
            : "url(" + canvas.toDataURL() + ") " + hot_x + " " + hot_y + ", auto";

        var urlBuf = _malloc(url.length + 1);
        stringToUTF8(url, urlBuf, url.length + 1);

        return urlBuf;
    }, surface->w, surface->h, hot_x, hot_y, conv_surf->pixels);
    /* *INDENT-ON* */ /* clang-format on */

    SDL_FreeSurface(conv_surf);

    return Emscripten_CreateCursorFromString(cursor_url, SDL_TRUE);
}

static SDL_Cursor *Emscripten_CreateSystemCursor(SDL_SystemCursor id)
{
    const char *cursor_name = SDL_GetCSSCursorName(id, NULL);

    return Emscripten_CreateCursorFromString(cursor_name, SDL_FALSE);
}

static void Emscripten_FreeCursor(SDL_Cursor *cursor)
{
    Emscripten_CursorData *curdata;
    if (cursor) {
        curdata = (Emscripten_CursorData *)cursor->driverdata;

        if (curdata) {
            if (curdata->is_custom) {
                SDL_free((char *)curdata->system_cursor);
            }
            SDL_free(cursor->driverdata);
        }

        SDL_free(cursor);
    }
}

static int Emscripten_ShowCursor(SDL_Cursor *cursor)
{
    Emscripten_CursorData *curdata;
    if (SDL_GetMouseFocus() != NULL) {
        if (cursor && cursor->driverdata) {
            curdata = (Emscripten_CursorData *)cursor->driverdata;

            if (curdata->system_cursor) {
                /* *INDENT-OFF* */ /* clang-format off */
                MAIN_THREAD_EM_ASM({
                    if (Module['canvas']) {
                        Module['canvas'].style['cursor'] = UTF8ToString($0);
                    }
                }, curdata->system_cursor);
                /* *INDENT-ON* */ /* clang-format on */
            }
        } else {
            /* *INDENT-OFF* */ /* clang-format off */
            MAIN_THREAD_EM_ASM(
                if (Module['canvas']) {
                    Module['canvas'].style['cursor'] = 'none';
                }
            );
            /* *INDENT-ON* */ /* clang-format on */
        }
    }
    return 0;
}

static void Emscripten_WarpMouse(SDL_Window *window, int x, int y)
{
    SDL_Unsupported();
}

static int Emscripten_SetRelativeMouseMode(SDL_bool enabled)
{
    SDL_Window *window;
    SDL_WindowData *window_data;

    /* TODO: pointer lock isn't actually enabled yet */
    if (enabled) {
        window = SDL_GetMouseFocus();
        if (!window) {
            return -1;
        }

        window_data = (SDL_WindowData *)window->driverdata;

        if (emscripten_request_pointerlock(window_data->canvas_id, 1) >= EMSCRIPTEN_RESULT_SUCCESS) {
            return 0;
        }
    } else {
        if (emscripten_exit_pointerlock() >= EMSCRIPTEN_RESULT_SUCCESS) {
            return 0;
        }
    }
    return -1;
}

void Emscripten_InitMouse(void)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    mouse->CreateCursor = Emscripten_CreateCursor;
    mouse->ShowCursor = Emscripten_ShowCursor;
    mouse->FreeCursor = Emscripten_FreeCursor;
    mouse->WarpMouse = Emscripten_WarpMouse;
    mouse->CreateSystemCursor = Emscripten_CreateSystemCursor;
    mouse->SetRelativeMouseMode = Emscripten_SetRelativeMouseMode;

    SDL_SetDefaultCursor(Emscripten_CreateDefaultCursor());
}

void Emscripten_FiniMouse(void)
{
}

#endif /* SDL_VIDEO_DRIVER_EMSCRIPTEN */

/* vi: set ts=4 sw=4 expandtab: */
