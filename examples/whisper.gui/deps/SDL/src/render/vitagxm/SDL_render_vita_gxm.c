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

#if SDL_VIDEO_RENDER_VITA_GXM

#include "SDL_hints.h"
#include "../SDL_sysrender.h"
#include "SDL_log.h"

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>

#include "SDL_render_vita_gxm_types.h"
#include "SDL_render_vita_gxm_tools.h"
#include "SDL_render_vita_gxm_memory.h"

#include <psp2/common_dialog.h>

/* #define DEBUG_RAZOR */

#ifdef DEBUG_RAZOR
#include <psp2/sysmodule.h>
#endif

static SDL_Renderer *VITA_GXM_CreateRenderer(SDL_Window *window, Uint32 flags);

static void VITA_GXM_WindowEvent(SDL_Renderer *renderer, const SDL_WindowEvent *event);

static SDL_bool VITA_GXM_SupportsBlendMode(SDL_Renderer *renderer, SDL_BlendMode blendMode);

static int VITA_GXM_CreateTexture(SDL_Renderer *renderer, SDL_Texture *texture);

static int VITA_GXM_UpdateTexture(SDL_Renderer *renderer, SDL_Texture *texture,
                                  const SDL_Rect *rect, const void *pixels, int pitch);

static int VITA_GXM_UpdateTextureYUV(SDL_Renderer *renderer, SDL_Texture *texture,
                                     const SDL_Rect *rect,
                                     const Uint8 *Yplane, int Ypitch,
                                     const Uint8 *Uplane, int Upitch,
                                     const Uint8 *Vplane, int Vpitch);

static int VITA_GXM_UpdateTextureNV(SDL_Renderer *renderer, SDL_Texture *texture,
                                    const SDL_Rect *rect,
                                    const Uint8 *Yplane, int Ypitch,
                                    const Uint8 *UVplane, int UVpitch);

static int VITA_GXM_LockTexture(SDL_Renderer *renderer, SDL_Texture *texture,
                                const SDL_Rect *rect, void **pixels, int *pitch);

static void VITA_GXM_UnlockTexture(SDL_Renderer *renderer,
                                   SDL_Texture *texture);

static void VITA_GXM_SetTextureScaleMode(SDL_Renderer *renderer, SDL_Texture *texture, SDL_ScaleMode scaleMode);

static int VITA_GXM_SetRenderTarget(SDL_Renderer *renderer,
                                    SDL_Texture *texture);

static int VITA_GXM_QueueSetViewport(SDL_Renderer *renderer, SDL_RenderCommand *cmd);

static int VITA_GXM_QueueSetDrawColor(SDL_Renderer *renderer, SDL_RenderCommand *cmd);

static int VITA_GXM_QueueDrawPoints(SDL_Renderer *renderer, SDL_RenderCommand *cmd, const SDL_FPoint *points, int count);
static int VITA_GXM_QueueDrawLines(SDL_Renderer *renderer, SDL_RenderCommand *cmd, const SDL_FPoint *points, int count);

static int VITA_GXM_QueueGeometry(SDL_Renderer *renderer, SDL_RenderCommand *cmd, SDL_Texture *texture,
                                  const float *xy, int xy_stride, const SDL_Color *color, int color_stride, const float *uv, int uv_stride,
                                  int num_vertices, const void *indices, int num_indices, int size_indices,
                                  float scale_x, float scale_y);

static int VITA_GXM_RenderClear(SDL_Renderer *renderer, SDL_RenderCommand *cmd);

static int VITA_GXM_RunCommandQueue(SDL_Renderer *renderer, SDL_RenderCommand *cmd, void *vertices, size_t vertsize);

static int VITA_GXM_RenderReadPixels(SDL_Renderer *renderer, const SDL_Rect *rect,
                                     Uint32 pixel_format, void *pixels, int pitch);

static int VITA_GXM_RenderPresent(SDL_Renderer *renderer);
static void VITA_GXM_DestroyTexture(SDL_Renderer *renderer, SDL_Texture *texture);
static void VITA_GXM_DestroyRenderer(SDL_Renderer *renderer);

SDL_RenderDriver VITA_GXM_RenderDriver = {
    .CreateRenderer = VITA_GXM_CreateRenderer,
    .info = {
        .name = "VITA gxm",
        .flags = SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC | SDL_RENDERER_TARGETTEXTURE,
        .num_texture_formats = 8,
        .texture_formats = {
            [0] = SDL_PIXELFORMAT_ABGR8888,
            [1] = SDL_PIXELFORMAT_ARGB8888,
            [2] = SDL_PIXELFORMAT_RGB565,
            [3] = SDL_PIXELFORMAT_BGR565,
            [4] = SDL_PIXELFORMAT_YV12,
            [5] = SDL_PIXELFORMAT_IYUV,
            [6] = SDL_PIXELFORMAT_NV12,
            [7] = SDL_PIXELFORMAT_NV21,
        },
        .max_texture_width = 4096,
        .max_texture_height = 4096,
    }
};

static int PixelFormatToVITAFMT(Uint32 format)
{
    switch (format) {
    case SDL_PIXELFORMAT_ARGB8888:
        return SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ARGB;
    case SDL_PIXELFORMAT_RGB888:
        return SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ARGB;
    case SDL_PIXELFORMAT_BGR888:
        return SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ABGR;
    case SDL_PIXELFORMAT_ABGR8888:
        return SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ABGR;
    case SDL_PIXELFORMAT_RGB565:
        return SCE_GXM_TEXTURE_FORMAT_U5U6U5_RGB;
    case SDL_PIXELFORMAT_BGR565:
        return SCE_GXM_TEXTURE_FORMAT_U5U6U5_BGR;
    case SDL_PIXELFORMAT_YV12:
        return SCE_GXM_TEXTURE_FORMAT_YVU420P3_CSC0;
    case SDL_PIXELFORMAT_IYUV:
        return SCE_GXM_TEXTURE_FORMAT_YUV420P3_CSC0;
    // should be the other way around. looks like SCE bug.
    case SDL_PIXELFORMAT_NV12:
        return SCE_GXM_TEXTURE_FORMAT_YVU420P2_CSC0;
    case SDL_PIXELFORMAT_NV21:
        return SCE_GXM_TEXTURE_FORMAT_YUV420P2_CSC0;
    default:
        return SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ABGR;
    }
}

void StartDrawing(SDL_Renderer *renderer)
{
    VITA_GXM_RenderData *data = (VITA_GXM_RenderData *)renderer->driverdata;
    if (data->drawing) {
        return;
    }

    data->drawstate.texture = NULL;
    data->drawstate.vertex_program = NULL;
    data->drawstate.fragment_program = NULL;
    data->drawstate.last_command = -1;
    data->drawstate.viewport_dirty = SDL_TRUE;

    // reset blend mode
    //    data->currentBlendMode = SDL_BLENDMODE_BLEND;
    //    fragment_programs *in = &data->blendFragmentPrograms.blend_mode_blend;
    //    data->colorFragmentProgram = in->color;
    //    data->textureFragmentProgram = in->texture;

    if (!renderer->target) {
        sceGxmBeginScene(
            data->gxm_context,
            0,
            data->renderTarget,
            NULL,
            NULL,
            data->displayBufferSync[data->backBufferIndex],
            &data->displaySurface[data->backBufferIndex],
            &data->depthSurface);
    } else {
        VITA_GXM_TextureData *vita_texture = (VITA_GXM_TextureData *)renderer->target->driverdata;

        sceGxmBeginScene(
            data->gxm_context,
            0,
            vita_texture->tex->gxm_rendertarget,
            NULL,
            NULL,
            NULL,
            &vita_texture->tex->gxm_colorsurface,
            &vita_texture->tex->gxm_depthstencil);
    }

    //    unset_clip_rectangle(data);

    data->drawing = SDL_TRUE;
}

static int VITA_GXM_SetVSync(SDL_Renderer *renderer, const int vsync)
{
    VITA_GXM_RenderData *data = renderer->driverdata;
    if (vsync) {
        data->displayData.wait_vblank = SDL_TRUE;
        renderer->info.flags |= SDL_RENDERER_PRESENTVSYNC;
    } else {
        data->displayData.wait_vblank = SDL_FALSE;
        renderer->info.flags &= ~SDL_RENDERER_PRESENTVSYNC;
    }
    return 0;
}

SDL_Renderer *VITA_GXM_CreateRenderer(SDL_Window *window, Uint32 flags)
{
    SDL_Renderer *renderer;
    VITA_GXM_RenderData *data;

    renderer = (SDL_Renderer *)SDL_calloc(1, sizeof(*renderer));
    if (!renderer) {
        SDL_OutOfMemory();
        return NULL;
    }

    data = (VITA_GXM_RenderData *)SDL_calloc(1, sizeof(VITA_GXM_RenderData));
    if (!data) {
        SDL_free(renderer);
        SDL_OutOfMemory();
        return NULL;
    }

    renderer->WindowEvent = VITA_GXM_WindowEvent;
    renderer->SupportsBlendMode = VITA_GXM_SupportsBlendMode;
    renderer->CreateTexture = VITA_GXM_CreateTexture;
    renderer->UpdateTexture = VITA_GXM_UpdateTexture;
#if SDL_HAVE_YUV
    renderer->UpdateTextureYUV = VITA_GXM_UpdateTextureYUV;
    renderer->UpdateTextureNV = VITA_GXM_UpdateTextureNV;
#endif
    renderer->LockTexture = VITA_GXM_LockTexture;
    renderer->UnlockTexture = VITA_GXM_UnlockTexture;
    renderer->SetTextureScaleMode = VITA_GXM_SetTextureScaleMode;
    renderer->SetRenderTarget = VITA_GXM_SetRenderTarget;
    renderer->QueueSetViewport = VITA_GXM_QueueSetViewport;
    renderer->QueueSetDrawColor = VITA_GXM_QueueSetDrawColor;
    renderer->QueueDrawPoints = VITA_GXM_QueueDrawPoints;
    renderer->QueueDrawLines = VITA_GXM_QueueDrawLines;
    renderer->QueueGeometry = VITA_GXM_QueueGeometry;
    renderer->RunCommandQueue = VITA_GXM_RunCommandQueue;
    renderer->RenderReadPixels = VITA_GXM_RenderReadPixels;
    renderer->RenderPresent = VITA_GXM_RenderPresent;
    renderer->DestroyTexture = VITA_GXM_DestroyTexture;
    renderer->DestroyRenderer = VITA_GXM_DestroyRenderer;
    renderer->SetVSync = VITA_GXM_SetVSync;

    renderer->info = VITA_GXM_RenderDriver.info;
    renderer->info.flags = (SDL_RENDERER_ACCELERATED | SDL_RENDERER_TARGETTEXTURE);
    renderer->driverdata = data;
    renderer->window = window;

    data->initialized = SDL_TRUE;

    if (flags & SDL_RENDERER_PRESENTVSYNC) {
        data->displayData.wait_vblank = SDL_TRUE;
    } else {
        data->displayData.wait_vblank = SDL_FALSE;
    }

#ifdef DEBUG_RAZOR
    sceSysmoduleLoadModule(SCE_SYSMODULE_RAZOR_HUD);
    sceSysmoduleLoadModule(SCE_SYSMODULE_RAZOR_CAPTURE);
#endif

    if (gxm_init(renderer) != 0) {
        SDL_free(data);
        SDL_free(renderer);
        return NULL;
    }

    return renderer;
}

static void VITA_GXM_WindowEvent(SDL_Renderer *renderer, const SDL_WindowEvent *event)
{
}

static SDL_bool VITA_GXM_SupportsBlendMode(SDL_Renderer *renderer, SDL_BlendMode blendMode)
{
    // only for custom modes. we build all modes on init, so no custom modes, sorry
    return SDL_FALSE;
}

static int VITA_GXM_CreateTexture(SDL_Renderer *renderer, SDL_Texture *texture)
{
    VITA_GXM_RenderData *data = (VITA_GXM_RenderData *)renderer->driverdata;
    VITA_GXM_TextureData *vita_texture = (VITA_GXM_TextureData *)SDL_calloc(1, sizeof(VITA_GXM_TextureData));

    if (!vita_texture) {
        return SDL_OutOfMemory();
    }

    vita_texture->tex = create_gxm_texture(
        data,
        texture->w,
        texture->h,
        PixelFormatToVITAFMT(texture->format),
        (texture->access == SDL_TEXTUREACCESS_TARGET),
        &(vita_texture->w),
        &(vita_texture->h),
        &(vita_texture->pitch),
        &(vita_texture->wscale));

    if (!vita_texture->tex) {
        SDL_free(vita_texture);
        return SDL_OutOfMemory();
    }

    texture->driverdata = vita_texture;

    VITA_GXM_SetTextureScaleMode(renderer, texture, texture->scaleMode);

#if SDL_HAVE_YUV
    vita_texture->yuv = ((texture->format == SDL_PIXELFORMAT_IYUV) || (texture->format == SDL_PIXELFORMAT_YV12));
    vita_texture->nv12 = ((texture->format == SDL_PIXELFORMAT_NV12) || (texture->format == SDL_PIXELFORMAT_NV21));
#endif

    return 0;
}

static void VITA_GXM_SetYUVProfile(SDL_Renderer *renderer, SDL_Texture *texture)
{
    VITA_GXM_RenderData *data = (VITA_GXM_RenderData *)renderer->driverdata;
    int ret = 0;
    switch (SDL_GetYUVConversionModeForResolution(texture->w, texture->h)) {
    case SDL_YUV_CONVERSION_BT601:
        ret = sceGxmSetYuvProfile(data->gxm_context, 0, SCE_GXM_YUV_PROFILE_BT601_STANDARD);
        break;
    case SDL_YUV_CONVERSION_BT709:
        ret = sceGxmSetYuvProfile(data->gxm_context, 0, SCE_GXM_YUV_PROFILE_BT709_STANDARD);
        break;
    case SDL_YUV_CONVERSION_JPEG:
    default:
        SDL_LogError(SDL_LOG_CATEGORY_RENDER, "Unsupported YUV profile: %d\n", SDL_GetYUVConversionModeForResolution(texture->w, texture->h));
        break;
    }

    if (ret < 0) {
        SDL_LogError(SDL_LOG_CATEGORY_RENDER, "Setting YUV profile failed: %x\n", ret);
    }
}

static int VITA_GXM_UpdateTexture(SDL_Renderer *renderer, SDL_Texture *texture,
                                  const SDL_Rect *rect, const void *pixels, int pitch)
{
    VITA_GXM_TextureData *vita_texture = (VITA_GXM_TextureData *)texture->driverdata;
    Uint8 *dst;
    int row, length, dpitch;

#if SDL_HAVE_YUV
    if (vita_texture->yuv || vita_texture->nv12) {
        VITA_GXM_SetYUVProfile(renderer, texture);
    }
#endif

    VITA_GXM_LockTexture(renderer, texture, rect, (void **)&dst, &dpitch);
    length = rect->w * SDL_BYTESPERPIXEL(texture->format);
    if (length == pitch && length == dpitch) {
        SDL_memcpy(dst, pixels, length * rect->h);
        pixels += pitch * rect->h;
    } else {
        for (row = 0; row < rect->h; ++row) {
            SDL_memcpy(dst, pixels, length);
            pixels += pitch;
            dst += dpitch;
        }
    }

#if SDL_HAVE_YUV
    if (vita_texture->yuv) {
        void *Udst;
        void *Vdst;
        int uv_pitch = (dpitch + 1) / 2;
        int uv_src_pitch = (pitch + 1) / 2;
        SDL_Rect UVrect = { rect->x / 2, rect->y / 2, (rect->w + 1) / 2, (rect->h + 1) / 2 };

        // skip Y plane
        Uint8 *Dpixels = gxm_texture_get_datap(vita_texture->tex) + (vita_texture->pitch * vita_texture->h);

        Udst = Dpixels + (UVrect.y * uv_pitch) + UVrect.x;
        Vdst = Dpixels + (uv_pitch * ((vita_texture->h + 1) / 2)) + (UVrect.y * uv_pitch) + UVrect.x;

        length = UVrect.w;

        // U plane
        if (length == uv_src_pitch && length == uv_pitch) {
            SDL_memcpy(Udst, pixels, length * UVrect.h);
            pixels += uv_src_pitch * UVrect.h;
        } else {
            for (row = 0; row < UVrect.h; ++row) {
                SDL_memcpy(Udst, pixels, length);
                pixels += uv_src_pitch;
                Udst += uv_pitch;
            }
        }

        // V plane
        if (length == uv_src_pitch && length == uv_pitch) {
            SDL_memcpy(Vdst, pixels, length * UVrect.h);
        } else {
            for (row = 0; row < UVrect.h; ++row) {
                SDL_memcpy(Vdst, pixels, length);
                pixels += uv_src_pitch;
                Vdst += uv_pitch;
            }
        }

    } else if (vita_texture->nv12) {
        void *UVdst;
        int uv_pitch = 2 * ((dpitch + 1) / 2);
        int uv_src_pitch = 2 * ((pitch + 1) / 2);
        SDL_Rect UVrect = { rect->x / 2, rect->y / 2, (rect->w + 1) / 2, (rect->h + 1) / 2 };

        // skip Y plane
        void *Dpixels = (void *)((Uint8 *)gxm_texture_get_datap(vita_texture->tex) + (vita_texture->pitch * vita_texture->h));
        UVdst = Dpixels + (UVrect.y * uv_pitch) + UVrect.x;

        length = UVrect.w * 2;

        // UV plane
        if (length == uv_src_pitch && length == uv_pitch) {
            SDL_memcpy(UVdst, pixels, length * UVrect.h);
        } else {
            for (row = 0; row < UVrect.h; ++row) {
                SDL_memcpy(UVdst, pixels, length);
                pixels += uv_src_pitch;
                UVdst += uv_pitch;
            }
        }
    }
#endif

    return 0;
}

#if SDL_HAVE_YUV
static int VITA_GXM_UpdateTextureYUV(SDL_Renderer *renderer, SDL_Texture *texture,
                                     const SDL_Rect *rect,
                                     const Uint8 *Yplane, int Ypitch,
                                     const Uint8 *Uplane, int Upitch,
                                     const Uint8 *Vplane, int Vpitch)
{
    Uint8 *dst;
    int row, length, dpitch;
    SDL_Rect UVrect = { rect->x / 2, rect->y / 2, (rect->w + 1) / 2, (rect->h + 1) / 2 };

    VITA_GXM_SetYUVProfile(renderer, texture);

    // copy Y plane
    // obtain pixels via locking so that texture is flushed
    VITA_GXM_LockTexture(renderer, texture, rect, (void **)&dst, &dpitch);

    length = rect->w;

    if (length == Ypitch && length == dpitch) {
        SDL_memcpy(dst, Yplane, length * rect->h);
    } else {
        for (row = 0; row < rect->h; ++row) {
            SDL_memcpy(dst, Yplane, length);
            Yplane += Ypitch;
            dst += dpitch;
        }
    }

    // U/V planes
    {
        void *Udst;
        void *Vdst;
        VITA_GXM_TextureData *vita_texture = (VITA_GXM_TextureData *)texture->driverdata;
        int uv_pitch = (dpitch + 1) / 2;

        // skip Y plane
        void *pixels = (void *)((Uint8 *)gxm_texture_get_datap(vita_texture->tex) + (vita_texture->pitch * vita_texture->h));

        if (texture->format == SDL_PIXELFORMAT_YV12) { // YVU
            Vdst = pixels + (UVrect.y * uv_pitch) + UVrect.x;
            Udst = pixels + (uv_pitch * ((vita_texture->h + 1) / 2)) + (UVrect.y * uv_pitch) + UVrect.x;
        } else { // YUV
            Udst = pixels + (UVrect.y * uv_pitch) + UVrect.x;
            Vdst = pixels + (uv_pitch * ((vita_texture->h + 1) / 2)) + (UVrect.y * uv_pitch) + UVrect.x;
        }

        length = UVrect.w;

        // U plane
        if (length == Upitch && length == uv_pitch) {
            SDL_memcpy(Udst, Uplane, length * UVrect.h);
        } else {
            for (row = 0; row < UVrect.h; ++row) {
                SDL_memcpy(Udst, Uplane, length);
                Uplane += Upitch;
                Udst += uv_pitch;
            }
        }

        // V plane
        if (length == Vpitch && length == uv_pitch) {
            SDL_memcpy(Vdst, Vplane, length * UVrect.h);
        } else {
            for (row = 0; row < UVrect.h; ++row) {
                SDL_memcpy(Vdst, Vplane, length);
                Vplane += Vpitch;
                Vdst += uv_pitch;
            }
        }
    }

    return 0;
}

static int VITA_GXM_UpdateTextureNV(SDL_Renderer *renderer, SDL_Texture *texture,
                                    const SDL_Rect *rect,
                                    const Uint8 *Yplane, int Ypitch,
                                    const Uint8 *UVplane, int UVpitch)
{

    Uint8 *dst;
    int row, length, dpitch;
    SDL_Rect UVrect = { rect->x / 2, rect->y / 2, (rect->w + 1) / 2, (rect->h + 1) / 2 };

    VITA_GXM_SetYUVProfile(renderer, texture);

    // copy Y plane
    VITA_GXM_LockTexture(renderer, texture, rect, (void **)&dst, &dpitch);

    length = rect->w * SDL_BYTESPERPIXEL(texture->format);

    if (length == Ypitch && length == dpitch) {
        SDL_memcpy(dst, Yplane, length * rect->h);
    } else {
        for (row = 0; row < rect->h; ++row) {
            SDL_memcpy(dst, Yplane, length);
            Yplane += Ypitch;
            dst += dpitch;
        }
    }

    // UV plane
    {
        void *UVdst;
        VITA_GXM_TextureData *vita_texture = (VITA_GXM_TextureData *)texture->driverdata;
        int uv_pitch = 2 * ((dpitch + 1) / 2);

        // skip Y plane
        void *pixels = (void *)((Uint8 *)gxm_texture_get_datap(vita_texture->tex) + (vita_texture->pitch * vita_texture->h));

        UVdst = pixels + (UVrect.y * uv_pitch) + UVrect.x;

        length = UVrect.w * 2;

        // UV plane
        if (length == UVpitch && length == uv_pitch) {
            SDL_memcpy(UVdst, UVplane, length * UVrect.h);
        } else {
            for (row = 0; row < UVrect.h; ++row) {
                SDL_memcpy(UVdst, UVplane, length);
                UVplane += UVpitch;
                UVdst += uv_pitch;
            }
        }
    }

    return 0;
}

#endif

static int VITA_GXM_LockTexture(SDL_Renderer *renderer, SDL_Texture *texture,
                                const SDL_Rect *rect, void **pixels, int *pitch)
{
    VITA_GXM_RenderData *data = (VITA_GXM_RenderData *)renderer->driverdata;
    VITA_GXM_TextureData *vita_texture = (VITA_GXM_TextureData *)texture->driverdata;

    *pixels =
        (void *)((Uint8 *)gxm_texture_get_datap(vita_texture->tex) + (rect->y * vita_texture->pitch) + rect->x * SDL_BYTESPERPIXEL(texture->format));
    *pitch = vita_texture->pitch;

    // make sure that rendering is finished on render target textures
    if (vita_texture->tex->gxm_rendertarget) {
        sceGxmFinish(data->gxm_context);
    }

    return 0;
}

static void VITA_GXM_UnlockTexture(SDL_Renderer *renderer, SDL_Texture *texture)
{
    // No need to update texture data on ps vita.
    // VITA_GXM_LockTexture already returns a pointer to the texture pixels buffer.
    // This really improves framerate when using lock/unlock.
}

static void VITA_GXM_SetTextureScaleMode(SDL_Renderer *renderer, SDL_Texture *texture, SDL_ScaleMode scaleMode)
{
    VITA_GXM_TextureData *vita_texture = (VITA_GXM_TextureData *)texture->driverdata;

    /*
     set texture filtering according to scaleMode
     suported hint values are nearest (0, default) or linear (1)
     vitaScaleMode is either SCE_GXM_TEXTURE_FILTER_POINT (good for tile-map)
     or SCE_GXM_TEXTURE_FILTER_LINEAR (good for scaling)
     */

    int vitaScaleMode = (scaleMode == SDL_ScaleModeNearest
                             ? SCE_GXM_TEXTURE_FILTER_POINT
                             : SCE_GXM_TEXTURE_FILTER_LINEAR);
    gxm_texture_set_filters(vita_texture->tex, vitaScaleMode, vitaScaleMode);

    return;
}

static int VITA_GXM_SetRenderTarget(SDL_Renderer *renderer, SDL_Texture *texture)
{
    return 0;
}

static void VITA_GXM_SetBlendMode(VITA_GXM_RenderData *data, int blendMode)
{
    if (blendMode != data->currentBlendMode) {
        fragment_programs *in = &data->blendFragmentPrograms.blend_mode_blend;

        switch (blendMode) {
        case SDL_BLENDMODE_NONE:
            in = &data->blendFragmentPrograms.blend_mode_none;
            break;
        case SDL_BLENDMODE_BLEND:
            in = &data->blendFragmentPrograms.blend_mode_blend;
            break;
        case SDL_BLENDMODE_ADD:
            in = &data->blendFragmentPrograms.blend_mode_add;
            break;
        case SDL_BLENDMODE_MOD:
            in = &data->blendFragmentPrograms.blend_mode_mod;
            break;
        case SDL_BLENDMODE_MUL:
            in = &data->blendFragmentPrograms.blend_mode_mul;
            break;
        }
        data->colorFragmentProgram = in->color;
        data->textureFragmentProgram = in->texture;
        data->currentBlendMode = blendMode;
    }
}

static int VITA_GXM_QueueSetViewport(SDL_Renderer *renderer, SDL_RenderCommand *cmd)
{
    return 0;
}

static int VITA_GXM_QueueSetDrawColor(SDL_Renderer *renderer, SDL_RenderCommand *cmd)
{
    VITA_GXM_RenderData *data = (VITA_GXM_RenderData *)renderer->driverdata;

    data->drawstate.color.r = cmd->data.color.r;
    data->drawstate.color.g = cmd->data.color.g;
    data->drawstate.color.b = cmd->data.color.b;
    data->drawstate.color.a = cmd->data.color.a;

    return 0;
}

static int VITA_GXM_QueueDrawPoints(SDL_Renderer *renderer, SDL_RenderCommand *cmd, const SDL_FPoint *points, int count)
{
    VITA_GXM_RenderData *data = (VITA_GXM_RenderData *)renderer->driverdata;

    SDL_Color color = data->drawstate.color;

    color_vertex *vertex = (color_vertex *)pool_malloc(
        data,
        count * sizeof(color_vertex));

    cmd->data.draw.first = (size_t)vertex;
    cmd->data.draw.count = count;

    for (int i = 0; i < count; i++) {
        vertex[i].x = points[i].x;
        vertex[i].y = points[i].y;
        vertex[i].color = color;
    }

    return 0;
}

static int VITA_GXM_QueueDrawLines(SDL_Renderer *renderer, SDL_RenderCommand *cmd, const SDL_FPoint *points, int count)
{
    VITA_GXM_RenderData *data = (VITA_GXM_RenderData *)renderer->driverdata;
    SDL_Color color = data->drawstate.color;

    color_vertex *vertex = (color_vertex *)pool_malloc(
        data,
        (count - 1) * 2 * sizeof(color_vertex));

    cmd->data.draw.first = (size_t)vertex;
    cmd->data.draw.count = (count - 1) * 2;

    for (int i = 0; i < count - 1; i++) {
        vertex[i * 2].x = points[i].x;
        vertex[i * 2].y = points[i].y;
        vertex[i * 2].color = color;

        vertex[i * 2 + 1].x = points[i + 1].x;
        vertex[i * 2 + 1].y = points[i + 1].y;
        vertex[i * 2 + 1].color = color;
    }

    return 0;
}

static int VITA_GXM_QueueGeometry(SDL_Renderer *renderer, SDL_RenderCommand *cmd, SDL_Texture *texture,
                                  const float *xy, int xy_stride, const SDL_Color *color, int color_stride, const float *uv, int uv_stride,
                                  int num_vertices, const void *indices, int num_indices, int size_indices,
                                  float scale_x, float scale_y)
{
    VITA_GXM_RenderData *data = (VITA_GXM_RenderData *)renderer->driverdata;
    int i;
    int count = indices ? num_indices : num_vertices;

    cmd->data.draw.count = count;
    size_indices = indices ? size_indices : 0;

    if (texture) {
        VITA_GXM_TextureData *vita_texture = (VITA_GXM_TextureData *)texture->driverdata;
        texture_vertex *vertices;

        vertices = (texture_vertex *)pool_malloc(
            data,
            count * sizeof(texture_vertex));

        if (!vertices) {
            return -1;
        }

        for (i = 0; i < count; i++) {
            int j;
            float *xy_;
            float *uv_;
            SDL_Color col_;
            if (size_indices == 4) {
                j = ((const Uint32 *)indices)[i];
            } else if (size_indices == 2) {
                j = ((const Uint16 *)indices)[i];
            } else if (size_indices == 1) {
                j = ((const Uint8 *)indices)[i];
            } else {
                j = i;
            }

            xy_ = (float *)((char *)xy + j * xy_stride);
            col_ = *(SDL_Color *)((char *)color + j * color_stride);
            uv_ = (float *)((char *)uv + j * uv_stride);

            vertices[i].x = xy_[0] * scale_x;
            vertices[i].y = xy_[1] * scale_y;
            vertices[i].u = uv_[0] * vita_texture->wscale;
            vertices[i].v = uv_[1];
            vertices[i].color = col_;
        }

        cmd->data.draw.first = (size_t)vertices;

    } else {
        color_vertex *vertices;

        vertices = (color_vertex *)pool_malloc(
            data,
            count * sizeof(color_vertex));

        if (!vertices) {
            return -1;
        }

        for (i = 0; i < count; i++) {
            int j;
            float *xy_;
            SDL_Color col_;
            if (size_indices == 4) {
                j = ((const Uint32 *)indices)[i];
            } else if (size_indices == 2) {
                j = ((const Uint16 *)indices)[i];
            } else if (size_indices == 1) {
                j = ((const Uint8 *)indices)[i];
            } else {
                j = i;
            }

            xy_ = (float *)((char *)xy + j * xy_stride);
            col_ = *(SDL_Color *)((char *)color + j * color_stride);

            vertices[i].x = xy_[0] * scale_x;
            vertices[i].y = xy_[1] * scale_y;
            vertices[i].color = col_;
        }
        cmd->data.draw.first = (size_t)vertices;
    }

    return 0;
}

static int VITA_GXM_RenderClear(SDL_Renderer *renderer, SDL_RenderCommand *cmd)
{
    void *color_buffer;
    float clear_color[4];

    VITA_GXM_RenderData *data = (VITA_GXM_RenderData *)renderer->driverdata;
    unset_clip_rectangle(data);

    clear_color[0] = (cmd->data.color.r) / 255.0f;
    clear_color[1] = (cmd->data.color.g) / 255.0f;
    clear_color[2] = (cmd->data.color.b) / 255.0f;
    clear_color[3] = (cmd->data.color.a) / 255.0f;

    // set clear shaders
    data->drawstate.fragment_program = data->clearFragmentProgram;
    data->drawstate.vertex_program = data->clearVertexProgram;
    sceGxmSetVertexProgram(data->gxm_context, data->clearVertexProgram);
    sceGxmSetFragmentProgram(data->gxm_context, data->clearFragmentProgram);

    // set the clear color
    sceGxmReserveFragmentDefaultUniformBuffer(data->gxm_context, &color_buffer);
    sceGxmSetUniformDataF(color_buffer, data->clearClearColorParam, 0, 4, clear_color);

    // draw the clear triangle
    sceGxmSetVertexStream(data->gxm_context, 0, data->clearVertices);
    sceGxmDraw(data->gxm_context, SCE_GXM_PRIMITIVE_TRIANGLES, SCE_GXM_INDEX_FORMAT_U16, data->linearIndices, 3);

    data->drawstate.cliprect_dirty = SDL_TRUE;
    return 0;
}

static int SetDrawState(VITA_GXM_RenderData *data, const SDL_RenderCommand *cmd)
{
    SDL_Texture *texture = cmd->data.draw.texture;
    const SDL_BlendMode blend = cmd->data.draw.blend;
    SceGxmFragmentProgram *fragment_program;
    SceGxmVertexProgram *vertex_program;
    SDL_bool matrix_updated = SDL_FALSE;
    SDL_bool program_updated = SDL_FALSE;

    if (data->drawstate.viewport_dirty) {
        const SDL_Rect *viewport = &data->drawstate.viewport;

        float sw = viewport->w / 2.;
        float sh = viewport->h / 2.;

        float x_scale = sw;
        float x_off = viewport->x + sw;
        float y_scale = -(sh);
        float y_off = viewport->y + sh;

        sceGxmSetViewport(data->gxm_context, x_off, x_scale, y_off, y_scale, 0.5f, 0.5f);

        if (viewport->w && viewport->h) {
            init_orthographic_matrix(data->ortho_matrix,
                                     (float)0,
                                     (float)viewport->w,
                                     (float)viewport->h,
                                     (float)0,
                                     0.0f, 1.0f);
            matrix_updated = SDL_TRUE;
        }

        data->drawstate.viewport_dirty = SDL_FALSE;
    }

    if (data->drawstate.cliprect_enabled_dirty) {
        if (!data->drawstate.cliprect_enabled) {
            unset_clip_rectangle(data);
        }
        data->drawstate.cliprect_enabled_dirty = SDL_FALSE;
    }

    if (data->drawstate.cliprect_enabled && data->drawstate.cliprect_dirty) {
        const SDL_Rect *rect = &data->drawstate.cliprect;
        set_clip_rectangle(data, rect->x, rect->y, rect->x + rect->w, rect->y + rect->h);
        data->drawstate.cliprect_dirty = SDL_FALSE;
    }

    VITA_GXM_SetBlendMode(data, blend); // do that first, to select appropriate shaders

    if (texture) {
        vertex_program = data->textureVertexProgram;
        fragment_program = data->textureFragmentProgram;
    } else {
        vertex_program = data->colorVertexProgram;
        fragment_program = data->colorFragmentProgram;
    }

    if (data->drawstate.vertex_program != vertex_program) {
        data->drawstate.vertex_program = vertex_program;
        sceGxmSetVertexProgram(data->gxm_context, vertex_program);
        program_updated = SDL_TRUE;
    }

    if (data->drawstate.fragment_program != fragment_program) {
        data->drawstate.fragment_program = fragment_program;
        sceGxmSetFragmentProgram(data->gxm_context, fragment_program);
        program_updated = SDL_TRUE;
    }

    if (program_updated || matrix_updated) {
        if (data->drawstate.fragment_program == data->textureFragmentProgram) {
            void *vertex_wvp_buffer;
            sceGxmReserveVertexDefaultUniformBuffer(data->gxm_context, &vertex_wvp_buffer);
            sceGxmSetUniformDataF(vertex_wvp_buffer, data->textureWvpParam, 0, 16, data->ortho_matrix);
        } else { // color
            void *vertexDefaultBuffer;
            sceGxmReserveVertexDefaultUniformBuffer(data->gxm_context, &vertexDefaultBuffer);
            sceGxmSetUniformDataF(vertexDefaultBuffer, data->colorWvpParam, 0, 16, data->ortho_matrix);
        }
    }

    if (texture != data->drawstate.texture) {
        if (texture) {
            VITA_GXM_TextureData *vita_texture = (VITA_GXM_TextureData *)cmd->data.draw.texture->driverdata;
            sceGxmSetFragmentTexture(data->gxm_context, 0, &vita_texture->tex->gxm_tex);
        }
        data->drawstate.texture = texture;
    }

    /* all drawing commands use this */
    sceGxmSetVertexStream(data->gxm_context, 0, (const void *)cmd->data.draw.first);

    return 0;
}

static int VITA_GXM_RunCommandQueue(SDL_Renderer *renderer, SDL_RenderCommand *cmd, void *vertices, size_t vertsize)
{
    VITA_GXM_RenderData *data = (VITA_GXM_RenderData *)renderer->driverdata;
    StartDrawing(renderer);

    data->drawstate.target = renderer->target;
    if (!data->drawstate.target) {
        int w, h;
        SDL_GL_GetDrawableSize(renderer->window, &w, &h);
        if ((w != data->drawstate.drawablew) || (h != data->drawstate.drawableh)) {
            data->drawstate.viewport_dirty = SDL_TRUE; // if the window dimensions changed, invalidate the current viewport, etc.
            data->drawstate.cliprect_dirty = SDL_TRUE;
            data->drawstate.drawablew = w;
            data->drawstate.drawableh = h;
        }
    }

    while (cmd) {
        switch (cmd->command) {

        case SDL_RENDERCMD_SETVIEWPORT:
        {
            SDL_Rect *viewport = &data->drawstate.viewport;
            if (SDL_memcmp(viewport, &cmd->data.viewport.rect, sizeof(cmd->data.viewport.rect)) != 0) {
                SDL_copyp(viewport, &cmd->data.viewport.rect);
                data->drawstate.viewport_dirty = SDL_TRUE;
                data->drawstate.cliprect_dirty = SDL_TRUE;
            }
            break;
        }

        case SDL_RENDERCMD_SETCLIPRECT:
        {
            const SDL_Rect *rect = &cmd->data.cliprect.rect;
            if (data->drawstate.cliprect_enabled != cmd->data.cliprect.enabled) {
                data->drawstate.cliprect_enabled = cmd->data.cliprect.enabled;
                data->drawstate.cliprect_enabled_dirty = SDL_TRUE;
            }

            if (SDL_memcmp(&data->drawstate.cliprect, rect, sizeof(*rect)) != 0) {
                SDL_copyp(&data->drawstate.cliprect, rect);
                data->drawstate.cliprect_dirty = SDL_TRUE;
            }
            break;
        }

        case SDL_RENDERCMD_SETDRAWCOLOR:
        {
            break;
        }

        case SDL_RENDERCMD_CLEAR:
        {
            VITA_GXM_RenderClear(renderer, cmd);
            break;
        }

        case SDL_RENDERCMD_FILL_RECTS: /* unused */
            break;

        case SDL_RENDERCMD_COPY: /* unused */
            break;

        case SDL_RENDERCMD_COPY_EX: /* unused */
            break;

        case SDL_RENDERCMD_DRAW_POINTS:
        case SDL_RENDERCMD_DRAW_LINES:
        case SDL_RENDERCMD_GEOMETRY:
        {
            SDL_Texture *thistexture = cmd->data.draw.texture;
            SDL_BlendMode thisblend = cmd->data.draw.blend;
            const SDL_RenderCommandType thiscmdtype = cmd->command;
            SDL_RenderCommand *finalcmd = cmd;
            SDL_RenderCommand *nextcmd = cmd->next;
            size_t count = cmd->data.draw.count;
            int ret;
            while (nextcmd) {
                const SDL_RenderCommandType nextcmdtype = nextcmd->command;
                if (nextcmdtype != thiscmdtype) {
                    break; /* can't go any further on this draw call, different render command up next. */
                } else if (nextcmd->data.draw.texture != thistexture || nextcmd->data.draw.blend != thisblend) {
                    break; /* can't go any further on this draw call, different texture/blendmode copy up next. */
                } else {
                    finalcmd = nextcmd; /* we can combine copy operations here. Mark this one as the furthest okay command. */
                    count += nextcmd->data.draw.count;
                }
                nextcmd = nextcmd->next;
            }

            ret = SetDrawState(data, cmd);

            if (ret == 0) {
                int op = SCE_GXM_PRIMITIVE_TRIANGLES;

                if (thiscmdtype == SDL_RENDERCMD_DRAW_POINTS) {
                    sceGxmSetFrontPolygonMode(data->gxm_context, SCE_GXM_POLYGON_MODE_POINT);
                    op = SCE_GXM_PRIMITIVE_POINTS;
                } else if (thiscmdtype == SDL_RENDERCMD_DRAW_LINES) {
                    sceGxmSetFrontPolygonMode(data->gxm_context, SCE_GXM_POLYGON_MODE_LINE);
                    op = SCE_GXM_PRIMITIVE_LINES;
                }

                sceGxmDraw(data->gxm_context, op, SCE_GXM_INDEX_FORMAT_U16, data->linearIndices, count);

                if (thiscmdtype == SDL_RENDERCMD_DRAW_POINTS || thiscmdtype == SDL_RENDERCMD_DRAW_LINES) {
                    sceGxmSetFrontPolygonMode(data->gxm_context, SCE_GXM_POLYGON_MODE_TRIANGLE_FILL);
                }
            }

            cmd = finalcmd; /* skip any copy commands we just combined in here. */
            break;
        }

        case SDL_RENDERCMD_NO_OP:
            break;
        }
        data->drawstate.last_command = cmd->command;
        cmd = cmd->next;
    }

    sceGxmEndScene(data->gxm_context, NULL, NULL);
    data->drawing = SDL_FALSE;

    return 0;
}

void read_pixels(int x, int y, size_t width, size_t height, void *data)
{
    SceDisplayFrameBuf pParam;
    int i, j;
    Uint32 *out32;
    Uint32 *in32;

    pParam.size = sizeof(SceDisplayFrameBuf);

    sceDisplayGetFrameBuf(&pParam, SCE_DISPLAY_SETBUF_NEXTFRAME);

    out32 = (Uint32 *)data;
    in32 = (Uint32 *)pParam.base;

    in32 += (x + y * pParam.pitch);

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            out32[(height - (i + 1)) * width + j] = in32[j];
        }
        in32 += pParam.pitch;
    }
}

static int VITA_GXM_RenderReadPixels(SDL_Renderer *renderer, const SDL_Rect *rect,
                                     Uint32 pixel_format, void *pixels, int pitch)
{
    Uint32 temp_format = renderer->target ? renderer->target->format : SDL_PIXELFORMAT_ABGR8888;
    size_t buflen;
    void *temp_pixels;
    int temp_pitch;
    Uint8 *src, *dst, *tmp;
    int w, h, length, rows;
    int status;

    // TODO: read from texture rendertarget. Although no-one sane should do it.
    if (renderer->target) {
        return SDL_Unsupported();
    }

    temp_pitch = rect->w * SDL_BYTESPERPIXEL(temp_format);
    buflen = rect->h * temp_pitch;
    if (buflen == 0) {
        return 0; /* nothing to do. */
    }

    temp_pixels = SDL_malloc(buflen);
    if (!temp_pixels) {
        return SDL_OutOfMemory();
    }

    SDL_GetRendererOutputSize(renderer, &w, &h);

    read_pixels(rect->x, renderer->target ? rect->y : (h - rect->y) - rect->h,
                rect->w, rect->h, temp_pixels);

    /* Flip the rows to be top-down if necessary */

    if (!renderer->target) {
        SDL_bool isstack;
        length = rect->w * SDL_BYTESPERPIXEL(temp_format);
        src = (Uint8 *)temp_pixels + (rect->h - 1) * temp_pitch;
        dst = (Uint8 *)temp_pixels;
        tmp = SDL_small_alloc(Uint8, length, &isstack);
        rows = rect->h / 2;
        while (rows--) {
            SDL_memcpy(tmp, dst, length);
            SDL_memcpy(dst, src, length);
            SDL_memcpy(src, tmp, length);
            dst += temp_pitch;
            src -= temp_pitch;
        }
        SDL_small_free(tmp, isstack);
    }

    status = SDL_ConvertPixels(rect->w, rect->h,
                               temp_format, temp_pixels, temp_pitch,
                               pixel_format, pixels, pitch);
    SDL_free(temp_pixels);

    return status;
}

static int VITA_GXM_RenderPresent(SDL_Renderer *renderer)
{
    VITA_GXM_RenderData *data = (VITA_GXM_RenderData *)renderer->driverdata;
    SceCommonDialogUpdateParam updateParam;

    data->displayData.address = data->displayBufferData[data->backBufferIndex];

    SDL_memset(&updateParam, 0, sizeof(updateParam));

    updateParam.renderTarget.colorFormat = VITA_GXM_COLOR_FORMAT;
    updateParam.renderTarget.surfaceType = SCE_GXM_COLOR_SURFACE_LINEAR;
    updateParam.renderTarget.width = VITA_GXM_SCREEN_WIDTH;
    updateParam.renderTarget.height = VITA_GXM_SCREEN_HEIGHT;
    updateParam.renderTarget.strideInPixels = VITA_GXM_SCREEN_STRIDE;

    updateParam.renderTarget.colorSurfaceData = data->displayBufferData[data->backBufferIndex];
    updateParam.renderTarget.depthSurfaceData = data->depthBufferData;

    updateParam.displaySyncObject = (SceGxmSyncObject *)data->displayBufferSync[data->backBufferIndex];

    sceCommonDialogUpdate(&updateParam);

#ifdef DEBUG_RAZOR
    sceGxmPadHeartbeat(
        (const SceGxmColorSurface *)&data->displaySurface[data->backBufferIndex],
        (SceGxmSyncObject *)data->displayBufferSync[data->backBufferIndex]);
#endif

    sceGxmDisplayQueueAddEntry(
        data->displayBufferSync[data->frontBufferIndex], // OLD fb
        data->displayBufferSync[data->backBufferIndex],  // NEW fb
        &data->displayData);

    // update buffer indices
    data->frontBufferIndex = data->backBufferIndex;
    data->backBufferIndex = (data->backBufferIndex + 1) % VITA_GXM_BUFFERS;
    data->pool_index = 0;

    data->current_pool = (data->current_pool + 1) % 2;
    return 0;
}

static void VITA_GXM_DestroyTexture(SDL_Renderer *renderer, SDL_Texture *texture)
{
    VITA_GXM_RenderData *data = (VITA_GXM_RenderData *)renderer->driverdata;
    VITA_GXM_TextureData *vita_texture = (VITA_GXM_TextureData *)texture->driverdata;

    if (!data) {
        return;
    }

    if (!vita_texture) {
        return;
    }

    if (!vita_texture->tex) {
        return;
    }

    sceGxmFinish(data->gxm_context);

    free_gxm_texture(data, vita_texture->tex);

    SDL_free(vita_texture);

    texture->driverdata = NULL;
}

static void VITA_GXM_DestroyRenderer(SDL_Renderer *renderer)
{
    VITA_GXM_RenderData *data = (VITA_GXM_RenderData *)renderer->driverdata;
    if (data) {
        if (!data->initialized) {
            return;
        }

        gxm_finish(renderer);

        data->initialized = SDL_FALSE;
        data->drawing = SDL_FALSE;
        SDL_free(data);
    }
    SDL_free(renderer);
}

#endif /* SDL_VIDEO_RENDER_VITA_GXM */

/* vi: set ts=4 sw=4 expandtab: */
