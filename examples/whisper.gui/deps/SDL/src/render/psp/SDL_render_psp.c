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

#if SDL_VIDEO_RENDER_PSP

#include "SDL_hints.h"
#include "../SDL_sysrender.h"

#include <pspkernel.h>
#include <pspdisplay.h>
#include <pspgu.h>
#include <pspgum.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <pspge.h>
#include <stdarg.h>
#include <stdlib.h>
#include <vram.h>

/* PSP renderer implementation, based on the PGE  */

#define PSP_SCREEN_WIDTH  480
#define PSP_SCREEN_HEIGHT 272

#define PSP_FRAME_BUFFER_WIDTH 512
#define PSP_FRAME_BUFFER_SIZE  (PSP_FRAME_BUFFER_WIDTH * PSP_SCREEN_HEIGHT)

static unsigned int __attribute__((aligned(16))) DisplayList[262144];

#define COL5650(r, g, b, a) ((r >> 3) | ((g >> 2) << 5) | ((b >> 3) << 11))
#define COL5551(r, g, b, a) ((r >> 3) | ((g >> 3) << 5) | ((b >> 3) << 10) | (a > 0 ? 0x7000 : 0))
#define COL4444(r, g, b, a) ((r >> 4) | ((g >> 4) << 4) | ((b >> 4) << 8) | ((a >> 4) << 12))
#define COL8888(r, g, b, a) ((r) | ((g) << 8) | ((b) << 16) | ((a) << 24))

/**
 * Holds psp specific texture data
 *
 * Part of a hot-list of textures that are used as render targets
 * When short of vram we spill Least-Recently-Used render targets to system memory
 */
typedef struct PSP_TextureData
{
    void *data;                 /**< Image data. */
    unsigned int size;          /**< Size of data in bytes. */
    unsigned int width;         /**< Image width. */
    unsigned int height;        /**< Image height. */
    unsigned int textureWidth;  /**< Texture width (power of two). */
    unsigned int textureHeight; /**< Texture height (power of two). */
    unsigned int bits;          /**< Image bits per pixel. */
    unsigned int format;        /**< Image format - one of ::pgePixelFormat. */
    unsigned int pitch;
    SDL_bool swizzled;                /**< Is image swizzled. */
    struct PSP_TextureData *prevhotw; /**< More recently used render target */
    struct PSP_TextureData *nexthotw; /**< Less recently used render target */
} PSP_TextureData;

typedef struct
{
    SDL_BlendMode mode;
    unsigned int color;
    int shadeModel;
    SDL_Texture *texture;
} PSP_BlendState;

typedef struct
{
    void *frontbuffer;         /**< main screen buffer */
    void *backbuffer;          /**< buffer presented to display */
    SDL_Texture *boundTarget;  /**< currently bound rendertarget */
    SDL_bool initialized;      /**< is driver initialized */
    SDL_bool displayListAvail; /**< is the display list already initialized for this frame */
    unsigned int psm;          /**< format of the display buffers */
    unsigned int bpp;          /**< bits per pixel of the main display */

    SDL_bool vsync;                       /**< whether we do vsync */
    PSP_BlendState blendState;            /**< current blend mode */
    PSP_TextureData *most_recent_target;  /**< start of render target LRU double linked list */
    PSP_TextureData *least_recent_target; /**< end of the LRU list */

    SDL_bool vblank_not_reached; /**< whether vblank wasn't reached */
} PSP_RenderData;

typedef struct
{
    float x, y, z;
} VertV;

typedef struct
{
    float u, v;
    float x, y, z;
} VertTV;

typedef struct
{
    SDL_Color col;
    float x, y, z;
} VertCV;

typedef struct
{
    float u, v;
    SDL_Color col;
    float x, y, z;
} VertTCV;

#define PI 3.14159265358979f

#define radToDeg(x) ((x)*180.f / PI)
#define degToRad(x) ((x)*PI / 180.f)

static float MathAbs(float x)
{
    float result;

    __asm__ volatile(
        "mtv      %1, S000\n"
        "vabs.s   S000, S000\n"
        "mfv      %0, S000\n"
        : "=r"(result)
        : "r"(x));

    return result;
}

static void MathSincos(float r, float *s, float *c)
{
    __asm__ volatile(
        "mtv      %2, S002\n"
        "vcst.s   S003, VFPU_2_PI\n"
        "vmul.s   S002, S002, S003\n"
        "vrot.p   C000, S002, [s, c]\n"
        "mfv      %0, S000\n"
        "mfv      %1, S001\n"
        : "=r"(*s), "=r"(*c)
        : "r"(r));
}

static void Swap(float *a, float *b)
{
    float n = *a;
    *a = *b;
    *b = n;
}

static inline int InVram(void *data)
{
    return data < (void *)0x04200000;
}

/* Return next power of 2 */
static int TextureNextPow2(unsigned int w)
{
    unsigned int n = 2;
    if (w == 0) {
        return 0;
    }

    while (w > n) {
        n <<= 1;
    }

    return n;
}

static void psp_on_vblank(u32 sub, PSP_RenderData *data)
{
    if (data) {
        data->vblank_not_reached = SDL_FALSE;
    }
}

static int PixelFormatToPSPFMT(Uint32 format)
{
    switch (format) {
    case SDL_PIXELFORMAT_BGR565:
        return GU_PSM_5650;
    case SDL_PIXELFORMAT_ABGR1555:
        return GU_PSM_5551;
    case SDL_PIXELFORMAT_ABGR4444:
        return GU_PSM_4444;
    case SDL_PIXELFORMAT_ABGR8888:
        return GU_PSM_8888;
    default:
        return GU_PSM_8888;
    }
}

/// SECTION render target LRU management
static void LRUTargetRelink(PSP_TextureData *psp_texture)
{
    if (psp_texture->prevhotw) {
        psp_texture->prevhotw->nexthotw = psp_texture->nexthotw;
    }
    if (psp_texture->nexthotw) {
        psp_texture->nexthotw->prevhotw = psp_texture->prevhotw;
    }
}

static void LRUTargetPushFront(PSP_RenderData *data, PSP_TextureData *psp_texture)
{
    psp_texture->nexthotw = data->most_recent_target;
    if (data->most_recent_target) {
        data->most_recent_target->prevhotw = psp_texture;
    }
    data->most_recent_target = psp_texture;
    if (!data->least_recent_target) {
        data->least_recent_target = psp_texture;
    }
}

static void LRUTargetRemove(PSP_RenderData *data, PSP_TextureData *psp_texture)
{
    LRUTargetRelink(psp_texture);
    if (data->most_recent_target == psp_texture) {
        data->most_recent_target = psp_texture->nexthotw;
    }
    if (data->least_recent_target == psp_texture) {
        data->least_recent_target = psp_texture->prevhotw;
    }
    psp_texture->prevhotw = NULL;
    psp_texture->nexthotw = NULL;
}

static void LRUTargetBringFront(PSP_RenderData *data, PSP_TextureData *psp_texture)
{
    if (data->most_recent_target == psp_texture) {
        return; // nothing to do
    }
    LRUTargetRemove(data, psp_texture);
    LRUTargetPushFront(data, psp_texture);
}

static void TextureStorageFree(void *storage)
{
    if (InVram(storage)) {
        vfree(storage);
    } else {
        SDL_free(storage);
    }
}

static int TextureSwizzle(PSP_TextureData *psp_texture, void *dst)
{
    int bytewidth, height;
    int rowblocks, rowblocksadd;
    int i, j;
    unsigned int blockaddress = 0;
    unsigned int *src = NULL;
    unsigned char *data = NULL;

    if (psp_texture->swizzled) {
        return 1;
    }

    bytewidth = psp_texture->textureWidth * (psp_texture->bits >> 3);
    height = psp_texture->size / bytewidth;

    rowblocks = (bytewidth >> 4);
    rowblocksadd = (rowblocks - 1) << 7;

    src = (unsigned int *)psp_texture->data;

    data = dst;
    if (!data) {
        data = SDL_malloc(psp_texture->size);
    }

    if (!data) {
        return SDL_OutOfMemory();
    }

    for (j = 0; j < height; j++, blockaddress += 16) {
        unsigned int *block;

        block = (unsigned int *)&data[blockaddress];

        for (i = 0; i < rowblocks; i++) {
            *block++ = *src++;
            *block++ = *src++;
            *block++ = *src++;
            *block++ = *src++;
            block += 28;
        }

        if ((j & 0x7) == 0x7) {
            blockaddress += rowblocksadd;
        }
    }

    TextureStorageFree(psp_texture->data);
    psp_texture->data = data;
    psp_texture->swizzled = SDL_TRUE;

    sceKernelDcacheWritebackRange(psp_texture->data, psp_texture->size);
    return 1;
}

static int TextureUnswizzle(PSP_TextureData *psp_texture, void *dst)
{
    int bytewidth, height;
    int widthblocks, heightblocks;
    int dstpitch, dstrow;
    int blockx, blocky;
    int j;
    unsigned int *src = NULL;
    unsigned char *data = NULL;
    unsigned char *ydst = NULL;

    if (!psp_texture->swizzled) {
        return 1;
    }

    bytewidth = psp_texture->textureWidth * (psp_texture->bits >> 3);
    height = psp_texture->size / bytewidth;

    widthblocks = bytewidth / 16;
    heightblocks = height / 8;

    dstpitch = (bytewidth - 16) / 4;
    dstrow = bytewidth * 8;

    src = (unsigned int *)psp_texture->data;

    data = dst;

    if (!data) {
        data = SDL_malloc(psp_texture->size);
    }

    if (!data) {
        return SDL_OutOfMemory();
    }

    ydst = (unsigned char *)data;

    for (blocky = 0; blocky < heightblocks; ++blocky) {
        unsigned char *xdst = ydst;

        for (blockx = 0; blockx < widthblocks; ++blockx) {
            unsigned int *block;

            block = (unsigned int *)xdst;

            for (j = 0; j < 8; ++j) {
                *(block++) = *(src++);
                *(block++) = *(src++);
                *(block++) = *(src++);
                *(block++) = *(src++);
                block += dstpitch;
            }

            xdst += 16;
        }

        ydst += dstrow;
    }

    TextureStorageFree(psp_texture->data);

    psp_texture->data = data;

    psp_texture->swizzled = SDL_FALSE;

    sceKernelDcacheWritebackRange(psp_texture->data, psp_texture->size);
    return 1;
}

static int TextureSpillToSram(PSP_RenderData *data, PSP_TextureData *psp_texture)
{
    // Assumes the texture is in VRAM
    if (psp_texture->swizzled) {
        // Texture was swizzled in vram, just copy to system memory
        void *sdata = SDL_malloc(psp_texture->size);
        if (!sdata) {
            return SDL_OutOfMemory();
        }

        SDL_memcpy(sdata, psp_texture->data, psp_texture->size);
        vfree(psp_texture->data);
        psp_texture->data = sdata;
        return 0;
    } else {
        return TextureSwizzle(psp_texture, NULL); // Will realloc in sysram
    }
}

static int TexturePromoteToVram(PSP_RenderData *data, PSP_TextureData *psp_texture, SDL_bool target)
{
    // Assumes texture in sram and a large enough continuous block in vram
    void *tdata = vramalloc(psp_texture->size);
    if (psp_texture->swizzled && target) {
        return TextureUnswizzle(psp_texture, tdata);
    } else {
        SDL_memcpy(tdata, psp_texture->data, psp_texture->size);
        SDL_free(psp_texture->data);
        psp_texture->data = tdata;
        return 0;
    }
}

static int TextureSpillLRU(PSP_RenderData *data, size_t wanted)
{
    PSP_TextureData *lru = data->least_recent_target;
    if (lru) {
        if (TextureSpillToSram(data, lru) < 0) {
            return -1;
        }
        LRUTargetRemove(data, lru);
    } else {
        // Asked to spill but there nothing to spill
        return SDL_SetError("Could not spill more VRAM to system memory. VRAM : %dKB,(%dKB), wanted %dKB", vmemavail() / 1024, vlargestblock() / 1024, wanted / 1024);
    }
    return 0;
}

static int TextureSpillTargetsForSpace(PSP_RenderData *data, size_t size)
{
    while (vlargestblock() < size) {
        if (TextureSpillLRU(data, size) < 0) {
            return -1;
        }
    }
    return 0;
}

static int TextureBindAsTarget(PSP_RenderData *data, PSP_TextureData *psp_texture)
{
    unsigned int dstFormat;

    if (!InVram(psp_texture->data)) {
        // Bring back the texture in vram
        if (TextureSpillTargetsForSpace(data, psp_texture->size) < 0) {
            return -1;
        }
        if (TexturePromoteToVram(data, psp_texture, SDL_TRUE) < 0) {
            return -1;
        }
    }
    LRUTargetBringFront(data, psp_texture);
    sceGuDrawBufferList(psp_texture->format, vrelptr(psp_texture->data), psp_texture->textureWidth);

    // Stencil alpha dst hack
    dstFormat = psp_texture->format;
    if (dstFormat == GU_PSM_5551) {
        sceGuEnable(GU_STENCIL_TEST);
        sceGuStencilOp(GU_REPLACE, GU_REPLACE, GU_REPLACE);
        sceGuStencilFunc(GU_GEQUAL, 0xff, 0xff);
        sceGuEnable(GU_ALPHA_TEST);
        sceGuAlphaFunc(GU_GREATER, 0x00, 0xff);
    } else {
        sceGuDisable(GU_STENCIL_TEST);
        sceGuDisable(GU_ALPHA_TEST);
    }
    return 0;
}

static void PSP_WindowEvent(SDL_Renderer *renderer, const SDL_WindowEvent *event)
{
}

static int PSP_CreateTexture(SDL_Renderer *renderer, SDL_Texture *texture)
{
    PSP_RenderData *data = renderer->driverdata;
    PSP_TextureData *psp_texture = (PSP_TextureData *)SDL_calloc(1, sizeof(*psp_texture));

    if (!psp_texture) {
        return SDL_OutOfMemory();
    }

    psp_texture->swizzled = SDL_FALSE;
    psp_texture->width = texture->w;
    psp_texture->height = texture->h;
    psp_texture->textureHeight = TextureNextPow2(texture->h);
    psp_texture->textureWidth = TextureNextPow2(texture->w);
    psp_texture->format = PixelFormatToPSPFMT(texture->format);

    switch (psp_texture->format) {
    case GU_PSM_5650:
    case GU_PSM_5551:
    case GU_PSM_4444:
        psp_texture->bits = 16;
        break;

    case GU_PSM_8888:
        psp_texture->bits = 32;
        break;

    default:
        SDL_free(psp_texture);
        return -1;
    }

    psp_texture->pitch = psp_texture->textureWidth * SDL_BYTESPERPIXEL(texture->format);
    psp_texture->size = psp_texture->textureHeight * psp_texture->pitch;
    if (texture->access & SDL_TEXTUREACCESS_TARGET) {
        if (TextureSpillTargetsForSpace(renderer->driverdata, psp_texture->size) < 0) {
            SDL_free(psp_texture);
            return -1;
        }
        psp_texture->data = vramalloc(psp_texture->size);
        if (psp_texture->data) {
            LRUTargetPushFront(data, psp_texture);
        }
    } else {
        psp_texture->data = SDL_calloc(1, psp_texture->size);
    }

    if (!psp_texture->data) {
        SDL_free(psp_texture);
        return SDL_OutOfMemory();
    }
    texture->driverdata = psp_texture;

    return 0;
}

static int TextureShouldSwizzle(PSP_TextureData *psp_texture, SDL_Texture *texture)
{
    return !((texture->access == SDL_TEXTUREACCESS_TARGET) && InVram(psp_texture->data)) && texture->access != SDL_TEXTUREACCESS_STREAMING && (texture->w >= 16 || texture->h >= 16);
}

static void TextureActivate(SDL_Texture *texture)
{
    PSP_TextureData *psp_texture = (PSP_TextureData *)texture->driverdata;
    int scaleMode = (texture->scaleMode == SDL_ScaleModeNearest) ? GU_NEAREST : GU_LINEAR;

    /* Swizzling is useless with small textures. */
    if (TextureShouldSwizzle(psp_texture, texture)) {
        TextureSwizzle(psp_texture, NULL);
    }

    sceGuTexWrap(GU_REPEAT, GU_REPEAT);
    sceGuTexMode(psp_texture->format, 0, 0, psp_texture->swizzled);
    sceGuTexFilter(scaleMode, scaleMode); /* GU_NEAREST good for tile-map */
    /* GU_LINEAR good for scaling */
    sceGuTexImage(0, psp_texture->textureWidth, psp_texture->textureHeight, psp_texture->textureWidth, psp_texture->data);
}

static int PSP_LockTexture(SDL_Renderer *renderer, SDL_Texture *texture,
                           const SDL_Rect *rect, void **pixels, int *pitch);

static int PSP_UpdateTexture(SDL_Renderer *renderer, SDL_Texture *texture,
                             const SDL_Rect *rect, const void *pixels, int pitch)
{
    /*  PSP_TextureData *psp_texture = (PSP_TextureData *) texture->driverdata; */
    const Uint8 *src;
    Uint8 *dst;
    int row, length, dpitch;
    src = pixels;

    PSP_LockTexture(renderer, texture, rect, (void **)&dst, &dpitch);
    length = rect->w * SDL_BYTESPERPIXEL(texture->format);
    if (length == pitch && length == dpitch) {
        SDL_memcpy(dst, src, length * rect->h);
    } else {
        for (row = 0; row < rect->h; ++row) {
            SDL_memcpy(dst, src, length);
            src += pitch;
            dst += dpitch;
        }
    }

    sceKernelDcacheWritebackAll();
    return 0;
}

static int PSP_LockTexture(SDL_Renderer *renderer, SDL_Texture *texture,
                           const SDL_Rect *rect, void **pixels, int *pitch)
{
    PSP_TextureData *psp_texture = (PSP_TextureData *)texture->driverdata;

    *pixels =
        (void *)((Uint8 *)psp_texture->data + rect->y * psp_texture->pitch +
                 rect->x * SDL_BYTESPERPIXEL(texture->format));
    *pitch = psp_texture->pitch;
    return 0;
}

static void PSP_UnlockTexture(SDL_Renderer *renderer, SDL_Texture *texture)
{
    PSP_TextureData *psp_texture = (PSP_TextureData *)texture->driverdata;
    SDL_Rect rect;

    /* We do whole texture updates, at least for now */
    rect.x = 0;
    rect.y = 0;
    rect.w = texture->w;
    rect.h = texture->h;
    PSP_UpdateTexture(renderer, texture, &rect, psp_texture->data, psp_texture->pitch);
}

static void PSP_SetTextureScaleMode(SDL_Renderer *renderer, SDL_Texture *texture, SDL_ScaleMode scaleMode)
{
    /* Nothing to do because TextureActivate takes care of it */
}

static int PSP_SetRenderTarget(SDL_Renderer *renderer, SDL_Texture *texture)
{
    return 0;
}

static int PSP_QueueSetViewport(SDL_Renderer *renderer, SDL_RenderCommand *cmd)
{
    return 0; /* nothing to do in this backend. */
}

static int PSP_QueueDrawPoints(SDL_Renderer *renderer, SDL_RenderCommand *cmd, const SDL_FPoint *points, int count)
{
    VertV *verts = (VertV *)SDL_AllocateRenderVertices(renderer, count * sizeof(VertV), 4, &cmd->data.draw.first);
    int i;

    if (!verts) {
        return -1;
    }

    cmd->data.draw.count = count;

    for (i = 0; i < count; i++, verts++, points++) {
        verts->x = points->x;
        verts->y = points->y;
        verts->z = 0.0f;
    }

    return 0;
}

static int PSP_QueueGeometry(SDL_Renderer *renderer, SDL_RenderCommand *cmd, SDL_Texture *texture,
                             const float *xy, int xy_stride, const SDL_Color *color, int color_stride, const float *uv, int uv_stride,
                             int num_vertices, const void *indices, int num_indices, int size_indices,
                             float scale_x, float scale_y)
{
    int i;
    int count = indices ? num_indices : num_vertices;

    cmd->data.draw.count = count;
    size_indices = indices ? size_indices : 0;

    if (!texture) {
        VertCV *verts;
        verts = (VertCV *)SDL_AllocateRenderVertices(renderer, count * sizeof(VertCV), 4, &cmd->data.draw.first);
        if (!verts) {
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

            verts->x = xy_[0] * scale_x;
            verts->y = xy_[1] * scale_y;
            verts->z = 0;

            verts->col = col_;

            verts++;
        }
    } else {
        PSP_TextureData *psp_texture = (PSP_TextureData *)texture->driverdata;
        VertTCV *verts;
        verts = (VertTCV *)SDL_AllocateRenderVertices(renderer, count * sizeof(VertTCV), 4, &cmd->data.draw.first);
        if (!verts) {
            return -1;
        }

        for (i = 0; i < count; i++) {
            int j;
            float *xy_;
            SDL_Color col_;
            float *uv_;

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

            verts->x = xy_[0] * scale_x;
            verts->y = xy_[1] * scale_y;
            verts->z = 0;

            verts->col = col_;

            verts->u = uv_[0] * psp_texture->textureWidth;
            verts->v = uv_[1] * psp_texture->textureHeight;

            verts++;
        }
    }

    return 0;
}

static int PSP_QueueFillRects(SDL_Renderer *renderer, SDL_RenderCommand *cmd, const SDL_FRect *rects, int count)
{
    VertV *verts = (VertV *)SDL_AllocateRenderVertices(renderer, count * 2 * sizeof(VertV), 4, &cmd->data.draw.first);
    int i;

    if (!verts) {
        return -1;
    }

    cmd->data.draw.count = count;
    for (i = 0; i < count; i++, rects++) {
        verts->x = rects->x;
        verts->y = rects->y;
        verts->z = 0.0f;
        verts++;

        verts->x = rects->x + rects->w + 0.5f;
        verts->y = rects->y + rects->h + 0.5f;
        verts->z = 0.0f;
        verts++;
    }

    return 0;
}

static int PSP_QueueCopy(SDL_Renderer *renderer, SDL_RenderCommand *cmd, SDL_Texture *texture,
                         const SDL_Rect *srcrect, const SDL_FRect *dstrect)
{
    VertTV *verts;
    const float x = dstrect->x;
    const float y = dstrect->y;
    const float width = dstrect->w;
    const float height = dstrect->h;

    const float u0 = srcrect->x;
    const float v0 = srcrect->y;
    const float u1 = srcrect->x + srcrect->w;
    const float v1 = srcrect->y + srcrect->h;

    if ((MathAbs(u1) - MathAbs(u0)) < 64.0f) {
        verts = (VertTV *)SDL_AllocateRenderVertices(renderer, 2 * sizeof(VertTV), 4, &cmd->data.draw.first);
        if (!verts) {
            return -1;
        }

        cmd->data.draw.count = 1;

        verts->u = u0;
        verts->v = v0;
        verts->x = x;
        verts->y = y;
        verts->z = 0;
        verts++;

        verts->u = u1;
        verts->v = v1;
        verts->x = x + width;
        verts->y = y + height;
        verts->z = 0;
        verts++;
    } else {
        float start, end;
        float curU = u0;
        float curX = x;
        const float endX = x + width;
        const float slice = 64.0f;
        const size_t count = SDL_ceilf(width / slice);
        size_t i;
        float ustep = (u1 - u0) / width * slice;

        if (ustep < 0.0f) {
            ustep = -ustep;
        }

        cmd->data.draw.count = count;

        verts = (VertTV *)SDL_AllocateRenderVertices(renderer, count * 2 * sizeof(VertTV), 4, &cmd->data.draw.first);
        if (!verts) {
            return -1;
        }

        for (i = 0, start = 0, end = width; i < count; i++, start += slice) {
            const float polyWidth = ((curX + slice) > endX) ? (endX - curX) : slice;
            const float sourceWidth = ((curU + ustep) > u1) ? (u1 - curU) : ustep;

            SDL_assert(start < end);

            verts->u = curU;
            verts->v = v0;
            verts->x = curX;
            verts->y = y;
            verts->z = 0;
            verts++;

            curU += sourceWidth;
            curX += polyWidth;

            verts->u = curU;
            verts->v = v1;
            verts->x = curX;
            verts->y = (y + height);
            verts->z = 0;
            verts++;
        }
    }

    return 0;
}

static int PSP_QueueCopyEx(SDL_Renderer *renderer, SDL_RenderCommand *cmd, SDL_Texture *texture,
                           const SDL_Rect *srcrect, const SDL_FRect *dstrect,
                           const double angle, const SDL_FPoint *center, const SDL_RendererFlip flip, float scale_x, float scale_y)
{
    VertTV *verts = (VertTV *)SDL_AllocateRenderVertices(renderer, 4 * sizeof(VertTV), 4, &cmd->data.draw.first);
    const float centerx = center->x;
    const float centery = center->y;
    const float x = dstrect->x + centerx;
    const float y = dstrect->y + centery;
    const float width = dstrect->w - centerx;
    const float height = dstrect->h - centery;
    float s, c;
    float cw1, sw1, ch1, sh1, cw2, sw2, ch2, sh2;

    float u0 = srcrect->x;
    float v0 = srcrect->y;
    float u1 = srcrect->x + srcrect->w;
    float v1 = srcrect->y + srcrect->h;

    if (!verts) {
        return -1;
    }

    cmd->data.draw.count = 1;

    MathSincos(degToRad(360 - angle), &s, &c);

    cw1 = c * -centerx;
    sw1 = s * -centerx;
    ch1 = c * -centery;
    sh1 = s * -centery;
    cw2 = c * width;
    sw2 = s * width;
    ch2 = c * height;
    sh2 = s * height;

    if (flip & SDL_FLIP_VERTICAL) {
        Swap(&v0, &v1);
    }

    if (flip & SDL_FLIP_HORIZONTAL) {
        Swap(&u0, &u1);
    }

    verts->u = u0;
    verts->v = v0;
    verts->x = x + cw1 + sh1;
    verts->y = y - sw1 + ch1;
    verts->z = 0;
    verts++;

    verts->u = u0;
    verts->v = v1;
    verts->x = x + cw1 + sh2;
    verts->y = y - sw1 + ch2;
    verts->z = 0;
    verts++;

    verts->u = u1;
    verts->v = v1;
    verts->x = x + cw2 + sh2;
    verts->y = y - sw2 + ch2;
    verts->z = 0;
    verts++;

    verts->u = u1;
    verts->v = v0;
    verts->x = x + cw2 + sh1;
    verts->y = y - sw2 + ch1;
    verts->z = 0;

    if (scale_x != 1.0f || scale_y != 1.0f) {
        verts->x *= scale_x;
        verts->y *= scale_y;
        verts--;
        verts->x *= scale_x;
        verts->y *= scale_y;
        verts--;
        verts->x *= scale_x;
        verts->y *= scale_y;
        verts--;
        verts->x *= scale_x;
        verts->y *= scale_y;
    }

    return 0;
}

static void ResetBlendState(PSP_BlendState *state)
{
    sceGuColor(0xffffffff);
    state->color = 0xffffffff;
    state->mode = SDL_BLENDMODE_INVALID;
    state->texture = NULL;
    sceGuDisable(GU_TEXTURE_2D);
    sceGuShadeModel(GU_SMOOTH);
    state->shadeModel = GU_SMOOTH;
}

static void StartDrawing(SDL_Renderer *renderer)
{
    PSP_RenderData *data = (PSP_RenderData *)renderer->driverdata;

    // Check if we need to start GU displaylist
    if (!data->displayListAvail) {
        sceGuStart(GU_DIRECT, DisplayList);
        data->displayListAvail = SDL_TRUE;
        // ResetBlendState(&data->blendState);
    }

    // Check if we need a draw buffer change
    if (renderer->target != data->boundTarget) {
        SDL_Texture *texture = renderer->target;
        if (texture) {
            PSP_TextureData *psp_texture = (PSP_TextureData *)texture->driverdata;
            // Set target, registering LRU
            TextureBindAsTarget(data, psp_texture);
        } else {
            // Set target back to screen
            sceGuDrawBufferList(data->psm, vrelptr(data->frontbuffer), PSP_FRAME_BUFFER_WIDTH);
        }
        data->boundTarget = texture;
    }
}

static void PSP_SetBlendState(PSP_RenderData *data, PSP_BlendState *state)
{
    PSP_BlendState *current = &data->blendState;

    if (state->mode != current->mode) {
        switch (state->mode) {
        case SDL_BLENDMODE_NONE:
            sceGuTexFunc(GU_TFX_REPLACE, GU_TCC_RGBA);
            sceGuDisable(GU_BLEND);
            break;
        case SDL_BLENDMODE_BLEND:
            sceGuTexFunc(GU_TFX_MODULATE, GU_TCC_RGBA);
            sceGuBlendFunc(GU_ADD, GU_SRC_ALPHA, GU_ONE_MINUS_SRC_ALPHA, 0, 0);
            sceGuEnable(GU_BLEND);
            break;
        case SDL_BLENDMODE_ADD:
            sceGuTexFunc(GU_TFX_MODULATE, GU_TCC_RGBA);
            sceGuBlendFunc(GU_ADD, GU_SRC_ALPHA, GU_FIX, 0, 0x00FFFFFF);
            sceGuEnable(GU_BLEND);
            break;
        case SDL_BLENDMODE_MOD:
            sceGuTexFunc(GU_TFX_MODULATE, GU_TCC_RGBA);
            sceGuBlendFunc(GU_ADD, GU_FIX, GU_SRC_COLOR, 0, 0);
            sceGuEnable(GU_BLEND);
            break;
        case SDL_BLENDMODE_MUL:
            sceGuTexFunc(GU_TFX_MODULATE, GU_TCC_RGBA);
            /* FIXME SDL_BLENDMODE_MUL is simplified, and dstA is in fact un-changed.*/
            sceGuBlendFunc(GU_ADD, GU_DST_COLOR, GU_ONE_MINUS_SRC_ALPHA, 0, 0);
            sceGuEnable(GU_BLEND);
            break;
        case SDL_BLENDMODE_INVALID:
            break;
        }
    }

    if (state->color != current->color) {
        sceGuColor(state->color);
    }

    if (state->shadeModel != current->shadeModel) {
        sceGuShadeModel(state->shadeModel);
    }

    if (state->texture != current->texture) {
        if (state->texture) {
            TextureActivate(state->texture);
            sceGuEnable(GU_TEXTURE_2D);
        } else {
            sceGuDisable(GU_TEXTURE_2D);
        }
    }

    *current = *state;
}

static int PSP_RunCommandQueue(SDL_Renderer *renderer, SDL_RenderCommand *cmd, void *vertices, size_t vertsize)
{
    PSP_RenderData *data = (PSP_RenderData *)renderer->driverdata;
    Uint8 *gpumem = NULL;
    StartDrawing(renderer);

    /* note that before the renderer interface change, this would do extrememly small
       batches with sceGuGetMemory()--a few vertices at a time--and it's not clear that
       this won't fail if you try to push 100,000 draw calls in a single batch.
       I don't know what the limits on PSP hardware are. It might be useful to have
       rendering backends report a reasonable maximum, so the higher level can flush
       if we appear to be exceeding that. */
    gpumem = (Uint8 *)sceGuGetMemory(vertsize);
    if (!gpumem) {
        return SDL_SetError("Couldn't obtain a %d-byte vertex buffer!", (int)vertsize);
    }
    SDL_memcpy(gpumem, vertices, vertsize);

    while (cmd) {
        switch (cmd->command) {
        case SDL_RENDERCMD_SETDRAWCOLOR:
        {
            break; /* !!! FIXME: we could cache drawstate like color */
        }

        case SDL_RENDERCMD_SETVIEWPORT:
        {
            SDL_Rect *viewport = &cmd->data.viewport.rect;
            sceGuOffset(2048 - (viewport->w >> 1), 2048 - (viewport->h >> 1));
            sceGuViewport(2048, 2048, viewport->w, viewport->h);
            sceGuScissor(viewport->x, viewport->y, viewport->w, viewport->h);
            /* FIXME: We need to update the clip rect too, see https://github.com/libsdl-org/SDL/issues/9094 */
            break;
        }

        case SDL_RENDERCMD_SETCLIPRECT:
        {
            const SDL_Rect *rect = &cmd->data.cliprect.rect;
            if (cmd->data.cliprect.enabled) {
                sceGuEnable(GU_SCISSOR_TEST);
                sceGuScissor(rect->x, rect->y, rect->w, rect->h);
            } else {
                sceGuDisable(GU_SCISSOR_TEST);
            }
            break;
        }

        case SDL_RENDERCMD_CLEAR:
        {
            const Uint8 r = cmd->data.color.r;
            const Uint8 g = cmd->data.color.g;
            const Uint8 b = cmd->data.color.b;
            const Uint8 a = cmd->data.color.a;
            sceGuClearColor(GU_RGBA(r, g, b, a));
            sceGuClearStencil(a);
            sceGuClear(GU_COLOR_BUFFER_BIT | GU_STENCIL_BUFFER_BIT);
            break;
        }

        case SDL_RENDERCMD_DRAW_POINTS:
        {
            const size_t count = cmd->data.draw.count;
            const VertV *verts = (VertV *)(gpumem + cmd->data.draw.first);
            const Uint8 r = cmd->data.draw.r;
            const Uint8 g = cmd->data.draw.g;
            const Uint8 b = cmd->data.draw.b;
            const Uint8 a = cmd->data.draw.a;
            PSP_BlendState state = {
                .color = GU_RGBA(r, g, b, a),
                .texture = NULL,
                .mode = cmd->data.draw.blend,
                .shadeModel = GU_FLAT
            };
            PSP_SetBlendState(data, &state);
            sceGuDrawArray(GU_POINTS, GU_VERTEX_32BITF | GU_TRANSFORM_2D, count, 0, verts);
            break;
        }

        case SDL_RENDERCMD_DRAW_LINES:
        {
            const size_t count = cmd->data.draw.count;
            const VertV *verts = (VertV *)(gpumem + cmd->data.draw.first);
            const Uint8 r = cmd->data.draw.r;
            const Uint8 g = cmd->data.draw.g;
            const Uint8 b = cmd->data.draw.b;
            const Uint8 a = cmd->data.draw.a;
            PSP_BlendState state = {
                .color = GU_RGBA(r, g, b, a),
                .texture = NULL,
                .mode = cmd->data.draw.blend,
                .shadeModel = GU_FLAT
            };
            PSP_SetBlendState(data, &state);
            sceGuDrawArray(GU_LINE_STRIP, GU_VERTEX_32BITF | GU_TRANSFORM_2D, count, 0, verts);
            break;
        }

        case SDL_RENDERCMD_FILL_RECTS:
        {
            const size_t count = cmd->data.draw.count;
            const VertV *verts = (VertV *)(gpumem + cmd->data.draw.first);
            const Uint8 r = cmd->data.draw.r;
            const Uint8 g = cmd->data.draw.g;
            const Uint8 b = cmd->data.draw.b;
            const Uint8 a = cmd->data.draw.a;
            PSP_BlendState state = {
                .color = GU_RGBA(r, g, b, a),
                .texture = NULL,
                .mode = cmd->data.draw.blend,
                .shadeModel = GU_FLAT
            };
            PSP_SetBlendState(data, &state);
            sceGuDrawArray(GU_SPRITES, GU_VERTEX_32BITF | GU_TRANSFORM_2D, 2 * count, 0, verts);
            break;
        }

        case SDL_RENDERCMD_COPY:
        {
            const size_t count = cmd->data.draw.count;
            const VertTV *verts = (VertTV *)(gpumem + cmd->data.draw.first);
            const Uint8 a = cmd->data.draw.a;
            const Uint8 r = cmd->data.draw.r;
            const Uint8 g = cmd->data.draw.g;
            const Uint8 b = cmd->data.draw.b;
            PSP_BlendState state = {
                .color = GU_RGBA(r, g, b, a),
                .texture = cmd->data.draw.texture,
                .mode = cmd->data.draw.blend,
                .shadeModel = GU_SMOOTH
            };
            PSP_SetBlendState(data, &state);
            sceGuDrawArray(GU_SPRITES, GU_TEXTURE_32BITF | GU_VERTEX_32BITF | GU_TRANSFORM_2D, 2 * count, 0, verts);
            break;
        }

        case SDL_RENDERCMD_COPY_EX:
        {
            const VertTV *verts = (VertTV *)(gpumem + cmd->data.draw.first);
            const Uint8 a = cmd->data.draw.a;
            const Uint8 r = cmd->data.draw.r;
            const Uint8 g = cmd->data.draw.g;
            const Uint8 b = cmd->data.draw.b;
            PSP_BlendState state = {
                .color = GU_RGBA(r, g, b, a),
                .texture = cmd->data.draw.texture,
                .mode = cmd->data.draw.blend,
                .shadeModel = GU_SMOOTH
            };
            PSP_SetBlendState(data, &state);
            sceGuDrawArray(GU_TRIANGLE_FAN, GU_TEXTURE_32BITF | GU_VERTEX_32BITF | GU_TRANSFORM_2D, 4, 0, verts);
            break;
        }

        case SDL_RENDERCMD_GEOMETRY:
        {
            const size_t count = cmd->data.draw.count;
            if (!cmd->data.draw.texture) {
                const VertCV *verts = (VertCV *)(gpumem + cmd->data.draw.first);
                sceGuDisable(GU_TEXTURE_2D);
                /* In GU_SMOOTH mode */
                sceGuDrawArray(GU_TRIANGLES, GU_COLOR_8888 | GU_VERTEX_32BITF | GU_TRANSFORM_2D, count, 0, verts);
                sceGuEnable(GU_TEXTURE_2D);
            } else {
                const VertTCV *verts = (VertTCV *)(gpumem + cmd->data.draw.first);
                const Uint8 a = cmd->data.draw.a;
                const Uint8 r = cmd->data.draw.r;
                const Uint8 g = cmd->data.draw.g;
                const Uint8 b = cmd->data.draw.b;
                PSP_BlendState state = {
                    .color = GU_RGBA(r, g, b, a),
                    .texture = NULL,
                    .mode = cmd->data.draw.blend,
                    .shadeModel = GU_FLAT
                };
                TextureActivate(cmd->data.draw.texture);
                PSP_SetBlendState(data, &state);
                sceGuDrawArray(GU_TRIANGLES, GU_TEXTURE_32BITF | GU_COLOR_8888 | GU_VERTEX_32BITF | GU_TRANSFORM_2D, count, 0, verts);
            }
            break;
        }

        case SDL_RENDERCMD_NO_OP:
            break;
        }

        cmd = cmd->next;
    }

    return 0;
}

static int PSP_RenderReadPixels(SDL_Renderer *renderer, const SDL_Rect *rect,
                                Uint32 pixel_format, void *pixels, int pitch)
{
    return SDL_Unsupported();
}

static int PSP_RenderPresent(SDL_Renderer *renderer)
{
    PSP_RenderData *data = (PSP_RenderData *)renderer->driverdata;
    if (!data->displayListAvail) {
        return -1;
    }

    data->displayListAvail = SDL_FALSE;
    sceGuFinish();
    sceGuSync(0, 0);

    if ((data->vsync) && (data->vblank_not_reached)) {
        sceDisplayWaitVblankStart();
    }
    data->vblank_not_reached = SDL_TRUE;

    data->backbuffer = data->frontbuffer;
    data->frontbuffer = vabsptr(sceGuSwapBuffers());

    return 0;
}

static void PSP_DestroyTexture(SDL_Renderer *renderer, SDL_Texture *texture)
{
    PSP_RenderData *renderdata = (PSP_RenderData *)renderer->driverdata;
    PSP_TextureData *psp_texture = (PSP_TextureData *)texture->driverdata;

    if (!renderdata) {
        return;
    }

    if (!psp_texture) {
        return;
    }

    LRUTargetRemove(renderdata, psp_texture);
    TextureStorageFree(psp_texture->data);
    SDL_free(psp_texture);
    texture->driverdata = NULL;
}

static void PSP_DestroyRenderer(SDL_Renderer *renderer)
{
    PSP_RenderData *data = (PSP_RenderData *)renderer->driverdata;
    if (data) {
        if (!data->initialized) {
            return;
        }

        StartDrawing(renderer);

        sceKernelDisableSubIntr(PSP_VBLANK_INT, 0);
        sceKernelReleaseSubIntrHandler(PSP_VBLANK_INT, 0);
        sceDisplayWaitVblankStart();
        sceGuDisplay(GU_FALSE);
        sceGuTerm();
        vfree(data->backbuffer);
        vfree(data->frontbuffer);

        data->initialized = SDL_FALSE;
        data->displayListAvail = SDL_FALSE;
        SDL_free(data);
    }
    SDL_free(renderer);
}

static int PSP_SetVSync(SDL_Renderer *renderer, const int vsync)
{
    PSP_RenderData *data = renderer->driverdata;
    data->vsync = vsync;
    return 0;
}

SDL_Renderer *PSP_CreateRenderer(SDL_Window *window, Uint32 flags)
{

    SDL_Renderer *renderer;
    PSP_RenderData *data;
    int pixelformat;
    void *doublebuffer = NULL;

    renderer = (SDL_Renderer *)SDL_calloc(1, sizeof(*renderer));
    if (!renderer) {
        SDL_OutOfMemory();
        return NULL;
    }

    data = (PSP_RenderData *)SDL_calloc(1, sizeof(*data));
    if (!data) {
        PSP_DestroyRenderer(renderer);
        SDL_OutOfMemory();
        return NULL;
    }

    renderer->WindowEvent = PSP_WindowEvent;
    renderer->CreateTexture = PSP_CreateTexture;
    renderer->UpdateTexture = PSP_UpdateTexture;
    renderer->LockTexture = PSP_LockTexture;
    renderer->UnlockTexture = PSP_UnlockTexture;
    renderer->SetTextureScaleMode = PSP_SetTextureScaleMode;
    renderer->SetRenderTarget = PSP_SetRenderTarget;
    renderer->QueueSetViewport = PSP_QueueSetViewport;
    renderer->QueueSetDrawColor = PSP_QueueSetViewport; /* SetViewport and SetDrawColor are (currently) no-ops. */
    renderer->QueueDrawPoints = PSP_QueueDrawPoints;
    renderer->QueueDrawLines = PSP_QueueDrawPoints; /* lines and points queue vertices the same way. */
    renderer->QueueGeometry = PSP_QueueGeometry;
    renderer->QueueFillRects = PSP_QueueFillRects;
    renderer->QueueCopy = PSP_QueueCopy;
    renderer->QueueCopyEx = PSP_QueueCopyEx;
    renderer->RunCommandQueue = PSP_RunCommandQueue;
    renderer->RenderReadPixels = PSP_RenderReadPixels;
    renderer->RenderPresent = PSP_RenderPresent;
    renderer->DestroyTexture = PSP_DestroyTexture;
    renderer->DestroyRenderer = PSP_DestroyRenderer;
    renderer->SetVSync = PSP_SetVSync;
    renderer->info = PSP_RenderDriver.info;
    renderer->info.flags = (SDL_RENDERER_ACCELERATED | SDL_RENDERER_TARGETTEXTURE);
    renderer->driverdata = data;
    renderer->window = window;

    data->initialized = SDL_TRUE;
    data->most_recent_target = NULL;
    data->least_recent_target = NULL;

    if (flags & SDL_RENDERER_PRESENTVSYNC) {
        data->vsync = SDL_TRUE;
    } else {
        data->vsync = SDL_FALSE;
    }

    pixelformat = PixelFormatToPSPFMT(SDL_GetWindowPixelFormat(window));
    switch (pixelformat) {
    case GU_PSM_4444:
    case GU_PSM_5650:
    case GU_PSM_5551:
        data->bpp = 2;
        data->psm = pixelformat;
        break;
    default:
        data->bpp = 4;
        data->psm = GU_PSM_8888;
        break;
    }

    doublebuffer = vramalloc(PSP_FRAME_BUFFER_SIZE * data->bpp * 2);
    data->backbuffer = doublebuffer;
    data->frontbuffer = ((uint8_t *)doublebuffer) + PSP_FRAME_BUFFER_SIZE * data->bpp;

    sceGuInit();
    /* setup GU */
    sceGuStart(GU_DIRECT, DisplayList);
    sceGuDrawBuffer(data->psm, vrelptr(data->frontbuffer), PSP_FRAME_BUFFER_WIDTH);
    sceGuDispBuffer(PSP_SCREEN_WIDTH, PSP_SCREEN_HEIGHT, vrelptr(data->backbuffer), PSP_FRAME_BUFFER_WIDTH);

    sceGuOffset(2048 - (PSP_SCREEN_WIDTH >> 1), 2048 - (PSP_SCREEN_HEIGHT >> 1));
    sceGuViewport(2048, 2048, PSP_SCREEN_WIDTH, PSP_SCREEN_HEIGHT);

    sceGuDisable(GU_DEPTH_TEST);

    /* Scissoring */
    sceGuScissor(0, 0, PSP_SCREEN_WIDTH, PSP_SCREEN_HEIGHT);
    sceGuEnable(GU_SCISSOR_TEST);

    /* Backface culling */
    /*
    FIXME: Culling probably un-needed ? It can conflict with SDL_RENDERCMD_GEOMETRY
    sceGuFrontFace(GU_CCW);
    sceGuEnable(GU_CULL_FACE);
    */

    // Setup initial blend state
    ResetBlendState(&data->blendState);

    sceGuFinish();
    sceGuSync(0, 0);
    sceDisplayWaitVblankStartCB();
    sceGuDisplay(GU_TRUE);

    /* Improve performance when VSYC is enabled and it is not reaching the 60 FPS */
    data->vblank_not_reached = SDL_TRUE;
    sceKernelRegisterSubIntrHandler(PSP_VBLANK_INT, 0, psp_on_vblank, data);
    sceKernelEnableSubIntr(PSP_VBLANK_INT, 0);

    return renderer;
}

SDL_RenderDriver PSP_RenderDriver = {
    .CreateRenderer = PSP_CreateRenderer,
    .info = {
        .name = "PSP",
        .flags = SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC | SDL_RENDERER_TARGETTEXTURE,
        .num_texture_formats = 4,
        .texture_formats = {
            [0] = SDL_PIXELFORMAT_BGR565,
            [1] = SDL_PIXELFORMAT_ABGR1555,
            [2] = SDL_PIXELFORMAT_ABGR4444,
            [3] = SDL_PIXELFORMAT_ABGR8888,
        },
        .max_texture_width = 512,
        .max_texture_height = 512,
    }
};

#endif /* SDL_VIDEO_RENDER_PSP */

/* vi: set ts=4 sw=4 expandtab: */
