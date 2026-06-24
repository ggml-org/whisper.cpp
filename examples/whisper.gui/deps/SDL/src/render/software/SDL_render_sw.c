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

#if SDL_VIDEO_RENDER_SW

#include "../SDL_sysrender.h"
#include "SDL_render_sw_c.h"
#include "SDL_hints.h"

#include "SDL_draw.h"
#include "SDL_blendfillrect.h"
#include "SDL_blendline.h"
#include "SDL_blendpoint.h"
#include "SDL_drawline.h"
#include "SDL_drawpoint.h"
#include "SDL_rotate.h"
#include "SDL_triangle.h"

/* SDL surface based renderer implementation */

typedef struct
{
    const SDL_Rect *viewport;
    const SDL_Rect *cliprect;
    SDL_bool surface_cliprect_dirty;
} SW_DrawStateCache;

typedef struct
{
    SDL_Surface *surface;
    SDL_Surface *window;
} SW_RenderData;

static SDL_Surface *SW_ActivateRenderer(SDL_Renderer *renderer)
{
    SW_RenderData *data = (SW_RenderData *)renderer->driverdata;

    if (!data->surface) {
        data->surface = data->window;
    }
    if (!data->surface) {
        SDL_Surface *surface = SDL_GetWindowSurface(renderer->window);
        if (surface) {
            data->surface = data->window = surface;
        }
    }
    return data->surface;
}

static void SW_WindowEvent(SDL_Renderer *renderer, const SDL_WindowEvent *event)
{
    SW_RenderData *data = (SW_RenderData *)renderer->driverdata;

    if (event->event == SDL_WINDOWEVENT_SIZE_CHANGED) {
        data->surface = NULL;
        data->window = NULL;
    }
}

static int SW_GetOutputSize(SDL_Renderer *renderer, int *w, int *h)
{
    SW_RenderData *data = (SW_RenderData *)renderer->driverdata;

    if (data->surface) {
        if (w) {
            *w = data->surface->w;
        }
        if (h) {
            *h = data->surface->h;
        }
        return 0;
    }

    if (renderer->window) {
        SDL_GetWindowSizeInPixels(renderer->window, w, h);
        return 0;
    }

    return SDL_SetError("Software renderer doesn't have an output surface");
}

static int SW_CreateTexture(SDL_Renderer *renderer, SDL_Texture *texture)
{
    int bpp;
    Uint32 Rmask, Gmask, Bmask, Amask;

    if (!SDL_PixelFormatEnumToMasks(texture->format, &bpp, &Rmask, &Gmask, &Bmask, &Amask)) {
        return SDL_SetError("Unknown texture format");
    }

    texture->driverdata =
        SDL_CreateRGBSurface(0, texture->w, texture->h, bpp, Rmask, Gmask,
                             Bmask, Amask);
    SDL_SetSurfaceColorMod(texture->driverdata, texture->color.r, texture->color.g, texture->color.b);
    SDL_SetSurfaceAlphaMod(texture->driverdata, texture->color.a);
    SDL_SetSurfaceBlendMode(texture->driverdata, texture->blendMode);

    /* Only RLE encode textures without an alpha channel since the RLE coder
     * discards the color values of pixels with an alpha value of zero.
     */
    if (texture->access == SDL_TEXTUREACCESS_STATIC && !Amask) {
        SDL_SetSurfaceRLE(texture->driverdata, 1);
    }

    if (!texture->driverdata) {
        return -1;
    }
    return 0;
}

static int SW_UpdateTexture(SDL_Renderer *renderer, SDL_Texture *texture,
                            const SDL_Rect *rect, const void *pixels, int pitch)
{
    SDL_Surface *surface = (SDL_Surface *)texture->driverdata;
    Uint8 *src, *dst;
    int row;
    size_t length;

    if (SDL_MUSTLOCK(surface)) {
        SDL_LockSurface(surface);
    }
    src = (Uint8 *)pixels;
    dst = (Uint8 *)surface->pixels +
          rect->y * surface->pitch +
          rect->x * surface->format->BytesPerPixel;
    length = (size_t)rect->w * surface->format->BytesPerPixel;
    for (row = 0; row < rect->h; ++row) {
        SDL_memcpy(dst, src, length);
        src += pitch;
        dst += surface->pitch;
    }
    if (SDL_MUSTLOCK(surface)) {
        SDL_UnlockSurface(surface);
    }
    return 0;
}

static int SW_LockTexture(SDL_Renderer *renderer, SDL_Texture *texture,
                          const SDL_Rect *rect, void **pixels, int *pitch)
{
    SDL_Surface *surface = (SDL_Surface *)texture->driverdata;

    *pixels =
        (void *)((Uint8 *)surface->pixels + rect->y * surface->pitch +
                 rect->x * surface->format->BytesPerPixel);
    *pitch = surface->pitch;
    return 0;
}

static void SW_UnlockTexture(SDL_Renderer *renderer, SDL_Texture *texture)
{
}

static void SW_SetTextureScaleMode(SDL_Renderer *renderer, SDL_Texture *texture, SDL_ScaleMode scaleMode)
{
}

static int SW_SetRenderTarget(SDL_Renderer *renderer, SDL_Texture *texture)
{
    SW_RenderData *data = (SW_RenderData *)renderer->driverdata;

    if (texture) {
        data->surface = (SDL_Surface *)texture->driverdata;
    } else {
        data->surface = data->window;
    }
    return 0;
}

static int SW_QueueSetViewport(SDL_Renderer *renderer, SDL_RenderCommand *cmd)
{
    return 0; /* nothing to do in this backend. */
}

static int SW_QueueDrawPoints(SDL_Renderer *renderer, SDL_RenderCommand *cmd, const SDL_FPoint *points, int count)
{
    SDL_Point *verts = (SDL_Point *)SDL_AllocateRenderVertices(renderer, count * sizeof(SDL_Point), 0, &cmd->data.draw.first);
    int i;

    if (!verts) {
        return -1;
    }

    cmd->data.draw.count = count;

    for (i = 0; i < count; i++, verts++, points++) {
        verts->x = (int)points->x;
        verts->y = (int)points->y;
    }

    return 0;
}

static int SW_QueueFillRects(SDL_Renderer *renderer, SDL_RenderCommand *cmd, const SDL_FRect *rects, int count)
{
    SDL_Rect *verts = (SDL_Rect *)SDL_AllocateRenderVertices(renderer, count * sizeof(SDL_Rect), 0, &cmd->data.draw.first);
    int i;

    if (!verts) {
        return -1;
    }

    cmd->data.draw.count = count;

    for (i = 0; i < count; i++, verts++, rects++) {
        verts->x = (int)rects->x;
        verts->y = (int)rects->y;
        verts->w = SDL_max((int)rects->w, 1);
        verts->h = SDL_max((int)rects->h, 1);
    }

    return 0;
}

static int SW_QueueCopy(SDL_Renderer *renderer, SDL_RenderCommand *cmd, SDL_Texture *texture,
                        const SDL_Rect *srcrect, const SDL_FRect *dstrect)
{
    SDL_Rect *verts = (SDL_Rect *)SDL_AllocateRenderVertices(renderer, 2 * sizeof(SDL_Rect), 0, &cmd->data.draw.first);

    if (!verts) {
        return -1;
    }

    cmd->data.draw.count = 1;

    SDL_copyp(verts, srcrect);
    verts++;

    verts->x = (int)dstrect->x;
    verts->y = (int)dstrect->y;
    verts->w = (int)dstrect->w;
    verts->h = (int)dstrect->h;

    return 0;
}

typedef struct CopyExData
{
    SDL_Rect srcrect;
    SDL_Rect dstrect;
    double angle;
    SDL_FPoint center;
    SDL_RendererFlip flip;
    float scale_x;
    float scale_y;
} CopyExData;

static int SW_QueueCopyEx(SDL_Renderer *renderer, SDL_RenderCommand *cmd, SDL_Texture *texture,
                          const SDL_Rect *srcrect, const SDL_FRect *dstrect,
                          const double angle, const SDL_FPoint *center, const SDL_RendererFlip flip, float scale_x, float scale_y)
{
    CopyExData *verts = (CopyExData *)SDL_AllocateRenderVertices(renderer, sizeof(CopyExData), 0, &cmd->data.draw.first);

    if (!verts) {
        return -1;
    }

    cmd->data.draw.count = 1;

    SDL_copyp(&verts->srcrect, srcrect);

    verts->dstrect.x = (int)dstrect->x;
    verts->dstrect.y = (int)dstrect->y;
    verts->dstrect.w = (int)dstrect->w;
    verts->dstrect.h = (int)dstrect->h;
    verts->angle = angle;
    SDL_copyp(&verts->center, center);
    verts->flip = flip;
    verts->scale_x = scale_x;
    verts->scale_y = scale_y;

    return 0;
}

static int Blit_to_Screen(SDL_Surface *src, SDL_Rect *srcrect, SDL_Surface *surface, SDL_Rect *dstrect,
                          float scale_x, float scale_y, SDL_ScaleMode scaleMode)
{
    int retval;
    /* Renderer scaling, if needed */
    if (scale_x != 1.0f || scale_y != 1.0f) {
        SDL_Rect r;
        r.x = (int)((float)dstrect->x * scale_x);
        r.y = (int)((float)dstrect->y * scale_y);
        r.w = (int)((float)dstrect->w * scale_x);
        r.h = (int)((float)dstrect->h * scale_y);
        retval = SDL_PrivateUpperBlitScaled(src, srcrect, surface, &r, scaleMode);
    } else {
        retval = SDL_BlitSurface(src, srcrect, surface, dstrect);
    }
    return retval;
}

static int SW_RenderCopyEx(SDL_Renderer *renderer, SDL_Surface *surface, SDL_Texture *texture,
                           const SDL_Rect *srcrect, const SDL_Rect *final_rect,
                           const double angle, const SDL_FPoint *center, const SDL_RendererFlip flip, float scale_x, float scale_y)
{
    SDL_Surface *src = (SDL_Surface *)texture->driverdata;
    SDL_Rect tmp_rect;
    SDL_Surface *src_clone, *src_rotated, *src_scaled;
    SDL_Surface *mask = NULL, *mask_rotated = NULL;
    int retval = 0;
    SDL_BlendMode blendmode;
    Uint8 alphaMod, rMod, gMod, bMod;
    int applyModulation = SDL_FALSE;
    int blitRequired = SDL_FALSE;
    int isOpaque = SDL_FALSE;

    if (!surface) {
        return -1;
    }

    tmp_rect.x = 0;
    tmp_rect.y = 0;
    tmp_rect.w = final_rect->w;
    tmp_rect.h = final_rect->h;

    /* It is possible to encounter an RLE encoded surface here and locking it is
     * necessary because this code is going to access the pixel buffer directly.
     */
    if (SDL_MUSTLOCK(src)) {
        SDL_LockSurface(src);
    }

    /* Clone the source surface but use its pixel buffer directly.
     * The original source surface must be treated as read-only.
     */
    src_clone = SDL_CreateRGBSurfaceFrom(src->pixels, src->w, src->h, src->format->BitsPerPixel, src->pitch,
                                         src->format->Rmask, src->format->Gmask,
                                         src->format->Bmask, src->format->Amask);
    if (!src_clone) {
        if (SDL_MUSTLOCK(src)) {
            SDL_UnlockSurface(src);
        }
        return -1;
    }

    SDL_GetSurfaceBlendMode(src, &blendmode);
    SDL_GetSurfaceAlphaMod(src, &alphaMod);
    SDL_GetSurfaceColorMod(src, &rMod, &gMod, &bMod);

    /* SDLgfx_rotateSurface only accepts 32-bit surfaces with a 8888 layout. Everything else has to be converted. */
    if (src->format->BitsPerPixel != 32 || SDL_PIXELLAYOUT(src->format->format) != SDL_PACKEDLAYOUT_8888 || !src->format->Amask) {
        blitRequired = SDL_TRUE;
    }

    /* If scaling and cropping is necessary, it has to be taken care of before the rotation. */
    if (!(srcrect->w == final_rect->w && srcrect->h == final_rect->h && srcrect->x == 0 && srcrect->y == 0)) {
        blitRequired = SDL_TRUE;
    }

    /* srcrect is not selecting the whole src surface, so cropping is needed */
    if (!(srcrect->w == src->w && srcrect->h == src->h && srcrect->x == 0 && srcrect->y == 0)) {
        blitRequired = SDL_TRUE;
    }

    /* The color and alpha modulation has to be applied before the rotation when using the NONE, MOD or MUL blend modes. */
    if ((blendmode == SDL_BLENDMODE_NONE || blendmode == SDL_BLENDMODE_MOD || blendmode == SDL_BLENDMODE_MUL) && (alphaMod & rMod & gMod & bMod) != 255) {
        applyModulation = SDL_TRUE;
        SDL_SetSurfaceAlphaMod(src_clone, alphaMod);
        SDL_SetSurfaceColorMod(src_clone, rMod, gMod, bMod);
    }

    /* Opaque surfaces are much easier to handle with the NONE blend mode. */
    if (blendmode == SDL_BLENDMODE_NONE && !src->format->Amask && alphaMod == 255) {
        isOpaque = SDL_TRUE;
    }

    /* The NONE blend mode requires a mask for non-opaque surfaces. This mask will be used
     * to clear the pixels in the destination surface. The other steps are explained below.
     */
    if (blendmode == SDL_BLENDMODE_NONE && !isOpaque) {
        mask = SDL_CreateRGBSurface(0, final_rect->w, final_rect->h, 32,
                                    0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000);
        if (!mask) {
            retval = -1;
        } else {
            SDL_SetSurfaceBlendMode(mask, SDL_BLENDMODE_MOD);
        }
    }

    /* Create a new surface should there be a format mismatch or if scaling, cropping,
     * or modulation is required. It's possible to use the source surface directly otherwise.
     */
    if (!retval && (blitRequired || applyModulation)) {
        SDL_Rect scale_rect = tmp_rect;
        src_scaled = SDL_CreateRGBSurface(0, final_rect->w, final_rect->h, 32,
                                          0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000);
        if (!src_scaled) {
            retval = -1;
        } else {
            SDL_SetSurfaceBlendMode(src_clone, SDL_BLENDMODE_NONE);
            retval = SDL_PrivateUpperBlitScaled(src_clone, srcrect, src_scaled, &scale_rect, texture->scaleMode);
            SDL_FreeSurface(src_clone);
            src_clone = src_scaled;
            src_scaled = NULL;
        }
    }

    /* SDLgfx_rotateSurface is going to make decisions depending on the blend mode. */
    SDL_SetSurfaceBlendMode(src_clone, blendmode);

    if (!retval) {
        SDL_Rect rect_dest;
        double cangle, sangle;

        SDLgfx_rotozoomSurfaceSizeTrig(tmp_rect.w, tmp_rect.h, angle, center,
                                       &rect_dest, &cangle, &sangle);
        src_rotated = SDLgfx_rotateSurface(src_clone, angle,
                                           (texture->scaleMode == SDL_ScaleModeNearest) ? 0 : 1, flip & SDL_FLIP_HORIZONTAL, flip & SDL_FLIP_VERTICAL,
                                           &rect_dest, cangle, sangle, center);
        if (!src_rotated) {
            retval = -1;
        }
        if (!retval && mask) {
            /* The mask needed for the NONE blend mode gets rotated with the same parameters. */
            mask_rotated = SDLgfx_rotateSurface(mask, angle,
                                                SDL_FALSE, 0, 0,
                                                &rect_dest, cangle, sangle, center);
            if (!mask_rotated) {
                retval = -1;
            }
        }
        if (!retval) {

            tmp_rect.x = final_rect->x + rect_dest.x;
            tmp_rect.y = final_rect->y + rect_dest.y;
            tmp_rect.w = rect_dest.w;
            tmp_rect.h = rect_dest.h;

            /* The NONE blend mode needs some special care with non-opaque surfaces.
             * Other blend modes or opaque surfaces can be blitted directly.
             */
            if (blendmode != SDL_BLENDMODE_NONE || isOpaque) {
                if (applyModulation == SDL_FALSE) {
                    /* If the modulation wasn't already applied, make it happen now. */
                    SDL_SetSurfaceAlphaMod(src_rotated, alphaMod);
                    SDL_SetSurfaceColorMod(src_rotated, rMod, gMod, bMod);
                }
                /* Renderer scaling, if needed */
                retval = Blit_to_Screen(src_rotated, NULL, surface, &tmp_rect, scale_x, scale_y, texture->scaleMode);
            } else {
                /* The NONE blend mode requires three steps to get the pixels onto the destination surface.
                 * First, the area where the rotated pixels will be blitted to get set to zero.
                 * This is accomplished by simply blitting a mask with the NONE blend mode.
                 * The colorkey set by the rotate function will discard the correct pixels.
                 */
                SDL_Rect mask_rect = tmp_rect;
                SDL_SetSurfaceBlendMode(mask_rotated, SDL_BLENDMODE_NONE);
                /* Renderer scaling, if needed */
                retval = Blit_to_Screen(mask_rotated, NULL, surface, &mask_rect, scale_x, scale_y, texture->scaleMode);
                if (!retval) {
                    /* The next step copies the alpha value. This is done with the BLEND blend mode and
                     * by modulating the source colors with 0. Since the destination is all zeros, this
                     * will effectively set the destination alpha to the source alpha.
                     */
                    SDL_SetSurfaceColorMod(src_rotated, 0, 0, 0);
                    mask_rect = tmp_rect;
                    /* Renderer scaling, if needed */
                    retval = Blit_to_Screen(src_rotated, NULL, surface, &mask_rect, scale_x, scale_y, texture->scaleMode);
                    if (!retval) {
                        /* The last step gets the color values in place. The ADD blend mode simply adds them to
                         * the destination (where the color values are all zero). However, because the ADD blend
                         * mode modulates the colors with the alpha channel, a surface without an alpha mask needs
                         * to be created. This makes all source pixels opaque and the colors get copied correctly.
                         */
                        SDL_Surface *src_rotated_rgb;
                        src_rotated_rgb = SDL_CreateRGBSurfaceFrom(src_rotated->pixels, src_rotated->w, src_rotated->h,
                                                                   src_rotated->format->BitsPerPixel, src_rotated->pitch,
                                                                   src_rotated->format->Rmask, src_rotated->format->Gmask,
                                                                   src_rotated->format->Bmask, 0);
                        if (!src_rotated_rgb) {
                            retval = -1;
                        } else {
                            SDL_SetSurfaceBlendMode(src_rotated_rgb, SDL_BLENDMODE_ADD);
                            /* Renderer scaling, if needed */
                            retval = Blit_to_Screen(src_rotated_rgb, NULL, surface, &tmp_rect, scale_x, scale_y, texture->scaleMode);
                            SDL_FreeSurface(src_rotated_rgb);
                        }
                    }
                }
                SDL_FreeSurface(mask_rotated);
            }
            if (src_rotated) {
                SDL_FreeSurface(src_rotated);
            }
        }
    }

    if (SDL_MUSTLOCK(src)) {
        SDL_UnlockSurface(src);
    }
    if (mask) {
        SDL_FreeSurface(mask);
    }
    if (src_clone) {
        SDL_FreeSurface(src_clone);
    }
    return retval;
}

typedef struct GeometryFillData
{
    SDL_Point dst;
    SDL_Color color;
} GeometryFillData;

typedef struct GeometryCopyData
{
    SDL_Point src;
    SDL_Point dst;
    SDL_Color color;
} GeometryCopyData;

static int SW_QueueGeometry(SDL_Renderer *renderer, SDL_RenderCommand *cmd, SDL_Texture *texture,
                            const float *xy, int xy_stride, const SDL_Color *color, int color_stride, const float *uv, int uv_stride,
                            int num_vertices, const void *indices, int num_indices, int size_indices,
                            float scale_x, float scale_y)
{
    int i;
    int count = indices ? num_indices : num_vertices;
    void *verts;
    size_t sz = texture ? sizeof(GeometryCopyData) : sizeof(GeometryFillData);

    verts = SDL_AllocateRenderVertices(renderer, count * sz, 0, &cmd->data.draw.first);
    if (!verts) {
        return -1;
    }

    cmd->data.draw.count = count;
    size_indices = indices ? size_indices : 0;

    if (texture) {
        GeometryCopyData *ptr = (GeometryCopyData *)verts;
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

            ptr->src.x = (int)(uv_[0] * texture->w);
            ptr->src.y = (int)(uv_[1] * texture->h);

            ptr->dst.x = (int)(xy_[0] * scale_x);
            ptr->dst.y = (int)(xy_[1] * scale_y);
            trianglepoint_2_fixedpoint(&ptr->dst);

            ptr->color = col_;

            ptr++;
        }
    } else {
        GeometryFillData *ptr = (GeometryFillData *)verts;

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

            ptr->dst.x = (int)(xy_[0] * scale_x);
            ptr->dst.y = (int)(xy_[1] * scale_y);
            trianglepoint_2_fixedpoint(&ptr->dst);

            ptr->color = col_;

            ptr++;
        }
    }
    return 0;
}

static void PrepTextureForCopy(const SDL_RenderCommand *cmd)
{
    const Uint8 r = cmd->data.draw.r;
    const Uint8 g = cmd->data.draw.g;
    const Uint8 b = cmd->data.draw.b;
    const Uint8 a = cmd->data.draw.a;
    const SDL_BlendMode blend = cmd->data.draw.blend;
    SDL_Texture *texture = cmd->data.draw.texture;
    SDL_Surface *surface = (SDL_Surface *)texture->driverdata;
    const SDL_bool colormod = ((r & g & b) != 0xFF);
    const SDL_bool alphamod = (a != 0xFF);
    const SDL_bool blending = ((blend == SDL_BLENDMODE_ADD) || (blend == SDL_BLENDMODE_MOD) || (blend == SDL_BLENDMODE_MUL));

    if (colormod || alphamod || blending) {
        SDL_SetSurfaceRLE(surface, 0);
    }

    /* !!! FIXME: we can probably avoid some of these calls. */
    SDL_SetSurfaceColorMod(surface, r, g, b);
    SDL_SetSurfaceAlphaMod(surface, a);
    SDL_SetSurfaceBlendMode(surface, blend);
}

static void SetDrawState(SDL_Surface *surface, SW_DrawStateCache *drawstate)
{
    if (drawstate->surface_cliprect_dirty) {
        const SDL_Rect *viewport = drawstate->viewport;
        const SDL_Rect *cliprect = drawstate->cliprect;
        SDL_assert_release(viewport != NULL); /* the higher level should have forced a SDL_RENDERCMD_SETVIEWPORT */

        if (cliprect) {
            SDL_Rect clip_rect;
            clip_rect.x = cliprect->x + viewport->x;
            clip_rect.y = cliprect->y + viewport->y;
            clip_rect.w = cliprect->w;
            clip_rect.h = cliprect->h;
            SDL_IntersectRect(viewport, &clip_rect, &clip_rect);
            SDL_SetClipRect(surface, &clip_rect);
        } else {
            SDL_SetClipRect(surface, drawstate->viewport);
        }
        drawstate->surface_cliprect_dirty = SDL_FALSE;
    }
}

static int SW_RunCommandQueue(SDL_Renderer *renderer, SDL_RenderCommand *cmd, void *vertices, size_t vertsize)
{
    SDL_Surface *surface = SW_ActivateRenderer(renderer);
    SW_DrawStateCache drawstate;

    if (!surface) {
        return -1;
    }

    drawstate.viewport = NULL;
    drawstate.cliprect = NULL;
    drawstate.surface_cliprect_dirty = SDL_TRUE;

    while (cmd) {
        switch (cmd->command) {
            case SDL_RENDERCMD_SETDRAWCOLOR: {
                break;  /* Not used in this backend. */
            }

            case SDL_RENDERCMD_SETVIEWPORT: {
                drawstate.viewport = &cmd->data.viewport.rect;
                drawstate.surface_cliprect_dirty = SDL_TRUE;
                break;
            }

            case SDL_RENDERCMD_SETCLIPRECT: {
                drawstate.cliprect = cmd->data.cliprect.enabled ? &cmd->data.cliprect.rect : NULL;
                drawstate.surface_cliprect_dirty = SDL_TRUE;
                break;
            }

            case SDL_RENDERCMD_CLEAR: {
                const Uint8 r = cmd->data.color.r;
                const Uint8 g = cmd->data.color.g;
                const Uint8 b = cmd->data.color.b;
                const Uint8 a = cmd->data.color.a;
                /* By definition the clear ignores the clip rect */
                SDL_SetClipRect(surface, NULL);
                SDL_FillRect(surface, NULL, SDL_MapRGBA(surface->format, r, g, b, a));
                drawstate.surface_cliprect_dirty = SDL_TRUE;
                break;
            }

            case SDL_RENDERCMD_DRAW_POINTS: {
                const Uint8 r = cmd->data.draw.r;
                const Uint8 g = cmd->data.draw.g;
                const Uint8 b = cmd->data.draw.b;
                const Uint8 a = cmd->data.draw.a;
                const int count = (int) cmd->data.draw.count;
                SDL_Point *verts = (SDL_Point *) (((Uint8 *) vertices) + cmd->data.draw.first);
                const SDL_BlendMode blend = cmd->data.draw.blend;
                SetDrawState(surface, &drawstate);

                /* Apply viewport */
                if (drawstate.viewport && (drawstate.viewport->x || drawstate.viewport->y)) {
                    int i;
                    for (i = 0; i < count; i++) {
                        verts[i].x += drawstate.viewport->x;
                        verts[i].y += drawstate.viewport->y;
                    }
                }

                if (blend == SDL_BLENDMODE_NONE) {
                    SDL_DrawPoints(surface, verts, count, SDL_MapRGBA(surface->format, r, g, b, a));
                } else {
                    SDL_BlendPoints(surface, verts, count, blend, r, g, b, a);
                }
                break;
            }

            case SDL_RENDERCMD_DRAW_LINES: {
                const Uint8 r = cmd->data.draw.r;
                const Uint8 g = cmd->data.draw.g;
                const Uint8 b = cmd->data.draw.b;
                const Uint8 a = cmd->data.draw.a;
                const int count = (int) cmd->data.draw.count;
                SDL_Point *verts = (SDL_Point *) (((Uint8 *) vertices) + cmd->data.draw.first);
                const SDL_BlendMode blend = cmd->data.draw.blend;
                SetDrawState(surface, &drawstate);

                /* Apply viewport */
                if (drawstate.viewport && (drawstate.viewport->x || drawstate.viewport->y)) {
                    int i;
                    for (i = 0; i < count; i++) {
                        verts[i].x += drawstate.viewport->x;
                        verts[i].y += drawstate.viewport->y;
                    }
                }

                if (blend == SDL_BLENDMODE_NONE) {
                    SDL_DrawLines(surface, verts, count, SDL_MapRGBA(surface->format, r, g, b, a));
                } else {
                    SDL_BlendLines(surface, verts, count, blend, r, g, b, a);
                }
                break;
            }

            case SDL_RENDERCMD_FILL_RECTS: {
                const Uint8 r = cmd->data.draw.r;
                const Uint8 g = cmd->data.draw.g;
                const Uint8 b = cmd->data.draw.b;
                const Uint8 a = cmd->data.draw.a;
                const int count = (int) cmd->data.draw.count;
                SDL_Rect *verts = (SDL_Rect *) (((Uint8 *) vertices) + cmd->data.draw.first);
                const SDL_BlendMode blend = cmd->data.draw.blend;
                SetDrawState(surface, &drawstate);

                /* Apply viewport */
                if (drawstate.viewport && (drawstate.viewport->x || drawstate.viewport->y)) {
                    int i;
                    for (i = 0; i < count; i++) {
                        verts[i].x += drawstate.viewport->x;
                        verts[i].y += drawstate.viewport->y;
                    }
                }

                if (blend == SDL_BLENDMODE_NONE) {
                    SDL_FillRects(surface, verts, count, SDL_MapRGBA(surface->format, r, g, b, a));
                } else {
                    SDL_BlendFillRects(surface, verts, count, blend, r, g, b, a);
                }
                break;
            }

            case SDL_RENDERCMD_COPY: {
                SDL_Rect *verts = (SDL_Rect *) (((Uint8 *) vertices) + cmd->data.draw.first);
                const SDL_Rect *srcrect = verts;
                SDL_Rect *dstrect = verts + 1;
                SDL_Texture *texture = cmd->data.draw.texture;
                SDL_Surface *src = (SDL_Surface *) texture->driverdata;

                SetDrawState(surface, &drawstate);

                PrepTextureForCopy(cmd);

                /* Apply viewport */
                if (drawstate.viewport && (drawstate.viewport->x || drawstate.viewport->y)) {
                    dstrect->x += drawstate.viewport->x;
                    dstrect->y += drawstate.viewport->y;
                }

                if ( srcrect->w == dstrect->w && srcrect->h == dstrect->h ) {
                    SDL_BlitSurface(src, srcrect, surface, dstrect);
                } else {
                    /* If scaling is ever done, permanently disable RLE (which doesn't support scaling)
                     * to avoid potentially frequent RLE encoding/decoding.
                     */
                    SDL_SetSurfaceRLE(surface, 0);

                    /* Prevent to do scaling + clipping on viewport boundaries as it may lose proportion */
                    if (dstrect->x < 0 || dstrect->y < 0 || dstrect->x + dstrect->w > surface->w || dstrect->y + dstrect->h > surface->h) {
                        SDL_Surface *tmp = SDL_CreateRGBSurfaceWithFormat(0, dstrect->w, dstrect->h, 0, src->format->format);
                        /* Scale to an intermediate surface, then blit */
                        if (tmp) {
                            SDL_Rect r;
                            SDL_BlendMode blendmode;
                            Uint8 alphaMod, rMod, gMod, bMod;

                            SDL_GetSurfaceBlendMode(src, &blendmode);
                            SDL_GetSurfaceAlphaMod(src, &alphaMod);
                            SDL_GetSurfaceColorMod(src, &rMod, &gMod, &bMod);

                            r.x = 0;
                            r.y = 0;
                            r.w = dstrect->w;
                            r.h = dstrect->h;

                            SDL_SetSurfaceBlendMode(src, SDL_BLENDMODE_NONE);
                            SDL_SetSurfaceColorMod(src, 255, 255, 255);
                            SDL_SetSurfaceAlphaMod(src, 255);

                            SDL_PrivateUpperBlitScaled(src, srcrect, tmp, &r, texture->scaleMode);

                            SDL_SetSurfaceColorMod(tmp, rMod, gMod, bMod);
                            SDL_SetSurfaceAlphaMod(tmp, alphaMod);
                            SDL_SetSurfaceBlendMode(tmp, blendmode);

                            SDL_BlitSurface(tmp, NULL, surface, dstrect);
                            SDL_FreeSurface(tmp);
                            /* No need to set back r/g/b/a/blendmode to 'src' since it's done in PrepTextureForCopy() */
                        }
                    } else{
                        SDL_PrivateUpperBlitScaled(src, srcrect, surface, dstrect, texture->scaleMode);
                    }
                }
                break;
            }

            case SDL_RENDERCMD_COPY_EX: {
                CopyExData *copydata = (CopyExData *) (((Uint8 *) vertices) + cmd->data.draw.first);
                SetDrawState(surface, &drawstate);
                PrepTextureForCopy(cmd);

                /* Apply viewport */
                if (drawstate.viewport && (drawstate.viewport->x || drawstate.viewport->y)) {
                    copydata->dstrect.x += drawstate.viewport->x;
                    copydata->dstrect.y += drawstate.viewport->y;
                }

                SW_RenderCopyEx(renderer, surface, cmd->data.draw.texture, &copydata->srcrect,
                                &copydata->dstrect, copydata->angle, &copydata->center, copydata->flip,
                                copydata->scale_x, copydata->scale_y);
                break;
            }

            case SDL_RENDERCMD_GEOMETRY: {
                int i;
                SDL_Rect *verts = (SDL_Rect *) (((Uint8 *) vertices) + cmd->data.draw.first);
                const int count = (int) cmd->data.draw.count;
                SDL_Texture *texture = cmd->data.draw.texture;
                const SDL_BlendMode blend = cmd->data.draw.blend;

                SetDrawState(surface, &drawstate);

                if (texture) {
                    SDL_Surface *src = (SDL_Surface *) texture->driverdata;

                    GeometryCopyData *ptr = (GeometryCopyData *) verts;

                    PrepTextureForCopy(cmd);

                    /* Apply viewport */
                    if (drawstate.viewport && (drawstate.viewport->x || drawstate.viewport->y)) {
                        SDL_Point vp;
                        vp.x = drawstate.viewport->x;
                        vp.y = drawstate.viewport->y;
                        trianglepoint_2_fixedpoint(&vp);
                        for (i = 0; i < count; i++) {
                            ptr[i].dst.x += vp.x;
                            ptr[i].dst.y += vp.y;
                        }
                    }

                    for (i = 0; i < count; i += 3, ptr += 3) {
                        SDL_SW_BlitTriangle(
                                src,
                                &(ptr[0].src), &(ptr[1].src), &(ptr[2].src),
                                surface,
                                &(ptr[0].dst), &(ptr[1].dst), &(ptr[2].dst),
                                ptr[0].color, ptr[1].color, ptr[2].color);
                    }
                } else {
                    GeometryFillData *ptr = (GeometryFillData *) verts;

                    /* Apply viewport */
                    if (drawstate.viewport && (drawstate.viewport->x || drawstate.viewport->y)) {
                        SDL_Point vp;
                        vp.x = drawstate.viewport->x;
                        vp.y = drawstate.viewport->y;
                        trianglepoint_2_fixedpoint(&vp);
                        for (i = 0; i < count; i++) {
                            ptr[i].dst.x += vp.x;
                            ptr[i].dst.y += vp.y;
                        }
                    }

                    for (i = 0; i < count; i += 3, ptr += 3) {
                        SDL_SW_FillTriangle(surface, &(ptr[0].dst), &(ptr[1].dst), &(ptr[2].dst), blend, ptr[0].color, ptr[1].color, ptr[2].color);
                    }
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

static int SW_RenderReadPixels(SDL_Renderer *renderer, const SDL_Rect *rect,
                               Uint32 format, void *pixels, int pitch)
{
    SDL_Surface *surface = SW_ActivateRenderer(renderer);
    Uint32 src_format;
    void *src_pixels;

    if (!surface) {
        return -1;
    }

    /* NOTE: The rect is already adjusted according to the viewport by
     * SDL_RenderReadPixels.
     */

    if (rect->x < 0 || rect->x + rect->w > surface->w ||
        rect->y < 0 || rect->y + rect->h > surface->h) {
        return SDL_SetError("Tried to read outside of surface bounds");
    }

    src_format = surface->format->format;
    src_pixels = (void *)((Uint8 *)surface->pixels +
                          rect->y * surface->pitch +
                          rect->x * surface->format->BytesPerPixel);

    return SDL_ConvertPixels(rect->w, rect->h,
                             src_format, src_pixels, surface->pitch,
                             format, pixels, pitch);
}

static int SW_RenderPresent(SDL_Renderer *renderer)
{
    SDL_Window *window = renderer->window;

    if (!window) {
        return -1;
    }
    return SDL_UpdateWindowSurface(window);
}

static void SW_DestroyTexture(SDL_Renderer *renderer, SDL_Texture *texture)
{
    SDL_Surface *surface = (SDL_Surface *)texture->driverdata;

    SDL_FreeSurface(surface);
}

static void SW_DestroyRenderer(SDL_Renderer *renderer)
{
    SDL_Window *window = renderer->window;
    SW_RenderData *data = (SW_RenderData *)renderer->driverdata;

    if (window) {
        SDL_DestroyWindowSurface(window);
    }
    SDL_free(data);
    SDL_free(renderer);
}

SDL_Renderer *SW_CreateRendererForSurface(SDL_Surface *surface)
{
    SDL_Renderer *renderer;
    SW_RenderData *data;

    if (!surface) {
        SDL_InvalidParamError("surface");
        return NULL;
    }

    renderer = (SDL_Renderer *)SDL_calloc(1, sizeof(*renderer));
    if (!renderer) {
        SDL_OutOfMemory();
        return NULL;
    }

    data = (SW_RenderData *)SDL_calloc(1, sizeof(*data));
    if (!data) {
        SW_DestroyRenderer(renderer);
        SDL_OutOfMemory();
        return NULL;
    }
    data->surface = surface;
    data->window = surface;

    renderer->WindowEvent = SW_WindowEvent;
    renderer->GetOutputSize = SW_GetOutputSize;
    renderer->CreateTexture = SW_CreateTexture;
    renderer->UpdateTexture = SW_UpdateTexture;
    renderer->LockTexture = SW_LockTexture;
    renderer->UnlockTexture = SW_UnlockTexture;
    renderer->SetTextureScaleMode = SW_SetTextureScaleMode;
    renderer->SetRenderTarget = SW_SetRenderTarget;
    renderer->QueueSetViewport = SW_QueueSetViewport;
    renderer->QueueSetDrawColor = SW_QueueSetViewport; /* SetViewport and SetDrawColor are (currently) no-ops. */
    renderer->QueueDrawPoints = SW_QueueDrawPoints;
    renderer->QueueDrawLines = SW_QueueDrawPoints; /* lines and points queue vertices the same way. */
    renderer->QueueFillRects = SW_QueueFillRects;
    renderer->QueueCopy = SW_QueueCopy;
    renderer->QueueCopyEx = SW_QueueCopyEx;
    renderer->QueueGeometry = SW_QueueGeometry;
    renderer->RunCommandQueue = SW_RunCommandQueue;
    renderer->RenderReadPixels = SW_RenderReadPixels;
    renderer->RenderPresent = SW_RenderPresent;
    renderer->DestroyTexture = SW_DestroyTexture;
    renderer->DestroyRenderer = SW_DestroyRenderer;
    renderer->info = SW_RenderDriver.info;
    renderer->driverdata = data;

    return renderer;
}

static SDL_Renderer *SW_CreateRenderer(SDL_Window *window, Uint32 flags)
{
    const char *hint;
    SDL_Surface *surface;
    SDL_bool no_hint_set;

    /* Set the vsync hint based on our flags, if it's not already set */
    hint = SDL_GetHint(SDL_HINT_RENDER_VSYNC);
    if (!hint || !*hint) {
        no_hint_set = SDL_TRUE;
    } else {
        no_hint_set = SDL_FALSE;
    }

    if (no_hint_set) {
        SDL_SetHint(SDL_HINT_RENDER_VSYNC, (flags & SDL_RENDERER_PRESENTVSYNC) ? "1" : "0");
    }

    surface = SDL_GetWindowSurface(window);

    /* Reset the vsync hint if we set it above */
    if (no_hint_set) {
        SDL_SetHint(SDL_HINT_RENDER_VSYNC, "");
    }

    if (!surface) {
        return NULL;
    }
    return SW_CreateRendererForSurface(surface);
}

SDL_RenderDriver SW_RenderDriver = {
    SW_CreateRenderer,
    {
     "software",
     SDL_RENDERER_SOFTWARE | SDL_RENDERER_PRESENTVSYNC | SDL_RENDERER_TARGETTEXTURE,
     8,
     {
      SDL_PIXELFORMAT_ARGB8888,
      SDL_PIXELFORMAT_ABGR8888,
      SDL_PIXELFORMAT_RGBA8888,
      SDL_PIXELFORMAT_BGRA8888,
      SDL_PIXELFORMAT_RGB888,
      SDL_PIXELFORMAT_BGR888,
      SDL_PIXELFORMAT_RGB565,
      SDL_PIXELFORMAT_RGB555
     },
     0,
     0}
};

#endif /* SDL_VIDEO_RENDER_SW */

/* vi: set ts=4 sw=4 expandtab: */
