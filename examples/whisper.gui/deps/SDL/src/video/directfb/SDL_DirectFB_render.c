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
#include "SDL_DirectFB_window.h"
#include "SDL_DirectFB_modes.h"

#include "SDL_syswm.h"
#include "SDL_DirectFB_shape.h"

#include "../SDL_sysvideo.h"
#include "../../render/SDL_sysrender.h"

#ifndef DFB_VERSION_ATLEAST

#define DFB_VERSIONNUM(X, Y, Z)                     \
    ((X)*1000 + (Y)*100 + (Z))

#define DFB_COMPILEDVERSION \
    DFB_VERSIONNUM(DIRECTFB_MAJOR_VERSION, DIRECTFB_MINOR_VERSION, DIRECTFB_MICRO_VERSION)

#define DFB_VERSION_ATLEAST(X, Y, Z) \
    (DFB_COMPILEDVERSION >= DFB_VERSIONNUM(X, Y, Z))

#define SDL_DFB_CHECK(x)    x

#endif

/* the following is not yet tested ... */
#define USE_DISPLAY_PALETTE         (0)


#define SDL_DFB_RENDERERDATA(rend) DirectFB_RenderData *renddata = ((rend) ? (DirectFB_RenderData *) (rend)->driverdata : NULL)
#define SDL_DFB_WINDOWSURFACE(win)  IDirectFBSurface *destsurf = ((DFB_WindowData *) ((win)->driverdata))->surface;

typedef struct
{
    SDL_Window *window;
    DFBSurfaceFlipFlags flipflags;
    int size_changed;
    int lastBlendMode;
    DFBSurfaceBlittingFlags blitFlags;
    DFBSurfaceDrawingFlags drawFlags;
    IDirectFBSurface* target;
} DirectFB_RenderData;

typedef struct
{
    IDirectFBSurface *surface;
    Uint32 format;
    void *pixels;
    int pitch;
    IDirectFBPalette *palette;
    int isDirty;

    SDL_VideoDisplay *display;      /* only for yuv textures */

#if (DFB_VERSION_ATLEAST(1,2,0))
    DFBSurfaceRenderOptions render_options;
#endif
} DirectFB_TextureData;

static void SDLtoDFBRect(const SDL_Rect * sr, DFBRectangle * dr)
{
    dr->x = sr->x;
    dr->y = sr->y;
    dr->h = sr->h;
    dr->w = sr->w;
}
static void SDLtoDFBRect_Float(const SDL_FRect * sr, DFBRectangle * dr)
{
    dr->x = sr->x;
    dr->y = sr->y;
    dr->h = sr->h;
    dr->w = sr->w;
}


static int TextureHasAlpha(DirectFB_TextureData * data)
{
    /* Drawing primitive ? */
    if (!data)
        return 0;

    return (DFB_PIXELFORMAT_HAS_ALPHA(DirectFB_SDLToDFBPixelFormat(data->format)) ? 1 : 0);
#if 0
    switch (data->format) {
    case SDL_PIXELFORMAT_INDEX4LSB:
    case SDL_PIXELFORMAT_INDEX4MSB:
    case SDL_PIXELFORMAT_ARGB4444:
    case SDL_PIXELFORMAT_ARGB1555:
    case SDL_PIXELFORMAT_ARGB8888:
    case SDL_PIXELFORMAT_RGBA8888:
    case SDL_PIXELFORMAT_ABGR8888:
    case SDL_PIXELFORMAT_BGRA8888:
    case SDL_PIXELFORMAT_ARGB2101010:
       return 1;
    default:
        return 0;
    }
#endif
}

static SDL_INLINE IDirectFBSurface *get_dfb_surface(SDL_Window *window)
{
    SDL_SysWMinfo wm_info;
    SDL_memset(&wm_info, 0, sizeof(SDL_SysWMinfo));

    SDL_VERSION(&wm_info.version);
    if (!SDL_GetWindowWMInfo(window, &wm_info)) {
        return NULL;
    }

    return wm_info.info.dfb.surface;
}

static SDL_INLINE IDirectFBWindow *get_dfb_window(SDL_Window *window)
{
    SDL_SysWMinfo wm_info;
    SDL_memset(&wm_info, 0, sizeof(SDL_SysWMinfo));

    SDL_VERSION(&wm_info.version);
    if (!SDL_GetWindowWMInfo(window, &wm_info)) {
        return NULL;
    }

    return wm_info.info.dfb.window;
}

static void SetBlendMode(DirectFB_RenderData * data, int blendMode, DirectFB_TextureData * source)
{
    IDirectFBSurface *destsurf = data->target;

    /* FIXME: check for format change */
    if (1 || data->lastBlendMode != blendMode) {
        switch (blendMode) {
        case SDL_BLENDMODE_NONE:
                                           /**< No blending */
            data->blitFlags = DSBLIT_NOFX;
            data->drawFlags = DSDRAW_NOFX;
            SDL_DFB_CHECK(destsurf->SetSrcBlendFunction(destsurf, DSBF_ONE));
            SDL_DFB_CHECK(destsurf->SetDstBlendFunction(destsurf, DSBF_ZERO));
            break;
#if 0
        case SDL_BLENDMODE_MASK:
            data->blitFlags =  DSBLIT_BLEND_ALPHACHANNEL;
            data->drawFlags = DSDRAW_BLEND;
            SDL_DFB_CHECK(destsurf->SetSrcBlendFunction(destsurf, DSBF_SRCALPHA));
            SDL_DFB_CHECK(destsurf->SetDstBlendFunction(destsurf, DSBF_INVSRCALPHA));
            break;
#endif
        case SDL_BLENDMODE_BLEND:
            data->blitFlags = DSBLIT_BLEND_ALPHACHANNEL;
            data->drawFlags = DSDRAW_BLEND;
            SDL_DFB_CHECK(destsurf->SetSrcBlendFunction(destsurf, DSBF_SRCALPHA));
            SDL_DFB_CHECK(destsurf->SetDstBlendFunction(destsurf, DSBF_INVSRCALPHA));
            break;
        case SDL_BLENDMODE_ADD:
            data->blitFlags = DSBLIT_BLEND_ALPHACHANNEL;
            data->drawFlags = DSDRAW_BLEND;
            /* FIXME: SRCALPHA kills performance on radeon ...
             * It will be cheaper to copy the surface to a temporary surface and premultiply
             */
            if (source && TextureHasAlpha(source))
                SDL_DFB_CHECK(destsurf->SetSrcBlendFunction(destsurf, DSBF_SRCALPHA));
            else
                SDL_DFB_CHECK(destsurf->SetSrcBlendFunction(destsurf, DSBF_ONE));
            SDL_DFB_CHECK(destsurf->SetDstBlendFunction(destsurf, DSBF_ONE));
            break;
        case SDL_BLENDMODE_MOD:
            data->blitFlags = DSBLIT_BLEND_ALPHACHANNEL;
            data->drawFlags = DSDRAW_BLEND;
            SDL_DFB_CHECK(destsurf->SetSrcBlendFunction(destsurf, DSBF_ZERO));
            SDL_DFB_CHECK(destsurf->SetDstBlendFunction(destsurf, DSBF_SRCCOLOR));

            break;
        case SDL_BLENDMODE_MUL:
            data->blitFlags = DSBLIT_BLEND_ALPHACHANNEL;
            data->drawFlags = DSDRAW_BLEND;
            /* FIXME SDL_BLENDMODE_MUL is simplified, and dstA is in fact un-changed.*/
            SDL_DFB_CHECK(destsurf->SetSrcBlendFunction(destsurf, DSBF_DESTCOLOR));
            SDL_DFB_CHECK(destsurf->SetDstBlendFunction(destsurf, DSBF_INVSRCALPHA));

            break;
        }
        data->lastBlendMode = blendMode;
    }
}

static int PrepareDraw(SDL_Renderer * renderer, const SDL_RenderCommand *cmd)
{
    DirectFB_RenderData *data = (DirectFB_RenderData *) renderer->driverdata;
    IDirectFBSurface *destsurf = data->target;
    Uint8 r = cmd->data.draw.r;
    Uint8 g = cmd->data.draw.g;
    Uint8 b = cmd->data.draw.b;
    Uint8 a = cmd->data.draw.a;

    SetBlendMode(data, cmd->data.draw.blend, NULL);
    SDL_DFB_CHECKERR(destsurf->SetDrawingFlags(destsurf, data->drawFlags));

    switch (renderer->blendMode) {
    case SDL_BLENDMODE_NONE:
    /* case SDL_BLENDMODE_MASK: */
    case SDL_BLENDMODE_BLEND:
        break;
    case SDL_BLENDMODE_ADD:
    case SDL_BLENDMODE_MOD:
    case SDL_BLENDMODE_MUL:
        r = ((int) r * (int) a) / 255;
        g = ((int) g * (int) a) / 255;
        b = ((int) b * (int) a) / 255;
        a = 255;
        break;
    case SDL_BLENDMODE_INVALID: break;
    }

    SDL_DFB_CHECKERR(destsurf->SetColor(destsurf, r, g, b, a));
    return 0;
  error:
    return -1;
}

static void DirectFB_WindowEvent(SDL_Renderer * renderer, const SDL_WindowEvent *event)
{
    SDL_DFB_RENDERERDATA(renderer);

    if (event->event == SDL_WINDOWEVENT_SIZE_CHANGED) {
        /* Rebind the context to the window area and update matrices */
        /* SDL_CurrentContext = NULL; */
        /* data->updateSize = SDL_TRUE; */
        renddata->size_changed = SDL_TRUE;
   }
}

static void DirectFB_ActivateRenderer(SDL_Renderer * renderer)
{
    SDL_DFB_RENDERERDATA(renderer);

    if (renddata->size_changed /* || windata->wm_needs_redraw */) {
        renddata->size_changed = SDL_FALSE;
    }
}

static int DirectFB_AcquireVidLayer(SDL_Renderer * renderer, SDL_Texture * texture)
{
    SDL_Window *window = renderer->window;
    SDL_VideoDisplay *display = SDL_GetDisplayForWindow(window);
    SDL_DFB_DEVICEDATA(display->device);
    DFB_DisplayData *dispdata = (DFB_DisplayData *) display->driverdata;
    DirectFB_TextureData *data = texture->driverdata;
    DFBDisplayLayerConfig layconf;
    DFBResult ret;

    if (devdata->use_yuv_direct && (dispdata->vidID >= 0)
        && (!dispdata->vidIDinuse)
        && SDL_ISPIXELFORMAT_FOURCC(data->format)) {
        layconf.flags =
            DLCONF_WIDTH | DLCONF_HEIGHT | DLCONF_PIXELFORMAT |
            DLCONF_SURFACE_CAPS;
        layconf.width = texture->w;
        layconf.height = texture->h;
        layconf.pixelformat = DirectFB_SDLToDFBPixelFormat(data->format);
        layconf.surface_caps = DSCAPS_VIDEOONLY | DSCAPS_DOUBLE;

        SDL_DFB_CHECKERR(devdata->dfb->GetDisplayLayer(devdata->dfb,
                                                       dispdata->vidID,
                                                       &dispdata->vidlayer));
        SDL_DFB_CHECKERR(dispdata->
                         vidlayer->SetCooperativeLevel(dispdata->vidlayer,
                                                       DLSCL_EXCLUSIVE));

        if (devdata->use_yuv_underlays) {
            ret = dispdata->vidlayer->SetLevel(dispdata->vidlayer, -1);
            if (ret != DFB_OK)
                SDL_DFB_DEBUG("Underlay Setlevel not supported\n");
        }
        SDL_DFB_CHECKERR(dispdata->
                         vidlayer->SetConfiguration(dispdata->vidlayer,
                                                    &layconf));
        SDL_DFB_CHECKERR(dispdata->
                         vidlayer->GetSurface(dispdata->vidlayer,
                                              &data->surface));
        dispdata->vidIDinuse = 1;
        data->display = display;
        return 0;
    }
    return 1;
  error:
    if (dispdata->vidlayer) {
        SDL_DFB_RELEASE(data->surface);
        SDL_DFB_CHECKERR(dispdata->
                         vidlayer->SetCooperativeLevel(dispdata->vidlayer,
                                                       DLSCL_ADMINISTRATIVE));
        SDL_DFB_RELEASE(dispdata->vidlayer);
    }
    return 1;
}


/* Copy the SDL_Surface palette to the DirectFB texture palette */
void DirectFB_SetTexturePalette(SDL_Renderer *renderer, SDL_Texture *texture, SDL_Palette *pal)
{
    int i;
    DFBColor dfbpal[256];
    DirectFB_TextureData *data = (DirectFB_TextureData *) texture->driverdata;
    for (i = 0; i < pal->ncolors; i++) {
            dfbpal[i].a = pal->colors[i].a;
            dfbpal[i].r = pal->colors[i].r;
            dfbpal[i].g = pal->colors[i].g;
            dfbpal[i].b = pal->colors[i].b;
    }
    data->palette->SetEntries(data->palette, dfbpal, pal->ncolors, 0);
}

static int DirectFB_CreateTexture(SDL_Renderer * renderer, SDL_Texture * texture)
{
    SDL_Window *window = renderer->window;
    SDL_VideoDisplay *display = SDL_GetDisplayForWindow(window);
    SDL_DFB_DEVICEDATA(display->device);
    DirectFB_TextureData *data;
    DFBSurfaceDescription dsc;
    DFBSurfacePixelFormat pixelformat;

    DirectFB_ActivateRenderer(renderer);

    SDL_DFB_ALLOC_CLEAR(data, sizeof(*data));
    texture->driverdata = data;

    /* find the right pixelformat */
    pixelformat = DirectFB_SDLToDFBPixelFormat(texture->format);
    if (pixelformat == DSPF_UNKNOWN) {
        SDL_SetError("Unknown pixel format %d", data->format);
        goto error;
    }

    data->format = texture->format;
    data->pitch = texture->w * DFB_BYTES_PER_PIXEL(pixelformat);

    if (DirectFB_AcquireVidLayer(renderer, texture) != 0) {
        /* fill surface description */
        dsc.flags =
            DSDESC_WIDTH | DSDESC_HEIGHT | DSDESC_PIXELFORMAT | DSDESC_CAPS;
        dsc.width = texture->w;
        dsc.height = texture->h;
        if(texture->format == SDL_PIXELFORMAT_YV12 ||
           texture->format == SDL_PIXELFORMAT_IYUV) {
           /* dfb has problems with odd sizes -make them even internally */
           dsc.width += (dsc.width % 2);
           dsc.height += (dsc.height % 2);
        }
        /* <1.2 Never use DSCAPS_VIDEOONLY here. It kills performance
         * No DSCAPS_SYSTEMONLY either - let dfb decide
         * 1.2: DSCAPS_SYSTEMONLY boosts performance by factor ~8
         * Depends on other settings as well. Let dfb decide.
         */
        dsc.caps = DSCAPS_PREMULTIPLIED;
#if 0
        if (texture->access == SDL_TEXTUREACCESS_STREAMING)
            dsc.caps |= DSCAPS_SYSTEMONLY;
        else
            dsc.caps |= DSCAPS_VIDEOONLY;
#endif

        dsc.pixelformat = pixelformat;
        data->pixels = NULL;

        /* Create the surface */
        SDL_DFB_CHECKERR(devdata->dfb->CreateSurface(devdata->dfb, &dsc,
                                                     &data->surface));
        if (SDL_ISPIXELFORMAT_INDEXED(data->format)
            && !SDL_ISPIXELFORMAT_FOURCC(data->format)) {
#if 1
            SDL_DFB_CHECKERR(data->surface->GetPalette(data->surface, &data->palette));
#else
            /* DFB has issues with blitting LUT8 surfaces.
             * Creating a new palette does not help.
             */
            DFBPaletteDescription pal_desc;
            pal_desc.flags = DPDESC_SIZE; /* | DPDESC_ENTRIES */
            pal_desc.size = 256;
            SDL_DFB_CHECKERR(devdata->dfb->CreatePalette(devdata->dfb, &pal_desc,&data->palette));
            SDL_DFB_CHECKERR(data->surface->SetPalette(data->surface, data->palette));
#endif
        }

    }
#if (DFB_VERSION_ATLEAST(1,2,0))
    data->render_options = DSRO_NONE;
#endif
    if (texture->access == SDL_TEXTUREACCESS_STREAMING) {
        /* 3 plane YUVs return 1 bpp, but we need more space for other planes */
        if(texture->format == SDL_PIXELFORMAT_YV12 ||
           texture->format == SDL_PIXELFORMAT_IYUV) {
            SDL_DFB_ALLOC_CLEAR(data->pixels, (texture->h * data->pitch  + ((texture->h + texture->h % 2) * (data->pitch + data->pitch % 2) * 2) / 4));
        } else {
            SDL_DFB_ALLOC_CLEAR(data->pixels, texture->h * data->pitch);
        }
    }

    return 0;

  error:
    SDL_DFB_RELEASE(data->palette);
    SDL_DFB_RELEASE(data->surface);
    SDL_DFB_FREE(texture->driverdata);
    return -1;
}

#if 0
static int DirectFB_SetTextureScaleMode(SDL_Renderer * renderer, SDL_Texture * texture)
{
#if (DFB_VERSION_ATLEAST(1,2,0))

    DirectFB_TextureData *data = (DirectFB_TextureData *) texture->driverdata;

    switch (texture->scaleMode) {
    case SDL_SCALEMODE_NONE:
    case SDL_SCALEMODE_FAST:
        data->render_options = DSRO_NONE;
        break;
    case SDL_SCALEMODE_SLOW:
        data->render_options = DSRO_SMOOTH_UPSCALE | DSRO_SMOOTH_DOWNSCALE;
        break;
    case SDL_SCALEMODE_BEST:
        data->render_options =
            DSRO_SMOOTH_UPSCALE | DSRO_SMOOTH_DOWNSCALE | DSRO_ANTIALIAS;
        break;
    default:
        data->render_options = DSRO_NONE;
        texture->scaleMode = SDL_SCALEMODE_NONE;
        return SDL_Unsupported();
    }
#endif
    return 0;
}
#endif

static int DirectFB_UpdateTexture(SDL_Renderer * renderer, SDL_Texture * texture,
                       const SDL_Rect * rect, const void *pixels, int pitch)
{
    DirectFB_TextureData *data = (DirectFB_TextureData *) texture->driverdata;
    Uint8 *dpixels;
    int dpitch;
    Uint8 *src, *dst;
    int row;
    size_t length;
    int bpp = DFB_BYTES_PER_PIXEL(DirectFB_SDLToDFBPixelFormat(texture->format));
    /* FIXME: SDL_BYTESPERPIXEL(texture->format) broken for yuv yv12 3 planes */

    DirectFB_ActivateRenderer(renderer);

    if ((texture->format == SDL_PIXELFORMAT_YV12) ||
        (texture->format == SDL_PIXELFORMAT_IYUV)) {
        bpp = 1;
    }

    SDL_DFB_CHECKERR(data->surface->Lock(data->surface,
                                         DSLF_WRITE | DSLF_READ,
                                         ((void **) &dpixels), &dpitch));
    src = (Uint8 *) pixels;
    dst = (Uint8 *) dpixels + rect->y * dpitch + rect->x * bpp;
    length = rect->w * bpp;
    for (row = 0; row < rect->h; ++row) {
        SDL_memcpy(dst, src, length);
        src += pitch;
        dst += dpitch;
    }
    /* copy other planes for 3 plane formats */
    if ((texture->format == SDL_PIXELFORMAT_YV12) ||
        (texture->format == SDL_PIXELFORMAT_IYUV)) {
        src = (Uint8 *) pixels + texture->h * pitch;
        dst = (Uint8 *) dpixels + texture->h * dpitch + rect->y * dpitch / 4 + rect->x * bpp / 2;
        for (row = 0; row < rect->h / 2 + (rect->h & 1); ++row) {
            SDL_memcpy(dst, src, length / 2);
            src += pitch / 2;
            dst += dpitch / 2;
        }
        src = (Uint8 *) pixels + texture->h * pitch + texture->h * pitch / 4;
        dst = (Uint8 *) dpixels + texture->h * dpitch + texture->h * dpitch / 4 + rect->y * dpitch / 4 + rect->x * bpp / 2;
        for (row = 0; row < rect->h / 2 + (rect->h & 1); ++row) {
            SDL_memcpy(dst, src, length / 2);
            src += pitch / 2;
            dst += dpitch / 2;
        }
    }
    SDL_DFB_CHECKERR(data->surface->Unlock(data->surface));
    data->isDirty = 0;
    return 0;
  error:
    return 1;

}

static int DirectFB_LockTexture(SDL_Renderer * renderer, SDL_Texture * texture,
                     const SDL_Rect * rect, void **pixels, int *pitch)
{
    DirectFB_TextureData *texturedata =
        (DirectFB_TextureData *) texture->driverdata;

    DirectFB_ActivateRenderer(renderer);

#if 0
    if (markDirty) {
        SDL_AddDirtyRect(&texturedata->dirty, rect);
    }
#endif

    if (texturedata->display) {
        void *fdata;
        int fpitch;

        SDL_DFB_CHECKERR(texturedata->surface->Lock(texturedata->surface,
                                                    DSLF_WRITE | DSLF_READ,
                                                    &fdata, &fpitch));
        *pitch = fpitch;
        *pixels = fdata;
    } else {
        *pixels =
            (void *) ((Uint8 *) texturedata->pixels +
                      rect->y * texturedata->pitch +
                      rect->x * DFB_BYTES_PER_PIXEL(DirectFB_SDLToDFBPixelFormat(texture->format)));
        *pitch = texturedata->pitch;
        texturedata->isDirty = 1;
    }
    return 0;

  error:
    return -1;
}

static void DirectFB_UnlockTexture(SDL_Renderer * renderer, SDL_Texture * texture)
{
    DirectFB_TextureData *texturedata =
        (DirectFB_TextureData *) texture->driverdata;

    DirectFB_ActivateRenderer(renderer);

    if (texturedata->display) {
        SDL_DFB_CHECK(texturedata->surface->Unlock(texturedata->surface));
        texturedata->pixels = NULL;
    }
}

static void DirectFB_SetTextureScaleMode(SDL_Renderer * renderer, SDL_Texture * texture, SDL_ScaleMode scaleMode)
{
}

#if 0
static void DirectFB_DirtyTexture(SDL_Renderer * renderer, SDL_Texture * texture,
                      int numrects, const SDL_Rect * rects)
{
    DirectFB_TextureData *data = (DirectFB_TextureData *) texture->driverdata;
    int i;

    for (i = 0; i < numrects; ++i) {
        SDL_AddDirtyRect(&data->dirty, &rects[i]);
    }
}
#endif

static int DirectFB_SetRenderTarget(SDL_Renderer * renderer, SDL_Texture * texture)
{
    DirectFB_RenderData *data = (DirectFB_RenderData *) renderer->driverdata;
    DirectFB_TextureData *tex_data = NULL;

    DirectFB_ActivateRenderer(renderer);
    if (texture) {
        tex_data = (DirectFB_TextureData *) texture->driverdata;
        data->target = tex_data->surface;
    } else {
        data->target = get_dfb_surface(data->window);
    }
    data->lastBlendMode = 0;
    return 0;
}


static int DirectFB_QueueSetViewport(SDL_Renderer * renderer, SDL_RenderCommand *cmd)
{
    return 0;  /* nothing to do in this backend. */
}

static int DirectFB_QueueDrawPoints(SDL_Renderer * renderer, SDL_RenderCommand *cmd, const SDL_FPoint *points, int count)
{
    const size_t len = count * sizeof(SDL_FPoint);
    SDL_FPoint *verts = (SDL_FPoint *) SDL_AllocateRenderVertices(renderer, len, 0, &cmd->data.draw.first);

    if (!verts) {
        return -1;
    }

    cmd->data.draw.count = count;
    SDL_memcpy(verts, points, len);
    return 0;
}

static int DirectFB_QueueGeometry(SDL_Renderer *renderer, SDL_RenderCommand *cmd, SDL_Texture *texture,
        const float *xy, int xy_stride, const SDL_Color *color, int color_stride, const float *uv, int uv_stride,
        int num_vertices, const void *indices, int num_indices, int size_indices,
        float scale_x, float scale_y)
{
    int i;
    int count = indices ? num_indices : num_vertices;
    float *verts;
    int sz = 2 + 4 + (texture ? 2 : 0);

    verts = (float *) SDL_AllocateRenderVertices(renderer, count * sz * sizeof(float), 0, &cmd->data.draw.first);
    if (!verts) {
        return -1;
    }

    cmd->data.draw.count = count;
    size_indices = indices ? size_indices : 0;

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

        xy_ = (float *)((char*)xy + j * xy_stride);
        col_ = *(SDL_Color *)((char*)color + j * color_stride);

        *(verts++) = xy_[0] * scale_x;
        *(verts++) = xy_[1] * scale_y;

        *(verts++) = col_.r;
        *(verts++) = col_.g;
        *(verts++) = col_.b;
        *(verts++) = col_.a;

        if (texture) {
            float *uv_ = (float *)((char*)uv + j * uv_stride);
            *(verts++) = uv_[0];
            *(verts++) = uv_[1];
        }
    }
    return 0;
}

static int DirectFB_QueueFillRects(SDL_Renderer * renderer, SDL_RenderCommand *cmd, const SDL_FRect * rects, int count)
{
    const size_t len = count * sizeof(SDL_FRect);
    SDL_FRect *verts = (SDL_FRect *) SDL_AllocateRenderVertices(renderer, len, 0, &cmd->data.draw.first);

    if (!verts) {
        return -1;
    }

    cmd->data.draw.count = count;
    SDL_memcpy(verts, rects, len);
    return 0;
}

static int DirectFB_QueueCopy(SDL_Renderer * renderer, SDL_RenderCommand *cmd, SDL_Texture * texture,
             const SDL_Rect * srcrect, const SDL_FRect * dstrect)
{
    DFBRectangle *verts = (DFBRectangle *) SDL_AllocateRenderVertices(renderer, 2 * sizeof(DFBRectangle), 0, &cmd->data.draw.first);

    if (!verts) {
        return -1;
    }

    cmd->data.draw.count = 1;

    SDLtoDFBRect(srcrect, verts++);
    SDLtoDFBRect_Float(dstrect, verts);

    return 0;
}

static int DirectFB_RunCommandQueue(SDL_Renderer * renderer, SDL_RenderCommand *cmd, void *vertices, size_t vertsize)
{
    /* !!! FIXME: there are probably some good optimization wins in here if someone wants to look it over. */
    DirectFB_RenderData *data = (DirectFB_RenderData *) renderer->driverdata;
    IDirectFBSurface *destsurf = data->target;
    DFBRegion clip_region;
    size_t i;

    DirectFB_ActivateRenderer(renderer);

    SDL_zero(clip_region);  /* in theory, this always gets set before use. */

    while (cmd) {
        switch (cmd->command) {
            case SDL_RENDERCMD_SETDRAWCOLOR:
                break;  /* not used here */

            case SDL_RENDERCMD_SETVIEWPORT: {
                const SDL_Rect *viewport = &cmd->data.viewport.rect;
                clip_region.x1 = viewport->x;
                clip_region.y1 = viewport->y;
                clip_region.x2 = clip_region.x1 + viewport->w - 1;
                clip_region.y2 = clip_region.y1 + viewport->h - 1;
                destsurf->SetClip(destsurf, &clip_region);
                break;
            }

            case SDL_RENDERCMD_SETCLIPRECT: {
                /* !!! FIXME: how does this SetClip interact with the one in SETVIEWPORT? */
                if (cmd->data.cliprect.enabled) {
                    const SDL_Rect *rect = &cmd->data.cliprect.rect;
                    clip_region.x1 = rect->x;
                    clip_region.x2 = rect->x + rect->w;
                    clip_region.y1 = rect->y;
                    clip_region.y2 = rect->y + rect->h;
                    destsurf->SetClip(destsurf, &clip_region);
                }
                break;
            }

            case SDL_RENDERCMD_CLEAR: {
                const Uint8 r = cmd->data.color.r;
                const Uint8 g = cmd->data.color.g;
                const Uint8 b = cmd->data.color.b;
                const Uint8 a = cmd->data.color.a;
                destsurf->Clear(destsurf, r, g, b, a);
                break;
            }

            case SDL_RENDERCMD_DRAW_POINTS: {
                const size_t count = cmd->data.draw.count;
                const SDL_FPoint *points = (SDL_FPoint *) (((Uint8 *) vertices) + cmd->data.draw.first);
                PrepareDraw(renderer, cmd);
                for (i = 0; i < count; i++) {
                    const int x = points[i].x + clip_region.x1;
                    const int y = points[i].y + clip_region.y1;
                    destsurf->DrawLine(destsurf, x, y, x, y);
                }
                break;
            }

            case SDL_RENDERCMD_DRAW_LINES: {
                const SDL_FPoint *points = (SDL_FPoint *) (((Uint8 *) vertices) + cmd->data.draw.first);
                const size_t count = cmd->data.draw.count;

                PrepareDraw(renderer, cmd);

                #if (DFB_VERSION_ATLEAST(1,2,0))  /* !!! FIXME: should this be set once, somewhere else? */
                destsurf->SetRenderOptions(destsurf, DSRO_ANTIALIAS);
                #endif

                for (i = 0; i < count - 1; i++) {
                    const int x1 = points[i].x + clip_region.x1;
                    const int y1 = points[i].y + clip_region.y1;
                    const int x2 = points[i + 1].x + clip_region.x1;
                    const int y2 = points[i + 1].y + clip_region.y1;
                    destsurf->DrawLine(destsurf, x1, y1, x2, y2);
                }
                break;
            }

            case SDL_RENDERCMD_FILL_RECTS: {
                const SDL_FRect *rects = (SDL_FRect *) (((Uint8 *) vertices) + cmd->data.draw.first);
                const size_t count = cmd->data.draw.count;

                PrepareDraw(renderer, cmd);

                for (i = 0; i < count; i++, rects++) {
                    destsurf->FillRectangle(destsurf, rects->x + clip_region.x1, rects->y + clip_region.y1, rects->w, rects->h);
                }
                break;
            }

            case SDL_RENDERCMD_COPY: {
                SDL_Texture *texture = cmd->data.draw.texture;
                const Uint8 r = cmd->data.draw.r;
                const Uint8 g = cmd->data.draw.g;
                const Uint8 b = cmd->data.draw.b;
                const Uint8 a = cmd->data.draw.a;
                DFBRectangle *verts = (DFBRectangle *) (((Uint8 *) vertices) + cmd->data.draw.first);
                DirectFB_TextureData *texturedata = (DirectFB_TextureData *) texture->driverdata;
                DFBRectangle *sr = verts++;
                DFBRectangle *dr = verts;

                dr->x += clip_region.x1;
                dr->y += clip_region.y1;

                if (texturedata->display) {
                    int px, py;
                    SDL_Window *window = renderer->window;
                    IDirectFBWindow *dfbwin = get_dfb_window(window);
                    SDL_DFB_WINDOWDATA(window);
                    SDL_VideoDisplay *display = texturedata->display;
                    DFB_DisplayData *dispdata = (DFB_DisplayData *) display->driverdata;

                    dispdata->vidlayer->SetSourceRectangle(dispdata->vidlayer, sr->x, sr->y, sr->w, sr->h);
                    dfbwin->GetPosition(dfbwin, &px, &py);
                    px += windata->client.x;
                    py += windata->client.y;
                    dispdata->vidlayer->SetScreenRectangle(dispdata->vidlayer, px + dr->x, py + dr->y, dr->w, dr->h);
                } else {
                    DFBSurfaceBlittingFlags flags = 0;
                    if (texturedata->isDirty) {
                        const SDL_Rect rect = { 0, 0, texture->w, texture->h };
                        DirectFB_UpdateTexture(renderer, texture, &rect, texturedata->pixels, texturedata->pitch);
                    }

                    if (a != 0xFF) {
                        flags |= DSBLIT_BLEND_COLORALPHA;
                    }

                    if ((r & g & b) != 0xFF) {
                        flags |= DSBLIT_COLORIZE;
                    }

                    destsurf->SetColor(destsurf, r, g, b, a);

                    /* ???? flags |= DSBLIT_SRC_PREMULTCOLOR; */

                    SetBlendMode(data, texture->blendMode, texturedata);

                    destsurf->SetBlittingFlags(destsurf, data->blitFlags | flags);

#if (DFB_VERSION_ATLEAST(1,2,0))
                    destsurf->SetRenderOptions(destsurf, texturedata->render_options);
#endif

                    if (sr->w == dr->w && sr->h == dr->h) {
                        destsurf->Blit(destsurf, texturedata->surface, sr, dr->x, dr->y);
                    } else {
                        destsurf->StretchBlit(destsurf, texturedata->surface, sr, dr);
                    }
                }
                break;
            }


            case SDL_RENDERCMD_GEOMETRY: {
                const float *verts = (float *) (((Uint8 *) vertices) + cmd->data.draw.first);
                SDL_Texture *texture = cmd->data.draw.texture;
                const size_t count = cmd->data.draw.count;

                Uint8 save_r = cmd->data.draw.r;
                Uint8 save_g = cmd->data.draw.g;
                Uint8 save_b = cmd->data.draw.b;
                Uint8 save_a = cmd->data.draw.a;

                int j;
                for (j = 0; j < count; j += 3)
                {
                    float x1, y1, r1, g1, b1, a1, u1, v1;
                    float x2, y2, r2, g2, b2, a2, u2, v2;
                    float x3, y3, r3, g3, b3, a3, u3, v3;

                    x1 = *(verts++);
                    y1 = *(verts++);
                    r1 = *(verts++);
                    g1 = *(verts++);
                    b1 = *(verts++);
                    a1 = *(verts++);
                    if (texture) {
                        u1 = *(verts++);
                        v1 = *(verts++);
                    }
                    x2 = *(verts++);
                    y2 = *(verts++);
                    r2 = *(verts++);
                    g2 = *(verts++);
                    b2 = *(verts++);
                    a2 = *(verts++);
                    if (texture) {
                        u2 = *(verts++);
                        v2 = *(verts++);
                    }
                    x3 = *(verts++);
                    y3 = *(verts++);
                    r3 = *(verts++);
                    g3 = *(verts++);
                    b3 = *(verts++);
                    a3 = *(verts++);
                    if (texture) {
                        u3 = *(verts++);
                        v3 = *(verts++);
                    }


                    if (texture) {
                        DFBVertex vertices[3];

                        DirectFB_TextureData *texturedata = (DirectFB_TextureData *) texture->driverdata;

                        DFBSurfaceBlittingFlags flags = 0;

                        int r = (r1 + r2 + r3) / 3;
                        int g = (g1 + g2 + g3) / 3;
                        int b = (b1 + b2 + b3) / 3;
                        int a = (a1 + a2 + a3) / 3;


                        if (texturedata->isDirty) {
                            const SDL_Rect rect = { 0, 0, texture->w, texture->h };
                            DirectFB_UpdateTexture(renderer, texture, &rect, texturedata->pixels, texturedata->pitch);
                        }

                        if (a != 0xFF) {
                            flags |= DSBLIT_BLEND_COLORALPHA;
                        }

                        if ((r & g & b) != 0xFF) {
                            flags |= DSBLIT_COLORIZE;
                        }

                        destsurf->SetColor(destsurf, r, g, b, a);

                        /* ???? flags |= DSBLIT_SRC_PREMULTCOLOR; */

                        SetBlendMode(data, texture->blendMode, texturedata);

                        destsurf->SetBlittingFlags(destsurf, data->blitFlags | flags);

#if (DFB_VERSION_ATLEAST(1,2,0))
                        destsurf->SetRenderOptions(destsurf, texturedata->render_options);
#endif

                        vertices[0].x = x1;
                        vertices[0].y = y1;
                        vertices[0].z = 0;
                        vertices[0].w = 0;
                        vertices[0].s = u1;
                        vertices[0].t = v1;

                        vertices[1].x = x2;
                        vertices[1].y = y2;
                        vertices[1].z = 0;
                        vertices[1].w = 0;
                        vertices[1].s = u2;
                        vertices[1].t = v2;

                        vertices[2].x = x3;
                        vertices[2].y = y3;
                        vertices[2].z = 0;
                        vertices[2].w = 0;
                        vertices[2].s = u3;
                        vertices[2].t = v3;

                        destsurf->TextureTriangles(destsurf, texturedata->surface, vertices, NULL, 3, DTTF_LIST);
                    } else {
                        DFBTriangle tris;
                        tris.x1 = x1;
                        tris.y1 = y1;
                        tris.x2 = x2;
                        tris.y2 = y2;
                        tris.x3 = x3;
                        tris.y3 = y3;

                        cmd->data.draw.r = (r1 + r2 + r3) / 3;
                        cmd->data.draw.g = (g1 + g2 + g3) / 3;
                        cmd->data.draw.b = (b1 + b2 + b3) / 3;
                        cmd->data.draw.a = (a1 + a2 + a3) / 3;

                        PrepareDraw(renderer, cmd);

                        destsurf->FillTriangles(destsurf, &tris, 1);
                    }
                }

                cmd->data.draw.r = save_r;
                cmd->data.draw.g = save_g;
                cmd->data.draw.b = save_b;
                cmd->data.draw.a = save_a;
                break;
            }

            case SDL_RENDERCMD_COPY_EX: /* unused */
                break;

            case SDL_RENDERCMD_NO_OP:
                break;
        }

        cmd = cmd->next;
    }

    return 0;
}


static int DirectFB_RenderPresent(SDL_Renderer * renderer)
{
    DirectFB_RenderData *data = (DirectFB_RenderData *) renderer->driverdata;
    SDL_Window *window = renderer->window;
    SDL_DFB_WINDOWDATA(window);
    SDL_ShapeData *shape_data = (window->shaper ? window->shaper->driverdata : NULL);

    DirectFB_ActivateRenderer(renderer);

    if (shape_data && shape_data->surface) {
        /* saturate the window surface alpha channel */
        SDL_DFB_CHECK(windata->window_surface->SetSrcBlendFunction(windata->window_surface, DSBF_ONE));
        SDL_DFB_CHECK(windata->window_surface->SetDstBlendFunction(windata->window_surface, DSBF_ONE));
        SDL_DFB_CHECK(windata->window_surface->SetDrawingFlags(windata->window_surface, DSDRAW_BLEND));
        SDL_DFB_CHECK(windata->window_surface->SetColor(windata->window_surface, 0, 0, 0, 0xff));
        SDL_DFB_CHECK(windata->window_surface->FillRectangle(windata->window_surface, 0,0, windata->size.w, windata->size.h));

        /* blit the mask */
        SDL_DFB_CHECK(windata->surface->SetSrcBlendFunction(windata->surface, DSBF_DESTCOLOR));
        SDL_DFB_CHECK(windata->surface->SetDstBlendFunction(windata->surface, DSBF_ZERO));
        SDL_DFB_CHECK(windata->surface->SetBlittingFlags(windata->surface, DSBLIT_BLEND_ALPHACHANNEL));
#if (DFB_VERSION_ATLEAST(1,2,0))
        SDL_DFB_CHECK(windata->surface->SetRenderOptions(windata->surface, DSRO_NONE));
#endif
        SDL_DFB_CHECK(windata->surface->Blit(windata->surface, shape_data->surface, NULL, 0, 0));
    }

    /* Send the data to the display */
    SDL_DFB_CHECK(windata->window_surface->Flip(windata->window_surface, NULL,
                                                data->flipflags));
    return 0;
}

static void DirectFB_DestroyTexture(SDL_Renderer * renderer, SDL_Texture * texture)
{
    DirectFB_TextureData *data = (DirectFB_TextureData *) texture->driverdata;

    DirectFB_ActivateRenderer(renderer);

    if (!data) {
        return;
    }
    SDL_DFB_RELEASE(data->palette);
    SDL_DFB_RELEASE(data->surface);
    if (data->display) {
        DFB_DisplayData *dispdata =
            (DFB_DisplayData *) data->display->driverdata;
        dispdata->vidIDinuse = 0;
        /* FIXME: Shouldn't we reset the cooperative level */
        SDL_DFB_CHECK(dispdata->vidlayer->SetCooperativeLevel(dispdata->vidlayer,
                                                DLSCL_ADMINISTRATIVE));
        SDL_DFB_RELEASE(dispdata->vidlayer);
    }
    SDL_DFB_FREE(data->pixels);
    SDL_free(data);
    texture->driverdata = NULL;
}

static void DirectFB_DestroyRenderer(SDL_Renderer * renderer)
{
    DirectFB_RenderData *data = (DirectFB_RenderData *) renderer->driverdata;
#if 0
    SDL_VideoDisplay *display = SDL_GetDisplayForWindow(data->window);
    if (display->palette) {
        SDL_DelPaletteWatch(display->palette, DisplayPaletteChanged, data);
    }
#endif

    SDL_free(data);
    SDL_free(renderer);
}

static int DirectFB_RenderReadPixels(SDL_Renderer * renderer, const SDL_Rect * rect,
                     Uint32 format, void * pixels, int pitch)
{
    Uint32 sdl_format;
    unsigned char* laypixels;
    int laypitch;
    DFBSurfacePixelFormat dfb_format;
    DirectFB_RenderData *data = (DirectFB_RenderData *) renderer->driverdata;
    IDirectFBSurface *winsurf = data->target;

    DirectFB_ActivateRenderer(renderer);

    winsurf->GetPixelFormat(winsurf, &dfb_format);
    sdl_format = DirectFB_DFBToSDLPixelFormat(dfb_format);
    winsurf->Lock(winsurf, DSLF_READ, (void **) &laypixels, &laypitch);

    laypixels += (rect->y * laypitch + rect->x * SDL_BYTESPERPIXEL(sdl_format) );
    SDL_ConvertPixels(rect->w, rect->h,
                      sdl_format, laypixels, laypitch,
                      format, pixels, pitch);

    winsurf->Unlock(winsurf);

    return 0;
}

#if 0
static int DirectFB_RenderWritePixels(SDL_Renderer * renderer, const SDL_Rect * rect,
                      Uint32 format, const void * pixels, int pitch)
{
    SDL_Window *window = renderer->window;
    SDL_DFB_WINDOWDATA(window);
    Uint32 sdl_format;
    unsigned char* laypixels;
    int laypitch;
    DFBSurfacePixelFormat dfb_format;

    SDL_DFB_CHECK(windata->surface->GetPixelFormat(windata->surface, &dfb_format));
    sdl_format = DirectFB_DFBToSDLPixelFormat(dfb_format);

    SDL_DFB_CHECK(windata->surface->Lock(windata->surface, DSLF_WRITE, (void **) &laypixels, &laypitch));

    laypixels += (rect->y * laypitch + rect->x * SDL_BYTESPERPIXEL(sdl_format) );
    SDL_ConvertPixels(rect->w, rect->h,
                      format, pixels, pitch,
                      sdl_format, laypixels, laypitch);

    SDL_DFB_CHECK(windata->surface->Unlock(windata->surface));

    return 0;
}
#endif


SDL_Renderer *DirectFB_CreateRenderer(SDL_Window * window, Uint32 flags)
{
    IDirectFBSurface *winsurf = get_dfb_surface(window);
    /*SDL_VideoDisplay *display = SDL_GetDisplayForWindow(window);*/
    SDL_Renderer *renderer = NULL;
    DirectFB_RenderData *data = NULL;
    DFBSurfaceCapabilities scaps;

    if (!winsurf) {
        return NULL;
    }

    SDL_DFB_ALLOC_CLEAR(renderer, sizeof(*renderer));
    SDL_DFB_ALLOC_CLEAR(data, sizeof(*data));

    renderer->WindowEvent = DirectFB_WindowEvent;
    renderer->CreateTexture = DirectFB_CreateTexture;
    renderer->UpdateTexture = DirectFB_UpdateTexture;
    renderer->LockTexture = DirectFB_LockTexture;
    renderer->UnlockTexture = DirectFB_UnlockTexture;
    renderer->SetTextureScaleMode = DirectFB_SetTextureScaleMode;
    renderer->QueueSetViewport = DirectFB_QueueSetViewport;
    renderer->QueueSetDrawColor = DirectFB_QueueSetViewport;  /* SetViewport and SetDrawColor are (currently) no-ops. */
    renderer->QueueDrawPoints = DirectFB_QueueDrawPoints;
    renderer->QueueDrawLines = DirectFB_QueueDrawPoints;  /* lines and points queue vertices the same way. */
    renderer->QueueGeometry = DirectFB_QueueGeometry;
    renderer->QueueFillRects = DirectFB_QueueFillRects;
    renderer->QueueCopy = DirectFB_QueueCopy;
    renderer->RunCommandQueue = DirectFB_RunCommandQueue;
    renderer->RenderPresent = DirectFB_RenderPresent;

    /* FIXME: Yet to be tested */
    renderer->RenderReadPixels = DirectFB_RenderReadPixels;
    /* renderer->RenderWritePixels = DirectFB_RenderWritePixels; */

    renderer->DestroyTexture = DirectFB_DestroyTexture;
    renderer->DestroyRenderer = DirectFB_DestroyRenderer;
    renderer->SetRenderTarget = DirectFB_SetRenderTarget;

    renderer->info = DirectFB_RenderDriver.info;
    renderer->window = window;      /* SDL window */
    renderer->driverdata = data;

    renderer->info.flags =
        SDL_RENDERER_ACCELERATED | SDL_RENDERER_TARGETTEXTURE;

    data->window = window;
    data->target = winsurf;

    data->flipflags = DSFLIP_PIPELINE | DSFLIP_BLIT;

    if (flags & SDL_RENDERER_PRESENTVSYNC) {
        data->flipflags |= DSFLIP_WAITFORSYNC | DSFLIP_ONSYNC;
        renderer->info.flags |= SDL_RENDERER_PRESENTVSYNC;
    } else
        data->flipflags |= DSFLIP_ONSYNC;

    SDL_DFB_CHECKERR(winsurf->GetCapabilities(winsurf, &scaps));

#if 0
    if (scaps & DSCAPS_DOUBLE)
        renderer->info.flags |= SDL_RENDERER_PRESENTFLIP2;
    else if (scaps & DSCAPS_TRIPLE)
        renderer->info.flags |= SDL_RENDERER_PRESENTFLIP3;
    else
        renderer->info.flags |= SDL_RENDERER_SINGLEBUFFER;
#endif

    DirectFB_SetSupportedPixelFormats(&renderer->info);

#if 0
    /* Set up a palette watch on the display palette */
    if (display-> palette) {
        SDL_AddPaletteWatch(display->palette, DisplayPaletteChanged, data);
    }
#endif

    return renderer;

  error:
    SDL_DFB_FREE(renderer);
    SDL_DFB_FREE(data);
    return NULL;
}


SDL_RenderDriver DirectFB_RenderDriver = {
    DirectFB_CreateRenderer,
    {
     "directfb",
     (SDL_RENDERER_PRESENTVSYNC | SDL_RENDERER_ACCELERATED),
     /* (SDL_TEXTUREMODULATE_NONE | SDL_TEXTUREMODULATE_COLOR |
      SDL_TEXTUREMODULATE_ALPHA),
      (SDL_BLENDMODE_NONE | SDL_BLENDMODE_MASK | SDL_BLENDMODE_BLEND |
      SDL_BLENDMODE_ADD | SDL_BLENDMODE_MOD),
     (SDL_SCALEMODE_NONE | SDL_SCALEMODE_FAST |
      SDL_SCALEMODE_SLOW | SDL_SCALEMODE_BEST), */
     0,
     {
             /* formats filled in later */
     },
     0,
     0}
};

#endif /* SDL_VIDEO_DRIVER_DIRECTFB */

/* vi: set ts=4 sw=4 expandtab: */
