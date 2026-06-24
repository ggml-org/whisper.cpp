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

#ifndef SDL_sysrender_h_
#define SDL_sysrender_h_

#include "SDL_render.h"
#include "SDL_events.h"
#include "SDL_mutex.h"
#include "SDL_yuv_sw_c.h"

/* Set up for C function definitions, even when using C++ */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * A rectangle, with the origin at the upper left (double precision).
 */
typedef struct SDL_DRect
{
    double x;
    double y;
    double w;
    double h;
} SDL_DRect;

/* The SDL 2D rendering system */

typedef struct SDL_RenderDriver SDL_RenderDriver;

/* Define the SDL texture structure */
struct SDL_Texture
{
    const void *magic;
    Uint32 format;           /**< The pixel format of the texture */
    int access;              /**< SDL_TextureAccess */
    int w;                   /**< The width of the texture */
    int h;                   /**< The height of the texture */
    int modMode;             /**< The texture modulation mode */
    SDL_BlendMode blendMode; /**< The texture blend mode */
    SDL_ScaleMode scaleMode; /**< The texture scale mode */
    SDL_Color color;         /**< Texture modulation values */

    SDL_Renderer *renderer;

    /* Support for formats not supported directly by the renderer */
    SDL_Texture *native;
    SDL_SW_YUVTexture *yuv;
    void *pixels;
    int pitch;
    SDL_Rect locked_rect;
    SDL_Surface *locked_surface; /**< Locked region exposed as a SDL surface */

    Uint32 last_command_generation; /* last command queue generation this texture was in. */

    void *driverdata; /**< Driver specific texture representation */
    void *userdata;

    SDL_Texture *prev;
    SDL_Texture *next;
};

typedef enum
{
    SDL_RENDERCMD_NO_OP,
    SDL_RENDERCMD_SETVIEWPORT,
    SDL_RENDERCMD_SETCLIPRECT,
    SDL_RENDERCMD_SETDRAWCOLOR,
    SDL_RENDERCMD_CLEAR,
    SDL_RENDERCMD_DRAW_POINTS,
    SDL_RENDERCMD_DRAW_LINES,
    SDL_RENDERCMD_FILL_RECTS,
    SDL_RENDERCMD_COPY,
    SDL_RENDERCMD_COPY_EX,
    SDL_RENDERCMD_GEOMETRY
} SDL_RenderCommandType;

typedef struct SDL_RenderCommand
{
    SDL_RenderCommandType command;
    union
    {
        struct
        {
            size_t first;
            SDL_Rect rect;
        } viewport;
        struct
        {
            SDL_bool enabled;
            SDL_Rect rect;
        } cliprect;
        struct
        {
            size_t first;
            size_t count;
            Uint8 r, g, b, a;
            SDL_BlendMode blend;
            SDL_Texture *texture;
        } draw;
        struct
        {
            size_t first;
            Uint8 r, g, b, a;
        } color;
    } data;
    struct SDL_RenderCommand *next;
} SDL_RenderCommand;

typedef struct SDL_VertexSolid
{
    SDL_FPoint position;
    SDL_Color color;
} SDL_VertexSolid;

typedef enum
{
    SDL_RENDERLINEMETHOD_POINTS,
    SDL_RENDERLINEMETHOD_LINES,
    SDL_RENDERLINEMETHOD_GEOMETRY,
} SDL_RenderLineMethod;

/* Define the SDL renderer structure */
struct SDL_Renderer
{
    const void *magic;

    void (*WindowEvent)(SDL_Renderer *renderer, const SDL_WindowEvent *event);
    int (*GetOutputSize)(SDL_Renderer *renderer, int *w, int *h);
    SDL_bool (*SupportsBlendMode)(SDL_Renderer *renderer, SDL_BlendMode blendMode);
    int (*CreateTexture)(SDL_Renderer *renderer, SDL_Texture *texture);
    int (*QueueSetViewport)(SDL_Renderer *renderer, SDL_RenderCommand *cmd);
    int (*QueueSetDrawColor)(SDL_Renderer *renderer, SDL_RenderCommand *cmd);
    int (*QueueDrawPoints)(SDL_Renderer *renderer, SDL_RenderCommand *cmd, const SDL_FPoint *points,
                           int count);
    int (*QueueDrawLines)(SDL_Renderer *renderer, SDL_RenderCommand *cmd, const SDL_FPoint *points,
                          int count);
    int (*QueueFillRects)(SDL_Renderer *renderer, SDL_RenderCommand *cmd, const SDL_FRect *rects,
                          int count);
    int (*QueueCopy)(SDL_Renderer *renderer, SDL_RenderCommand *cmd, SDL_Texture *texture,
                     const SDL_Rect *srcrect, const SDL_FRect *dstrect);
    int (*QueueCopyEx)(SDL_Renderer *renderer, SDL_RenderCommand *cmd, SDL_Texture *texture,
                       const SDL_Rect *srcquad, const SDL_FRect *dstrect,
                       const double angle, const SDL_FPoint *center, const SDL_RendererFlip flip, float scale_x, float scale_y);
    int (*QueueGeometry)(SDL_Renderer *renderer, SDL_RenderCommand *cmd, SDL_Texture *texture,
                         const float *xy, int xy_stride, const SDL_Color *color, int color_stride, const float *uv, int uv_stride,
                         int num_vertices, const void *indices, int num_indices, int size_indices,
                         float scale_x, float scale_y);

    int (*RunCommandQueue)(SDL_Renderer *renderer, SDL_RenderCommand *cmd, void *vertices, size_t vertsize);
    int (*UpdateTexture)(SDL_Renderer *renderer, SDL_Texture *texture,
                         const SDL_Rect *rect, const void *pixels,
                         int pitch);
#if SDL_HAVE_YUV
    int (*UpdateTextureYUV)(SDL_Renderer *renderer, SDL_Texture *texture,
                            const SDL_Rect *rect,
                            const Uint8 *Yplane, int Ypitch,
                            const Uint8 *Uplane, int Upitch,
                            const Uint8 *Vplane, int Vpitch);
    int (*UpdateTextureNV)(SDL_Renderer *renderer, SDL_Texture *texture,
                           const SDL_Rect *rect,
                           const Uint8 *Yplane, int Ypitch,
                           const Uint8 *UVplane, int UVpitch);
#endif
    int (*LockTexture)(SDL_Renderer *renderer, SDL_Texture *texture,
                       const SDL_Rect *rect, void **pixels, int *pitch);
    void (*UnlockTexture)(SDL_Renderer *renderer, SDL_Texture *texture);
    void (*SetTextureScaleMode)(SDL_Renderer *renderer, SDL_Texture *texture, SDL_ScaleMode scaleMode);
    int (*SetRenderTarget)(SDL_Renderer *renderer, SDL_Texture *texture);
    int (*RenderReadPixels)(SDL_Renderer *renderer, const SDL_Rect *rect,
                            Uint32 format, void *pixels, int pitch);
    int (*RenderPresent)(SDL_Renderer *renderer);
    void (*DestroyTexture)(SDL_Renderer *renderer, SDL_Texture *texture);

    void (*DestroyRenderer)(SDL_Renderer *renderer);

    int (*SetVSync)(SDL_Renderer *renderer, int vsync);

    int (*GL_BindTexture)(SDL_Renderer *renderer, SDL_Texture *texture, float *texw, float *texh);
    int (*GL_UnbindTexture)(SDL_Renderer *renderer, SDL_Texture *texture);

    void *(*GetMetalLayer)(SDL_Renderer *renderer);
    void *(*GetMetalCommandEncoder)(SDL_Renderer *renderer);

    /* The current renderer info */
    SDL_RendererInfo info;

    /* The window associated with the renderer */
    SDL_Window *window;
    SDL_bool hidden;

    /* Whether we should simulate vsync */
    SDL_bool wanted_vsync;
    SDL_bool simulate_vsync;
    Uint32 simulate_vsync_interval;
    Uint32 last_present;

    /* The logical resolution for rendering */
    int logical_w;
    int logical_h;
    int logical_w_backup;
    int logical_h_backup;

    /* Whether or not to force the viewport to even integer intervals */
    SDL_bool integer_scale;

    /* The drawable area within the window */
    SDL_DRect viewport;
    SDL_DRect viewport_backup;

    /* The clip rectangle within the window */
    SDL_DRect clip_rect;
    SDL_DRect clip_rect_backup;

    /* Whether or not the clipping rectangle is used. */
    SDL_bool clipping_enabled;
    SDL_bool clipping_enabled_backup;

    /* The render output coordinate scale */
    SDL_FPoint scale;
    SDL_FPoint scale_backup;

    /* The pixel to point coordinate scale */
    SDL_FPoint dpi_scale;

    /* Whether or not to scale relative mouse motion */
    SDL_bool relative_scaling;

    /* The method of drawing lines */
    SDL_RenderLineMethod line_method;

    /* List of triangle indices to draw rects */
    int rect_index_order[6];

    /* Remainder from scaled relative motion */
    float xrel;
    float yrel;

    /* The list of textures */
    SDL_Texture *textures;
    SDL_Texture *target;
    SDL_mutex *target_mutex;

    SDL_Color color;         /**< Color for drawing operations values */
    SDL_BlendMode blendMode; /**< The drawing blend mode */

    SDL_bool always_batch;
    SDL_bool batching;
    SDL_RenderCommand *render_commands;
    SDL_RenderCommand *render_commands_tail;
    SDL_RenderCommand *render_commands_pool;
    Uint32 render_command_generation;
    Uint32 last_queued_color;
    SDL_DRect last_queued_viewport;
    SDL_DRect last_queued_cliprect;
    SDL_bool last_queued_cliprect_enabled;
    SDL_bool color_queued;
    SDL_bool viewport_queued;
    SDL_bool cliprect_queued;

    void *vertex_data;
    size_t vertex_data_used;
    size_t vertex_data_allocation;

    void *driverdata;
};

/* Define the SDL render driver structure */
struct SDL_RenderDriver
{
    SDL_Renderer *(*CreateRenderer)(SDL_Window *window, Uint32 flags);

    /* Info about the renderer capabilities */
    SDL_RendererInfo info;
};

/* Not all of these are available in a given build. Use #ifdefs, etc. */
extern SDL_RenderDriver D3D_RenderDriver;
extern SDL_RenderDriver D3D11_RenderDriver;
extern SDL_RenderDriver D3D12_RenderDriver;
extern SDL_RenderDriver GL_RenderDriver;
extern SDL_RenderDriver GLES2_RenderDriver;
extern SDL_RenderDriver GLES_RenderDriver;
extern SDL_RenderDriver DirectFB_RenderDriver;
extern SDL_RenderDriver METAL_RenderDriver;
extern SDL_RenderDriver PS2_RenderDriver;
extern SDL_RenderDriver PSP_RenderDriver;
extern SDL_RenderDriver SW_RenderDriver;
extern SDL_RenderDriver VITA_GXM_RenderDriver;

/* Blend mode functions */
extern SDL_BlendFactor SDL_GetBlendModeSrcColorFactor(SDL_BlendMode blendMode);
extern SDL_BlendFactor SDL_GetBlendModeDstColorFactor(SDL_BlendMode blendMode);
extern SDL_BlendOperation SDL_GetBlendModeColorOperation(SDL_BlendMode blendMode);
extern SDL_BlendFactor SDL_GetBlendModeSrcAlphaFactor(SDL_BlendMode blendMode);
extern SDL_BlendFactor SDL_GetBlendModeDstAlphaFactor(SDL_BlendMode blendMode);
extern SDL_BlendOperation SDL_GetBlendModeAlphaOperation(SDL_BlendMode blendMode);

/* drivers call this during their Queue*() methods to make space in a array that are used
   for a vertex buffer during RunCommandQueue(). Pointers returned here are only valid until
   the next call, because it might be in an array that gets realloc()'d. */
extern void *SDL_AllocateRenderVertices(SDL_Renderer *renderer, const size_t numbytes, const size_t alignment, size_t *offset);

extern int SDL_PrivateLowerBlitScaled(SDL_Surface *src, SDL_Rect *srcrect, SDL_Surface *dst, SDL_Rect *dstrect, SDL_ScaleMode scaleMode);
extern int SDL_PrivateUpperBlitScaled(SDL_Surface *src, const SDL_Rect *srcrect, SDL_Surface *dst, SDL_Rect *dstrect, SDL_ScaleMode scaleMode);

/* Ends C function definitions when using C++ */
#ifdef __cplusplus
}
#endif

#endif /* SDL_sysrender_h_ */

/* vi: set ts=4 sw=4 expandtab: */
