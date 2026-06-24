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

/* The SDL 2D rendering system */

#include "SDL_hints.h"
#include "SDL_render.h"
#include "SDL_timer.h"
#include "SDL_sysrender.h"
#include "software/SDL_render_sw_c.h"
#include "../video/SDL_pixels_c.h"

#if defined(__ANDROID__)
#include "../core/android/SDL_android.h"
#endif

/* as a courtesy to iOS apps, we don't try to draw when in the background, as
that will crash the app. However, these apps _should_ have used
SDL_AddEventWatch to catch SDL_APP_WILLENTERBACKGROUND events and stopped
drawing themselves. Other platforms still draw, as the compositor can use it,
and more importantly: drawing to render targets isn't lost. But I still think
this should probably be removed at some point in the future.  --ryan. */
#if defined(__IPHONEOS__) || defined(__TVOS__) || defined(__ANDROID__)
#define DONT_DRAW_WHILE_HIDDEN 1
#else
#define DONT_DRAW_WHILE_HIDDEN 0
#endif

#define SDL_WINDOWRENDERDATA "_SDL_WindowRenderData"

#define CHECK_RENDERER_MAGIC(renderer, retval)             \
    if (!renderer || renderer->magic != &renderer_magic) { \
        SDL_InvalidParamError("renderer");                 \
        return retval;                                     \
    }

#define CHECK_TEXTURE_MAGIC(texture, retval)            \
    if (!texture || texture->magic != &texture_magic) { \
        SDL_InvalidParamError("texture");               \
        return retval;                                  \
    }

/* Predefined blend modes */
#define SDL_COMPOSE_BLENDMODE(srcColorFactor, dstColorFactor, colorOperation, \
                              srcAlphaFactor, dstAlphaFactor, alphaOperation) \
    (SDL_BlendMode)(((Uint32)colorOperation << 0) |                           \
                    ((Uint32)srcColorFactor << 4) |                           \
                    ((Uint32)dstColorFactor << 8) |                           \
                    ((Uint32)alphaOperation << 16) |                          \
                    ((Uint32)srcAlphaFactor << 20) |                          \
                    ((Uint32)dstAlphaFactor << 24))

#define SDL_BLENDMODE_NONE_FULL                                                              \
    SDL_COMPOSE_BLENDMODE(SDL_BLENDFACTOR_ONE, SDL_BLENDFACTOR_ZERO, SDL_BLENDOPERATION_ADD, \
                          SDL_BLENDFACTOR_ONE, SDL_BLENDFACTOR_ZERO, SDL_BLENDOPERATION_ADD)

#define SDL_BLENDMODE_BLEND_FULL                                                                                  \
    SDL_COMPOSE_BLENDMODE(SDL_BLENDFACTOR_SRC_ALPHA, SDL_BLENDFACTOR_ONE_MINUS_SRC_ALPHA, SDL_BLENDOPERATION_ADD, \
                          SDL_BLENDFACTOR_ONE, SDL_BLENDFACTOR_ONE_MINUS_SRC_ALPHA, SDL_BLENDOPERATION_ADD)

#define SDL_BLENDMODE_ADD_FULL                                                                    \
    SDL_COMPOSE_BLENDMODE(SDL_BLENDFACTOR_SRC_ALPHA, SDL_BLENDFACTOR_ONE, SDL_BLENDOPERATION_ADD, \
                          SDL_BLENDFACTOR_ZERO, SDL_BLENDFACTOR_ONE, SDL_BLENDOPERATION_ADD)

#define SDL_BLENDMODE_MOD_FULL                                                                     \
    SDL_COMPOSE_BLENDMODE(SDL_BLENDFACTOR_ZERO, SDL_BLENDFACTOR_SRC_COLOR, SDL_BLENDOPERATION_ADD, \
                          SDL_BLENDFACTOR_ZERO, SDL_BLENDFACTOR_ONE, SDL_BLENDOPERATION_ADD)

#define SDL_BLENDMODE_MUL_FULL                                                                                    \
    SDL_COMPOSE_BLENDMODE(SDL_BLENDFACTOR_DST_COLOR, SDL_BLENDFACTOR_ONE_MINUS_SRC_ALPHA, SDL_BLENDOPERATION_ADD, \
                          SDL_BLENDFACTOR_ZERO, SDL_BLENDFACTOR_ONE, SDL_BLENDOPERATION_ADD)

#ifndef SDL_RENDER_DISABLED
static const SDL_RenderDriver *render_drivers[] = {
#if SDL_VIDEO_RENDER_D3D
    &D3D_RenderDriver,
#endif
#if SDL_VIDEO_RENDER_D3D11
    &D3D11_RenderDriver,
#endif
#if SDL_VIDEO_RENDER_D3D12
    &D3D12_RenderDriver,
#endif
#if SDL_VIDEO_RENDER_METAL
    &METAL_RenderDriver,
#endif
#if SDL_VIDEO_RENDER_OGL
    &GL_RenderDriver,
#endif
#if SDL_VIDEO_RENDER_OGL_ES2
    &GLES2_RenderDriver,
#endif
#if SDL_VIDEO_RENDER_OGL_ES
    &GLES_RenderDriver,
#endif
#if SDL_VIDEO_RENDER_DIRECTFB
    &DirectFB_RenderDriver,
#endif
#if SDL_VIDEO_RENDER_PS2
    &PS2_RenderDriver,
#endif
#if SDL_VIDEO_RENDER_PSP
    &PSP_RenderDriver,
#endif
#if SDL_VIDEO_RENDER_VITA_GXM
    &VITA_GXM_RenderDriver,
#endif
#if SDL_VIDEO_RENDER_SW
    &SW_RenderDriver
#endif
};
#endif /* !SDL_RENDER_DISABLED */

static char renderer_magic;
static char texture_magic;

static SDL_INLINE void DebugLogRenderCommands(const SDL_RenderCommand *cmd)
{
#if 0
    unsigned int i = 1;
    SDL_Log("Render commands to flush:");
    while (cmd) {
        switch (cmd->command) {
            case SDL_RENDERCMD_NO_OP:
                SDL_Log(" %u. no-op", i++);
                break;

            case SDL_RENDERCMD_SETVIEWPORT:
                SDL_Log(" %u. set viewport (first=%u, rect={(%d, %d), %dx%d})", i++,
                        (unsigned int) cmd->data.viewport.first,
                        cmd->data.viewport.rect.x, cmd->data.viewport.rect.y,
                        cmd->data.viewport.rect.w, cmd->data.viewport.rect.h);
                break;

            case SDL_RENDERCMD_SETCLIPRECT:
                SDL_Log(" %u. set cliprect (enabled=%s, rect={(%d, %d), %dx%d})", i++,
                        cmd->data.cliprect.enabled ? "true" : "false",
                        cmd->data.cliprect.rect.x, cmd->data.cliprect.rect.y,
                        cmd->data.cliprect.rect.w, cmd->data.cliprect.rect.h);
                break;

            case SDL_RENDERCMD_SETDRAWCOLOR:
                SDL_Log(" %u. set draw color (first=%u, r=%d, g=%d, b=%d, a=%d)", i++,
                        (unsigned int) cmd->data.color.first,
                        (int) cmd->data.color.r, (int) cmd->data.color.g,
                        (int) cmd->data.color.b, (int) cmd->data.color.a);
                break;

            case SDL_RENDERCMD_CLEAR:
                SDL_Log(" %u. clear (first=%u, r=%d, g=%d, b=%d, a=%d)", i++,
                        (unsigned int) cmd->data.color.first,
                        (int) cmd->data.color.r, (int) cmd->data.color.g,
                        (int) cmd->data.color.b, (int) cmd->data.color.a);
                break;

            case SDL_RENDERCMD_DRAW_POINTS:
                SDL_Log(" %u. draw points (first=%u, count=%u, r=%d, g=%d, b=%d, a=%d, blend=%d)", i++,
                        (unsigned int) cmd->data.draw.first,
                        (unsigned int) cmd->data.draw.count,
                        (int) cmd->data.draw.r, (int) cmd->data.draw.g,
                        (int) cmd->data.draw.b, (int) cmd->data.draw.a,
                        (int) cmd->data.draw.blend);
                break;

            case SDL_RENDERCMD_DRAW_LINES:
                SDL_Log(" %u. draw lines (first=%u, count=%u, r=%d, g=%d, b=%d, a=%d, blend=%d)", i++,
                        (unsigned int) cmd->data.draw.first,
                        (unsigned int) cmd->data.draw.count,
                        (int) cmd->data.draw.r, (int) cmd->data.draw.g,
                        (int) cmd->data.draw.b, (int) cmd->data.draw.a,
                        (int) cmd->data.draw.blend);
                break;

            case SDL_RENDERCMD_FILL_RECTS:
                SDL_Log(" %u. fill rects (first=%u, count=%u, r=%d, g=%d, b=%d, a=%d, blend=%d)", i++,
                        (unsigned int) cmd->data.draw.first,
                        (unsigned int) cmd->data.draw.count,
                        (int) cmd->data.draw.r, (int) cmd->data.draw.g,
                        (int) cmd->data.draw.b, (int) cmd->data.draw.a,
                        (int) cmd->data.draw.blend);
                break;

            case SDL_RENDERCMD_COPY:
                SDL_Log(" %u. copy (first=%u, count=%u, r=%d, g=%d, b=%d, a=%d, blend=%d, tex=%p)", i++,
                        (unsigned int) cmd->data.draw.first,
                        (unsigned int) cmd->data.draw.count,
                        (int) cmd->data.draw.r, (int) cmd->data.draw.g,
                        (int) cmd->data.draw.b, (int) cmd->data.draw.a,
                        (int) cmd->data.draw.blend, cmd->data.draw.texture);
                break;


            case SDL_RENDERCMD_COPY_EX:
                SDL_Log(" %u. copyex (first=%u, count=%u, r=%d, g=%d, b=%d, a=%d, blend=%d, tex=%p)", i++,
                        (unsigned int) cmd->data.draw.first,
                        (unsigned int) cmd->data.draw.count,
                        (int) cmd->data.draw.r, (int) cmd->data.draw.g,
                        (int) cmd->data.draw.b, (int) cmd->data.draw.a,
                        (int) cmd->data.draw.blend, cmd->data.draw.texture);
                break;

            case SDL_RENDERCMD_GEOMETRY:
                SDL_Log(" %u. geometry (first=%u, count=%u, r=%d, g=%d, b=%d, a=%d, blend=%d, tex=%p)", i++,
                        (unsigned int) cmd->data.draw.first,
                        (unsigned int) cmd->data.draw.count,
                        (int) cmd->data.draw.r, (int) cmd->data.draw.g,
                        (int) cmd->data.draw.b, (int) cmd->data.draw.a,
                        (int) cmd->data.draw.blend, cmd->data.draw.texture);
                break;

        }
        cmd = cmd->next;
    }
#endif
}

static int FlushRenderCommands(SDL_Renderer *renderer)
{
    int retval;

    SDL_assert((renderer->render_commands == NULL) == (renderer->render_commands_tail == NULL));

    if (!renderer->render_commands) { /* nothing to do! */
        SDL_assert(renderer->vertex_data_used == 0);
        return 0;
    }

    DebugLogRenderCommands(renderer->render_commands);

    retval = renderer->RunCommandQueue(renderer, renderer->render_commands, renderer->vertex_data, renderer->vertex_data_used);

    /* Move the whole render command queue to the unused pool so we can reuse them next time. */
    if (renderer->render_commands_tail) {
        renderer->render_commands_tail->next = renderer->render_commands_pool;
        renderer->render_commands_pool = renderer->render_commands;
        renderer->render_commands_tail = NULL;
        renderer->render_commands = NULL;
    }
    renderer->vertex_data_used = 0;
    renderer->render_command_generation++;
    renderer->color_queued = SDL_FALSE;
    renderer->viewport_queued = SDL_FALSE;
    renderer->cliprect_queued = SDL_FALSE;
    return retval;
}

static int FlushRenderCommandsIfTextureNeeded(SDL_Texture *texture)
{
    SDL_Renderer *renderer = texture->renderer;
    if (texture->last_command_generation == renderer->render_command_generation) {
        /* the current command queue depends on this texture, flush the queue now before it changes */
        return FlushRenderCommands(renderer);
    }
    return 0;
}

static SDL_INLINE int FlushRenderCommandsIfNotBatching(SDL_Renderer *renderer)
{
    return renderer->batching ? 0 : FlushRenderCommands(renderer);
}

int SDL_RenderFlush(SDL_Renderer *renderer)
{
    return FlushRenderCommands(renderer);
}

void *SDL_AllocateRenderVertices(SDL_Renderer *renderer, const size_t numbytes, const size_t alignment, size_t *offset)
{
    const size_t needed = renderer->vertex_data_used + numbytes + alignment;
    const size_t current_offset = renderer->vertex_data_used;

    const size_t aligner = (alignment && ((current_offset & (alignment - 1)) != 0)) ? (alignment - (current_offset & (alignment - 1))) : 0;
    const size_t aligned = current_offset + aligner;

    if (renderer->vertex_data_allocation < needed) {
        const size_t current_allocation = renderer->vertex_data ? renderer->vertex_data_allocation : 1024;
        size_t newsize = current_allocation * 2;
        void *ptr;
        while (newsize < needed) {
            newsize *= 2;
        }

        ptr = SDL_realloc(renderer->vertex_data, newsize);

        if (!ptr) {
            SDL_OutOfMemory();
            return NULL;
        }
        renderer->vertex_data = ptr;
        renderer->vertex_data_allocation = newsize;
    }

    if (offset) {
        *offset = aligned;
    }

    renderer->vertex_data_used += aligner + numbytes;

    return ((Uint8 *)renderer->vertex_data) + aligned;
}

static SDL_RenderCommand *AllocateRenderCommand(SDL_Renderer *renderer)
{
    SDL_RenderCommand *retval = NULL;

    /* !!! FIXME: are there threading limitations in SDL's render API? If not, we need to mutex this. */
    retval = renderer->render_commands_pool;
    if (retval) {
        renderer->render_commands_pool = retval->next;
        retval->next = NULL;
    } else {
        retval = SDL_calloc(1, sizeof(*retval));
        if (!retval) {
            SDL_OutOfMemory();
            return NULL;
        }
    }

    SDL_assert((renderer->render_commands == NULL) == (renderer->render_commands_tail == NULL));
    if (renderer->render_commands_tail) {
        renderer->render_commands_tail->next = retval;
    } else {
        renderer->render_commands = retval;
    }
    renderer->render_commands_tail = retval;

    return retval;
}

static int QueueCmdSetViewport(SDL_Renderer *renderer)
{
    int retval = 0;
    if (!renderer->viewport_queued || (SDL_memcmp(&renderer->viewport, &renderer->last_queued_viewport, sizeof(SDL_DRect)) != 0)) {
        SDL_RenderCommand *cmd = AllocateRenderCommand(renderer);
        retval = -1;
        if (cmd) {
            cmd->command = SDL_RENDERCMD_SETVIEWPORT;
            cmd->data.viewport.first = 0; /* render backend will fill this in. */
            /* Convert SDL_DRect to SDL_Rect */
            cmd->data.viewport.rect.x = (int)SDL_floor(renderer->viewport.x);
            cmd->data.viewport.rect.y = (int)SDL_floor(renderer->viewport.y);
            cmd->data.viewport.rect.w = (int)SDL_floor(renderer->viewport.w);
            cmd->data.viewport.rect.h = (int)SDL_floor(renderer->viewport.h);
            retval = renderer->QueueSetViewport(renderer, cmd);
            if (retval < 0) {
                cmd->command = SDL_RENDERCMD_NO_OP;
            } else {
                SDL_copyp(&renderer->last_queued_viewport, &renderer->viewport);
                renderer->viewport_queued = SDL_TRUE;
            }
        }
    }
    return retval;
}

static int QueueCmdSetClipRect(SDL_Renderer *renderer)
{
    int retval = 0;
    if ((!renderer->cliprect_queued) ||
        (renderer->clipping_enabled != renderer->last_queued_cliprect_enabled) ||
        (SDL_memcmp(&renderer->clip_rect, &renderer->last_queued_cliprect, sizeof(SDL_DRect)) != 0)) {
        SDL_RenderCommand *cmd = AllocateRenderCommand(renderer);
        if (!cmd) {
            retval = -1;
        } else {
            cmd->command = SDL_RENDERCMD_SETCLIPRECT;
            cmd->data.cliprect.enabled = renderer->clipping_enabled;
            /* Convert SDL_DRect to SDL_Rect */
            cmd->data.cliprect.rect.x = (int)SDL_floor(renderer->clip_rect.x);
            cmd->data.cliprect.rect.y = (int)SDL_floor(renderer->clip_rect.y);
            cmd->data.cliprect.rect.w = (int)SDL_floor(renderer->clip_rect.w);
            cmd->data.cliprect.rect.h = (int)SDL_floor(renderer->clip_rect.h);
            SDL_copyp(&renderer->last_queued_cliprect, &renderer->clip_rect);
            renderer->last_queued_cliprect_enabled = renderer->clipping_enabled;
            renderer->cliprect_queued = SDL_TRUE;
        }
    }
    return retval;
}

static int QueueCmdSetDrawColor(SDL_Renderer *renderer, SDL_Color *col)
{
    const Uint32 color = (((Uint32)col->a << 24) | (col->r << 16) | (col->g << 8) | col->b);
    int retval = 0;

    if (!renderer->color_queued || (color != renderer->last_queued_color)) {
        SDL_RenderCommand *cmd = AllocateRenderCommand(renderer);
        retval = -1;

        if (cmd) {
            cmd->command = SDL_RENDERCMD_SETDRAWCOLOR;
            cmd->data.color.first = 0; /* render backend will fill this in. */
            cmd->data.color.r = col->r;
            cmd->data.color.g = col->g;
            cmd->data.color.b = col->b;
            cmd->data.color.a = col->a;
            retval = renderer->QueueSetDrawColor(renderer, cmd);
            if (retval < 0) {
                cmd->command = SDL_RENDERCMD_NO_OP;
            } else {
                renderer->last_queued_color = color;
                renderer->color_queued = SDL_TRUE;
            }
        }
    }
    return retval;
}

static int QueueCmdClear(SDL_Renderer *renderer)
{
    SDL_RenderCommand *cmd = AllocateRenderCommand(renderer);
    if (!cmd) {
        return -1;
    }

    cmd->command = SDL_RENDERCMD_CLEAR;
    cmd->data.color.first = 0;
    cmd->data.color.r = renderer->color.r;
    cmd->data.color.g = renderer->color.g;
    cmd->data.color.b = renderer->color.b;
    cmd->data.color.a = renderer->color.a;
    return 0;
}

static SDL_RenderCommand *PrepQueueCmdDraw(SDL_Renderer *renderer, const SDL_RenderCommandType cmdtype, SDL_Texture *texture)
{
    SDL_RenderCommand *cmd = NULL;
    int retval = 0;
    SDL_Color *color;
    SDL_BlendMode blendMode;

    if (texture) {
        color = &texture->color;
        blendMode = texture->blendMode;
    } else {
        color = &renderer->color;
        blendMode = renderer->blendMode;
    }

    if (cmdtype != SDL_RENDERCMD_GEOMETRY) {
        /* !!! FIXME: drop this draw if viewport w or h is zero. */
        retval = QueueCmdSetDrawColor(renderer, color);
    }

    /* Set the viewport and clip rect directly before draws, so the backends
     * don't have to worry about that state not being valid at draw time. */
    if (retval == 0 && !renderer->viewport_queued) {
        retval = QueueCmdSetViewport(renderer);
    }
    if (retval == 0 && !renderer->cliprect_queued) {
        retval = QueueCmdSetClipRect(renderer);
    }

    if (retval == 0) {
        cmd = AllocateRenderCommand(renderer);
        if (cmd) {
            cmd->command = cmdtype;
            cmd->data.draw.first = 0; /* render backend will fill this in. */
            cmd->data.draw.count = 0; /* render backend will fill this in. */
            cmd->data.draw.r = color->r;
            cmd->data.draw.g = color->g;
            cmd->data.draw.b = color->b;
            cmd->data.draw.a = color->a;
            cmd->data.draw.blend = blendMode;
            cmd->data.draw.texture = texture;
        }
    }
    return cmd;
}

static int QueueCmdDrawPoints(SDL_Renderer *renderer, const SDL_FPoint *points, const int count)
{
    SDL_RenderCommand *cmd = PrepQueueCmdDraw(renderer, SDL_RENDERCMD_DRAW_POINTS, NULL);
    int retval = -1;
    if (cmd) {
        retval = renderer->QueueDrawPoints(renderer, cmd, points, count);
        if (retval < 0) {
            cmd->command = SDL_RENDERCMD_NO_OP;
        }
    }
    return retval;
}

static int QueueCmdDrawLines(SDL_Renderer *renderer, const SDL_FPoint *points, const int count)
{
    SDL_RenderCommand *cmd = PrepQueueCmdDraw(renderer, SDL_RENDERCMD_DRAW_LINES, NULL);
    int retval = -1;
    if (cmd) {
        retval = renderer->QueueDrawLines(renderer, cmd, points, count);
        if (retval < 0) {
            cmd->command = SDL_RENDERCMD_NO_OP;
        }
    }
    return retval;
}

static int QueueCmdFillRects(SDL_Renderer *renderer, const SDL_FRect *rects, const int count)
{
    SDL_RenderCommand *cmd;
    int retval = -1;
    const int use_rendergeometry = (!renderer->QueueFillRects);

    cmd = PrepQueueCmdDraw(renderer, (use_rendergeometry ? SDL_RENDERCMD_GEOMETRY : SDL_RENDERCMD_FILL_RECTS), NULL);

    if (cmd) {
        if (use_rendergeometry) {
            SDL_bool isstack1;
            SDL_bool isstack2;
            float *xy = SDL_small_alloc(float, 4 * 2 * count, &isstack1);
            int *indices = SDL_small_alloc(int, 6 * count, &isstack2);

            if (xy && indices) {
                int i;
                float *ptr_xy = xy;
                int *ptr_indices = indices;
                const int xy_stride = 2 * sizeof(float);
                const int num_vertices = 4 * count;
                const int num_indices = 6 * count;
                const int size_indices = 4;
                int cur_index = 0;
                const int *rect_index_order = renderer->rect_index_order;

                for (i = 0; i < count; ++i) {
                    float minx, miny, maxx, maxy;

                    minx = rects[i].x;
                    miny = rects[i].y;
                    maxx = rects[i].x + rects[i].w;
                    maxy = rects[i].y + rects[i].h;

                    *ptr_xy++ = minx;
                    *ptr_xy++ = miny;
                    *ptr_xy++ = maxx;
                    *ptr_xy++ = miny;
                    *ptr_xy++ = maxx;
                    *ptr_xy++ = maxy;
                    *ptr_xy++ = minx;
                    *ptr_xy++ = maxy;

                    *ptr_indices++ = cur_index + rect_index_order[0];
                    *ptr_indices++ = cur_index + rect_index_order[1];
                    *ptr_indices++ = cur_index + rect_index_order[2];
                    *ptr_indices++ = cur_index + rect_index_order[3];
                    *ptr_indices++ = cur_index + rect_index_order[4];
                    *ptr_indices++ = cur_index + rect_index_order[5];
                    cur_index += 4;
                }

                retval = renderer->QueueGeometry(renderer, cmd, NULL,
                                                 xy, xy_stride, &renderer->color, 0 /* color_stride */, NULL, 0,
                                                 num_vertices, indices, num_indices, size_indices,
                                                 1.0f, 1.0f);

                if (retval < 0) {
                    cmd->command = SDL_RENDERCMD_NO_OP;
                }
            }
            SDL_small_free(xy, isstack1);
            SDL_small_free(indices, isstack2);

        } else {
            retval = renderer->QueueFillRects(renderer, cmd, rects, count);
            if (retval < 0) {
                cmd->command = SDL_RENDERCMD_NO_OP;
            }
        }
    }
    return retval;
}

static int QueueCmdCopy(SDL_Renderer *renderer, SDL_Texture *texture, const SDL_Rect *srcrect, const SDL_FRect *dstrect)
{
    SDL_RenderCommand *cmd = PrepQueueCmdDraw(renderer, SDL_RENDERCMD_COPY, texture);
    int retval = -1;
    if (cmd) {
        retval = renderer->QueueCopy(renderer, cmd, texture, srcrect, dstrect);
        if (retval < 0) {
            cmd->command = SDL_RENDERCMD_NO_OP;
        }
    }
    return retval;
}

static int QueueCmdCopyEx(SDL_Renderer *renderer, SDL_Texture *texture,
                          const SDL_Rect *srcquad, const SDL_FRect *dstrect,
                          const double angle, const SDL_FPoint *center, const SDL_RendererFlip flip, float scale_x, float scale_y)
{
    SDL_RenderCommand *cmd = PrepQueueCmdDraw(renderer, SDL_RENDERCMD_COPY_EX, texture);
    int retval = -1;
    if (cmd) {
        retval = renderer->QueueCopyEx(renderer, cmd, texture, srcquad, dstrect, angle, center, flip, scale_x, scale_y);
        if (retval < 0) {
            cmd->command = SDL_RENDERCMD_NO_OP;
        }
    }
    return retval;
}

static int QueueCmdGeometry(SDL_Renderer *renderer, SDL_Texture *texture,
                            const float *xy, int xy_stride,
                            const SDL_Color *color, int color_stride,
                            const float *uv, int uv_stride,
                            int num_vertices,
                            const void *indices, int num_indices, int size_indices,
                            float scale_x, float scale_y)
{
    SDL_RenderCommand *cmd;
    int retval = -1;
    cmd = PrepQueueCmdDraw(renderer, SDL_RENDERCMD_GEOMETRY, texture);
    if (cmd) {
        retval = renderer->QueueGeometry(renderer, cmd, texture,
                                         xy, xy_stride,
                                         color, color_stride, uv, uv_stride,
                                         num_vertices, indices, num_indices, size_indices,
                                         scale_x, scale_y);
        if (retval < 0) {
            cmd->command = SDL_RENDERCMD_NO_OP;
        }
    }
    return retval;
}

static int UpdateLogicalSize(SDL_Renderer *renderer, SDL_bool flush_viewport_cmd);

int SDL_GetNumRenderDrivers(void)
{
#ifndef SDL_RENDER_DISABLED
    return SDL_arraysize(render_drivers);
#else
    return 0;
#endif
}

int SDL_GetRenderDriverInfo(int index, SDL_RendererInfo *info)
{
#ifndef SDL_RENDER_DISABLED
    if (index < 0 || index >= SDL_GetNumRenderDrivers()) {
        return SDL_SetError("index must be in the range of 0 - %d",
                            SDL_GetNumRenderDrivers() - 1);
    }
    *info = render_drivers[index]->info;
    return 0;
#else
    return SDL_SetError("SDL not built with rendering support");
#endif
}

static void GetWindowViewportValues(SDL_Renderer *renderer, int *logical_w, int *logical_h, SDL_DRect *viewport, SDL_FPoint *scale)
{
    SDL_LockMutex(renderer->target_mutex);
    *logical_w = renderer->target ? renderer->logical_w_backup : renderer->logical_w;
    *logical_h = renderer->target ? renderer->logical_h_backup : renderer->logical_h;
    *viewport = renderer->target ? renderer->viewport_backup : renderer->viewport;
    *scale = renderer->target ? renderer->scale_backup : renderer->scale;
    SDL_UnlockMutex(renderer->target_mutex);
}

static int SDLCALL SDL_RendererEventWatch(void *userdata, SDL_Event *event)
{
    SDL_Renderer *renderer = (SDL_Renderer *)userdata;

    if (event->type == SDL_WINDOWEVENT) {
        SDL_Window *window = SDL_GetWindowFromID(event->window.windowID);
        if (window == renderer->window) {
            if (renderer->WindowEvent) {
                renderer->WindowEvent(renderer, &event->window);
            }

            /* In addition to size changes, we also want to do this block for
             * window display changes as well! If the new display has a new DPI,
             * we need to update the viewport for the new window/drawable ratio.
             */
            if (event->window.event == SDL_WINDOWEVENT_SIZE_CHANGED ||
                event->window.event == SDL_WINDOWEVENT_DISPLAY_CHANGED) {
                /* Make sure we're operating on the default render target */
                SDL_Texture *saved_target = SDL_GetRenderTarget(renderer);
                if (saved_target) {
                    SDL_SetRenderTarget(renderer, NULL);
                }

                /* Update the DPI scale if the window has been resized. */
                if (window && renderer->GetOutputSize) {
                    int window_w, window_h;
                    int output_w, output_h;
                    if (renderer->GetOutputSize(renderer, &output_w, &output_h) == 0) {
                        SDL_GetWindowSize(renderer->window, &window_w, &window_h);
                        renderer->dpi_scale.x = (float)window_w / output_w;
                        renderer->dpi_scale.y = (float)window_h / output_h;
                    }
                }

                if (renderer->logical_w) {
#if defined(__ANDROID__)
                    /* Don't immediatly flush because the app may be in
                     * background, and the egl context shouldn't be used. */
                    SDL_bool flush_viewport_cmd = SDL_FALSE;
#else
                    SDL_bool flush_viewport_cmd = SDL_TRUE;
#endif
                    UpdateLogicalSize(renderer, flush_viewport_cmd);
                } else {
                    /* Window was resized, reset viewport */
                    int w, h;

                    if (renderer->GetOutputSize) {
                        renderer->GetOutputSize(renderer, &w, &h);
                    } else {
                        SDL_GetWindowSize(renderer->window, &w, &h);
                    }

                    renderer->viewport.x = (double)0;
                    renderer->viewport.y = (double)0;
                    renderer->viewport.w = (double)w;
                    renderer->viewport.h = (double)h;
                    QueueCmdSetViewport(renderer);
#if defined(__ANDROID__)
                    /* Don't immediatly flush because the app may be in
                     * background, and the egl context shouldn't be used. */
#else
                    FlushRenderCommandsIfNotBatching(renderer);
#endif
                }

                if (saved_target) {
                    SDL_SetRenderTarget(renderer, saved_target);
                }
            } else if (event->window.event == SDL_WINDOWEVENT_HIDDEN) {
                renderer->hidden = SDL_TRUE;
            } else if (event->window.event == SDL_WINDOWEVENT_SHOWN) {
                if (!(SDL_GetWindowFlags(window) & SDL_WINDOW_MINIMIZED)) {
                    renderer->hidden = SDL_FALSE;
                }
            } else if (event->window.event == SDL_WINDOWEVENT_MINIMIZED) {
                renderer->hidden = SDL_TRUE;
            } else if (event->window.event == SDL_WINDOWEVENT_RESTORED ||
                       event->window.event == SDL_WINDOWEVENT_MAXIMIZED) {
                if (!(SDL_GetWindowFlags(window) & SDL_WINDOW_HIDDEN)) {
                    renderer->hidden = SDL_FALSE;
                }
            }
        }
    } else if (event->type == SDL_MOUSEMOTION) {
        SDL_Window *window = SDL_GetWindowFromID(event->motion.windowID);
        if (window == renderer->window) {
            int logical_w, logical_h;
            SDL_DRect viewport;
            SDL_FPoint scale;
            GetWindowViewportValues(renderer, &logical_w, &logical_h, &viewport, &scale);
            if (logical_w) {
                event->motion.x -= (int)(viewport.x * renderer->dpi_scale.x);
                event->motion.y -= (int)(viewport.y * renderer->dpi_scale.y);
                event->motion.x = (int)(event->motion.x / (scale.x * renderer->dpi_scale.x));
                event->motion.y = (int)(event->motion.y / (scale.y * renderer->dpi_scale.y));
                if (event->motion.xrel != 0 && renderer->relative_scaling) {
                    float rel = renderer->xrel + event->motion.xrel / (scale.x * renderer->dpi_scale.x);
                    float truncated = SDL_truncf(rel);
                    renderer->xrel = rel - truncated;
                    event->motion.xrel = (Sint32)truncated;
                }
                if (event->motion.yrel != 0 && renderer->relative_scaling) {
                    float rel = renderer->yrel + event->motion.yrel / (scale.y * renderer->dpi_scale.y);
                    float truncated = SDL_truncf(rel);
                    renderer->yrel = rel - truncated;
                    event->motion.yrel = (Sint32)truncated;
                }
            }
        }
    } else if (event->type == SDL_MOUSEBUTTONDOWN ||
               event->type == SDL_MOUSEBUTTONUP) {
        SDL_Window *window = SDL_GetWindowFromID(event->button.windowID);
        if (window == renderer->window) {
            int logical_w, logical_h;
            SDL_DRect viewport;
            SDL_FPoint scale;
            GetWindowViewportValues(renderer, &logical_w, &logical_h, &viewport, &scale);
            if (logical_w) {
                event->button.x -= (int)(viewport.x * renderer->dpi_scale.x);
                event->button.y -= (int)(viewport.y * renderer->dpi_scale.y);
                event->button.x = (int)(event->button.x / (scale.x * renderer->dpi_scale.x));
                event->button.y = (int)(event->button.y / (scale.y * renderer->dpi_scale.y));
            }
        }
    } else if (event->type == SDL_MOUSEWHEEL) {
        SDL_Window *window = SDL_GetWindowFromID(event->wheel.windowID);
        if (window == renderer->window) {
            int logical_w, logical_h;
            SDL_DRect viewport;
            SDL_FPoint scale;
            GetWindowViewportValues(renderer, &logical_w, &logical_h, &viewport, &scale);
            if (logical_w) {
                event->wheel.mouseX -= (int)(viewport.x * renderer->dpi_scale.x);
                event->wheel.mouseY -= (int)(viewport.y * renderer->dpi_scale.y);
                event->wheel.mouseX = (int)(event->wheel.mouseX / (scale.x * renderer->dpi_scale.x));
                event->wheel.mouseY = (int)(event->wheel.mouseY / (scale.y * renderer->dpi_scale.y));
            }
        }
    } else if (event->type == SDL_FINGERDOWN ||
               event->type == SDL_FINGERUP ||
               event->type == SDL_FINGERMOTION) {
        int logical_w, logical_h;
        float physical_w, physical_h;
        SDL_DRect viewport;
        SDL_FPoint scale;
        GetWindowViewportValues(renderer, &logical_w, &logical_h, &viewport, &scale);

        /* !!! FIXME: we probably should drop events that are outside of the
           !!! FIXME: viewport, but we can't do that from an event watcher,
           !!! FIXME: and we would have to track if a touch happened outside
           !!! FIXME: the viewport and then slid into it to insert extra
           !!! FIXME: events, which is a mess, so for now we just clamp these
           !!! FIXME: events to the edge. */

        if (renderer->GetOutputSize) {
            int w, h;
            renderer->GetOutputSize(renderer, &w, &h);
            physical_w = (float)w;
            physical_h = (float)h;
        } else {
            int w, h;
            SDL_GetWindowSize(renderer->window, &w, &h);
            physical_w = ((float)w) * renderer->dpi_scale.x;
            physical_h = ((float)h) * renderer->dpi_scale.y;
        }

        if (physical_w == 0.0f) { /* nowhere for the touch to go, avoid division by zero and put it dead center. */
            event->tfinger.x = 0.5f;
        } else {
            const float normalized_viewport_x = ((float)viewport.x) / physical_w;
            const float normalized_viewport_w = ((float)viewport.w) / physical_w;
            if (event->tfinger.x <= normalized_viewport_x) {
                event->tfinger.x = 0.0f; /* to the left of the viewport, clamp to the edge. */
            } else if (event->tfinger.x >= (normalized_viewport_x + normalized_viewport_w)) {
                event->tfinger.x = 1.0f; /* to the right of the viewport, clamp to the edge. */
            } else {
                event->tfinger.x = (event->tfinger.x - normalized_viewport_x) / normalized_viewport_w;
            }
        }

        if (physical_h == 0.0f) { /* nowhere for the touch to go, avoid division by zero and put it dead center. */
            event->tfinger.y = 0.5f;
        } else {
            const float normalized_viewport_y = ((float)viewport.y) / physical_h;
            const float normalized_viewport_h = ((float)viewport.h) / physical_h;
            if (event->tfinger.y <= normalized_viewport_y) {
                event->tfinger.y = 0.0f; /* to the left of the viewport, clamp to the edge. */
            } else if (event->tfinger.y >= (normalized_viewport_y + normalized_viewport_h)) {
                event->tfinger.y = 1.0f; /* to the right of the viewport, clamp to the edge. */
            } else {
                event->tfinger.y = (event->tfinger.y - normalized_viewport_y) / normalized_viewport_h;
            }
        }
    }

    return 0;
}

int SDL_CreateWindowAndRenderer(int width, int height, Uint32 window_flags,
                                SDL_Window **window, SDL_Renderer **renderer)
{
    *window = SDL_CreateWindow(NULL, SDL_WINDOWPOS_UNDEFINED,
                               SDL_WINDOWPOS_UNDEFINED,
                               width, height, window_flags);
    if (!*window) {
        *renderer = NULL;
        return -1;
    }

    *renderer = SDL_CreateRenderer(*window, -1, 0);
    if (!*renderer) {
        return -1;
    }

    return 0;
}

#ifndef SDL_RENDER_DISABLED
static SDL_INLINE void VerifyDrawQueueFunctions(const SDL_Renderer *renderer)
{
    /* all of these functions are required to be implemented, even as no-ops, so we don't
        have to check that they aren't NULL over and over. */
    SDL_assert(renderer->QueueSetViewport != NULL);
    SDL_assert(renderer->QueueSetDrawColor != NULL);
    SDL_assert(renderer->QueueDrawPoints != NULL);
    SDL_assert(renderer->QueueDrawLines != NULL || renderer->QueueGeometry != NULL);
    SDL_assert(renderer->QueueFillRects != NULL || renderer->QueueGeometry != NULL);
    SDL_assert(renderer->QueueCopy != NULL || renderer->QueueGeometry != NULL);
    SDL_assert(renderer->RunCommandQueue != NULL);
}

static SDL_RenderLineMethod SDL_GetRenderLineMethod(void)
{
    const char *hint = SDL_GetHint(SDL_HINT_RENDER_LINE_METHOD);

    int method = 0;
    if (hint) {
        method = SDL_atoi(hint);
    }
    switch (method) {
    case 1:
        return SDL_RENDERLINEMETHOD_POINTS;
    case 2:
        return SDL_RENDERLINEMETHOD_LINES;
    case 3:
        return SDL_RENDERLINEMETHOD_GEOMETRY;
    default:
        return SDL_RENDERLINEMETHOD_POINTS;
    }
}

static void SDL_CalculateSimulatedVSyncInterval(SDL_Renderer *renderer, SDL_Window *window)
{
    /* FIXME: SDL refresh rate API should return numerator/denominator */
    int refresh_rate = 0;
    int display_index = SDL_GetWindowDisplayIndex(window);
    SDL_DisplayMode mode;

    if (display_index < 0) {
        display_index = 0;
    }
    if (SDL_GetDesktopDisplayMode(display_index, &mode) == 0) {
        refresh_rate = mode.refresh_rate;
    }
    if (!refresh_rate) {
        /* Pick a good default refresh rate */
        refresh_rate = 60;
    }
    renderer->simulate_vsync_interval = (1000 / refresh_rate);
}
#endif /* !SDL_RENDER_DISABLED */

SDL_Renderer *SDL_CreateRenderer(SDL_Window *window, int index, Uint32 flags)
{
#ifndef SDL_RENDER_DISABLED
    SDL_Renderer *renderer = NULL;
    int n = SDL_GetNumRenderDrivers();
    SDL_bool batching = SDL_TRUE;
    const char *hint;

#if defined(__ANDROID__)
    Android_ActivityMutex_Lock_Running();
#endif

    if (!window) {
        SDL_InvalidParamError("window");
        goto error;
    }

    if (SDL_HasWindowSurface(window)) {
        SDL_SetError("Surface already associated with window");
        goto error;
    }

    if (SDL_GetRenderer(window)) {
        SDL_SetError("Renderer already associated with window");
        goto error;
    }

    hint = SDL_GetHint(SDL_HINT_RENDER_VSYNC);
    if (hint && *hint) {
        if (SDL_GetHintBoolean(SDL_HINT_RENDER_VSYNC, SDL_TRUE)) {
            flags |= SDL_RENDERER_PRESENTVSYNC;
        } else {
            flags &= ~SDL_RENDERER_PRESENTVSYNC;
        }
    }

    if (index < 0) {
        hint = SDL_GetHint(SDL_HINT_RENDER_DRIVER);
        if (hint) {
            for (index = 0; index < n; ++index) {
                const SDL_RenderDriver *driver = render_drivers[index];

                if (SDL_strcasecmp(hint, driver->info.name) == 0) {
                    /* Create a new renderer instance */
                    renderer = driver->CreateRenderer(window, flags);
                    if (renderer) {
                        batching = SDL_FALSE;
                    }
                    break;
                }
            }
        }

        if (!renderer) {
            for (index = 0; index < n; ++index) {
                const SDL_RenderDriver *driver = render_drivers[index];

                if ((driver->info.flags & flags) == flags) {
                    /* Create a new renderer instance */
                    renderer = driver->CreateRenderer(window, flags);
                    if (renderer) {
                        /* Yay, we got one! */
                        break;
                    }
                }
            }
        }
        if (!renderer) {
            SDL_SetError("Couldn't find matching render driver");
            goto error;
        }
    } else {
        if (index >= n) {
            SDL_SetError("index must be -1 or in the range of 0 - %d",
                         n - 1);
            goto error;
        }
        /* Create a new renderer instance */
        renderer = render_drivers[index]->CreateRenderer(window, flags);
        batching = SDL_FALSE;
        if (!renderer) {
            goto error;
        }
    }

    if (flags & SDL_RENDERER_PRESENTVSYNC) {
        renderer->wanted_vsync = SDL_TRUE;

        if (!(renderer->info.flags & SDL_RENDERER_PRESENTVSYNC)) {
            renderer->simulate_vsync = SDL_TRUE;
            renderer->info.flags |= SDL_RENDERER_PRESENTVSYNC;
        }
    }
    SDL_CalculateSimulatedVSyncInterval(renderer, window);

    VerifyDrawQueueFunctions(renderer);

    /* let app/user override batching decisions. */
    if (renderer->always_batch) {
        batching = SDL_TRUE;
    } else if (SDL_GetHint(SDL_HINT_RENDER_BATCHING)) {
        batching = SDL_GetHintBoolean(SDL_HINT_RENDER_BATCHING, SDL_TRUE);
    }

    renderer->batching = batching;
    renderer->magic = &renderer_magic;
    renderer->window = window;
    renderer->target_mutex = SDL_CreateMutex();
    renderer->scale.x = 1.0f;
    renderer->scale.y = 1.0f;
    renderer->dpi_scale.x = 1.0f;
    renderer->dpi_scale.y = 1.0f;

    /* Default value, if not specified by the renderer back-end */
    if (renderer->rect_index_order[0] == 0 && renderer->rect_index_order[1] == 0) {
        renderer->rect_index_order[0] = 0;
        renderer->rect_index_order[1] = 1;
        renderer->rect_index_order[2] = 2;
        renderer->rect_index_order[3] = 0;
        renderer->rect_index_order[4] = 2;
        renderer->rect_index_order[5] = 3;
    }

    /* new textures start at zero, so we start at 1 so first render doesn't flush by accident. */
    renderer->render_command_generation = 1;

    if (renderer->GetOutputSize) {
        int window_w, window_h;
        int output_w, output_h;
        if (renderer->GetOutputSize(renderer, &output_w, &output_h) == 0) {
            SDL_GetWindowSize(renderer->window, &window_w, &window_h);
            renderer->dpi_scale.x = (float)window_w / output_w;
            renderer->dpi_scale.y = (float)window_h / output_h;
        }
    }

    renderer->relative_scaling = SDL_GetHintBoolean(SDL_HINT_MOUSE_RELATIVE_SCALING, SDL_TRUE);

    renderer->line_method = SDL_GetRenderLineMethod();

    if (SDL_GetWindowFlags(window) & (SDL_WINDOW_HIDDEN | SDL_WINDOW_MINIMIZED)) {
        renderer->hidden = SDL_TRUE;
    } else {
        renderer->hidden = SDL_FALSE;
    }

    SDL_SetWindowData(window, SDL_WINDOWRENDERDATA, renderer);

    SDL_RenderSetViewport(renderer, NULL);

    SDL_AddEventWatch(SDL_RendererEventWatch, renderer);

    SDL_LogInfo(SDL_LOG_CATEGORY_RENDER,
                "Created renderer: %s", renderer->info.name);

#if defined(__ANDROID__)
    Android_ActivityMutex_Unlock();
#endif
    return renderer;

error:

#if defined(__ANDROID__)
    Android_ActivityMutex_Unlock();
#endif
    return NULL;

#else
    SDL_SetError("SDL not built with rendering support");
    return NULL;
#endif
}

SDL_Renderer *SDL_CreateSoftwareRenderer(SDL_Surface *surface)
{
#if SDL_VIDEO_RENDER_SW
    SDL_Renderer *renderer;

    renderer = SW_CreateRendererForSurface(surface);

    if (renderer) {
        VerifyDrawQueueFunctions(renderer);
        renderer->magic = &renderer_magic;
        renderer->target_mutex = SDL_CreateMutex();
        renderer->scale.x = 1.0f;
        renderer->scale.y = 1.0f;

        /* new textures start at zero, so we start at 1 so first render doesn't flush by accident. */
        renderer->render_command_generation = 1;

        /* Software renderer always uses line method, for speed */
        renderer->line_method = SDL_RENDERLINEMETHOD_LINES;

        SDL_RenderSetViewport(renderer, NULL);
    }
    return renderer;
#else
    SDL_SetError("SDL not built with rendering support");
    return NULL;
#endif /* !SDL_RENDER_DISABLED */
}

SDL_Renderer *SDL_GetRenderer(SDL_Window *window)
{
    return (SDL_Renderer *)SDL_GetWindowData(window, SDL_WINDOWRENDERDATA);
}

SDL_Window *SDL_RenderGetWindow(SDL_Renderer *renderer)
{
    CHECK_RENDERER_MAGIC(renderer, NULL);
    return renderer->window;
}

int SDL_GetRendererInfo(SDL_Renderer *renderer, SDL_RendererInfo *info)
{
    CHECK_RENDERER_MAGIC(renderer, -1);

    *info = renderer->info;
    return 0;
}

int SDL_GetRendererOutputSize(SDL_Renderer *renderer, int *w, int *h)
{
    CHECK_RENDERER_MAGIC(renderer, -1);

    if (renderer->target) {
        return SDL_QueryTexture(renderer->target, NULL, NULL, w, h);
    } else if (renderer->GetOutputSize) {
        return renderer->GetOutputSize(renderer, w, h);
    } else if (renderer->window) {
        SDL_GetWindowSize(renderer->window, w, h);
        return 0;
    } else {
        SDL_assert(0 && "This should never happen");
        return SDL_SetError("Renderer doesn't support querying output size");
    }
}

static SDL_bool IsSupportedBlendMode(SDL_Renderer *renderer, SDL_BlendMode blendMode)
{
    switch (blendMode) {
    /* These are required to be supported by all renderers */
    case SDL_BLENDMODE_NONE:
    case SDL_BLENDMODE_BLEND:
    case SDL_BLENDMODE_ADD:
    case SDL_BLENDMODE_MOD:
    case SDL_BLENDMODE_MUL:
        return SDL_TRUE;

    default:
        return renderer->SupportsBlendMode && renderer->SupportsBlendMode(renderer, blendMode);
    }
}

static SDL_bool IsSupportedFormat(SDL_Renderer *renderer, Uint32 format)
{
    Uint32 i;

    for (i = 0; i < renderer->info.num_texture_formats; ++i) {
        if (renderer->info.texture_formats[i] == format) {
            return SDL_TRUE;
        }
    }
    return SDL_FALSE;
}

static Uint32 GetClosestSupportedFormat(SDL_Renderer *renderer, Uint32 format)
{
    Uint32 i;

    if (SDL_ISPIXELFORMAT_FOURCC(format)) {
        /* Look for an exact match */
        for (i = 0; i < renderer->info.num_texture_formats; ++i) {
            if (renderer->info.texture_formats[i] == format) {
                return renderer->info.texture_formats[i];
            }
        }
    } else {
        SDL_bool hasAlpha = SDL_ISPIXELFORMAT_ALPHA(format);

        /* We just want to match the first format that has the same channels */
        for (i = 0; i < renderer->info.num_texture_formats; ++i) {
            if (!SDL_ISPIXELFORMAT_FOURCC(renderer->info.texture_formats[i]) &&
                SDL_ISPIXELFORMAT_ALPHA(renderer->info.texture_formats[i]) == hasAlpha) {
                return renderer->info.texture_formats[i];
            }
        }
    }
    return renderer->info.texture_formats[0];
}

static SDL_ScaleMode SDL_GetScaleMode(void)
{
    const char *hint = SDL_GetHint(SDL_HINT_RENDER_SCALE_QUALITY);

    if (!hint || SDL_strcasecmp(hint, "nearest") == 0) {
        return SDL_ScaleModeNearest;
    } else if (SDL_strcasecmp(hint, "linear") == 0) {
        return SDL_ScaleModeLinear;
    } else if (SDL_strcasecmp(hint, "best") == 0) {
        return SDL_ScaleModeBest;
    } else {
        return (SDL_ScaleMode)SDL_atoi(hint);
    }
}

SDL_Texture *SDL_CreateTexture(SDL_Renderer *renderer, Uint32 format, int access, int w, int h)
{
    SDL_Texture *texture;
    SDL_bool texture_is_fourcc_and_target;

    CHECK_RENDERER_MAGIC(renderer, NULL);

    if (!format) {
        format = renderer->info.texture_formats[0];
    }
    if (SDL_BYTESPERPIXEL(format) == 0) {
        SDL_SetError("Invalid texture format");
        return NULL;
    }
    if (SDL_ISPIXELFORMAT_INDEXED(format)) {
        if (!IsSupportedFormat(renderer, format)) {
            SDL_SetError("Palettized textures are not supported");
            return NULL;
        }
    }
    if (w <= 0 || h <= 0) {
        SDL_SetError("Texture dimensions can't be 0");
        return NULL;
    }
    if ((renderer->info.max_texture_width && w > renderer->info.max_texture_width) ||
        (renderer->info.max_texture_height && h > renderer->info.max_texture_height)) {
        SDL_SetError("Texture dimensions are limited to %dx%d", renderer->info.max_texture_width, renderer->info.max_texture_height);
        return NULL;
    }
    texture = (SDL_Texture *)SDL_calloc(1, sizeof(*texture));
    if (!texture) {
        SDL_OutOfMemory();
        return NULL;
    }
    texture->magic = &texture_magic;
    texture->format = format;
    texture->access = access;
    texture->w = w;
    texture->h = h;
    texture->color.r = 255;
    texture->color.g = 255;
    texture->color.b = 255;
    texture->color.a = 255;
    texture->scaleMode = SDL_GetScaleMode();
    texture->renderer = renderer;
    texture->next = renderer->textures;
    if (renderer->textures) {
        renderer->textures->prev = texture;
    }
    renderer->textures = texture;

    /* FOURCC format cannot be used directly by renderer back-ends for target texture */
    texture_is_fourcc_and_target = (access == SDL_TEXTUREACCESS_TARGET && SDL_ISPIXELFORMAT_FOURCC(texture->format));

    if (texture_is_fourcc_and_target == SDL_FALSE && IsSupportedFormat(renderer, format)) {
        if (renderer->CreateTexture(renderer, texture) < 0) {
            SDL_DestroyTexture(texture);
            return NULL;
        }
    } else {
        int closest_format;

        if (texture_is_fourcc_and_target == SDL_FALSE) {
            closest_format = GetClosestSupportedFormat(renderer, format);
        } else {
            closest_format = renderer->info.texture_formats[0];
        }

        texture->native = SDL_CreateTexture(renderer, closest_format, access, w, h);
        if (!texture->native) {
            SDL_DestroyTexture(texture);
            return NULL;
        }

        /* Swap textures to have texture before texture->native in the list */
        texture->native->next = texture->next;
        if (texture->native->next) {
            texture->native->next->prev = texture->native;
        }
        texture->prev = texture->native->prev;
        if (texture->prev) {
            texture->prev->next = texture;
        }
        texture->native->prev = texture;
        texture->next = texture->native;
        renderer->textures = texture;

        if (SDL_ISPIXELFORMAT_FOURCC(texture->format)) {
#if SDL_HAVE_YUV
            texture->yuv = SDL_SW_CreateYUVTexture(format, w, h);
#else
            SDL_SetError("SDL not built with YUV support");
#endif
            if (!texture->yuv) {
                SDL_DestroyTexture(texture);
                return NULL;
            }
        } else if (access == SDL_TEXTUREACCESS_STREAMING) {
            /* The pitch is 4 byte aligned */
            texture->pitch = (((w * SDL_BYTESPERPIXEL(format)) + 3) & ~3);
            texture->pixels = SDL_calloc(1, (size_t)texture->pitch * h);
            if (!texture->pixels) {
                SDL_DestroyTexture(texture);
                return NULL;
            }
        }
    }
    return texture;
}

SDL_Texture *SDL_CreateTextureFromSurface(SDL_Renderer *renderer, SDL_Surface *surface)
{
    const SDL_PixelFormat *fmt;
    SDL_bool needAlpha;
    SDL_bool direct_update;
    int i;
    Uint32 format = SDL_PIXELFORMAT_UNKNOWN;
    SDL_Texture *texture;

    CHECK_RENDERER_MAGIC(renderer, NULL);

    if (!surface) {
        SDL_InvalidParamError("SDL_CreateTextureFromSurface(): surface");
        return NULL;
    }

    /* See what the best texture format is */
    fmt = surface->format;
    if (fmt->Amask || SDL_HasColorKey(surface)) {
        needAlpha = SDL_TRUE;
    } else {
        needAlpha = SDL_FALSE;
    }

    /* If Palette contains alpha values, promotes to alpha format */
    if (fmt->palette) {
        SDL_bool is_opaque, has_alpha_channel;
        SDL_DetectPalette(fmt->palette, &is_opaque, &has_alpha_channel);
        if (!is_opaque) {
            needAlpha = SDL_TRUE;
        }
    }

    /* Try to have the best pixel format for the texture */
    /* No alpha, but a colorkey => promote to alpha */
    if (!fmt->Amask && SDL_HasColorKey(surface)) {
        if (fmt->format == SDL_PIXELFORMAT_RGB888) {
            for (i = 0; i < (int)renderer->info.num_texture_formats; ++i) {
                if (renderer->info.texture_formats[i] == SDL_PIXELFORMAT_ARGB8888) {
                    format = SDL_PIXELFORMAT_ARGB8888;
                    break;
                }
            }
        } else if (fmt->format == SDL_PIXELFORMAT_BGR888) {
            for (i = 0; i < (int)renderer->info.num_texture_formats; ++i) {
                if (renderer->info.texture_formats[i] == SDL_PIXELFORMAT_ABGR8888) {
                    format = SDL_PIXELFORMAT_ABGR8888;
                    break;
                }
            }
        }
    } else {
        /* Exact match would be fine */
        for (i = 0; i < (int)renderer->info.num_texture_formats; ++i) {
            if (renderer->info.texture_formats[i] == fmt->format) {
                format = fmt->format;
                break;
            }
        }
    }

    /* Fallback, choose a valid pixel format */
    if (format == SDL_PIXELFORMAT_UNKNOWN) {
        format = renderer->info.texture_formats[0];
        for (i = 0; i < (int)renderer->info.num_texture_formats; ++i) {
            if (!SDL_ISPIXELFORMAT_FOURCC(renderer->info.texture_formats[i]) &&
                SDL_ISPIXELFORMAT_ALPHA(renderer->info.texture_formats[i]) == needAlpha) {
                format = renderer->info.texture_formats[i];
                break;
            }
        }
    }

    texture = SDL_CreateTexture(renderer, format, SDL_TEXTUREACCESS_STATIC,
                                surface->w, surface->h);
    if (!texture) {
        return NULL;
    }

    if (format == surface->format->format) {
        if (surface->format->Amask && SDL_HasColorKey(surface)) {
            /* Surface and Renderer formats are identicals.
             * Intermediate conversion is needed to convert color key to alpha (SDL_ConvertColorkeyToAlpha()). */
            direct_update = SDL_FALSE;
        } else {
            /* Update Texture directly */
            direct_update = SDL_TRUE;
        }
    } else {
        /* Surface and Renderer formats are differents, it needs an intermediate conversion. */
        direct_update = SDL_FALSE;
    }

    if (direct_update) {
        if (SDL_MUSTLOCK(surface)) {
            SDL_LockSurface(surface);
            SDL_UpdateTexture(texture, NULL, surface->pixels, surface->pitch);
            SDL_UnlockSurface(surface);
        } else {
            SDL_UpdateTexture(texture, NULL, surface->pixels, surface->pitch);
        }

#if SDL_VIDEO_RENDER_DIRECTFB
        /* DirectFB allows palette format for textures.
         * Copy SDL_Surface palette to the texture */
        if (SDL_ISPIXELFORMAT_INDEXED(format)) {
            if (SDL_strcasecmp(renderer->info.name, "directfb") == 0) {
                extern void DirectFB_SetTexturePalette(SDL_Renderer *renderer, SDL_Texture *texture, SDL_Palette *pal);
                DirectFB_SetTexturePalette(renderer, texture, surface->format->palette);
            }
        }
#endif

    } else {
        SDL_PixelFormat *dst_fmt;
        SDL_Surface *temp = NULL;

        /* Set up a destination surface for the texture update */
        dst_fmt = SDL_AllocFormat(format);
        if (!dst_fmt) {
            SDL_DestroyTexture(texture);
            return NULL;
        }
        temp = SDL_ConvertSurface(surface, dst_fmt, 0);
        SDL_FreeFormat(dst_fmt);
        if (temp) {
            SDL_UpdateTexture(texture, NULL, temp->pixels, temp->pitch);
            SDL_FreeSurface(temp);
        } else {
            SDL_DestroyTexture(texture);
            return NULL;
        }
    }

    {
        Uint8 r, g, b, a;
        SDL_BlendMode blendMode;

        SDL_GetSurfaceColorMod(surface, &r, &g, &b);
        SDL_SetTextureColorMod(texture, r, g, b);

        SDL_GetSurfaceAlphaMod(surface, &a);
        SDL_SetTextureAlphaMod(texture, a);

        if (SDL_HasColorKey(surface)) {
            /* We converted to a texture with alpha format */
            SDL_SetTextureBlendMode(texture, SDL_BLENDMODE_BLEND);
        } else {
            SDL_GetSurfaceBlendMode(surface, &blendMode);
            SDL_SetTextureBlendMode(texture, blendMode);
        }
    }
    return texture;
}

int SDL_QueryTexture(SDL_Texture *texture, Uint32 *format, int *access,
                     int *w, int *h)
{
    CHECK_TEXTURE_MAGIC(texture, -1);

    if (format) {
        *format = texture->format;
    }
    if (access) {
        *access = texture->access;
    }
    if (w) {
        *w = texture->w;
    }
    if (h) {
        *h = texture->h;
    }
    return 0;
}

int SDL_SetTextureColorMod(SDL_Texture *texture, Uint8 r, Uint8 g, Uint8 b)
{
    CHECK_TEXTURE_MAGIC(texture, -1);

    if (r < 255 || g < 255 || b < 255) {
        texture->modMode |= SDL_TEXTUREMODULATE_COLOR;
    } else {
        texture->modMode &= ~SDL_TEXTUREMODULATE_COLOR;
    }
    texture->color.r = r;
    texture->color.g = g;
    texture->color.b = b;
    if (texture->native) {
        return SDL_SetTextureColorMod(texture->native, r, g, b);
    }
    return 0;
}

int SDL_GetTextureColorMod(SDL_Texture *texture, Uint8 *r, Uint8 *g,
                           Uint8 *b)
{
    CHECK_TEXTURE_MAGIC(texture, -1);

    if (r) {
        *r = texture->color.r;
    }
    if (g) {
        *g = texture->color.g;
    }
    if (b) {
        *b = texture->color.b;
    }
    return 0;
}

int SDL_SetTextureAlphaMod(SDL_Texture *texture, Uint8 alpha)
{
    CHECK_TEXTURE_MAGIC(texture, -1);

    if (alpha < 255) {
        texture->modMode |= SDL_TEXTUREMODULATE_ALPHA;
    } else {
        texture->modMode &= ~SDL_TEXTUREMODULATE_ALPHA;
    }
    texture->color.a = alpha;
    if (texture->native) {
        return SDL_SetTextureAlphaMod(texture->native, alpha);
    }
    return 0;
}

int SDL_GetTextureAlphaMod(SDL_Texture *texture, Uint8 *alpha)
{
    CHECK_TEXTURE_MAGIC(texture, -1);

    if (alpha) {
        *alpha = texture->color.a;
    }
    return 0;
}

int SDL_SetTextureBlendMode(SDL_Texture *texture, SDL_BlendMode blendMode)
{
    SDL_Renderer *renderer;

    CHECK_TEXTURE_MAGIC(texture, -1);

    renderer = texture->renderer;
    if (!IsSupportedBlendMode(renderer, blendMode)) {
        return SDL_Unsupported();
    }
    texture->blendMode = blendMode;
    if (texture->native) {
        return SDL_SetTextureBlendMode(texture->native, blendMode);
    }
    return 0;
}

int SDL_GetTextureBlendMode(SDL_Texture *texture, SDL_BlendMode *blendMode)
{
    CHECK_TEXTURE_MAGIC(texture, -1);

    if (blendMode) {
        *blendMode = texture->blendMode;
    }
    return 0;
}

int SDL_SetTextureScaleMode(SDL_Texture *texture, SDL_ScaleMode scaleMode)
{
    SDL_Renderer *renderer;

    CHECK_TEXTURE_MAGIC(texture, -1);

    renderer = texture->renderer;
    texture->scaleMode = scaleMode;
    if (texture->native) {
        return SDL_SetTextureScaleMode(texture->native, scaleMode);
    } else {
        renderer->SetTextureScaleMode(renderer, texture, scaleMode);
    }
    return 0;
}

int SDL_GetTextureScaleMode(SDL_Texture *texture, SDL_ScaleMode *scaleMode)
{
    CHECK_TEXTURE_MAGIC(texture, -1);

    if (scaleMode) {
        *scaleMode = texture->scaleMode;
    }
    return 0;
}

int SDL_SetTextureUserData(SDL_Texture *texture, void *userdata)
{
    CHECK_TEXTURE_MAGIC(texture, -1);

    texture->userdata = userdata;
    return 0;
}

void *SDL_GetTextureUserData(SDL_Texture *texture)
{
    CHECK_TEXTURE_MAGIC(texture, NULL);

    return texture->userdata;
}

#if SDL_HAVE_YUV
static int SDL_UpdateTextureYUV(SDL_Texture *texture, const SDL_Rect *rect,
                                const void *pixels, int pitch)
{
    SDL_Texture *native = texture->native;
    SDL_Rect full_rect;

    if (SDL_SW_UpdateYUVTexture(texture->yuv, rect, pixels, pitch) < 0) {
        return -1;
    }

    full_rect.x = 0;
    full_rect.y = 0;
    full_rect.w = texture->w;
    full_rect.h = texture->h;
    rect = &full_rect;

    if (texture->access == SDL_TEXTUREACCESS_STREAMING) {
        /* We can lock the texture and copy to it */
        void *native_pixels = NULL;
        int native_pitch = 0;

        if (SDL_LockTexture(native, rect, &native_pixels, &native_pitch) < 0) {
            return -1;
        }
        SDL_SW_CopyYUVToRGB(texture->yuv, rect, native->format,
                            rect->w, rect->h, native_pixels, native_pitch);
        SDL_UnlockTexture(native);
    } else {
        /* Use a temporary buffer for updating */
        const int temp_pitch = (((rect->w * SDL_BYTESPERPIXEL(native->format)) + 3) & ~3);
        const size_t alloclen = (size_t)rect->h * temp_pitch;
        if (alloclen > 0) {
            void *temp_pixels = SDL_malloc(alloclen);
            if (!temp_pixels) {
                return SDL_OutOfMemory();
            }
            SDL_SW_CopyYUVToRGB(texture->yuv, rect, native->format,
                                rect->w, rect->h, temp_pixels, temp_pitch);
            SDL_UpdateTexture(native, rect, temp_pixels, temp_pitch);
            SDL_free(temp_pixels);
        }
    }
    return 0;
}
#endif /* SDL_HAVE_YUV */

static int SDL_UpdateTextureNative(SDL_Texture *texture, const SDL_Rect *rect,
                                   const void *pixels, int pitch)
{
    SDL_Texture *native = texture->native;

    if (!rect->w || !rect->h) {
        return 0; /* nothing to do. */
    }

    if (texture->access == SDL_TEXTUREACCESS_STREAMING) {
        /* We can lock the texture and copy to it */
        void *native_pixels = NULL;
        int native_pitch = 0;

        if (SDL_LockTexture(native, rect, &native_pixels, &native_pitch) < 0) {
            return -1;
        }
        SDL_ConvertPixels(rect->w, rect->h,
                          texture->format, pixels, pitch,
                          native->format, native_pixels, native_pitch);
        SDL_UnlockTexture(native);
    } else {
        /* Use a temporary buffer for updating */
        const int temp_pitch = (((rect->w * SDL_BYTESPERPIXEL(native->format)) + 3) & ~3);
        const size_t alloclen = (size_t)rect->h * temp_pitch;
        if (alloclen > 0) {
            void *temp_pixels = SDL_malloc(alloclen);
            if (!temp_pixels) {
                return SDL_OutOfMemory();
            }
            SDL_ConvertPixels(rect->w, rect->h,
                              texture->format, pixels, pitch,
                              native->format, temp_pixels, temp_pitch);
            SDL_UpdateTexture(native, rect, temp_pixels, temp_pitch);
            SDL_free(temp_pixels);
        }
    }
    return 0;
}

int SDL_UpdateTexture(SDL_Texture *texture, const SDL_Rect *rect,
                      const void *pixels, int pitch)
{
    SDL_Rect real_rect;

    CHECK_TEXTURE_MAGIC(texture, -1);

    if (!pixels) {
        return SDL_InvalidParamError("pixels");
    }
    if (!pitch) {
        return SDL_InvalidParamError("pitch");
    }

    real_rect.x = 0;
    real_rect.y = 0;
    real_rect.w = texture->w;
    real_rect.h = texture->h;
    if (rect) {
        if (!SDL_IntersectRect(rect, &real_rect, &real_rect)) {
            return 0;
        }
    }

    if (real_rect.w == 0 || real_rect.h == 0) {
        return 0; /* nothing to do. */
#if SDL_HAVE_YUV
    } else if (texture->yuv) {
        return SDL_UpdateTextureYUV(texture, &real_rect, pixels, pitch);
#endif
    } else if (texture->native) {
        return SDL_UpdateTextureNative(texture, &real_rect, pixels, pitch);
    } else {
        SDL_Renderer *renderer = texture->renderer;
        if (FlushRenderCommandsIfTextureNeeded(texture) < 0) {
            return -1;
        }
        return renderer->UpdateTexture(renderer, texture, &real_rect, pixels, pitch);
    }
}

#if SDL_HAVE_YUV
static int SDL_UpdateTextureYUVPlanar(SDL_Texture *texture, const SDL_Rect *rect,
                                      const Uint8 *Yplane, int Ypitch,
                                      const Uint8 *Uplane, int Upitch,
                                      const Uint8 *Vplane, int Vpitch)
{
    SDL_Texture *native = texture->native;
    SDL_Rect full_rect;

    if (SDL_SW_UpdateYUVTexturePlanar(texture->yuv, rect, Yplane, Ypitch, Uplane, Upitch, Vplane, Vpitch) < 0) {
        return -1;
    }

    full_rect.x = 0;
    full_rect.y = 0;
    full_rect.w = texture->w;
    full_rect.h = texture->h;
    rect = &full_rect;

    if (!rect->w || !rect->h) {
        return 0; /* nothing to do. */
    }

    if (texture->access == SDL_TEXTUREACCESS_STREAMING) {
        /* We can lock the texture and copy to it */
        void *native_pixels = NULL;
        int native_pitch = 0;

        if (SDL_LockTexture(native, rect, &native_pixels, &native_pitch) < 0) {
            return -1;
        }
        SDL_SW_CopyYUVToRGB(texture->yuv, rect, native->format,
                            rect->w, rect->h, native_pixels, native_pitch);
        SDL_UnlockTexture(native);
    } else {
        /* Use a temporary buffer for updating */
        const int temp_pitch = (((rect->w * SDL_BYTESPERPIXEL(native->format)) + 3) & ~3);
        const size_t alloclen = (size_t)rect->h * temp_pitch;
        if (alloclen > 0) {
            void *temp_pixels = SDL_malloc(alloclen);
            if (!temp_pixels) {
                return SDL_OutOfMemory();
            }
            SDL_SW_CopyYUVToRGB(texture->yuv, rect, native->format,
                                rect->w, rect->h, temp_pixels, temp_pitch);
            SDL_UpdateTexture(native, rect, temp_pixels, temp_pitch);
            SDL_free(temp_pixels);
        }
    }
    return 0;
}

static int SDL_UpdateTextureNVPlanar(SDL_Texture *texture, const SDL_Rect *rect,
                                     const Uint8 *Yplane, int Ypitch,
                                     const Uint8 *UVplane, int UVpitch)
{
    SDL_Texture *native = texture->native;
    SDL_Rect full_rect;

    if (SDL_SW_UpdateNVTexturePlanar(texture->yuv, rect, Yplane, Ypitch, UVplane, UVpitch) < 0) {
        return -1;
    }

    full_rect.x = 0;
    full_rect.y = 0;
    full_rect.w = texture->w;
    full_rect.h = texture->h;
    rect = &full_rect;

    if (!rect->w || !rect->h) {
        return 0; /* nothing to do. */
    }

    if (texture->access == SDL_TEXTUREACCESS_STREAMING) {
        /* We can lock the texture and copy to it */
        void *native_pixels = NULL;
        int native_pitch = 0;

        if (SDL_LockTexture(native, rect, &native_pixels, &native_pitch) < 0) {
            return -1;
        }
        SDL_SW_CopyYUVToRGB(texture->yuv, rect, native->format,
                            rect->w, rect->h, native_pixels, native_pitch);
        SDL_UnlockTexture(native);
    } else {
        /* Use a temporary buffer for updating */
        const int temp_pitch = (((rect->w * SDL_BYTESPERPIXEL(native->format)) + 3) & ~3);
        const size_t alloclen = (size_t)rect->h * temp_pitch;
        if (alloclen > 0) {
            void *temp_pixels = SDL_malloc(alloclen);
            if (!temp_pixels) {
                return SDL_OutOfMemory();
            }
            SDL_SW_CopyYUVToRGB(texture->yuv, rect, native->format,
                                rect->w, rect->h, temp_pixels, temp_pitch);
            SDL_UpdateTexture(native, rect, temp_pixels, temp_pitch);
            SDL_free(temp_pixels);
        }
    }
    return 0;
}

#endif /* SDL_HAVE_YUV */

int SDL_UpdateYUVTexture(SDL_Texture *texture, const SDL_Rect *rect,
                         const Uint8 *Yplane, int Ypitch,
                         const Uint8 *Uplane, int Upitch,
                         const Uint8 *Vplane, int Vpitch)
{
#if SDL_HAVE_YUV
    SDL_Renderer *renderer;
    SDL_Rect real_rect;

    CHECK_TEXTURE_MAGIC(texture, -1);

    if (!Yplane) {
        return SDL_InvalidParamError("Yplane");
    }
    if (!Ypitch) {
        return SDL_InvalidParamError("Ypitch");
    }
    if (!Uplane) {
        return SDL_InvalidParamError("Uplane");
    }
    if (!Upitch) {
        return SDL_InvalidParamError("Upitch");
    }
    if (!Vplane) {
        return SDL_InvalidParamError("Vplane");
    }
    if (!Vpitch) {
        return SDL_InvalidParamError("Vpitch");
    }

    if (texture->format != SDL_PIXELFORMAT_YV12 &&
        texture->format != SDL_PIXELFORMAT_IYUV) {
        return SDL_SetError("Texture format must by YV12 or IYUV");
    }

    real_rect.x = 0;
    real_rect.y = 0;
    real_rect.w = texture->w;
    real_rect.h = texture->h;
    if (rect) {
        SDL_IntersectRect(rect, &real_rect, &real_rect);
    }

    if (real_rect.w == 0 || real_rect.h == 0) {
        return 0; /* nothing to do. */
    }

    if (texture->yuv) {
        return SDL_UpdateTextureYUVPlanar(texture, &real_rect, Yplane, Ypitch, Uplane, Upitch, Vplane, Vpitch);
    } else {
        SDL_assert(!texture->native);
        renderer = texture->renderer;
        SDL_assert(renderer->UpdateTextureYUV);
        if (renderer->UpdateTextureYUV) {
            if (FlushRenderCommandsIfTextureNeeded(texture) < 0) {
                return -1;
            }
            return renderer->UpdateTextureYUV(renderer, texture, &real_rect, Yplane, Ypitch, Uplane, Upitch, Vplane, Vpitch);
        } else {
            return SDL_Unsupported();
        }
    }
#else
    return -1;
#endif
}

int SDL_UpdateNVTexture(SDL_Texture *texture, const SDL_Rect *rect,
                        const Uint8 *Yplane, int Ypitch,
                        const Uint8 *UVplane, int UVpitch)
{
#if SDL_HAVE_YUV
    SDL_Renderer *renderer;
    SDL_Rect real_rect;

    CHECK_TEXTURE_MAGIC(texture, -1);

    if (!Yplane) {
        return SDL_InvalidParamError("Yplane");
    }
    if (!Ypitch) {
        return SDL_InvalidParamError("Ypitch");
    }
    if (!UVplane) {
        return SDL_InvalidParamError("UVplane");
    }
    if (!UVpitch) {
        return SDL_InvalidParamError("UVpitch");
    }

    if (texture->format != SDL_PIXELFORMAT_NV12 &&
        texture->format != SDL_PIXELFORMAT_NV21) {
        return SDL_SetError("Texture format must by NV12 or NV21");
    }

    real_rect.x = 0;
    real_rect.y = 0;
    real_rect.w = texture->w;
    real_rect.h = texture->h;
    if (rect) {
        SDL_IntersectRect(rect, &real_rect, &real_rect);
    }

    if (real_rect.w == 0 || real_rect.h == 0) {
        return 0; /* nothing to do. */
    }

    if (texture->yuv) {
        return SDL_UpdateTextureNVPlanar(texture, &real_rect, Yplane, Ypitch, UVplane, UVpitch);
    } else {
        SDL_assert(!texture->native);
        renderer = texture->renderer;
        SDL_assert(renderer->UpdateTextureNV);
        if (renderer->UpdateTextureNV) {
            if (FlushRenderCommandsIfTextureNeeded(texture) < 0) {
                return -1;
            }
            return renderer->UpdateTextureNV(renderer, texture, &real_rect, Yplane, Ypitch, UVplane, UVpitch);
        } else {
            return SDL_Unsupported();
        }
    }
#else
    return -1;
#endif
}

#if SDL_HAVE_YUV
static int SDL_LockTextureYUV(SDL_Texture *texture, const SDL_Rect *rect,
                              void **pixels, int *pitch)
{
    return SDL_SW_LockYUVTexture(texture->yuv, rect, pixels, pitch);
}
#endif /* SDL_HAVE_YUV */

static int SDL_LockTextureNative(SDL_Texture *texture, const SDL_Rect *rect,
                                 void **pixels, int *pitch)
{
    texture->locked_rect = *rect;
    *pixels = (void *)((Uint8 *)texture->pixels +
                       rect->y * texture->pitch +
                       rect->x * SDL_BYTESPERPIXEL(texture->format));
    *pitch = texture->pitch;
    return 0;
}

int SDL_LockTexture(SDL_Texture *texture, const SDL_Rect *rect,
                    void **pixels, int *pitch)
{
    SDL_Rect full_rect;

    CHECK_TEXTURE_MAGIC(texture, -1);

    if (texture->access != SDL_TEXTUREACCESS_STREAMING) {
        return SDL_SetError("SDL_LockTexture(): texture must be streaming");
    }

    if (!rect) {
        full_rect.x = 0;
        full_rect.y = 0;
        full_rect.w = texture->w;
        full_rect.h = texture->h;
        rect = &full_rect;
    }

#if SDL_HAVE_YUV
    if (texture->yuv) {
        if (FlushRenderCommandsIfTextureNeeded(texture) < 0) {
            return -1;
        }
        return SDL_LockTextureYUV(texture, rect, pixels, pitch);
    } else
#endif
        if (texture->native) {
        /* Calls a real SDL_LockTexture/SDL_UnlockTexture on unlock, flushing then. */
        return SDL_LockTextureNative(texture, rect, pixels, pitch);
    } else {
        SDL_Renderer *renderer = texture->renderer;
        if (FlushRenderCommandsIfTextureNeeded(texture) < 0) {
            return -1;
        }
        return renderer->LockTexture(renderer, texture, rect, pixels, pitch);
    }
}

int SDL_LockTextureToSurface(SDL_Texture *texture, const SDL_Rect *rect,
                             SDL_Surface **surface)
{
    SDL_Rect real_rect;
    void *pixels = NULL;
    int pitch = 0; /* fix static analysis */
    int ret;

    if (!texture || !surface) {
        return -1;
    }

    real_rect.x = 0;
    real_rect.y = 0;
    real_rect.w = texture->w;
    real_rect.h = texture->h;
    if (rect) {
        SDL_IntersectRect(rect, &real_rect, &real_rect);
    }

    ret = SDL_LockTexture(texture, &real_rect, &pixels, &pitch);
    if (ret < 0) {
        return ret;
    }

    texture->locked_surface = SDL_CreateRGBSurfaceWithFormatFrom(pixels, real_rect.w, real_rect.h, 0, pitch, texture->format);
    if (!texture->locked_surface) {
        SDL_UnlockTexture(texture);
        return -1;
    }

    *surface = texture->locked_surface;
    return 0;
}

#if SDL_HAVE_YUV
static void SDL_UnlockTextureYUV(SDL_Texture *texture)
{
    SDL_Texture *native = texture->native;
    void *native_pixels = NULL;
    int native_pitch = 0;
    SDL_Rect rect;

    rect.x = 0;
    rect.y = 0;
    rect.w = texture->w;
    rect.h = texture->h;

    if (SDL_LockTexture(native, &rect, &native_pixels, &native_pitch) < 0) {
        return;
    }
    SDL_SW_CopyYUVToRGB(texture->yuv, &rect, native->format,
                        rect.w, rect.h, native_pixels, native_pitch);
    SDL_UnlockTexture(native);
}
#endif /* SDL_HAVE_YUV */

static void SDL_UnlockTextureNative(SDL_Texture *texture)
{
    SDL_Texture *native = texture->native;
    void *native_pixels = NULL;
    int native_pitch = 0;
    const SDL_Rect *rect = &texture->locked_rect;
    const void *pixels = (void *)((Uint8 *)texture->pixels +
                                  rect->y * texture->pitch +
                                  rect->x * SDL_BYTESPERPIXEL(texture->format));
    int pitch = texture->pitch;

    if (SDL_LockTexture(native, rect, &native_pixels, &native_pitch) < 0) {
        return;
    }
    SDL_ConvertPixels(rect->w, rect->h,
                      texture->format, pixels, pitch,
                      native->format, native_pixels, native_pitch);
    SDL_UnlockTexture(native);
}

void SDL_UnlockTexture(SDL_Texture *texture)
{
    CHECK_TEXTURE_MAGIC(texture, );

    if (texture->access != SDL_TEXTUREACCESS_STREAMING) {
        return;
    }
#if SDL_HAVE_YUV
    if (texture->yuv) {
        SDL_UnlockTextureYUV(texture);
    } else
#endif
        if (texture->native) {
        SDL_UnlockTextureNative(texture);
    } else {
        SDL_Renderer *renderer = texture->renderer;
        renderer->UnlockTexture(renderer, texture);
    }

    SDL_FreeSurface(texture->locked_surface);
    texture->locked_surface = NULL;
}

SDL_bool SDL_RenderTargetSupported(SDL_Renderer *renderer)
{
    if (!renderer || !renderer->SetRenderTarget) {
        return SDL_FALSE;
    }
    return (renderer->info.flags & SDL_RENDERER_TARGETTEXTURE) != 0;
}

int SDL_SetRenderTarget(SDL_Renderer *renderer, SDL_Texture *texture)
{
    if (!SDL_RenderTargetSupported(renderer)) {
        return SDL_Unsupported();
    }

    /* texture == NULL is valid and means reset the target to the window */
    if (texture) {
        CHECK_TEXTURE_MAGIC(texture, -1);
        if (renderer != texture->renderer) {
            return SDL_SetError("Texture was not created with this renderer");
        }
        if (texture->access != SDL_TEXTUREACCESS_TARGET) {
            return SDL_SetError("Texture not created with SDL_TEXTUREACCESS_TARGET");
        }
        if (texture->native) {
            /* Always render to the native texture */
            texture = texture->native;
        }
    }

    if (texture == renderer->target) {
        /* Nothing to do! */
        return 0;
    }

    FlushRenderCommands(renderer); /* time to send everything to the GPU! */

    SDL_LockMutex(renderer->target_mutex);

    if (texture && !renderer->target) {
        /* Make a backup of the viewport */
        renderer->viewport_backup = renderer->viewport;
        renderer->clip_rect_backup = renderer->clip_rect;
        renderer->clipping_enabled_backup = renderer->clipping_enabled;
        renderer->scale_backup = renderer->scale;
        renderer->logical_w_backup = renderer->logical_w;
        renderer->logical_h_backup = renderer->logical_h;
    }
    renderer->target = texture;

    if (renderer->SetRenderTarget(renderer, texture) < 0) {
        SDL_UnlockMutex(renderer->target_mutex);
        return -1;
    }

    if (texture) {
        renderer->viewport.x = (double)0;
        renderer->viewport.y = (double)0;
        renderer->viewport.w = (double)texture->w;
        renderer->viewport.h = (double)texture->h;
        SDL_zero(renderer->clip_rect);
        renderer->clipping_enabled = SDL_FALSE;
        renderer->scale.x = 1.0f;
        renderer->scale.y = 1.0f;
        renderer->logical_w = texture->w;
        renderer->logical_h = texture->h;
    } else {
        renderer->viewport = renderer->viewport_backup;
        renderer->clip_rect = renderer->clip_rect_backup;
        renderer->clipping_enabled = renderer->clipping_enabled_backup;
        renderer->scale = renderer->scale_backup;
        renderer->logical_w = renderer->logical_w_backup;
        renderer->logical_h = renderer->logical_h_backup;
    }

    SDL_UnlockMutex(renderer->target_mutex);

    if (QueueCmdSetViewport(renderer) < 0) {
        return -1;
    }
    if (QueueCmdSetClipRect(renderer) < 0) {
        return -1;
    }

    /* All set! */
    return FlushRenderCommandsIfNotBatching(renderer);
}

SDL_Texture *SDL_GetRenderTarget(SDL_Renderer *renderer)
{
    CHECK_RENDERER_MAGIC(renderer, NULL);

    return renderer->target;
}

static int UpdateLogicalSize(SDL_Renderer *renderer, SDL_bool flush_viewport_cmd)
{
    int w = 1, h = 1;
    float want_aspect;
    float real_aspect;
    float scale;
    SDL_Rect viewport;
    /* 0 is for letterbox, 1 is for overscan */
    int scale_policy = 0;
    const char *hint;

    if (!renderer->logical_w || !renderer->logical_h) {
        return 0;
    }
    if (SDL_GetRendererOutputSize(renderer, &w, &h) < 0) {
        return -1;
    }

    hint = SDL_GetHint(SDL_HINT_RENDER_LOGICAL_SIZE_MODE);
    if (hint && (*hint == '1' || SDL_strcasecmp(hint, "overscan") == 0)) {
#if SDL_VIDEO_RENDER_D3D
        SDL_bool overscan_supported = SDL_TRUE;
        /* Unfortunately, Direct3D 9 doesn't support negative viewport numbers
           which the overscan implementation relies on.
        */
        if (SDL_strcasecmp(SDL_GetCurrentVideoDriver(), "direct3d") == 0) {
            overscan_supported = SDL_FALSE;
        }
        if (overscan_supported) {
            scale_policy = 1;
        }
#else
        scale_policy = 1;
#endif
    }

    want_aspect = (float)renderer->logical_w / renderer->logical_h;
    real_aspect = (float)w / h;

    /* Clear the scale because we're setting viewport in output coordinates */
    SDL_RenderSetScale(renderer, 1.0f, 1.0f);

    if (renderer->integer_scale) {
        if (want_aspect > real_aspect) {
            scale = (float)(w / renderer->logical_w); /* This an integer division! */
        } else {
            scale = (float)(h / renderer->logical_h); /* This an integer division! */
        }

        if (scale < 1.0f) {
            scale = 1.0f;
        }

        viewport.w = (int)SDL_floor(renderer->logical_w * scale);
        viewport.x = (w - viewport.w) / 2;
        viewport.h = (int)SDL_floor(renderer->logical_h * scale);
        viewport.y = (h - viewport.h) / 2;
    } else if (SDL_fabs(want_aspect - real_aspect) < 0.0001) {
        /* The aspect ratios are the same, just scale appropriately */
        scale = (float)w / renderer->logical_w;

        SDL_zero(viewport);
        SDL_GetRendererOutputSize(renderer, &viewport.w, &viewport.h);
    } else if (want_aspect > real_aspect) {
        if (scale_policy == 1) {
            /* We want a wider aspect ratio than is available -
             zoom so logical height matches the real height
             and the width will grow off the screen
             */
            scale = (float)h / renderer->logical_h;
            viewport.y = 0;
            viewport.h = h;
            viewport.w = (int)SDL_floor(renderer->logical_w * scale);
            viewport.x = (w - viewport.w) / 2;
        } else {
            /* We want a wider aspect ratio than is available - letterbox it */
            scale = (float)w / renderer->logical_w;
            viewport.x = 0;
            viewport.w = w;
            viewport.h = (int)SDL_floor(renderer->logical_h * scale);
            viewport.y = (h - viewport.h) / 2;
        }
    } else {
        if (scale_policy == 1) {
            /* We want a narrower aspect ratio than is available -
             zoom so logical width matches the real width
             and the height will grow off the screen
             */
            scale = (float)w / renderer->logical_w;
            viewport.x = 0;
            viewport.w = w;
            viewport.h = (int)SDL_floor(renderer->logical_h * scale);
            viewport.y = (h - viewport.h) / 2;
        } else {
            /* We want a narrower aspect ratio than is available - use side-bars */
            scale = (float)h / renderer->logical_h;
            viewport.y = 0;
            viewport.h = h;
            viewport.w = (int)SDL_floor(renderer->logical_w * scale);
            viewport.x = (w - viewport.w) / 2;
        }
    }

    /* Set the new viewport */
    renderer->viewport.x = (double)viewport.x * renderer->scale.x;
    renderer->viewport.y = (double)viewport.y * renderer->scale.y;
    renderer->viewport.w = (double)viewport.w * renderer->scale.x;
    renderer->viewport.h = (double)viewport.h * renderer->scale.y;
    QueueCmdSetViewport(renderer);
    if (flush_viewport_cmd) {
        FlushRenderCommandsIfNotBatching(renderer);
    }

    /* Set the new scale */
    SDL_RenderSetScale(renderer, scale, scale);

    return 0;
}

int SDL_RenderSetLogicalSize(SDL_Renderer *renderer, int w, int h)
{
    CHECK_RENDERER_MAGIC(renderer, -1);

    if (!w || !h) {
        /* Clear any previous logical resolution */
        renderer->logical_w = 0;
        renderer->logical_h = 0;
        SDL_RenderSetViewport(renderer, NULL);
        SDL_RenderSetScale(renderer, 1.0f, 1.0f);
        return 0;
    }

    renderer->logical_w = w;
    renderer->logical_h = h;

    return UpdateLogicalSize(renderer, SDL_TRUE);
}

void SDL_RenderGetLogicalSize(SDL_Renderer *renderer, int *w, int *h)
{
    CHECK_RENDERER_MAGIC(renderer, );

    if (w) {
        *w = renderer->logical_w;
    }
    if (h) {
        *h = renderer->logical_h;
    }
}

int SDL_RenderSetIntegerScale(SDL_Renderer *renderer, SDL_bool enable)
{
    CHECK_RENDERER_MAGIC(renderer, -1);

    renderer->integer_scale = enable;

    return UpdateLogicalSize(renderer, SDL_TRUE);
}

SDL_bool SDL_RenderGetIntegerScale(SDL_Renderer *renderer)
{
    CHECK_RENDERER_MAGIC(renderer, SDL_FALSE);

    return renderer->integer_scale;
}

int SDL_RenderSetViewport(SDL_Renderer *renderer, const SDL_Rect *rect)
{
    int retval;
    CHECK_RENDERER_MAGIC(renderer, -1);

    if (rect) {
        renderer->viewport.x = (double)rect->x * renderer->scale.x;
        renderer->viewport.y = (double)rect->y * renderer->scale.y;
        renderer->viewport.w = (double)rect->w * renderer->scale.x;
        renderer->viewport.h = (double)rect->h * renderer->scale.y;
    } else {
        int w, h;
        if (SDL_GetRendererOutputSize(renderer, &w, &h) < 0) {
            return -1;
        }
        renderer->viewport.x = 0.0;
        renderer->viewport.y = 0.0;
        /* NOLINTBEGIN(clang-analyzer-core.uninitialized.Assign): SDL_GetRendererOutputSize cannot fail */
        renderer->viewport.w = (double)w;
        renderer->viewport.h = (double)h;
        /* NOLINTEND(clang-analyzer-core.uninitialized.Assign) */
    }
    retval = QueueCmdSetViewport(renderer);
    return retval < 0 ? retval : FlushRenderCommandsIfNotBatching(renderer);
}

void SDL_RenderGetViewport(SDL_Renderer *renderer, SDL_Rect *rect)
{
    CHECK_RENDERER_MAGIC(renderer, );

    if (rect) {
        rect->x = (int)SDL_floor(renderer->viewport.x / renderer->scale.x);
        rect->y = (int)SDL_floor(renderer->viewport.y / renderer->scale.y);
        rect->w = (int)SDL_floor(renderer->viewport.w / renderer->scale.x);
        rect->h = (int)SDL_floor(renderer->viewport.h / renderer->scale.y);
    }
}

static void RenderGetViewportSize(SDL_Renderer *renderer, SDL_FRect *rect)
{
    rect->x = 0.0f;
    rect->y = 0.0f;
    rect->w = (float)(renderer->viewport.w / renderer->scale.x);
    rect->h = (float)(renderer->viewport.h / renderer->scale.y);
}

int SDL_RenderSetClipRect(SDL_Renderer *renderer, const SDL_Rect *rect)
{
    int retval;
    CHECK_RENDERER_MAGIC(renderer, -1)

    if (rect && rect->w >= 0 && rect->h >= 0) {
        renderer->clipping_enabled = SDL_TRUE;
        renderer->clip_rect.x = (double)rect->x * renderer->scale.x;
        renderer->clip_rect.y = (double)rect->y * renderer->scale.y;
        renderer->clip_rect.w = (double)rect->w * renderer->scale.x;
        renderer->clip_rect.h = (double)rect->h * renderer->scale.y;
    } else {
        renderer->clipping_enabled = SDL_FALSE;
        SDL_zero(renderer->clip_rect);
    }

    retval = QueueCmdSetClipRect(renderer);
    return retval < 0 ? retval : FlushRenderCommandsIfNotBatching(renderer);
}

void SDL_RenderGetClipRect(SDL_Renderer *renderer, SDL_Rect *rect)
{
    CHECK_RENDERER_MAGIC(renderer, )

    if (rect) {
        rect->x = (int)SDL_floor(renderer->clip_rect.x / renderer->scale.x);
        rect->y = (int)SDL_floor(renderer->clip_rect.y / renderer->scale.y);
        rect->w = (int)SDL_floor(renderer->clip_rect.w / renderer->scale.x);
        rect->h = (int)SDL_floor(renderer->clip_rect.h / renderer->scale.y);
    }
}

SDL_bool SDL_RenderIsClipEnabled(SDL_Renderer *renderer)
{
    CHECK_RENDERER_MAGIC(renderer, SDL_FALSE)
    return renderer->clipping_enabled;
}

int SDL_RenderSetScale(SDL_Renderer *renderer, float scaleX, float scaleY)
{
    CHECK_RENDERER_MAGIC(renderer, -1);

    renderer->scale.x = scaleX;
    renderer->scale.y = scaleY;
    return 0;
}

void SDL_RenderGetScale(SDL_Renderer *renderer, float *scaleX, float *scaleY)
{
    CHECK_RENDERER_MAGIC(renderer, );

    if (scaleX) {
        *scaleX = renderer->scale.x;
    }
    if (scaleY) {
        *scaleY = renderer->scale.y;
    }
}

void SDL_RenderWindowToLogical(SDL_Renderer *renderer, int windowX, int windowY, float *logicalX, float *logicalY)
{
    float window_physical_x, window_physical_y;

    CHECK_RENDERER_MAGIC(renderer, );

    window_physical_x = ((float)windowX) / renderer->dpi_scale.x;
    window_physical_y = ((float)windowY) / renderer->dpi_scale.y;

    if (logicalX) {
        *logicalX = (float)((window_physical_x - renderer->viewport.x) / renderer->scale.x);
    }
    if (logicalY) {
        *logicalY = (float)((window_physical_y - renderer->viewport.y) / renderer->scale.y);
    }
}

void SDL_RenderLogicalToWindow(SDL_Renderer *renderer, float logicalX, float logicalY, int *windowX, int *windowY)
{
    float window_physical_x, window_physical_y;

    CHECK_RENDERER_MAGIC(renderer, );

    window_physical_x = (float)((logicalX * renderer->scale.x) + renderer->viewport.x);
    window_physical_y = (float)((logicalY * renderer->scale.y) + renderer->viewport.y);

    if (windowX) {
        *windowX = (int)(window_physical_x * renderer->dpi_scale.x);
    }
    if (windowY) {
        *windowY = (int)(window_physical_y * renderer->dpi_scale.y);
    }
}

int SDL_SetRenderDrawColor(SDL_Renderer *renderer,
                           Uint8 r, Uint8 g, Uint8 b, Uint8 a)
{
    CHECK_RENDERER_MAGIC(renderer, -1);

    renderer->color.r = r;
    renderer->color.g = g;
    renderer->color.b = b;
    renderer->color.a = a;
    return 0;
}

int SDL_GetRenderDrawColor(SDL_Renderer *renderer,
                           Uint8 *r, Uint8 *g, Uint8 *b, Uint8 *a)
{
    CHECK_RENDERER_MAGIC(renderer, -1);

    if (r) {
        *r = renderer->color.r;
    }
    if (g) {
        *g = renderer->color.g;
    }
    if (b) {
        *b = renderer->color.b;
    }
    if (a) {
        *a = renderer->color.a;
    }
    return 0;
}

int SDL_SetRenderDrawBlendMode(SDL_Renderer *renderer, SDL_BlendMode blendMode)
{
    CHECK_RENDERER_MAGIC(renderer, -1);

    if (!IsSupportedBlendMode(renderer, blendMode)) {
        return SDL_Unsupported();
    }
    renderer->blendMode = blendMode;
    return 0;
}

int SDL_GetRenderDrawBlendMode(SDL_Renderer *renderer, SDL_BlendMode *blendMode)
{
    CHECK_RENDERER_MAGIC(renderer, -1);

    *blendMode = renderer->blendMode;
    return 0;
}

int SDL_RenderClear(SDL_Renderer *renderer)
{
    int retval;
    CHECK_RENDERER_MAGIC(renderer, -1);
    retval = QueueCmdClear(renderer);
    return retval < 0 ? retval : FlushRenderCommandsIfNotBatching(renderer);
}

/* !!! FIXME: delete all the duplicate code for the integer versions in 2.1,
   !!! FIXME:  making the floating point versions the only available APIs. */

int SDL_RenderDrawPoint(SDL_Renderer *renderer, int x, int y)
{
    SDL_FPoint fpoint;
    fpoint.x = (float)x;
    fpoint.y = (float)y;
    return SDL_RenderDrawPointsF(renderer, &fpoint, 1);
}

int SDL_RenderDrawPointF(SDL_Renderer *renderer, float x, float y)
{
    SDL_FPoint fpoint;
    fpoint.x = x;
    fpoint.y = y;
    return SDL_RenderDrawPointsF(renderer, &fpoint, 1);
}

static int RenderDrawPointsWithRects(SDL_Renderer *renderer,
                                     const SDL_Point *points, const int count)
{
    int retval;
    SDL_bool isstack;
    SDL_FRect *frects;
    int i;

    if (count < 1) {
        return 0;
    }

    frects = SDL_small_alloc(SDL_FRect, count, &isstack);
    if (!frects) {
        return SDL_OutOfMemory();
    }

    for (i = 0; i < count; ++i) {
        frects[i].x = points[i].x * renderer->scale.x;
        frects[i].y = points[i].y * renderer->scale.y;
        frects[i].w = renderer->scale.x;
        frects[i].h = renderer->scale.y;
    }

    retval = QueueCmdFillRects(renderer, frects, count);

    SDL_small_free(frects, isstack);

    return retval;
}

int SDL_RenderDrawPoints(SDL_Renderer *renderer,
                         const SDL_Point *points, int count)
{
    SDL_FPoint *fpoints;
    int i;
    int retval;
    SDL_bool isstack;

    CHECK_RENDERER_MAGIC(renderer, -1);

    if (!points) {
        return SDL_InvalidParamError("SDL_RenderDrawPoints(): points");
    }
    if (count < 1) {
        return 0;
    }

#if DONT_DRAW_WHILE_HIDDEN
    /* Don't draw while we're hidden */
    if (renderer->hidden) {
        return 0;
    }
#endif

    if (renderer->scale.x != 1.0f || renderer->scale.y != 1.0f) {
        retval = RenderDrawPointsWithRects(renderer, points, count);
    } else {
        fpoints = SDL_small_alloc(SDL_FPoint, count, &isstack);
        if (!fpoints) {
            return SDL_OutOfMemory();
        }
        for (i = 0; i < count; ++i) {
            fpoints[i].x = (float)points[i].x;
            fpoints[i].y = (float)points[i].y;
        }

        retval = QueueCmdDrawPoints(renderer, fpoints, count);

        SDL_small_free(fpoints, isstack);
    }
    return retval < 0 ? retval : FlushRenderCommandsIfNotBatching(renderer);
}

static int RenderDrawPointsWithRectsF(SDL_Renderer *renderer,
                                      const SDL_FPoint *fpoints, const int count)
{
    int retval;
    SDL_bool isstack;
    SDL_FRect *frects;
    int i;

    if (count < 1) {
        return 0;
    }

    frects = SDL_small_alloc(SDL_FRect, count, &isstack);
    if (!frects) {
        return SDL_OutOfMemory();
    }

    for (i = 0; i < count; ++i) {
        frects[i].x = fpoints[i].x * renderer->scale.x;
        frects[i].y = fpoints[i].y * renderer->scale.y;
        frects[i].w = renderer->scale.x;
        frects[i].h = renderer->scale.y;
    }

    retval = QueueCmdFillRects(renderer, frects, count);

    SDL_small_free(frects, isstack);

    return retval;
}

int SDL_RenderDrawPointsF(SDL_Renderer *renderer,
                          const SDL_FPoint *points, int count)
{
    int retval;

    CHECK_RENDERER_MAGIC(renderer, -1);

    if (!points) {
        return SDL_InvalidParamError("SDL_RenderDrawPointsF(): points");
    }
    if (count < 1) {
        return 0;
    }

#if DONT_DRAW_WHILE_HIDDEN
    /* Don't draw while we're hidden */
    if (renderer->hidden) {
        return 0;
    }
#endif

    if (renderer->scale.x != 1.0f || renderer->scale.y != 1.0f) {
        retval = RenderDrawPointsWithRectsF(renderer, points, count);
    } else {
        retval = QueueCmdDrawPoints(renderer, points, count);
    }
    return retval < 0 ? retval : FlushRenderCommandsIfNotBatching(renderer);
}

int SDL_RenderDrawLine(SDL_Renderer *renderer, int x1, int y1, int x2, int y2)
{
    SDL_FPoint points[2];
    points[0].x = (float)x1;
    points[0].y = (float)y1;
    points[1].x = (float)x2;
    points[1].y = (float)y2;
    return SDL_RenderDrawLinesF(renderer, points, 2);
}

int SDL_RenderDrawLineF(SDL_Renderer *renderer, float x1, float y1, float x2, float y2)
{
    SDL_FPoint points[2];
    points[0].x = x1;
    points[0].y = y1;
    points[1].x = x2;
    points[1].y = y2;
    return SDL_RenderDrawLinesF(renderer, points, 2);
}

static int RenderDrawLineBresenham(SDL_Renderer *renderer, int x1, int y1, int x2, int y2, SDL_bool draw_last)
{
    int i, deltax, deltay, numpixels;
    int d, dinc1, dinc2;
    int x, xinc1, xinc2;
    int y, yinc1, yinc2;
    int retval;
    SDL_bool isstack;
    SDL_FPoint *points;
    SDL_Rect clip_rect;

    /* the backend might clip this further to the clipping rect, but we
       just want a basic safety against generating millions of points for
       massive lines. */
    clip_rect.x = 0;
    clip_rect.y = 0;
    clip_rect.w = (int) renderer->viewport.w;
    clip_rect.h = (int) renderer->viewport.h;
    if (!SDL_IntersectRectAndLine(&clip_rect, &x1, &y1, &x2, &y2)) {
        return 0;
    }

    deltax = SDL_abs(x2 - x1);
    deltay = SDL_abs(y2 - y1);

    if (deltax >= deltay) {
        numpixels = deltax + 1;
        d = (2 * deltay) - deltax;
        dinc1 = deltay * 2;
        dinc2 = (deltay - deltax) * 2;
        xinc1 = 1;
        xinc2 = 1;
        yinc1 = 0;
        yinc2 = 1;
    } else {
        numpixels = deltay + 1;
        d = (2 * deltax) - deltay;
        dinc1 = deltax * 2;
        dinc2 = (deltax - deltay) * 2;
        xinc1 = 0;
        xinc2 = 1;
        yinc1 = 1;
        yinc2 = 1;
    }

    if (x1 > x2) {
        xinc1 = -xinc1;
        xinc2 = -xinc2;
    }
    if (y1 > y2) {
        yinc1 = -yinc1;
        yinc2 = -yinc2;
    }

    x = x1;
    y = y1;

    if (!draw_last) {
        --numpixels;
    }

    points = SDL_small_alloc(SDL_FPoint, numpixels, &isstack);
    if (!points) {
        return SDL_OutOfMemory();
    }
    for (i = 0; i < numpixels; ++i) {
        points[i].x = (float)x;
        points[i].y = (float)y;

        if (d < 0) {
            d += dinc1;
            x += xinc1;
            y += yinc1;
        } else {
            d += dinc2;
            x += xinc2;
            y += yinc2;
        }
    }

    if (renderer->scale.x != 1.0f || renderer->scale.y != 1.0f) {
        retval = RenderDrawPointsWithRectsF(renderer, points, numpixels);
    } else {
        retval = QueueCmdDrawPoints(renderer, points, numpixels);
    }

    SDL_small_free(points, isstack);

    return retval;
}

static int RenderDrawLinesWithRectsF(SDL_Renderer *renderer,
                                     const SDL_FPoint *points, const int count)
{
    const float scale_x = renderer->scale.x;
    const float scale_y = renderer->scale.y;
    SDL_FRect *frect;
    SDL_FRect *frects;
    int i, nrects = 0;
    int retval = 0;
    SDL_bool isstack;
    SDL_bool drew_line = SDL_FALSE;
    SDL_bool draw_last = SDL_FALSE;

    frects = SDL_small_alloc(SDL_FRect, count - 1, &isstack);
    if (!frects) {
        return SDL_OutOfMemory();
    }

    for (i = 0; i < count - 1; ++i) {
        SDL_bool same_x = (points[i].x == points[i + 1].x);
        SDL_bool same_y = (points[i].y == points[i + 1].y);

        if (i == (count - 2)) {
            if (!drew_line || points[i + 1].x != points[0].x || points[i + 1].y != points[0].y) {
                draw_last = SDL_TRUE;
            }
        } else {
            if (same_x && same_y) {
                continue;
            }
        }
        if (same_x) {
            const float minY = SDL_min(points[i].y, points[i + 1].y);
            const float maxY = SDL_max(points[i].y, points[i + 1].y);

            frect = &frects[nrects++];
            frect->x = points[i].x * scale_x;
            frect->y = minY * scale_y;
            frect->w = scale_x;
            frect->h = (maxY - minY + draw_last) * scale_y;
            if (!draw_last && points[i + 1].y < points[i].y) {
                frect->y += scale_y;
            }
        } else if (same_y) {
            const float minX = SDL_min(points[i].x, points[i + 1].x);
            const float maxX = SDL_max(points[i].x, points[i + 1].x);

            frect = &frects[nrects++];
            frect->x = minX * scale_x;
            frect->y = points[i].y * scale_y;
            frect->w = (maxX - minX + draw_last) * scale_x;
            frect->h = scale_y;
            if (!draw_last && points[i + 1].x < points[i].x) {
                frect->x += scale_x;
            }
        } else {
            retval += RenderDrawLineBresenham(renderer, (int)SDL_roundf(points[i].x), (int)SDL_roundf(points[i].y),
                                              (int)SDL_roundf(points[i + 1].x), (int)SDL_roundf(points[i + 1].y), draw_last);
        }
        drew_line = SDL_TRUE;
    }

    if (nrects) {
        retval += QueueCmdFillRects(renderer, frects, nrects);
    }

    SDL_small_free(frects, isstack);

    if (retval < 0) {
        retval = -1;
    }
    return retval;
}

int SDL_RenderDrawLines(SDL_Renderer *renderer,
                        const SDL_Point *points, int count)
{
    SDL_FPoint *fpoints;
    int i;
    int retval;
    SDL_bool isstack;

    CHECK_RENDERER_MAGIC(renderer, -1);

    if (!points) {
        return SDL_InvalidParamError("SDL_RenderDrawLines(): points");
    }
    if (count < 2) {
        return 0;
    }

#if DONT_DRAW_WHILE_HIDDEN
    /* Don't draw while we're hidden */
    if (renderer->hidden) {
        return 0;
    }
#endif

    fpoints = SDL_small_alloc(SDL_FPoint, count, &isstack);
    if (!fpoints) {
        return SDL_OutOfMemory();
    }

    for (i = 0; i < count; ++i) {
        fpoints[i].x = (float)points[i].x;
        fpoints[i].y = (float)points[i].y;
    }

    retval = SDL_RenderDrawLinesF(renderer, fpoints, count);

    SDL_small_free(fpoints, isstack);

    return retval;
}

int SDL_RenderDrawLinesF(SDL_Renderer *renderer,
                         const SDL_FPoint *points, int count)
{
    int retval = 0;

    CHECK_RENDERER_MAGIC(renderer, -1);

    if (!points) {
        return SDL_InvalidParamError("SDL_RenderDrawLinesF(): points");
    }
    if (count < 2) {
        return 0;
    }

#if DONT_DRAW_WHILE_HIDDEN
    /* Don't draw while we're hidden */
    if (renderer->hidden) {
        return 0;
    }
#endif

    if (renderer->line_method == SDL_RENDERLINEMETHOD_POINTS) {
        retval = RenderDrawLinesWithRectsF(renderer, points, count);
    } else if (renderer->line_method == SDL_RENDERLINEMETHOD_GEOMETRY) {
        SDL_bool isstack1;
        SDL_bool isstack2;
        const float scale_x = renderer->scale.x;
        const float scale_y = renderer->scale.y;
        float *xy = SDL_small_alloc(float, 4 * 2 * count, &isstack1);
        int *indices = SDL_small_alloc(int,
                                       (4) * 3 * (count - 1) + (2) * 3 * (count), &isstack2);

        if (xy && indices) {
            int i;
            float *ptr_xy = xy;
            int *ptr_indices = indices;
            const int xy_stride = 2 * sizeof(float);
            int num_vertices = 4 * count;
            int num_indices = 0;
            const int size_indices = 4;
            int cur_index = -4;
            const int is_looping = (points[0].x == points[count - 1].x && points[0].y == points[count - 1].y);
            SDL_FPoint p; /* previous point */
            p.x = p.y = 0.0f;
            /*       p            q

                    0----1------ 4----5
                    | \  |``\    | \  |
                    |  \ |   ` `\|  \ |
                    3----2-------7----6
            */
            for (i = 0; i < count; ++i) {
                SDL_FPoint q = points[i]; /* current point */

                q.x *= scale_x;
                q.y *= scale_y;

                *ptr_xy++ = q.x;
                *ptr_xy++ = q.y;
                *ptr_xy++ = q.x + scale_x;
                *ptr_xy++ = q.y;
                *ptr_xy++ = q.x + scale_x;
                *ptr_xy++ = q.y + scale_y;
                *ptr_xy++ = q.x;
                *ptr_xy++ = q.y + scale_y;

#define ADD_TRIANGLE(i1, i2, i3)     \
    *ptr_indices++ = cur_index + i1; \
    *ptr_indices++ = cur_index + i2; \
    *ptr_indices++ = cur_index + i3; \
    num_indices += 3;

                /* closed polyline, don´t draw twice the point */
                if (i || is_looping == 0) {
                    ADD_TRIANGLE(4, 5, 6)
                    ADD_TRIANGLE(4, 6, 7)
                }

                /* first point only, no segment */
                if (i == 0) {
                    p = q;
                    cur_index += 4;
                    continue;
                }

                /* draw segment */
                if (p.y == q.y) {
                    if (p.x < q.x) {
                        ADD_TRIANGLE(1, 4, 7)
                        ADD_TRIANGLE(1, 7, 2)
                    } else {
                        ADD_TRIANGLE(5, 0, 3)
                        ADD_TRIANGLE(5, 3, 6)
                    }
                } else if (p.x == q.x) {
                    if (p.y < q.y) {
                        ADD_TRIANGLE(2, 5, 4)
                        ADD_TRIANGLE(2, 4, 3)
                    } else {
                        ADD_TRIANGLE(6, 1, 0)
                        ADD_TRIANGLE(6, 0, 7)
                    }
                } else {
                    if (p.y < q.y) {
                        if (p.x < q.x) {
                            ADD_TRIANGLE(1, 5, 4)
                            ADD_TRIANGLE(1, 4, 2)
                            ADD_TRIANGLE(2, 4, 7)
                            ADD_TRIANGLE(2, 7, 3)
                        } else {
                            ADD_TRIANGLE(4, 0, 5)
                            ADD_TRIANGLE(5, 0, 3)
                            ADD_TRIANGLE(5, 3, 6)
                            ADD_TRIANGLE(6, 3, 2)
                        }
                    } else {
                        if (p.x < q.x) {
                            ADD_TRIANGLE(0, 4, 7)
                            ADD_TRIANGLE(0, 7, 1)
                            ADD_TRIANGLE(1, 7, 6)
                            ADD_TRIANGLE(1, 6, 2)
                        } else {
                            ADD_TRIANGLE(6, 5, 1)
                            ADD_TRIANGLE(6, 1, 0)
                            ADD_TRIANGLE(7, 6, 0)
                            ADD_TRIANGLE(7, 0, 3)
                        }
                    }
                }

                p = q;
                cur_index += 4;
            }

            retval = QueueCmdGeometry(renderer, NULL,
                                      xy, xy_stride, &renderer->color, 0 /* color_stride */, NULL, 0,
                                      num_vertices, indices, num_indices, size_indices,
                                      1.0f, 1.0f);
        }

        SDL_small_free(xy, isstack1);
        SDL_small_free(indices, isstack2);

    } else if (renderer->scale.x != 1.0f || renderer->scale.y != 1.0f) {
        retval = RenderDrawLinesWithRectsF(renderer, points, count);
    } else {
        retval = QueueCmdDrawLines(renderer, points, count);
    }

    return retval < 0 ? retval : FlushRenderCommandsIfNotBatching(renderer);
}

int SDL_RenderDrawRect(SDL_Renderer *renderer, const SDL_Rect *rect)
{
    SDL_FRect frect;
    SDL_FRect *prect = NULL;

    if (rect) {
        frect.x = (float)rect->x;
        frect.y = (float)rect->y;
        frect.w = (float)rect->w;
        frect.h = (float)rect->h;
        prect = &frect;
    }

    return SDL_RenderDrawRectF(renderer, prect);
}

int SDL_RenderDrawRectF(SDL_Renderer *renderer, const SDL_FRect *rect)
{
    SDL_FRect frect;
    SDL_FPoint points[5];

    CHECK_RENDERER_MAGIC(renderer, -1);

    /* If 'rect' == NULL, then outline the whole surface */
    if (!rect) {
        RenderGetViewportSize(renderer, &frect);
        rect = &frect;
    }

    points[0].x = rect->x;
    points[0].y = rect->y;
    points[1].x = rect->x + rect->w - 1;
    points[1].y = rect->y;
    points[2].x = rect->x + rect->w - 1;
    points[2].y = rect->y + rect->h - 1;
    points[3].x = rect->x;
    points[3].y = rect->y + rect->h - 1;
    points[4].x = rect->x;
    points[4].y = rect->y;
    return SDL_RenderDrawLinesF(renderer, points, 5);
}

int SDL_RenderDrawRects(SDL_Renderer *renderer,
                        const SDL_Rect *rects, int count)
{
    int i;

    CHECK_RENDERER_MAGIC(renderer, -1);

    if (!rects) {
        return SDL_InvalidParamError("SDL_RenderDrawRects(): rects");
    }
    if (count < 1) {
        return 0;
    }

#if DONT_DRAW_WHILE_HIDDEN
    /* Don't draw while we're hidden */
    if (renderer->hidden) {
        return 0;
    }
#endif

    for (i = 0; i < count; ++i) {
        if (SDL_RenderDrawRect(renderer, &rects[i]) < 0) {
            return -1;
        }
    }
    return 0;
}

int SDL_RenderDrawRectsF(SDL_Renderer *renderer,
                         const SDL_FRect *rects, int count)
{
    int i;

    CHECK_RENDERER_MAGIC(renderer, -1);

    if (!rects) {
        return SDL_InvalidParamError("SDL_RenderDrawRectsF(): rects");
    }
    if (count < 1) {
        return 0;
    }

#if DONT_DRAW_WHILE_HIDDEN
    /* Don't draw while we're hidden */
    if (renderer->hidden) {
        return 0;
    }
#endif

    for (i = 0; i < count; ++i) {
        if (SDL_RenderDrawRectF(renderer, &rects[i]) < 0) {
            return -1;
        }
    }
    return 0;
}

int SDL_RenderFillRect(SDL_Renderer *renderer, const SDL_Rect *rect)
{
    SDL_FRect frect;

    CHECK_RENDERER_MAGIC(renderer, -1);

    /* If 'rect' == NULL, then outline the whole surface */
    if (rect) {
        frect.x = (float)rect->x;
        frect.y = (float)rect->y;
        frect.w = (float)rect->w;
        frect.h = (float)rect->h;
    } else {
        RenderGetViewportSize(renderer, &frect);
    }
    return SDL_RenderFillRectsF(renderer, &frect, 1);
}

int SDL_RenderFillRectF(SDL_Renderer *renderer, const SDL_FRect *rect)
{
    SDL_FRect frect;

    CHECK_RENDERER_MAGIC(renderer, -1);

    /* If 'rect' == NULL, then outline the whole surface */
    if (!rect) {
        RenderGetViewportSize(renderer, &frect);
        rect = &frect;
    }
    return SDL_RenderFillRectsF(renderer, rect, 1);
}

int SDL_RenderFillRects(SDL_Renderer *renderer,
                        const SDL_Rect *rects, int count)
{
    SDL_FRect *frects;
    int i;
    int retval;
    SDL_bool isstack;

    CHECK_RENDERER_MAGIC(renderer, -1);

    if (!rects) {
        return SDL_InvalidParamError("SDL_RenderFillRects(): rects");
    }
    if (count < 1) {
        return 0;
    }

#if DONT_DRAW_WHILE_HIDDEN
    /* Don't draw while we're hidden */
    if (renderer->hidden) {
        return 0;
    }
#endif

    frects = SDL_small_alloc(SDL_FRect, count, &isstack);
    if (!frects) {
        return SDL_OutOfMemory();
    }
    for (i = 0; i < count; ++i) {
        frects[i].x = rects[i].x * renderer->scale.x;
        frects[i].y = rects[i].y * renderer->scale.y;
        frects[i].w = rects[i].w * renderer->scale.x;
        frects[i].h = rects[i].h * renderer->scale.y;
    }

    retval = QueueCmdFillRects(renderer, frects, count);

    SDL_small_free(frects, isstack);

    return retval < 0 ? retval : FlushRenderCommandsIfNotBatching(renderer);
}

int SDL_RenderFillRectsF(SDL_Renderer *renderer,
                         const SDL_FRect *rects, int count)
{
    SDL_FRect *frects;
    int i;
    int retval;
    SDL_bool isstack;

    CHECK_RENDERER_MAGIC(renderer, -1);

    if (!rects) {
        return SDL_InvalidParamError("SDL_RenderFillRectsF(): rects");
    }
    if (count < 1) {
        return 0;
    }

#if DONT_DRAW_WHILE_HIDDEN
    /* Don't draw while we're hidden */
    if (renderer->hidden) {
        return 0;
    }
#endif

    frects = SDL_small_alloc(SDL_FRect, count, &isstack);
    if (!frects) {
        return SDL_OutOfMemory();
    }
    for (i = 0; i < count; ++i) {
        frects[i].x = rects[i].x * renderer->scale.x;
        frects[i].y = rects[i].y * renderer->scale.y;
        frects[i].w = rects[i].w * renderer->scale.x;
        frects[i].h = rects[i].h * renderer->scale.y;
    }

    retval = QueueCmdFillRects(renderer, frects, count);

    SDL_small_free(frects, isstack);

    return retval < 0 ? retval : FlushRenderCommandsIfNotBatching(renderer);
}

int SDL_RenderCopy(SDL_Renderer *renderer, SDL_Texture *texture,
                   const SDL_Rect *srcrect, const SDL_Rect *dstrect)
{
    SDL_FRect dstfrect;
    SDL_FRect *pdstfrect = NULL;
    if (dstrect) {
        dstfrect.x = (float)dstrect->x;
        dstfrect.y = (float)dstrect->y;
        dstfrect.w = (float)dstrect->w;
        dstfrect.h = (float)dstrect->h;
        pdstfrect = &dstfrect;
    }
    return SDL_RenderCopyF(renderer, texture, srcrect, pdstfrect);
}

int SDL_RenderCopyF(SDL_Renderer *renderer, SDL_Texture *texture,
                    const SDL_Rect *srcrect, const SDL_FRect *dstrect)
{
    SDL_Rect real_srcrect;
    SDL_FRect real_dstrect;
    int retval;
    int use_rendergeometry;

    CHECK_RENDERER_MAGIC(renderer, -1);
    CHECK_TEXTURE_MAGIC(texture, -1);

    if (renderer != texture->renderer) {
        return SDL_SetError("Texture was not created with this renderer");
    }

#if DONT_DRAW_WHILE_HIDDEN
    /* Don't draw while we're hidden */
    if (renderer->hidden) {
        return 0;
    }
#endif

    use_rendergeometry = (!renderer->QueueCopy);

    real_srcrect.x = 0;
    real_srcrect.y = 0;
    real_srcrect.w = texture->w;
    real_srcrect.h = texture->h;
    if (srcrect) {
        if (!SDL_IntersectRect(srcrect, &real_srcrect, &real_srcrect)) {
            return 0;
        }
    }

    RenderGetViewportSize(renderer, &real_dstrect);
    if (dstrect) {
        if (!SDL_HasIntersectionF(dstrect, &real_dstrect)) {
            return 0;
        }
        real_dstrect = *dstrect;
    }

    if (texture->native) {
        texture = texture->native;
    }

    texture->last_command_generation = renderer->render_command_generation;

    if (use_rendergeometry) {
        float xy[8];
        const int xy_stride = 2 * sizeof(float);
        float uv[8];
        const int uv_stride = 2 * sizeof(float);
        const int num_vertices = 4;
        const int *indices = renderer->rect_index_order;
        const int num_indices = 6;
        const int size_indices = 4;
        float minu, minv, maxu, maxv;
        float minx, miny, maxx, maxy;

        minu = (float)(real_srcrect.x) / (float)texture->w;
        minv = (float)(real_srcrect.y) / (float)texture->h;
        maxu = (float)(real_srcrect.x + real_srcrect.w) / (float)texture->w;
        maxv = (float)(real_srcrect.y + real_srcrect.h) / (float)texture->h;

        minx = real_dstrect.x;
        miny = real_dstrect.y;
        maxx = real_dstrect.x + real_dstrect.w;
        maxy = real_dstrect.y + real_dstrect.h;

        uv[0] = minu;
        uv[1] = minv;
        uv[2] = maxu;
        uv[3] = minv;
        uv[4] = maxu;
        uv[5] = maxv;
        uv[6] = minu;
        uv[7] = maxv;

        xy[0] = minx;
        xy[1] = miny;
        xy[2] = maxx;
        xy[3] = miny;
        xy[4] = maxx;
        xy[5] = maxy;
        xy[6] = minx;
        xy[7] = maxy;

        retval = QueueCmdGeometry(renderer, texture,
                                  xy, xy_stride, &texture->color, 0 /* color_stride */, uv, uv_stride,
                                  num_vertices,
                                  indices, num_indices, size_indices,
                                  renderer->scale.x, renderer->scale.y);
    } else {

        real_dstrect.x *= renderer->scale.x;
        real_dstrect.y *= renderer->scale.y;
        real_dstrect.w *= renderer->scale.x;
        real_dstrect.h *= renderer->scale.y;

        retval = QueueCmdCopy(renderer, texture, &real_srcrect, &real_dstrect);
    }
    return retval < 0 ? retval : FlushRenderCommandsIfNotBatching(renderer);
}

int SDL_RenderCopyEx(SDL_Renderer *renderer, SDL_Texture *texture,
                     const SDL_Rect *srcrect, const SDL_Rect *dstrect,
                     const double angle, const SDL_Point *center, const SDL_RendererFlip flip)
{
    SDL_FRect dstfrect;
    SDL_FRect *pdstfrect = NULL;
    SDL_FPoint fcenter;
    SDL_FPoint *pfcenter = NULL;

    if (dstrect) {
        dstfrect.x = (float)dstrect->x;
        dstfrect.y = (float)dstrect->y;
        dstfrect.w = (float)dstrect->w;
        dstfrect.h = (float)dstrect->h;
        pdstfrect = &dstfrect;
    }

    if (center) {
        fcenter.x = (float)center->x;
        fcenter.y = (float)center->y;
        pfcenter = &fcenter;
    }

    return SDL_RenderCopyExF(renderer, texture, srcrect, pdstfrect, angle, pfcenter, flip);
}

int SDL_RenderCopyExF(SDL_Renderer *renderer, SDL_Texture *texture,
                      const SDL_Rect *srcrect, const SDL_FRect *dstrect,
                      const double angle, const SDL_FPoint *center, const SDL_RendererFlip flip)
{
    SDL_Rect real_srcrect;
    SDL_FRect real_dstrect;
    SDL_FPoint real_center;
    int retval;
    int use_rendergeometry;

    if (flip == SDL_FLIP_NONE && (int)(angle / 360) == angle / 360) { /* fast path when we don't need rotation or flipping */
        return SDL_RenderCopyF(renderer, texture, srcrect, dstrect);
    }

    CHECK_RENDERER_MAGIC(renderer, -1);
    CHECK_TEXTURE_MAGIC(texture, -1);

    if (renderer != texture->renderer) {
        return SDL_SetError("Texture was not created with this renderer");
    }
    if (!renderer->QueueCopyEx && !renderer->QueueGeometry) {
        return SDL_SetError("Renderer does not support RenderCopyEx");
    }

#if DONT_DRAW_WHILE_HIDDEN
    /* Don't draw while we're hidden */
    if (renderer->hidden) {
        return 0;
    }
#endif

    use_rendergeometry = (!renderer->QueueCopyEx);

    real_srcrect.x = 0;
    real_srcrect.y = 0;
    real_srcrect.w = texture->w;
    real_srcrect.h = texture->h;
    if (srcrect) {
        if (!SDL_IntersectRect(srcrect, &real_srcrect, &real_srcrect)) {
            return 0;
        }
    }

    /* We don't intersect the dstrect with the viewport as RenderCopy does because of potential rotation clipping issues... TODO: should we? */
    if (dstrect) {
        real_dstrect = *dstrect;
    } else {
        RenderGetViewportSize(renderer, &real_dstrect);
    }

    if (texture->native) {
        texture = texture->native;
    }

    if (center) {
        real_center = *center;
    } else {
        real_center.x = real_dstrect.w / 2.0f;
        real_center.y = real_dstrect.h / 2.0f;
    }

    texture->last_command_generation = renderer->render_command_generation;

    if (use_rendergeometry) {
        float xy[8];
        const int xy_stride = 2 * sizeof(float);
        float uv[8];
        const int uv_stride = 2 * sizeof(float);
        const int num_vertices = 4;
        const int *indices = renderer->rect_index_order;
        const int num_indices = 6;
        const int size_indices = 4;
        float minu, minv, maxu, maxv;
        float minx, miny, maxx, maxy;
        float centerx, centery;

        float s_minx, s_miny, s_maxx, s_maxy;
        float c_minx, c_miny, c_maxx, c_maxy;

        const float radian_angle = (float)((M_PI * angle) / 180.0);
        const float s = SDL_sinf(radian_angle);
        const float c = SDL_cosf(radian_angle);

        minu = (float)(real_srcrect.x) / (float)texture->w;
        minv = (float)(real_srcrect.y) / (float)texture->h;
        maxu = (float)(real_srcrect.x + real_srcrect.w) / (float)texture->w;
        maxv = (float)(real_srcrect.y + real_srcrect.h) / (float)texture->h;

        centerx = real_center.x + real_dstrect.x;
        centery = real_center.y + real_dstrect.y;

        if (flip & SDL_FLIP_HORIZONTAL) {
            minx = real_dstrect.x + real_dstrect.w;
            maxx = real_dstrect.x;
        } else {
            minx = real_dstrect.x;
            maxx = real_dstrect.x + real_dstrect.w;
        }

        if (flip & SDL_FLIP_VERTICAL) {
            miny = real_dstrect.y + real_dstrect.h;
            maxy = real_dstrect.y;
        } else {
            miny = real_dstrect.y;
            maxy = real_dstrect.y + real_dstrect.h;
        }

        uv[0] = minu;
        uv[1] = minv;
        uv[2] = maxu;
        uv[3] = minv;
        uv[4] = maxu;
        uv[5] = maxv;
        uv[6] = minu;
        uv[7] = maxv;

        /* apply rotation with 2x2 matrix ( c -s )
         *                                ( s  c ) */
        s_minx = s * (minx - centerx);
        s_miny = s * (miny - centery);
        s_maxx = s * (maxx - centerx);
        s_maxy = s * (maxy - centery);
        c_minx = c * (minx - centerx);
        c_miny = c * (miny - centery);
        c_maxx = c * (maxx - centerx);
        c_maxy = c * (maxy - centery);

        /* (minx, miny) */
        xy[0] = (c_minx - s_miny) + centerx;
        xy[1] = (s_minx + c_miny) + centery;
        /* (maxx, miny) */
        xy[2] = (c_maxx - s_miny) + centerx;
        xy[3] = (s_maxx + c_miny) + centery;
        /* (maxx, maxy) */
        xy[4] = (c_maxx - s_maxy) + centerx;
        xy[5] = (s_maxx + c_maxy) + centery;
        /* (minx, maxy) */
        xy[6] = (c_minx - s_maxy) + centerx;
        xy[7] = (s_minx + c_maxy) + centery;

        retval = QueueCmdGeometry(renderer, texture,
                                  xy, xy_stride, &texture->color, 0 /* color_stride */, uv, uv_stride,
                                  num_vertices,
                                  indices, num_indices, size_indices,
                                  renderer->scale.x, renderer->scale.y);
    } else {

        retval = QueueCmdCopyEx(renderer, texture, &real_srcrect, &real_dstrect, angle, &real_center, flip, renderer->scale.x, renderer->scale.y);
    }
    return retval < 0 ? retval : FlushRenderCommandsIfNotBatching(renderer);
}

int SDL_RenderGeometry(SDL_Renderer *renderer,
                       SDL_Texture *texture,
                       const SDL_Vertex *vertices, int num_vertices,
                       const int *indices, int num_indices)
{
    if (vertices) {
        const float *xy = &vertices->position.x;
        int xy_stride = sizeof(SDL_Vertex);
        const SDL_Color *color = &vertices->color;
        int color_stride = sizeof(SDL_Vertex);
        const float *uv = &vertices->tex_coord.x;
        int uv_stride = sizeof(SDL_Vertex);
        int size_indices = 4;
        return SDL_RenderGeometryRaw(renderer, texture, xy, xy_stride, color, color_stride, uv, uv_stride, num_vertices, indices, num_indices, size_indices);
    } else {
        return SDL_InvalidParamError("vertices");
    }
}

static int remap_one_indice(
    int prev,
    int k,
    SDL_Texture *texture,
    const float *xy, int xy_stride,
    const SDL_Color *color, int color_stride,
    const float *uv, int uv_stride)
{
    const float *xy0_, *xy1_, *uv0_, *uv1_;
    int col0_, col1_;
    xy0_ = (const float *)((const char *)xy + prev * xy_stride);
    xy1_ = (const float *)((const char *)xy + k * xy_stride);
    if (xy0_[0] != xy1_[0]) {
        return k;
    }
    if (xy0_[1] != xy1_[1]) {
        return k;
    }
    if (texture) {
        uv0_ = (const float *)((const char *)uv + prev * uv_stride);
        uv1_ = (const float *)((const char *)uv + k * uv_stride);
        if (uv0_[0] != uv1_[0]) {
            return k;
        }
        if (uv0_[1] != uv1_[1]) {
            return k;
        }
    }
    col0_ = *(const int *)((const char *)color + prev * color_stride);
    col1_ = *(const int *)((const char *)color + k * color_stride);

    if (col0_ != col1_) {
        return k;
    }

    return prev;
}

static int remap_indices(
    int prev[3],
    int k,
    SDL_Texture *texture,
    const float *xy, int xy_stride,
    const SDL_Color *color, int color_stride,
    const float *uv, int uv_stride)
{
    int i;
    if (prev[0] == -1) {
        return k;
    }

    for (i = 0; i < 3; i++) {
        int new_k = remap_one_indice(prev[i], k, texture, xy, xy_stride, color, color_stride, uv, uv_stride);
        if (new_k != k) {
            return new_k;
        }
    }
    return k;
}

#define DEBUG_SW_RENDER_GEOMETRY 0
/* For the software renderer, try to reinterpret triangles as SDL_Rect */
static int SDLCALL SDL_SW_RenderGeometryRaw(SDL_Renderer *renderer,
                                            SDL_Texture *texture,
                                            const float *xy, int xy_stride,
                                            const SDL_Color *color, int color_stride,
                                            const float *uv, int uv_stride,
                                            int num_vertices,
                                            const void *indices, int num_indices, int size_indices)
{
    int i;
    int retval = 0;
    int count = indices ? num_indices : num_vertices;
    int prev[3]; /* Previous triangle vertex indices */
    int texw = 0, texh = 0;
    SDL_BlendMode blendMode = SDL_BLENDMODE_NONE;
    Uint8 r = 0, g = 0, b = 0, a = 0;

    /* Save */
    SDL_GetRenderDrawBlendMode(renderer, &blendMode);
    SDL_GetRenderDrawColor(renderer, &r, &g, &b, &a);

    if (texture) {
        SDL_QueryTexture(texture, NULL, NULL, &texw, &texh);
    }

    prev[0] = -1;
    prev[1] = -1;
    prev[2] = -1;
    size_indices = indices ? size_indices : 0;

    for (i = 0; i < count; i += 3) {
        int k0, k1, k2; /* Current triangle indices */
        int is_quad = 1;
#if DEBUG_SW_RENDER_GEOMETRY
        int is_uniform = 1;
        int is_rectangle = 1;
#endif
        int A = -1;  /* Top left vertex */
        int B = -1;  /* Bottom right vertex */
        int C = -1;  /* Third vertex of current triangle */
        int C2 = -1; /* Last, vertex of previous triangle */

        if (size_indices == 4) {
            k0 = ((const Uint32 *)indices)[i];
            k1 = ((const Uint32 *)indices)[i + 1];
            k2 = ((const Uint32 *)indices)[i + 2];
        } else if (size_indices == 2) {
            k0 = ((const Uint16 *)indices)[i];
            k1 = ((const Uint16 *)indices)[i + 1];
            k2 = ((const Uint16 *)indices)[i + 2];
        } else if (size_indices == 1) {
            k0 = ((const Uint8 *)indices)[i];
            k1 = ((const Uint8 *)indices)[i + 1];
            k2 = ((const Uint8 *)indices)[i + 2];
        } else {
            /* Vertices were not provided by indices. Maybe some are duplicated.
             * We try to indentificate the duplicates by comparing with the previous three vertices */
            k0 = remap_indices(prev, i, texture, xy, xy_stride, color, color_stride, uv, uv_stride);
            k1 = remap_indices(prev, i + 1, texture, xy, xy_stride, color, color_stride, uv, uv_stride);
            k2 = remap_indices(prev, i + 2, texture, xy, xy_stride, color, color_stride, uv, uv_stride);
        }

        if (prev[0] == -1) {
            prev[0] = k0;
            prev[1] = k1;
            prev[2] = k2;
            continue;
        }

        /* Two triangles forming a quadialateral,
         * prev and current triangles must have exactly 2 common vertices */
        {
            int cnt = 0, j = 3;
            while (j--) {
                int p = prev[j];
                if (p == k0 || p == k1 || p == k2) {
                    cnt++;
                }
            }
            is_quad = (cnt == 2);
        }

        /* Identify vertices */
        if (is_quad) {
            const float *xy0_, *xy1_, *xy2_;
            float x0, x1, x2;
            float y0, y1, y2;
            xy0_ = (const float *)((const char *)xy + k0 * xy_stride);
            xy1_ = (const float *)((const char *)xy + k1 * xy_stride);
            xy2_ = (const float *)((const char *)xy + k2 * xy_stride);
            x0 = xy0_[0];
            y0 = xy0_[1];
            x1 = xy1_[0];
            y1 = xy1_[1];
            x2 = xy2_[0];
            y2 = xy2_[1];

            /* Find top-left */
            if (x0 <= x1 && y0 <= y1) {
                if (x0 <= x2 && y0 <= y2) {
                    A = k0;
                } else {
                    A = k2;
                }
            } else {
                if (x1 <= x2 && y1 <= y2) {
                    A = k1;
                } else {
                    A = k2;
                }
            }

            /* Find bottom-right */
            if (x0 >= x1 && y0 >= y1) {
                if (x0 >= x2 && y0 >= y2) {
                    B = k0;
                } else {
                    B = k2;
                }
            } else {
                if (x1 >= x2 && y1 >= y2) {
                    B = k1;
                } else {
                    B = k2;
                }
            }

            /* Find C */
            if (k0 != A && k0 != B) {
                C = k0;
            } else if (k1 != A && k1 != B) {
                C = k1;
            } else {
                C = k2;
            }

            /* Find C2 */
            if (prev[0] != A && prev[0] != B) {
                C2 = prev[0];
            } else if (prev[1] != A && prev[1] != B) {
                C2 = prev[1];
            } else {
                C2 = prev[2];
            }

            xy0_ = (const float *)((const char *)xy + A * xy_stride);
            xy1_ = (const float *)((const char *)xy + B * xy_stride);
            xy2_ = (const float *)((const char *)xy + C * xy_stride);
            x0 = xy0_[0];
            y0 = xy0_[1];
            x1 = xy1_[0];
            y1 = xy1_[1];
            x2 = xy2_[0];
            y2 = xy2_[1];

            /* Check if triangle A B C is rectangle */
            if ((x0 == x2 && y1 == y2) || (y0 == y2 && x1 == x2)) {
                /* ok */
            } else {
                is_quad = 0;
#if DEBUG_SW_RENDER_GEOMETRY
                is_rectangle = 0;
#endif
            }

            xy2_ = (const float *)((const char *)xy + C2 * xy_stride);
            x2 = xy2_[0];
            y2 = xy2_[1];

            /* Check if triangle A B C2 is rectangle */
            if ((x0 == x2 && y1 == y2) || (y0 == y2 && x1 == x2)) {
                /* ok */
            } else {
                is_quad = 0;
#if DEBUG_SW_RENDER_GEOMETRY
                is_rectangle = 0;
#endif
            }
        }

        /* Check if uniformly colored */
        if (is_quad) {
            const int col0_ = *(const int *)((const char *)color + A * color_stride);
            const int col1_ = *(const int *)((const char *)color + B * color_stride);
            const int col2_ = *(const int *)((const char *)color + C * color_stride);
            const int col3_ = *(const int *)((const char *)color + C2 * color_stride);
            if (col0_ == col1_ && col0_ == col2_ && col0_ == col3_) {
                /* ok */
            } else {
                is_quad = 0;
#if DEBUG_SW_RENDER_GEOMETRY
                is_uniform = 0;
#endif
            }
        }

        /* Start rendering rect */
        if (is_quad) {
            SDL_Rect s;
            SDL_FRect d;
            const float *xy0_, *xy1_, *uv0_, *uv1_;
            SDL_Color col0_ = *(const SDL_Color *)((const char *)color + k0 * color_stride);

            xy0_ = (const float *)((const char *)xy + A * xy_stride);
            xy1_ = (const float *)((const char *)xy + B * xy_stride);

            if (texture) {
                uv0_ = (const float *)((const char *)uv + A * uv_stride);
                uv1_ = (const float *)((const char *)uv + B * uv_stride);
                s.x = (int)(uv0_[0] * texw);
                s.y = (int)(uv0_[1] * texh);
                s.w = (int)(uv1_[0] * texw - s.x);
                s.h = (int)(uv1_[1] * texh - s.y);
            }

            d.x = xy0_[0];
            d.y = xy0_[1];
            d.w = xy1_[0] - d.x;
            d.h = xy1_[1] - d.y;

            /* Rect + texture */
            if (texture && s.w != 0 && s.h != 0) {
                SDL_SetTextureAlphaMod(texture, col0_.a);
                SDL_SetTextureColorMod(texture, col0_.r, col0_.g, col0_.b);
                if (s.w > 0 && s.h > 0) {
                    SDL_RenderCopyF(renderer, texture, &s, &d);
                } else {
                    int flags = 0;
                    if (s.w < 0) {
                        flags |= SDL_FLIP_HORIZONTAL;
                        s.w *= -1;
                        s.x -= s.w;
                    }
                    if (s.h < 0) {
                        flags |= SDL_FLIP_VERTICAL;
                        s.h *= -1;
                        s.y -= s.h;
                    }
                    SDL_RenderCopyExF(renderer, texture, &s, &d, 0, NULL, flags);
                }

#if DEBUG_SW_RENDER_GEOMETRY
                SDL_Log("Rect-COPY: RGB %d %d %d - Alpha:%d - texture=%p: src=(%d,%d, %d x %d) dst (%f, %f, %f x %f)", col0_.r, col0_.g, col0_.b, col0_.a,
                        (void *)texture, s.x, s.y, s.w, s.h, d.x, d.y, d.w, d.h);
#endif
            } else if (d.w != 0.0f && d.h != 0.0f) { /* Rect, no texture */
                SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
                SDL_SetRenderDrawColor(renderer, col0_.r, col0_.g, col0_.b, col0_.a);
                SDL_RenderFillRectF(renderer, &d);
#if DEBUG_SW_RENDER_GEOMETRY
                SDL_Log("Rect-FILL: RGB %d %d %d - Alpha:%d - texture=%p: dst (%f, %f, %f x %f)", col0_.r, col0_.g, col0_.b, col0_.a,
                        (void *)texture, d.x, d.y, d.w, d.h);
            } else {
                SDL_Log("Rect-DISMISS: RGB %d %d %d - Alpha:%d - texture=%p: src=(%d,%d, %d x %d) dst (%f, %f, %f x %f)", col0_.r, col0_.g, col0_.b, col0_.a,
                        (void *)texture, s.x, s.y, s.w, s.h, d.x, d.y, d.w, d.h);
#endif
            }

            prev[0] = -1;
        } else {
            /* Render triangles */
            if (prev[0] != -1) {
#if DEBUG_SW_RENDER_GEOMETRY
                SDL_Log("Triangle %d %d %d - is_uniform:%d is_rectangle:%d", prev[0], prev[1], prev[2], is_uniform, is_rectangle);
#endif
                retval = QueueCmdGeometry(renderer, texture,
                                          xy, xy_stride, color, color_stride, uv, uv_stride,
                                          num_vertices, prev, 3, 4, renderer->scale.x, renderer->scale.y);
                if (retval < 0) {
                    goto end;
                } else {
                    FlushRenderCommandsIfNotBatching(renderer);
                }
            }

            prev[0] = k0;
            prev[1] = k1;
            prev[2] = k2;
        }
    } /* End for (), next triangle */

    if (prev[0] != -1) {
        /* flush the last triangle */
#if DEBUG_SW_RENDER_GEOMETRY
        SDL_Log("Last triangle %d %d %d", prev[0], prev[1], prev[2]);
#endif
        retval = QueueCmdGeometry(renderer, texture,
                                  xy, xy_stride, color, color_stride, uv, uv_stride,
                                  num_vertices, prev, 3, 4, renderer->scale.x, renderer->scale.y);
        if (retval < 0) {
            goto end;
        } else {
            FlushRenderCommandsIfNotBatching(renderer);
        }
    }

end:
    /* Restore */
    SDL_SetRenderDrawBlendMode(renderer, blendMode);
    SDL_SetRenderDrawColor(renderer, r, g, b, a);

    return retval;
}

int SDL_RenderGeometryRaw(SDL_Renderer *renderer,
                          SDL_Texture *texture,
                          const float *xy, int xy_stride,
                          const SDL_Color *color, int color_stride,
                          const float *uv, int uv_stride,
                          int num_vertices,
                          const void *indices, int num_indices, int size_indices)
{
    int i;
    int retval = 0;
    int count = indices ? num_indices : num_vertices;

    CHECK_RENDERER_MAGIC(renderer, -1);

    if (!renderer->QueueGeometry) {
        return SDL_Unsupported();
    }

    if (texture) {
        CHECK_TEXTURE_MAGIC(texture, -1);

        if (renderer != texture->renderer) {
            return SDL_SetError("Texture was not created with this renderer");
        }
    }

    if (!xy) {
        return SDL_InvalidParamError("xy");
    }

    if (!color) {
        return SDL_InvalidParamError("color");
    }

    if (texture && !uv) {
        return SDL_InvalidParamError("uv");
    }

    if (count % 3 != 0) {
        return SDL_InvalidParamError(indices ? "num_indices" : "num_vertices");
    }

    if (indices) {
        if (size_indices != 1 && size_indices != 2 && size_indices != 4) {
            return SDL_InvalidParamError("size_indices");
        }
    } else {
        size_indices = 0;
    }

#if DONT_DRAW_WHILE_HIDDEN
    /* Don't draw while we're hidden */
    if (renderer->hidden) {
        return 0;
    }
#endif

    if (num_vertices < 3) {
        return 0;
    }

    if (texture && texture->native) {
        texture = texture->native;
    }

    if (texture) {
        for (i = 0; i < num_vertices; ++i) {
            const float *uv_ = (const float *)((const char *)uv + i * uv_stride);
            float u = uv_[0];
            float v = uv_[1];
            if (u < 0.0f || v < 0.0f || u > 1.0f || v > 1.0f) {
                return SDL_SetError("Values of 'uv' out of bounds %f %f at %d/%d", u, v, i, num_vertices);
            }
        }
    }

    if (indices) {
        for (i = 0; i < num_indices; ++i) {
            int j;
            if (size_indices == 4) {
                j = ((const Uint32 *)indices)[i];
            } else if (size_indices == 2) {
                j = ((const Uint16 *)indices)[i];
            } else {
                j = ((const Uint8 *)indices)[i];
            }
            if (j < 0 || j >= num_vertices) {
                return SDL_SetError("Values of 'indices' out of bounds");
            }
        }
    }

    if (texture) {
        texture->last_command_generation = renderer->render_command_generation;
    }

    /* For the software renderer, try to reinterpret triangles as SDL_Rect */
    if (renderer->info.flags & SDL_RENDERER_SOFTWARE) {
        return SDL_SW_RenderGeometryRaw(renderer, texture,
                                        xy, xy_stride, color, color_stride, uv, uv_stride, num_vertices,
                                        indices, num_indices, size_indices);
    }

    retval = QueueCmdGeometry(renderer, texture,
                              xy, xy_stride, color, color_stride, uv, uv_stride,
                              num_vertices,
                              indices, num_indices, size_indices,
                              renderer->scale.x, renderer->scale.y);

    return retval < 0 ? retval : FlushRenderCommandsIfNotBatching(renderer);
}

int SDL_RenderReadPixels(SDL_Renderer *renderer, const SDL_Rect *rect,
                         Uint32 format, void *pixels, int pitch)
{
    SDL_Rect real_rect;

    CHECK_RENDERER_MAGIC(renderer, -1);

    if (!renderer->RenderReadPixels) {
        return SDL_Unsupported();
    }

    FlushRenderCommands(renderer); /* we need to render before we read the results. */

    if (!format) {
        if (!renderer->target) {
            format = SDL_GetWindowPixelFormat(renderer->window);
        } else {
            format = renderer->target->format;
        }
    }

    real_rect.x = (int)SDL_floor(renderer->viewport.x);
    real_rect.y = (int)SDL_floor(renderer->viewport.y);
    real_rect.w = (int)SDL_floor(renderer->viewport.w);
    real_rect.h = (int)SDL_floor(renderer->viewport.h);
    if (rect) {
        if (!SDL_IntersectRect(rect, &real_rect, &real_rect)) {
            return 0;
        }
        if (real_rect.y > rect->y) {
            pixels = (Uint8 *)pixels + pitch * (real_rect.y - rect->y);
        }
        if (real_rect.x > rect->x) {
            int bpp = SDL_BYTESPERPIXEL(format);
            pixels = (Uint8 *)pixels + bpp * (real_rect.x - rect->x);
        }
    }

    return renderer->RenderReadPixels(renderer, &real_rect,
                                      format, pixels, pitch);
}

static void SDL_RenderSimulateVSync(SDL_Renderer *renderer)
{
    Uint32 now, elapsed;
    const Uint32 interval = renderer->simulate_vsync_interval;

    if (!interval) {
        /* We can't do sub-ms delay, so just return here */
        return;
    }

    now = SDL_GetTicks();
    elapsed = (now - renderer->last_present);
    if (elapsed < interval) {
        Uint32 duration = (interval - elapsed);
        SDL_Delay(duration);
        now = SDL_GetTicks();
    }

    elapsed = (now - renderer->last_present);
    if (!renderer->last_present || elapsed > 1000) {
        /* It's been too long, reset the presentation timeline */
        renderer->last_present = now;
    } else {
        renderer->last_present += (elapsed / interval) * interval;
    }
}

void SDL_RenderPresent(SDL_Renderer *renderer)
{
    SDL_bool presented = SDL_TRUE;

    CHECK_RENDERER_MAGIC(renderer, );

    FlushRenderCommands(renderer); /* time to send everything to the GPU! */

#if DONT_DRAW_WHILE_HIDDEN
    /* Don't present while we're hidden */
    if (renderer->hidden) {
        presented = SDL_FALSE;
    } else
#endif
        if (renderer->RenderPresent(renderer) < 0) {
        presented = SDL_FALSE;
    }

    if (renderer->simulate_vsync ||
        (!presented && renderer->wanted_vsync)) {
        SDL_RenderSimulateVSync(renderer);
    }
}

void SDL_DestroyTexture(SDL_Texture *texture)
{
    SDL_Renderer *renderer;

    CHECK_TEXTURE_MAGIC(texture, );

    renderer = texture->renderer;
    if (texture == renderer->target) {
        SDL_SetRenderTarget(renderer, NULL); /* implies command queue flush */
    } else {
        FlushRenderCommandsIfTextureNeeded(texture);
    }

    texture->magic = NULL;

    if (texture->next) {
        texture->next->prev = texture->prev;
    }
    if (texture->prev) {
        texture->prev->next = texture->next;
    } else {
        renderer->textures = texture->next;
    }

    if (texture->native) {
        SDL_DestroyTexture(texture->native);
    }
#if SDL_HAVE_YUV
    if (texture->yuv) {
        SDL_SW_DestroyYUVTexture(texture->yuv);
    }
#endif
    SDL_free(texture->pixels);

    renderer->DestroyTexture(renderer, texture);

    SDL_FreeSurface(texture->locked_surface);
    texture->locked_surface = NULL;

    SDL_free(texture);
}

void SDL_DestroyRenderer(SDL_Renderer *renderer)
{
    SDL_RenderCommand *cmd;

    CHECK_RENDERER_MAGIC(renderer, );

    SDL_DelEventWatch(SDL_RendererEventWatch, renderer);

    if (renderer->render_commands_tail) {
        renderer->render_commands_tail->next = renderer->render_commands_pool;
        cmd = renderer->render_commands;
    } else {
        cmd = renderer->render_commands_pool;
    }

    renderer->render_commands_pool = NULL;
    renderer->render_commands_tail = NULL;
    renderer->render_commands = NULL;

    while (cmd) {
        SDL_RenderCommand *next = cmd->next;
        SDL_free(cmd);
        cmd = next;
    }

    SDL_free(renderer->vertex_data);

    /* Free existing textures for this renderer */
    while (renderer->textures) {
        SDL_Texture *tex = renderer->textures;
        (void)tex;
        SDL_DestroyTexture(renderer->textures);
        SDL_assert(tex != renderer->textures); /* satisfy static analysis. */
    }

    if (renderer->window) {
        SDL_SetWindowData(renderer->window, SDL_WINDOWRENDERDATA, NULL);
    }

    /* It's no longer magical... */
    renderer->magic = NULL;

    /* Free the target mutex */
    SDL_DestroyMutex(renderer->target_mutex);
    renderer->target_mutex = NULL;

    /* Free the renderer instance */
    renderer->DestroyRenderer(renderer);
}

int SDL_GL_BindTexture(SDL_Texture *texture, float *texw, float *texh)
{
    SDL_Renderer *renderer;

    CHECK_TEXTURE_MAGIC(texture, -1);
    renderer = texture->renderer;
    if (texture->native) {
        return SDL_GL_BindTexture(texture->native, texw, texh);
    } else if (renderer && renderer->GL_BindTexture) {
        FlushRenderCommandsIfTextureNeeded(texture); /* in case the app is going to mess with it. */
        return renderer->GL_BindTexture(renderer, texture, texw, texh);
    } else {
        return SDL_Unsupported();
    }
}

int SDL_GL_UnbindTexture(SDL_Texture *texture)
{
    SDL_Renderer *renderer;

    CHECK_TEXTURE_MAGIC(texture, -1);
    renderer = texture->renderer;
    if (texture->native) {
        return SDL_GL_UnbindTexture(texture->native);
    } else if (renderer && renderer->GL_UnbindTexture) {
        FlushRenderCommandsIfTextureNeeded(texture); /* in case the app messed with it. */
        return renderer->GL_UnbindTexture(renderer, texture);
    }

    return SDL_Unsupported();
}

void *SDL_RenderGetMetalLayer(SDL_Renderer *renderer)
{
    CHECK_RENDERER_MAGIC(renderer, NULL);

    if (renderer->GetMetalLayer) {
        FlushRenderCommands(renderer); /* in case the app is going to mess with it. */
        return renderer->GetMetalLayer(renderer);
    }
    return NULL;
}

void *SDL_RenderGetMetalCommandEncoder(SDL_Renderer *renderer)
{
    CHECK_RENDERER_MAGIC(renderer, NULL);

    if (renderer->GetMetalCommandEncoder) {
        FlushRenderCommands(renderer); /* in case the app is going to mess with it. */
        return renderer->GetMetalCommandEncoder(renderer);
    }
    return NULL;
}

static SDL_BlendMode SDL_GetShortBlendMode(SDL_BlendMode blendMode)
{
    if (blendMode == SDL_BLENDMODE_NONE_FULL) {
        return SDL_BLENDMODE_NONE;
    }
    if (blendMode == SDL_BLENDMODE_BLEND_FULL) {
        return SDL_BLENDMODE_BLEND;
    }
    if (blendMode == SDL_BLENDMODE_ADD_FULL) {
        return SDL_BLENDMODE_ADD;
    }
    if (blendMode == SDL_BLENDMODE_MOD_FULL) {
        return SDL_BLENDMODE_MOD;
    }
    if (blendMode == SDL_BLENDMODE_MUL_FULL) {
        return SDL_BLENDMODE_MUL;
    }
    return blendMode;
}

static SDL_BlendMode SDL_GetLongBlendMode(SDL_BlendMode blendMode)
{
    if (blendMode == SDL_BLENDMODE_NONE) {
        return SDL_BLENDMODE_NONE_FULL;
    }
    if (blendMode == SDL_BLENDMODE_BLEND) {
        return SDL_BLENDMODE_BLEND_FULL;
    }
    if (blendMode == SDL_BLENDMODE_ADD) {
        return SDL_BLENDMODE_ADD_FULL;
    }
    if (blendMode == SDL_BLENDMODE_MOD) {
        return SDL_BLENDMODE_MOD_FULL;
    }
    if (blendMode == SDL_BLENDMODE_MUL) {
        return SDL_BLENDMODE_MUL_FULL;
    }
    return blendMode;
}

SDL_BlendMode SDL_ComposeCustomBlendMode(SDL_BlendFactor srcColorFactor, SDL_BlendFactor dstColorFactor,
                           SDL_BlendOperation colorOperation,
                           SDL_BlendFactor srcAlphaFactor, SDL_BlendFactor dstAlphaFactor,
                           SDL_BlendOperation alphaOperation)
{
    SDL_BlendMode blendMode = SDL_COMPOSE_BLENDMODE(srcColorFactor, dstColorFactor, colorOperation,
                                                    srcAlphaFactor, dstAlphaFactor, alphaOperation);
    return SDL_GetShortBlendMode(blendMode);
}

SDL_BlendFactor SDL_GetBlendModeSrcColorFactor(SDL_BlendMode blendMode)
{
    blendMode = SDL_GetLongBlendMode(blendMode);
    return (SDL_BlendFactor)(((Uint32)blendMode >> 4) & 0xF);
}

SDL_BlendFactor SDL_GetBlendModeDstColorFactor(SDL_BlendMode blendMode)
{
    blendMode = SDL_GetLongBlendMode(blendMode);
    return (SDL_BlendFactor)(((Uint32)blendMode >> 8) & 0xF);
}

SDL_BlendOperation SDL_GetBlendModeColorOperation(SDL_BlendMode blendMode)
{
    blendMode = SDL_GetLongBlendMode(blendMode);
    return (SDL_BlendOperation)(((Uint32)blendMode >> 0) & 0xF);
}

SDL_BlendFactor SDL_GetBlendModeSrcAlphaFactor(SDL_BlendMode blendMode)
{
    blendMode = SDL_GetLongBlendMode(blendMode);
    return (SDL_BlendFactor)(((Uint32)blendMode >> 20) & 0xF);
}

SDL_BlendFactor SDL_GetBlendModeDstAlphaFactor(SDL_BlendMode blendMode)
{
    blendMode = SDL_GetLongBlendMode(blendMode);
    return (SDL_BlendFactor)(((Uint32)blendMode >> 24) & 0xF);
}

SDL_BlendOperation SDL_GetBlendModeAlphaOperation(SDL_BlendMode blendMode)
{
    blendMode = SDL_GetLongBlendMode(blendMode);
    return (SDL_BlendOperation)(((Uint32)blendMode >> 16) & 0xF);
}

int SDL_RenderSetVSync(SDL_Renderer *renderer, int vsync)
{
    CHECK_RENDERER_MAGIC(renderer, -1);

    if (vsync != 0 && vsync != 1) {
        return SDL_Unsupported();
    }

    renderer->wanted_vsync = vsync ? SDL_TRUE : SDL_FALSE;

    if (!renderer->SetVSync ||
        renderer->SetVSync(renderer, vsync) != 0) {
        renderer->simulate_vsync = vsync ? SDL_TRUE : SDL_FALSE;
        if (renderer->simulate_vsync) {
            renderer->info.flags |= SDL_RENDERER_PRESENTVSYNC;
        } else {
            renderer->info.flags &= ~SDL_RENDERER_PRESENTVSYNC;
        }
    } else {
        renderer->simulate_vsync = SDL_FALSE;
    }
    return 0;
}

/* vi: set ts=4 sw=4 expandtab: */
