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

#if SDL_VIDEO_RENDER_OGL_ES

#include "SDL_hints.h"
#include "../../video/SDL_sysvideo.h" /* For SDL_GL_SwapWindowWithResult */
#include "SDL_opengles.h"
#include "../SDL_sysrender.h"
#include "../../SDL_utils_c.h"

/* To prevent unnecessary window recreation,
 * these should match the defaults selected in SDL_GL_ResetAttributes
 */

#define RENDERER_CONTEXT_MAJOR 1
#define RENDERER_CONTEXT_MINOR 1

#if defined(SDL_VIDEO_DRIVER_PANDORA)

/* Empty function stub to get OpenGL ES 1.x support without  */
/* OpenGL ES extension GL_OES_draw_texture supported         */
GL_API void GL_APIENTRY
glDrawTexiOES(GLint x, GLint y, GLint z, GLint width, GLint height)
{
    return;
}

#endif /* SDL_VIDEO_DRIVER_PANDORA */

/* OpenGL ES 1.1 renderer implementation, based on the OpenGL renderer */

/* Used to re-create the window with OpenGL ES capability */
extern int SDL_RecreateWindow(SDL_Window *window, Uint32 flags);

static const float inv255f = 1.0f / 255.0f;

typedef struct GLES_FBOList GLES_FBOList;

struct GLES_FBOList
{
    Uint32 w, h;
    GLuint FBO;
    GLES_FBOList *next;
};

typedef struct
{
    SDL_Rect viewport;
    SDL_bool viewport_dirty;
    SDL_Texture *texture;
    SDL_Texture *target;
    int drawablew;
    int drawableh;
    SDL_BlendMode blend;
    SDL_bool cliprect_enabled_dirty;
    SDL_bool cliprect_enabled;
    SDL_bool cliprect_dirty;
    SDL_Rect cliprect;
    SDL_bool texturing;
    Uint32 color;
    Uint32 clear_color;
} GLES_DrawStateCache;

typedef struct
{
    SDL_GLContext context;

#define SDL_PROC(ret, func, params) ret (APIENTRY *func) params;
#define SDL_PROC_OES                SDL_PROC
#include "SDL_glesfuncs.h"
#undef SDL_PROC
#undef SDL_PROC_OES
    SDL_bool GL_OES_framebuffer_object_supported;
    GLES_FBOList *framebuffers;
    GLuint window_framebuffer;

    SDL_bool GL_OES_blend_func_separate_supported;
    SDL_bool GL_OES_blend_equation_separate_supported;
    SDL_bool GL_OES_blend_subtract_supported;
    SDL_bool GL_EXT_blend_minmax_supported;

    GLES_DrawStateCache drawstate;
} GLES_RenderData;

typedef struct
{
    GLuint texture;
    GLenum type;
    GLfloat texw;
    GLfloat texh;
    GLenum format;
    GLenum formattype;
    void *pixels;
    int pitch;
    GLES_FBOList *fbo;
} GLES_TextureData;

static int GLES_SetError(const char *prefix, GLenum result)
{
    const char *error;

    switch (result) {
    case GL_NO_ERROR:
        error = "GL_NO_ERROR";
        break;
    case GL_INVALID_ENUM:
        error = "GL_INVALID_ENUM";
        break;
    case GL_INVALID_VALUE:
        error = "GL_INVALID_VALUE";
        break;
    case GL_INVALID_OPERATION:
        error = "GL_INVALID_OPERATION";
        break;
    case GL_STACK_OVERFLOW:
        error = "GL_STACK_OVERFLOW";
        break;
    case GL_STACK_UNDERFLOW:
        error = "GL_STACK_UNDERFLOW";
        break;
    case GL_OUT_OF_MEMORY:
        error = "GL_OUT_OF_MEMORY";
        break;
    default:
        error = "UNKNOWN";
        break;
    }
    return SDL_SetError("%s: %s", prefix, error);
}

static int GLES_LoadFunctions(GLES_RenderData *data)
{
#ifdef SDL_VIDEO_DRIVER_UIKIT
#define __SDL_NOGETPROCADDR__
#elif defined(SDL_VIDEO_DRIVER_ANDROID)
#define __SDL_NOGETPROCADDR__
#elif defined(SDL_VIDEO_DRIVER_PANDORA)
#define __SDL_NOGETPROCADDR__
#endif

#ifdef __SDL_NOGETPROCADDR__
#define SDL_PROC(ret, func, params)     data->func = func;
#define SDL_PROC_OES(ret, func, params) data->func = func;
#else
#define SDL_PROC(ret, func, params)                                                           \
    do {                                                                                      \
        data->func = SDL_GL_GetProcAddress(#func);                                            \
        if (!data->func) {                                                                    \
            return SDL_SetError("Couldn't load GLES function %s: %s", #func, SDL_GetError()); \
        }                                                                                     \
    } while (0);
#define SDL_PROC_OES(ret, func, params)            \
    do {                                           \
        data->func = SDL_GL_GetProcAddress(#func); \
    } while (0);
#endif /* __SDL_NOGETPROCADDR__ */

#include "SDL_glesfuncs.h"
#undef SDL_PROC
#undef SDL_PROC_OES
    return 0;
}

static GLES_FBOList *GLES_GetFBO(GLES_RenderData *data, Uint32 w, Uint32 h)
{
    GLES_FBOList *result = data->framebuffers;
    while ((result) && ((result->w != w) || (result->h != h))) {
        result = result->next;
    }
    if (!result) {
        result = SDL_malloc(sizeof(GLES_FBOList));
        result->w = w;
        result->h = h;
        data->glGenFramebuffersOES(1, &result->FBO);
        result->next = data->framebuffers;
        data->framebuffers = result;
    }
    return result;
}

static int GLES_ActivateRenderer(SDL_Renderer *renderer)
{
    GLES_RenderData *data = (GLES_RenderData *)renderer->driverdata;

    if (SDL_GL_GetCurrentContext() != data->context) {
        if (SDL_GL_MakeCurrent(renderer->window, data->context) < 0) {
            return -1;
        }
    }

    return 0;
}

static void GLES_WindowEvent(SDL_Renderer *renderer, const SDL_WindowEvent *event)
{
    GLES_RenderData *data = (GLES_RenderData *)renderer->driverdata;

    if (event->event == SDL_WINDOWEVENT_MINIMIZED) {
        /* According to Apple documentation, we need to finish drawing NOW! */
        data->glFinish();
    }
}

static int GLES_GetOutputSize(SDL_Renderer *renderer, int *w, int *h)
{
    SDL_GL_GetDrawableSize(renderer->window, w, h);
    return 0;
}

static GLenum GetBlendFunc(SDL_BlendFactor factor)
{
    switch (factor) {
    case SDL_BLENDFACTOR_ZERO:
        return GL_ZERO;
    case SDL_BLENDFACTOR_ONE:
        return GL_ONE;
    case SDL_BLENDFACTOR_SRC_COLOR:
        return GL_SRC_COLOR;
    case SDL_BLENDFACTOR_ONE_MINUS_SRC_COLOR:
        return GL_ONE_MINUS_SRC_COLOR;
    case SDL_BLENDFACTOR_SRC_ALPHA:
        return GL_SRC_ALPHA;
    case SDL_BLENDFACTOR_ONE_MINUS_SRC_ALPHA:
        return GL_ONE_MINUS_SRC_ALPHA;
    case SDL_BLENDFACTOR_DST_COLOR:
        return GL_DST_COLOR;
    case SDL_BLENDFACTOR_ONE_MINUS_DST_COLOR:
        return GL_ONE_MINUS_DST_COLOR;
    case SDL_BLENDFACTOR_DST_ALPHA:
        return GL_DST_ALPHA;
    case SDL_BLENDFACTOR_ONE_MINUS_DST_ALPHA:
        return GL_ONE_MINUS_DST_ALPHA;
    default:
        return GL_INVALID_ENUM;
    }
}

static GLenum GetBlendEquation(SDL_BlendOperation operation)
{
    switch (operation) {
    case SDL_BLENDOPERATION_ADD:
        return GL_FUNC_ADD_OES;
    case SDL_BLENDOPERATION_SUBTRACT:
        return GL_FUNC_SUBTRACT_OES;
    case SDL_BLENDOPERATION_REV_SUBTRACT:
        return GL_FUNC_REVERSE_SUBTRACT_OES;
    case SDL_BLENDOPERATION_MINIMUM:
        return GL_MIN_EXT;
    case SDL_BLENDOPERATION_MAXIMUM:
        return GL_MAX_EXT;
    default:
        return GL_INVALID_ENUM;
    }
}

static SDL_bool GLES_SupportsBlendMode(SDL_Renderer *renderer, SDL_BlendMode blendMode)
{
    GLES_RenderData *data = (GLES_RenderData *)renderer->driverdata;
    SDL_BlendFactor srcColorFactor = SDL_GetBlendModeSrcColorFactor(blendMode);
    SDL_BlendFactor srcAlphaFactor = SDL_GetBlendModeSrcAlphaFactor(blendMode);
    SDL_BlendOperation colorOperation = SDL_GetBlendModeColorOperation(blendMode);
    SDL_BlendFactor dstColorFactor = SDL_GetBlendModeDstColorFactor(blendMode);
    SDL_BlendFactor dstAlphaFactor = SDL_GetBlendModeDstAlphaFactor(blendMode);
    SDL_BlendOperation alphaOperation = SDL_GetBlendModeAlphaOperation(blendMode);

    if (GetBlendFunc(srcColorFactor) == GL_INVALID_ENUM ||
        GetBlendFunc(srcAlphaFactor) == GL_INVALID_ENUM ||
        GetBlendEquation(colorOperation) == GL_INVALID_ENUM ||
        GetBlendFunc(dstColorFactor) == GL_INVALID_ENUM ||
        GetBlendFunc(dstAlphaFactor) == GL_INVALID_ENUM ||
        GetBlendEquation(alphaOperation) == GL_INVALID_ENUM) {
        return SDL_FALSE;
    }
    if ((srcColorFactor != srcAlphaFactor || dstColorFactor != dstAlphaFactor) && !data->GL_OES_blend_func_separate_supported) {
        return SDL_FALSE;
    }
    if (colorOperation != alphaOperation && !data->GL_OES_blend_equation_separate_supported) {
        return SDL_FALSE;
    }
    if (colorOperation != SDL_BLENDOPERATION_ADD && !data->GL_OES_blend_subtract_supported) {
        return SDL_FALSE;
    }
    if (colorOperation == SDL_BLENDOPERATION_MINIMUM && !data->GL_EXT_blend_minmax_supported) {
        return SDL_FALSE;
    }
    if (colorOperation == SDL_BLENDOPERATION_MAXIMUM && !data->GL_EXT_blend_minmax_supported) {
        return SDL_FALSE;
    }
    return SDL_TRUE;
}

static int GLES_CreateTexture(SDL_Renderer *renderer, SDL_Texture *texture)
{
    GLES_RenderData *renderdata = (GLES_RenderData *)renderer->driverdata;
    GLES_TextureData *data;
    GLint internalFormat;
    GLenum format, type;
    int texture_w, texture_h;
    GLenum scaleMode;
    GLenum result;

    GLES_ActivateRenderer(renderer);

    switch (texture->format) {
    case SDL_PIXELFORMAT_RGBA32:
        internalFormat = GL_RGBA;
        format = GL_RGBA;
        type = GL_UNSIGNED_BYTE;
        break;
    default:
        return SDL_SetError("Texture format not supported");
    }

    data = (GLES_TextureData *)SDL_calloc(1, sizeof(*data));
    if (!data) {
        return SDL_OutOfMemory();
    }

    if (texture->access == SDL_TEXTUREACCESS_STREAMING) {
        data->pitch = texture->w * SDL_BYTESPERPIXEL(texture->format);
        data->pixels = SDL_calloc(1, texture->h * data->pitch);
        if (!data->pixels) {
            SDL_free(data);
            return SDL_OutOfMemory();
        }
    }

    if (texture->access == SDL_TEXTUREACCESS_TARGET) {
        if (!renderdata->GL_OES_framebuffer_object_supported) {
            SDL_free(data);
            return SDL_SetError("GL_OES_framebuffer_object not supported");
        }
        data->fbo = GLES_GetFBO(renderer->driverdata, texture->w, texture->h);
    } else {
        data->fbo = NULL;
    }

    renderdata->glGetError();
    renderdata->glEnable(GL_TEXTURE_2D);
    renderdata->glGenTextures(1, &data->texture);
    result = renderdata->glGetError();
    if (result != GL_NO_ERROR) {
        if (texture->access == SDL_TEXTUREACCESS_STREAMING) {
            SDL_free(data->pixels);
        }
        SDL_free(data);
        return GLES_SetError("glGenTextures()", result);
    }

    data->type = GL_TEXTURE_2D;
    /* no NPOV textures allowed in OpenGL ES (yet) */
    texture_w = SDL_powerof2(texture->w);
    texture_h = SDL_powerof2(texture->h);
    data->texw = (GLfloat)texture->w / texture_w;
    data->texh = (GLfloat)texture->h / texture_h;

    data->format = format;
    data->formattype = type;
    scaleMode = (texture->scaleMode == SDL_ScaleModeNearest) ? GL_NEAREST : GL_LINEAR;
    renderdata->glBindTexture(data->type, data->texture);
    renderdata->glTexParameteri(data->type, GL_TEXTURE_MIN_FILTER, scaleMode);
    renderdata->glTexParameteri(data->type, GL_TEXTURE_MAG_FILTER, scaleMode);
    renderdata->glTexParameteri(data->type, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    renderdata->glTexParameteri(data->type, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    renderdata->glTexImage2D(data->type, 0, internalFormat, texture_w,
                             texture_h, 0, format, type, NULL);
    renderdata->glDisable(GL_TEXTURE_2D);
    renderdata->drawstate.texture = texture;
    renderdata->drawstate.texturing = SDL_FALSE;

    result = renderdata->glGetError();
    if (result != GL_NO_ERROR) {
        if (texture->access == SDL_TEXTUREACCESS_STREAMING) {
            SDL_free(data->pixels);
        }
        SDL_free(data);
        return GLES_SetError("glTexImage2D()", result);
    }

    texture->driverdata = data;
    return 0;
}

static int GLES_UpdateTexture(SDL_Renderer *renderer, SDL_Texture *texture,
                              const SDL_Rect *rect, const void *pixels, int pitch)
{
    GLES_RenderData *renderdata = (GLES_RenderData *)renderer->driverdata;
    GLES_TextureData *data = (GLES_TextureData *)texture->driverdata;
    Uint8 *blob = NULL;
    Uint8 *src;
    int srcPitch;
    int y;

    GLES_ActivateRenderer(renderer);

    /* Bail out if we're supposed to update an empty rectangle */
    if (rect->w <= 0 || rect->h <= 0) {
        return 0;
    }

    /* Reformat the texture data into a tightly packed array */
    srcPitch = rect->w * SDL_BYTESPERPIXEL(texture->format);
    src = (Uint8 *)pixels;
    if (pitch != srcPitch) {
        blob = (Uint8 *)SDL_malloc(srcPitch * rect->h);
        if (!blob) {
            return SDL_OutOfMemory();
        }
        src = blob;
        for (y = 0; y < rect->h; ++y) {
            SDL_memcpy(src, pixels, srcPitch);
            src += srcPitch;
            pixels = (Uint8 *)pixels + pitch;
        }
        src = blob;
    }

    /* Create a texture subimage with the supplied data */
    renderdata->glGetError();
    renderdata->glEnable(data->type);
    renderdata->glBindTexture(data->type, data->texture);
    renderdata->glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    renderdata->glTexSubImage2D(data->type,
                                0,
                                rect->x,
                                rect->y,
                                rect->w,
                                rect->h,
                                data->format,
                                data->formattype,
                                src);
    renderdata->glDisable(data->type);
    SDL_free(blob);

    renderdata->drawstate.texture = texture;
    renderdata->drawstate.texturing = SDL_FALSE;

    if (renderdata->glGetError() != GL_NO_ERROR) {
        return SDL_SetError("Failed to update texture");
    }
    return 0;
}

static int GLES_LockTexture(SDL_Renderer *renderer, SDL_Texture *texture,
                            const SDL_Rect *rect, void **pixels, int *pitch)
{
    GLES_TextureData *data = (GLES_TextureData *)texture->driverdata;

    *pixels =
        (void *)((Uint8 *)data->pixels + rect->y * data->pitch +
                 rect->x * SDL_BYTESPERPIXEL(texture->format));
    *pitch = data->pitch;
    return 0;
}

static void GLES_UnlockTexture(SDL_Renderer *renderer, SDL_Texture *texture)
{
    GLES_TextureData *data = (GLES_TextureData *)texture->driverdata;
    SDL_Rect rect;

    /* We do whole texture updates, at least for now */
    rect.x = 0;
    rect.y = 0;
    rect.w = texture->w;
    rect.h = texture->h;
    GLES_UpdateTexture(renderer, texture, &rect, data->pixels, data->pitch);
}

static void GLES_SetTextureScaleMode(SDL_Renderer *renderer, SDL_Texture *texture, SDL_ScaleMode scaleMode)
{
    GLES_RenderData *renderdata = (GLES_RenderData *)renderer->driverdata;
    GLES_TextureData *data = (GLES_TextureData *)texture->driverdata;
    GLenum glScaleMode = (scaleMode == SDL_ScaleModeNearest) ? GL_NEAREST : GL_LINEAR;

    renderdata->glBindTexture(data->type, data->texture);
    renderdata->glTexParameteri(data->type, GL_TEXTURE_MIN_FILTER, glScaleMode);
    renderdata->glTexParameteri(data->type, GL_TEXTURE_MAG_FILTER, glScaleMode);
}

static int GLES_SetRenderTarget(SDL_Renderer *renderer, SDL_Texture *texture)
{
    GLES_RenderData *data = (GLES_RenderData *)renderer->driverdata;
    GLES_TextureData *texturedata = NULL;
    GLenum status;

    if (!data->GL_OES_framebuffer_object_supported) {
        return SDL_SetError("Can't enable render target support in this renderer");
    }

    data->drawstate.viewport_dirty = SDL_TRUE;

    if (!texture) {
        data->glBindFramebufferOES(GL_FRAMEBUFFER_OES, data->window_framebuffer);
        return 0;
    }

    texturedata = (GLES_TextureData *)texture->driverdata;
    data->glBindFramebufferOES(GL_FRAMEBUFFER_OES, texturedata->fbo->FBO);
    /* TODO: check if texture pixel format allows this operation */
    data->glFramebufferTexture2DOES(GL_FRAMEBUFFER_OES, GL_COLOR_ATTACHMENT0_OES, texturedata->type, texturedata->texture, 0);
    /* Check FBO status */
    status = data->glCheckFramebufferStatusOES(GL_FRAMEBUFFER_OES);
    if (status != GL_FRAMEBUFFER_COMPLETE_OES) {
        return SDL_SetError("glFramebufferTexture2DOES() failed");
    }
    return 0;
}

static int GLES_QueueSetViewport(SDL_Renderer *renderer, SDL_RenderCommand *cmd)
{
    return 0; /* nothing to do in this backend. */
}

static int GLES_QueueDrawPoints(SDL_Renderer *renderer, SDL_RenderCommand *cmd, const SDL_FPoint *points, int count)
{
    GLfloat *verts = (GLfloat *)SDL_AllocateRenderVertices(renderer, count * 2 * sizeof(GLfloat), 0, &cmd->data.draw.first);
    int i;

    if (!verts) {
        return -1;
    }

    cmd->data.draw.count = count;
    for (i = 0; i < count; i++) {
        *(verts++) = 0.5f + points[i].x;
        *(verts++) = 0.5f + points[i].y;
    }

    return 0;
}

static int GLES_QueueDrawLines(SDL_Renderer *renderer, SDL_RenderCommand *cmd, const SDL_FPoint *points, int count)
{
    int i;
    GLfloat prevx, prevy;
    const size_t vertlen = (sizeof(GLfloat) * 2) * count;
    GLfloat *verts = (GLfloat *)SDL_AllocateRenderVertices(renderer, vertlen, 0, &cmd->data.draw.first);

    if (!verts) {
        return -1;
    }
    cmd->data.draw.count = count;

    /* 0.5f offset to hit the center of the pixel. */
    prevx = 0.5f + points->x;
    prevy = 0.5f + points->y;
    *(verts++) = prevx;
    *(verts++) = prevy;

    /* bump the end of each line segment out a quarter of a pixel, to provoke
       the diamond-exit rule. Without this, you won't just drop the last
       pixel of the last line segment, but you might also drop pixels at the
       edge of any given line segment along the way too. */
    for (i = 1; i < count; i++) {
        const GLfloat xstart = prevx;
        const GLfloat ystart = prevy;
        const GLfloat xend = points[i].x + 0.5f; /* 0.5f to hit pixel center. */
        const GLfloat yend = points[i].y + 0.5f;
        /* bump a little in the direction we are moving in. */
        const GLfloat deltax = xend - xstart;
        const GLfloat deltay = yend - ystart;
        const GLfloat angle = SDL_atan2f(deltay, deltax);
        prevx = xend + (SDL_cosf(angle) * 0.25f);
        prevy = yend + (SDL_sinf(angle) * 0.25f);
        *(verts++) = prevx;
        *(verts++) = prevy;
    }

    return 0;
}

static int GLES_QueueGeometry(SDL_Renderer *renderer, SDL_RenderCommand *cmd, SDL_Texture *texture,
                              const float *xy, int xy_stride, const SDL_Color *color, int color_stride, const float *uv, int uv_stride,
                              int num_vertices, const void *indices, int num_indices, int size_indices,
                              float scale_x, float scale_y)
{
    GLES_TextureData *texturedata = NULL;
    int i;
    int count = indices ? num_indices : num_vertices;
    GLfloat *verts;
    int sz = 2 + 4 + (texture ? 2 : 0);

    verts = (GLfloat *)SDL_AllocateRenderVertices(renderer, count * sz * sizeof(GLfloat), 0, &cmd->data.draw.first);
    if (!verts) {
        return -1;
    }

    if (texture) {
        texturedata = (GLES_TextureData *)texture->driverdata;
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

        xy_ = (float *)((char *)xy + j * xy_stride);
        col_ = *(SDL_Color *)((char *)color + j * color_stride);

        *(verts++) = xy_[0] * scale_x;
        *(verts++) = xy_[1] * scale_y;

        *(verts++) = col_.r * inv255f;
        *(verts++) = col_.g * inv255f;
        *(verts++) = col_.b * inv255f;
        *(verts++) = col_.a * inv255f;

        if (texture) {
            float *uv_ = (float *)((char *)uv + j * uv_stride);
            *(verts++) = uv_[0] * texturedata->texw;
            *(verts++) = uv_[1] * texturedata->texh;
        }
    }
    return 0;
}

static void SetDrawState(GLES_RenderData *data, const SDL_RenderCommand *cmd)
{
    const SDL_BlendMode blend = cmd->data.draw.blend;
    const Uint8 r = cmd->data.draw.r;
    const Uint8 g = cmd->data.draw.g;
    const Uint8 b = cmd->data.draw.b;
    const Uint8 a = cmd->data.draw.a;
    const Uint32 color = (((Uint32)a << 24) | (r << 16) | (g << 8) | b);

    if (color != data->drawstate.color) {
        const GLfloat fr = ((GLfloat)r) * inv255f;
        const GLfloat fg = ((GLfloat)g) * inv255f;
        const GLfloat fb = ((GLfloat)b) * inv255f;
        const GLfloat fa = ((GLfloat)a) * inv255f;
        data->glColor4f(fr, fg, fb, fa);
        data->drawstate.color = color;
    }

    if (data->drawstate.viewport_dirty) {
        const SDL_Rect *viewport = &data->drawstate.viewport;
        const SDL_bool istarget = (data->drawstate.target != NULL);
        data->glMatrixMode(GL_PROJECTION);
        data->glLoadIdentity();
        data->glViewport(viewport->x,
                         istarget ? viewport->y : (data->drawstate.drawableh - viewport->y - viewport->h),
                         viewport->w, viewport->h);
        if (viewport->w && viewport->h) {
            data->glOrthof((GLfloat)0, (GLfloat)viewport->w,
                           (GLfloat)(istarget ? 0 : viewport->h),
                           (GLfloat)(istarget ? viewport->h : 0),
                           0.0, 1.0);
        }
        data->glMatrixMode(GL_MODELVIEW);
        data->drawstate.viewport_dirty = SDL_FALSE;
    }

    if (data->drawstate.cliprect_enabled_dirty) {
        if (data->drawstate.cliprect_enabled) {
            data->glEnable(GL_SCISSOR_TEST);
        } else {
            data->glDisable(GL_SCISSOR_TEST);
        }
        data->drawstate.cliprect_enabled_dirty = SDL_FALSE;
    }

    if (data->drawstate.cliprect_enabled && data->drawstate.cliprect_dirty) {
        const SDL_Rect *viewport = &data->drawstate.viewport;
        const SDL_Rect *rect = &data->drawstate.cliprect;
        const SDL_bool istarget = (data->drawstate.target != NULL);
        data->glScissor(viewport->x + rect->x,
                        istarget ? viewport->y + rect->y : data->drawstate.drawableh - viewport->y - rect->y - rect->h,
                        rect->w, rect->h);
        data->drawstate.cliprect_dirty = SDL_FALSE;
    }

    if (blend != data->drawstate.blend) {
        if (blend == SDL_BLENDMODE_NONE) {
            data->glDisable(GL_BLEND);
        } else {
            data->glEnable(GL_BLEND);
            if (data->GL_OES_blend_func_separate_supported) {
                data->glBlendFuncSeparateOES(GetBlendFunc(SDL_GetBlendModeSrcColorFactor(blend)),
                                             GetBlendFunc(SDL_GetBlendModeDstColorFactor(blend)),
                                             GetBlendFunc(SDL_GetBlendModeSrcAlphaFactor(blend)),
                                             GetBlendFunc(SDL_GetBlendModeDstAlphaFactor(blend)));
            } else {
                data->glBlendFunc(GetBlendFunc(SDL_GetBlendModeSrcColorFactor(blend)),
                                  GetBlendFunc(SDL_GetBlendModeDstColorFactor(blend)));
            }
            if (data->GL_OES_blend_equation_separate_supported) {
                data->glBlendEquationSeparateOES(GetBlendEquation(SDL_GetBlendModeColorOperation(blend)),
                                                 GetBlendEquation(SDL_GetBlendModeAlphaOperation(blend)));
            } else if (data->GL_OES_blend_subtract_supported) {
                data->glBlendEquationOES(GetBlendEquation(SDL_GetBlendModeColorOperation(blend)));
            }
        }
        data->drawstate.blend = blend;
    }

    if ((cmd->data.draw.texture != NULL) != data->drawstate.texturing) {
        if (cmd->data.draw.texture == NULL) {
            data->glDisable(GL_TEXTURE_2D);
            data->glDisableClientState(GL_TEXTURE_COORD_ARRAY);
            data->drawstate.texturing = SDL_FALSE;
        } else {
            data->glEnable(GL_TEXTURE_2D);
            data->glEnableClientState(GL_TEXTURE_COORD_ARRAY);
            data->drawstate.texturing = SDL_TRUE;
        }
    }
}

static void SetCopyState(GLES_RenderData *data, const SDL_RenderCommand *cmd)
{
    SDL_Texture *texture = cmd->data.draw.texture;
    SetDrawState(data, cmd);

    if (texture != data->drawstate.texture) {
        GLES_TextureData *texturedata = (GLES_TextureData *)texture->driverdata;
        data->glBindTexture(GL_TEXTURE_2D, texturedata->texture);
        data->drawstate.texture = texture;
    }
}

static int GLES_RunCommandQueue(SDL_Renderer *renderer, SDL_RenderCommand *cmd, void *vertices, size_t vertsize)
{
    GLES_RenderData *data = (GLES_RenderData *)renderer->driverdata;

    if (GLES_ActivateRenderer(renderer) < 0) {
        return -1;
    }

    data->drawstate.target = renderer->target;

    if (!renderer->target) {
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
        case SDL_RENDERCMD_SETDRAWCOLOR:
        {
            break; /* not used in this render backend. */
        }

        case SDL_RENDERCMD_SETVIEWPORT:
        {
            SDL_Rect *viewport = &data->drawstate.viewport;
            if (SDL_memcmp(viewport, &cmd->data.viewport.rect, sizeof(cmd->data.viewport.rect)) != 0) {
                SDL_copyp(viewport, &cmd->data.viewport.rect);
                data->drawstate.viewport_dirty = SDL_TRUE;
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

        case SDL_RENDERCMD_CLEAR:
        {
            const Uint8 r = cmd->data.color.r;
            const Uint8 g = cmd->data.color.g;
            const Uint8 b = cmd->data.color.b;
            const Uint8 a = cmd->data.color.a;
            const Uint32 color = (((Uint32)a << 24) | (r << 16) | (g << 8) | b);
            if (color != data->drawstate.clear_color) {
                const GLfloat fr = ((GLfloat)r) * inv255f;
                const GLfloat fg = ((GLfloat)g) * inv255f;
                const GLfloat fb = ((GLfloat)b) * inv255f;
                const GLfloat fa = ((GLfloat)a) * inv255f;
                data->glClearColor(fr, fg, fb, fa);
                data->drawstate.clear_color = color;
            }

            if (data->drawstate.cliprect_enabled || data->drawstate.cliprect_enabled_dirty) {
                data->glDisable(GL_SCISSOR_TEST);
                data->drawstate.cliprect_enabled_dirty = data->drawstate.cliprect_enabled;
            }

            data->glClear(GL_COLOR_BUFFER_BIT);

            break;
        }

        case SDL_RENDERCMD_DRAW_POINTS:
        {
            const size_t count = cmd->data.draw.count;
            const GLfloat *verts = (GLfloat *)(((Uint8 *)vertices) + cmd->data.draw.first);
            SetDrawState(data, cmd);
            data->glVertexPointer(2, GL_FLOAT, 0, verts);
            data->glDrawArrays(GL_POINTS, 0, (GLsizei)count);
            break;
        }

        case SDL_RENDERCMD_DRAW_LINES:
        {
            const GLfloat *verts = (GLfloat *)(((Uint8 *)vertices) + cmd->data.draw.first);
            const size_t count = cmd->data.draw.count;
            SDL_assert(count >= 2);
            SetDrawState(data, cmd);
            data->glVertexPointer(2, GL_FLOAT, 0, verts);
            data->glDrawArrays(GL_LINE_STRIP, 0, (GLsizei)count);
            break;
        }

        case SDL_RENDERCMD_FILL_RECTS: /* unused */
            break;

        case SDL_RENDERCMD_COPY: /* unused */
            break;

        case SDL_RENDERCMD_COPY_EX: /* unused */
            break;

        case SDL_RENDERCMD_GEOMETRY:
        {
            const GLfloat *verts = (GLfloat *)(((Uint8 *)vertices) + cmd->data.draw.first);
            SDL_Texture *texture = cmd->data.draw.texture;
            const size_t count = cmd->data.draw.count;
            int stride = (2 + 4 + (texture ? 2 : 0)) * sizeof(float);

            if (texture) {
                SetCopyState(data, cmd);
            } else {
                SetDrawState(data, cmd);
            }

            data->glEnableClientState(GL_COLOR_ARRAY);

            data->glVertexPointer(2, GL_FLOAT, stride, verts);
            data->glColorPointer(4, GL_FLOAT, stride, verts + 2);
            if (texture) {
                data->glTexCoordPointer(2, GL_FLOAT, stride, verts + 2 + 4);
            }

            data->glDrawArrays(GL_TRIANGLES, 0, (GLsizei)count);

            data->glDisableClientState(GL_COLOR_ARRAY);
            break;
        }

        case SDL_RENDERCMD_NO_OP:
            break;
        }

        cmd = cmd->next;
    }

    return 0;
}

static int GLES_RenderReadPixels(SDL_Renderer *renderer, const SDL_Rect *rect,
                                 Uint32 pixel_format, void *pixels, int pitch)
{
    GLES_RenderData *data = (GLES_RenderData *)renderer->driverdata;
    Uint32 temp_format = renderer->target ? renderer->target->format : SDL_PIXELFORMAT_RGBA32;
    void *temp_pixels;
    int temp_pitch;
    Uint8 *src, *dst, *tmp;
    int w, h, length, rows;
    int status;

    GLES_ActivateRenderer(renderer);

    temp_pitch = rect->w * SDL_BYTESPERPIXEL(temp_format);
    temp_pixels = SDL_malloc(rect->h * temp_pitch);
    if (!temp_pixels) {
        return SDL_OutOfMemory();
    }

    SDL_GetRendererOutputSize(renderer, &w, &h);

    data->glPixelStorei(GL_PACK_ALIGNMENT, 1);

    data->glReadPixels(rect->x, renderer->target ? rect->y : (h - rect->y) - rect->h,
                       rect->w, rect->h, GL_RGBA, GL_UNSIGNED_BYTE, temp_pixels);

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

static int GLES_RenderPresent(SDL_Renderer *renderer)
{
    GLES_ActivateRenderer(renderer);

    return SDL_GL_SwapWindowWithResult(renderer->window);
}

static void GLES_DestroyTexture(SDL_Renderer *renderer, SDL_Texture *texture)
{
    GLES_RenderData *renderdata = (GLES_RenderData *)renderer->driverdata;

    GLES_TextureData *data = (GLES_TextureData *)texture->driverdata;

    GLES_ActivateRenderer(renderer);

    if (renderdata->drawstate.texture == texture) {
        renderdata->drawstate.texture = NULL;
    }
    if (renderdata->drawstate.target == texture) {
        renderdata->drawstate.target = NULL;
    }

    if (!data) {
        return;
    }
    if (data->texture) {
        renderdata->glDeleteTextures(1, &data->texture);
    }
    SDL_free(data->pixels);
    SDL_free(data);
    texture->driverdata = NULL;
}

static void GLES_DestroyRenderer(SDL_Renderer *renderer)
{
    GLES_RenderData *data = (GLES_RenderData *)renderer->driverdata;

    if (data) {
        if (data->context) {
            while (data->framebuffers) {
                GLES_FBOList *nextnode = data->framebuffers->next;
                data->glDeleteFramebuffersOES(1, &data->framebuffers->FBO);
                SDL_free(data->framebuffers);
                data->framebuffers = nextnode;
            }
            SDL_GL_DeleteContext(data->context);
        }
        SDL_free(data);
    }
    SDL_free(renderer);
}

static int GLES_BindTexture(SDL_Renderer *renderer, SDL_Texture *texture, float *texw, float *texh)
{
    GLES_RenderData *data = (GLES_RenderData *)renderer->driverdata;
    GLES_TextureData *texturedata = (GLES_TextureData *)texture->driverdata;
    GLES_ActivateRenderer(renderer);

    data->glEnable(GL_TEXTURE_2D);
    data->glBindTexture(texturedata->type, texturedata->texture);

    data->drawstate.texture = texture;
    data->drawstate.texturing = SDL_TRUE;

    if (texw) {
        *texw = (float)texturedata->texw;
    }
    if (texh) {
        *texh = (float)texturedata->texh;
    }

    return 0;
}

static int GLES_UnbindTexture(SDL_Renderer *renderer, SDL_Texture *texture)
{
    GLES_RenderData *data = (GLES_RenderData *)renderer->driverdata;
    GLES_TextureData *texturedata = (GLES_TextureData *)texture->driverdata;
    GLES_ActivateRenderer(renderer);
    data->glDisable(texturedata->type);

    data->drawstate.texture = NULL;
    data->drawstate.texturing = SDL_FALSE;

    return 0;
}

static int GLES_SetVSync(SDL_Renderer *renderer, const int vsync)
{
    int retval;
    if (vsync) {
        retval = SDL_GL_SetSwapInterval(1);
    } else {
        retval = SDL_GL_SetSwapInterval(0);
    }
    if (retval != 0) {
        return retval;
    }
    if (SDL_GL_GetSwapInterval() != 0) {
        renderer->info.flags |= SDL_RENDERER_PRESENTVSYNC;
    } else {
        renderer->info.flags &= ~SDL_RENDERER_PRESENTVSYNC;
    }
    return retval;
}

static SDL_Renderer *GLES_CreateRenderer(SDL_Window *window, Uint32 flags)
{
    SDL_Renderer *renderer;
    GLES_RenderData *data;
    GLint value;
    Uint32 window_flags;
    int profile_mask = 0, major = 0, minor = 0;
    SDL_bool changed_window = SDL_FALSE;

    SDL_GL_GetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, &profile_mask);
    SDL_GL_GetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, &major);
    SDL_GL_GetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, &minor);

    window_flags = SDL_GetWindowFlags(window);
    if (!(window_flags & SDL_WINDOW_OPENGL) ||
        profile_mask != SDL_GL_CONTEXT_PROFILE_ES || major != RENDERER_CONTEXT_MAJOR || minor != RENDERER_CONTEXT_MINOR) {

        changed_window = SDL_TRUE;
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, RENDERER_CONTEXT_MAJOR);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, RENDERER_CONTEXT_MINOR);

        if (SDL_RecreateWindow(window, (window_flags & ~(SDL_WINDOW_VULKAN | SDL_WINDOW_METAL)) | SDL_WINDOW_OPENGL) < 0) {
            goto error;
        }
    }

    renderer = (SDL_Renderer *)SDL_calloc(1, sizeof(*renderer));
    if (!renderer) {
        SDL_OutOfMemory();
        goto error;
    }

    data = (GLES_RenderData *)SDL_calloc(1, sizeof(*data));
    if (!data) {
        GLES_DestroyRenderer(renderer);
        SDL_OutOfMemory();
        goto error;
    }

    renderer->WindowEvent = GLES_WindowEvent;
    renderer->GetOutputSize = GLES_GetOutputSize;
    renderer->SupportsBlendMode = GLES_SupportsBlendMode;
    renderer->CreateTexture = GLES_CreateTexture;
    renderer->UpdateTexture = GLES_UpdateTexture;
    renderer->LockTexture = GLES_LockTexture;
    renderer->UnlockTexture = GLES_UnlockTexture;
    renderer->SetTextureScaleMode = GLES_SetTextureScaleMode;
    renderer->SetRenderTarget = GLES_SetRenderTarget;
    renderer->QueueSetViewport = GLES_QueueSetViewport;
    renderer->QueueSetDrawColor = GLES_QueueSetViewport; /* SetViewport and SetDrawColor are (currently) no-ops. */
    renderer->QueueDrawPoints = GLES_QueueDrawPoints;
    renderer->QueueDrawLines = GLES_QueueDrawLines;
    renderer->QueueGeometry = GLES_QueueGeometry;
    renderer->RunCommandQueue = GLES_RunCommandQueue;
    renderer->RenderReadPixels = GLES_RenderReadPixels;
    renderer->RenderPresent = GLES_RenderPresent;
    renderer->DestroyTexture = GLES_DestroyTexture;
    renderer->DestroyRenderer = GLES_DestroyRenderer;
    renderer->SetVSync = GLES_SetVSync;
    renderer->GL_BindTexture = GLES_BindTexture;
    renderer->GL_UnbindTexture = GLES_UnbindTexture;
    renderer->info = GLES_RenderDriver.info;
    renderer->info.flags = SDL_RENDERER_ACCELERATED;
    renderer->driverdata = data;
    renderer->window = window;

    data->context = SDL_GL_CreateContext(window);
    if (!data->context) {
        GLES_DestroyRenderer(renderer);
        goto error;
    }
    if (SDL_GL_MakeCurrent(window, data->context) < 0) {
        GLES_DestroyRenderer(renderer);
        goto error;
    }

    if (GLES_LoadFunctions(data) < 0) {
        GLES_DestroyRenderer(renderer);
        goto error;
    }

    if (flags & SDL_RENDERER_PRESENTVSYNC) {
        SDL_GL_SetSwapInterval(1);
    } else {
        SDL_GL_SetSwapInterval(0);
    }
    if (SDL_GL_GetSwapInterval() != 0) {
        renderer->info.flags |= SDL_RENDERER_PRESENTVSYNC;
    }

    value = 0;
    data->glGetIntegerv(GL_MAX_TEXTURE_SIZE, &value);
    renderer->info.max_texture_width = value;
    value = 0;
    data->glGetIntegerv(GL_MAX_TEXTURE_SIZE, &value);
    renderer->info.max_texture_height = value;

    /* Android does not report GL_OES_framebuffer_object but the functionality seems to be there anyway */
    if (SDL_GL_ExtensionSupported("GL_OES_framebuffer_object") || data->glGenFramebuffersOES) {
        data->GL_OES_framebuffer_object_supported = SDL_TRUE;
        renderer->info.flags |= SDL_RENDERER_TARGETTEXTURE;

        value = 0;
        data->glGetIntegerv(GL_FRAMEBUFFER_BINDING_OES, &value);
        data->window_framebuffer = (GLuint)value;
    }
    data->framebuffers = NULL;

    if (SDL_GL_ExtensionSupported("GL_OES_blend_func_separate")) {
        data->GL_OES_blend_func_separate_supported = SDL_TRUE;
    }
    if (SDL_GL_ExtensionSupported("GL_OES_blend_equation_separate")) {
        data->GL_OES_blend_equation_separate_supported = SDL_TRUE;
    }
    if (SDL_GL_ExtensionSupported("GL_OES_blend_subtract")) {
        data->GL_OES_blend_subtract_supported = SDL_TRUE;
    }
    if (SDL_GL_ExtensionSupported("GL_EXT_blend_minmax")) {
        data->GL_EXT_blend_minmax_supported = SDL_TRUE;
    }

    /* Set up parameters for rendering */
    data->glDisable(GL_DEPTH_TEST);
    data->glDisable(GL_CULL_FACE);

    data->glMatrixMode(GL_MODELVIEW);
    data->glLoadIdentity();

    data->glEnableClientState(GL_VERTEX_ARRAY);
    data->glDisableClientState(GL_TEXTURE_COORD_ARRAY);

    data->glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    data->drawstate.blend = SDL_BLENDMODE_INVALID;
    data->drawstate.color = 0xFFFFFFFF;
    data->drawstate.clear_color = 0xFFFFFFFF;

    return renderer;

error:
    if (changed_window) {
        /* Uh oh, better try to put it back... */
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, profile_mask);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, major);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, minor);
        SDL_RecreateWindow(window, window_flags);
    }
    return NULL;
}

SDL_RenderDriver GLES_RenderDriver = {
    GLES_CreateRenderer,
    { "opengles",
      (SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC),
      1,
      { SDL_PIXELFORMAT_RGBA32 },
      0,
      0 }
};

#endif /* SDL_VIDEO_RENDER_OGL_ES */

/* vi: set ts=4 sw=4 expandtab: */
