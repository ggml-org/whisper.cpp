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

#if SDL_VIDEO_RENDER_OGL_ES2

#include "SDL_hints.h"
#include "SDL_video.h"
#include "SDL_opengles2.h"
#include "SDL_shaders_gles2.h"
#include "SDL_stdinc.h"

/* *INDENT-OFF* */ /* clang-format off */

/*************************************************************************************************
 * Vertex/fragment shader source                                                                 *
 *************************************************************************************************/

static const char GLES2_Fragment_Include_Best_Texture_Precision[] = \
"#ifdef GL_FRAGMENT_PRECISION_HIGH\n"                           \
"#define SDL_TEXCOORD_PRECISION highp\n"                        \
"#else\n"                                                       \
"#define SDL_TEXCOORD_PRECISION mediump\n"                      \
"#endif\n"                                                      \
"\n"                                                            \
"precision mediump float;\n"                                    \
"\n"                                                            \
;

static const char GLES2_Fragment_Include_Medium_Texture_Precision[] = \
"#define SDL_TEXCOORD_PRECISION mediump\n"                      \
"precision mediump float;\n"                                    \
"\n"                                                            \
;

static const char GLES2_Fragment_Include_High_Texture_Precision[] = \
"#define SDL_TEXCOORD_PRECISION highp\n"                        \
"precision mediump float;\n"                                    \
"\n"                                                            \
;

static const char GLES2_Fragment_Include_Undef_Precision[] =    \
"#define mediump\n"                                             \
"#define highp\n"                                               \
"#define lowp\n"                                                \
"#define SDL_TEXCOORD_PRECISION\n"                              \
"\n"                                                            \
;

static const char GLES2_Vertex_Default[] =                      \
"uniform mat4 u_projection;\n"                                  \
"attribute vec2 a_position;\n"                                  \
"attribute vec4 a_color;\n"                                     \
"attribute vec2 a_texCoord;\n"                                  \
"varying vec2 v_texCoord;\n"                                    \
"varying vec4 v_color;\n"                                       \
"\n"                                                            \
"void main()\n"                                                 \
"{\n"                                                           \
"    v_texCoord = a_texCoord;\n"                                \
"    gl_Position = u_projection * vec4(a_position, 0.0, 1.0);\n" \
"    gl_PointSize = 1.0;\n"                                     \
"    v_color = a_color;\n"                                      \
"}\n"                                                           \
;

static const char GLES2_Fragment_Solid[] =                      \
"varying mediump vec4 v_color;\n"                               \
"\n"                                                            \
"void main()\n"                                                 \
"{\n"                                                           \
"    gl_FragColor = v_color;\n"                                 \
"}\n"                                                           \
;

static const char GLES2_Fragment_TextureABGR[] =                \
"uniform sampler2D u_texture;\n"                                \
"varying mediump vec4 v_color;\n"                               \
"varying SDL_TEXCOORD_PRECISION vec2 v_texCoord;\n"             \
"\n"                                                            \
"void main()\n"                                                 \
"{\n"                                                           \
"    gl_FragColor = texture2D(u_texture, v_texCoord);\n"        \
"    gl_FragColor *= v_color;\n"                                \
"}\n"                                                           \
;

/* ARGB to ABGR conversion */
static const char GLES2_Fragment_TextureARGB[] =                \
"uniform sampler2D u_texture;\n"                                \
"varying mediump vec4 v_color;\n"                               \
"varying SDL_TEXCOORD_PRECISION vec2 v_texCoord;\n"             \
"\n"                                                            \
"void main()\n"                                                 \
"{\n"                                                           \
"    mediump vec4 abgr = texture2D(u_texture, v_texCoord);\n"   \
"    gl_FragColor = abgr;\n"                                    \
"    gl_FragColor.r = abgr.b;\n"                                \
"    gl_FragColor.b = abgr.r;\n"                                \
"    gl_FragColor *= v_color;\n"                                \
"}\n"                                                           \
;

/* RGB to ABGR conversion */
static const char GLES2_Fragment_TextureRGB[] =                 \
"uniform sampler2D u_texture;\n"                                \
"varying mediump vec4 v_color;\n"                               \
"varying SDL_TEXCOORD_PRECISION vec2 v_texCoord;\n"             \
"\n"                                                            \
"void main()\n"                                                 \
"{\n"                                                           \
"    mediump vec4 abgr = texture2D(u_texture, v_texCoord);\n"   \
"    gl_FragColor = abgr;\n"                                    \
"    gl_FragColor.r = abgr.b;\n"                                \
"    gl_FragColor.b = abgr.r;\n"                                \
"    gl_FragColor.a = 1.0;\n"                                   \
"    gl_FragColor *= v_color;\n"                                \
"}\n"                                                           \
;

/* BGR to ABGR conversion */
static const char GLES2_Fragment_TextureBGR[] =                 \
"uniform sampler2D u_texture;\n"                                \
"varying mediump vec4 v_color;\n"                               \
"varying SDL_TEXCOORD_PRECISION vec2 v_texCoord;\n"             \
"\n"                                                            \
"void main()\n"                                                 \
"{\n"                                                           \
"    mediump vec4 abgr = texture2D(u_texture, v_texCoord);\n"   \
"    gl_FragColor = abgr;\n"                                    \
"    gl_FragColor.a = 1.0;\n"                                   \
"    gl_FragColor *= v_color;\n"                                \
"}\n"                                                           \
;

#if SDL_HAVE_YUV

#define JPEG_SHADER_CONSTANTS                                   \
"// YUV offset \n"                                              \
"const vec3 offset = vec3(0, -0.501960814, -0.501960814);\n"    \
"\n"                                                            \
"// RGB coefficients \n"                                        \
"const mat3 matrix = mat3( 1,       1,        1,\n"             \
"                          0,      -0.3441,   1.772,\n"         \
"                          1.402,  -0.7141,   0);\n"            \
"\n"                                                            \

#define BT601_SHADER_CONSTANTS                                  \
"// YUV offset \n"                                              \
"const vec3 offset = vec3(-0.0627451017, -0.501960814, -0.501960814);\n" \
"\n"                                                            \
"// RGB coefficients \n"                                        \
"const mat3 matrix = mat3( 1.1644,  1.1644,   1.1644,\n"        \
"                          0,      -0.3918,   2.0172,\n"        \
"                          1.596,  -0.813,    0);\n"            \
"\n"                                                            \

#define BT709_SHADER_CONSTANTS                                  \
"// YUV offset \n"                                              \
"const vec3 offset = vec3(-0.0627451017, -0.501960814, -0.501960814);\n" \
"\n"                                                            \
"// RGB coefficients \n"                                        \
"const mat3 matrix = mat3( 1.1644,  1.1644,   1.1644,\n"        \
"                          0,      -0.2132,   2.1124,\n"        \
"                          1.7927, -0.5329,   0);\n"            \
"\n"                                                            \


#define YUV_SHADER_PROLOGUE                                     \
"uniform sampler2D u_texture;\n"                                \
"uniform sampler2D u_texture_u;\n"                              \
"uniform sampler2D u_texture_v;\n"                              \
"varying mediump vec4 v_color;\n"                               \
"varying SDL_TEXCOORD_PRECISION vec2 v_texCoord;\n"             \
"\n"                                                            \

#define YUV_SHADER_BODY                                         \
"void main()\n"                                                 \
"{\n"                                                           \
"    mediump vec3 yuv;\n"                                       \
"    lowp vec3 rgb;\n"                                          \
"\n"                                                            \
"    // Get the YUV values \n"                                  \
"    yuv.x = texture2D(u_texture,   v_texCoord).r;\n"           \
"    yuv.y = texture2D(u_texture_u, v_texCoord).r;\n"           \
"    yuv.z = texture2D(u_texture_v, v_texCoord).r;\n"           \
"\n"                                                            \
"    // Do the color transform \n"                              \
"    yuv += offset;\n"                                          \
"    rgb = matrix * yuv;\n"                                     \
"\n"                                                            \
"    // That was easy. :) \n"                                   \
"    gl_FragColor = vec4(rgb, 1);\n"                            \
"    gl_FragColor *= v_color;\n"                                \
"}"                                                             \

#define NV12_RA_SHADER_BODY                                     \
"void main()\n"                                                 \
"{\n"                                                           \
"    mediump vec3 yuv;\n"                                       \
"    lowp vec3 rgb;\n"                                          \
"\n"                                                            \
"    // Get the YUV values \n"                                  \
"    yuv.x = texture2D(u_texture,   v_texCoord).r;\n"           \
"    yuv.yz = texture2D(u_texture_u, v_texCoord).ra;\n"         \
"\n"                                                            \
"    // Do the color transform \n"                              \
"    yuv += offset;\n"                                          \
"    rgb = matrix * yuv;\n"                                     \
"\n"                                                            \
"    // That was easy. :) \n"                                   \
"    gl_FragColor = vec4(rgb, 1);\n"                            \
"    gl_FragColor *= v_color;\n"                                \
"}"                                                             \

#define NV12_RG_SHADER_BODY                                     \
"void main()\n"                                                 \
"{\n"                                                           \
"    mediump vec3 yuv;\n"                                       \
"    lowp vec3 rgb;\n"                                          \
"\n"                                                            \
"    // Get the YUV values \n"                                  \
"    yuv.x = texture2D(u_texture,   v_texCoord).r;\n"           \
"    yuv.yz = texture2D(u_texture_u, v_texCoord).rg;\n"         \
"\n"                                                            \
"    // Do the color transform \n"                              \
"    yuv += offset;\n"                                          \
"    rgb = matrix * yuv;\n"                                     \
"\n"                                                            \
"    // That was easy. :) \n"                                   \
"    gl_FragColor = vec4(rgb, 1);\n"                            \
"    gl_FragColor *= v_color;\n"                                \
"}"                                                             \

#define NV21_SHADER_BODY                                        \
"void main()\n"                                                 \
"{\n"                                                           \
"    mediump vec3 yuv;\n"                                       \
"    lowp vec3 rgb;\n"                                          \
"\n"                                                            \
"    // Get the YUV values \n"                                  \
"    yuv.x = texture2D(u_texture,   v_texCoord).r;\n"           \
"    yuv.yz = texture2D(u_texture_u, v_texCoord).ar;\n"         \
"\n"                                                            \
"    // Do the color transform \n"                              \
"    yuv += offset;\n"                                          \
"    rgb = matrix * yuv;\n"                                     \
"\n"                                                            \
"    // That was easy. :) \n"                                   \
"    gl_FragColor = vec4(rgb, 1);\n"                            \
"    gl_FragColor *= v_color;\n"                                \
"}"                                                             \

/* YUV to ABGR conversion */
static const char GLES2_Fragment_TextureYUVJPEG[] = \
        YUV_SHADER_PROLOGUE \
        JPEG_SHADER_CONSTANTS \
        YUV_SHADER_BODY \
;
static const char GLES2_Fragment_TextureYUVBT601[] = \
        YUV_SHADER_PROLOGUE \
        BT601_SHADER_CONSTANTS \
        YUV_SHADER_BODY \
;
static const char GLES2_Fragment_TextureYUVBT709[] = \
        YUV_SHADER_PROLOGUE \
        BT709_SHADER_CONSTANTS \
        YUV_SHADER_BODY \
;

/* NV12 to ABGR conversion */
static const char GLES2_Fragment_TextureNV12JPEG[] = \
        YUV_SHADER_PROLOGUE \
        JPEG_SHADER_CONSTANTS \
        NV12_RA_SHADER_BODY \
;
static const char GLES2_Fragment_TextureNV12BT601_RA[] = \
        YUV_SHADER_PROLOGUE \
        BT601_SHADER_CONSTANTS \
        NV12_RA_SHADER_BODY \
;
static const char GLES2_Fragment_TextureNV12BT601_RG[] = \
        YUV_SHADER_PROLOGUE \
        BT601_SHADER_CONSTANTS \
        NV12_RG_SHADER_BODY \
;
static const char GLES2_Fragment_TextureNV12BT709_RA[] = \
        YUV_SHADER_PROLOGUE \
        BT709_SHADER_CONSTANTS \
        NV12_RA_SHADER_BODY \
;
static const char GLES2_Fragment_TextureNV12BT709_RG[] = \
        YUV_SHADER_PROLOGUE \
        BT709_SHADER_CONSTANTS \
        NV12_RG_SHADER_BODY \
;

/* NV21 to ABGR conversion */
static const char GLES2_Fragment_TextureNV21JPEG[] = \
        YUV_SHADER_PROLOGUE \
        JPEG_SHADER_CONSTANTS \
        NV21_SHADER_BODY \
;
static const char GLES2_Fragment_TextureNV21BT601[] = \
        YUV_SHADER_PROLOGUE \
        BT601_SHADER_CONSTANTS \
        NV21_SHADER_BODY \
;
static const char GLES2_Fragment_TextureNV21BT709[] = \
        YUV_SHADER_PROLOGUE \
        BT709_SHADER_CONSTANTS \
        NV21_SHADER_BODY \
;
#endif

/* Custom Android video format texture */
static const char GLES2_Fragment_TextureExternalOES_Prologue[] = \
"#extension GL_OES_EGL_image_external : require\n"              \
"\n"                                                            \
;
static const char GLES2_Fragment_TextureExternalOES[] =         \
"uniform samplerExternalOES u_texture;\n"                       \
"varying mediump vec4 v_color;\n"                               \
"varying SDL_TEXCOORD_PRECISION vec2 v_texCoord;\n"             \
"\n"                                                            \
"void main()\n"                                                 \
"{\n"                                                           \
"    gl_FragColor = texture2D(u_texture, v_texCoord);\n"        \
"    gl_FragColor *= v_color;\n"                                \
"}\n"                                                           \
;

/* *INDENT-ON* */ /* clang-format on */

/*************************************************************************************************
 * Shader selector                                                                               *
 *************************************************************************************************/

const char *GLES2_GetShaderPrologue(GLES2_ShaderType type)
{
    switch (type) {
    case GLES2_SHADER_FRAGMENT_TEXTURE_EXTERNAL_OES:
        return GLES2_Fragment_TextureExternalOES_Prologue;
    default:
        return "";
    }
}

const char *GLES2_GetShaderInclude(GLES2_ShaderIncludeType type)
{
    switch (type) {
    case GLES2_SHADER_FRAGMENT_INCLUDE_UNDEF_PRECISION:
        return GLES2_Fragment_Include_Undef_Precision;
    case GLES2_SHADER_FRAGMENT_INCLUDE_BEST_TEXCOORD_PRECISION:
        return GLES2_Fragment_Include_Best_Texture_Precision;
    case GLES2_SHADER_FRAGMENT_INCLUDE_MEDIUM_TEXCOORD_PRECISION:
        return GLES2_Fragment_Include_Medium_Texture_Precision;
    case GLES2_SHADER_FRAGMENT_INCLUDE_HIGH_TEXCOORD_PRECISION:
        return GLES2_Fragment_Include_High_Texture_Precision;
    default:
        return "";
    }
}

GLES2_ShaderIncludeType GLES2_GetTexCoordPrecisionEnumFromHint(void)
{
    const char *texcoord_hint = SDL_GetHint("SDL_RENDER_OPENGLES2_TEXCOORD_PRECISION");
    GLES2_ShaderIncludeType value = GLES2_SHADER_FRAGMENT_INCLUDE_BEST_TEXCOORD_PRECISION;
    if (texcoord_hint) {
        if (SDL_strcmp(texcoord_hint, "undefined") == 0) {
            return GLES2_SHADER_FRAGMENT_INCLUDE_UNDEF_PRECISION;
        }
        if (SDL_strcmp(texcoord_hint, "high") == 0) {
            return GLES2_SHADER_FRAGMENT_INCLUDE_HIGH_TEXCOORD_PRECISION;
        }
        if (SDL_strcmp(texcoord_hint, "medium") == 0) {
            return GLES2_SHADER_FRAGMENT_INCLUDE_MEDIUM_TEXCOORD_PRECISION;
        }
    }
    return value;
}

const char *GLES2_GetShader(GLES2_ShaderType type)
{
    switch (type) {
    case GLES2_SHADER_VERTEX_DEFAULT:
        return GLES2_Vertex_Default;
    case GLES2_SHADER_FRAGMENT_SOLID:
        return GLES2_Fragment_Solid;
    case GLES2_SHADER_FRAGMENT_TEXTURE_ABGR:
        return GLES2_Fragment_TextureABGR;
    case GLES2_SHADER_FRAGMENT_TEXTURE_ARGB:
        return GLES2_Fragment_TextureARGB;
    case GLES2_SHADER_FRAGMENT_TEXTURE_RGB:
        return GLES2_Fragment_TextureRGB;
    case GLES2_SHADER_FRAGMENT_TEXTURE_BGR:
        return GLES2_Fragment_TextureBGR;
#if SDL_HAVE_YUV
    case GLES2_SHADER_FRAGMENT_TEXTURE_YUV_JPEG:
        return GLES2_Fragment_TextureYUVJPEG;
    case GLES2_SHADER_FRAGMENT_TEXTURE_YUV_BT601:
        return GLES2_Fragment_TextureYUVBT601;
    case GLES2_SHADER_FRAGMENT_TEXTURE_YUV_BT709:
        return GLES2_Fragment_TextureYUVBT709;
    case GLES2_SHADER_FRAGMENT_TEXTURE_NV12_JPEG:
        return GLES2_Fragment_TextureNV12JPEG;
    case GLES2_SHADER_FRAGMENT_TEXTURE_NV12_RA_BT601:
        return GLES2_Fragment_TextureNV12BT601_RA;
    case GLES2_SHADER_FRAGMENT_TEXTURE_NV12_RG_BT601:
        return GLES2_Fragment_TextureNV12BT601_RG;
    case GLES2_SHADER_FRAGMENT_TEXTURE_NV12_RA_BT709:
        return GLES2_Fragment_TextureNV12BT709_RA;
    case GLES2_SHADER_FRAGMENT_TEXTURE_NV12_RG_BT709:
        return GLES2_Fragment_TextureNV12BT709_RG;
    case GLES2_SHADER_FRAGMENT_TEXTURE_NV21_JPEG:
        return GLES2_Fragment_TextureNV21JPEG;
    case GLES2_SHADER_FRAGMENT_TEXTURE_NV21_BT601:
        return GLES2_Fragment_TextureNV21BT601;
    case GLES2_SHADER_FRAGMENT_TEXTURE_NV21_BT709:
        return GLES2_Fragment_TextureNV21BT709;
#endif
    case GLES2_SHADER_FRAGMENT_TEXTURE_EXTERNAL_OES:
        return GLES2_Fragment_TextureExternalOES;
    default:
        return NULL;
    }
}

#endif /* SDL_VIDEO_RENDER_OGL_ES2 */

/* vi: set ts=4 sw=4 expandtab: */
