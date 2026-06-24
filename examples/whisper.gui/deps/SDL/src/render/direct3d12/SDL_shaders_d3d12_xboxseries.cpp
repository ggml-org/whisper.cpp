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

#if SDL_VIDEO_RENDER_D3D12 && defined(__XBOXSERIES__)

#include <SDL_stdinc.h>

#include "../../core/windows/SDL_windows.h"
#include <d3d12_xs.h>

#include "SDL_shaders_d3d12.h"

#define SDL_COMPOSE_ERROR(str) SDL_STRINGIFY_ARG(__FUNCTION__) ", " str

/* Shader blob headers are generated with a pre-build step using buildshaders.bat */
#include "../VisualC-GDK/shaders/D3D12_PixelShader_Colors_Series.h"
#include "../VisualC-GDK/shaders/D3D12_PixelShader_Textures_Series.h"
#include "../VisualC-GDK/shaders/D3D12_PixelShader_NV12_BT601_Series.h"
#include "../VisualC-GDK/shaders/D3D12_PixelShader_NV12_BT709_Series.h"
#include "../VisualC-GDK/shaders/D3D12_PixelShader_NV12_JPEG_Series.h"
#include "../VisualC-GDK/shaders/D3D12_PixelShader_NV21_BT601_Series.h"
#include "../VisualC-GDK/shaders/D3D12_PixelShader_NV21_BT709_Series.h"
#include "../VisualC-GDK/shaders/D3D12_PixelShader_NV21_JPEG_Series.h"
#include "../VisualC-GDK/shaders/D3D12_PixelShader_YUV_BT601_Series.h"
#include "../VisualC-GDK/shaders/D3D12_PixelShader_YUV_BT709_Series.h"
#include "../VisualC-GDK/shaders/D3D12_PixelShader_YUV_JPEG_Series.h"

#include "../VisualC-GDK/shaders/D3D12_VertexShader_Color_Series.h"
#include "../VisualC-GDK/shaders/D3D12_VertexShader_Texture_Series.h"
#include "../VisualC-GDK/shaders/D3D12_VertexShader_NV_Series.h"
#include "../VisualC-GDK/shaders/D3D12_VertexShader_YUV_Series.h"

#include "../VisualC-GDK/shaders/D3D12_RootSig_Color_Series.h"
#include "../VisualC-GDK/shaders/D3D12_RootSig_Texture_Series.h"
#include "../VisualC-GDK/shaders/D3D12_RootSig_YUV_Series.h"
#include "../VisualC-GDK/shaders/D3D12_RootSig_NV_Series.h"

static struct
{
    const void *ps_shader_data;
    SIZE_T ps_shader_size;
    const void *vs_shader_data;
    SIZE_T vs_shader_size;
    D3D12_RootSignature root_sig;
} D3D12_shaders[NUM_SHADERS] = {
    { D3D12_PixelShader_Colors, sizeof(D3D12_PixelShader_Colors),
      D3D12_VertexShader_Color, sizeof(D3D12_VertexShader_Color),
      ROOTSIG_COLOR },
    { D3D12_PixelShader_Textures, sizeof(D3D12_PixelShader_Textures),
      D3D12_VertexShader_Texture, sizeof(D3D12_VertexShader_Texture),
      ROOTSIG_TEXTURE },
#if SDL_HAVE_YUV
    { D3D12_PixelShader_YUV_JPEG, sizeof(D3D12_PixelShader_YUV_JPEG),
      D3D12_VertexShader_YUV, sizeof(D3D12_VertexShader_YUV),
      ROOTSIG_YUV },
    { D3D12_PixelShader_YUV_BT601, sizeof(D3D12_PixelShader_YUV_BT601),
      D3D12_VertexShader_YUV, sizeof(D3D12_VertexShader_YUV),
      ROOTSIG_YUV },
    { D3D12_PixelShader_YUV_BT709, sizeof(D3D12_PixelShader_YUV_BT709),
      D3D12_VertexShader_YUV, sizeof(D3D12_VertexShader_YUV),
      ROOTSIG_YUV },
    { D3D12_PixelShader_NV12_JPEG, sizeof(D3D12_PixelShader_NV12_JPEG),
      D3D12_VertexShader_NV, sizeof(D3D12_VertexShader_NV),
      ROOTSIG_NV },
    { D3D12_PixelShader_NV12_BT601, sizeof(D3D12_PixelShader_NV12_BT601),
      D3D12_VertexShader_NV, sizeof(D3D12_VertexShader_NV),
      ROOTSIG_NV },
    { D3D12_PixelShader_NV12_BT709, sizeof(D3D12_PixelShader_NV12_BT709),
      D3D12_VertexShader_NV, sizeof(D3D12_VertexShader_NV),
      ROOTSIG_NV },
    { D3D12_PixelShader_NV21_JPEG, sizeof(D3D12_PixelShader_NV21_JPEG),
      D3D12_VertexShader_NV, sizeof(D3D12_VertexShader_NV),
      ROOTSIG_NV },
    { D3D12_PixelShader_NV21_BT601, sizeof(D3D12_PixelShader_NV21_BT601),
      D3D12_VertexShader_NV, sizeof(D3D12_VertexShader_NV),
      ROOTSIG_NV },
    { D3D12_PixelShader_NV21_BT709, sizeof(D3D12_PixelShader_NV21_BT709),
      D3D12_VertexShader_NV, sizeof(D3D12_VertexShader_NV),
      ROOTSIG_NV },
#endif
};

static struct
{
    const void *rs_shader_data;
    SIZE_T rs_shader_size;
} D3D12_rootsigs[NUM_ROOTSIGS] = {
    { D3D12_RootSig_Color, sizeof(D3D12_RootSig_Color) },
    { D3D12_RootSig_Texture, sizeof(D3D12_RootSig_Texture) },
#if SDL_HAVE_YUV
    { D3D12_RootSig_YUV, sizeof(D3D12_RootSig_YUV) },
    { D3D12_RootSig_NV, sizeof(D3D12_RootSig_NV) },
#endif
};

extern "C" void
D3D12_GetVertexShader(D3D12_Shader shader, D3D12_SHADER_BYTECODE *outBytecode)
{
    outBytecode->pShaderBytecode = D3D12_shaders[shader].vs_shader_data;
    outBytecode->BytecodeLength = D3D12_shaders[shader].vs_shader_size;
}

extern "C" void
D3D12_GetPixelShader(D3D12_Shader shader, D3D12_SHADER_BYTECODE *outBytecode)
{
    outBytecode->pShaderBytecode = D3D12_shaders[shader].ps_shader_data;
    outBytecode->BytecodeLength = D3D12_shaders[shader].ps_shader_size;
}

extern "C" D3D12_RootSignature
D3D12_GetRootSignatureType(D3D12_Shader shader)
{
    return D3D12_shaders[shader].root_sig;
}

extern "C" void
D3D12_GetRootSignatureData(D3D12_RootSignature rootSig, D3D12_SHADER_BYTECODE *outBytecode)
{
    outBytecode->pShaderBytecode = D3D12_rootsigs[rootSig].rs_shader_data;
    outBytecode->BytecodeLength = D3D12_rootsigs[rootSig].rs_shader_size;
}

#endif /* SDL_VIDEO_RENDER_D3D12 && defined(__XBOXSERIES__) */

/* vi: set ts=4 sw=4 expandtab: */
