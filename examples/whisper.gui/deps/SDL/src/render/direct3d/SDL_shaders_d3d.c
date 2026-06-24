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

#include "SDL_render.h"
#include "SDL_system.h"

#if SDL_VIDEO_RENDER_D3D

#include "../../core/windows/SDL_windows.h"

#include <d3d9.h>

#include "SDL_shaders_d3d.h"

/* The shaders here were compiled with:

       fxc /T ps_2_0 /Fo"<OUTPUT FILE>" "<INPUT FILE>"

   Shader object code was converted to a list of DWORDs via the following
   *nix style command (available separately from Windows + MSVC):

     hexdump -v -e '6/4 "0x%08.8x, " "\n"' <FILE>
*/

/* --- D3D9_PixelShader_YUV_JPEG.hlsl ---
    Texture2D theTextureY : register(t0);
    Texture2D theTextureU : register(t1);
    Texture2D theTextureV : register(t2);
    SamplerState theSampler = sampler_state
    {
        addressU = Clamp;
        addressV = Clamp;
        mipfilter = NONE;
        minfilter = LINEAR;
        magfilter = LINEAR;
    };

    struct PixelShaderInput
    {
        float4 pos : SV_POSITION;
        float2 tex : TEXCOORD0;
        float4 color : COLOR0;
    };

    float4 main(PixelShaderInput input) : SV_TARGET
    {
        const float3 offset = {0.0, -0.501960814, -0.501960814};
        const float3 Rcoeff = {1.0000,  0.0000,  1.4020};
        const float3 Gcoeff = {1.0000, -0.3441, -0.7141};
        const float3 Bcoeff = {1.0000,  1.7720,  0.0000};

        float4 Output;

        float3 yuv;
        yuv.x = theTextureY.Sample(theSampler, input.tex).r;
        yuv.y = theTextureU.Sample(theSampler, input.tex).r;
        yuv.z = theTextureV.Sample(theSampler, input.tex).r;

        yuv += offset;
        Output.r = dot(yuv, Rcoeff);
        Output.g = dot(yuv, Gcoeff);
        Output.b = dot(yuv, Bcoeff);
        Output.a = 1.0f;

        return Output * input.color;
    }
*/
static const DWORD D3D9_PixelShader_YUV_JPEG[] = {
    0xffff0200, 0x0044fffe, 0x42415443, 0x0000001c, 0x000000d7, 0xffff0200,
    0x00000003, 0x0000001c, 0x00000100, 0x000000d0, 0x00000058, 0x00010003,
    0x00000001, 0x00000070, 0x00000000, 0x00000080, 0x00020003, 0x00000001,
    0x00000098, 0x00000000, 0x000000a8, 0x00000003, 0x00000001, 0x000000c0,
    0x00000000, 0x53656874, 0x6c706d61, 0x742b7265, 0x65546568, 0x72757478,
    0xab005565, 0x00070004, 0x00040001, 0x00000001, 0x00000000, 0x53656874,
    0x6c706d61, 0x742b7265, 0x65546568, 0x72757478, 0xab005665, 0x00070004,
    0x00040001, 0x00000001, 0x00000000, 0x53656874, 0x6c706d61, 0x742b7265,
    0x65546568, 0x72757478, 0xab005965, 0x00070004, 0x00040001, 0x00000001,
    0x00000000, 0x325f7370, 0x4d00305f, 0x6f726369, 0x74666f73, 0x29522820,
    0x534c4820, 0x6853204c, 0x72656461, 0x6d6f4320, 0x656c6970, 0x2e362072,
    0x36392e33, 0x312e3030, 0x34383336, 0xababab00, 0x05000051, 0xa00f0000,
    0x00000000, 0xbf008081, 0xbf008081, 0x3f800000, 0x05000051, 0xa00f0001,
    0x3f800000, 0x00000000, 0x3fb374bc, 0x00000000, 0x05000051, 0xa00f0002,
    0x3f800000, 0xbeb02de0, 0xbf36cf42, 0x00000000, 0x05000051, 0xa00f0003,
    0x3f800000, 0x3fe2d0e5, 0x00000000, 0x00000000, 0x0200001f, 0x80000000,
    0xb0030000, 0x0200001f, 0x80000000, 0x900f0000, 0x0200001f, 0x90000000,
    0xa00f0800, 0x0200001f, 0x90000000, 0xa00f0801, 0x0200001f, 0x90000000,
    0xa00f0802, 0x03000042, 0x800f0000, 0xb0e40000, 0xa0e40800, 0x03000042,
    0x800f0001, 0xb0e40000, 0xa0e40801, 0x03000042, 0x800f0002, 0xb0e40000,
    0xa0e40802, 0x02000001, 0x80020000, 0x80000001, 0x02000001, 0x80040000,
    0x80000002, 0x03000002, 0x80070000, 0x80e40000, 0xa0e40000, 0x03000008,
    0x80010001, 0x80e40000, 0xa0e40001, 0x03000008, 0x80020001, 0x80e40000,
    0xa0e40002, 0x0400005a, 0x80040001, 0x80e40000, 0xa0e40003, 0xa0aa0003,
    0x02000001, 0x80080001, 0xa0ff0000, 0x03000005, 0x800f0000, 0x80e40001,
    0x90e40000, 0x02000001, 0x800f0800, 0x80e40000, 0x0000ffff
};

/* --- D3D9_PixelShader_YUV_BT601.hlsl ---
    Texture2D theTextureY : register(t0);
    Texture2D theTextureU : register(t1);
    Texture2D theTextureV : register(t2);
    SamplerState theSampler = sampler_state
    {
        addressU = Clamp;
        addressV = Clamp;
        mipfilter = NONE;
        minfilter = LINEAR;
        magfilter = LINEAR;
    };

    struct PixelShaderInput
    {
        float4 pos : SV_POSITION;
        float2 tex : TEXCOORD0;
        float4 color : COLOR0;
    };

    float4 main(PixelShaderInput input) : SV_TARGET
    {
        const float3 offset = {-0.0627451017, -0.501960814, -0.501960814};
        const float3 Rcoeff = {1.1644,  0.0000,  1.5960};
        const float3 Gcoeff = {1.1644, -0.3918, -0.8130};
        const float3 Bcoeff = {1.1644,  2.0172,  0.0000};

        float4 Output;

        float3 yuv;
        yuv.x = theTextureY.Sample(theSampler, input.tex).r;
        yuv.y = theTextureU.Sample(theSampler, input.tex).r;
        yuv.z = theTextureV.Sample(theSampler, input.tex).r;

        yuv += offset;
        Output.r = dot(yuv, Rcoeff);
        Output.g = dot(yuv, Gcoeff);
        Output.b = dot(yuv, Bcoeff);
        Output.a = 1.0f;

        return Output * input.color;
    }
*/
static const DWORD D3D9_PixelShader_YUV_BT601[] = {
    0xffff0200, 0x0044fffe, 0x42415443, 0x0000001c, 0x000000d7, 0xffff0200,
    0x00000003, 0x0000001c, 0x00000100, 0x000000d0, 0x00000058, 0x00010003,
    0x00000001, 0x00000070, 0x00000000, 0x00000080, 0x00020003, 0x00000001,
    0x00000098, 0x00000000, 0x000000a8, 0x00000003, 0x00000001, 0x000000c0,
    0x00000000, 0x53656874, 0x6c706d61, 0x742b7265, 0x65546568, 0x72757478,
    0xab005565, 0x00070004, 0x00040001, 0x00000001, 0x00000000, 0x53656874,
    0x6c706d61, 0x742b7265, 0x65546568, 0x72757478, 0xab005665, 0x00070004,
    0x00040001, 0x00000001, 0x00000000, 0x53656874, 0x6c706d61, 0x742b7265,
    0x65546568, 0x72757478, 0xab005965, 0x00070004, 0x00040001, 0x00000001,
    0x00000000, 0x325f7370, 0x4d00305f, 0x6f726369, 0x74666f73, 0x29522820,
    0x534c4820, 0x6853204c, 0x72656461, 0x6d6f4320, 0x656c6970, 0x2e362072,
    0x36392e33, 0x312e3030, 0x34383336, 0xababab00, 0x05000051, 0xa00f0000,
    0xbd808081, 0xbf008081, 0xbf008081, 0x3f800000, 0x05000051, 0xa00f0001,
    0x3f950b0f, 0x00000000, 0x3fcc49ba, 0x00000000, 0x05000051, 0xa00f0002,
    0x3f950b0f, 0xbec89a02, 0xbf5020c5, 0x00000000, 0x05000051, 0xa00f0003,
    0x3f950b0f, 0x400119ce, 0x00000000, 0x00000000, 0x0200001f, 0x80000000,
    0xb0030000, 0x0200001f, 0x80000000, 0x900f0000, 0x0200001f, 0x90000000,
    0xa00f0800, 0x0200001f, 0x90000000, 0xa00f0801, 0x0200001f, 0x90000000,
    0xa00f0802, 0x03000042, 0x800f0000, 0xb0e40000, 0xa0e40800, 0x03000042,
    0x800f0001, 0xb0e40000, 0xa0e40801, 0x03000042, 0x800f0002, 0xb0e40000,
    0xa0e40802, 0x02000001, 0x80020000, 0x80000001, 0x02000001, 0x80040000,
    0x80000002, 0x03000002, 0x80070000, 0x80e40000, 0xa0e40000, 0x03000008,
    0x80010001, 0x80e40000, 0xa0e40001, 0x03000008, 0x80020001, 0x80e40000,
    0xa0e40002, 0x0400005a, 0x80040001, 0x80e40000, 0xa0e40003, 0xa0aa0003,
    0x02000001, 0x80080001, 0xa0ff0000, 0x03000005, 0x800f0000, 0x80e40001,
    0x90e40000, 0x02000001, 0x800f0800, 0x80e40000, 0x0000ffff
};

/* --- D3D9_PixelShader_YUV_BT709.hlsl ---
    Texture2D theTextureY : register(t0);
    Texture2D theTextureU : register(t1);
    Texture2D theTextureV : register(t2);
    SamplerState theSampler = sampler_state
    {
        addressU = Clamp;
        addressV = Clamp;
        mipfilter = NONE;
        minfilter = LINEAR;
        magfilter = LINEAR;
    };

    struct PixelShaderInput
    {
        float4 pos : SV_POSITION;
        float2 tex : TEXCOORD0;
        float4 color : COLOR0;
    };

    float4 main(PixelShaderInput input) : SV_TARGET
    {
        const float3 offset = {-0.0627451017, -0.501960814, -0.501960814};
        const float3 Rcoeff = {1.1644,  0.0000,  1.7927};
        const float3 Gcoeff = {1.1644, -0.2132, -0.5329};
        const float3 Bcoeff = {1.1644,  2.1124,  0.0000};

        float4 Output;

        float3 yuv;
        yuv.x = theTextureY.Sample(theSampler, input.tex).r;
        yuv.y = theTextureU.Sample(theSampler, input.tex).r;
        yuv.z = theTextureV.Sample(theSampler, input.tex).r;

        yuv += offset;
        Output.r = dot(yuv, Rcoeff);
        Output.g = dot(yuv, Gcoeff);
        Output.b = dot(yuv, Bcoeff);
        Output.a = 1.0f;

        return Output * input.color;
    }
*/
static const DWORD D3D9_PixelShader_YUV_BT709[] = {
    0xffff0200, 0x0044fffe, 0x42415443, 0x0000001c, 0x000000d7, 0xffff0200,
    0x00000003, 0x0000001c, 0x00000100, 0x000000d0, 0x00000058, 0x00010003,
    0x00000001, 0x00000070, 0x00000000, 0x00000080, 0x00020003, 0x00000001,
    0x00000098, 0x00000000, 0x000000a8, 0x00000003, 0x00000001, 0x000000c0,
    0x00000000, 0x53656874, 0x6c706d61, 0x742b7265, 0x65546568, 0x72757478,
    0xab005565, 0x00070004, 0x00040001, 0x00000001, 0x00000000, 0x53656874,
    0x6c706d61, 0x742b7265, 0x65546568, 0x72757478, 0xab005665, 0x00070004,
    0x00040001, 0x00000001, 0x00000000, 0x53656874, 0x6c706d61, 0x742b7265,
    0x65546568, 0x72757478, 0xab005965, 0x00070004, 0x00040001, 0x00000001,
    0x00000000, 0x325f7370, 0x4d00305f, 0x6f726369, 0x74666f73, 0x29522820,
    0x534c4820, 0x6853204c, 0x72656461, 0x6d6f4320, 0x656c6970, 0x2e362072,
    0x36392e33, 0x312e3030, 0x34383336, 0xababab00, 0x05000051, 0xa00f0000,
    0xbd808081, 0xbf008081, 0xbf008081, 0x3f800000, 0x05000051, 0xa00f0001,
    0x3f950b0f, 0x00000000, 0x3fe57732, 0x00000000, 0x05000051, 0xa00f0002,
    0x3f950b0f, 0xbe5a511a, 0xbf086c22, 0x00000000, 0x05000051, 0xa00f0003,
    0x3f950b0f, 0x40073190, 0x00000000, 0x00000000, 0x0200001f, 0x80000000,
    0xb0030000, 0x0200001f, 0x80000000, 0x900f0000, 0x0200001f, 0x90000000,
    0xa00f0800, 0x0200001f, 0x90000000, 0xa00f0801, 0x0200001f, 0x90000000,
    0xa00f0802, 0x03000042, 0x800f0000, 0xb0e40000, 0xa0e40800, 0x03000042,
    0x800f0001, 0xb0e40000, 0xa0e40801, 0x03000042, 0x800f0002, 0xb0e40000,
    0xa0e40802, 0x02000001, 0x80020000, 0x80000001, 0x02000001, 0x80040000,
    0x80000002, 0x03000002, 0x80070000, 0x80e40000, 0xa0e40000, 0x03000008,
    0x80010001, 0x80e40000, 0xa0e40001, 0x03000008, 0x80020001, 0x80e40000,
    0xa0e40002, 0x0400005a, 0x80040001, 0x80e40000, 0xa0e40003, 0xa0aa0003,
    0x02000001, 0x80080001, 0xa0ff0000, 0x03000005, 0x800f0000, 0x80e40001,
    0x90e40000, 0x02000001, 0x800f0800, 0x80e40000, 0x0000ffff
};

static const DWORD *D3D9_shaders[] = {
    D3D9_PixelShader_YUV_JPEG,
    D3D9_PixelShader_YUV_BT601,
    D3D9_PixelShader_YUV_BT709,
};

HRESULT D3D9_CreatePixelShader(IDirect3DDevice9 *d3dDevice, D3D9_Shader shader, IDirect3DPixelShader9 **pixelShader)
{
    return IDirect3DDevice9_CreatePixelShader(d3dDevice, D3D9_shaders[shader], pixelShader);
}

#endif /* SDL_VIDEO_RENDER_D3D */

/* vi: set ts=4 sw=4 expandtab: */
