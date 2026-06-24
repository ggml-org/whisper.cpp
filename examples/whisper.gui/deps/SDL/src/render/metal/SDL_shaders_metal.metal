#include <metal_texture>
#include <metal_matrix>

using namespace metal;

struct SolidVertexInput
{
    float2 position [[attribute(0)]];
    float4 color    [[attribute(1)]];
};

struct SolidVertexOutput
{
    float4 position [[position]];
    float4 color;
    float pointSize [[point_size]];
};

vertex SolidVertexOutput SDL_Solid_vertex(SolidVertexInput in [[stage_in]],
                                          constant float4x4 &projection [[buffer(2)]],
                                          constant float4x4 &transform [[buffer(3)]])
{
    SolidVertexOutput v;
    v.position = (projection * transform) * float4(in.position, 0.0f, 1.0f);
    v.color = in.color;
    v.pointSize = 1.0f;
    return v;
}

fragment float4 SDL_Solid_fragment(SolidVertexInput in [[stage_in]])
{
    return in.color;
}

struct CopyVertexInput
{
    float2 position [[attribute(0)]];
    float4 color    [[attribute(1)]];
    float2 texcoord [[attribute(2)]];
};

struct CopyVertexOutput
{
    float4 position [[position]];
    float4 color;
    float2 texcoord;
};

vertex CopyVertexOutput SDL_Copy_vertex(CopyVertexInput in [[stage_in]],
                                        constant float4x4 &projection [[buffer(2)]],
                                        constant float4x4 &transform [[buffer(3)]])
{
    CopyVertexOutput v;
    v.position = (projection * transform) * float4(in.position, 0.0f, 1.0f);
    v.color = in.color;
    v.texcoord = in.texcoord;
    return v;
}

fragment float4 SDL_Copy_fragment(CopyVertexOutput vert [[stage_in]],
                                  texture2d<float> tex [[texture(0)]],
                                  sampler s [[sampler(0)]])
{
    return tex.sample(s, vert.texcoord) * vert.color;
}

struct YUVDecode
{
    float3 offset;
    float3 Rcoeff;
    float3 Gcoeff;
    float3 Bcoeff;
};

fragment float4 SDL_YUV_fragment(CopyVertexOutput vert [[stage_in]],
                                 constant YUVDecode &decode [[buffer(1)]],
                                 texture2d<float> texY [[texture(0)]],
                                 texture2d_array<float> texUV [[texture(1)]],
                                 sampler s [[sampler(0)]])
{
    float3 yuv;
    yuv.x = texY.sample(s, vert.texcoord).r;
    yuv.y = texUV.sample(s, vert.texcoord, 0).r;
    yuv.z = texUV.sample(s, vert.texcoord, 1).r;

    yuv += decode.offset;

    return vert.color * float4(dot(yuv, decode.Rcoeff), dot(yuv, decode.Gcoeff), dot(yuv, decode.Bcoeff), 1.0);
}

fragment float4 SDL_NV12_fragment(CopyVertexOutput vert [[stage_in]],
                                 constant YUVDecode &decode [[buffer(1)]],
                                 texture2d<float> texY [[texture(0)]],
                                 texture2d<float> texUV [[texture(1)]],
                                 sampler s [[sampler(0)]])
{
    float3 yuv;
    yuv.x = texY.sample(s, vert.texcoord).r;
    yuv.yz = texUV.sample(s, vert.texcoord).rg;

    yuv += decode.offset;

    return vert.color * float4(dot(yuv, decode.Rcoeff), dot(yuv, decode.Gcoeff), dot(yuv, decode.Bcoeff), 1.0);
}

fragment float4 SDL_NV21_fragment(CopyVertexOutput vert [[stage_in]],
                                 constant YUVDecode &decode [[buffer(1)]],
                                 texture2d<float> texY [[texture(0)]],
                                 texture2d<float> texUV [[texture(1)]],
                                 sampler s [[sampler(0)]])
{
    float3 yuv;
    yuv.x = texY.sample(s, vert.texcoord).r;
    yuv.yz = texUV.sample(s, vert.texcoord).gr;

    yuv += decode.offset;

    return vert.color * float4(dot(yuv, decode.Rcoeff), dot(yuv, decode.Gcoeff), dot(yuv, decode.Bcoeff), 1.0);
}

