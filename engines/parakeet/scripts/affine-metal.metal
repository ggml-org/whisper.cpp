/*
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Experimental affine GGUF carrier matmul. The 64x32, K32 swizzled staging
// follows ggml's MIT-licensed kernel_mul_mm_simd (speech-ggml 6bb85ad6).
// Unlike stock Q4_0/Q8_0, carrier block scales MUST equal 1; true FP32
// minimum/step are per output channel. No full dense matrix is materialized.
struct AffineArgs { uint M, N, K, bits; };

inline half affine_value(device const uchar * codes, device const float * step,
                         device const float * minimum, uint row, uint k,
                         constant AffineArgs & a) {
#pragma clang fp contract(off)
    uint block = row*(a.K/32) + k/32;
    uint lane = k%32;
    uint code;
    if (a.bits == 4) {
        uint packed = codes[block*18 + 2 + lane%16];
        code = (packed >> ((lane/16)*4)) & 15;
    } else {
        code = uint(int(as_type<char>(codes[block*34 + 2 + lane])) + 128);
    }
    float scaled = float(code)*step[row];
    return half(scaled + minimum[row]);
}

kernel void affine_restore(device const uchar * codes [[buffer(0)]],
                           device const float * step [[buffer(1)]],
                           device const float * minimum [[buffer(2)]],
                           device half * result [[buffer(3)]],
                           constant AffineArgs & a [[buffer(4)]],
                           uint i [[thread_position_in_grid]]) {
    if (i<a.N*a.K) result[i]=affine_value(codes,step,minimum,i/a.K,i%a.K,a);
}

kernel void affine_mm(device const uchar * codes [[buffer(0)]],
                      device const float * step [[buffer(1)]],
                      device const float * minimum [[buffer(2)]],
                      device const float * x [[buffer(3)]],
                      device float * result [[buffer(4)]],
                      constant AffineArgs & a [[buffer(5)]],
                      uint2 group [[threadgroup_position_in_grid]],
                      ushort tid [[thread_index_in_threadgroup]],
                      ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half sa[64*32];
    threadgroup half sb[32*32];
    threadgroup float tail[64*32];
    const uint r0=group.y*64, r1=group.x*32;
    const ushort nr0=min(uint(64),a.N-r0), nr1=min(uint(32),a.M-r1);
    const ushort lr0=min(ushort(tid/2),ushort(nr0-1));
    const ushort lr1=min(ushort(tid/4),ushort(nr1-1));
    const ushort il0=tid%2, iy=8*(tid%4);
    simdgroup_float8x8 mc[8];
    for (ushort i=0;i<8;++i) mc[i]=make_filled_simdgroup_matrix<float,8>(0.f);
    for (uint k=0;k<a.K;k+=32) {
        half values[16];
        for (ushort i=0;i<16;++i) values[i]=affine_value(codes,step,minimum,r0+lr0,k+16*il0+i,a);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (ushort i=0;i<16;++i) {
            const ushort sx=2*il0+i/8,sy=(tid/2)/8,lx=(tid/2)%8,ly=i%8;
            sa[64*(8*sx+sy)+8*ly+lx]=values[i];
        }
        for (ushort i=0;i<8;++i) {
            const ushort sx=tid%4,sy=(tid/4)/8,ly=(tid/4)%8;
            sb[64*(4*sx+sy)+8*ly+i]=half(x[(r1+lr1)*a.K+k+iy+i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half * lsma=sa+4*64*(sg%2);
        threadgroup const half * lsmb=sb+2*64*(sg/2);
        for (ushort ik=0;ik<4;++ik) {
            simdgroup_half8x8 ma[4],mb[2];
            for (ushort i=0;i<4;++i) simdgroup_load(ma[i],lsma+64*i,8);
            for (ushort i=0;i<2;++i) simdgroup_load(mb[i],lsmb+64*i,8);
            for (ushort i=0;i<8;++i) simdgroup_multiply_accumulate(mc[i],mb[i/4],ma[i%4],mc[i]);
            lsma+=8*64;lsmb+=4*64;
        }
    }
    if (nr0==64 && nr1==32) {
        device float * out=result+r0+32*(sg&1)+(r1+16*(sg>>1))*a.N;
        for (ushort i=0;i<8;++i) simdgroup_store(mc[i],out+8*(i%4)+8*a.N*(i/4),a.N);
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * out=tail+32*(sg&1)+16*(sg>>1)*64;
        for (ushort i=0;i<8;++i) simdgroup_store(mc[i],out+8*(i%4)+8*64*(i/4),64);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i=tid;i<nr1*64;i+=128) {
            uint row=i/64,col=i%64;
            if(col<nr0) result[(r1+row)*a.N+r0+col]=tail[i];
        }
    }
}
