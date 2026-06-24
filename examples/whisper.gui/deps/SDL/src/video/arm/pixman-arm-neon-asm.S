/*
 * Copyright © 2009 Nokia Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Author:  Siarhei Siamashka (siarhei.siamashka@nokia.com)
 */

/*
 * Copyright (c) 2018 RISC OS Open Ltd
 *
 * This software is provided 'as-is', without any express or implied
 * warranty.  In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

/* Prevent the stack from becoming executable for no reason... */
#if defined(__linux__) && defined(__ELF__)
.section .note.GNU-stack,"",%progbits
#endif

    .text
    .fpu neon
    .arch armv7a
    .object_arch armv4
    .eabi_attribute 10, 0 /* suppress Tag_FP_arch */
    .eabi_attribute 12, 0 /* suppress Tag_Advanced_SIMD_arch */
    .arm
    .altmacro
    .p2align 2

#include "pixman-arm-asm.h"
#include "pixman-arm-neon-asm.h"

/* Global configuration options and preferences */

/*
 * The code can optionally make use of unaligned memory accesses to improve
 * performance of handling leading/trailing pixels for each scanline.
 * Configuration variable RESPECT_STRICT_ALIGNMENT can be set to 0 for
 * example in linux if unaligned memory accesses are not configured to
 * generate.exceptions.
 */
.set RESPECT_STRICT_ALIGNMENT, 1

/*
 * Set default prefetch type. There is a choice between the following options:
 *
 * PREFETCH_TYPE_NONE (may be useful for the ARM cores where PLD is set to work
 * as NOP to workaround some HW bugs or for whatever other reason)
 *
 * PREFETCH_TYPE_SIMPLE (may be useful for simple single-issue ARM cores where
 * advanced prefetch intruduces heavy overhead)
 *
 * PREFETCH_TYPE_ADVANCED (useful for superscalar cores such as ARM Cortex-A8
 * which can run ARM and NEON instructions simultaneously so that extra ARM
 * instructions do not add (many) extra cycles, but improve prefetch efficiency)
 *
 * Note: some types of function can't support advanced prefetch and fallback
 *       to simple one (those which handle 24bpp pixels)
 */
.set PREFETCH_TYPE_DEFAULT, PREFETCH_TYPE_ADVANCED

/* Prefetch distance in pixels for simple prefetch */
.set PREFETCH_DISTANCE_SIMPLE, 64

/******************************************************************************/

/* We can actually do significantly better than the Pixman macros, at least for
 * the case of fills, by using a carefully scheduled inner loop. Cortex-A53
 * shows an improvement of up to 78% in ideal cases (large fills to L1 cache).
 */

.macro generate_fillrect_function name, bpp, log2Bpp
/*
 * void name(int32_t w, int32_t h, uint8_t *dst, int32_t dst_stride, uint8_t src);
 * On entry:
 * a1 = width, pixels
 * a2 = height, rows
 * a3 = pointer to top-left destination pixel
 * a4 = stride, pixels
 * [sp] = pixel value to fill with
 * Within the function:
 * v1 = width remaining
 * v2 = vst offset
 * v3 = alternate pointer
 * ip = data ARM register
 */
pixman_asm_function name
    vld1.\bpp   {d0[],d1[]}, [sp]
    sub         a4, a1
    vld1.\bpp   {d2[],d3[]}, [sp]
    cmp         a1, #(15+64) >> \log2Bpp
    push        {v1-v3,lr}
    vmov        ip, s0
    blo         51f

    /* Long-row case */
    mov         v2, #64
1:  mov         v1, a1
    ands        v3, a3, #15
    beq         2f
    /* Leading pixels */
    rsb         v3, v3, #16  /* number of leading bytes until 16-byte aligned */
    sub         v1, v1, v3, lsr #\log2Bpp
    rbit        v3, v3
.if bpp <= 16
.if bpp == 8
    tst         a3, #1       /* bit 0 unaffected by rsb so can avoid register interlock */
    strneb      ip, [a3], #1
    tst         v3, #1<<30
.else
    tst         a3, #2       /* bit 1 unaffected by rsb (assuming halfword alignment) so can avoid register interlock */
.endif
    strneh      ip, [a3], #2
.endif
    movs        v3, v3, lsl #3
    vstmcs      a3!, {s0}
    vstmmi      a3!, {d0}
2:  sub         v1, v1, #64 >> \log2Bpp /* simplifies inner loop termination */
    add         v3, a3, #32
    /* Inner loop */
3:  vst1.\bpp   {q0-q1}, [a3 :128], v2
    subs        v1, v1, #64 >> \log2Bpp
    vst1.\bpp   {q0-q1}, [v3 :128], v2
    bhs         3b
    /* Trailing pixels */
4:  movs        v1, v1, lsl #27 + \log2Bpp
    bcc         5f
    vst1.\bpp   {q0-q1}, [a3 :128]!
5:  bpl         6f
    vst1.\bpp   {q0}, [a3 :128]!
6:  movs        v1, v1, lsl #2
    vstmcs      a3!, {d0}
    vstmmi      a3!, {s0}
.if bpp <= 16
    movs        v1, v1, lsl #2
    strcsh      ip, [a3], #2
.if bpp == 8
    strmib      ip, [a3], #1
.endif
.endif
    subs        a2, a2, #1
    add         a3, a3, a4, lsl #\log2Bpp
    bhi         1b
    pop         {v1-v3,pc}

    /* Short-row case */
51: movs        v1, a1
.if bpp == 8
    tst         a3, #3
    beq         53f
52: subs        v1, v1, #1
    blo         57f
    strb        ip, [a3], #1
    tst         a3, #3
    bne         52b
.elseif bpp == 16
    tstne       a3, #2
    subne       v1, v1, #1
    strneh      ip, [a3], #2
.endif
53: cmp         v1, #32 >> \log2Bpp
    bcc         54f
    vst1.\bpp   {q0-q1}, [a3]!
    sub         v1, v1, #32 >> \log2Bpp
    /* Trailing pixels */
54: movs        v1, v1, lsl #27 + \log2Bpp
    bcc         55f
    vst1.\bpp   {q0-q1}, [a3]!
55: bpl         56f
    vst1.\bpp   {q0}, [a3]!
56: movs        v1, v1, lsl #2
    vstmcs      a3!, {d0}
    vstmmi      a3!, {s0}
.if bpp <= 16
    movs        v1, v1, lsl #2
    strcsh      ip, [a3], #2
.if bpp == 8
    strmib      ip, [a3], #1
.endif
.endif
    subs        a2, a2, #1
    add         a3, a3, a4, lsl #\log2Bpp
    bhi         51b
57: pop         {v1-v3,pc}

.endfunc
.endm

generate_fillrect_function FillRect32ARMNEONAsm, 32, 2
generate_fillrect_function FillRect16ARMNEONAsm, 16, 1
generate_fillrect_function FillRect8ARMNEONAsm,  8,  0

/******************************************************************************/

.macro RGBtoRGBPixelAlpha_process_pixblock_head
    vmvn        d30, d3  /* get inverted source alpha */
    vmov        d31, d7  /* dest alpha is always unchanged */
    vmull.u8    q14, d0, d3
    vmlal.u8    q14, d4, d30
    vmull.u8    q0, d1, d3
    vmlal.u8    q0, d5, d30
    vmull.u8    q1, d2, d3
    vmlal.u8    q1, d6, d30
    vrshr.u16   q2, q14, #8
    vrshr.u16   q3, q0, #8
    vraddhn.u16 d28, q14, q2
    vrshr.u16   q2, q1, #8
    vraddhn.u16 d29, q0, q3
    vraddhn.u16 d30, q1, q2
.endm

.macro RGBtoRGBPixelAlpha_process_pixblock_tail
    /* nothing */
.endm

.macro RGBtoRGBPixelAlpha_process_pixblock_tail_head
    vld4.8      {d0-d3}, [SRC]!
                                    PF add PF_X, PF_X, #8
        vst4.8      {d28-d31}, [DST_W :128]!
                                    PF tst PF_CTL, #0xF
    vld4.8      {d4-d7}, [DST_R :128]!
                                    PF addne PF_X, PF_X, #8
    vmvn        d30, d3  /* get inverted source alpha */
    vmov        d31, d7  /* dest alpha is always unchanged */
    vmull.u8    q14, d0, d3
                                    PF subne PF_CTL, PF_CTL, #1
    vmlal.u8    q14, d4, d30
                                    PF cmp PF_X, ORIG_W
    vmull.u8    q0, d1, d3
                                    PF pld, [PF_SRC, PF_X, lsl #src_bpp_shift]
    vmlal.u8    q0, d5, d30
                                    PF pld, [PF_DST, PF_X, lsl #dst_bpp_shift]
    vmull.u8    q1, d2, d3
                                    PF subge PF_X, PF_X, ORIG_W
    vmlal.u8    q1, d6, d30
                                    PF subges PF_CTL, PF_CTL, #0x10
    vrshr.u16   q2, q14, #8
                                    PF ldrgeb DUMMY, [PF_SRC, SRC_STRIDE, lsl #src_bpp_shift]!
    vrshr.u16   q3, q0, #8
                                    PF ldrgeb DUMMY, [PF_DST, DST_STRIDE, lsl #dst_bpp_shift]!
    vraddhn.u16 d28, q14, q2
    vrshr.u16   q2, q1, #8
    vraddhn.u16 d29, q0, q3
    vraddhn.u16 d30, q1, q2
.endm

generate_composite_function \
    BlitRGBtoRGBPixelAlphaARMNEONAsm, 32, 0, 32, \
    FLAG_DST_READWRITE | FLAG_DEINTERLEAVE_32BPP, \
    8, /* number of pixels, processed in a single block */ \
    5, /* prefetch distance */ \
    default_init, \
    default_cleanup, \
    RGBtoRGBPixelAlpha_process_pixblock_head, \
    RGBtoRGBPixelAlpha_process_pixblock_tail, \
    RGBtoRGBPixelAlpha_process_pixblock_tail_head

 /******************************************************************************/

.macro ARGBto565PixelAlpha_process_pixblock_head
    vmvn        d6, d3
    vshr.u8     d1, #2
    vshr.u8     d3, #3
    vshr.u8     d0, #3
    vshrn.u16   d7, q2, #3
    vshrn.u16   d25, q2, #8
    vbic.i16    q2, #0xe0
    vshr.u8     d6, #3
    vshr.u8     d7, #2
    vshr.u8     d2, #3
    vmovn.u16   d24, q2
    vshr.u8     d25, #3
    vmull.u8    q13, d1, d3
    vmlal.u8    q13, d7, d6
    vmull.u8    q14, d0, d3
    vmlal.u8    q14, d24, d6
    vmull.u8    q15, d2, d3
    vmlal.u8    q15, d25, d6
.endm

.macro ARGBto565PixelAlpha_process_pixblock_tail
    vsra.u16    q13, #5
    vsra.u16    q14, #5
    vsra.u16    q15, #5
    vrshr.u16   q13, #5
    vrshr.u16   q14, #5
    vrshr.u16   q15, #5
    vsli.u16    q14, q13, #5
    vsli.u16    q14, q15, #11
.endm

.macro ARGBto565PixelAlpha_process_pixblock_tail_head
    vld4.8      {d0-d3}, [SRC]!
                                    PF add PF_X, PF_X, #8
        vsra.u16    q13, #5
                                    PF tst PF_CTL, #0xF
        vsra.u16    q14, #5
                                    PF addne PF_X, PF_X, #8
        vsra.u16    q15, #5
                                    PF subne PF_CTL, PF_CTL, #1
        vrshr.u16   q13, #5
                                    PF cmp PF_X, ORIG_W
        vrshr.u16   q14, #5
                                    PF pld, [PF_SRC, PF_X, lsl #src_bpp_shift]
        vrshr.u16   q15, #5
                                    PF pld, [PF_DST, PF_X, lsl #dst_bpp_shift]
    vld1.8      {d4-d5}, [DST_R]!
                                    PF subge PF_X, PF_X, ORIG_W
        vsli.u16    q14, q13, #5
                                    PF subges PF_CTL, PF_CTL, #0x10
        vsli.u16    q14, q15, #11
                                    PF ldrgeb DUMMY, [PF_SRC, SRC_STRIDE, lsl #src_bpp_shift]!
        vst1.8      {q14}, [DST_W :128]!
    vmvn        d6, d3
    vshr.u8     d1, #2
    vshr.u8     d3, #3
    vshr.u8     d0, #3
    vshrn.u16   d7, q2, #3
    vshrn.u16   d25, q2, #8
    vbic.i16    q2, #0xe0
                                    PF ldrgeb DUMMY, [PF_DST, DST_STRIDE, lsl #dst_bpp_shift]!
    vshr.u8     d6, #3
    vshr.u8     d7, #2
    vshr.u8     d2, #3
    vmovn.u16   d24, q2
    vshr.u8     d25, #3
    vmull.u8    q13, d1, d3
    vmlal.u8    q13, d7, d6
    vmull.u8    q14, d0, d3
    vmlal.u8    q14, d24, d6
    vmull.u8    q15, d2, d3
    vmlal.u8    q15, d25, d6
.endm

generate_composite_function \
    BlitARGBto565PixelAlphaARMNEONAsm, 32, 0, 16, \
    FLAG_DST_READWRITE | FLAG_DEINTERLEAVE_32BPP, \
    8, /* number of pixels, processed in a single block */ \
    6, /* prefetch distance */ \
    default_init, \
    default_cleanup, \
    ARGBto565PixelAlpha_process_pixblock_head, \
    ARGBto565PixelAlpha_process_pixblock_tail, \
    ARGBto565PixelAlpha_process_pixblock_tail_head
