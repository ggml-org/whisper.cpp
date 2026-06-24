/*
 * Copyright (c) 2016 RISC OS Open Ltd
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

/* Prevent the stack from becoming executable */
#if defined(__linux__) && defined(__ELF__)
.section .note.GNU-stack,"",%progbits
#endif

	.text
	.arch armv6
	.object_arch armv4
	.arm
	.altmacro
	.p2align 2

#include "pixman-arm-asm.h"
#include "pixman-arm-simd-asm.h"

/* A head macro should do all processing which results in an output of up to
 * 16 bytes, as far as the final load instruction. The corresponding tail macro
 * should complete the processing of the up-to-16 bytes. The calling macro will
 * sometimes choose to insert a preload or a decrement of X between them.
 *   cond           ARM condition code for code block
 *   numbytes       Number of output bytes that should be generated this time
 *   firstreg       First WK register in which to place output
 *   unaligned_src  Whether to use non-wordaligned loads of source image
 *   unaligned_mask Whether to use non-wordaligned loads of mask image
 *   preload        If outputting 16 bytes causes 64 bytes to be read, whether an extra preload should be output
 */

/******************************************************************************/

.macro FillRect32_init
        ldr     SRC, [sp, #ARGS_STACK_OFFSET]
        mov     STRIDE_S, SRC
        mov     MASK, SRC
        mov     STRIDE_M, SRC
.endm

.macro FillRect16_init
        ldrh    SRC, [sp, #ARGS_STACK_OFFSET]
        orr     SRC, SRC, lsl #16
        mov     STRIDE_S, SRC
        mov     MASK, SRC
        mov     STRIDE_M, SRC
.endm

.macro FillRect8_init
        ldrb    SRC, [sp, #ARGS_STACK_OFFSET]
        orr     SRC, SRC, lsl #8
        orr     SRC, SRC, lsl #16
        mov     STRIDE_S, SRC
        mov     MASK, SRC
        mov     STRIDE_M, SRC
.endm

.macro FillRect_process_tail  cond, numbytes, firstreg
    WK4     .req    SRC
    WK5     .req    STRIDE_S
    WK6     .req    MASK
    WK7     .req    STRIDE_M
        pixst   cond, numbytes, 4, DST
    .unreq  WK4
    .unreq  WK5
    .unreq  WK6
    .unreq  WK7
.endm

generate_composite_function \
    FillRect32ARMSIMDAsm, 0, 0, 32, \
    FLAG_DST_WRITEONLY | FLAG_COND_EXEC | FLAG_PROCESS_PRESERVES_PSR | FLAG_PROCESS_DOES_STORE | FLAG_PROCESS_PRESERVES_SCRATCH \
    0, /* prefetch distance doesn't apply */ \
    FillRect32_init \
    nop_macro, /* newline */ \
    nop_macro /* cleanup */ \
    nop_macro /* process head */ \
    FillRect_process_tail

generate_composite_function \
    FillRect16ARMSIMDAsm, 0, 0, 16, \
    FLAG_DST_WRITEONLY | FLAG_COND_EXEC | FLAG_PROCESS_PRESERVES_PSR | FLAG_PROCESS_DOES_STORE | FLAG_PROCESS_PRESERVES_SCRATCH \
    0, /* prefetch distance doesn't apply */ \
    FillRect16_init \
    nop_macro, /* newline */ \
    nop_macro /* cleanup */ \
    nop_macro /* process head */ \
    FillRect_process_tail

generate_composite_function \
    FillRect8ARMSIMDAsm, 0, 0, 8, \
    FLAG_DST_WRITEONLY | FLAG_COND_EXEC | FLAG_PROCESS_PRESERVES_PSR | FLAG_PROCESS_DOES_STORE | FLAG_PROCESS_PRESERVES_SCRATCH \
    0, /* prefetch distance doesn't apply */ \
    FillRect8_init \
    nop_macro, /* newline */ \
    nop_macro /* cleanup */ \
    nop_macro /* process head */ \
    FillRect_process_tail

/******************************************************************************/

/* This differs from the over_8888_8888 routine in Pixman in that the destination
 * alpha component is always left unchanged, and RGB components are not
 * premultiplied by alpha. It differs from BlitRGBtoRGBPixelAlpha in that
 * renormalisation is done by multiplying by 257/256 (with rounding) rather than
 * simply shifting right by 8 bits - removing the need to special-case alpha=0xff.
 */

.macro RGBtoRGBPixelAlpha_init
        line_saved_regs STRIDE_S, ORIG_W
        mov     MASK, #0x80
.endm

.macro RGBtoRGBPixelAlpha_1pixel_translucent  s, d, tmp0, tmp1, tmp2, tmp3, half
        uxtb    tmp3, s
        uxtb    tmp0, d
        sub     tmp0, tmp3, tmp0
        uxtb    tmp3, s, ror #16
        uxtb    tmp1, d, ror #16
        sub     tmp1, tmp3, tmp1
        uxtb    tmp3, s, ror #8
        mov     s, s, lsr #24
        uxtb    tmp2, d, ror #8
        sub     tmp2, tmp3, tmp2
        smlabb  tmp0, tmp0, s, half
        smlabb  tmp1, tmp1, s, half
        smlabb  tmp2, tmp2, s, half
        add     tmp0, tmp0, asr #8
        add     tmp1, tmp1, asr #8
        add     tmp2, tmp2, asr #8
        pkhbt   tmp0, tmp0, tmp1, lsl #16
        and     tmp2, tmp2, #0xff00
        uxtb16  tmp0, tmp0, ror #8
        orr     tmp0, tmp0, tmp2
        uadd8   d, d, tmp0
.endm

.macro RGBtoRGBPixelAlpha_1pixel_opaque  s, d
        and     d, d, #0xff000000
        bic     s, s, #0xff000000
        orr     d, d, s
.endm

.macro RGBtoRGBPixelAlpha_process_head  cond, numbytes, firstreg, unaligned_src, unaligned_mask, preload
 .if numbytes == 16
        ldm     SRC!, {WK0, WK1}
        ldm     SRC!, {STRIDE_S, STRIDE_M}
        ldrd    WK2, WK3, [DST], #16
        orr     SCRATCH, WK0, WK1
        and     ORIG_W, WK0, WK1
        orr     SCRATCH, SCRATCH, STRIDE_S
        and     ORIG_W, ORIG_W, STRIDE_S
        orr     SCRATCH, SCRATCH, STRIDE_M
        and     ORIG_W, ORIG_W, STRIDE_M
        tst     SCRATCH, #0xff000000
 .elseif numbytes == 8
        ldm     SRC!, {WK0, WK1}
        ldm     DST!, {WK2, WK3}
        orr     SCRATCH, WK0, WK1
        and     ORIG_W, WK0, WK1
        tst     SCRATCH, #0xff000000
 .else // numbytes == 4
        ldr     WK0, [SRC], #4
        ldr     WK2, [DST], #4
        tst     WK0, #0xff000000
 .endif
.endm

.macro RGBtoRGBPixelAlpha_process_tail  cond, numbytes, firstreg
        beq     20f @ all transparent
 .if numbytes == 16
        cmp     ORIG_W, #0xff000000
        bhs     10f @ all opaque
        RGBtoRGBPixelAlpha_1pixel_translucent WK0, WK2, STRIDE_S, STRIDE_M, SCRATCH, ORIG_W, MASK
        RGBtoRGBPixelAlpha_1pixel_translucent WK1, WK3, STRIDE_S, STRIDE_M, SCRATCH, ORIG_W, MASK
        strd    WK2, WK3, [DST, #-16]
        ldrd    WK0, WK1, [SRC, #-8]
        ldrd    WK2, WK3, [DST, #-8]
        RGBtoRGBPixelAlpha_1pixel_translucent WK0, WK2, STRIDE_S, STRIDE_M, SCRATCH, ORIG_W, MASK
        RGBtoRGBPixelAlpha_1pixel_translucent WK1, WK3, STRIDE_S, STRIDE_M, SCRATCH, ORIG_W, MASK
        b       19f
10:     RGBtoRGBPixelAlpha_1pixel_opaque WK0, WK2
        RGBtoRGBPixelAlpha_1pixel_opaque WK1, WK3
        strd    WK2, WK3, [DST, #-16]
        ldrd    WK0, WK1, [SRC, #-8]
        ldrd    WK2, WK3, [DST, #-8]
        RGBtoRGBPixelAlpha_1pixel_opaque WK0, WK2
        RGBtoRGBPixelAlpha_1pixel_opaque WK1, WK3
19:     strd    WK2, WK3, [DST, #-8]
 .elseif numbytes == 8
        cmp     ORIG_W, #0xff000000
        bhs     10f @ all opaque
        RGBtoRGBPixelAlpha_1pixel_translucent WK0, WK2, STRIDE_S, STRIDE_M, SCRATCH, ORIG_W, MASK
        RGBtoRGBPixelAlpha_1pixel_translucent WK1, WK3, STRIDE_S, STRIDE_M, SCRATCH, ORIG_W, MASK
        b       19f
10:     RGBtoRGBPixelAlpha_1pixel_opaque WK0, WK2
        RGBtoRGBPixelAlpha_1pixel_opaque WK1, WK3
19:     strd    WK2, WK3, [DST, #-8]
 .else // numbytes == 4
        cmp     WK0, #0xff000000
        bhs     10f @ opaque
        RGBtoRGBPixelAlpha_1pixel_translucent WK0, WK2, STRIDE_S, STRIDE_M, SCRATCH, ORIG_W, MASK
        b       19f
10:     RGBtoRGBPixelAlpha_1pixel_opaque WK0, WK2
19:     str     WK2, [DST, #-4]
 .endif
20:
.endm

generate_composite_function \
    BlitRGBtoRGBPixelAlphaARMSIMDAsm, 32, 0, 32, \
    FLAG_DST_READWRITE | FLAG_BRANCH_OVER | FLAG_PROCESS_CORRUPTS_PSR | FLAG_PROCESS_DOES_STORE | FLAG_SPILL_LINE_VARS | FLAG_PROCESS_CORRUPTS_WK0, \
    2, /* prefetch distance */ \
    RGBtoRGBPixelAlpha_init, \
    nop_macro, /* newline */ \
    nop_macro, /* cleanup */ \
    RGBtoRGBPixelAlpha_process_head, \
    RGBtoRGBPixelAlpha_process_tail

/******************************************************************************/

.macro ARGBto565PixelAlpha_init
        line_saved_regs STRIDE_D, STRIDE_S, ORIG_W
        mov     MASK, #0x001f
        mov     STRIDE_M, #0x0010
        orr     MASK, MASK, MASK, lsl #16
        orr     STRIDE_M, STRIDE_M, STRIDE_M, lsl #16
.endm

.macro ARGBto565PixelAlpha_newline
        mov     STRIDE_S, #0x0200
.endm

/* On entry:
 * s1 holds 1 32bpp source pixel
 * d holds 1 16bpp destination pixel
 * rbmask, rbhalf, ghalf hold 0x001f001f, 0x00100010, 0x00000200 respectively
 * other registers are temporaries
 * On exit:
 * Constant registers preserved
 */

.macro ARGBto565PixelAlpha_1pixel_translucent  s, d, rbmask, rbhalf, ghalf, alpha, rb, g, misc
        mov     alpha, s, lsr #27
        and     misc, s, #0xfc00
        and     g, d, #0x07e0
        pkhbt   rb, d, d, lsl #5
        rsb     misc, g, misc, lsr #5
        and     s, rbmask, s, lsr #3
        and     rb, rbmask, rb
        sub     s, s, rb
        smlabb  misc, misc, alpha, ghalf
        mla     s, s, alpha, rbhalf
        add     misc, misc, misc, lsl #5
        add     g, g, misc, asr #10
        add     s, s, s, lsl #5
        and     g, g, #0x07e0
        add     rb, rb, s, asr #10
        and     rb, rb, rbmask
        pkhbt   rb, rb, rb, lsl #11
        orr     d, rb, g
        orr     d, d, rb, lsr #16
.endm

/* On entry:
 * s1 holds 1 32bpp source pixel
 * d holds 1 16bpp destination pixel
 * rbmask holds 0x001f001f
 * On exit:
 * Constant registers preserved
 */

.macro ARGBto565PixelAlpha_1pixel_opaque  s, d, rbmask
        and     d, rbmask, s, lsr #3
        and     s, s, #0xfc00
        orr     d, d, d, lsr #5
        orr     d, d, s, lsr #5
.endm

/* On entry:
 * s1, s2 hold 2 32bpp source pixels
 * d holds 2 16bpp destination pixels
 * rbmask, rbhalf, ghalf hold 0x001f001f, 0x00100010, 0x00000200 respectively
 * other registers are temporaries
 * On exit:
 * Constant registers preserved
 * Blended results have been written through destination pointer
 */

.macro ARGBto565PixelAlpha_2pixels_translucent  s1, s2, d, rbmask, rbhalf, ghalf, alpha, rb, g, misc
        mov     alpha, s1, lsr #27
        and     misc, s1, #0xfc00
        and     g, d, #0x07e0
        pkhbt   rb, d, d, lsl #5
        rsb     misc, g, misc, lsr #5
        and     s1, rbmask, s1, lsr #3
        and     rb, rbmask, rb
        sub     s1, s1, rb
        smlabb  misc, misc, alpha, ghalf
        mla     s1, s1, alpha, rbhalf
          uxth    d, d, ror #16
        add     misc, misc, misc, lsl #5
          mov     alpha, s2, lsr #27
        add     g, g, misc, asr #10
        add     s1, s1, s1, lsl #5
        and     g, g, #0x07e0
        add     rb, rb, s1, asr #10
        and     rb, rb, rbmask
          and     misc, s2, #0xfc00
        pkhbt   rb, rb, rb, lsl #11
          and     s1, d, #0x07e0
          pkhbt   d, d, d, lsl #5
          rsb     misc, s1, misc, lsr #5
          and     s2, rbmask, s2, lsr #3
          and     d, rbmask, d
          sub     s2, s2, d
          smlabb  misc, misc, alpha, ghalf
          mla     s2, s2, alpha, rbhalf
        orr     alpha, rb, g
          add     misc, misc, misc, lsl #5
        orr     alpha, alpha, rb, lsr #16
          add     s1, s1, misc, asr #10
          add     s2, s2, s2, lsl #5
          and     s1, s1, #0x07e0
          add     d, d, s2, asr #10
          and     d, d, rbmask
        strh    alpha, [DST, #-4]
          pkhbt   d, d, d, lsl #11
          orr     alpha, d, s1
          orr     alpha, alpha, d, lsr #16
          strh    alpha, [DST, #-2]
.endm

/* On entry:
 * s1, s2 hold 2 32bpp source pixels
 * rbmask holds 0x001f001f
 * other registers are temporaries
 * On exit:
 * Constant registers preserved
 * Blended results have been written through destination pointer
 */

.macro ARGBto565PixelAlpha_2pixels_opaque  s1, s2, d, rbmask, g
        and     g, s1, #0xfc00
        and     d, rbmask, s1, lsr #3
          and     s1, rbmask, s2, lsr #3
        orr     d, d, d, lsr #5
        orr     d, d, g, lsr #5
          and     g, s2, #0xfc00
        strh    d, [DST, #-4]
          orr     s1, s1, s1, lsr #5
          orr     s1, s1, g, lsr #5
          strh    s1, [DST, #-2]
.endm

.macro ARGBto565PixelAlpha_2pixels_head
        ldrd    WK0, WK1, [SRC], #8
        ldr     WK2, [DST], #4
        orr     SCRATCH, WK0, WK1
        and     ORIG_W, WK0, WK1
        tst     SCRATCH, #0xff000000
.endm

.macro ARGBto565PixelAlpha_2pixels_tail
        beq     20f @ all transparent
        cmp     ORIG_W, #0xff000000
        bhs     10f @ all opaque
        ARGBto565PixelAlpha_2pixels_translucent  WK0, WK1, WK2, MASK, STRIDE_M, STRIDE_S, STRIDE_D, WK3, SCRATCH, ORIG_W
        b       20f
10:     ARGBto565PixelAlpha_2pixels_opaque  WK0, WK1, WK2, MASK, SCRATCH
20:
.endm

.macro ARGBto565PixelAlpha_process_head  cond, numbytes, firstreg, unaligned_src, unaligned_mask, preload
 .if numbytes == 16
        ARGBto565PixelAlpha_2pixels_head
        ARGBto565PixelAlpha_2pixels_tail
        ARGBto565PixelAlpha_2pixels_head
        ARGBto565PixelAlpha_2pixels_tail
 .endif
 .if numbytes >= 8
        ARGBto565PixelAlpha_2pixels_head
        ARGBto565PixelAlpha_2pixels_tail
 .endif
 .if numbytes >= 4
        ARGBto565PixelAlpha_2pixels_head
 .else // numbytes == 2
        ldr     WK0, [SRC], #4
        ldrh    WK2, [DST], #2
        tst     WK0, #0xff000000
 .endif
.endm

.macro ARGBto565PixelAlpha_process_tail  cond, numbytes, firstreg
 .if numbytes >= 4
        ARGBto565PixelAlpha_2pixels_tail
 .else // numbytes == 2
        beq     20f @ all transparent
        cmp     WK0, #0xff000000
        bhs     10f @ opaque
        ARGBto565PixelAlpha_1pixel_translucent  WK0, WK2, MASK, STRIDE_M, STRIDE_S, STRIDE_D, WK3, SCRATCH, ORIG_W
        b       19f
10:     ARGBto565PixelAlpha_1pixel_opaque WK0, WK2, MASK
19:     strh    WK2, [DST, #-2]
20:
 .endif
.endm

generate_composite_function \
    BlitARGBto565PixelAlphaARMSIMDAsm, 32, 0, 16, \
    FLAG_DST_READWRITE | FLAG_BRANCH_OVER | FLAG_PROCESS_CORRUPTS_PSR | FLAG_PROCESS_DOES_STORE | FLAG_SPILL_LINE_VARS | FLAG_PROCESS_CORRUPTS_WK0, \
    2, /* prefetch distance */ \
    ARGBto565PixelAlpha_init, \
    ARGBto565PixelAlpha_newline, \
    nop_macro, /* cleanup */ \
    ARGBto565PixelAlpha_process_head, \
    ARGBto565PixelAlpha_process_tail

 /******************************************************************************/

.macro BGR888toRGB888_1pixel cond, reg, tmp
        uxtb16&cond  tmp, WK&reg, ror #8
        uxtb16&cond  WK&reg, WK&reg, ror #16
        orr&cond     WK&reg, WK&reg, tmp, lsl #8
.endm

.macro BGR888toRGB888_2pixels cond, reg1, reg2, tmp1, tmp2
        uxtb16&cond  tmp1, WK&reg1, ror #8
        uxtb16&cond  WK&reg1, WK&reg1, ror #16
        uxtb16&cond  tmp2, WK&reg2, ror #8
        uxtb16&cond  WK&reg2, WK&reg2, ror #16
        orr&cond     WK&reg1, WK&reg1, tmp1, lsl #8
        orr&cond     WK&reg2, WK&reg2, tmp2, lsl #8
.endm

.macro BGR888toRGB888_process_head  cond, numbytes, firstreg, unaligned_src, unaligned_mask, preload
        pixld   cond, numbytes, firstreg, SRC, unaligned_src
.endm

.macro BGR888toRGB888_process_tail  cond, numbytes, firstreg
 .if numbytes >= 8
        BGR888toRGB888_2pixels cond, %(firstreg+0), %(firstreg+1), MASK, STRIDE_M
  .if numbytes == 16
        BGR888toRGB888_2pixels cond, %(firstreg+2), %(firstreg+3), MASK, STRIDE_M
  .endif
 .else @ numbytes == 4
        BGR888toRGB888_1pixel cond, %(firstreg+0), MASK
 .endif
.endm

generate_composite_function \
    Blit_BGR888_RGB888ARMSIMDAsm, 32, 0, 32, \
    FLAG_DST_WRITEONLY | FLAG_COND_EXEC | FLAG_PROCESS_PRESERVES_SCRATCH, \
    2, /* prefetch distance */ \
    nop_macro, /* init */ \
    nop_macro, /* newline */ \
    nop_macro, /* cleanup */ \
    BGR888toRGB888_process_head, \
    BGR888toRGB888_process_tail

/******************************************************************************/

.macro RGB444toRGB888_init
        ldr     MASK, =0x0f0f0f0f
        /* Set GE[3:0] to 0101 so SEL instructions do what we want */
        msr     CPSR_s, #0x50000
.endm

.macro RGB444toRGB888_1pixel reg, mask, tmp
        pkhbt   WK&reg, WK&reg, WK&reg, lsl #12      @ 0000aaaarrrrggggaaaarrrrggggbbbb
        and     WK&reg, mask, WK&reg                 @ 0000aaaa0000gggg0000rrrr0000bbbb
        orr     WK&reg, WK&reg, WK&reg, lsl #4       @ aaaaaaaaggggggggrrrrrrrrbbbbbbbb
        pkhtb   tmp, WK&reg, WK&reg, asr #8          @ aaaaaaaaggggggggggggggggrrrrrrrr
        pkhbt   WK&reg, WK&reg, WK&reg, lsl #8       @ ggggggggrrrrrrrrrrrrrrrrbbbbbbbb
        sel     WK&reg, WK&reg, tmp                  @ aaaaaaaarrrrrrrrggggggggbbbbbbbb
.endm

.macro RGB444toRGB888_2pixels in, out1, out2, mask, tmp1, tmp2
        and     tmp1, mask, WK&in                    @ 0000RRRR0000BBBB0000rrrr0000bbbb
        and     tmp2, mask, WK&in, lsr #4            @ 0000AAAA0000GGGG0000aaaa0000gggg
        orr     tmp1, tmp1, tmp1, lsl #4             @ RRRRRRRRBBBBBBBBrrrrrrrrbbbbbbbb
        orr     tmp2, tmp2, tmp2, lsl #4             @ AAAAAAAAGGGGGGGGaaaaaaaagggggggg
        pkhtb   WK&out2, tmp2, tmp1, asr #16         @ AAAAAAAAGGGGGGGGRRRRRRRRBBBBBBBB
        pkhbt   WK&out1, tmp1, tmp2, lsl #16         @ aaaaaaaaggggggggrrrrrrrrbbbbbbbb
        pkhtb   tmp2, WK&out2, WK&out2, asr #8       @ AAAAAAAAGGGGGGGGGGGGGGGGRRRRRRRR
        pkhtb   tmp1, WK&out1, WK&out1, asr #8       @ aaaaaaaaggggggggggggggggrrrrrrrr
        pkhbt   WK&out1, WK&out1, WK&out1, lsl #8    @ ggggggggrrrrrrrrrrrrrrrrbbbbbbbb
        pkhbt   WK&out2, WK&out2, WK&out2, lsl #8    @ GGGGGGGGRRRRRRRRRRRRRRRRBBBBBBBB
        sel     WK&out1, WK&out1, tmp1               @ aaaaaaaarrrrrrrrggggggggbbbbbbbb
        sel     WK&out2, WK&out2, tmp2               @ AAAAAAAARRRRRRRRGGGGGGGGBBBBBBBB
.endm

.macro RGB444toRGB888_process_head  cond, numbytes, firstreg, unaligned_src, unaligned_mask, preload
        pixld   cond, numbytes/2, firstreg, SRC, unaligned_src
.endm

.macro RGB444toRGB888_process_tail  cond, numbytes, firstreg
 .if numbytes >= 8
  .if numbytes == 16
        RGB444toRGB888_2pixels %(firstreg+1), %(firstreg+2), %(firstreg+3), MASK, STRIDE_M, SCRATCH
  .endif
        RGB444toRGB888_2pixels %(firstreg+0), %(firstreg+0), %(firstreg+1), MASK, STRIDE_M, SCRATCH
 .else @ numbytes == 4
        RGB444toRGB888_1pixel %(firstreg+0), MASK, SCRATCH
 .endif
.endm

generate_composite_function \
    Blit_RGB444_RGB888ARMSIMDAsm, 16, 0, 32, \
    FLAG_DST_WRITEONLY | FLAG_BRANCH_OVER, \
    2, /* prefetch distance */ \
    RGB444toRGB888_init, \
    nop_macro, /* newline */ \
    nop_macro, /* cleanup */ \
    RGB444toRGB888_process_head, \
    RGB444toRGB888_process_tail
