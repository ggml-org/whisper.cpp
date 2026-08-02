//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "ggml-impl.h"
#include "common.hpp"
#include "dequantize.hpp"
#include "quants.hpp"
#include "getrows.hpp"


// True if dst->src[0] has been in-place reordered by opt_for_reorder into a
// split layout. get_rows must use a reorder-aware dequantize in that case.
static bool ggml_sycl_get_rows_src0_reordered(const ggml_tensor * dst) {
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)dst->src[0]->extra;
    return extra && extra->optimized_feature.reorder;
}


// Reorder-aware get_rows for src0 that has been in-place reordered by
// opt_for_reorder into a split layout (qs region, then scales/d region).
// Standard k_get_rows reads (block_q_t*)src0_row which assumes interleaved
// d+qs per block; that is wrong after reorder. This kernel indexes the split
// layout directly using the GLOBAL block index (i01*nblocks_per_row + ib) via
// block_q_t<type>::get_block_offset / get_d_offset, then calls the per-element
// reorder dequantize. Covers Q1_0/Q4_0/Q8_0 (the types with a per-element
// dequantize_kernel_t_reorder). K-quants need block-cooperative dequantize and
// are handled separately.
template <ggml_type gtype, dequantize_kernel_t_reorder dq_reorder, typename dst_t>
static void k_get_rows_reorder(
            const void * src0, const int32_t * src1, dst_t * dst,
            int64_t ne00, int64_t ne01, /*int64_t ne02, int64_t ne03,*/
            /*int64_t ne10, int64_t ne11,*/ int64_t ne12, /*int64_t ne13,*/
            /*size_t s0,*/ size_t s1, size_t s2, size_t s3,
            /*size_t nb00,*/ size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            const sycl::nd_item<3> &item_ct1) {

    using block_type   = ggml_sycl_reordered::block_q_t<gtype>;
    using block_traits = typename block_type::traits;
    constexpr int qk = block_traits::qk;
    constexpr int qr = block_traits::qr;

    const int i00 = (item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2)) * 2;
    const int i10 = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1);
    const int i11 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) / ne12;
    const int i12 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) % ne12;

    if (i00 >= ne00) {
        return;
    }

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;

    // For 3D/4D src0 (ne02/ne03 > 1) we offset by the slice; nb02/nb03 describe
    // the per-slice byte offset of the START of that slice's split region.
    const char * slice = (const char *)src0 + i11*nb02 + i12*nb03;
    const int nblocks_per_row = ne00 / qk;
    const int total_nblocks  = ne01 * nblocks_per_row;

    const int ib  = i00 / qk;            // block index within the row
    const int iqs = (i00 % qk) / qr;     // quant index within block
    const int gblock = i01 * nblocks_per_row + ib;  // global block index

    // block_q_t gives the split-layout offsets for a given (global) block index.
    // qs_ptr = this block's qs start (get_block_offset already includes gblock).
    // d_ptr  = the d-SECTION start (block_index 0); dq_reorder adds gblock
    // internally to index the scale (see dequantize_q8_0_reorder: d = *(half*)(d_ptr+ib)).
    const auto bx_off = block_type::get_block_offset(gblock, total_nblocks);
    const auto d_sec  = block_type::get_d_offset(ne01, ne00, 0);
    const char * base = slice;
    const void * qs_ptr  = base + bx_off.first;
    const void * d_ptr   = base + d_sec.first;

    dfloat2 v;
    dq_reorder(d_ptr, gblock, qs_ptr, iqs, v);

    const int iybs = i00 - i00 % qk;
    const int y_offset = qr == 1 ? 1 : qk / 2;
    dst_row[iybs + iqs + 0] = v.x();
    dst_row[iybs + iqs + y_offset] = v.y();
}


template <typename dst_t>
static void k_get_rows_q4_K_reorder(
            const void * src0, const int32_t * src1, dst_t * dst,
            int64_t ne00, int64_t ne01, int64_t ne12,
            size_t s1, size_t s2, size_t s3, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            uint8_t * scales_local, const sycl::nd_item<3> &item_ct1) {
#if QK_K == 256
    const int64_t ib_row = item_ct1.get_group(2);   // block index within row
    const int64_t i10    = item_ct1.get_group(1);   // batch
    const int64_t tid_local = item_ct1.get_local_id(2);
    const int64_t il = tid_local / 8;
    const int64_t ir = tid_local % 8;

    const int64_t i11 = i10 / ne12;
    const int64_t i12 = i10 % ne12;
    const int64_t i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
    dst_t * y = dst_row + ib_row * QK_K + 64 * il + 4 * ir;

    const char * slice = (const char *)src0 + i11*nb02 + i12*nb03;
    const int nblocks_per_row = ne00 / QK_K;
    const int total_nblocks  = ne01 * nblocks_per_row;
    const int gblock = i01 * nblocks_per_row + ib_row;  // global block index

    const uint8_t * base = (const uint8_t *)slice;
    const size_t qs_offset     = (size_t)gblock * (QK_K / 2);
    const size_t scales_offset = (size_t)total_nblocks * (QK_K / 2) + (size_t)gblock * K_SCALE_SIZE;
    const size_t dm_offset     = (size_t)total_nblocks * (QK_K / 2) + (size_t)total_nblocks * K_SCALE_SIZE + (size_t)gblock * sizeof(ggml_half2);
    const uint8_t * qs_ptr     = base + qs_offset;
    const uint8_t * scales_ptr = base + scales_offset;
    const ggml_half2 dm_values = *reinterpret_cast<const ggml_half2 *>(base + dm_offset);
    const float dall = dm_values.x();
    const float dmin = dm_values.y();

    if (tid_local < 12) scales_local[tid_local] = scales_ptr[tid_local];
    item_ct1.barrier(sycl::access::fence_space::local_space);
    dequantize_q4_K_common(y, qs_ptr, dall, dmin, scales_local, (int)il, (int)ir);
#else
    GGML_UNUSED(src0);GGML_UNUSED(src1);GGML_UNUSED(dst);GGML_UNUSED(ne00);GGML_UNUSED(ne01);
    GGML_UNUSED(ne12);GGML_UNUSED(s1);GGML_UNUSED(s2);GGML_UNUSED(s3);GGML_UNUSED(nb02);GGML_UNUSED(nb03);
    GGML_UNUSED(s10);GGML_UNUSED(s11);GGML_UNUSED(s12);GGML_UNUSED(scales_local);GGML_UNUSED(item_ct1);
    GGML_ABORT("Q4_K reorder get_rows not supported for QK_K != 256");
#endif
}

template <typename dst_t>
static void k_get_rows_q5_K_reorder(
            const void * src0, const int32_t * src1, dst_t * dst,
            int64_t ne00, int64_t ne01, int64_t ne12,
            size_t s1, size_t s2, size_t s3, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            uint8_t * scales_local, const sycl::nd_item<3> &item_ct1) {
#if QK_K == 256
    const int64_t ib_row = item_ct1.get_group(2);
    const int64_t i10    = item_ct1.get_group(1);
    const int64_t tid    = item_ct1.get_local_id(2);
    const int64_t il = tid / 16;
    const int64_t ir = tid % 16;
    const int64_t is = 2 * il;

    const int64_t i11 = i10 / ne12;
    const int64_t i12 = i10 % ne12;
    const int64_t i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
    dst_t * y = dst_row + ib_row * QK_K + 64 * il + 2 * ir;

    const char * slice = (const char *)src0 + i11*nb02 + i12*nb03;
    const int nblocks_per_row = ne00 / QK_K;
    const int total_nblocks  = ne01 * nblocks_per_row;
    const int gblock = i01 * nblocks_per_row + ib_row;

    const uint8_t * base = (const uint8_t *)slice;
    const size_t qs_offset     = (size_t)gblock * (QK_K / 2);
    const size_t qh_offset     = (size_t)total_nblocks * (QK_K / 2) + (size_t)gblock * (QK_K / 8);
    const size_t scales_offset = (size_t)total_nblocks * (QK_K / 2) + (size_t)total_nblocks * (QK_K / 8) + (size_t)gblock * K_SCALE_SIZE;
    const size_t dm_offset     = (size_t)total_nblocks * (QK_K / 2) + (size_t)total_nblocks * (QK_K / 8) + (size_t)total_nblocks * K_SCALE_SIZE + (size_t)gblock * sizeof(ggml_half2);
    const uint8_t * qs_ptr     = base + qs_offset;
    const uint8_t * qh_ptr     = base + qh_offset;
    const uint8_t * scales_ptr = base + scales_offset;
    const ggml_half2 dm_values = *reinterpret_cast<const ggml_half2 *>(base + dm_offset);
    const float dall = dm_values.x();
    const float dmin = dm_values.y();

    const uint8_t * ql = qs_ptr + 32 * il + 2 * ir;
    const uint8_t * qh = qh_ptr + 2 * ir;
    if (tid < K_SCALE_SIZE) scales_local[tid] = scales_ptr[tid];
    item_ct1.barrier(sycl::access::fence_space::local_space);

    uint8_t sc, m;
    get_scale_min_k4(is + 0, scales_local, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, scales_local, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;

    uint8_t hm = 1 << (2 * il);
    y[ 0] = d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1;
    y[ 1] = d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1;
    hm <<= 1;
    y[32] = d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2;
    y[33] = d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2;
#else
    GGML_UNUSED(src0);GGML_UNUSED(src1);GGML_UNUSED(dst);GGML_UNUSED(ne00);GGML_UNUSED(ne01);
    GGML_UNUSED(ne12);GGML_UNUSED(s1);GGML_UNUSED(s2);GGML_UNUSED(s3);GGML_UNUSED(nb02);GGML_UNUSED(nb03);
    GGML_UNUSED(s10);GGML_UNUSED(s11);GGML_UNUSED(s12);GGML_UNUSED(scales_local);GGML_UNUSED(item_ct1);
    GGML_ABORT("Q5_K reorder get_rows not supported for QK_K != 256");
#endif
}

template <typename dst_t>
static void k_get_rows_q3_K_reorder(
            const void * src0, const int32_t * src1, dst_t * dst,
            int64_t ne00, int64_t ne01, int64_t ne12,
            size_t s1, size_t s2, size_t s3, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            const sycl::nd_item<3> &item_ct1) {
#if QK_K == 256
    const int64_t ib_row = item_ct1.get_group(2);
    const int64_t i10    = item_ct1.get_group(1);
    const int64_t tid    = item_ct1.get_local_id(2);

    const int64_t i11 = i10 / ne12;
    const int64_t i12 = i10 % ne12;
    const int64_t i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;

    const char * slice = (const char *)src0 + i11*nb02 + i12*nb03;
    const int nblocks_per_row = ne00 / QK_K;
    const int total_nblocks  = ne01 * nblocks_per_row;
    const int gblock = i01 * nblocks_per_row + ib_row;

    const uint8_t * base = (const uint8_t *)slice;
    const size_t qs_offset     = (size_t)gblock * (QK_K / 4);
    const size_t hmask_offset  = (size_t)total_nblocks * (QK_K / 4) + (size_t)gblock * (QK_K / 8);
    const size_t scales_offset = (size_t)total_nblocks * (QK_K / 4) + (size_t)total_nblocks * (QK_K / 8) + (size_t)gblock * 12;
    const size_t d_offset      = (size_t)total_nblocks * (QK_K / 4) + (size_t)total_nblocks * (QK_K / 8) + (size_t)total_nblocks * 12 + (size_t)gblock * sizeof(ggml_half);
    const uint8_t * qs     = base + qs_offset;
    const uint8_t * hmask  = base + hmask_offset;
    const uint8_t * scales = base + scales_offset;
    const float     d_all  = static_cast<float>(*reinterpret_cast<const ggml_half *>(base + d_offset));

    const int64_t r    = tid / 4;
    const int64_t tid4 = r / 2;
    const int64_t is0  = r % 2;
    const int64_t l0   = 16 * is0 + 4 * (tid % 4);
    const int64_t n    = tid4 / 4;
    const int64_t j    = tid4 - 4 * n;
    const int64_t is   = 8 * n + 2 * j + is0;
    const int     shift = 2 * j;
    uint8_t       m    = 1 << (4 * n + j);

    uint8_t us = is < 4
        ? (scales[is - 0] & 0xF) | (((scales[is + 8] >> 0) & 3) << 4)
        : is < 8
            ? (scales[is - 0] & 0xF) | (((scales[is + 4] >> 2) & 3) << 4)
            : is < 12
                ? (scales[is - 8] >> 4) | (((scales[is + 0] >> 4) & 3) << 4)
                : (scales[is - 8] >> 4) | (((scales[is - 4] >> 6) & 3) << 4);
    const float dl = d_all * (us - 32);

    dst_t * y = dst_row + ib_row * QK_K + 128 * n + 32 * j;
    const uint8_t * q  = qs + 32 * n;
    const uint8_t * hm = hmask;
    for (int l = l0; l < l0 + 4; ++l) {
        y[l] = dl * ((int8_t) ((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
    }
#else
    GGML_UNUSED(src0);GGML_UNUSED(src1);GGML_UNUSED(dst);GGML_UNUSED(ne00);GGML_UNUSED(ne01);
    GGML_UNUSED(ne12);GGML_UNUSED(s1);GGML_UNUSED(s2);GGML_UNUSED(s3);GGML_UNUSED(nb02);GGML_UNUSED(nb03);
    GGML_UNUSED(s10);GGML_UNUSED(s11);GGML_UNUSED(s12);GGML_UNUSED(item_ct1);
    GGML_ABORT("Q3_K reorder get_rows not supported for QK_K != 256");
#endif
}

template <typename dst_t>
static void k_get_rows_q6_K_reorder(
            const void * src0, const int32_t * src1, dst_t * dst,
            int64_t ne00, int64_t ne01, int64_t ne12,
            size_t s1, size_t s2, size_t s3, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            const sycl::nd_item<3> &item_ct1) {
#if QK_K == 256
    const int64_t ib_row = item_ct1.get_group(2);
    const int64_t i10    = item_ct1.get_group(1);
    const int64_t tid    = item_ct1.get_local_id(2);
    const int64_t ip  = tid / 32;
    const int64_t il  = tid - 32 * ip;
    const int64_t is  = 8 * ip + il / 16;

    const int64_t i11 = i10 / ne12;
    const int64_t i12 = i10 % ne12;
    const int64_t i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
    dst_t * y = dst_row + ib_row * QK_K + 128 * ip + il;

    const char * slice = (const char *)src0 + i11*nb02 + i12*nb03;
    const int nblocks_per_row = ne00 / QK_K;
    const int total_nblocks  = ne01 * nblocks_per_row;
    const int gblock = i01 * nblocks_per_row + ib_row;

    const uint8_t * base = (const uint8_t *)slice;
    const size_t ql_offset          = (size_t)gblock * (QK_K / 2);
    const size_t qh_offset         = (size_t)total_nblocks * (QK_K / 2) + (size_t)gblock * (QK_K / 4);
    const size_t base_scales_offset = (size_t)total_nblocks * (QK_K / 2) + (size_t)total_nblocks * (QK_K / 4) + (size_t)gblock * (QK_K / 16);
    const size_t base_d_offset      = ((QK_K / 2) + (QK_K / 4) + (QK_K / 16)) * (size_t)total_nblocks;
    const uint8_t * ql_ptr     = base + ql_offset;
    const uint8_t * qh_ptr     = base + qh_offset;
    const uint8_t * scales_ptr = base + base_scales_offset;
    const ggml_half * d = (const ggml_half *)(base + base_d_offset) + gblock;

    const uint8_t * ql = ql_ptr + 64 * ip + il;
    const uint8_t   qh = *(qh_ptr + 32 * ip + il);
    const int8_t *  sc = reinterpret_cast<const int8_t *>(scales_ptr + is);

    y[0]  = *d * sc[0] * ((int8_t) ((ql[0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = *d * sc[2] * ((int8_t) ((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = *d * sc[4] * ((int8_t) ((ql[0] >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = *d * sc[6] * ((int8_t) ((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32);
#else
    GGML_UNUSED(src0);GGML_UNUSED(src1);GGML_UNUSED(dst);GGML_UNUSED(ne00);GGML_UNUSED(ne01);
    GGML_UNUSED(ne12);GGML_UNUSED(s1);GGML_UNUSED(s2);GGML_UNUSED(s3);GGML_UNUSED(nb02);GGML_UNUSED(nb03);
    GGML_UNUSED(s10);GGML_UNUSED(s11);GGML_UNUSED(s12);GGML_UNUSED(item_ct1);
    GGML_ABORT("Q6_K reorder get_rows not supported for QK_K != 256");
#endif
}


template<int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void k_get_rows(
            const void * src0, const int32_t * src1, dst_t * dst,
            int64_t ne00, /*int64_t ne01, int64_t ne02, int64_t ne03,*/
            /*int64_t ne10, int64_t ne11,*/ int64_t ne12, /*int64_t ne13,*/
            /*size_t s0,*/ size_t s1, size_t s2, size_t s3,
            /*size_t nb00,*/ size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            const sycl::nd_item<3> &item_ct1/*, size_t s13*/) {

    const int i00 = (item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2)) *
                    2;
    const int i10 = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1);
    const int i11 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) /
                    ne12;
    const int i12 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) %
                    ne12;

    if (i00 >= ne00) {
        return;
    }

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
    const void * src0_row = (const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03;

    const int ib = i00/qk; // block index
    const int iqs = (i00%qk)/qr; // quant index
    const int iybs = i00 - i00%qk; // dst block start index
    const int y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(src0_row, ib, iqs, v);

    dst_row[iybs + iqs + 0] = v.x();
    dst_row[iybs + iqs + y_offset] = v.y();
}

template<typename src0_t, typename dst_t>
static void k_get_rows_float(
            const src0_t * src0, const int32_t * src1, dst_t * dst,
            int64_t ne00, /*int64_t ne01, int64_t ne02, int64_t ne03,*/
            /*int64_t ne10, int64_t ne11,*/ int64_t ne12, /*int64_t ne13,*/
            /*size_t s0,*/ size_t s1, size_t s2, size_t s3,
            /*size_t nb00,*/ size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            const sycl::nd_item<3> &item_ct1/*, size_t s13*/) {

    const int i00 = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2);
    const int i10 = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1);
    const int i11 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) /
                    ne12;
    const int i12 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) %
                    ne12;

    if (i00 >= ne00) {
        return;
    }

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
    const src0_t * src0_row = (const src0_t *)((const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03);

    dst_row[i00] = src0_row[i00];
}

template <int qk, int qr, dequantize_kernel_t dq>
static void get_rows_sycl(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                          ggml_tensor *dst, const void *src0_dd,
                          const int32_t *src1_dd, float *dst_dd,
                          queue_ptr stream) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const sycl::range<3> block_dims(1, 1, SYCL_GET_ROWS_BLOCK_SIZE);
    const int block_num_x = (ne00 + 2*SYCL_GET_ROWS_BLOCK_SIZE - 1) / (2*SYCL_GET_ROWS_BLOCK_SIZE);
    const sycl::range<3> block_nums(ne11 * ne12, ne10, block_num_x);

    // strides in elements
    //const size_t s0 = nb0 / ggml_element_size(dst);
    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);
    //const size_t s13 = nb13 / ggml_element_size(src1);

    GGML_ASSERT(ne00 % 2 == 0);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             k_get_rows<qk, qr, dq>(
                                 src0_dd, src1_dd, dst_dd, ne00, ne12, s1, s2,
                                 s3, nb01, nb02, nb03, s10, s11, s12, item_ct1);
                         });

    GGML_UNUSED(dst);
    GGML_UNUSED(ctx);
}

// Reorder-aware get_rows for src0 split into [qs|scales|d] by opt_for_reorder.
// Covers Q1_0/Q4_0/Q8_0 (types with a per-element dequantize_kernel_t_reorder).
template <ggml_type gtype, dequantize_kernel_t_reorder dq_reorder>
static void get_rows_sycl_reorder(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                          ggml_tensor *dst, const void *src0_dd,
                          const int32_t *src1_dd, float *dst_dd,
                          queue_ptr stream) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const sycl::range<3> block_dims(1, 1, SYCL_GET_ROWS_BLOCK_SIZE);
    const int block_num_x = (ne00 + 2*SYCL_GET_ROWS_BLOCK_SIZE - 1) / (2*SYCL_GET_ROWS_BLOCK_SIZE);
    const sycl::range<3> block_nums(ne11 * ne12, ne10, block_num_x);

    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);

    GGML_ASSERT(ne00 % 2 == 0);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             k_get_rows_reorder<gtype, dq_reorder, float>(
                                 src0_dd, src1_dd, dst_dd, ne00, ne01, ne12, s1, s2,
                                 s3, nb01, nb02, nb03, s10, s11, s12, item_ct1);
                         });

    GGML_UNUSED(dst);
    GGML_UNUSED(ctx);
}

// K-quant reorder get_rows launchers. grid = (1, n11*n12, nblocks_per_row);
// work-group size = 32 (Q4_K) / 64 (Q5_K, Q6_K) / 128 (Q3_K). Each group handles
// one (batch, block_in_row).
#define GGML_SYCL_GET_ROWS_K_LOCALS(ne00_, ne01_, ne12_, s1_, s2_, s3_, nb02_, nb03_, s10_, s11_, s12_) \
    const int64_t ne00 = ne00_; const int64_t ne01 = ne01_; const int64_t ne12 = ne12_; \
    const size_t s1 = s1_; const size_t s2 = s2_; const size_t s3 = s3_; \
    const size_t nb02 = nb02_; const size_t nb03 = nb03_; \
    const size_t s10 = s10_; const size_t s11 = s11_; const size_t s12 = s12_;

static void get_rows_sycl_q4_K_reorder(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                          ggml_tensor *dst, const void *src0_dd,
                          const int32_t *src1_dd, float *dst_dd, queue_ptr stream) {
    GGML_TENSOR_BINARY_OP_LOCALS
    const int nblocks_per_row = ne00 / QK_K;
    const sycl::range<3> block_dims(1, 1, 32);
    const sycl::range<3> block_nums(1, ne11 * ne12, nblocks_per_row);
    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);
    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> scale_acc(sycl::range<1>(12), cgh);
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                k_get_rows_q4_K_reorder<float>(src0_dd, src1_dd, dst_dd, ne00, ne01, ne12,
                    s1, s2, s3, nb02, nb03, s10, s11, s12, get_pointer(scale_acc), item_ct1);
            });
    });
    GGML_UNUSED(dst); GGML_UNUSED(ctx);
}

static void get_rows_sycl_q5_K_reorder(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                          ggml_tensor *dst, const void *src0_dd,
                          const int32_t *src1_dd, float *dst_dd, queue_ptr stream) {
    GGML_TENSOR_BINARY_OP_LOCALS
    const int nblocks_per_row = ne00 / QK_K;
    const sycl::range<3> block_dims(1, 1, 64);
    const sycl::range<3> block_nums(1, ne11 * ne12, nblocks_per_row);
    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);
    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> scale_acc(sycl::range<1>(K_SCALE_SIZE), cgh);
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                k_get_rows_q5_K_reorder<float>(src0_dd, src1_dd, dst_dd, ne00, ne01, ne12,
                    s1, s2, s3, nb02, nb03, s10, s11, s12, get_pointer(scale_acc), item_ct1);
            });
    });
    GGML_UNUSED(dst); GGML_UNUSED(ctx);
}

static void get_rows_sycl_q3_K_reorder(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                          ggml_tensor *dst, const void *src0_dd,
                          const int32_t *src1_dd, float *dst_dd, queue_ptr stream) {
    GGML_TENSOR_BINARY_OP_LOCALS
    const int nblocks_per_row = ne00 / QK_K;
    const sycl::range<3> block_dims(1, 1, 64);
    const sycl::range<3> block_nums(1, ne11 * ne12, nblocks_per_row);
    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);
    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            k_get_rows_q3_K_reorder<float>(src0_dd, src1_dd, dst_dd, ne00, ne01, ne12,
                s1, s2, s3, nb02, nb03, s10, s11, s12, item_ct1);
        });
    GGML_UNUSED(dst); GGML_UNUSED(ctx);
}

static void get_rows_sycl_q6_K_reorder(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                          ggml_tensor *dst, const void *src0_dd,
                          const int32_t *src1_dd, float *dst_dd, queue_ptr stream) {
    GGML_TENSOR_BINARY_OP_LOCALS
    const int nblocks_per_row = ne00 / QK_K;
    const sycl::range<3> block_dims(1, 1, 64);
    const sycl::range<3> block_nums(1, ne11 * ne12, nblocks_per_row);
    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);
    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            k_get_rows_q6_K_reorder<float>(src0_dd, src1_dd, dst_dd, ne00, ne01, ne12,
                s1, s2, s3, nb02, nb03, s10, s11, s12, item_ct1);
        });
    GGML_UNUSED(dst); GGML_UNUSED(ctx);
}

template <typename src0_t, typename dst_t>
static void get_rows_sycl_float(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                const ggml_tensor *src1, ggml_tensor *dst,
                                const src0_t *src0_dd, const int32_t *src1_dd,
                                dst_t *dst_dd, queue_ptr stream) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const sycl::range<3> block_dims(1, 1, SYCL_GET_ROWS_BLOCK_SIZE);
    const int block_num_x = (ne00 + SYCL_GET_ROWS_BLOCK_SIZE - 1) / SYCL_GET_ROWS_BLOCK_SIZE;
    const sycl::range<3> block_nums(ne11 * ne12, ne10, block_num_x);

    // strides in elements
    //const size_t s0 = nb0 / ggml_element_size(dst);
    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);
    //const size_t s13 = nb13 / ggml_element_size(src1);

    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                k_get_rows_float(src0_dd, src1_dd, dst_dd, ne00, ne12, s1, s2,
                                 s3, nb01, nb02, nb03, s10, s11, s12, item_ct1);
            });
    }

    GGML_UNUSED(dst);
    GGML_UNUSED(ctx);
}

void ggml_sycl_op_get_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_I32 );

    GGML_ASSERT(dst->src[0]->nb[0] == ggml_type_size(dst->src[0]->type));
    GGML_ASSERT(dst->src[1]->nb[0] == ggml_type_size(dst->src[1]->type));
    GGML_ASSERT(dst->nb[0] == ggml_type_size(dst->type));

    const int32_t * src1_i32 = (const int32_t *) dst->src[1]->data;
    /* TODO: Refactor and remove duplicates */
    switch (dst->src[0]->type) {
        case GGML_TYPE_F16:
            get_rows_sycl_float(ctx, dst->src[0], dst->src[1], dst, (const sycl::half *)dst->src[0]->data,
                                src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_BF16:
            get_rows_sycl_float(ctx, dst->src[0], dst->src[1], dst, (const sycl::ext::oneapi::bfloat16 *)dst->src[0]->data,
                                src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_F32:
            get_rows_sycl_float(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_I32:
            get_rows_sycl_float(ctx, dst->src[0], dst->src[1], dst, (const int32_t *)dst->src[0]->data,
            src1_i32, (int32_t *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q1_0:
            if (ggml_sycl_get_rows_src0_reordered(dst)) {
                GGML_ABORT("get_rows reorder for Q1_0 not yet implemented (block_q_t<GGML_TYPE_Q1_0> is unspecialized)");
            }
            get_rows_sycl<QK1_0, 1, dequantize_q1_0>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_MXFP4:
            get_rows_sycl<QK_MXFP4, 2, dequantize_mxfp4>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_NVFP4:
            get_rows_sycl<QK_NVFP4, 1, dequantize_nvfp4>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_IQ2_XXS:
            get_rows_sycl<QK_K, 1, dequantize_iq2_xxs>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_IQ2_XS:
            get_rows_sycl<QK_K, 1, dequantize_iq2_xs>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_IQ2_S:
            get_rows_sycl<QK_K, 1, dequantize_iq2_s>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_IQ3_XXS:
            get_rows_sycl<QK_K, 1, dequantize_iq3_xxs>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_IQ1_S:
            get_rows_sycl<QK_K, 1, dequantize_iq1_s>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_IQ1_M:
            get_rows_sycl<QK_K, 1, dequantize_iq1_m>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_IQ3_S:
            get_rows_sycl<QK_K, 1, dequantize_iq3_s>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_IQ4_NL:
            get_rows_sycl<QK4_NL, 1, dequantize_iq4_nl>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_IQ4_XS:
            get_rows_sycl<QK_K, 1, dequantize_iq4_xs>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q2_K:
            get_rows_sycl<QK_K, 1, dequantize_q2_K>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q3_K:
            if (ggml_sycl_get_rows_src0_reordered(dst)) {
                get_rows_sycl_q3_K_reorder(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
                src1_i32, (float *)dst->data, ctx.stream());
            } else {
                get_rows_sycl<QK_K, 1, dequantize_q3_K>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
                src1_i32, (float *)dst->data, ctx.stream());
            }
            break;
        case GGML_TYPE_Q4_0:
            if (ggml_sycl_get_rows_src0_reordered(dst)) {
                get_rows_sycl_reorder<GGML_TYPE_Q4_0, dequantize_q4_0_reorder>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
                src1_i32, (float *)dst->data, ctx.stream());
            } else {
                get_rows_sycl<QK4_0, QR4_0, dequantize_q4_0>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
                src1_i32, (float *)dst->data, ctx.stream());
            }
            break;
        case GGML_TYPE_Q4_1:
            get_rows_sycl<QK4_1, QR4_1, dequantize_q4_1>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q4_K:
            if (ggml_sycl_get_rows_src0_reordered(dst)) {
                get_rows_sycl_q4_K_reorder(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
                src1_i32, (float *)dst->data, ctx.stream());
            } else {
                get_rows_sycl<QK_K, 1, dequantize_q4_K>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
                src1_i32, (float *)dst->data, ctx.stream());
            }
            break;
        case GGML_TYPE_Q5_0:
            get_rows_sycl<QK5_0, QR5_0, dequantize_q5_0>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q5_1:
            get_rows_sycl<QK5_1, QR5_1, dequantize_q5_1>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q5_K:
            if (ggml_sycl_get_rows_src0_reordered(dst)) {
                get_rows_sycl_q5_K_reorder(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
                src1_i32, (float *)dst->data, ctx.stream());
            } else {
                get_rows_sycl<QK_K, 1, dequantize_q5_K>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
                src1_i32, (float *)dst->data, ctx.stream());
            }
            break;
        case GGML_TYPE_Q6_K:
            if (ggml_sycl_get_rows_src0_reordered(dst)) {
                get_rows_sycl_q6_K_reorder(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
                src1_i32, (float *)dst->data, ctx.stream());
            } else {
                get_rows_sycl<QK_K, 1, dequantize_q6_K>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
                src1_i32, (float *)dst->data, ctx.stream());
            }
            break;
        case GGML_TYPE_Q8_0:
            if (ggml_sycl_get_rows_src0_reordered(dst)) {
                get_rows_sycl_reorder<GGML_TYPE_Q8_0, dequantize_q8_0_reorder>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
                src1_i32, (float *)dst->data, ctx.stream());
            } else {
                get_rows_sycl<QK8_0, QR8_0, dequantize_q8_0>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
                src1_i32, (float *)dst->data, ctx.stream());
            }
            break;
        default:
            // TODO: k-quants
            GGML_LOG_ERROR("%s: unsupported type: %s\n", __func__, ggml_type_name(dst->src[0]->type));
            GGML_ABORT("fatal error");
    }
}
