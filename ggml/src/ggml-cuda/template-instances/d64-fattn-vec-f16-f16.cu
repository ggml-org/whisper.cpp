// consigliere patch 0002 (GGML_CUDA_FA_D64_F16_ONLY): the head-dim-64 F16/F16 case of the upstream
// fattn-vec-instance-f16-f16.cu, which also holds head dims 128 and 256. The d64- prefix keeps this
// file out of every upstream glob.

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(64, GGML_TYPE_F16, GGML_TYPE_F16);
