// consigliere patch 0002 (GGML_CUDA_FA_D64_F16_ONLY): the head-dim-64 cases of the upstream
// fattn-mma-f16-instance-*.cu files, which mix head dims. The d64- prefix keeps this file out of
// every upstream glob, so a build with the option OFF never compiles these instances twice.

#include "../fattn-mma-f16.cuh"

DECL_FATTN_MMA_F16_CASE(64, 64, 16, 1);
DECL_FATTN_MMA_F16_CASE(64, 64, 16, 2);
DECL_FATTN_MMA_F16_CASE(64, 64, 16, 4);
DECL_FATTN_MMA_F16_CASE(64, 64, 1, 8);
DECL_FATTN_MMA_F16_CASE(64, 64, 2, 4);
DECL_FATTN_MMA_F16_CASE(64, 64, 2, 8);
DECL_FATTN_MMA_F16_CASE(64, 64, 32, 1);
DECL_FATTN_MMA_F16_CASE(64, 64, 32, 2);
DECL_FATTN_MMA_F16_CASE(64, 64, 4, 2);
DECL_FATTN_MMA_F16_CASE(64, 64, 4, 4);
DECL_FATTN_MMA_F16_CASE(64, 64, 4, 8);
DECL_FATTN_MMA_F16_CASE(64, 64, 64, 1);
DECL_FATTN_MMA_F16_CASE(64, 64, 8, 1);
DECL_FATTN_MMA_F16_CASE(64, 64, 8, 2);
DECL_FATTN_MMA_F16_CASE(64, 64, 8, 4);
DECL_FATTN_MMA_F16_CASE(64, 64, 8, 8);
