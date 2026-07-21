#pragma once

#include <functional>
#include <utility>
#include <vector>

namespace tts_cpp::lavasr::dsp {

// Per-chunk output planes (real, imag), each [L * F], for chunk index c.
using ChunkPlanes = std::function<std::pair<const float *, const float *>(int c)>;

// Stitch per-chunk core outputs back onto the [T][F] grid: plain copy of chunk 0's
// first T frames when T<=L, else squared-Hann weighted overlap-add + normalize.
void overlap_add_normalize(const std::vector<int> & starts, int T, int F, int L,
                           const ChunkPlanes & chunk_out,
                           std::vector<float> & outRe, std::vector<float> & outIm);

} // namespace tts_cpp::lavasr::dsp
