// Copyright 2026 Celerium-Ce
#include "include/magnitude_kernel.cuh"

#include "include/cuda_utils.h"

namespace {

constexpr int kBlockDim = 256;

// Grid-stride loop over the flattened num_frames x num_bins matrix: at
// this project's scale (tens of thousands of bins per file) a single
// launch with a modest number of blocks comfortably covers every element
// without needing a 2D launch configuration.
__global__ void MagnitudeKernel(const cufftComplex* __restrict__ spectrum,
                                 int total_bins,
                                 float* __restrict__ magnitude_db) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_bins;
       i += blockDim.x * gridDim.x) {
    const cufftComplex bin = spectrum[i];
    const float magnitude = sqrtf(bin.x * bin.x + bin.y * bin.y);
    magnitude_db[i] = 20.0f * log10f(magnitude + 1e-6f);
  }
}

}  // namespace

void LaunchMagnitudeKernel(const cufftComplex* d_spectrum, int num_frames,
                            int num_bins, float* d_magnitude_db,
                            cudaStream_t stream) {
  const int total_bins = num_frames * num_bins;
  const int grid = (total_bins + kBlockDim - 1) / kBlockDim;
  MagnitudeKernel<<<grid, kBlockDim, 0, stream>>>(d_spectrum, total_bins,
                                                   d_magnitude_db);
  CUDA_CHECK(cudaGetLastError());
}
