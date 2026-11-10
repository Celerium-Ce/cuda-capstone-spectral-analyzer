// Copyright 2026 Celerium-Ce
#ifndef INCLUDE_MAGNITUDE_KERNEL_CUH_
#define INCLUDE_MAGNITUDE_KERNEL_CUH_

#include <cuda_runtime.h>
#include <cufft.h>

// Converts `num_frames` x `num_bins` complex FFT bins (row-major) into
// magnitude expressed in decibels (20 * log10(|z| + eps)), writing into
// `d_magnitude_db` of the same shape. Runs on `stream`.
void LaunchMagnitudeKernel(const cufftComplex* d_spectrum, int num_frames,
                            int num_bins, float* d_magnitude_db,
                            cudaStream_t stream);

#endif  // INCLUDE_MAGNITUDE_KERNEL_CUH_
