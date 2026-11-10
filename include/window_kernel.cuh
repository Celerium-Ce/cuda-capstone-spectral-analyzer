// Copyright 2026 Celerium-Ce
#ifndef INCLUDE_WINDOW_KERNEL_CUH_
#define INCLUDE_WINDOW_KERNEL_CUH_

#include <cuda_runtime.h>

// Slices `num_frames` overlapping windows of length `frame_size` out of
// `d_signal` (`hop_size` samples apart, zero-padding past the end of the
// signal) and multiplies each by a Hann window, writing the result into
// `d_frames` (row-major, `num_frames` x `frame_size`) ready for a batched
// real-to-complex FFT. Runs on `stream`.
void LaunchWindowKernel(const float* d_signal, int signal_len, int frame_size,
                         int hop_size, int num_frames, float* d_frames,
                         cudaStream_t stream);

#endif  // INCLUDE_WINDOW_KERNEL_CUH_
