// Copyright 2026 Celerium-Ce
#include "include/window_kernel.cuh"

#include "include/cuda_utils.h"

namespace {

// One block per frame, one thread per sample within the frame (the driver
// only ever calls this with frame_size <= 1024, the max threads per
// block, so every frame is handled by a single block with no inner loop).
__global__ void WindowKernel(const float* __restrict__ signal, int signal_len,
                              int frame_size, int hop_size,
                              float* __restrict__ frames) {
  const int frame = blockIdx.x;
  const int n = threadIdx.x;
  if (n >= frame_size) {
    return;
  }

  const int sample_idx = frame * hop_size + n;
  const float sample = (sample_idx < signal_len) ? signal[sample_idx] : 0.0f;

  constexpr float kPi = 3.14159265358979323846f;
  const float hann = 0.5f * (1.0f - cosf(2.0f * kPi * n / (frame_size - 1)));

  frames[frame * frame_size + n] = sample * hann;
}

}  // namespace

void LaunchWindowKernel(const float* d_signal, int signal_len, int frame_size,
                         int hop_size, int num_frames, float* d_frames,
                         cudaStream_t stream) {
  const dim3 block(frame_size);
  const dim3 grid(num_frames);
  WindowKernel<<<grid, block, 0, stream>>>(d_signal, signal_len, frame_size,
                                            hop_size, d_frames);
  CUDA_CHECK(cudaGetLastError());
}
