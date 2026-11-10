// Copyright 2026 Celerium-Ce
#ifndef INCLUDE_CUDA_UTILS_H_
#define INCLUDE_CUDA_UTILS_H_

#include <cuda_runtime.h>
#include <cufft.h>

#include <cstdio>
#include <cstdlib>

// Aborts with a message identifying the call site if a CUDA runtime call
// fails. Used at every CUDA API call site so failures are caught close to
// their source instead of surfacing as a later, unrelated crash.
#define CUDA_CHECK(call)                                                 \
  do {                                                                   \
    cudaError_t cuda_check_status_ = (call);                             \
    if (cuda_check_status_ != cudaSuccess) {                             \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
              cudaGetErrorString(cuda_check_status_));                   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

// Same idea as CUDA_CHECK, but for cuFFT status codes (cuFFT has no
// cudaGetErrorString equivalent, so this just prints the numeric code).
#define CUFFT_CHECK(call)                                                \
  do {                                                                   \
    cufftResult cufft_check_status_ = (call);                            \
    if (cufft_check_status_ != CUFFT_SUCCESS) {                          \
      fprintf(stderr, "cuFFT error at %s:%d: status=%d\n", __FILE__,     \
              __LINE__, static_cast<int>(cufft_check_status_));          \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

#endif  // INCLUDE_CUDA_UTILS_H_
