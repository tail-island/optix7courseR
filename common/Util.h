#pragma once

#include <cstdlib>
#include <iostream>
#include <string>

#include <cuda_runtime.h>
#include <optix.h>

namespace osc {

#define CUDA_CHECK(call)                                                                                                            \
  {                                                                                                                                 \
    if (call != cudaSuccess) {                                                                                                      \
      auto error = cudaGetLastError();                                                                                              \
      std::cerr << "CUDA call (" << #call << ") failed. " << cudaGetErrorName(error) << " (" << cudaGetErrorString(error) << ")\n"; \
      std::exit(2);                                                                                                                 \
    }                                                                                                                               \
  }

#define CU_CHECK(call)                                                   \
  {                                                                      \
    if (call != CUDA_SUCCESS) {                                          \
      std::cerr << "CU call (" << #call << ") failed. " << call << "\n"; \
      std::exit(2);                                                      \
    }                                                                    \
  }

#define OPTIX_CHECK(call)                                                   \
  {                                                                         \
    if (call != OPTIX_SUCCESS) {                                            \
      std::cerr << "Optix call (" << #call << ") failed. " << call << "\n"; \
      std::exit(2);                                                         \
    }                                                                       \
  }

inline auto optixLogCallback(unsigned int Level, const char *Tag, const char *Message, void *) noexcept {
  std::cerr << "[" << Level << "][" << Tag << "]: " << Message << "\n";
}

} // namespace osc