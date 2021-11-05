#pragma once

#include <cstdlib>
#include <iostream>
#include <string>

#include <cuda_runtime.h>
#include <optix.h>

#define CUDA_CHECK(call)                                                                                                                       \
  {                                                                                                                                            \
    if (call != cudaSuccess) {                                                                                                                 \
      auto error = cudaGetLastError();                                                                                                         \
      std::cerr << "CUDA call (" << #call << ") failed. " << cudaGetErrorName(error) << " (" << cudaGetErrorString(error) << ")" << std::endl; \
      std::exit(2);                                                                                                                            \
    }                                                                                                                                          \
  }

#define CU_CHECK(call)                                                        \
  {                                                                           \
    if (call != CUDA_SUCCESS) {                                               \
      std::cerr << "CU call (" << #call << ") failed. " << call << std::endl; \
      std::exit(2);                                                           \
    }                                                                         \
  }

#define OPTIX_CHECK(call)                                                        \
  {                                                                              \
    if (call != OPTIX_SUCCESS) {                                                 \
      std::cerr << "Optix call (" << #call << ") failed. " << call << std::endl; \
      std::exit(2);                                                              \
    }                                                                            \
  }

namespace osc {
namespace common {

inline auto optixLogCallback(unsigned int level, const char *tag, const char *message, void *) noexcept {
  std::cerr << "[" << level << "][" << tag << "]: " << message << std::endl;
}

} // namespace common
} // namespace osc
