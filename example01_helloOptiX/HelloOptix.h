#pragma once

#include <iostream>

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include "../common/Util.h"

namespace osc {

inline auto getCudaDeviceCount() {
  auto Result = 0;

  CUDA_CHECK(cudaGetDeviceCount(&Result));

  return Result;
}

inline auto init() {
  // CUDAを実行可能なデバイスがあるかチェックします。

  if (getCudaDeviceCount() == 0) {
    throw std::runtime_error("#osc: no CUDA capable devices found!");
  }

  // OptiXを初期化します。

  OPTIX_CHECK(optixInit());
}

inline auto helloOptiX() {
  std::cout << "#osc: initializing optix...\n";
  osc::init();
  std::cout << "#osc: successfully initialized optix... yay!\n";

  // for this simple hello-world example, don't do anything else
  // ...

  std::cout << "#osc: done. clean exit.\n";
}

} // namespace osc
