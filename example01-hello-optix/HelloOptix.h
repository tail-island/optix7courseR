#pragma once

#include <iostream>

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include "../common/Util.h"

namespace osc {

inline auto getCudaDeviceCount() {
  auto result = 0;

  CUDA_CHECK(cudaGetDeviceCount(&result));

  return result;
}

inline auto init() {
  // CUDAを実行可能なデバイスがあるかチェックします。

  if (getCudaDeviceCount() == 0) {
    throw std::runtime_error("#osc: no CUDA capable devices found!");
  }

  // OptiXを初期化します。

  OPTIX_CHECK(optixInit());
}

inline auto helloOptix() {
  std::cout << "#osc: initializing optix...\n";
  osc::init();
  std::cout << "#osc: successfully initialized optix... yay!\n";

  // とりあえずは、OptiXの初期化まで。環境が正しく構築されているかを確認して終了とします。

  std::cout << "#osc: done. clean exit.\n";
}

} // namespace osc
