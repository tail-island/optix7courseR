#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace osc {

struct LaunchParams {
  float4 *imageBuffer;
  int frameId;
};

} // namespace osc
