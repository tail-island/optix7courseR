#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace osc {

struct LaunchParams {
  float3 *imageBuffer;
  int frameId;
};

} // namespace osc
