#pragma once

#include <cstdint>

#include <cuda_runtime.h>
#include <Eigen/Core>
#include <optix.h>

namespace osc {

enum class RayType {
  Radiance,
  Size
};

struct Camera {
  float3 origin;
  float3 u;
  float3 v;
  float3 w;
};

struct LaunchParams {
  float4 *imageBuffer;

  Camera camera;

  OptixTraversableHandle traversableHandle;
};

} // namespace osc
