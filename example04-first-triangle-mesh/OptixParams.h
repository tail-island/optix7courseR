#pragma once

#include <cstdint>

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
  std::uint32_t *imageBuffer;

  Camera camera;

  OptixTraversableHandle traversableHandle;
};

} // namespace osc
