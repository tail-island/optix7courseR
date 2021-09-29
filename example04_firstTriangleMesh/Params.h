#pragma once

#include <cstdint>

#include <optix.h>
#include <Eigen/Core>

namespace osc {

struct LaunchParams {
  std::uint32_t *ImageBuffer;
  int Width;
  int Height;

  OptixTraversableHandle TraversableHandle;
};

struct Camera {
  Eigen::Vector3f Origin;
  Eigen::Vector3f U;
  Eigen::Vector3f V;
  Eigen::Vector3f W;
};

struct RaygenData {
  Camera Camera;
};

} // namespace osc
