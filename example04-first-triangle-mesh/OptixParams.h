#pragma once

#include <cstdint>

#include <Eigen/Core>
#include <optix.h>

namespace osc {

enum class RayType {
  Radiance,
  Size
};

struct LaunchParams {
  std::uint32_t *imageBuffer;
  OptixTraversableHandle traversableHandle;
};

struct Camera {
  Eigen::Vector3f origin;
  Eigen::Vector3f u;
  Eigen::Vector3f v;
  Eigen::Vector3f w;
};

struct RaygenData {
  Camera camera;
};

} // namespace osc
