#pragma once

#include <cstdint>

#include <Eigen/Core>
#include <optix.h>

namespace osc {

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

struct TriangleMeshes {
  Eigen::Vector3f *vertices;
  Eigen::Vector3f *normals;
  Eigen::Vector2f *textureCoordinates;
  Eigen::Vector3i *indices;
  Eigen::Vector3f color;
  cudaTextureObject_t textureObject;
};

struct HitgroupData {
  TriangleMeshes triangleMeshes;
};

} // namespace osc
