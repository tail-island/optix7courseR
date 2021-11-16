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

struct TriangleMeshes {
  Eigen::Vector3f *vertices;
  Eigen::Vector3f *normals;
  Eigen::Vector2f *textureCoordinates;
  Eigen::Vector3i *indices;

  bool hasTextureObject;
  cudaTextureObject_t textureObject;

  Eigen::Vector3f color;
};

struct HitgroupData {
  TriangleMeshes triangleMeshes;
};

} // namespace osc