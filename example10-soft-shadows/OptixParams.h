#pragma once

#include <cstdint>

#include <cuda_runtime.h>
#include <Eigen/Core>
#include <optix.h>

namespace osc {

enum class RayType {
  Radiance,
  Shadow,
  Size
};

struct Light {
  float3 origin;
  float3 u;
  float3 v;
  float3 power;
};

struct Camera {
  float3 origin;
  float3 u;
  float3 v;
  float3 w;
};

struct LaunchParams {
  float3 *imageBuffer;
  int frameId;

  Light light;
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

} // namespace osc
