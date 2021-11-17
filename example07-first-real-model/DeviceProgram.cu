#include <tuple>

#pragma nv_diag_suppress 20236

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <optix.h>
#include <optix_device.h>
#include <optix_stubs.h>

#include "OptixParams.h"

namespace osc {

extern "C" {
__constant__ LaunchParams optixLaunchParams;
}

// OptiXのペイロードはunsigned int×n個で扱いづらいので、構造体へのポインタに変換します。

inline __device__ auto getPayloadParams(void *payloadPointer) noexcept {
  auto p = reinterpret_cast<std::uint64_t>(payloadPointer);

  return std::make_tuple(static_cast<std::uint32_t>(p >> 32), static_cast<std::uint32_t>(p));
}

inline __device__ auto getPayloadPointer() noexcept {
  return reinterpret_cast<void *>(static_cast<std::uint64_t>(optixGetPayload_0()) << 32 | static_cast<std::uint64_t>(optixGetPayload_1()));
}

// 光を生成します。

extern "C" __global__ void __raygen__renderFrame() {
  const auto &x = optixGetLaunchIndex().x;
  const auto &y = optixGetLaunchIndex().y;

  auto &origin = *reinterpret_cast<Eigen::Vector3f *>(&optixLaunchParams.camera.origin);

  const auto &u = *reinterpret_cast<Eigen::Vector3f *>(&optixLaunchParams.camera.u);
  const auto &v = *reinterpret_cast<Eigen::Vector3f *>(&optixLaunchParams.camera.v);
  const auto &w = *reinterpret_cast<Eigen::Vector3f *>(&optixLaunchParams.camera.w);

  auto direction = ((static_cast<float>(x) / optixGetLaunchDimensions().x * 2 - 1) * u + (static_cast<float>(y) / optixGetLaunchDimensions().y * 2 - 1) * v + w).normalized();

  auto color = Eigen::Vector3f{0};

  auto [payloadParam0, payloadParam1] = getPayloadParams(&color);

  optixTrace(
      optixLaunchParams.traversableHandle,
      *reinterpret_cast<float3 *>(&origin),
      *reinterpret_cast<float3 *>(&direction),
      0.0f,                                // tmin
      1e20f,                               // tmax
      0.0f,                                // rayTime
      OptixVisibilityMask(255),            //
      OPTIX_RAY_FLAG_DISABLE_ANYHIT,       // rayFlags,
      static_cast<int>(RayType::Radiance), // SBToffset
      static_cast<int>(RayType::Size),     // SBTstride
      static_cast<int>(RayType::Radiance), // missSBTIndex
      payloadParam0,                       // ペイロードではunsigned intしか使えません……。
      payloadParam1);

  optixLaunchParams.imageBuffer[x + y * optixGetLaunchDimensions().x] = float4{color.x(), color.y(), color.z(), 1};
}

// 物体に光が衝突した場合の処理です。衝突判定は自動でやってくれるみたい。

extern "C" __global__ void __closesthit__radiance() {
  const auto &triangleMeshes = reinterpret_cast<HitgroupData *>(optixGetSbtDataPointer())->triangleMeshes;

  const auto &index = triangleMeshes.indices[optixGetPrimitiveIndex()];

  // ポリゴンの法線を取得します。

  const auto normal = [&] {
    const auto &vertex1 = triangleMeshes.vertices[index.x()];
    const auto &vertex2 = triangleMeshes.vertices[index.y()];
    const auto &vertex3 = triangleMeshes.vertices[index.z()];

    return (vertex2 - vertex1).cross(vertex3 - vertex1).normalized();
  }();

  // レイの向きを取得します。

  const auto rayDirection = [&] {
    auto result = optixGetWorldRayDirection();

    return *reinterpret_cast<Eigen::Vector3f *>(&result);
  }();

  // 光源とかはとりあえず考慮しないで、レイとポリゴンが垂直なほど明るくなるということで。カメラにライトが付いているとでも思って、納得してください……。

  *reinterpret_cast<Eigen::Vector3f *>(getPayloadPointer()) = triangleMeshes.color * (0.2 + 0.8 * std::abs(normal.dot(rayDirection)));
}

// 物体に光が衝突しそうな場合の処理？

extern "C" __global__ void __anyhit__radiance() {
  ; // とりあえず、なにもしません。
}

// トレースした光が物体に衝突しなかった場合の処理です。

extern "C" __global__ void __miss__radiance() {
  *reinterpret_cast<Eigen::Vector3f *>(getPayloadPointer()) = Eigen::Vector3f{1, 1, 1}; // とりあえず、背景は真っ白にします。
}

} // namespace osc
