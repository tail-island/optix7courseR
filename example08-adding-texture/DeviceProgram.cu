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

enum { // TODO: enum classに書き換える。
  SURFACE_RAY_TYPE = 0,
  RAY_TYPE_COUNT
};

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

  const auto &camera = reinterpret_cast<RaygenData *>(optixGetSbtDataPointer())->camera;

  auto origin = camera.origin;
  auto direction = ((static_cast<float>(x) / optixGetLaunchDimensions().x * 2 - 1) * camera.u + (static_cast<float>(y) / optixGetLaunchDimensions().y * 2 - 1) * camera.v + camera.w).normalized();

  auto color = Eigen::Vector3f{0};

  auto [payloadParam0, payloadParam1] = getPayloadParams(&color);

  optixTrace(
      optixLaunchParams.traversableHandle,
      *reinterpret_cast<float3 *>(&origin),
      *reinterpret_cast<float3 *>(&direction),
      0.0f,                          // tmin
      1e20f,                         // tmax
      0.0f,                          // rayTime
      OptixVisibilityMask(255),      //
      OPTIX_RAY_FLAG_DISABLE_ANYHIT, // rayFlags,
      SURFACE_RAY_TYPE,              // SBToffset
      RAY_TYPE_COUNT,                // SBTstride
      SURFACE_RAY_TYPE,              // missSBTIndex
      payloadParam0,                 // ペイロードではunsigned intしか使えません……。
      payloadParam1);

  const auto r = static_cast<int>(255.5 * color.x()); // intへのキャストは小数点以下切り捨てなので、255よりも少し大きい値を使用しました。
  const auto g = static_cast<int>(255.5 * color.y());
  const auto b = static_cast<int>(255.5 * color.z());

  optixLaunchParams.imageBuffer[x + y * optixGetLaunchDimensions().x] = r << 0 | g << 8 | b << 16 | 0xff000000;
}

// 物体に光が衝突した場合の処理です。衝突判定は自動でやってくれるみたい。

extern "C" __global__ void __closesthit__radiance() {
  const auto &triangleMeshes = reinterpret_cast<HitgroupData *>(optixGetSbtDataPointer())->triangleMeshes;

  const auto &index = triangleMeshes.indices[optixGetPrimitiveIndex()];

  const auto u = optixGetTriangleBarycentrics().x;
  const auto v = optixGetTriangleBarycentrics().y;

  // 法線を取得します。

  const auto triangleMeshNormal = [&] {
    return ((1 - u - v) * triangleMeshes.normals[index.x()] + u * triangleMeshes.normals[index.y()] + v * triangleMeshes.normals[index.z()]).normalized();
  }();

  // ポリゴンの色を取得します。

  const auto color = [&] {
    if (!triangleMeshes.hasTextureObject) {
      return triangleMeshes.color;
    }

    const auto textureCoordinate = (1 - u - v) * triangleMeshes.textureCoordinates[index.x()] + u * triangleMeshes.textureCoordinates[index.y()] + v * triangleMeshes.textureCoordinates[index.z()];
    const auto textureColor = tex2D<float4>(triangleMeshes.textureObject, textureCoordinate.x(), textureCoordinate.y());

    return Eigen::Vector3f{textureColor.x, textureColor.y, textureColor.z};
  }();

  // レイの向きを取得します。

  const auto rayDirection = [&] {
    auto result = optixGetWorldRayDirection();

    return *reinterpret_cast<Eigen::Vector3f *>(&result);
  }();

  // 光源とかはとりあえず考慮しないで、レイとポリゴンが垂直なほど明るくなるということで。カメラにライトが付いているとでも思って、納得してください……。

  *reinterpret_cast<Eigen::Vector3f *>(getPayloadPointer()) = color * (0.2 + 0.8 * std::fabs(triangleMeshNormal.dot(rayDirection)));
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
