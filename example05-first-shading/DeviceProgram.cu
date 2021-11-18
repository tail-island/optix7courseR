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

// レイを生成します。

extern "C" __global__ void __raygen__renderFrame() {
  const auto &x = optixGetLaunchIndex().x;
  const auto &y = optixGetLaunchIndex().y;

  // カメラの情報を取得します。

  auto &origin = *reinterpret_cast<Eigen::Vector3f *>(&optixLaunchParams.camera.origin);

  const auto &u = *reinterpret_cast<Eigen::Vector3f *>(&optixLaunchParams.camera.u);
  const auto &v = *reinterpret_cast<Eigen::Vector3f *>(&optixLaunchParams.camera.v);
  const auto &w = *reinterpret_cast<Eigen::Vector3f *>(&optixLaunchParams.camera.w);

  // レイの方向を計算します。

  auto direction = ((static_cast<float>(x) / optixGetLaunchDimensions().x * 2 - 1) * u + (static_cast<float>(y) / optixGetLaunchDimensions().y * 2 - 1) * v + w).normalized();

  // ピクセルの色を表現する変数を用意します。この値をoptixTraceして設定します。

  auto color = Eigen::Vector3f{0};
  auto [payloadParam0, payloadParam1] = getPayloadParams(&color);

  // optixTraceして、レイをトレースします。

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

  // optixTraceで設定されたcolorを使用して、イメージ・バッファーに値を設定します。

  optixLaunchParams.imageBuffer[x + y * optixGetLaunchDimensions().x] = float3{color.x(), color.y(), color.z()};
}

// 物体にレイが衝突した場合の処理です。衝突判定は自動でやってくれます。

extern "C" __global__ void __closesthit__radiance() {
  const auto &triangleMeshes = reinterpret_cast<HitgroupData *>(optixGetSbtDataPointer())->triangleMeshes;

  // ポリゴンの法線を取得します。

  const auto normal = [&] {
    const auto &index = triangleMeshes.indices[optixGetPrimitiveIndex()];

    const auto &vector1 = triangleMeshes.vertices[index.x()];
    const auto &vector2 = triangleMeshes.vertices[index.y()];
    const auto &vector3 = triangleMeshes.vertices[index.z()];

    return (vector2 - vector1).cross(vector3 - vector1).normalized();
  }();

  // レイの向きを取得します。

  const auto rayDirection = [&] {
    auto result = optixGetWorldRayDirection();

    return *reinterpret_cast<Eigen::Vector3f *>(&result);
  }();

  // 色は、光源とかはとりあえず考慮しないで、レイとポリゴンが垂直なほど明るくなるということで。カメラにライトが付いているとでも思って、納得してください……。

  *reinterpret_cast<Eigen::Vector3f *>(getPayloadPointer()) = triangleMeshes.color * (0.2 + 0.8 * std::abs(normal.dot(rayDirection)));
}

// 物体にレイが衝突しそうな場合の処理です。このコースでは最後まで使用しません。

extern "C" __global__ void __anyhit__radiance() {
  ; // このコースでは、なにもしません。
}

// トレースした光が物体に衝突しなかった場合の処理です。

extern "C" __global__ void __miss__radiance() {
  *reinterpret_cast<Eigen::Vector3f *>(getPayloadPointer()) = Eigen::Vector3f{1, 1, 1}; // とりあえず、背景は真っ白にします。
}

} // namespace osc
