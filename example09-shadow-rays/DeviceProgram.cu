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

  // レイが衝突した場所の色を取得します。

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

  // レイが衝突した場所の（カメラに向いた面の）法線を取得します。

  const auto normal = [&] {
    auto result = static_cast<Eigen::Vector3f>((1 - u - v) * triangleMeshes.normals[index.x()] + u * triangleMeshes.normals[index.y()] + v * triangleMeshes.normals[index.z()]);

    if (result.dot(rayDirection) > 0) {
      result *= -1;
    }

    return result.normalized();
  }();

  // レイが衝突した場所（から、同じポリゴンに再衝突しないように法線方向に少しずらした場所）を取得します。

  auto hitPosition = [&] {
    return static_cast<Eigen::Vector3f>((1 - u - v) * triangleMeshes.vertices[index.x()] + u * triangleMeshes.vertices[index.y()] + v * triangleMeshes.vertices[index.z()] + normal * 1e-3f); // Eigenは必要になるまで計算を遅らせるので、static_castしないとoptixTraceで計算途中の値をreinterpret_castされちゃう……。
  }();

  // レイが衝突した場所から光源への方向を取得します。

  auto lightDirection = [&] {
    return static_cast<Eigen::Vector3f>((*reinterpret_cast<Eigen::Vector3f *>(&optixLaunchParams.lightPosition) - hitPosition).normalized());
  }();

  // レイが衝突した場所から光源が見えるかを表現する変数を用意します。この値を、optixTraceして設定します。

  bool isLightVisible = false;

  // 影を生成するためのレイを使用して、optixTraceします。

  auto [payloadParam0, payloadParam1] = getPayloadParams(&isLightVisible);

  optixTrace(
      optixLaunchParams.traversableHandle,
      *reinterpret_cast<float3 *>(&hitPosition),
      *reinterpret_cast<float3 *>(&lightDirection),
      0.0f,
      1e20f,
      0.0f,
      OptixVisibilityMask(255),
      OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
      static_cast<int>(RayType::Shadow),
      static_cast<int>(RayType::Size),
      static_cast<int>(RayType::Shadow),
      payloadParam0,
      payloadParam1);

  // 色を設定します。光源が見えない場合でも、0.3の明るさで表示します。

  *reinterpret_cast<Eigen::Vector3f *>(getPayloadPointer()) = color * (0.3 + 0.7 * (isLightVisible ? std::abs(normal.dot(lightDirection)) : 0));
}

// 物体に光が衝突しそうな場合の処理？

extern "C" __global__ void __anyhit__radiance() {
  ; // 表面の色彩のためのレイでは、何もしません。
}

// トレースした光が物体に衝突しなかった場合の処理です。

extern "C" __global__ void __miss__radiance() {
  *reinterpret_cast<Eigen::Vector3f *>(getPayloadPointer()) = Eigen::Vector3f{1, 1, 1}; // とりあえず、背景は真っ白にします。
}

// 影を生成するためのレイが、物体に衝突した場合の処理。

extern "C" __global__ void __closesthit__shadow() {
  ; // 影を生成するためのレイでは、何もしません。
}

// 影を生成するためのレイが、物体に衝突しそうな場合の処理？

extern "C" __global__ void __anyhit__shadow() {
  ; // 影を生成するためのレイでは、何もしません。
}

// 影を生成するためのレイが、物体に衝突しなかった場合の処理。

extern "C" __global__ void __miss__shadow() {
  *reinterpret_cast<bool *>(getPayloadPointer()) = true; // 影を生成するためのレイが何にもぶつからなかった＝光源に辿り着けた＝明るさはマックスで。
}

} // namespace osc