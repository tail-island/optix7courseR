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

// 色は、とりあえずポリゴンのインデックスから適当に作ります。

inline __device__ auto getRandomColor(unsigned int seed) noexcept {
  const auto r = seed * 13 * 17 + 0x234235;
  const auto g = seed * 7 * 3 * 5 + 0x773477;
  const auto b = seed * 11 * 19 + 0x223766;

  return Eigen::Vector3f((r & 0x00ff) / 255.0f, (g & 0x00ff) / 255.0f, (b & 0x00ff) / 255.0f);
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
      0.0f,                               // tmin
      1e20f,                              // tmax
      0.0f,                               // rayTime
      OptixVisibilityMask(255),           //
      OPTIX_RAY_FLAG_DISABLE_ANYHIT,      // rayFlags,
      static_cast<int>(RayType::Surface), // SBToffset
      static_cast<int>(RayType::Size),    // SBTstride
      static_cast<int>(RayType::Surface), // missSBTIndex
      payloadParam0,                      // ペイロードではunsigned intしか使えません……。
      payloadParam1);

  const auto r = static_cast<int>(255.5 * color.x()); // intへのキャストは小数点以下切り捨てなので、255よりも少し大きい値を使用しました。
  const auto g = static_cast<int>(255.5 * color.y());
  const auto b = static_cast<int>(255.5 * color.z());

  optixLaunchParams.imageBuffer[x + y * optixGetLaunchDimensions().x] = r << 0 | g << 8 | b << 16 | 0xff000000;
}

// 物体に光が衝突した場合の処理です。衝突判定は自動でやってくれるみたい。

extern "C" __global__ void __closesthit__radiance() {
  *reinterpret_cast<Eigen::Vector3f *>(getPayloadPointer()) = getRandomColor(optixGetPrimitiveIndex()); // とりあえず、光が衝突したポリゴンのインデックスをシードにして、ランダムな色を割り当てます。
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
