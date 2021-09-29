#include <tuple>

#include <Eigen/Core>
#include <optix.h>
#include <optix_device.h>
#include <optix_stubs.h>

#include "Params.h"

namespace osc {

extern "C" {
  __constant__ osc::LaunchParams OptixLaunchParams;
}

enum { // TODO: enum classに書き換える。
  SURFACE_RAY_TYPE = 0,
  RAY_TYPE_COUNT
};

// OptiXのペイロードはunsigned int×n個で扱いづらいので、構造体へのポインタにします。

inline __device__ auto getPayloadParams(void *PayloadPointer) noexcept {
  auto P = reinterpret_cast<std::uint64_t>(PayloadPointer);

  return std::make_tuple(static_cast<std::uint32_t>(P >> 32), static_cast<std::uint32_t>(P));
}

inline __device__ auto getPayloadPointer() noexcept {
  return reinterpret_cast<void *>(static_cast<std::uint64_t>(optixGetPayload_0()) << 32 | static_cast<std::uint64_t>(optixGetPayload_1()));
}

// 色は、とりあえずポリゴンのインデックスから適当に作ります。

inline __device__ auto getRandomColor(unsigned int Seed) noexcept {
  const auto R = Seed * 13 * 17 + 0x234235;
  const auto G = Seed * 7 * 3 * 5 + 0x773477;
  const auto B = Seed * 11 * 19 + 0x223766;

  return Eigen::Vector3f((R & 0x00ff) / 255.0f, (G & 0x00ff) / 255.0f, (B & 0x00ff) / 255.0f);
}

// 光を生成します。

extern "C" __global__ void __raygen__renderFrame() {
  const auto &X = optixGetLaunchIndex().x;
  const auto &Y = optixGetLaunchIndex().y;

  const auto &Camera = reinterpret_cast<osc::RaygenData *>(optixGetSbtDataPointer())->Camera;

  auto Origin = Camera.Origin;
  auto Direction = ((static_cast<float>(X) / OptixLaunchParams.Width * 2 - 1) * Camera.U + (static_cast<float>(Y) / OptixLaunchParams.Height * 2 - 1) * Camera.V + Camera.W).normalized();

  auto Color = Eigen::Vector3f{0};

  auto [PayloadParam0, PayloadParam1] = getPayloadParams(&Color);

  optixTrace(
      OptixLaunchParams.TraversableHandle,
      *reinterpret_cast<float3 *>(&Origin),
      *reinterpret_cast<float3 *>(&Direction),
      0.f,                           // tmin
      1e20f,                         // tmax
      0.0f,                          // rayTime
      OptixVisibilityMask(255),      //
      OPTIX_RAY_FLAG_DISABLE_ANYHIT, // rayFlags,
      SURFACE_RAY_TYPE,              // SBToffset
      RAY_TYPE_COUNT,                // SBTstride
      SURFACE_RAY_TYPE,              // missSBTIndex
      PayloadParam0,                 // ペイロードではunsigned intしか使えない……。
      PayloadParam1);

  const auto R = static_cast<int>(255.5 * Color[0]);  // intへのキャストは小数点以下切り捨てなので、255よりも少し大きい値を使用しました。
  const auto G = static_cast<int>(255.5 * Color[1]);
  const auto B = static_cast<int>(255.5 * Color[2]);

  OptixLaunchParams.ImageBuffer[X + Y * OptixLaunchParams.Width] = 0xff000000 | R << 0 | G << 8 | B << 16;
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
