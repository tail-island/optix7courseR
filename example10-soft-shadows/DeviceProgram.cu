#include <tuple>

#pragma nv_diag_suppress 20236

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <curand_kernel.h>
#include <optix.h>
#include <optix_device.h>
#include <optix_stubs.h>

#include "OptixParams.h"

#define PIXEL_SAMPLE_SIZE 16
#define LIGHT_SAMPLE_SIZE 1

namespace osc {

extern "C" {
__constant__ LaunchParams optixLaunchParams;
}

struct HitgroupData {
  TriangleMeshes triangleMeshes;
};

// OptiXのペイロードはunsigned int×n個で扱いづらいので、構造体へのポインタに変換します。

inline __device__ auto getPayloadParams(void *payloadPointer) noexcept {
  auto p = reinterpret_cast<std::uint64_t>(payloadPointer);

  return std::make_tuple(static_cast<std::uint32_t>(p >> 32), static_cast<std::uint32_t>(p));
}

inline __device__ auto getPayloadPointer() noexcept {
  return reinterpret_cast<void *>(static_cast<std::uint64_t>(optixGetPayload_0()) << 32 | static_cast<std::uint64_t>(optixGetPayload_1()));
}

// ピクセルの色を取得するためにoptixTraceする際のペイロードです。PRD（Per Ray Data）と呼ばれたりもします。

struct RadiancePayload {
  Eigen::Vector3f color;

  ::curandState *curandState; // 乱数を使用したいので、cuRANDのステータスを保持しておきます。
};

// 光を生成します。

extern "C" __global__ void __raygen__renderFrame() {
  const auto &x = optixGetLaunchIndex().x;
  const auto &y = optixGetLaunchIndex().y;

  // カメラの情報を取得します。

  auto &origin = *reinterpret_cast<Eigen::Vector3f *>(&optixLaunchParams.camera.origin);

  const auto &u = *reinterpret_cast<Eigen::Vector3f *>(&optixLaunchParams.camera.u);
  const auto &v = *reinterpret_cast<Eigen::Vector3f *>(&optixLaunchParams.camera.v);
  const auto &w = *reinterpret_cast<Eigen::Vector3f *>(&optixLaunchParams.camera.w);

  // cuRANDを初期化します。

  auto curandState = ::curandState{};
  curand_init(0, (optixGetLaunchIndex().y * optixGetLaunchDimensions().x + optixGetLaunchIndex().x) + (optixGetLaunchDimensions().x * optixGetLaunchDimensions().y * optixLaunchParams.frameId), 0, &curandState);

  // サンプリングしてピクセルの色を設定します。

  const auto color = [&] {
    auto result = Eigen::Vector3f{0, 0, 0};

    for (auto i = 0; i < PIXEL_SAMPLE_SIZE; ++i) {
      // レイの方向を、ランダム性を加えて計算します。

      auto direction = (((static_cast<float>(x) + curand_uniform(&curandState)) / optixGetLaunchDimensions().x * 2 - 1) * u + ((static_cast<float>(y) + curand_uniform(&curandState)) / optixGetLaunchDimensions().y * 2 - 1) * v + w).normalized();

      // ペイロードを表現する変数を用意します。optixTraceで呼び出される関数の中で使用されます。

      auto payload = RadiancePayload{Eigen::Vector3f{0, 0, 0}, &curandState};
      auto [payloadParam0, payloadParam1] = getPayloadParams(&payload);

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

      // optixTraceで設定されたcolorを使用して、ピクセルの色を設定します。

      result += payload.color / PIXEL_SAMPLE_SIZE;
    }

    return result;
  }();

  // イメージ・バッファーに値を設定します。

  optixLaunchParams.imageBuffer[x + y * optixGetLaunchDimensions().x] = float3{color.x(), color.y(), color.z()};
}

// 物体に光が衝突した場合の処理です。衝突判定は自動でやってくれます。

extern "C" __global__ void __closesthit__radiance() {
  const auto &triangleMeshes = reinterpret_cast<HitgroupData *>(optixGetSbtDataPointer())->triangleMeshes;

  const auto &index = triangleMeshes.indices[optixGetPrimitiveIndex()];

  const auto u = optixGetTriangleBarycentrics().x;
  const auto v = optixGetTriangleBarycentrics().y;

  auto &radiancePayload = *reinterpret_cast<RadiancePayload *>(getPayloadPointer());
  auto &curandState = *radiancePayload.curandState;

  // レイが衝突した場所の色を取得します。

  const auto color = [&] {
    if (!triangleMeshes.hasTextureObject) {
      return triangleMeshes.color;
    }

    const auto textureCoordinate = (1 - u - v) * triangleMeshes.textureCoordinates[index.x()] + u * triangleMeshes.textureCoordinates[index.y()] + v * triangleMeshes.textureCoordinates[index.z()];
    const auto textureColor = tex2D<float4>(triangleMeshes.textureObject, textureCoordinate.x(), textureCoordinate.y());

    return static_cast<Eigen::Vector3f>(Eigen::Vector3f{textureColor.x, textureColor.y, textureColor.z} * textureColor.w);
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

  // サンプリングして衝突点の色を設定します。

  radiancePayload.color = [&] {
    auto result = static_cast<Eigen::Vector3f>(color * 0.2); // 散乱光で少しだけ光があたっている状態から開始します。

    for (auto i = 0; i < LIGHT_SAMPLE_SIZE; ++i) {
      // レイが衝突した場所から光源への方向を取得します。

      auto toLight = [&] {
        const auto lightPosition = *reinterpret_cast<Eigen::Vector3f *>(&optixLaunchParams.light.origin) +
                                   *reinterpret_cast<Eigen::Vector3f *>(&optixLaunchParams.light.u) * curand_uniform(&curandState) + // ランダムを追加します。
                                   *reinterpret_cast<Eigen::Vector3f *>(&optixLaunchParams.light.v) * curand_uniform(&curandState);

        return static_cast<Eigen::Vector3f>(lightPosition - hitPosition);
      }();

      // レイが衝突した場所から光源が見えるかを表現する変数を用意します。この値をoptixTraceして設定します。boolでも良いのですけど、計算の都合上3次元ベクトルで。

      auto lightVisibility = Eigen::Vector3f{0, 0, 0};
      auto [payloadParam0, payloadParam1] = getPayloadParams(&lightVisibility);

      // 影を生成するためのレイを使用して、optixTraceします。

      optixTrace(
          optixLaunchParams.traversableHandle,
          *reinterpret_cast<float3 *>(&hitPosition),
          *reinterpret_cast<float3 *>(&toLight),
          0.0f,
          1.0f - 1e-3f, // toLightの距離までしかトレースしないようにします（- 1e-3fしているのは、hitPositionを移動したため）。そうしないと、光源の先にあるオブジェクトに衝突してしまう。。。
          0.0f,
          OptixVisibilityMask(255),
          OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
          static_cast<int>(RayType::Shadow),
          static_cast<int>(RayType::Size),
          static_cast<int>(RayType::Shadow),
          payloadParam0,
          payloadParam1);

      // 色を設定します。

      result += (color.array() * (*reinterpret_cast<Eigen::Vector3f *>(&optixLaunchParams.light.power) / std::pow(toLight.norm(), 2)).array() * (lightVisibility / LIGHT_SAMPLE_SIZE).array()).matrix() * std::abs(normal.dot(toLight.normalized()));
    }

    return result;
  }();
}

// 物体にレイが衝突しそうな場合の処理です。このコースでは最後まで使用しません。

extern "C" __global__ void __anyhit__radiance() {
  ; // 表面の色彩のためのレイでは、何もしません。
}

// トレースした光が物体に衝突しなかった場合の処理です。

extern "C" __global__ void __miss__radiance() {
  *reinterpret_cast<Eigen::Vector3f *>(getPayloadPointer()) = Eigen::Vector3f{1, 1, 1}; // とりあえず、背景は真っ白にします。
}

// 影を生成するためのレイが、物体に衝突した場合の処理です。

extern "C" __global__ void __closesthit__shadow() {
  ; // 影を生成するためのレイでは、何もしません。光源に向けたレイが衝突しなかったのなら明るくするという実装のため。
}

// 影を生成するためのレイが、物体に衝突しそうな場合の処理です。

extern "C" __global__ void __anyhit__shadow() {
  ; // 影を生成するためのレイでは、何もしません。光源に向けたレイが衝突しなかったのなら明るくするという実装のため。
}

// 影を生成するためのレイが、物体に衝突しなかった場合の処理です。

extern "C" __global__ void __miss__shadow() {
  *reinterpret_cast<Eigen::Vector3f *>(getPayloadPointer()) = Eigen::Vector3f{1, 1, 1}; // 影を生成するためのレイが何にもぶつからなかった＝光源に辿り着けた＝明るさはマックスで。
}

} // namespace osc
