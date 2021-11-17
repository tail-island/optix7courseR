#include <optix_device.h>

#include "OptixParams.h"

namespace osc {

// OptiXのデバイス・プログラムに渡すパラメーター。

extern "C" {
__constant__ LaunchParams optixLaunchParams;
}

// レイを生成します。

extern "C" __global__ void __raygen__renderFrame() {
  const auto &frameId = optixLaunchParams.frameId;
  const auto &x = optixGetLaunchIndex().x;
  const auto &y = optixGetLaunchIndex().y;

  if (frameId == 0 && x == 0 && y == 0) {
    printf("___raygen__renderFrame() is called.\n");
  }

  // 通常はレイを生成して、トレースして、その結果に基づいて出力を作成するのですが、とりあえず、テスト・パターンを生成してみます。

  const auto r = (x + frameId) % 256;
  const auto g = (y + frameId) % 256;
  const auto b = (x + y + frameId) % 256;

  optixLaunchParams.imageBuffer[x + y * optixGetLaunchDimensions().x] = float3{static_cast<float>(r) / 255, static_cast<float>(g) / 255, static_cast<float>(b) / 255};;
}

extern "C" __global__ void __closesthit__radiance() {
  ; // とりあえず、なにもしません。
}

extern "C" __global__ void __anyhit__radiance() {
  ; // とりあえず、なにもしません。
}

extern "C" __global__ void __miss__radiance() {
  ; // とりあえず、なにもしません。
}

} // namespace osc
