#include <optix_device.h>

#include "OptixParams.h"

namespace osc {

// OptiXのデバイス・プログラムに渡すパラメーター。

extern "C" {
__constant__ LaunchParams optixLaunchParams;
}

// レイを生成します。

extern "C" __global__ void __raygen__renderFrame() {
  const auto &x = optixGetLaunchIndex().x;
  const auto &y = optixGetLaunchIndex().y;

  if (x == 0 && y == 0) {
    printf("___raygen__renderFrame() is called.\n");
  }

  // 通常はレイを生成して、トレースして、その結果に基づいて出力を作成するのですが、とりあえず、テスト・パターンを生成してみます。

  const auto r = x % 256;
  const auto g = y % 256;
  const auto b = (x + y) % 256;

  optixLaunchParams.imageBuffer[x + y * optixGetLaunchDimensions().x] = r << 0 | g << 8 | b << 16 | 0xff000000;
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
