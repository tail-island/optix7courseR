# example10-soft-shadows

example09で影が付いたのですけど、なんだかとても不自然な画像でした……。我々は大気圏の中で日常を送っているので光が大気で散乱している状態を自然だと感じていて、だからこんなにパキっとした影がある画像を気持ち悪いと感じて当たり前でしょう。というわけで、example10ではもう少し影の境界をぼやけさせます。乱数を使用して、影になるかどうかを確率的に扱いましょう。あと、光源に近いほど明るく、光源から遠くなると暗くなるというのも実現します（太陽光ではなく、電球みたいな光源だと考えてください）。

さて、あるドットが影になったりならなかったりするには、レイが衝突した点から光源が見えたり見えなかったりすればよいわけ。そのためには、光源の位置が乱数でランダムに移動すると楽ちん。でも、野放図にどこまでも光源が移動できるとどのドットも影になったりならなかったりしてしまうので、光源が移動できる範囲を制限したい。あと、光源から遠くなると暗くなるのを実現するために、光源に光の強さを表現する変数も追加したい。これらはCPU側で設定してGPU側で参照する値なので、OptixParams.hに定義しましょう。

~~~c++
struct Light {
  float3 origin;
  float3 u; // 光源の範囲
  float3 v; // 光源の範囲
  float3 power; // RGB三原色での光の強さ
};

...

struct LaunchParams {
  float3 *imageBuffer;
  int frameId;

  Light light;
  Camera camera;

  OptixTraversableHandle traversableHandle;
};
~~~

次は乱数です。残念なことに、CUDAでは慣れ親しんだC++の標準ライブラリの\<random\>は使用できません（というか、数値演算の標準ライブラリくらいしか使えない……）。効率よく並列でランダムを生成するアルゴリズムは、標準ライブラリの\<random\>とは違いますしね。でもだからといって自前で乱数ライブラリを作るなんてのはやってられません（[元ネタ](https://github.com/ingowald/optix7course/blob/master/common/gdt/gdt/random/random.h)ではやっていますけど……）。というわけで、CUDAの乱数ライブラリであるcuRANDを使ってみましょう。cuRANDの内部はCPU側で動くモジュールとGPU側で動くモジュールに分かれていて、今回はGPU側で動くモジュールを使用します（CPU側で動くモジュールは大量の乱数を効率よく作成する場合にとても便利なので、別の機会にぜひ使用してみてください）。

というわけで、DeviceProgram.cuにGPU側のcuRANDを使用するためのヘッダを`#include`させます。

~~~c++
#include <curand_kernel.h>
~~~

これで乱数を作れるようになったのですけど、cuRANDで乱数を生成するときには`curandState`へのポインターが必要で、これをグローバル変数にしてしまうと並列動作する際に問題になりそうなので、`optxTrace()`のペイロードで渡すようにします。example09までのペイロードの情報は色が1つだけだったのでクラスを作らなくて大丈夫だったのですけど、example10では情報が2つになるのでクラスを作らなければなりません。

~~~c++
// ピクセルの色を取得するためにoptixTraceする際のペイロードです。PRD（Per Ray Data）と呼ばれたりもします。

struct RadiancePayload {
  Eigen::Vector3f color;

  ::curandState *curandState; // 乱数を使用したいので、cuRANDのステータスを保持しておきます。
};
~~~

これで準備ができたので、レイを生成しましょう。

~~~c++
extern "C" __global__ void __raygen__renderFrame() {
  const auto &x = optixGetLaunchIndex().x;
  const auto &y = optixGetLaunchIndex().y;

  ...

  // cuRANDを初期化します。

  auto curandState = ::curandState{};
  curand_init(0, (optixGetLaunchIndex().y * optixGetLaunchDimensions().x + optixGetLaunchIndex().x) + (optixGetLaunchDimensions().x * optixGetLaunchDimensions().y * optixLaunchParams.frameId), 0, &curandState);

  // サンプリングしてピクセルの色を設定します。

  const auto color = [&] {
    auto result = Eigen::Vector3f{0, 0, 0};

    for (auto i = 0; i < PIXEL_SAMPLE_SIZE; ++i) {
      // レイの方向を、ランダム性を加えて計算します。

      auto direction = (((static_cast<float>(x) + curand_uniform(&curandState)) / optixGetLaunchDimensions().x * 2 - 1) * u + ((static_cast<float>(y) + curand_uniform(&curandState)) / optixGetLaunchDimensions().y * 2 - 1) * v + w).normalized(); // optixTraceの都合で、const autoに出来ない……。

      // ペイロードを表現する変数を用意します。optixTraceで呼び出される関数の中で使用されます。

      auto payload = RadiancePayload{Eigen::Vector3f{0, 0, 0}, &curandState};
      auto [payloadParam0, payloadParam1] = getPayloadParams(&payload); // optixTraceの都合で、const autoに出来ない……。

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

  ...
}
~~~

cuRANDを初期化しているのと、レイの方向に乱数を加えていること、ループしてサンプリングしていることと、ペイロードの型が`osc::RadiancePayload`になったことがexample09との差異です。レイの方向に乱数を加えてサンプリングしているのは、アンチ・エイリアシングのためです。次に作成する`__closehit__radiance()`でも乱数を使用しますから、影の境界をぼやけさせる効果もあって一挙両得となるはず。

その`__closehit__radiance()`も修正します。

~~~c++
extern "C" __global__ void __closesthit__radiance() {
  ...

  auto &radiancePayload = *reinterpret_cast<RadiancePayload *>(getPayloadPointer());
  auto &curandState = *radiancePayload.curandState;

  // レイが衝突した場所の色を取得します。

  const auto color = ...;

  // レイの向きを取得します。

  const auto rayDirection = ...;

  // レイが衝突した場所の（カメラに向いた面の）法線を取得します。

  const auto normal = ...;

  // レイが衝突した場所（から、同じポリゴンに再衝突しないように法線方向に少しずらした場所）を取得します。

  auto hitPosition = ...;

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
          1.0f - 1e-3f, // toLightの距離までしかトレースしないようにします。そうしないと、光源の先にあるオブジェクトに衝突してしまう。。。
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
~~~

ループしてサンプリングしていることと、`toLight`に乱数を追加していることと、色を設定する際に光は距離の2乗で減衰するとして計算していることが、example09との違いです。diffを取るといろいろ違いが表示されますけど、それはサンプリングのためのループでインデントが変わったためで、実はexample09からあまり変更していないんです。

あとは、OptixParams.hの型変更に合わせてRenderer.hやWindow.hを修正して、あと、光源が電球となりましたので建物の中の2階に移動させれば終わりです。プログラムを実行して、たしかに影の境界がぼやけたけど、ドット単位でみると明るくなったり暗くなったりしていてなんかちょっと不本意な画像が表示されたら作業は終了です。お疲れさまでした。

![example10-soft-shadows-linux]()

![example10-soft-shadows-windows]()
