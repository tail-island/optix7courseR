# example09-shadow-rays

example08の画像がそこそこリアルに見えるのって、カメラから光が出ているという前提だからじゃね？　カメラから光が出ている場合の画像なので、影がなくてもおかしくはないというトリックなのでは？

……ごめんなさい、その通りです。example08までの方式だと、光源の位置をカメラ以外に変更した場合に、光が届かずに影になるはずの部分が影にならなくて変な画像になってしまいます。そこで、example09では影を生成するようにプログラムを修正していきましょう。example09ではとりあえず、光が直接当たる場合は明るく、当たらない場合は暗くするという簡単な方式を実現してみます。

さて、光が直接当たるというのはどういう状況かと言うと、他のオブジェクトに邪魔されずに光源が見えるという状況だと考えられます。OptiXで考えると、衝突した場所から光源に向かうレイを`optixTrace()`で生成して、そのレイが他のオブジェクトにぶつからずに`__miss_xxx()`になった場合に明るく、そうでない場合に暗くすればよいかなぁと。でも、その`__miss_xxx()`はカメラからレイを射出する場合の`__miss__radiance()`とは異なる処理にだけど、どうするの？

という問題を解決するために、OptiXは複数の種類のレイを扱えるように作られています。Aというレイの種類の場合は`__miss__a()`を、Bというレイの種類の場合は`__miss_b()`を使うという感じですね。さっそく、OptixParams.hに影用のレイを追加しましょう。

~~~c++
enum class RayType {
  Radiance,
  Shadow,
  Size
};
~~~

あと、光源の位置を自由に設定できるように、OptixParams.hの`osc::LaunchParams`も修正します。

~~~c++
struct LaunchParams {
  float3 *imageBuffer;

  float3 lightPosition;
  Camera camera;

  OptixTraversableHandle traversableHandle;
};
~~~

で、複数のレイを登録する形に、OptixState.hを修正していきます。

~~~c++
// 衝突判定のプログラム・グループ群を作成します。処理の実態は*.cuの中にあります。

hitgroupProgramGroups_ = [&] {
  std::cout << "#osc: creating HitgroupProgramGroups...\n";

  auto result = std::vector<OptixProgramGroup>{static_cast<int>(RayType::Size)};

  [&] {
    const auto programGroupDesc = [&] {
      auto result = OptixProgramGroupDesc{};

      result.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
      result.hitgroup.moduleCH = module_;
      result.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
      result.hitgroup.moduleAH = module_;
      result.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

      return result;
    }();

    const auto programGroupOptions = OptixProgramGroupOptions{};

    auto [log, logSize] = [&] {
      auto result = std::array<char, 2048>{};

      return std::make_tuple(result, std::size(result));
    }();

    OPTIX_CHECK(optixProgramGroupCreate(deviceContext_, &programGroupDesc, 1, &programGroupOptions, log.data(), &logSize, &result[static_cast<int>(RayType::Radiance)]));

    if (logSize > 1) { // 文字列の終端の'\0'があるので、１文字以上になります。
      std::cerr << log.data() << "\n";
    }
  }();

  [&] {
    const auto programGroupDesc = [&] {
      auto result = OptixProgramGroupDesc{};

      result.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
      result.hitgroup.moduleCH = module_;
      result.hitgroup.entryFunctionNameCH = "__closesthit__shadow";
      result.hitgroup.moduleAH = module_;
      result.hitgroup.entryFunctionNameAH = "__anyhit__shadow";

      return result;
    }();

    const auto programGroupOptions = OptixProgramGroupOptions{};

    auto [log, logSize] = [&] {
      auto result = std::array<char, 2048>{};

      return std::make_tuple(result, std::size(result));
    }();

    OPTIX_CHECK(optixProgramGroupCreate(deviceContext_, &programGroupDesc, 1, &programGroupOptions, log.data(), &logSize, &result[static_cast<int>(RayType::Shadow)]));

    if (logSize > 1) { // 文字列の終端の'\0'があるので、１文字以上になります。
      std::cerr << log.data() << "\n";
    }
  }();

  return result;
}();

// レイをトレースしても何かにぶつからなかった場合のプログラム・グループ群を作成します。処理の実態は*.cuの中にあります。

missProgramGroups_ = [&] {
  std::cout << "#osc: creating MissProgramGroups...\n";

  auto result = std::vector<OptixProgramGroup>{static_cast<int>(RayType::Size)};

  [&] {
    const auto programGroupDesc = [&] {
      auto result = OptixProgramGroupDesc{};

      result.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
      result.miss.module = module_;
      result.miss.entryFunctionName = "__miss__radiance";

      return result;
    }();

    const auto programGroupOptions = OptixProgramGroupOptions{};

    auto [log, logSize] = [&] {
      auto result = std::array<char, 2048>{};

      return std::make_tuple(result, std::size(result));
    }();

    OPTIX_CHECK(optixProgramGroupCreate(deviceContext_, &programGroupDesc, 1, &programGroupOptions, log.data(), &logSize, &result[static_cast<int>(RayType::Radiance)]));

    if (logSize > 1) { // 文字列の終端の'\0'があるので、１文字以上になります。
      std::cerr << log.data() << "\n";
    }
  }();

  [&] {
    const auto programGroupOptions = OptixProgramGroupOptions{};

    const auto programGroupDesc = [&] {
      auto result = OptixProgramGroupDesc{};

      result.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
      result.miss.module = module_;
      result.miss.entryFunctionName = "__miss__shadow";

      return result;
    }();

    auto [log, logSize] = [&] {
      auto result = std::array<char, 2048>{};

      return std::make_tuple(result, std::size(result));
    }();

    OPTIX_CHECK(optixProgramGroupCreate(deviceContext_, &programGroupDesc, 1, &programGroupOptions, log.data(), &logSize, &result[static_cast<int>(RayType::Shadow)]));

    if (logSize > 1) { // 文字列の終端の'\0'があるので、１文字以上になります。
      std::cerr << log.data() << "\n";
    }
  }();

  return result;
}();
~~~

大きく変更しているように見えますけど、example08までは`OptixProgramGroup`が1つだけだったのが`osc::RayType`の変更に合わせて2つになっただけで、コピー＆ペーストして、関数名（`__xxx_radiance`と`__xxx_shadow`の部分）と設定する値のインデックス（`RayType::Radiance`と`RayType::Shadow`の部分）を変更しただけです（今更だけど、ループにすれば良かったかも……。でも、C++23で追加されるだろう`zip()`がないと、複数の値がある場合のループ化はやりづらいしなぁ）。

さて、以上で準備は整いましたので、GPU側のプログラミングを始めます。まずは衝突した場所の情報を手に入れたいのですけど、普通に衝突した場所を取得すると、他のオブジェクトに衝突せずに光源が見えるかを計算する際に、計算誤差で自分自身と衝突してしまう危険性があるんですよ……。なので、実際の衝突位置からすこしだけずらして取得する処理を作成します。

~~~c++
// レイが衝突した場所の（カメラに向いた面の）法線を取得します。

const auto normal = [&] {
  auto result = static_cast<Eigen::Vector3f>((1 - u - v) * triangleMeshes.normals[index.x()] + u * triangleMeshes.normals[index.y()] + v * triangleMeshes.normals[index.z()]);

  if (result.dot(rayDirection) > 0) {
    result *= -1;
  }

  return result.normalized();
}();

// レイが衝突した場所（から、同じポリゴンに再衝突しないように法線方向に少しずらした場所）を取得します。

auto hitPosition = static_cast<Eigen::Vector3f>((1 - u - v) * triangleMeshes.vertices[index.x()] + u * triangleMeshes.vertices[index.y()] + v * triangleMeshes.vertices[index.z()] + normal * 1e-3f); // Eigenは必要になるまで計算を遅らせるので、static_castしないとoptixTraceで計算途中の値をreinterpret_castされちゃう……。 // optixTraceの都合で、const autoに出来ない……。
~~~

これで衝突した場所が手に入りましたから、光源に向かうベクトルを計算して、`optixTrace()`します。

~~~c++
// レイが衝突した場所から光源への方向を取得します。

auto toLight = [&] { // optixTraceの都合で、const autoに出来ない……。
  return static_cast<Eigen::Vector3f>(*reinterpret_cast<Eigen::Vector3f *>(&optixLaunchParams.lightPosition) - hitPosition);
}();

// レイが衝突した場所から光源が見えるかを表現する変数を用意します。この値をoptixTraceして設定します。

auto isLightVisible = false;
auto [payloadParam0, payloadParam1] = getPayloadParams(&isLightVisible); // optixTraceの都合で、const autoに出来ない……。

// 影を生成するためのレイを使用して、optixTraceします。

optixTrace(
    optixLaunchParams.traversableHandle,
    *reinterpret_cast<float3 *>(&hitPosition),
    *reinterpret_cast<float3 *>(&toLight),
    0.0f,
    1.0f, // toLightの距離までしかトレースしないようにします。
    0.0f,
    OptixVisibilityMask(255),
    OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
    static_cast<int>(RayType::Shadow),
    static_cast<int>(RayType::Size),
    static_cast<int>(RayType::Shadow),
    payloadParam0,
    payloadParam1);
~~~

レイの種類に`RayType::Shadow`を設定している点に注意してください。あと、`optixTrace()`のtmaxに`1.0f`を設定しているところも。これは、3番目の引数の`toLight`と同じ距離までしか衝突を調べないようにするためです。tmaxにたとえば`1e20`のような大きな値を設定してしまうと、光源の向こうのオブジェクトに衝突した場合まで障害物ありと判断されておかしな画像が生成されてしまいますからね。あと、元ネタの[optix7course](https://github.com/ingowald/optix7course/blob/master/example09_shadowRays/devicePrograms.cu)では、レイの出発点を法線にそって`1e-3f`移動させているのになぜかtminに`1e-3f`を設定して、で、法線と衝突点から光源へのベクトルは無関係なのになぜかtmaxに`1.0f - 1e-3f`を設定していたりするのですけど、ここは私が正しいと思う（あまり自信はないけどな）値に変更してみました（どちらにしろ誤差なので、生成される画像に違いはないでしょうし）。

あと、`RayType::Shadow`用の関数も作成しなければなりません。今回必要なのは`__miss_shadow()`だけなのですけど、全部を定義します。

~~~c++
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
  *reinterpret_cast<bool *>(getPayloadPointer()) = true; // 影を生成するためのレイが何にもぶつからなかった＝光源に辿り着けた＝明るい。
}
~~~

これで、光源までの経路に障害物がない場合にペイロードの値である`isLightVisible`が`true`になるようになりましたので、ピクセルの色設定するコードを作成します。

~~~c++
*reinterpret_cast<Eigen::Vector3f *>(getPayloadPointer()) = color * (0.2 + 0.8 * (isLightVisible ? normal.dot(toLight.normalized()) : 0));
~~~

影の部分を真っ暗にすると宇宙空間みたいに見えてしまうので、影であっても少しだけ明るく見えるようにしました。[元ネタのoptix7course](https://github.com/ingowald/optix7course/blob/master/example09_shadowRays/devicePrograms.cu)だとカメラへの角度でも明るさを変えている（だから影の部分でも明るい箇所と暗い箇所があるし、光がほぼ垂直にあたっているけどカメラと水平に近い床が暗くなっている）のですけど、理屈に合わない気がしたのと、[元ネタのexample10](https://github.com/ingowald/optix7course/blob/master/example10_softShadows/devicePrograms.cu)では光源との角度で明るさを計算していたので、こんな計算式にしてみました。

ともあれ、あとは、光源を設定するようにRenderrer.hやWindow.h、Main.cppを修正すれば終わりです。プログラムを実行して、パキッとした影が描画されることを確認できたら作業は終了です。お疲れさまでした。

![example09-shadow-rays-linux]()

![example09-shadow-rays-windows]()
