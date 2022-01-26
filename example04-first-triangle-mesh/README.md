# example04-first-triangle-mesh

テスト・パターンの表示はもう飽きた！　3Dオブジェクトを表示してグリグリしたい！　というわけで、ポリゴンで何か表示してみましょう。

そのためには3Dオブジェクトを表現するクラスが必要なので、Model.hを作成しました。

~~~c++
class Model final {
  std::vector<Eigen::Vector3f> vertices_;
  std::vector<Eigen::Vector3i> indices_;

public:
  const auto &getVertices() const noexcept {
    return vertices_;
  }

  const auto &getIndices() const noexcept {
    return indices_;
  }

  auto addRectangular(const Eigen::Vector3f &center, const Eigen::Vector3f &size) noexcept {
    const auto unitCubeVertices = std::vector<Eigen::Vector3f>{
        {0, 0, 0},
        {1, 0, 0},
        {0, 1, 0},
        {1, 1, 0},
        {0, 0, 1},
        {1, 0, 1},
        {0, 1, 1},
        {1, 1, 1}};

    const auto unitCubeIndices = std::vector<Eigen::Vector3i>{
        {0, 1, 3},
        {2, 3, 0},
        {5, 7, 6},
        {5, 6, 4},
        {0, 4, 5},
        {0, 5, 1},
        {2, 3, 7},
        {2, 7, 6},
        {1, 5, 7},
        {1, 7, 3},
        {4, 0, 2},
        {4, 2, 6}};

    const auto firstVertexIndex = std::size(vertices_);
    const auto affine = Eigen::Translation3f{center - size / 2} * Eigen::Scaling(size);

    std::transform(std::begin(unitCubeVertices), std::end(unitCubeVertices), std::back_inserter(vertices_), [&](const auto &unitCubeVertex) {
      return affine * unitCubeVertex;
    });

    std::transform(std::begin(unitCubeIndices), std::end(unitCubeIndices), std::back_inserter(indices_), [&](const auto &unitCubeIndex) {
      return firstVertexIndex + unitCubeIndex.array();
    });
  }
};
~~~

一般に、3Dオブジェクトはポリゴンと呼ばれる多角形で構成されたサーフェス・モデルで表現します。で、四角形以上だと座標設定を間違えた場合に表面が平らにならなくなってしまうので、多角形には絶対安全な三角形を使用します。あと、隣り合う三角形が頂点を共有する場合にメモリと処理を節約するために、頂点の集合と、三角形単位での頂点のインデックスへの集合という2つのデータで表現します（上のコードの`vertices_`が頂点の集合、`indices_`が頂点へのインデックスの集合です）。

![]()

で、三角形を一つ表示しても3Dグラフィックスっぽくないので、今回は`addRectangular()`で直方体を追加するようにしてみました。これを使用して、Main.cppで地面っぽい薄い直方体の上に立方体が浮いているモデルを作成します。

~~~c++
int main(int argc, char **argv) {
  const auto model = [] {
    auto result = osc::Model{};

    // 地面。
    result.addRectangular(Eigen::Vector3f{0, -1.5, 0}, Eigen::Vector3f{10, 0.1, 10});

    // 立方体。
    result.addRectangular(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(2, 2, 2));

    return result;
  }();

  osc::Window("Optix 7 Course Example", {Eigen::Vector3f{-10, 2, -12}, Eigen::Vector3f{0, 0, 0}, Eigen::Vector3f{0, 1, 0}}, model).run();
}
~~~

これでモデルはできたのですけど、このモデルのままではOptiXが理解できないので、OptiXが理解可能な型に変換する処理をOptixState.hの中に追加します。

~~~c++
// OptixのTraversableHandleを生成します。

verticesBuffer_.setSize(std::size(model.getVertices()));
verticesBuffer_.set(model.getVertices());

indicesBuffer_.setSize(std::size(model.getIndices()));
indicesBuffer_.set(model.getIndices());

[&] {
  const auto accelBuildOptions = [&] {
    auto result = OptixAccelBuildOptions{};

    result.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    result.motionOptions.numKeys = 1;
    result.operation = OPTIX_BUILD_OPERATION_BUILD;

    return result;
  }();

  const auto triangleArrayFlags = 0u;

  const auto buildInput = [&] {
    auto result = OptixBuildInput{};

    result.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    result.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    result.triangleArray.vertexStrideInBytes = sizeof(Eigen::Vector3f);
    result.triangleArray.numVertices = verticesBuffer_.getSize();
    result.triangleArray.vertexBuffers = &verticesBuffer_.getData();

    result.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    result.triangleArray.indexStrideInBytes = sizeof(Eigen::Vector3i);
    result.triangleArray.numIndexTriplets = indicesBuffer_.getSize();
    result.triangleArray.indexBuffer = indicesBuffer_.getData();

    result.triangleArray.flags = &triangleArrayFlags;
    result.triangleArray.numSbtRecords = 1;
    result.triangleArray.sbtIndexOffsetBuffer = 0;
    result.triangleArray.sbtIndexOffsetSizeInBytes = 0;
    result.triangleArray.sbtIndexOffsetStrideInBytes = 0;

    return result;
  }();

  const auto accelBufferSizes = [&] {
    auto result = OptixAccelBufferSizes{};

    OPTIX_CHECK(optixAccelComputeMemoryUsage(deviceContext_, &accelBuildOptions, &buildInput, 1, &result));

    return result;
  }();

  const auto compactedSizeBuffer = common::DeviceBuffer<std::uint64_t>{};

  const auto accelEmitDesc = [&] {
    auto result = OptixAccelEmitDesc{};

    result.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    result.result = compactedSizeBuffer.getData();

    return result;
  }();

  const auto tempBuffer = common::DeviceVectorBuffer<std::uint8_t>{accelBufferSizes.tempSizeInBytes};
  const auto outputBuffer = common::DeviceVectorBuffer<std::uint8_t>{accelBufferSizes.outputSizeInBytes};

  OPTIX_CHECK(optixAccelBuild(deviceContext_, 0, &accelBuildOptions, &buildInput, 1, tempBuffer.getData(), tempBuffer.getDataSize(), outputBuffer.getData(), outputBuffer.getDataSize(), &traversableHandle_, &accelEmitDesc, 1));

  traversableBuffer_.setSize(compactedSizeBuffer.get());

  OPTIX_CHECK(optixAccelCompact(deviceContext_, 0, traversableHandle_, traversableBuffer_.getData(), traversableBuffer_.getDataSize(), &traversableHandle_));
}();
~~~

やたらと面倒くさいコードですけど、OptiXがやれと言うのだからどうしょうもありません。もう、こういうもんなんだと納得して写経してください……。

ともあれ、OptiXが理解可能なデータができましたので、これをOptixParams.hの`osc::LaunchParams`経由でOptiXに渡します。`osc::LaunchParams`には、視点をマウスでグリグリ動かせるよう、カメラの情報も追加しました。

~~~c++
enum class RayType {
  Radiance,
  Size
};

struct Camera {
  float3 origin;
  float3 u;
  float3 v;
  float3 w;
};

struct LaunchParams {
  float3 *imageBuffer;

  Camera camera;

  OptixTraversableHandle traversableHandle;
};
~~~

あれ？　その上の`osc::RayType`って何？　そう思った方は、DeviceProgram.cuを見てみてください。全面的に書き換えられていて、この中で`RayType`を使用しています。まずは、レイを生成する`__raygen_renderFrame()`を見てみます。

~~~c++
extern "C" __global__ void __raygen__renderFrame() {
  const auto &x = optixGetLaunchIndex().x;
  const auto &y = optixGetLaunchIndex().y;

  // カメラの情報を取得します。

  auto &origin = *reinterpret_cast<Eigen::Vector3f *>(&optixLaunchParams.camera.origin);  // optixTraceの都合で、const autoに出来ない……。

  const auto &u = *reinterpret_cast<Eigen::Vector3f *>(&optixLaunchParams.camera.u);
  const auto &v = *reinterpret_cast<Eigen::Vector3f *>(&optixLaunchParams.camera.v);
  const auto &w = *reinterpret_cast<Eigen::Vector3f *>(&optixLaunchParams.camera.w);

  // レイの方向を計算します。

  auto direction = ((static_cast<float>(x) / optixGetLaunchDimensions().x * 2 - 1) * u + (static_cast<float>(y) / optixGetLaunchDimensions().y * 2 - 1) * v + w).normalized();  // optixTraceの都合で、const autoに出来ない……。

  // ピクセルの色を表現する変数を用意します。この値をoptixTraceして設定します。

  auto color = Eigen::Vector3f{0};
  auto [payloadParam0, payloadParam1] = getPayloadParams(&color);  // optixTraceの都合で、const autoに出来ない……。

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
~~~

このコードで重要なのは`optixTrace()`を呼び出しているところです（カメラ座標での変換については、UVW mappingで検索してみてください）。この`optixTrace()`をするだけで、レイを射出して、そのレイが何かに衝突すると全自動で`__closehit__xxx()`を呼び出してくれるという便利機能です。

で、衝突判定には3Dオブジェクトのモデルが必要なので、最初の引数に`optixLaunchParams.traversableHandle`を渡しているわけ。2番目の引数はレイの射出元で、3番目の引数はレイの方向。4番目の引数のtminと5番目の引数のtmaxは衝突判定をする範囲です。近くのオブジェクトとの衝突判定をしたくないならtminを大きく、遠くのオブジェクトとの衝突判定をしたくないならtmaxを小さくします。で、6番目の引数と7番目の引数は分からん（設定したことない。0と適当な値にしておけば良さそう）。8番目の引数はレイの動作で、今回はレイがオブジェクトの近くを通った場合の`__anyhit__xxx()`を無視するために`OPTIX_RAY_FLAG_DISABLE_ANYHIT`を設定しています。

9〜11番目までの`RayType`を使用している引数は、実はexample09で初めて意味がある使い方をするので説明が難しいのですけど、レイの種類を表しています。で、今回はレイを1種類しか使用していないのであまり意味がないというわけ。`RayType::Size`はC++の`enum`は要素の数を取得することができないためのハックで、最後に`Size`という要素を付け加えて要素数を表現させています。

残り、12番目と13番目の引数は一見しただけではなんだか分からないでしょう。これはレイ単位に用意されるペイロード（積載物）で、なんでペイロードなんてのが必要なのかと言うと、レイを追跡していく中で何か処理をした結果をどこかに置いておく必要があるためです。プログラマーとしてはこういうのは戻り値でやれよと思うのですけど、複数の値を戻して構造化束縛できるようになったのはC++17からなので、古いバージョンのC++もサポートしなければならないNVIDIAとしては戻り値にはしづらかったのでしょう。というわけで戻り値ではなく引数になっているのはしょうがないとしても、引数の型が`unsigned int`限定なのは酷い。これじゃあまりに使いづらいですから、データへのポインター（64bit）を32bitの値2つで表現することにしました。ポインターを`unsigned int`2つに分割したり、`unsigned int`2つを元のポインターに戻したりするのは、以下の2つの関数でやっています。

~~~c++
inline __device__ auto getPayloadParams(void *payloadPointer) noexcept {
  auto p = reinterpret_cast<std::uint64_t>(payloadPointer);

  return std::make_tuple(static_cast<std::uint32_t>(p >> 32), static_cast<std::uint32_t>(p));
}

inline __device__ auto getPayloadPointer() noexcept {
  return reinterpret_cast<void *>(static_cast<std::uint64_t>(optixGetPayload_0()) << 32 | static_cast<std::uint64_t>(optixGetPayload_1()));
}
~~~

というわけで、`optixTrace()`すると引数の`payloadParam0`と`payloadParam1`の元になった`color`に値が設定されるので、その後で`color.x()`とかを使って画像に色を設定できるというわけ。

さて、`optixTrace()`を呼び出すと全自動で`__closehit__xxx()`が呼び出されることを前で述べましたが、その`__closehit_xxx()`にも処理を書かなければなりません。今回は`__closehit_radiance()`を使用するようにOptixState.hで設定していますので、`__closehit_radiance()`の中に処理を記述します。

~~~c++
// 色は、とりあえずポリゴンのインデックスから適当に作ります。

inline __device__ auto getRandomColor(unsigned int seed) noexcept {
  const auto r = seed * 13 * 17 + 0x234235;
  const auto g = seed * 7 * 3 * 5 + 0x773477;
  const auto b = seed * 11 * 19 + 0x223766;

  return Eigen::Vector3f((r & 0x00ff) / 255.0f, (g & 0x00ff) / 255.0f, (b & 0x00ff) / 255.0f);
}

// 物体にレイが衝突した場合の処理です。衝突判定は自動でやってくれます。

extern "C" __global__ void __closesthit__radiance() {
  *reinterpret_cast<Eigen::Vector3f *>(getPayloadPointer()) = getRandomColor(optixGetPrimitiveIndex()); // とりあえず、光が衝突したポリゴンのインデックスをシードにして、ランダムな色を割り当てます。
}
~~~

`optixGetPrimitiveIndex()`で取得できる何番目のポリゴンに衝突したのかの情報から適当にランダムな色を決めて、その色を設定しています。この方式なら、ポリゴン単位で同じ色になるでしょう。

そうそう、レイを追跡してみたのだけどオブジェクトに衝突しなかった場合の`__miss_xxx()`も作成しなければなりません。今回は、背景は真っ白にしてみました。

~~~c++
extern "C" __global__ void __miss__radiance() {
  *reinterpret_cast<Eigen::Vector3f *>(getPayloadPointer()) = Eigen::Vector3f{1, 1, 1}; // とりあえず、背景は真っ白にします。
}
~~~

あとは、Window.hの`osc::Window`の継承元をマウスでカメラをグリグリできる`osc::common::CameraWindow`に変えてカメラ関係の処理をちょっと追加して、あと、Renderer.hにもカメラの処理を追加してあげれば完了です。

プログラムを実行して、どぎつい色の三角形で構成された3Dオブジェクトが表示されたら成功です。マウスでグリグリしてみてください。お疲れさまでした。

![example04-first-triangle-mesh-linux]()

![example04-first-triangle-mesh-windows]()
