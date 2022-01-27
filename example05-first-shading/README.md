# example05-first-shading

example04のような、どぎつい色がのっぺりと塗られている程度では3Dグラフィックスとは呼べない気がする……。example05で、影をつけてちょっとそれらしくしてみましょう。カメラから光が出ていると仮定して、光が垂直にあたっている場合は明るく、斜めの場合は暗くなるようにしてみます。

そのためにはどうすればよいか、整理しながら少しずつ考えてみます。

3Dオブジェクトはポリゴンで構成されていて、そのポリゴンの法線（面がどちらを向いているのか）は、三角形の頂点から計算できたはず。で、OptiXはレイ・トレーシングをしているわけで、だから現在のレイの向きは管理しているはず。

あとは、レイの向きと法線を明るさに変換できればよいわけで、それは内積として計算できるはず。ポリゴンの裏も表として扱いたいなら、向きが反対なので内積がマイナスになった場合に対応するために絶対値を求めればOK。

というわけで、ポリゴンの頂点が分かれば明るさを決定できそうということになりました。でも、これまでのやり方だと、そもそもポリゴンの頂点の情報を取得できません……。

だから、Shader Binding Table（OptiXのドキュメントではSBTと省略されていることが多い。具体的な型は`OptixShaderBindingTable`）にデータを格納することにしましょう。

まずは、三角形のポリゴンの集合を表現する型を定義します。この型はCPU側からもGPU側からもアクセスできなければならないので、OptixParams.hに定義します。

~~~c++
struct TriangleMeshes {
  Eigen::Vector3f *vertices;
  Eigen::Vector3i *indices;
  Eigen::Vector3f color;
};
~~~

その上で、Shader Binding Tableを作成しているのはOptixState.hの中で、衝突時の操作の際に使用されるデータの`HitgroupRecord`に属性を追加します。

~~~c+++
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord {
  char header[OPTIX_SBT_RECORD_HEADER_SIZE];

  TriangleMeshes triangleMeshes;
};
~~~

`HitgroupRecord`を作成する部分にも修正が必要です。

~~~c++
auto result = OptixShaderBindingTable{};

...

[&] {
  const auto hitgroupRecords = [&] {
    auto result = std::vector<HitgroupRecord>{};

    for (const auto &programGroup : hitgroupProgramGroups_) {
      result.emplace_back([&] {
        auto result = HitgroupRecord{};

        OPTIX_CHECK(optixSbtRecordPackHeader(programGroup, &result));

        result.triangleMeshes.vertices = reinterpret_cast<Eigen::Vector3f *>(verticesBuffer_.getData());
        result.triangleMeshes.indices = reinterpret_cast<Eigen::Vector3i *>(indicesBuffer_.getData());
        result.triangleMeshes.color = model.getColor();

        return result;
      }());
    }

    return result;
  }();

  result.hitgroupRecordBase = [&] {
    void *result;

    CUDA_CHECK(cudaMalloc(&result, sizeof(HitgroupRecord) * std::size(hitgroupRecords)));
    CUDA_CHECK(cudaMemcpy(result, hitgroupRecords.data(), sizeof(HitgroupRecord) * std::size(hitgroupRecords), cudaMemcpyHostToDevice));

    return reinterpret_cast<CUdeviceptr>(result);
  }();

  result.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
  result.hitgroupRecordCount = std::size(hitgroupRecords);
}();
~~~

すみません、まだ修正が必要でした。OptixState.hに定義した`HitgroupRecord`にはヘッダ情報が含まれていて、でも、GPU側のプログラムからShader Binding Tableのデータにアクセスする際にはこのヘッダが含まれていない型になるんですよ……。自由度が高くて素晴らしいと考えるべきなのか、高水準でよく気がつくAPIが提供されていないと考えるべきなのかは分かりませんけど、しょうがないので`DeviceProgram.cu`にGPU側のための型を追加します。

~~~c++
struct HitgroupData {
  TriangleMeshes triangleMeshes;
};
~~~

以上でポリゴンの情報にアクセスするための準備は完了です。さっそく、DeviceProgram.cuに明るさを設定する処理を追加しましょう。

~~~c++
extern "C" __global__ void __closesthit__radiance() {
  const auto &triangleMeshes = reinterpret_cast<HitgroupData *>(optixGetSbtDataPointer())->triangleMeshes;

  // ポリゴンの法線を取得します。

  const auto normal = [&] {
    const auto &index = triangleMeshes.indices[optixGetPrimitiveIndex()];

    const auto &vertex1 = triangleMeshes.vertices[index.x()];
    const auto &vertex2 = triangleMeshes.vertices[index.y()];
    const auto &vertex3 = triangleMeshes.vertices[index.z()];

    return (vertex2 - vertex1).cross(vertex3 - vertex1).normalized();
  }();

  // レイの向きを取得します。

  const auto rayDirection = [&] {
    auto result = optixGetWorldRayDirection();

    return *reinterpret_cast<Eigen::Vector3f *>(&result);
  }();

  // 色は、光源とかはとりあえず考慮しないで、レイとポリゴンが垂直なほど明るくなるということで。カメラにライトが付いているとでも思って、納得してください……。

  *reinterpret_cast<Eigen::Vector3f *>(getPayloadPointer()) = triangleMeshes.color * (0.2 + 0.8 * std::abs(normal.dot(rayDirection)));
}
~~~

あとは、今回はポリゴンの色をランダムではなく事前に設定するようにしましたので、Model.hとMain.cppにそのための小さな修正を加えれば作業は完了です。

プログラムを実行して、マウスで視点を変更すると3Dオブジェクトのポリゴンの明るさがイイ感じに変更されたら成功です。お疲れさまでした。

![example05-first-shading-linux]()

![example05-first-shading-windows]()
