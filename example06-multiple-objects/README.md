# example06-multiple-objects

example05でそこそこ3Dグラフィックスっぽくはなったのですけど、色が単色で寂しい……。ポリゴンの数だけ色を格納するようなデータ構造、たとえば以下のようなコードは……

~~~c++
// やっては駄目な修正。

struct TriangleMeshes {
  Eigen::Vector3f *vertices;
  Eigen::Vector3i *indices;
  Eigen::Vector3f *colors; // *をつけて集合にしてみる。
};
~~~

メモリの無駄遣いなのでやっては駄目です。複数のポリゴンで色が同じだったりしますもんね。色の情報程度ならメモリの無駄遣いもまぁ許容できるかもしれませんけど、これがテクスチャ・マッピングの画像だったりすると、メモリがいくらあっても足りなくなってしまうでしょう。

なので、少し工夫して、モデル→ポリゴンの集合でなく、モデル→オブジェクト→ポリゴンの集合になる形に、Model.hを修正します。

~~~c++
class Object final {
  std::vector<Eigen::Vector3f> vertices_;
  std::vector<Eigen::Vector3i> indices_;
  Eigen::Vector3f color_;

public:
  const auto &getVertices() const noexcept {
    return vertices_;
  }

  const auto &getIndices() const noexcept {
    return indices_;
  }

  const auto &getColor() const noexcept {
    return color_;
  }

  auto setColor(const Eigen::Vector3f &color) noexcept {
    color_ = color;
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

class Model final {
  std::vector<Object> objects_;

public:
  const auto &getObjects() const noexcept {
    return objects_;
  }

  auto addObject(const Object &object) noexcept {
    objects_.emplace_back(object);
  }
};
~~~

モデルを修正したので、OptixState.hも修正しましょう。

~~~c++
class OptixState final {
  ...

  std::vector<common::DeviceVectorBuffer<Eigen::Vector3f>> verticesBuffers_;
  std::vector<common::DeviceVectorBuffer<Eigen::Vector3i>> indicesBuffers_;

  ...

  OptixState(int deviceId, const Model &model) noexcept {

    ...

    // OptixのTraversableHandleを生成します。

    std::transform(std::begin(model.getObjects()), std::end(model.getObjects()), std::back_inserter(verticesBuffers_), [](const Object &object) {
      return common::DeviceVectorBuffer<Eigen::Vector3f>{object.getVertices()};
    });

    std::transform(std::begin(model.getObjects()), std::end(model.getObjects()), std::back_inserter(indicesBuffers_), [](const Object &object) {
      return common::DeviceVectorBuffer<Eigen::Vector3i>{object.getIndices()};
    });

    [&] {
      const auto accelBuildOptions = [&] {
        auto result = OptixAccelBuildOptions{};

        result.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        result.motionOptions.numKeys = 1;
        result.operation = OPTIX_BUILD_OPERATION_BUILD;

        return result;
      }();

      const auto triangleArrayFlags = 0u;

      const auto buildInputs = [&] {
        auto result = std::vector<OptixBuildInput>{};

        for (auto i = 0; i < static_cast<int>(std::size(model.getObjects())); ++i) {
          result.emplace_back([&] {
            auto result = OptixBuildInput{};

            result.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

            result.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            result.triangleArray.vertexStrideInBytes = sizeof(Eigen::Vector3f);
            result.triangleArray.numVertices = verticesBuffers_[i].getSize();
            result.triangleArray.vertexBuffers = &verticesBuffers_[i].getData();

            result.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            result.triangleArray.indexStrideInBytes = sizeof(Eigen::Vector3i);
            result.triangleArray.numIndexTriplets = indicesBuffers_[i].getSize();
            result.triangleArray.indexBuffer = indicesBuffers_[i].getData();

            result.triangleArray.flags = &triangleArrayFlags;
            result.triangleArray.numSbtRecords = 1;
            result.triangleArray.sbtIndexOffsetBuffer = 0;
            result.triangleArray.sbtIndexOffsetSizeInBytes = 0;
            result.triangleArray.sbtIndexOffsetStrideInBytes = 0;

            return result;
          }());
        }

        return result;
      }();

      const auto accelBufferSizes = [&] {
        auto result = OptixAccelBufferSizes{};

        OPTIX_CHECK(optixAccelComputeMemoryUsage(deviceContext_, &accelBuildOptions, buildInputs.data(), std::size(buildInputs), &result));

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

      OPTIX_CHECK(optixAccelBuild(deviceContext_, 0, &accelBuildOptions, buildInputs.data(), std::size(buildInputs), tempBuffer.getData(), tempBuffer.getDataSize(), outputBuffer.getData(), outputBuffer.getDataSize(), &traversableHandle_, &accelEmitDesc, 1));

      traversableBuffer_.setSize(compactedSizeBuffer.get());

      OPTIX_CHECK(optixAccelCompact(deviceContext_, 0, traversableHandle_, traversableBuffer_.getData(), traversableBuffer_.getDataSize(), &traversableHandle_));
    }();

    ...

    // OptiXのシェーダー・バインディング・テーブルを作成します。

    shaderBindingTable_ = [&] {

      ...

      auto result = OptixShaderBindingTable{};

      ...

      [&] {
        const auto hitgroupRecords = [&] {
          auto result = std::vector<HitgroupRecord>{};

          for (auto i = 0; i < static_cast<int>(std::size(model.getObjects())); ++i) {
            for (const auto &programGroup : hitgroupProgramGroups_) {
              result.emplace_back([&] {
                auto result = HitgroupRecord{};

                OPTIX_CHECK(optixSbtRecordPackHeader(programGroup, &result));

                result.triangleMeshes.vertices = reinterpret_cast<Eigen::Vector3f *>(verticesBuffers_[i].getData());
                result.triangleMeshes.indices = reinterpret_cast<Eigen::Vector3i *>(indicesBuffers_[i].getData());
                result.triangleMeshes.color = model.getObjects()[i].getColor();

                return result;
              }());
            }
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

      ...

    }();
  }

  ...

}
~~~

いろいろ変わっているように見えるかもしれませんけど、やっていることはモデルのオブジェクト分のループを追加しただけ（嘘だと思うならexample04とのdiffをとってみてください）で、実は簡単な作業です。

あとは、モデルの変更に合わせてMain.cppを修正して、で、DeviceProgram.cuは「そのまま」（嘘だとおもうなら、やっぱりdiffをとってみてください）で作業は終了です。OptiXはもともとオブジェクトの集合としてデータを扱えるようになっていて、これまではたまたまオブジェクトが一つしかないという扱いになっていたんですな。

では、プログラムを実行してください。地面の色と立方体の色が異なっていれば成功です。お疲れさまでした。

![example06-multiple-objects-linux]()

![example06-multiple-objects-windows]()
