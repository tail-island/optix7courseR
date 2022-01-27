# example08-textures

example07で表現できたのは形状だけでした。それでは寂しいので、example08ではポリゴンにテクスチャーを貼ってかっこよくしてみましょう。

まずは、Model.hにテクスチャーを表現するクラスを追加します。

~~~c++
class Texture final {
  std::vector<std::uint32_t> image_;
  Eigen::Vector2i imageSize_;

public:
  Texture(const std::vector<std::uint32_t> &image, const Eigen::Vector2i &imageSize) noexcept : image_(image), imageSize_(imageSize) {
    ;
  }

  const auto &getImage() const noexcept {
    return image_;
  }

  const auto &getImageSize() const noexcept {
    return imageSize_;
  }
};
~~~

で、この`osc::Texture`を使用するように`osc::Object`を修正します。

~~~c++
class Object final {
  std::vector<Eigen::Vector3f> vertices_;
  std::vector<Eigen::Vector3f> normals_; // 頂点法線。面法線じゃありません。
  std::vector<Eigen::Vector2f> textureCoordinates_;
  std::vector<Eigen::Vector3i> indices_;

  Eigen::Vector3f color_;
  int textureIndex_;

public:
  Object(const std::vector<Eigen::Vector3f> &vertices, const std::vector<Eigen::Vector3f> &normals, const std::vector<Eigen::Vector2f> &textureCoordinates, const std::vector<Eigen::Vector3i> &indices, const Eigen::Vector3f &diffuse, int diffuseTextureIndex) noexcept : vertices_(vertices), normals_(normals), textureCoordinates_(textureCoordinates), indices_(indices), color_(diffuse), textureIndex_(diffuseTextureIndex) {
    ;
  }

  const auto &getVertices() const noexcept {
    return vertices_;
  }

  const auto &getNormals() const noexcept {
    return normals_;
  }

  const auto &getTextureCoordinates() const noexcept {
    return textureCoordinates_;
  }

  const auto &getIndices() const noexcept {
    return indices_;
  }

  const auto &getColor() const noexcept {
    return color_;
  }

  auto getTextureIndex() const noexcept {
    return textureIndex_;
  }
};
~~~

あ、せっかくテクスチャーを貼ってもポリゴン面の単位での法線だと継ぎ目がカクカクしちゃうので、頂点の単位で法線を持つことにしました（`normals_`）。あと、頂点がテクスチャーのどの位置にあるのかを表現する`textureCoordinates_`も追加しています。

あとは、`osc::Model`の属性に`osc::Texture`の集合（`textures_`）を追加して、モデルを読み込む処理を修正します。テクスチャーの読み込みには、前にも利用した超絶便利な[stb](https://github.com/nothings/stb)を使用しました。

~~~c++
class Model final {
  std::vector<Texture> textures_;
  std::vector<Object> objects_;

  Eigen::AlignedBox3f boundBox_;

public:
  Model(const std::string &path) noexcept {
    auto attrib = tinyobj::attrib_t{};
    auto shapes = std::vector<tinyobj::shape_t>{};
    auto materials = std::vector<tinyobj::material_t>{};
    auto error = std::string{};
    auto modelPath = path.substr(0, path.rfind('/') + 1); // へっぽこなやり方でごめんなさい。。。

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &error, path.c_str(), modelPath.c_str())) {
      std::cerr << error << std::endl;
      std::exit(1);
    }

    // テクスチャーを読み込みます。

    auto textureIndices = std::unordered_map<int, int>{};

    [&] {
      auto textureIndex = 0;

      for (auto i = 0; i < static_cast<int>(std::size(materials)); ++i) {
        if (materials[i].diffuse_texname.empty()) {
          textureIndices.emplace(i, -1);
          continue;
        }

        textureIndices.emplace(i, textureIndex++);

        textures_.emplace_back([&] {
          auto imageSize = Eigen::Vector2i{};
          int n;

          const auto imagePath = [&] {
            auto result = modelPath + materials[i].diffuse_texname;

            std::replace(std::begin(result), std::end(result), '\\', '/');

            return result;
          }();

          auto image = reinterpret_cast<std::uint32_t *>(stbi_load(imagePath.c_str(), &imageSize.x(), &imageSize.y(), &n, STBI_rgb_alpha));

          // テクスチャーは左下が原点らしいので、上下を反転させます。

          for (auto y = 0; y < imageSize.y() / 2; ++y) {
            auto line1 = image + imageSize.x() * y;
            auto line2 = image + imageSize.x() * (imageSize.y() - 1 - y);

            for (auto x = 0; x < imageSize.x(); ++x) {
              std::swap(line1[x], line2[x]);
            }
          }

          return Texture{std::vector<std::uint32_t>(image, image + imageSize.x() * imageSize.y()), imageSize};
        }());
      }
    }();

    // オブジェクトを読み込みます。

    [&] {
      for (const auto &shape : shapes) {
        auto materialIds = std::set(std::begin(shape.mesh.material_ids), std::end(shape.mesh.material_ids));

        for (const auto &materialId : materialIds) {
          auto knownIndices = std::map<tinyobj::index_t, int>{};

          auto vertices = std::vector<Eigen::Vector3f>{};
          auto normals = std::vector<Eigen::Vector3f>{};
          auto textureCoordinates = std::vector<Eigen::Vector2f>{};
          auto indices = std::vector<Eigen::Vector3i>{};

          for (auto i = 0; i < std::size(shape.mesh.material_ids); ++i) {
            if (shape.mesh.material_ids[i] != materialId) {
              continue;
            }

            indices.emplace_back([&] {
              auto result = Eigen::Vector3i{};

              for (auto j = 0; j < 3; ++j) {
                result[j] = [&] {
                  const auto result = static_cast<int>(std::size(vertices));

                  const auto &index = shape.mesh.indices[i * 3 + j];

                  const auto [it, emplaced] = knownIndices.emplace(index, result);

                  if (!emplaced) {
                    return it->second;
                  }

                  vertices.emplace_back(Eigen::Vector3f{attrib.vertices[index.vertex_index * 3 + 0], attrib.vertices[index.vertex_index * 3 + 1], attrib.vertices[index.vertex_index * 3 + 2]});
                  normals.emplace_back(Eigen::Vector3f{attrib.normals[index.normal_index * 3 + 0], attrib.normals[index.normal_index * 3 + 1], attrib.normals[index.normal_index * 3 + 2]});
                  textureCoordinates.emplace_back(Eigen::Vector2f{attrib.texcoords[index.texcoord_index * 2 + 0], attrib.texcoords[index.texcoord_index * 2 + 1]});

                  return result;
                }();
              }

              return result;
            }());
          }

          objects_.emplace_back(vertices, normals, textureCoordinates, indices, *reinterpret_cast<Eigen::Vector3f *>(&materials[materialId].diffuse), textureIndices[materialId]);
        }
      }
    }();

    // バウンド・ボックスを作成します。

    [&] {
      for (const auto &object : objects_) {
        for (const auto &vertex : object.getVertices()) {
          boundBox_.extend(vertex);
        }
      }
    }();
  }

  const auto &getTextures() const noexcept {
    return textures_;
  }

  const auto &getObjects() const noexcept {
    return objects_;
  }

  const auto &getBoundBox() const noexcept {
    return boundBox_;
  }
};
~~~

このモデルの情報は最終的にはGPU側で動作するプログラムで使用したいわけなので、OptixParam.hの中の`osc::TriangleMeshes`を修正します。

~~~c++
struct TriangleMeshes {
  Eigen::Vector3f *vertices;
  Eigen::Vector3f *normals;
  Eigen::Vector2f *textureCoordinates;
  Eigen::Vector3i *indices;

  bool hasTextureObject;
  cudaTextureObject_t textureObject;

  Eigen::Vector3f color;
};
~~~

この中の`cudaTextureObject_t`ってのは、テクスチャーのオブジェクトを表現する型です。このコードを見ると、テクスチャーの実体を表現しているように思えて、だから異なるオブジェクトが同じテクスチャーを使う場合にメモリの無駄が発生するように感じたかもしれませんけど、実は`cudaTextureObject_t`は`unsigned long long`のエイリアスで、テクスチャーのIDを表現しているものなのでこれで大丈夫です。まぁ、その代わりに、テクスチャーの有無を表現する`hasTextureObject`という変数が追加しなければならなかったわけですけど……。

で、この`osc::TriangleMeshes`に追加された属性分のデータを作成する処理を、OptixState.hに追加します。

~~~c++
class OptixState final {

  ...

  std::vector<common::DeviceVectorBuffer<Eigen::Vector3f>> verticesBuffers_;
  std::vector<common::DeviceVectorBuffer<Eigen::Vector3f>> normalsBuffers_;
  std::vector<common::DeviceVectorBuffer<Eigen::Vector2f>> textureCoordinatesBuffers_;
  std::vector<common::DeviceVectorBuffer<Eigen::Vector3i>> indicesBuffers_;

  ...

  OptixState(int deviceId, const Model &model) noexcept {

    ...

    // テクスチャーを作成します。

    [&] {
      for (const auto &texture : model.getTextures()) {
        auto textureArray = cudaArray_t{};
        auto channelDesc = cudaCreateChannelDesc<uchar4>();

        CUDA_CHECK(cudaMallocArray(&textureArray, &channelDesc, texture.getImageSize().x(), texture.getImageSize().y()));

        CUDA_CHECK(cudaMemcpy2DToArray(textureArray,
                                       0,
                                       0,
                                       texture.getImage().data(),
                                       texture.getImageSize().x() * sizeof(std::uint32_t),
                                       texture.getImageSize().x() * sizeof(std::uint32_t),
                                       texture.getImageSize().y(),
                                       cudaMemcpyHostToDevice));

        textureArrays_.emplace_back(textureArray);

        auto resourceDesc = [&] {
          auto result = cudaResourceDesc{};

          result.resType = cudaResourceTypeArray;
          result.res.array.array = textureArray;

          return result;
        }();

        auto textureDesc = [&] {
          auto result = cudaTextureDesc{};

          result.addressMode[0] = cudaAddressModeWrap;
          result.addressMode[1] = cudaAddressModeWrap;
          result.filterMode = cudaFilterModeLinear;
          result.readMode = cudaReadModeNormalizedFloat;
          result.normalizedCoords = 1;
          result.maxAnisotropy = 1;
          result.maxMipmapLevelClamp = 99;
          result.minMipmapLevelClamp = 0;
          result.mipmapFilterMode = cudaFilterModePoint;
          result.borderColor[0] = 1.0f;
          result.sRGB = 0;

          return result;
        }();

        auto textureObject = cudaTextureObject_t{};

        CUDA_CHECK(cudaCreateTextureObject(&textureObject, &resourceDesc, &textureDesc, nullptr));

        textureObjects_.emplace_back(textureObject);
      }
    }();

    // 後続処理のために、モデルのオブジェクトの各属性を抽出します。

    std::transform(std::begin(model.getObjects()), std::end(model.getObjects()), std::back_inserter(verticesBuffers_), [](const Object &object) {
      return common::DeviceVectorBuffer<Eigen::Vector3f>{object.getVertices()};
    });

    std::transform(std::begin(model.getObjects()), std::end(model.getObjects()), std::back_inserter(normalsBuffers_), [](const Object &object) {
      return common::DeviceVectorBuffer<Eigen::Vector3f>{object.getNormals()};
    });

    std::transform(std::begin(model.getObjects()), std::end(model.getObjects()), std::back_inserter(textureCoordinatesBuffers_), [](const Object &object) {
      return common::DeviceVectorBuffer<Eigen::Vector2f>{object.getTextureCoordinates()};
    });

    std::transform(std::begin(model.getObjects()), std::end(model.getObjects()), std::back_inserter(indicesBuffers_), [](const Object &object) {
      return common::DeviceVectorBuffer<Eigen::Vector3i>{object.getIndices()};
    });

    ...


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
                result.triangleMeshes.normals = reinterpret_cast<Eigen::Vector3f *>(normalsBuffers_[i].getData());
                result.triangleMeshes.textureCoordinates = reinterpret_cast<Eigen::Vector2f *>(textureCoordinatesBuffers_[i].getData());
                result.triangleMeshes.indices = reinterpret_cast<Eigen::Vector3i *>(indicesBuffers_[i].getData());
                result.triangleMeshes.color = model.getObjects()[i].getColor();

                if (model.getObjects()[i].getTextureIndex() >= 0) {
                  result.triangleMeshes.hasTextureObject = true;
                  result.triangleMeshes.textureObject = textureObjects_[model.getObjects()[i].getTextureIndex()];
                } else {
                  result.triangleMeshes.hasTextureObject = false;
                }

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

      return result;
    }();
  }

  ...

};
~~~

テクスチャーはCUDAのテクスチャーのままで、OptiXだからといって特別なところはありません。残りの部分も、データ構造の変更に合わせているだけで、特別なことはしていません。

以上で情報がGPUに渡るようにありましたので、お待ちかねのGPU側のコードを書きましょう。

まずは、ポリゴンの頂点のインデックスと、重心座標系でレイが衝突した位置を取得します。

~~~c++
const auto &index = triangleMeshes.indices[optixGetPrimitiveIndex()];

const auto u = optixGetTriangleBarycentrics().x;
const auto v = optixGetTriangleBarycentrics().y;
~~~

ここまで分かれば、CUDAの機能を使用したテクスチャーのレイが衝突した部分の色を取得したり、法線を求めたりすることができます。

~~~c++
// レイが衝突した場所の色を取得します。

const auto color = [&] {
  if (!triangleMeshes.hasTextureObject) {
    return triangleMeshes.color;
  }

  const auto textureCoordinate = (1 - u - v) * triangleMeshes.textureCoordinates[index.x()] + u * triangleMeshes.textureCoordinates[index.y()] + v * triangleMeshes.textureCoordinates[index.z()];
  const auto textureColor = tex2D<float4>(triangleMeshes.textureObject, textureCoordinate.x(), textureCoordinate.y());

  return Eigen::Vector3f{textureColor.x, textureColor.y, textureColor.z};
}();

// レイが衝突した場所の法線を取得します。

const auto normal = [&] {
  return ((1 - u - v) * triangleMeshes.normals[index.x()] + u * triangleMeshes.normals[index.y()] + v * triangleMeshes.normals[index.z()]).normalized();
}();
~~~

で、残りはこれまでと同じ。これで、テクスチャーが張り付いた立派な3Dグラフィックスが表示されます。プログラムを実行して、テクスチャーで装飾された画像が表示されたら作業は終了です。お疲れさまでした。

![example08-textures-linux]()

![example08-textures-windows]()
