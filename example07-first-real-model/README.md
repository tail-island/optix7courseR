# example07-first-real-model

example06で標準的な3Dグラフィックスのモデルになりましたので、これなら既存のかっこいい3Dモデルを表示できるんじゃないかな？

というわけで、ネタ元である[optix7course](https://github.com/ingowald/optix7course)に従って、[McGuire Computer Graphics Archive](https://casual-effects.com/data/)のCrytek Spozaのモデルを使用してみましょう。モデルの読み込み処理は、ネタ元と同じ[tinyobjectloader](https://github.com/tinyobjloader/tinyobjloader)を使用します。

というわけで、さくっとモデルを書き換えます。

~~~c++
namespace std {
  inline auto operator<(const tinyobj::index_t &index1, const tinyobj::index_t &index2) {
    if (index1.vertex_index < index2.vertex_index) {
      return true;
    }

    if (index1.vertex_index > index2.vertex_index) {
      return false;
    }

    if (index1.normal_index < index2.normal_index) {
      return true;
    }

    if (index1.normal_index > index2.normal_index) {
      return false;
    }

    if (index1.texcoord_index < index2.texcoord_index) {
      return true;
    }

    if (index1.texcoord_index > index2.texcoord_index) {
      return false;
    }

    return false;
  }
}

namespace osc {

class Object final {
  std::vector<Eigen::Vector3f> vertices_;
  std::vector<Eigen::Vector3i> indices_;

  Eigen::Vector3f diffuse_;

public:
  Object(const std::vector<Eigen::Vector3f> &vertices, const std::vector<Eigen::Vector3i> &indices, const Eigen::Vector3f &diffuse) noexcept : vertices_(vertices), indices_(indices), diffuse_(diffuse) {
    ;
  }

  const auto &getVertices() const noexcept {
    return vertices_;
  }

  const auto &getIndices() const noexcept {
    return indices_;
  }

  const auto &getDiffuse() const noexcept {
    return diffuse_;
  }
};

inline auto getRandomColor(unsigned int seed) noexcept {
  const auto r = seed * 13 * 17 + 0x234235;
  const auto g = seed * 7 * 3 * 5 + 0x773477;
  const auto b = seed * 11 * 19 + 0x223766;

  return Eigen::Vector3f((r & 0x00ff) / 255.0f, (g & 0x00ff) / 255.0f, (b & 0x00ff) / 255.0f);
}

class Model final {
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

    // オブジェクトを読み込みます。

    [&] {
      for (const auto &shape : shapes) {
        auto materialIds = std::set(std::begin(shape.mesh.material_ids), std::end(shape.mesh.material_ids));

        for (const auto &materialId : materialIds) {
          auto knownIndices = std::map<tinyobj::index_t, int>{};

          auto vertices = std::vector<Eigen::Vector3f>{};
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

                  return result;
                }();
              }

              return result;
            }());
          }

          objects_.emplace_back(vertices, indices, getRandomColor(materialId));
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

  const auto &getObjects() const noexcept {
    return objects_;
  }

  const auto &getBoundBox() const noexcept {
    return boundBox_;
  }
};

} // namespace osc
~~~

tinyobjectloaderは便利で簡単で素晴らしい！　これで、モデルの読み込み処理が終了しました。

あとは、Main.cppのモデルを生成したりカメラの位置を調整したりする部分を修正すれば、他はそのままで作業は完了です。example06までのプログラムって、見た目はしょぼかったですけど、実はかなり本格派なプログラムだったのですね……。

実行してみましょう。オブジェクトにランダムな色を設定したのでちょっとアレだけどクロアチアのスポンザ宮殿っぽい画像が表示されたら作業は終了です。お疲れさまでした。

![example07-first-real-model-linux](https://raw.githubusercontent.com/tail-island/optix7courseR/main/image/example07-first-real-model-linux.png)

![example07-first-real-model-windows](https://raw.githubusercontent.com/tail-island/optix7courseR/main/image/example07-first-real-model-windows.png)
