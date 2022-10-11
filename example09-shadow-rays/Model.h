#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

namespace osc {

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

struct LessIndexT {
  auto operator()(const tinyobj::index_t &index1, const tinyobj::index_t &index2) const noexcept {
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
};

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
          auto knownIndices = std::map<tinyobj::index_t, int, LessIndexT>{};

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

} // namespace osc
