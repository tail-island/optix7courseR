#pragma once

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <set>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

namespace osc {

class Object final {
  std::vector<Eigen::Vector3f> vertices_;
  std::vector<Eigen::Vector3f> normals_; // 頂点法線。面法線じゃありません。
  std::vector<Eigen::Vector2f> texcoords_;
  std::vector<Eigen::Vector3i> indices_;

  Eigen::Vector3f diffuse_;

public:
  Object(const std::vector<Eigen::Vector3f> &vertices, const std::vector<Eigen::Vector3f> &normals, const std::vector<Eigen::Vector2f> &textureCoordinates, const std::vector<Eigen::Vector3i> &indices, const Eigen::Vector3f &diffuse) noexcept : vertices_(vertices), normals_(normals), texcoords_(textureCoordinates), indices_(indices), diffuse_(diffuse) {
    ;
  }

  const auto &getVertices() const noexcept {
    return vertices_;
  }

  const auto &getNormals() const noexcept {
    return normals_;
  }

  const auto &getTexcoords() const noexcept {
    return texcoords_;
  }

  const auto &getIndices() const noexcept {
    return indices_;
  }

  const auto &getDiffuse() const noexcept {
    return diffuse_;
  }
};

class Model final {
  std::vector<Object> objects_;
  Eigen::AlignedBox3f boundBox_;

  static auto getRandomDiffuse(unsigned int seed) noexcept {
    const auto r = seed * 13 * 17 + 0x234235;
    const auto g = seed * 7 * 3 * 5 + 0x773477;
    const auto b = seed * 11 * 19 + 0x223766;

    return Eigen::Vector3f((r & 0x00ff) / 255.0f, (g & 0x00ff) / 255.0f, (b & 0x00ff) / 255.0f);
  }

  static auto createObjects(const std::string &path) noexcept {
    auto result = std::vector<Object>{};

    auto attrib = tinyobj::attrib_t{};
    auto shapes = std::vector<tinyobj::shape_t>{};
    auto materials = std::vector<tinyobj::material_t>{};
    auto error = std::string{};
    auto materialPath = path.substr(0, path.rfind('/') + 1); // へっぽこなやり方でごめんなさい。。。

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &error, path.c_str(), materialPath.c_str())) {
      std::cerr << error << std::endl;
      std::exit(1);
    }

    for (const auto &shape : shapes) {
      auto materialIds = std::set(std::begin(shape.mesh.material_ids), std::end(shape.mesh.material_ids));

      for (const auto &materialId : materialIds) {
        auto vertices = std::vector<Eigen::Vector3f>{};
        auto normals = std::vector<Eigen::Vector3f>{};
        auto texcoords = std::vector<Eigen::Vector2f>{};
        auto indices = std::vector<Eigen::Vector3i>{};

        for (auto i = 0; i < std::size(shape.mesh.material_ids); ++i) {
          if (shape.mesh.material_ids[i] != materialId) {
            continue;
          }

          auto localIndex = Eigen::Vector3i{};

          for (auto j = 0; j < 3; ++j) {
            const auto &index = shape.mesh.indices[i * 3 + j];

            const auto &vertex = Eigen::Vector3f{attrib.vertices[index.vertex_index * 3 + 0], attrib.vertices[index.vertex_index * 3 + 1], attrib.vertices[index.vertex_index * 3 + 2]};

            localIndex[j] = [&] {
              const auto it = std::find(std::begin(vertices), std::end(vertices), vertex);

              if (it == std::end(vertices)) {
                vertices.emplace_back(vertex);

                const auto &normal = Eigen::Vector3f(attrib.normals[index.normal_index * 3 + 0], attrib.normals[index.normal_index * 3 + 1], attrib.normals[index.normal_index * 3 + 2]);
                normals.emplace_back(normal);

                const auto &texcoord = Eigen::Vector2f(attrib.texcoords[index.texcoord_index * 2 + 0], attrib.texcoords[index.texcoord_index * 2 + 1]);
                texcoords.emplace_back(texcoord);

                return static_cast<int>(std::size(vertices)) - 1;
              }

              return static_cast<int>(std::distance(std::begin(vertices), it));
            }();
          }

          indices.emplace_back(localIndex);
        }

        result.emplace_back(vertices, normals, texcoords, indices, getRandomDiffuse(materialId));
      }
    }

    return result;
  }

  static auto createBoundBox(const std::vector<Object> &objects) noexcept {
    auto result = Eigen::AlignedBox3f{};

    for (const auto &object : objects) {
      for (const auto &vertex : object.getVertices()) {
        result.extend(vertex);
      }
    }

    return result;
  }

public:
  Model(const std::string &path) noexcept : objects_(createObjects(path)), boundBox_(createBoundBox(objects_)) {
    ;
  }

  const auto &getObjects() const noexcept {
    return objects_;
  }

  const auto &getBoundBox() const noexcept {
    return boundBox_;
  }
};

} // namespace osc
