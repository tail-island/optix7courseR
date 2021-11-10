#pragma once

#include <algorithm>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace osc {

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

  auto addCube(const Eigen::Vector3f &center, const Eigen::Vector3f &size) noexcept {
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

} // namespace osc
