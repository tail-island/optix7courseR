#pragma once

#include <algorithm>
#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace osc {

class Model final {
  std::vector<Eigen::Vector3f> Vertexes;
  std::vector<Eigen::Vector3i> Indexes;

public:
  const auto &vertexes() const noexcept {
    return Vertexes;
  }

  const auto &indexes() const noexcept {
    return Indexes;
  }

  auto addCube(const Eigen::Vector3f &Center, const Eigen::Vector3f &Size) noexcept {
    const auto UnitCubeVertexes = std::vector<Eigen::Vector3f>{
        {0, 0, 0},
        {1, 0, 0},
        {0, 1, 0},
        {1, 1, 0},
        {0, 0, 1},
        {1, 0, 1},
        {0, 1, 1},
        {1, 1, 1}};

    const auto UnitCubeIndexes = std::vector<Eigen::Vector3i>{
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

    const auto FirstVertexIndex = std::size(Vertexes);
    const auto Affine = Eigen::Translation<float, 3>(Center - Size * 0.5) * Eigen::Scaling(Size);

    std::transform(std::begin(UnitCubeVertexes), std::end(UnitCubeVertexes), std::back_inserter(Vertexes), [&](const auto &UnitCubeVertex) {
      return Affine * UnitCubeVertex;
    });

    std::transform(std::begin(UnitCubeIndexes), std::end(UnitCubeIndexes), std::back_inserter(Indexes), [&](const auto &UnitCubeIndex) {
      return FirstVertexIndex + UnitCubeIndex.array();
    });
  }
};

} // namespace osc
