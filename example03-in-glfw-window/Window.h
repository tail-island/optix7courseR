#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "../common/Window.h"
#include "Renderer.h"

namespace osc {

class Window final : public common::Window {
  Renderer renderer_;

public:
  Window(const std::string &title) noexcept : common::Window{title}, renderer_{} {
    ;
  }

  void resize(const Eigen::Vector2i &size) noexcept override {
    renderer_.resize(size.x(), size.y());
  }

  std::vector<Eigen::Vector4f> render() noexcept override {
    return renderer_.render();
  }
};

} // namespace osc
