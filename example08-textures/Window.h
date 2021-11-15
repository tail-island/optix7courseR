#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "../common/Window.h"
#include "Renderer.h"

namespace osc {

class Window final : public common::CameraWindow {
  Renderer renderer_;

  auto setCamera() noexcept {
    const auto [u, v, w] = getUVW();

    renderer_.setCamera(Camera{camera_.getFrom(), u, v, w});
  }

public:
  Window(const std::string &title, const common::Camera &camera, const Model &model) noexcept : common::CameraWindow{title, camera}, renderer_{model} {
    setCamera();
  }

  void resize(const Eigen::Vector2i &size) noexcept override {
    setCamera();

    renderer_.resize(size.x(), size.y());
  }

  std::vector<std::uint32_t> render() noexcept override {
    if (isCameraMoved_) {
      setCamera();

      isCameraMoved_ = false;
    }

    return renderer_.render();
  }
};

} // namespace osc
