#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "../common/Window.h"
#include "Renderer.h"

namespace osc {

class Window final : public common::CameraWindow {
  Renderer renderer_;

  auto setCamera() noexcept {
    const auto [u, v, w] = getUVW();

    renderer_.setCamera(Camera{*reinterpret_cast<const float3 *>(&camera_.getFrom()), *reinterpret_cast<const float3 *>(&u), *reinterpret_cast<const float3 *>(&v), *reinterpret_cast<const float3 *>(&w)});
  }

public:
  Window(const std::string &title, const common::Camera &camera, const Model &model, const Light &light) noexcept : common::CameraWindow{title, camera}, renderer_{model, light} {
    setCamera();
  }

  void resize(const Eigen::Vector2i &size) noexcept override {
    setCamera();

    renderer_.resize(size.x(), size.y());
  }

  std::vector<Eigen::Vector4f> render() noexcept override {
    if (isCameraMoved_) {
      setCamera();

      isCameraMoved_ = false;
    }

    return renderer_.render();
  }
};

} // namespace osc
