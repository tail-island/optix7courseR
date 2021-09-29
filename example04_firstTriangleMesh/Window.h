#pragma once

#include "../common/WindowBase.h"
#include "Renderer.h"

namespace osc {

class Window final : public WindowBase {
  Renderer Renderer;

public:
  Window(const std::string &Title, const osc::Model& Model) : WindowBase(Title), Renderer(Model) {
    ;
  }

  void resize(int Width, int Height) noexcept {
    Renderer.resize(Width, Height);
  }

  std::vector<std::uint32_t> render() noexcept {
    const auto [U, V, W] = createUVW(Eigen::Vector3f{-10, 2, -12}, Eigen::Vector3f{0, 0, 0}, Eigen::Vector3f{0, 1, 0}, 35, static_cast<float>(Width) / Height);
    Renderer.setCamera(Camera{Eigen::Vector3f(-10, 2, -12), U, V, W});

    return Renderer.render();
  }
};

} // namespace osc
