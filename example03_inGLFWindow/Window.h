#pragma once

#include <cstdint>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include "../common/Window.h"
#include "Renderer.h"

namespace osc {

class Window final : public common::Window {
  Renderer Renderer;

public:
  Window(const std::string &Title) : common::Window(Title), Renderer() {
    ;
  }

  void resize() noexcept {
    Renderer.resize(Width, Height);
  }

  std::vector<std::uint32_t> render() noexcept {
    return Renderer.render();
  }
};

} // namespace osc
