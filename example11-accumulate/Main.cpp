#include "Model.h"
#include "Window.h"

int main(int argc, char **argv) {
  const auto model = osc::Model("../data/dabrovic-sponza/sponza.obj");

  const auto camera = osc::common::Camera{Eigen::Vector3f{-1293.07, 154.681f, -0.7304f}, model.getBoundBox().center() - Eigen::Vector3f{0, 400, 0}, Eigen::Vector3f{0, 1, 0}};

  const auto light = [&] {
    const auto size = 200.0f;

    return osc::Light{{-1000.0f - size, 800.0f, -size}, {2.0f * size, 0.0f, 0.0f}, {0.0f, 0.0f, 2.0f * size}, {3000000.0f, 3000000.0f, 3000000.0f}};
  }();

  osc::Window("Optix 7 Course Example", camera, model, light).run();
}
