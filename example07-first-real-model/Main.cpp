#include "Model.h"
#include "Window.h"

int main(int argc, char **argv) {
  const auto model = osc::Model("../data/dabrovic-sponza/sponza.obj");

  auto window = osc::Window("Optix 7 Course Example",
                            osc::common::Camera{Eigen::Vector3f{-1293.07, 154.681f, -0.7304f}, model.getBoundBox().center() - Eigen::Vector3f{0, 400, 0}, Eigen::Vector3f{0, 1, 0}},
                            model);

  window.run();
}
