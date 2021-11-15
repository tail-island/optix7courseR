#include "Model.h"
#include "Window.h"

int main(int argc, char **argv) {
  auto model = osc::Model("../data/dabrovic-sponza/sponza.obj");

  osc::Window("Optix 7 Course Example", {Eigen::Vector3f{-1293.07, 154.681f, -0.7304f}, model.getBoundBox().center() - Eigen::Vector3f{0, 400, 0}, Eigen::Vector3f{0, 1, 0}}, model).run();
}
