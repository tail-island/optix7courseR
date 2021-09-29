#include "Model.h"
#include "Window.h"

int main(int Argc, char **Argv) {
  auto Model = [] {
    auto Result = osc::Model();

    // 地面。
    Result.addCube(Eigen::Vector3f{0, -1.5, 0}, Eigen::Vector3f{10, .1, 10});

    // 立方体。
    Result.addCube(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(2, 2, 2));

    return Result;
  }();

  osc::Window("Optix 7 Course Example", Model).run();
}
