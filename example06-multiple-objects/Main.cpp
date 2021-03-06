#include "Model.h"
#include "Window.h"

int main(int argc, char **argv) {
  const auto model = [] {
    auto result = osc::Model{};

    // 地面。
    result.addObject([] {
      auto result = osc::Object{};

      result.addRectangular(Eigen::Vector3f{0, -1.5, 0}, Eigen::Vector3f{10, 0.1, 10});

      result.setColor(Eigen::Vector3f{0, 1, 0});

      return result;
    }());

    // 立方体。
    result.addObject([] {
      auto result = osc::Object{};

      result.addRectangular(Eigen::Vector3f{0, 0, 0}, Eigen::Vector3f{2, 2, 2});

      result.setColor(Eigen::Vector3f{0, 1, 1});

      return result;
    }());

    return result;
  }();

  osc::Window("Optix 7 Course Example", {Eigen::Vector3f{-10, 2, -12}, Eigen::Vector3f{0, 0, 0}, Eigen::Vector3f{0, 1, 0}}, model).run();
}
