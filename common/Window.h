#include <cstdint>
#include <iostream>
#include <tuple>
#include <vector>

#include <Eigen/Core>
#ifdef _WIN32
#include <windows.h>
#endif
#include <GL/gl.h>
#include <GLFW/glfw3.h>
#define _USE_MATH_DEFINES
#include <math.h>

namespace osc {
namespace common {

class Window {
protected:
  int Width;
  int Height;

  GLFWwindow *GLFWWindow;

public:
  Window(const std::string &Title) noexcept : Width(0), Height(0) {
    glfwSetErrorCallback([](auto error, auto description) {
      std::cerr << "Error: " << description << "\n";
    });

    if (!glfwInit()) {
      std::exit(1);
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2); // 本当はOpenGL4.xにしたいのですけど、そうするとglDrawPixelsで描画できません。
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0); // そもそもglDrawPixelsみたいな遅いのを使うな、CUDAで作った画像をホスト・メモリ経由で表示するのは無駄だと言われたらぐうの音も出ないのですけど、とりあえずは無視で。

    GLFWWindow = glfwCreateWindow(1200, 800, Title.c_str(), NULL, NULL);
    if (!GLFWWindow) {
      std::exit(1);
    }

    glfwSetWindowUserPointer(GLFWWindow, this);
    glfwMakeContextCurrent(GLFWWindow);
    glfwSwapInterval(1);
  }

  virtual ~Window() {
    glfwDestroyWindow(GLFWWindow);
    glfwTerminate();
  }

  std::tuple<double, double> cursorPos() const noexcept {
    double X;
    double Y;
    glfwGetCursorPos(GLFWWindow, &X, &Y);

    return std::make_tuple(X, Y);
  }

  virtual void key(int Key, int Action, int Mods) noexcept {
    ;
  }

  virtual void mouseButton(int Button, int Action, int Mods) noexcept {
    ;
  }

  virtual void cursorPos(double X, double Y) noexcept {
    ;
  }

  virtual void resize() noexcept = 0;

  virtual std::vector<std::uint32_t> render() noexcept = 0;

  auto run() noexcept {
    glfwGetFramebufferSize(GLFWWindow, &Width, &Height);
    resize();

    glfwSetFramebufferSizeCallback(GLFWWindow, [](GLFWwindow *GLFWWindow, int Width, int Height) {
      auto Window = static_cast<common::Window *>(glfwGetWindowUserPointer(GLFWWindow));

      Window->Width = Width;
      Window->Height = Height;

      Window->resize();
    });

    glfwSetKeyCallback(GLFWWindow, [](GLFWwindow *GLFWWindow, int Key, int Scancode, int Action, int Mods) {
      static_cast<Window *>(glfwGetWindowUserPointer(GLFWWindow))->key(Key, Action, Mods);
    });

    glfwSetMouseButtonCallback(GLFWWindow, [](GLFWwindow *GLFWWindow, int Button, int Action, int Mods) {
      static_cast<Window *>(glfwGetWindowUserPointer(GLFWWindow))->mouseButton(Button, Action, Mods);
    });

    glfwSetCursorPosCallback(GLFWWindow, [](GLFWwindow *GLFWWindow, double X, double Y) {
      static_cast<Window *>(glfwGetWindowUserPointer(GLFWWindow))->cursorPos(X, Y);
    });

    while (!glfwWindowShouldClose(GLFWWindow)) {
      glDrawPixels(Width, Height, GL_RGBA, GL_UNSIGNED_BYTE, render().data()); // TODO: glDrawPixelsを使うのをやめて、OpenGLのバージョンを上げる。

      glfwSwapBuffers(GLFWWindow);
      glfwPollEvents();
    }
  }

  class Camera final {
    Eigen::Vector3f From;
    Eigen::Vector3f Up;

    float PointOfInterestDistance;

    bool Modified;

  public:
    const auto& from() const noexcept {
      return From;
    }

    auto At() const noexcept {
      return 0;
    }

    const auto& up() const noexcept {
      return Up;
    }
  };

  inline auto createUVW(const Eigen::Vector3f &From, const Eigen::Vector3f &At, const Eigen::Vector3f &Up, float Fov, float AspectRatio) noexcept {
    auto W = At - From;
    auto U = W.cross(Up).normalized();
    auto V = U.cross(W).normalized();

    const auto vlen = W.norm() * tanf(0.5f * Fov * static_cast<float>(M_PI) / 180.0f);
    const auto ulen = vlen * AspectRatio;

    V *= vlen;
    U *= ulen;

    return std::make_tuple(U, V, W);
  }
};

} // namespace common
} // namespace osc