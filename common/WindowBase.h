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

class WindowBase {
protected:
  GLFWwindow *Handle;

  int Width;
  int Height;

  GLuint Texture;

public:
  WindowBase(const std::string &Title) noexcept : Width(0), Height(0), Texture(0) {
    glfwSetErrorCallback([](auto error, auto description) {
      std::cerr << "Error: " << description << "\n";
    });

    if (!glfwInit()) {
      std::exit(1);
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2); // 本当はOpenGL4.xにしたいのですけど、そうするとglDrawPixelsで描画できません。
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0); // そもそもglDrawPixelsみたいな遅いのを使うな、CUDAで作った画像をホスト・メモリ経由で表示するのはやめろと言われたらぐうの音も出ないのですけど、とりあえずは無視で。

    Handle = glfwCreateWindow(1200, 800, Title.c_str(), NULL, NULL);
    if (!Handle) {
      std::exit(1);
    }

    glfwSetWindowUserPointer(Handle, this);
    glfwMakeContextCurrent(Handle);
    glfwSwapInterval(1);
  }

  virtual ~WindowBase() {
    glfwDestroyWindow(Handle);
    glfwTerminate();
  }

  std::tuple<double, double> cursorPos() const noexcept {
    double X;
    double Y;
    glfwGetCursorPos(Handle, &X, &Y);

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
    glfwGetFramebufferSize(Handle, &Width, &Height);
    resize();

    glfwSetFramebufferSizeCallback(Handle, [](GLFWwindow *Window, int Width, int Height) {
      auto WindowBase = static_cast<osc::WindowBase *>(glfwGetWindowUserPointer(Window));

      WindowBase->Width = Width;
      WindowBase->Height = Height;

      WindowBase->resize();
    });

    glfwSetKeyCallback(Handle, [](GLFWwindow *Window, int Key, int Scancode, int Action, int Mods) {
      static_cast<WindowBase *>(glfwGetWindowUserPointer(Window))->key(Key, Action, Mods);
    });

    glfwSetMouseButtonCallback(Handle, [](GLFWwindow *Window, int Button, int Action, int Mods) {
      static_cast<WindowBase *>(glfwGetWindowUserPointer(Window))->mouseButton(Button, Action, Mods);
    });

    glfwSetCursorPosCallback(Handle, [](GLFWwindow *Window, double X, double Y) {
      static_cast<WindowBase *>(glfwGetWindowUserPointer(Window))->cursorPos(X, Y);
    });

    while (!glfwWindowShouldClose(Handle)) {
      glDrawPixels(Width, Height, GL_RGBA, GL_UNSIGNED_BYTE, render().data()); // TODO: glDrawPixelsを使うのをやめて、OpenGLのバージョンを上げる。

      glfwSwapBuffers(Handle);
      glfwPollEvents();
    }
  }

  // class Camera final {
  //   Eigen::Vector3f Position;
  //   Eigen::Vector3f UpVector;
  //   float PointOfInterestDistance;

  // public:
  // }

  inline auto createUVW(const Eigen::Vector3f& From, const Eigen::Vector3f& At, const Eigen::Vector3f& Up, float Fov, float AspectRatio) noexcept {
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

} // namespace osc