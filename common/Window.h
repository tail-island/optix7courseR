#include <cstdint>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
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
  GLFWwindow *glfwWindow_;

  auto getFrameBufferSize() const noexcept {
    int width;
    int height;
    glfwGetFramebufferSize(glfwWindow_, &width, &height);

    return Eigen::Vector2i{width, height};
  }

  auto getCursorPos() const noexcept {
    double x;
    double y;
    glfwGetCursorPos(glfwWindow_, &x, &y);

    return Eigen::Vector2f{x, y};
  }

  virtual void key(int key, int action, int modifiers) noexcept {
    ;
  }

  virtual void mouseButton(int button, int action, int modifiers) noexcept {
    ;
  }

  virtual void cursorPos(const Eigen::Vector2f &cursorPos) noexcept {
    ;
  }

  virtual void resize(const Eigen::Vector2i &size) noexcept = 0;

  virtual std::vector<std::uint32_t> render() noexcept = 0;

public:
  Window(const std::string &title) noexcept {
    glfwSetErrorCallback([](auto error, auto description) {
      std::cerr << "Error: " << description << std::endl;
    });

    if (!glfwInit()) {
      std::exit(1);
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2); // 本当はOpenGL4.xにしたいのですけど、そうするとglDrawPixelsで描画できません。
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0); // そもそもglDrawPixelsみたいな遅いのを使うな、CUDAで作った画像をホスト・メモリ経由で表示するのは無駄だと言われたらぐうの音も出ないのですけど、とりあえずは無視で。

    glfwWindow_ = glfwCreateWindow(1200, 800, title.c_str(), NULL, NULL);
    if (!glfwWindow_) {
      std::exit(1);
    }

    glfwSetWindowUserPointer(glfwWindow_, this);
    glfwMakeContextCurrent(glfwWindow_);
    glfwSwapInterval(1);
  }

  virtual ~Window() {
    glfwDestroyWindow(glfwWindow_);
    glfwTerminate();
  }

  auto run() noexcept {
    resize(getFrameBufferSize());

    glfwSetFramebufferSizeCallback(glfwWindow_, [](GLFWwindow *glfwWindow, int width, int height) {
      auto window = static_cast<common::Window *>(glfwGetWindowUserPointer(glfwWindow));

      window->resize(Eigen::Vector2i{width, height});
    });

    glfwSetKeyCallback(glfwWindow_, [](GLFWwindow *glfwWindow, int key, int scancode, int action, int modifiers) {
      static_cast<Window *>(glfwGetWindowUserPointer(glfwWindow))->key(key, action, modifiers);
    });

    glfwSetMouseButtonCallback(glfwWindow_, [](GLFWwindow *glfwWindow, int button, int action, int modifiers) {
      static_cast<Window *>(glfwGetWindowUserPointer(glfwWindow))->mouseButton(button, action, modifiers);
    });

    glfwSetCursorPosCallback(glfwWindow_, [](GLFWwindow *glfwWindow, double x, double y) {
      static_cast<Window *>(glfwGetWindowUserPointer(glfwWindow))->cursorPos(Eigen::Vector2f{x, y});
    });

    while (!glfwWindowShouldClose(glfwWindow_)) {
      auto frameBufferSize = getFrameBufferSize();

      glDrawPixels(frameBufferSize.x(), frameBufferSize.y(), GL_RGBA, GL_UNSIGNED_BYTE, render().data()); // TODO: glDrawPixelsを使うのをやめて、OpenGLのバージョンを上げる。

      glfwSwapBuffers(glfwWindow_);
      glfwPollEvents();
    }
  }
};

class Camera final {
  Eigen::Vector3f from_;
  Eigen::Vector3f at_;
  Eigen::Vector3f up_;

public:
  Camera(const Eigen::Vector3f &from, const Eigen::Vector3f &at, const Eigen::Vector3f &up) noexcept : from_{from}, at_{at}, up_{up} {
    ;
  }

  const auto &getFrom() const noexcept {
    return from_;
  }

  auto setFrom(const Eigen::Vector3f &from) noexcept {
    from_ = from;
  }

  const auto &getAt() const noexcept {
    return at_;
  }

  auto setAt(const Eigen::Vector3f &at) noexcept {
    at_ = at;
  }

  const auto &getUp() const noexcept {
    return up_;
  }

  auto setUp(const Eigen::Vector3f &up) noexcept {
    up_ = up;
  }
};

constexpr float radianPerPixel = 0.5f * static_cast<float>(M_PI) / 180.0f;
constexpr float stepPerPixel = 0.05f;

class CameraWindow : public Window {
protected:
  Camera camera_;

  bool isMouseDragLeft_;
  bool isMouseDragMiddle_;
  bool isMouseDragRight_;

  Eigen::Vector2f lastCursorPos_;

  bool isCameraMoved_;

  auto getUVW() const noexcept {
    auto w = camera_.getAt() - camera_.getFrom();
    auto u = w.cross(camera_.getUp()).normalized();
    auto v = u.cross(w).normalized();

    const auto vlen = w.norm() * tanf(0.5f * 35 * static_cast<float>(M_PI) / 180.0f);
    const auto ulen = vlen * getFrameBufferSize().x() / getFrameBufferSize().y();

    v *= vlen;
    u *= ulen;

    return std::make_tuple(u, v, w);
  }

  void mouseButton(int button, int action, int modifiers) noexcept override {
    switch (button) {
    case GLFW_MOUSE_BUTTON_LEFT:
      isMouseDragLeft_ = action == GLFW_PRESS;
      break;

    case GLFW_MOUSE_BUTTON_MIDDLE:
      isMouseDragMiddle_ = action == GLFW_PRESS;
      break;

    case GLFW_MOUSE_BUTTON_RIGHT:
      isMouseDragRight_ = action == GLFW_PRESS;
      break;
    };

    lastCursorPos_ = getCursorPos();
  }

  auto rotate(const Eigen::Vector2f &angle) noexcept {
    // atの座標系を作成します。
    auto z = (camera_.getFrom() - camera_.getAt()).normalized();
    auto x = z.cross(camera_.getUp()).normalized();
    auto y = x.cross(z).normalized();

    // atの座標系でカメラを回転します。
    camera_.setFrom(camera_.getAt() + Eigen::AngleAxisf(angle.y(), x) * Eigen::AngleAxisf(angle.x(), y) * (camera_.getFrom() - camera_.getAt()));

    isCameraMoved_ = true;
  }

  auto move(const Eigen::Vector3f &step) noexcept {
    // fromの座標系を作成します。
    auto z = (camera_.getAt() - camera_.getFrom()).normalized();
    auto x = z.cross(camera_.getUp()).normalized();
    auto y = x.cross(z).normalized();

    // 移動率（モデルの大きさによって移動距離をイイ感じに変えたい）を計算します。
    auto stepRatio = (camera_.getFrom() - camera_.getAt()).norm();

    // fromの座標系でfromとatを平行移動させます。
    const auto affine = Eigen::Translation3f(x * step.x() * stepRatio) * Eigen::Translation3f(y * step.y() * stepRatio) * Eigen::Translation3f(z * step.z() * stepRatio);
    camera_.setFrom(affine * camera_.getFrom());
    camera_.setAt(affine * camera_.getAt());

    isCameraMoved_ = true;
  }

  void cursorPos(const Eigen::Vector2f &cursorPos) noexcept override {
    if (isMouseDragLeft_) {
      rotate(Eigen::Vector2f{cursorPos.x() - lastCursorPos_.x(), -(cursorPos.y() - lastCursorPos_.y())} * -1 * radianPerPixel);
    }

    if (isMouseDragMiddle_) {
      move(Eigen::Vector3f{cursorPos.x() - lastCursorPos_.x(), -(cursorPos.y() - lastCursorPos_.y()), 0} * stepPerPixel);
    }

    if (isMouseDragRight_) {
      move(Eigen::Vector3f{0, 0, cursorPos.y() - lastCursorPos_.y()} * stepPerPixel);
    }

    lastCursorPos_ = cursorPos;
  }

public:
  CameraWindow(const std::string &title, const Camera &camera) noexcept : Window{title}, camera_{camera}, isMouseDragLeft_{false}, isMouseDragMiddle_{false}, isMouseDragRight_{false}, lastCursorPos_{0, 0}, isCameraMoved_(false) {
    ;
  }
};

} // namespace common
} // namespace osc
