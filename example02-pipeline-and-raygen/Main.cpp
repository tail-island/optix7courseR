#include <cstdint>
#include <iostream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "OptixState.h"
#include "Renderer.h"

constexpr int width = 1200;
constexpr int height = 1024;

int main(int argc, char **argv) {
  auto renderer = osc::Renderer{width, height};
  auto image = renderer.render();

  stbi_write_png("osc_example2.png", width, height, 4, image.data(), width * sizeof(std::uint32_t));

  std::cout << "Image rendereed and saved to osc_example2.png ... done.\n";
}
