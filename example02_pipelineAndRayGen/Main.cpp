#include <cstdint>
#include <iostream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "OptixState.h"
#include "Renderer.h"

constexpr int Width = 1200;
constexpr int Height = 1024;

int main(int Argc, char **Argv) {
  auto OptixState = osc::OptixState(0);
  auto Renderer = osc::Renderer(OptixState.stream(), OptixState.pipeline(), OptixState.shaderBindingTable(), Width, Height);
  auto Image = Renderer.render();

  stbi_write_png("osc_example2.png", Width, Height, 4, Image.data(), Width * sizeof(std::uint32_t));

  std::cout << "Image rendereed and saved to osc_example2.png ... done.\n";
}
