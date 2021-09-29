#pragma once

#include <cstdint>

namespace osc {

struct LaunchParams {
  std::uint32_t *ImageBuffer;
  int Width;
  int Height;
  int FrameID;
};

} // namespace osc
