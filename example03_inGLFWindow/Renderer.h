#pragma once

#include <cstdint>
#include <vector>

#include "../common/DeviceBuffer.h"
#include "../common/DeviceVectorBuffer.h"
#include "../common/Util.h"
#include "OptixState.h"
#include "Params.h"

namespace osc {

class Renderer final {
  OptixState OptixState;

  int Width;
  int Height;
  int FrameID;

  common::DeviceVectorBuffer<std::uint32_t> ImageBuffer;
  common::DeviceBuffer<LaunchParams> LaunchParamsBuffer;

public:
  Renderer() noexcept : OptixState(0), Width(0), Height(0), FrameID(0), ImageBuffer(Width * Height), LaunchParamsBuffer() {
    ;
  }

  void resize(int Width, int Height) noexcept {
    Renderer::Width = Width;
    Renderer::Height = Height;

    ImageBuffer.resize(Width * Height);
  }

  std::vector<std::uint32_t> render() noexcept {
    auto LaunchParams = osc::LaunchParams{reinterpret_cast<std::uint32_t *>(ImageBuffer.data()), Width, Height, FrameID};
    LaunchParamsBuffer.set(LaunchParams);

    OPTIX_CHECK(optixLaunch(OptixState.pipeline(), OptixState.stream(), LaunchParamsBuffer.data(), LaunchParamsBuffer.dataSize(), &OptixState.shaderBindingTable(), Width, Height, 1));

    FrameID++;

    CUDA_CHECK(cudaDeviceSynchronize());

    return ImageBuffer.get();
  }
};

} // namespace osc
