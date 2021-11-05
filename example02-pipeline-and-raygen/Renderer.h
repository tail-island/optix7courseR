#pragma once

#include <cstdint>

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include "../common/DeviceBuffer.h"
#include "../common/DeviceVectorBuffer.h"
#include "OptixParams.h"
#include "OptixState.h"

namespace osc {

class Renderer final {
  OptixState optixState_;

  int width_;
  int height_;

  common::DeviceVectorBuffer<std::uint32_t> imageBuffer_;
  common::DeviceBuffer<LaunchParams> optixLaunchParamsBuffer_;

public:
  Renderer(int width, int height) noexcept : optixState_{0}, width_{width}, height_{height}, imageBuffer_{static_cast<std::size_t>(width * height)}, optixLaunchParamsBuffer_{} {
    ;
  }

  auto render() noexcept {
    optixLaunchParamsBuffer_.set(LaunchParams{reinterpret_cast<std::uint32_t *>(imageBuffer_.getData())});

    OPTIX_CHECK(optixLaunch(optixState_.getPipeline(), optixState_.getStream(), optixLaunchParamsBuffer_.getData(), optixLaunchParamsBuffer_.getDataSize(), &optixState_.getShaderBindingTable(), width_, height_, 1));

    CUDA_CHECK(cudaDeviceSynchronize());

    return imageBuffer_.get();
  }
};

} // namespace osc
