#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "../common/DeviceBuffer.h"
#include "../common/DeviceVectorBuffer.h"
#include "../common/Util.h"
#include "Model.h"
#include "OptixParams.h"
#include "OptixState.h"

namespace osc {

class Renderer final {
  OptixState optixState_;

  int width_;
  int height_;

  common::DeviceVectorBuffer<std::uint32_t> imageBuffer_;
  Camera camera_;

  common::DeviceBuffer<LaunchParams> optixLaunchParamsBuffer_;

public:
  Renderer(const Model &model) noexcept : optixState_{0, model}, width_{0}, height_{0}, imageBuffer_{static_cast<std::size_t>(width_ * height_)}, optixLaunchParamsBuffer_{} {
    ;
  }

  auto resize(int width, int height) noexcept {
    width_ = width;
    height_ = height;

    imageBuffer_.setSize(width * height);
  }

  auto setCamera(const Camera &camera) noexcept {
    camera_ = camera;
  }

  auto render() noexcept {
    optixLaunchParamsBuffer_.set(LaunchParams{reinterpret_cast<std::uint32_t *>(imageBuffer_.getData()), camera_, {-907.108, 2205.875, -400.0267}, optixState_.getTraversableHandle()});

    OPTIX_CHECK(optixLaunch(optixState_.getPipeline(), optixState_.getStream(), optixLaunchParamsBuffer_.getData(), optixLaunchParamsBuffer_.getDataSize(), &optixState_.getShaderBindingTable(), width_, height_, 1));

    CUDA_CHECK(cudaDeviceSynchronize());

    return imageBuffer_.get();
  }
};

} // namespace osc
