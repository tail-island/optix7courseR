#pragma once

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <Eigen/Core>

#include "../common/DeviceBuffer.h"
#include "../common/DeviceVectorBuffer.h"
#include "../common/Util.h"
#include "OptixParams.h"
#include "OptixState.h"

namespace osc {

class Renderer final {
  OptixState optixState_;

  int width_;
  int height_;

  int frameId_;

  common::DeviceVectorBuffer<Eigen::Vector3f> imageBuffer_;
  common::DeviceBuffer<LaunchParams> optixLaunchParamsBuffer_;

public:
  Renderer() noexcept : optixState_{0}, width_{0}, height_{0}, frameId_{0}, imageBuffer_{static_cast<std::size_t>(width_ * height_)}, optixLaunchParamsBuffer_{} {
    ;
  }

  auto resize(int width, int height) noexcept {
    width_ = width;
    height_ = height;

    imageBuffer_.setSize(width * height);
  }

  auto render() noexcept {
    optixLaunchParamsBuffer_.set(LaunchParams{reinterpret_cast<float3 *>(imageBuffer_.getData()), frameId_});

    OPTIX_CHECK(optixLaunch(optixState_.getPipeline(), optixState_.getStream(), optixLaunchParamsBuffer_.getData(), optixLaunchParamsBuffer_.getDataSize(), &optixState_.getShaderBindingTable(), width_, height_, 1));

    frameId_++;

    CUDA_CHECK(cudaDeviceSynchronize());

    return imageBuffer_.get();
  }
};

} // namespace osc
