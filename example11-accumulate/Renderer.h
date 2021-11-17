#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <Eigen/Core>

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

  common::DeviceVectorBuffer<Eigen::Vector4f> imageBuffer_;
  int frameId_;
  Light light_;
  Camera camera_;

  common::DeviceBuffer<LaunchParams> optixLaunchParamsBuffer_;

public:
  Renderer(const Model &model, const Light &light) noexcept : optixState_{0, model}, width_{0}, height_{0}, imageBuffer_{static_cast<std::size_t>(width_ * height_)}, frameId_{0}, light_{light}, camera_{}, optixLaunchParamsBuffer_{} {
    ;
  }

  auto resize(int width, int height) noexcept {
    width_ = width;
    height_ = height;

    imageBuffer_.setSize(width * height);

    frameId_ = 0;
  }

  auto setCamera(const Camera &camera) noexcept {
    camera_ = camera;

    frameId_ = 0;
  }

  auto render() noexcept {
    optixLaunchParamsBuffer_.set(LaunchParams{reinterpret_cast<float4 *>(imageBuffer_.getData()), frameId_, light_, camera_, optixState_.getTraversableHandle()});
    frameId_++;

    OPTIX_CHECK(optixLaunch(optixState_.getPipeline(), optixState_.getStream(), optixLaunchParamsBuffer_.getData(), optixLaunchParamsBuffer_.getDataSize(), &optixState_.getShaderBindingTable(), width_, height_, 1));

    CUDA_CHECK(cudaDeviceSynchronize());

    return imageBuffer_.get();
  }
};

} // namespace osc
