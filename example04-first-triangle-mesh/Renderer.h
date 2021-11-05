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
  Model model_;
  OptixState optixState_;

  int width_;
  int height_;

  common::DeviceVectorBuffer<std::uint32_t> imageBuffer_;
  common::DeviceBuffer<LaunchParams> optixLaunchParamsBuffer_;

  common::DeviceVectorBuffer<Eigen::Vector3f> vertexesBuffer_;
  common::DeviceVectorBuffer<Eigen::Vector3i> indexesBuffer_;

  common::DeviceVectorBuffer<std::uint8_t> traversableBuffer_;
  OptixTraversableHandle traversableHandle_;

public:
  Renderer(const Model &model) noexcept : model_{model}, optixState_{0}, width_{0}, height_{0}, imageBuffer_{static_cast<std::size_t>(width_ * height_)}, optixLaunchParamsBuffer_{}, vertexesBuffer_{std::size(model.getVertexes())}, indexesBuffer_{std::size(model.getIndexes())} {
    vertexesBuffer_.set(model.getVertexes());
    indexesBuffer_.set(model.getIndexes());

    const auto vertexesBufferData = vertexesBuffer_.getData();       // なんでかポインターへのポインターが必要なので、変数を宣言しました。。。
    const auto triangleArrayFlags = std::array<std::uint32_t, 1>{0}; // なんでかポインターが必要なので、変数を宣言しました。。。

    const auto accelBuildOptions = [&] {
      auto result = OptixAccelBuildOptions{};

      result.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
      result.motionOptions.numKeys = 1;
      result.operation = OPTIX_BUILD_OPERATION_BUILD;

      return result;
    }();

    const auto buildInput = [&] {
      auto result = OptixBuildInput{};

      result.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

      result.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
      result.triangleArray.vertexStrideInBytes = sizeof(Eigen::Vector3f);
      result.triangleArray.numVertices = vertexesBuffer_.getSize();
      result.triangleArray.vertexBuffers = &vertexesBufferData;

      result.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
      result.triangleArray.indexStrideInBytes = sizeof(Eigen::Vector3i);
      result.triangleArray.numIndexTriplets = indexesBuffer_.getSize();
      result.triangleArray.indexBuffer = indexesBuffer_.getData();

      result.triangleArray.flags = triangleArrayFlags.data();
      result.triangleArray.numSbtRecords = 1;
      result.triangleArray.sbtIndexOffsetBuffer = 0;
      result.triangleArray.sbtIndexOffsetSizeInBytes = 0;
      result.triangleArray.sbtIndexOffsetStrideInBytes = 0;

      return result;
    }();

    const auto accelBufferSizes = [&] {
      auto result = OptixAccelBufferSizes{};

      OPTIX_CHECK(optixAccelComputeMemoryUsage(optixState_.getDeviceContext(), &accelBuildOptions, &buildInput, 1, &result));

      return result;
    }();

    const auto compactedSizeBuffer = common::DeviceBuffer<std::uint64_t>{};

    const auto accelEmitDesc = [&] {
      auto result = OptixAccelEmitDesc{};

      result.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
      result.result = compactedSizeBuffer.getData();

      return result;
    }();

    const auto tempBuffer = common::DeviceVectorBuffer<std::uint8_t>{accelBufferSizes.tempSizeInBytes};
    const auto outputBuffer = common::DeviceVectorBuffer<std::uint8_t>{accelBufferSizes.outputSizeInBytes};

    OPTIX_CHECK(optixAccelBuild(optixState_.getDeviceContext(), 0, &accelBuildOptions, &buildInput, 1, tempBuffer.getData(), tempBuffer.getDataSize(), outputBuffer.getData(), outputBuffer.getDataSize(), &traversableHandle_, &accelEmitDesc, 1));

    traversableBuffer_.setSize(compactedSizeBuffer.get());

    OPTIX_CHECK(optixAccelCompact(optixState_.getDeviceContext(), 0, traversableHandle_, traversableBuffer_.getData(), traversableBuffer_.getDataSize(), &traversableHandle_));

    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void resize(int width, int height) noexcept {
    width_ = width;
    height_ = height;

    imageBuffer_.setSize(width * height);
  }

  void setCamera(const Camera &camera) noexcept {
    auto record = [&] {
      auto result = RaygenRecord{};

      optixSbtRecordPackHeader(optixState_.getRaygenProgramGroups()[0], &result);

      return result;
    }();
    record.camera = camera;

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(optixState_.getShaderBindingTable().raygenRecord), &record, sizeof(RaygenRecord), cudaMemcpyHostToDevice)); // OptiXのサンプルだとこんな感じになっていたのですけど、カメラが2個になったらどうすればいいんだろ？
  }

  std::vector<std::uint32_t> render() noexcept {
    optixLaunchParamsBuffer_.set(LaunchParams{reinterpret_cast<std::uint32_t *>(imageBuffer_.getData()), traversableHandle_});

    OPTIX_CHECK(optixLaunch(optixState_.getPipeline(), optixState_.getStream(), optixLaunchParamsBuffer_.getData(), optixLaunchParamsBuffer_.getDataSize(), &optixState_.getShaderBindingTable(), width_, height_, 1));

    CUDA_CHECK(cudaDeviceSynchronize());

    return imageBuffer_.get();
  }
};

} // namespace osc
