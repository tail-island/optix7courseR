#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "../common/DeviceBuffer.h"
#include "../common/DeviceVectorBuffer.h"
#include "../common/Util.h"
#include "OptixState.h"
#include "Params.h"

namespace osc {

class Renderer final {
  Model Model;
  OptixState OptixState;

  int Width;
  int Height;

  common::DeviceVectorBuffer<std::uint32_t> ImageBuffer;
  common::DeviceBuffer<LaunchParams> LaunchParamsBuffer;

  common::DeviceVectorBuffer<Eigen::Vector3f> VertexesBuffer;
  common::DeviceVectorBuffer<Eigen::Vector3i> IndexesBuffer;

  common::DeviceVectorBuffer<std::uint8_t> TraversableBuffer;
  OptixTraversableHandle TraversableHandle;

public:
  Renderer(const osc::Model &Model) noexcept : Model(Model), OptixState(0), Width(0), Height(0), ImageBuffer(Width * Height), LaunchParamsBuffer(), VertexesBuffer(std::size(Model.vertexes())), IndexesBuffer(std::size(Model.indexes())) {
    VertexesBuffer.set(Model.vertexes());
    IndexesBuffer.set(Model.indexes());

    const auto VertexesBufferData = VertexesBuffer.data();           // なんでかポインターへのポインターが必要なので、変数を宣言しました。。。
    const auto TriangleArrayFlags = std::array<std::uint32_t, 1>{0}; // なんでかポインターが必要なので、変数を宣言しました。。。

    const auto AccelBuildOptions = [&] {
      auto Result = OptixAccelBuildOptions();

      Result.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
      Result.motionOptions.numKeys = 1;
      Result.operation = OPTIX_BUILD_OPERATION_BUILD;

      return Result;
    }();

    const auto BuildInput = [&] {
      auto Result = OptixBuildInput();

      Result.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

      Result.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
      Result.triangleArray.vertexStrideInBytes = sizeof(Eigen::Vector3f);
      Result.triangleArray.numVertices = VertexesBuffer.size();
      Result.triangleArray.vertexBuffers = &VertexesBufferData;

      Result.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
      Result.triangleArray.indexStrideInBytes = sizeof(Eigen::Vector3i);
      Result.triangleArray.numIndexTriplets = IndexesBuffer.size();
      Result.triangleArray.indexBuffer = IndexesBuffer.data();

      Result.triangleArray.flags = TriangleArrayFlags.data();
      Result.triangleArray.numSbtRecords = 1;
      Result.triangleArray.sbtIndexOffsetBuffer = 0;
      Result.triangleArray.sbtIndexOffsetSizeInBytes = 0;
      Result.triangleArray.sbtIndexOffsetStrideInBytes = 0;

      return Result;
    }();

    const auto AccelBufferSizes = [&] {
      auto Result = OptixAccelBufferSizes();

      OPTIX_CHECK(optixAccelComputeMemoryUsage(OptixState.deviceContext(), &AccelBuildOptions, &BuildInput, 1, &Result));

      return Result;
    }();

    const auto CompactedSizeBuffer = common::DeviceBuffer<std::uint64_t>();

    const auto AccelEmitDesc = [&] {
      auto Result = OptixAccelEmitDesc();

      Result.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
      Result.result = CompactedSizeBuffer.data();

      return Result;
    }();

    const auto TempBuffer = common::DeviceVectorBuffer<std::uint8_t>(AccelBufferSizes.tempSizeInBytes);
    const auto OutputBuffer = common::DeviceVectorBuffer<std::uint8_t>(AccelBufferSizes.outputSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(OptixState.deviceContext(), 0, &AccelBuildOptions, &BuildInput, 1, TempBuffer.data(), TempBuffer.dataSize(), OutputBuffer.data(), OutputBuffer.dataSize(), &TraversableHandle, &AccelEmitDesc, 1));

    TraversableBuffer.resize(CompactedSizeBuffer.get());

    OPTIX_CHECK(optixAccelCompact(OptixState.deviceContext(), 0, TraversableHandle, TraversableBuffer.data(), TraversableBuffer.dataSize(), &TraversableHandle));

    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void resize(int Width, int Height) noexcept {
    Renderer::Width = Width;
    Renderer::Height = Height;

    ImageBuffer.resize(Width * Height);
  }

  void setCamera(const Camera &Camera) noexcept {
    auto Record = [&] {
      auto Result = RaygenRecord();

      optixSbtRecordPackHeader(OptixState.raygenProgramGroups()[0], &Result);

      return Result;
    }();
    Record.Camera = Camera;
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(OptixState.shaderBindingTable().raygenRecord), &Record, sizeof(RaygenRecord), cudaMemcpyHostToDevice)); // OptiXのサンプルだとこんな感じになっていたのですけど、カメラが2個になったらどうすればいいんだろ？
  }

  std::vector<std::uint32_t> render() noexcept {
    auto LaunchParams = osc::LaunchParams{reinterpret_cast<std::uint32_t *>(ImageBuffer.data()), Width, Height, TraversableHandle};
    LaunchParamsBuffer.set(LaunchParams);

    OPTIX_CHECK(optixLaunch(OptixState.pipeline(), OptixState.stream(), LaunchParamsBuffer.data(), LaunchParamsBuffer.dataSize(), &OptixState.shaderBindingTable(), Width, Height, 1));

    CUDA_CHECK(cudaDeviceSynchronize());

    return ImageBuffer.get();
  }
};

} // namespace osc
