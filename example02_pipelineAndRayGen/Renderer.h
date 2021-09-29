#pragma once

#include <cstdint>

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include "../common/DeviceBuffer.h"
#include "../common/DeviceVectorBuffer.h"
#include "Params.h"

namespace osc {

class Renderer final {
  CUstream Stream;
  OptixPipeline Pipeline;
  OptixShaderBindingTable ShaderBindingTable;

  int Width;
  int Height;

  common::DeviceVectorBuffer<std::uint32_t> ImageBuffer;
  common::DeviceBuffer<LaunchParams> LaunchParamsBuffer;

public:
  Renderer(const CUstream &Stream, const OptixPipeline &Pipeline, const OptixShaderBindingTable &ShaderBindingTable, int Width, int Height) noexcept : Stream(Stream), Pipeline(Pipeline), ShaderBindingTable(ShaderBindingTable), Width(Width), Height(Height), ImageBuffer(Width * Height), LaunchParamsBuffer() {
    ;
  }

  auto render() noexcept {
    auto LaunchParams = osc::LaunchParams{reinterpret_cast<std::uint32_t *>(ImageBuffer.data()), Width, Height};
    LaunchParamsBuffer.set(LaunchParams);

    OPTIX_CHECK(optixLaunch(Pipeline, Stream, LaunchParamsBuffer.data(), LaunchParamsBuffer.dataSize(), &ShaderBindingTable, Width, Height, 1));

    CUDA_CHECK(cudaDeviceSynchronize());

    return ImageBuffer.get();
  }
};

} // namespace osc
