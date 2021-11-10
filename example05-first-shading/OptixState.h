#pragma once

#include <algorithm>
#include <array>
#include <iostream>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include "../common/Util.h"
#include "Model.h"
#include "OptixParams.h"

namespace osc {

extern "C" const char DeviceProgram[];

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord {
  char header[OPTIX_SBT_RECORD_HEADER_SIZE];

  Camera camera;
};

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord {
  char header[OPTIX_SBT_RECORD_HEADER_SIZE];

  TriangleMeshes triangleMeshes;
};

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
  char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

class OptixState final {
  CUstream stream_;
  OptixDeviceContext deviceContext_;

  common::DeviceVectorBuffer<Eigen::Vector3f> vertexesBuffer_;
  common::DeviceVectorBuffer<Eigen::Vector3i> indexesBuffer_;
  common::DeviceVectorBuffer<std::uint8_t> traversableBuffer_;
  OptixTraversableHandle traversableHandle_;

  OptixModule module_;
  std::vector<OptixProgramGroup> raygenProgramGroups_;
  std::vector<OptixProgramGroup> hitgroupProgramGroups_;
  std::vector<OptixProgramGroup> missProgramGroups_;
  OptixPipeline pipeline_;
  OptixShaderBindingTable shaderBindingTable_;

public:
  OptixState(int deviceId, const Model &model) noexcept {
    // CUDAデバイスがないと始まらないので、デバイスがあるか確認します。

    [&] {
      auto deviceProp = cudaDeviceProp{};
      CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, deviceId));

      std::cout << "#osc: running on device: " << deviceProp.name << "\n";
    }();

    // OptiXを初期化します。

    [&] {
      OPTIX_CHECK(optixInit());
    }();

    // CUDAのストリーム（実行スケジュール管理のための処理キュー）を作成します。

    stream_ = [&] {
      std::cout << "#osc: creating Stream...\n";

      auto result = CUstream{};

      CUDA_CHECK(cudaSetDevice(deviceId));
      CUDA_CHECK(cudaStreamCreate(&result));

      return result;
    }();

    // Optixのコンテキストを作成します。

    deviceContext_ = [&] {
      std::cout << "#osc: creating DeviceContext...\n";

      auto result = OptixDeviceContext{};

      auto cuContext = CUcontext{};
      CU_CHECK(cuCtxGetCurrent(&cuContext));

      OPTIX_CHECK(optixDeviceContextCreate(cuContext, 0, &result));
      OPTIX_CHECK(optixDeviceContextSetLogCallback(result, common::optixLogCallback, nullptr, 4));

      return result;
    }();

    // OptixのTraversableHandleを生成します。

    vertexesBuffer_.setSize(std::size(model.getVertexes()));
    vertexesBuffer_.set(model.getVertexes());

    indexesBuffer_.setSize(std::size(model.getIndexes()));
    indexesBuffer_.set(model.getIndexes());

    [&] {
      const auto accelBuildOptions = [&] {
        auto result = OptixAccelBuildOptions{};

        result.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        result.motionOptions.numKeys = 1;
        result.operation = OPTIX_BUILD_OPERATION_BUILD;

        return result;
      }();

      const auto triangleArrayFlags = 0u;

      const auto buildInput = [&] {
        auto result = OptixBuildInput{};

        result.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        result.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        result.triangleArray.vertexStrideInBytes = sizeof(Eigen::Vector3f);
        result.triangleArray.numVertices = vertexesBuffer_.getSize();
        result.triangleArray.vertexBuffers = &vertexesBuffer_.getData();

        result.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        result.triangleArray.indexStrideInBytes = sizeof(Eigen::Vector3i);
        result.triangleArray.numIndexTriplets = indexesBuffer_.getSize();
        result.triangleArray.indexBuffer = indexesBuffer_.getData();

        result.triangleArray.flags = &triangleArrayFlags;
        result.triangleArray.numSbtRecords = 1;
        result.triangleArray.sbtIndexOffsetBuffer = 0;
        result.triangleArray.sbtIndexOffsetSizeInBytes = 0;
        result.triangleArray.sbtIndexOffsetStrideInBytes = 0;

        return result;
      }();

      const auto accelBufferSizes = [&] {
        auto result = OptixAccelBufferSizes{};

        OPTIX_CHECK(optixAccelComputeMemoryUsage(deviceContext_, &accelBuildOptions, &buildInput, 1, &result));

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

      OPTIX_CHECK(optixAccelBuild(deviceContext_, 0, &accelBuildOptions, &buildInput, 1, tempBuffer.getData(), tempBuffer.getDataSize(), outputBuffer.getData(), outputBuffer.getDataSize(), &traversableHandle_, &accelEmitDesc, 1));

      traversableBuffer_.setSize(compactedSizeBuffer.get());

      OPTIX_CHECK(optixAccelCompact(deviceContext_, 0, traversableHandle_, traversableBuffer_.getData(), traversableBuffer_.getDataSize(), &traversableHandle_));
    }();

    // OptiXの処理パイプラインのオプションを作成します。

    const auto pipelineCompileOptions = [&] {
      std::cout << "#osc: creating PipelineCompileOptions...\n";

      auto result = OptixPipelineCompileOptions{};

      result.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
      result.usesMotionBlur = false;
      result.numPayloadValues = 2;
      result.numAttributeValues = 2;
      result.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
      result.pipelineLaunchParamsVariableName = "optixLaunchParams";

      return result;
    }();

    const auto pipelineLinkOptions = [&] {
      std::cout << "#osc: creating PipelineLinkOptions...\n";

      auto result = OptixPipelineLinkOptions{};

      result.maxTraceDepth = 2;

      return result;
    }();

    // OptiXのモジュールを作成します。*.cuファイル（をコンパイルしてPTXにしたものをbin2cで配列化したもの）を読み込んでいます。

    module_ = [&] {
      std::cout << "#osc: creating OptixModule...\n";

      auto result = OptixModule{};

      const auto moduleCompileOptions = [&] {
        auto result = OptixModuleCompileOptions{};

        result.maxRegisterCount = 50;
        result.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        result.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

        return result;
      }();

      const auto ptx = std::string(DeviceProgram);

      auto [log, logSize] = [&] {
        auto result = std::array<char, 2048>{};

        return std::make_tuple(result, std::size(result));
      }();

      OPTIX_CHECK(optixModuleCreateFromPTX(deviceContext_, &moduleCompileOptions, &pipelineCompileOptions, ptx.c_str(), std::size(ptx), log.data(), &logSize, &result));

      if (logSize > 1) { // 文字列の終端の'\0'があるので、１文字以上になります。
        std::cerr << log.data() << "\n";
      }

      return result;
    }();

    // レイ・トレースする光（光源から辿るのではなく、カメラから逆にトレースする）生成のプログラム・グループ群を作成します。処理の実態は*.cuの中にあります。

    raygenProgramGroups_ = [&] {
      std::cout << "#osc: creating RaygenProgramGroups...\n";

      auto result = std::vector<OptixProgramGroup>{1};

      const auto programGroupDesc = [&] {
        auto result = OptixProgramGroupDesc{};

        result.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        result.raygen.module = module_;
        result.raygen.entryFunctionName = "__raygen__renderFrame";

        return result;
      }();

      const auto programGroupOptions = OptixProgramGroupOptions{};

      auto [log, logSize] = [&] {
        auto result = std::array<char, 2048>{};

        return std::make_tuple(result, std::size(result));
      }();

      OPTIX_CHECK(optixProgramGroupCreate(deviceContext_, &programGroupDesc, 1, &programGroupOptions, log.data(), &logSize, &result[0]));

      if (logSize > 1) { // 文字列の終端の'\0'があるので、１文字以上になります。
        std::cerr << log.data() << "\n";
      }

      return result;
    }();

    // 衝突判定のプログラム・グループ群を作成します。処理の実態は*.cuの中にあります。

    hitgroupProgramGroups_ = [&] {
      std::cout << "#osc: creating HitgroupProgramGroups...\n";

      auto result = std::vector<OptixProgramGroup>{1};

      const auto programGroupOptions = OptixProgramGroupOptions{};

      const auto programGroupDesc = [&] {
        auto result = OptixProgramGroupDesc{};

        result.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        result.hitgroup.moduleCH = module_;
        result.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
        result.hitgroup.moduleAH = module_;
        result.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

        return result;
      }();

      auto [log, logSize] = [&] {
        auto result = std::array<char, 2048>{};

        return std::make_tuple(result, std::size(result));
      }();

      OPTIX_CHECK(optixProgramGroupCreate(deviceContext_, &programGroupDesc, 1, &programGroupOptions, log.data(), &logSize, &result[0]));

      if (logSize > 1) { // 文字列の終端の'\0'があるので、１文字以上になります。
        std::cerr << log.data() << "\n";
      }

      return result;
    }();

    // レイをトレースしても何かにぶつからなかった場合のプログラム・グループ群を作成します。処理の実態は*.cuの中にあります。

    missProgramGroups_ = [&] {
      std::cout << "#osc: creating MissProgramGroups...\n";

      auto result = std::vector<OptixProgramGroup>{1};

      const auto programGroupOptions = OptixProgramGroupOptions{};

      const auto programGroupDesc = [&] {
        auto result = OptixProgramGroupDesc{};

        result.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        result.miss.module = module_;
        result.miss.entryFunctionName = "__miss__radiance";

        return result;
      }();

      auto [log, logSize] = [&] {
        auto result = std::array<char, 2048>{};

        return std::make_tuple(result, std::size(result));
      }();

      OPTIX_CHECK(optixProgramGroupCreate(deviceContext_, &programGroupDesc, 1, &programGroupOptions, log.data(), &logSize, &result[0]));

      if (logSize > 1) { // 文字列の終端の'\0'があるので、１文字以上になります。
        std::cerr << log.data() << "\n";
      }

      return result;
    }();

    // OptiXの処理パイプラインを作成します。

    pipeline_ = [&] {
      std::cout << "#osc: creating Pipeline...\n";

      auto result = OptixPipeline{};

      const auto programGroups = [&] {
        auto result = std::vector<OptixProgramGroup>{};

        std::copy(std::begin(raygenProgramGroups_), std::end(raygenProgramGroups_), std::back_inserter(result));
        std::copy(std::begin(hitgroupProgramGroups_), std::end(hitgroupProgramGroups_), std::back_inserter(result));
        std::copy(std::begin(missProgramGroups_), std::end(missProgramGroups_), std::back_inserter(result));

        return result;
      }();

      auto [log, logSize] = [&] {
        auto result = std::array<char, 2048>{};

        return std::make_tuple(result, std::size(result));
      }();

      OPTIX_CHECK(optixPipelineCreate(deviceContext_, &pipelineCompileOptions, &pipelineLinkOptions, programGroups.data(), std::size(programGroups), log.data(), &logSize, &result));
      OPTIX_CHECK(optixPipelineSetStackSize(result, 2 * 1024, 2 * 1024, 2 * 1024, 1));

      if (logSize > 1) { // 文字列の終端の'\0'があるので、１文字以上になります。
        std::cerr << log.data() << "\n";
      }

      return result;
    }();

    // OptiXのシェーダー・バインディング・テーブル（なんなのか未だに分からない）を作成します。

    shaderBindingTable_ = [&] {
      std::cout << "#osc: creating ShaderBindingTable...\n";

      auto result = OptixShaderBindingTable{};

      [&] {
        const auto raygenRecords = [&] {
          auto result = std::vector<RaygenRecord>{};

          result.emplace_back([&] {
            auto result = RaygenRecord{};

            OPTIX_CHECK(optixSbtRecordPackHeader(raygenProgramGroups_[0], &result));

            return result;
          }());

          return result;
        }();

        result.raygenRecord = [&] {
          void *result;

          CUDA_CHECK(cudaMalloc(&result, sizeof(RaygenRecord) * std::size(raygenRecords)));
          CUDA_CHECK(cudaMemcpy(result, raygenRecords.data(), sizeof(RaygenRecord) * std::size(raygenRecords), cudaMemcpyHostToDevice));

          return reinterpret_cast<CUdeviceptr>(result);
        }();
      }();

      [&] {
        const auto hitgroupRecords = [&] {
          auto result = std::vector<HitgroupRecord>{};

          result.emplace_back([&] {
            auto result = HitgroupRecord{};

            OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupProgramGroups_[0], &result));

            result.triangleMeshes.vertexes = reinterpret_cast<Eigen::Vector3f *>(vertexesBuffer_.getData());
            result.triangleMeshes.indexes = reinterpret_cast<Eigen::Vector3i *>(indexesBuffer_.getData());
            result.triangleMeshes.color = model.getColor();

            return result;
          }());

          return result;
        }();

        result.hitgroupRecordBase = [&] {
          void *result;

          CUDA_CHECK(cudaMalloc(&result, sizeof(HitgroupRecord) * std::size(hitgroupRecords)));
          CUDA_CHECK(cudaMemcpy(result, hitgroupRecords.data(), sizeof(HitgroupRecord) * std::size(hitgroupRecords), cudaMemcpyHostToDevice));

          return reinterpret_cast<CUdeviceptr>(result);
        }();

        result.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
        result.hitgroupRecordCount = std::size(hitgroupRecords);
      }();

      [&] {
        const auto missRecords = [&] {
          auto result = std::vector<MissRecord>{};

          result.emplace_back([&] {
            auto result = MissRecord{};

            OPTIX_CHECK(optixSbtRecordPackHeader(missProgramGroups_[0], &result));

            return result;
          }());

          return result;
        }();

        result.missRecordBase = [&] {
          void *result;

          CUDA_CHECK(cudaMalloc(&result, sizeof(MissRecord) * std::size(missRecords)));
          CUDA_CHECK(cudaMemcpy(result, missRecords.data(), sizeof(MissRecord) * std::size(missRecords), cudaMemcpyHostToDevice));

          return reinterpret_cast<CUdeviceptr>(result);
        }();

        result.missRecordStrideInBytes = sizeof(MissRecord);
        result.missRecordCount = std::size(missRecords);
      }();

      return result;
    }();
  }

  ~OptixState() {
    OPTIX_CHECK(optixPipelineDestroy(pipeline_));

    for (auto programGroup : missProgramGroups_) {
      OPTIX_CHECK(optixProgramGroupDestroy(programGroup));
    }

    for (auto programGroup : hitgroupProgramGroups_) {
      OPTIX_CHECK(optixProgramGroupDestroy(programGroup));
    }

    for (auto programGroup : raygenProgramGroups_) {
      OPTIX_CHECK(optixProgramGroupDestroy(programGroup));
    }

    OPTIX_CHECK(optixModuleDestroy(module_));
    OPTIX_CHECK(optixDeviceContextDestroy(deviceContext_));

    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(shaderBindingTable_.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(shaderBindingTable_.hitgroupRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(shaderBindingTable_.raygenRecord)));
  }

  const auto &getStream() const noexcept {
    return stream_;
  }

  const auto &getDeviceContext() const noexcept {
    return deviceContext_;
  }

  const auto &getTraversableHandle() const noexcept {
    return traversableHandle_;
  }

  const auto &getPipeline() const noexcept {
    return pipeline_;
  }

  const auto &getRaygenProgramGroups() const noexcept {
    return raygenProgramGroups_;
  }

  const auto &getShaderBindingTable() const noexcept {
    return shaderBindingTable_;
  }
};

} // namespace osc
