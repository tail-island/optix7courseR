#pragma once

#include <algorithm>
#include <array>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include "../common/Util.h"
#include "Params.h"

namespace osc {

extern "C" const char DeviceProgram[];

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord {
  char Header[OPTIX_SBT_RECORD_HEADER_SIZE];

  Camera Camera;
};

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord {
  char Header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
  char Header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

class OptixState final {
  CUstream Stream;
  OptixDeviceContext DeviceContext;
  OptixModule Module;
  std::vector<OptixProgramGroup> RaygenProgramGroups;
  std::vector<OptixProgramGroup> HitgroupProgramGroups;
  std::vector<OptixProgramGroup> MissProgramGroups;
  OptixPipeline Pipeline;
  OptixShaderBindingTable ShaderBindingTable;

public:
  OptixState(int DeviceID) noexcept {
    // CUDAデバイスがないと始まらないので、デバイスがあるか確認します。

    [&] {
      auto DeviceProp = cudaDeviceProp{};
      CUDA_CHECK(cudaGetDeviceProperties(&DeviceProp, DeviceID));

      std::cout << "#osc: running on device: " << DeviceProp.name << "\n";
    }();

    // OptiXを初期化します。

    [&] {
      OPTIX_CHECK(optixInit());
    }();

    // CUDAのストリーム（実行スケジュール管理のための処理キュー）を作成します。

    Stream = [&] {
      std::cout << "#osc: creating Stream...\n";

      auto Result = CUstream();

      CUDA_CHECK(cudaSetDevice(DeviceID));
      CUDA_CHECK(cudaStreamCreate(&Result));

      return Result;
    }();

    // Optixのコンテキストを作成します。

    DeviceContext = [&] {
      std::cout << "#osc: creating DeviceContext...\n";

      auto Result = OptixDeviceContext{};

      auto CuContext = CUcontext{};
      CU_CHECK(cuCtxGetCurrent(&CuContext));

      OPTIX_CHECK(optixDeviceContextCreate(CuContext, 0, &Result));
      OPTIX_CHECK(optixDeviceContextSetLogCallback(Result, optixLogCallback, nullptr, 4));

      return Result;
    }();

    // OptiXの処理パイプラインのオプションを作成します。

    auto PipelineCompileOptions = [&] {
      std::cout << "#osc: creating PipelineCompileOptions...\n";

      auto Result = OptixPipelineCompileOptions{};

      Result.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
      Result.usesMotionBlur = false;
      Result.numPayloadValues = 2;
      Result.numAttributeValues = 2;
      Result.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
      Result.pipelineLaunchParamsVariableName = "OptixLaunchParams";

      return Result;
    }();

    auto PipelineLinkOptions = [&] {
      std::cout << "#osc: creating PipelineLinkOptions...\n";

      auto Result = OptixPipelineLinkOptions{};

      Result.maxTraceDepth = 2;

      return Result;
    }();

    // OptiXのモジュールを作成します。*.cuファイル（をコンパイルしてPTXにしたものをbin2cで配列化したもの）を読み込んでいます。

    Module = [&] {
      std::cout << "#osc: creating OptixModule...\n";

      auto Result = OptixModule{};

      auto ModuleCompileOptions = [&] {
        auto Result = OptixModuleCompileOptions{};

        Result.maxRegisterCount = 50;
        Result.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        Result.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

        return Result;
      }();

      const auto Ptx = std::string(DeviceProgram);

      auto [Log, LogSize] = [&] {
        auto Result = std::array<char, 2048>();

        return std::make_tuple(Result, Result.size());
      }();

      OPTIX_CHECK(optixModuleCreateFromPTX(DeviceContext, &ModuleCompileOptions, &PipelineCompileOptions, Ptx.c_str(), Ptx.size(), Log.data(), &LogSize, &Result));

      if (LogSize > 1) { // 文字列の終端の'\0'があるので、１文字以上になります。
        std::cerr << Log.data() << "\n";
      }

      return Result;
    }();

    // レイ・トレースする光（光源から辿るのではなく、カメラから逆にトレースする）のプログラム・グループ群を作成します。処理の実態は*.cuの中にあります。

    RaygenProgramGroups = [&] {
      std::cout << "#osc: creating RaygenProgramGroups...\n";

      auto Result = std::vector<OptixProgramGroup>(1);

      auto ProgramGroupDesc = [&] {
        auto Result = OptixProgramGroupDesc{};

        Result.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        Result.raygen.module = Module;
        Result.raygen.entryFunctionName = "__raygen__renderFrame";

        return Result;
      }();

      auto ProgramGroupOptions = OptixProgramGroupOptions{};

      auto [Log, LogSize] = [&] {
        auto Result = std::array<char, 2048>();

        return std::make_tuple(Result, Result.size());
      }();

      OPTIX_CHECK(optixProgramGroupCreate(DeviceContext, &ProgramGroupDesc, 1, &ProgramGroupOptions, Log.data(), &LogSize, &Result[0]));

      if (LogSize > 1) { // 文字列の終端の'\0'があるので、１文字以上になります。
        std::cerr << Log.data() << "\n";
      }

      return Result;
    }();

    // 衝突判定のプログラム・グループ群を作成します。処理の実態は*.cuの中にあります。

    HitgroupProgramGroups = [&] {
      std::cout << "#osc: creating HitgroupProgramGroups...\n";

      auto Result = std::vector<OptixProgramGroup>(1);

      auto ProgramGroupOptions = OptixProgramGroupOptions{};

      auto ProgramGroupDesc = [&] {
        auto Result = OptixProgramGroupDesc{};

        Result.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        Result.hitgroup.moduleCH = Module;
        Result.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
        Result.hitgroup.moduleAH = Module;
        Result.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

        return Result;
      }();

      auto [Log, LogSize] = [&] {
        auto Result = std::array<char, 2048>();

        return std::make_tuple(Result, Result.size());
      }();

      OPTIX_CHECK(optixProgramGroupCreate(DeviceContext, &ProgramGroupDesc, 1, &ProgramGroupOptions, Log.data(), &LogSize, &Result[0]));

      if (LogSize > 1) { // 文字列の終端の'\0'があるので、１文字以上になります。
        std::cerr << Log.data() << "\n";
      }

      return Result;
    }();

    // トレースしても何かにぶつからなかった場合のプログラム・グループ群を作成します。処理の実態は*.cuの中にあります。

    MissProgramGroups = [&] {
      std::cout << "#osc: creating MissProgramGroups...\n";

      auto Result = std::vector<OptixProgramGroup>(1);

      auto ProgramGroupOptions = OptixProgramGroupOptions{};

      auto ProgramGroupDesc = [&] {
        auto Result = OptixProgramGroupDesc{};

        Result.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        Result.miss.module = Module;
        Result.miss.entryFunctionName = "__miss__radiance";

        return Result;
      }();

      auto [Log, LogSize] = [&] {
        auto Result = std::array<char, 2048>();

        return std::make_tuple(Result, Result.size());
      }();

      OPTIX_CHECK(optixProgramGroupCreate(DeviceContext, &ProgramGroupDesc, 1, &ProgramGroupOptions, Log.data(), &LogSize, &Result[0]));

      if (LogSize > 1) { // 文字列の終端の'\0'があるので、１文字以上になります。
        std::cerr << Log.data() << "\n";
      }

      return Result;
    }();

    // OptiXの処理パイプラインを作成します。

    Pipeline = [&] {
      std::cout << "#osc: creating Pipeline...\n";

      auto Result = OptixPipeline{};

      auto ProgramGroups = [&] {
        auto Result = std::vector<OptixProgramGroup>{};

        std::copy(std::begin(RaygenProgramGroups), std::end(RaygenProgramGroups), std::back_inserter(Result));
        std::copy(std::begin(HitgroupProgramGroups), std::end(HitgroupProgramGroups), std::back_inserter(Result));
        std::copy(std::begin(MissProgramGroups), std::end(MissProgramGroups), std::back_inserter(Result));

        return Result;
      }();

      auto [Log, LogSize] = [&] {
        auto Result = std::array<char, 2048>();

        return std::make_tuple(Result, Result.size());
      }();

      OPTIX_CHECK(optixPipelineCreate(DeviceContext, &PipelineCompileOptions, &PipelineLinkOptions, ProgramGroups.data(), ProgramGroups.size(), Log.data(), &LogSize, &Result));
      OPTIX_CHECK(optixPipelineSetStackSize(Result, 2 * 1024, 2 * 1024, 2 * 1024, 1));

      if (LogSize > 1) { // 文字列の終端の'\0'があるので、１文字以上になります。
        std::cerr << Log.data() << "\n";
      }

      return Result;
    }();

    // OptiXのシェーダー・バインディング・テーブル（なんなのか未だに分からない）を作成します。

    ShaderBindingTable = [&] {
      std::cout << "#osc: creating ShaderBindingTable...\n";

      auto Result = OptixShaderBindingTable{};

      [&] {
        const auto RaygenRecords = [&] {
          auto Result = std::vector<RaygenRecord>{};

          std::transform(std::begin(RaygenProgramGroups), std::end(RaygenProgramGroups), std::back_inserter(Result), [](const auto &ProgramGroup) {
            auto Result = RaygenRecord();

            OPTIX_CHECK(optixSbtRecordPackHeader(ProgramGroup, &Result));

            return Result;
          });

          return Result;
        }();

        Result.raygenRecord = [&] {
          void *Result;

          CUDA_CHECK(cudaMalloc(&Result, sizeof(RaygenRecord) * std::size(RaygenRecords)));
          CUDA_CHECK(cudaMemcpy(Result, RaygenRecords.data(), sizeof(RaygenRecord) * std::size(RaygenRecords), cudaMemcpyHostToDevice));

          return reinterpret_cast<CUdeviceptr>(Result);
        }();
      }();

      [&] {
        const auto HitgroupRecords = [&] {
          auto Result = std::vector<HitgroupRecord>{};

          std::transform(std::begin(HitgroupProgramGroups), std::end(HitgroupProgramGroups), std::back_inserter(Result), [](const auto &ProgramGroup) {
            auto Result = HitgroupRecord();

            OPTIX_CHECK(optixSbtRecordPackHeader(ProgramGroup, &Result));

            return Result;
          });

          return Result;
        }();

        Result.hitgroupRecordBase = [&] {
          void *Result;

          CUDA_CHECK(cudaMalloc(&Result, sizeof(HitgroupRecord) * std::size(HitgroupRecords)));
          CUDA_CHECK(cudaMemcpy(Result, HitgroupRecords.data(), sizeof(HitgroupRecord) * std::size(HitgroupRecords), cudaMemcpyHostToDevice));

          return reinterpret_cast<CUdeviceptr>(Result);
        }();

        Result.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
        Result.hitgroupRecordCount = std::size(HitgroupRecords);
      }();

      [&] {
        const auto MissRecords = [&] {
          auto Result = std::vector<MissRecord>{};

          std::transform(std::begin(MissProgramGroups), std::end(MissProgramGroups), std::back_inserter(Result), [](const auto &ProgramGroup) {
            auto Result = MissRecord();

            OPTIX_CHECK(optixSbtRecordPackHeader(ProgramGroup, &Result));

            return Result;
          });

          return Result;
        }();

        Result.missRecordBase = [&] {
          void *Result;

          CUDA_CHECK(cudaMalloc(&Result, sizeof(MissRecord) * std::size(MissRecords)));
          CUDA_CHECK(cudaMemcpy(Result, MissRecords.data(), sizeof(MissRecord) * std::size(MissRecords), cudaMemcpyHostToDevice));

          return reinterpret_cast<CUdeviceptr>(Result);
        }();

        Result.missRecordStrideInBytes = sizeof(MissRecord);
        Result.missRecordCount = std::size(MissRecords);
      }();

      return Result;
    }();
  }

  ~OptixState() {
    OPTIX_CHECK(optixPipelineDestroy(Pipeline));

    for (auto ProgramGroup : MissProgramGroups) {
      OPTIX_CHECK(optixProgramGroupDestroy(ProgramGroup));
    }

    for (auto ProgramGroup : HitgroupProgramGroups) {
      OPTIX_CHECK(optixProgramGroupDestroy(ProgramGroup));
    }

    for (auto ProgramGroup : RaygenProgramGroups) {
      OPTIX_CHECK(optixProgramGroupDestroy(ProgramGroup));
    }

    OPTIX_CHECK(optixModuleDestroy(Module));
    OPTIX_CHECK(optixDeviceContextDestroy(DeviceContext));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(ShaderBindingTable.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(ShaderBindingTable.hitgroupRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(ShaderBindingTable.raygenRecord)));
  }

  const auto &stream() const {
    return Stream;
  }

  const auto &deviceContext() const {
    return DeviceContext;
  }

  const auto &pipeline() const {
    return Pipeline;
  }

  const auto &raygenProgramGroups() const {
    return RaygenProgramGroups;
  }

  const auto &shaderBindingTable() const {
    return ShaderBindingTable;
  }
};

} // namespace osc
