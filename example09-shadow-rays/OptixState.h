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

  std::vector<cudaArray_t> textureArrays_;
  std::vector<cudaTextureObject_t> textureObjects_;

  std::vector<common::DeviceVectorBuffer<Eigen::Vector3f>> verticesBuffers_;
  std::vector<common::DeviceVectorBuffer<Eigen::Vector3f>> normalsBuffers_;
  std::vector<common::DeviceVectorBuffer<Eigen::Vector2f>> textureCoordinatesBuffers_;
  std::vector<common::DeviceVectorBuffer<Eigen::Vector3i>> indicesBuffers_;

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

    // テクスチャーを作成します。

    [&] {
      for (const auto &texture : model.getTextures()) {
        auto textureArray = cudaArray_t{};
        auto channelDesc = cudaCreateChannelDesc<uchar4>();

        CUDA_CHECK(cudaMallocArray(&textureArray, &channelDesc, texture.getImageSize().x(), texture.getImageSize().y()));

        CUDA_CHECK(cudaMemcpy2DToArray(textureArray,
                                       0,
                                       0,
                                       texture.getImage().data(),
                                       texture.getImageSize().x() * sizeof(std::uint32_t),
                                       texture.getImageSize().x() * sizeof(std::uint32_t),
                                       texture.getImageSize().y(),
                                       cudaMemcpyHostToDevice));

        textureArrays_.emplace_back(textureArray);

        auto resourceDesc = [&] {
          auto result = cudaResourceDesc{};

          result.resType = cudaResourceTypeArray;
          result.res.array.array = textureArray;

          return result;
        }();

        auto textureDesc = [&] {
          auto result = cudaTextureDesc{};

          result.addressMode[0] = cudaAddressModeWrap;
          result.addressMode[1] = cudaAddressModeWrap;
          result.filterMode = cudaFilterModeLinear;
          result.readMode = cudaReadModeNormalizedFloat;
          result.normalizedCoords = 1;
          result.maxAnisotropy = 1;
          result.maxMipmapLevelClamp = 99;
          result.minMipmapLevelClamp = 0;
          result.mipmapFilterMode = cudaFilterModePoint;
          result.borderColor[0] = 1.0f;
          result.sRGB = 0;

          return result;
        }();

        auto textureObject = cudaTextureObject_t{};

        CUDA_CHECK(cudaCreateTextureObject(&textureObject, &resourceDesc, &textureDesc, nullptr));

        textureObjects_.emplace_back(textureObject);
      }
    }();

    // 後続処理のために、モデルのオブジェクトの各属性を抽出します。

    std::transform(std::begin(model.getObjects()), std::end(model.getObjects()), std::back_inserter(verticesBuffers_), [](const Object &object) {
      return common::DeviceVectorBuffer<Eigen::Vector3f>{object.getVertices()};
    });

    std::transform(std::begin(model.getObjects()), std::end(model.getObjects()), std::back_inserter(normalsBuffers_), [](const Object &object) {
      return common::DeviceVectorBuffer<Eigen::Vector3f>{object.getNormals()};
    });

    std::transform(std::begin(model.getObjects()), std::end(model.getObjects()), std::back_inserter(textureCoordinatesBuffers_), [](const Object &object) {
      return common::DeviceVectorBuffer<Eigen::Vector2f>{object.getTextureCoordinates()};
    });

    std::transform(std::begin(model.getObjects()), std::end(model.getObjects()), std::back_inserter(indicesBuffers_), [](const Object &object) {
      return common::DeviceVectorBuffer<Eigen::Vector3i>{object.getIndices()};
    });

    // OptixのTraversableHandleを生成します。

    [&] {
      const auto accelBuildOptions = [&] {
        auto result = OptixAccelBuildOptions{};

        result.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        result.motionOptions.numKeys = 1;
        result.operation = OPTIX_BUILD_OPERATION_BUILD;

        return result;
      }();

      const auto triangleArrayFlags = 0u;

      const auto buildInputs = [&] {
        auto result = std::vector<OptixBuildInput>{};

        for (auto i = 0; i < static_cast<int>(std::size(model.getObjects())); ++i) {
          result.emplace_back([&] {
            auto result = OptixBuildInput{};

            result.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

            result.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            result.triangleArray.vertexStrideInBytes = sizeof(Eigen::Vector3f);
            result.triangleArray.numVertices = verticesBuffers_[i].getSize();
            result.triangleArray.vertexBuffers = &verticesBuffers_[i].getData();

            result.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            result.triangleArray.indexStrideInBytes = sizeof(Eigen::Vector3i);
            result.triangleArray.numIndexTriplets = indicesBuffers_[i].getSize();
            result.triangleArray.indexBuffer = indicesBuffers_[i].getData();

            result.triangleArray.flags = &triangleArrayFlags;
            result.triangleArray.numSbtRecords = 1;
            result.triangleArray.sbtIndexOffsetBuffer = 0;
            result.triangleArray.sbtIndexOffsetSizeInBytes = 0;
            result.triangleArray.sbtIndexOffsetStrideInBytes = 0;

            return result;
          }());
        }

        return result;
      }();

      const auto accelBufferSizes = [&] {
        auto result = OptixAccelBufferSizes{};

        OPTIX_CHECK(optixAccelComputeMemoryUsage(deviceContext_, &accelBuildOptions, buildInputs.data(), std::size(buildInputs), &result));

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

      OPTIX_CHECK(optixAccelBuild(deviceContext_, 0, &accelBuildOptions, buildInputs.data(), std::size(buildInputs), tempBuffer.getData(), tempBuffer.getDataSize(), outputBuffer.getData(), outputBuffer.getDataSize(), &traversableHandle_, &accelEmitDesc, 1));

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

      auto result = std::vector<OptixProgramGroup>{static_cast<int>(RayType::Size)};

      [&] {
        const auto programGroupOptions = OptixProgramGroupOptions{};

        const auto programGroupDesc = [&] {
          auto result = OptixProgramGroupDesc{};

          result.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
          result.hitgroup.moduleCH = module_;
          result.hitgroup.entryFunctionNameCH = "__closesthit__surface";
          result.hitgroup.moduleAH = module_;
          result.hitgroup.entryFunctionNameAH = "__anyhit__surface";

          return result;
        }();

        auto [log, logSize] = [&] {
          auto result = std::array<char, 2048>{};

          return std::make_tuple(result, std::size(result));
        }();

        OPTIX_CHECK(optixProgramGroupCreate(deviceContext_, &programGroupDesc, 1, &programGroupOptions, log.data(), &logSize, &result[static_cast<int>(RayType::Surface)]));

        if (logSize > 1) { // 文字列の終端の'\0'があるので、１文字以上になります。
          std::cerr << log.data() << "\n";
        }
      }();

      [&] {
        const auto programGroupOptions = OptixProgramGroupOptions{};

        const auto programGroupDesc = [&] {
          auto result = OptixProgramGroupDesc{};

          result.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
          result.hitgroup.moduleCH = module_;
          result.hitgroup.entryFunctionNameCH = "__closesthit__shadow";
          result.hitgroup.moduleAH = module_;
          result.hitgroup.entryFunctionNameAH = "__anyhit__shadow";

          return result;
        }();

        auto [log, logSize] = [&] {
          auto result = std::array<char, 2048>{};

          return std::make_tuple(result, std::size(result));
        }();

        OPTIX_CHECK(optixProgramGroupCreate(deviceContext_, &programGroupDesc, 1, &programGroupOptions, log.data(), &logSize, &result[static_cast<int>(RayType::Shadow)]));

        if (logSize > 1) { // 文字列の終端の'\0'があるので、１文字以上になります。
          std::cerr << log.data() << "\n";
        }
      }();

      return result;
    }();

    // レイをトレースしても何かにぶつからなかった場合のプログラム・グループ群を作成します。処理の実態は*.cuの中にあります。

    missProgramGroups_ = [&] {
      std::cout << "#osc: creating MissProgramGroups...\n";

      auto result = std::vector<OptixProgramGroup>{static_cast<int>(RayType::Size)};

      [&] {
        const auto programGroupOptions = OptixProgramGroupOptions{};

        const auto programGroupDesc = [&] {
          auto result = OptixProgramGroupDesc{};

          result.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
          result.miss.module = module_;
          result.miss.entryFunctionName = "__miss__surface";

          return result;
        }();

        auto [log, logSize] = [&] {
          auto result = std::array<char, 2048>{};

          return std::make_tuple(result, std::size(result));
        }();

        OPTIX_CHECK(optixProgramGroupCreate(deviceContext_, &programGroupDesc, 1, &programGroupOptions, log.data(), &logSize, &result[static_cast<int>(RayType::Surface)]));

        if (logSize > 1) { // 文字列の終端の'\0'があるので、１文字以上になります。
          std::cerr << log.data() << "\n";
        }
      }();

      [&] {
        const auto programGroupOptions = OptixProgramGroupOptions{};

        const auto programGroupDesc = [&] {
          auto result = OptixProgramGroupDesc{};

          result.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
          result.miss.module = module_;
          result.miss.entryFunctionName = "__miss__shadow";

          return result;
        }();

        auto [log, logSize] = [&] {
          auto result = std::array<char, 2048>{};

          return std::make_tuple(result, std::size(result));
        }();

        OPTIX_CHECK(optixProgramGroupCreate(deviceContext_, &programGroupDesc, 1, &programGroupOptions, log.data(), &logSize, &result[static_cast<int>(RayType::Shadow)]));

        if (logSize > 1) { // 文字列の終端の'\0'があるので、１文字以上になります。
          std::cerr << log.data() << "\n";
        }
      }();

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

          for (const auto &programGroup : raygenProgramGroups_) {
            result.emplace_back([&] {
              auto result = RaygenRecord{};

              OPTIX_CHECK(optixSbtRecordPackHeader(programGroup, &result));

              return result;
            }());
          }

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

          for (auto i = 0; i < static_cast<int>(std::size(model.getObjects())); ++i) {
            for (const auto &programGroup : hitgroupProgramGroups_) {
              result.emplace_back([&] {
                auto result = HitgroupRecord{};

                OPTIX_CHECK(optixSbtRecordPackHeader(programGroup, &result));

                result.triangleMeshes.vertices = reinterpret_cast<Eigen::Vector3f *>(verticesBuffers_[i].getData());
                result.triangleMeshes.normals = reinterpret_cast<Eigen::Vector3f *>(normalsBuffers_[i].getData());
                result.triangleMeshes.textureCoordinates = reinterpret_cast<Eigen::Vector2f *>(textureCoordinatesBuffers_[i].getData());
                result.triangleMeshes.indices = reinterpret_cast<Eigen::Vector3i *>(indicesBuffers_[i].getData());
                result.triangleMeshes.color = model.getObjects()[i].getDiffuse();

                if (model.getObjects()[i].getDiffuseTextureIndex() >= 0) {
                  result.triangleMeshes.hasTextureObject = true;
                  result.triangleMeshes.textureObject = textureObjects_[model.getObjects()[i].getDiffuseTextureIndex()];
                } else {
                  result.triangleMeshes.hasTextureObject = false;
                }

                return result;
              }());
            }
          }

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

          for (const auto &programGroup : missProgramGroups_) {
            result.emplace_back([&] {
              auto result = MissRecord{};

              OPTIX_CHECK(optixSbtRecordPackHeader(programGroup, &result));

              return result;
            }());
          }

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
