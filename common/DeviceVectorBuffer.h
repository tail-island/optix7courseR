#pragma once

#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

#include "../common/util.h"

namespace osc {
namespace common {

template <typename T>
class DeviceVectorBuffer final {
  CUdeviceptr Data;
  std::size_t Size;

  auto newData() noexcept {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&Data), dataSize()));
  }

  auto deleteData() noexcept {
    if (!Data) {
      return;
    }

    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(Data)));
  }

public:
  DeviceVectorBuffer(std::size_t Size = 0) noexcept : Size(Size) {
    newData();
  }

  DeviceVectorBuffer(const DeviceVectorBuffer &Other) noexcept : Size(Other.Size) {
    newData();
  }

  DeviceVectorBuffer(DeviceVectorBuffer &&Other) noexcept : Data(Other.Data), Size(Other.Size) {
    Other.Data = 0;
    Other.Size = 0;
  }

  ~DeviceVectorBuffer() {
    deleteData();
  }

  auto data() const noexcept {
    return Data;
  }

  auto dataSize() const noexcept {
    return sizeof(T) * Size;
  }

  auto size() const noexcept {
    return Size;
  }

  auto set(const std::vector<T> &Data) noexcept {
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(DeviceVectorBuffer::Data), Data.data(), dataSize(), cudaMemcpyHostToDevice));
  }

  auto get() const noexcept {
    auto Result = std::vector<T>(Size);

    CUDA_CHECK(cudaMemcpy(Result.data(), reinterpret_cast<void *>(Data), dataSize(), cudaMemcpyDeviceToHost));

    return Result;
  }

  auto resize(std::size_t Size) noexcept {
    DeviceVectorBuffer::Size = Size;

    deleteData();
    newData();
  }
};

} // namespace common
} // namespace osc
