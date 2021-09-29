#pragma once

#include <cuda_runtime.h>

#include "../common/util.h"

namespace osc {

template <typename T>
class DeviceBuffer final {
  CUdeviceptr Data;

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
  DeviceBuffer() noexcept {
    newData();
  }

  DeviceBuffer(const DeviceBuffer &Other) noexcept {
    newData();
  }

  DeviceBuffer(DeviceBuffer &&Other) noexcept : Data(Other.Data) {
    Other.Data = 0;
  }

  ~DeviceBuffer() {
    deleteData();
  }

  auto data() const noexcept {
    return Data;
  }

  auto dataSize() const noexcept {
    return sizeof(T);
  }

  auto set(const T &Data) noexcept {
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(DeviceBuffer::Data), &Data, dataSize(), cudaMemcpyHostToDevice));
  }

  auto get() const noexcept {
    auto Result = T{};

    CUDA_CHECK(cudaMemcpy(&Result, reinterpret_cast<void *>(Data), dataSize(), cudaMemcpyDeviceToHost));

    return Result;
  }
};

} // namespace osc
