#pragma once

#include <cuda_runtime.h>

#include "../common/Util.h"

namespace osc {
namespace common {

template <typename T>
class DeviceBuffer final {
  CUdeviceptr data_;

  auto mallocData() noexcept {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&data_), getDataSize()));
  }

  auto freeData() noexcept {
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(data_)));
  }

public:
  DeviceBuffer() noexcept {
    mallocData();
  }

  DeviceBuffer(const DeviceBuffer<T> &other) noexcept : DeviceBuffer{} {
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(data_), reinterpret_cast<void *>(other.data_), getDataSize(), cudaMemcpyDeviceToDevice));
  }

  DeviceBuffer(DeviceBuffer<T> &&other) noexcept : data_{other.data_} {
    other.data_ = 0;
  }

  ~DeviceBuffer() {
    if (data_) {
      freeData();
    }
  }

  const auto &getData() const noexcept {
    return data_;
  }

  auto getDataSize() const noexcept {
    return sizeof(T);
  }

  auto set(const T &data) noexcept {
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(data_), &data, getDataSize(), cudaMemcpyHostToDevice));
  }

  auto get() const noexcept {
    auto result = T{};

    CUDA_CHECK(cudaMemcpy(&result, reinterpret_cast<void *>(data_), getDataSize(), cudaMemcpyDeviceToHost));

    return result;
  }
};

} // namespace common
} // namespace osc
