#pragma once

#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

#include "../common/Util.h"

namespace osc {
namespace common {

template <typename T>
class DeviceVectorBuffer final {
  CUdeviceptr data_;
  std::size_t size_;

  auto mallocData() noexcept {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&data_), getDataSize()));
  }

  auto freeData() noexcept {
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(data_)));
  }

public:
  DeviceVectorBuffer(std::size_t size = 0) noexcept : size_{size} {
    mallocData();
  }

  DeviceVectorBuffer(const DeviceVectorBuffer &other) noexcept : DeviceVectorBuffer{other.size_} {
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(data_), reinterpret_cast<void *>(other.data_), getDataSize(), cudaMemcpyDeviceToDevice));
  }

  DeviceVectorBuffer(DeviceVectorBuffer &&other) noexcept : data_{other.data_}, size_{other.size_} {
    other.data_ = 0;
    other.size_ = 0;
  }

  ~DeviceVectorBuffer() {
    if (data_) {
      freeData();
    }
  }

  const auto &getData() const noexcept {
    return data_;
  }

  auto getDataSize() const noexcept {
    return sizeof(T) * size_;
  }

  auto getSize() const noexcept {
    return size_;
  }

  auto setSize(std::size_t size) noexcept {
    DeviceVectorBuffer::size_ = size;

    freeData();
    mallocData();
  }

  auto set(const std::vector<T> &other) noexcept {
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(DeviceVectorBuffer::data_), other.data(), getDataSize(), cudaMemcpyHostToDevice));
  }

  auto get() const noexcept {
    auto result = std::vector<T>(size_);

    CUDA_CHECK(cudaMemcpy(result.data(), reinterpret_cast<void *>(data_), getDataSize(), cudaMemcpyDeviceToHost));

    return result;
  }
};

} // namespace common
} // namespace osc
