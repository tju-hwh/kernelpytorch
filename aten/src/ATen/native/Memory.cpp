#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <c10/util/Exception.h>
#include <c10/core/Storage.h>

namespace at {
namespace native {

bool is_pinned(const Tensor& self) {
  return detail::getCUDAHooks().isPinnedPtr(self.storage().data());
}

Tensor pin_memory(const Tensor& self) {
  if (!self.device().is_cpu()) {
    AT_ERROR("cannot pin '", self.toString(), "' only dense CPU tensors can be pinned");
  }
  if (self.is_pinned()) {
    TORCH_WARN("Already '", self.toString(), "' is pinned");
    return self;
  }
  auto* allocator = detail::getCUDAHooks().getPinnedMemoryAllocator();
  auto storage = Storage(
      Storage::use_byte_size_t(),
      detail::computeStorageNbytes(
          self.sizes(), self.strides(), self.dtype().itemsize()),
      allocator,
      /*resizable=*/false);
  auto tensor = at::empty({0}, self.options()).set_(storage, 0, self.sizes(), self.strides());
  tensor.copy_(self);
  return tensor;
}

Tensor cuda_host(const Tensor& self) {
  if (self.is_pinned() && self.device().is_cuda()) {
    TORCH_WARN("Already '", self.toString(), "' is pinned");
    return self;
  }
  auto* allocator = detail::getCUDAHooks().getCUDAHostAllocator();
  auto storage = Storage(
      Storage::use_byte_size_t(),
      detail::computeStorageNbytes(
        self.sizes(), self.strides(), self.dtype().itemsize()),
      allocator,
      /*resizable=*/false
      );

  auto specified_options = self.options();
  specified_options = specified_options.device(at::kCUDA);

  auto tensor = at::empty({0}, specified_options).set_(storage, 0, self.sizes(), self.strides());
  tensor.copy_(self);
  return tensor;
}

// Exposes at::has_internal_overlap as an operator for testing purposes
int64_t _debug_has_internal_overlap(const Tensor& self) {
  return static_cast<int64_t>(at::has_internal_overlap(self));
}

}
}
