#include <ATen/ATen.h>
#include <THC/THCTensorSync.h>
#include <c10/cuda/CUDAStream.h>

namespace at {
namespace native {

void record_tensor(Tensor& self, const Tensor& tensor, Device device) {
  void* old_ptr = self.data_ptr();
  void* new_ptr = tensor.data_ptr();

  if (self.is_cuda() && self.is_pinned())
    return;
  if (new_ptr == old_ptr) {
    return;
  }

  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  recordEvent(self, tensor, device.index(), stream);
}

Tensor& sync_device_(Tensor& self, Device device) {
  if (self.is_cuda() && self.is_pinned())
    return self;

  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  syncEvent(self.data_ptr(), device.index(), self.device().index(), stream);
  return self;
}

Tensor& sync_tensor_(Tensor& self, const Tensor& tensor) {
  if (self.is_cuda() && self.is_pinned())
    return self;
  if (!tensor.is_cuda())
    return self;

  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  syncEvent(self.data_ptr(), tensor.device().index(), self.device().index(), stream);
  return self;
}

}
}
