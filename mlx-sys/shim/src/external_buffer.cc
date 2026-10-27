#include "cxx_mlx_shim/external_buffer.h"
#include "cxx_mlx_shim/shim_helpers.h"

#include <stdexcept>

#if __has_include(<Metal/Metal.hpp>)
#include "mlx/allocator.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/event.h"
#include "mlx/primitives.h"

namespace cxx_mlx {
namespace {
namespace mx = mlx::core;

// Writes its first input into the buffer of its second (an external array)
// and outputs that buffer.
class CopyIntoExternal final : public mx::UnaryPrimitive {
 public:
  explicit CopyIntoExternal(mx::Stream stream) : UnaryPrimitive(stream) {}
  const char* name() const override { return "CopyIntoExternal"; }
  void eval_cpu(const std::vector<mx::array>&, mx::array&) override {
    throw std::runtime_error("CopyIntoExternal requires the GPU");
  }
  void eval_gpu(const std::vector<mx::array>& in, mx::array& out) override {
    out.copy_shared_buffer(in[1]);
    mx::copy_gpu_inplace(in[0], out, mx::CopyType::General, stream());
  }
};

// Outputs its first input's buffer after signaling an event on the GPU.
class HandoffSignal final : public mx::UnaryPrimitive {
 public:
  HandoffSignal(mx::Stream stream, mx::Event event, uint64_t value)
      : UnaryPrimitive(stream), event_(std::move(event)), value_(value) {}
  const char* name() const override { return "HandoffSignal"; }
  void eval_cpu(const std::vector<mx::array>&, mx::array&) override {
    throw std::runtime_error("HandoffSignal requires the GPU");
  }
  void eval_gpu(const std::vector<mx::array>& in, mx::array& out) override {
    out.copy_shared_buffer(in[0]);
    mx::Event event = event_;
    event.set_value(value_);
    event.signal(stream());
  }

 private:
  mx::Event event_;
  uint64_t value_;
};

// Outputs its input's buffer once the GPU has waited for an event.
class HandoffWait final : public mx::UnaryPrimitive {
 public:
  HandoffWait(mx::Stream stream, mx::Event event, uint64_t value)
      : UnaryPrimitive(stream), event_(std::move(event)), value_(value) {}
  const char* name() const override { return "HandoffWait"; }
  void eval_cpu(const std::vector<mx::array>&, mx::array&) override {
    throw std::runtime_error("HandoffWait requires the GPU");
  }
  void eval_gpu(const std::vector<mx::array>& in, mx::array& out) override {
    mx::Event event = event_;
    event.set_value(value_);
    event.wait(stream());
    out.copy_shared_buffer(in[0]);
  }

 private:
  mx::Event event_;
  uint64_t value_;
};

mx::Stream gpu_stream(bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  const auto stream =
      mx::to_stream(helpers::decode_stream_or_device(has_target, is_device_only, device_type, stream_index));
  if (stream.device != mx::Device::gpu) throw std::runtime_error("handoffs run on a GPU stream");
  return stream;
}
}  // namespace

std::unique_ptr<MlxArray> array_from_external_buffer(
    size_t address, size_t nbytes, rust::Slice<const int32_t> shape, uint8_t dtype_repr) {
  if (address == 0 || address % 16384 != 0 || nbytes == 0 || nbytes % 16384 != 0)
    throw std::runtime_error("external buffer must be page aligned in address and length");
  const auto dtype = helpers::dtype_from_repr(dtype_repr);
  mx::Shape s(shape.begin(), shape.end());
  size_t elements = 1;
  for (auto d : s) {
    if (d <= 0) throw std::runtime_error("external buffer shape must be positive");
    elements *= size_t(d);
  }
  if (elements * mx::size_of(dtype) > nbytes) throw std::runtime_error("external buffer too small for its shape");
  auto& d = mx::metal::device(mx::Device::gpu);
  MTL::Buffer* buffer = d.mtl_device()->newBuffer(
      reinterpret_cast<const void*>(address), nbytes, MTL::ResourceStorageModeShared, nullptr);
  if (!buffer) throw std::runtime_error("cannot wrap the external buffer as a Metal buffer");
  mx::array a(mx::allocator::Buffer(static_cast<void*>(buffer)), std::move(s), dtype,
              [](mx::allocator::Buffer b) { static_cast<MTL::Buffer*>(b.ptr())->release(); });
  return std::make_unique<MlxArray>(std::move(a));
}

std::unique_ptr<MlxArray> copy_into_external(
    const MlxArray& src, const MlxArray& dst, const ArrayVec& after,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  const auto& s = src;
  const auto& t = dst;
  if (s.shape() != t.shape() || s.dtype() != t.dtype())
    throw std::runtime_error("copy_into_external: shape or dtype differs");
  const auto stream = gpu_stream(has_target, is_device_only, device_type, stream_index);
  std::vector<mx::array> inputs{s, t};
  inputs.insert(inputs.end(), after.inner.begin(), after.inner.end());
  return std::make_unique<MlxArray>(
      mx::array(t.shape(), t.dtype(), std::make_shared<CopyIntoExternal>(stream), std::move(inputs)));
}

std::unique_ptr<HandoffEvent> handoff_event_new() {
  return std::make_unique<HandoffEvent>(HandoffEvent{mx::Event(mx::default_stream(mx::Device::gpu))});
}

std::unique_ptr<MlxArray> handoff_signal_after(
    const MlxArray& x, const ArrayVec& after, const HandoffEvent& event, uint64_t value,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  const auto stream = gpu_stream(has_target, is_device_only, device_type, stream_index);
  std::vector<mx::array> inputs{x};
  inputs.insert(inputs.end(), after.inner.begin(), after.inner.end());
  return std::make_unique<MlxArray>(mx::array(
      x.shape(), x.dtype(), std::make_shared<HandoffSignal>(stream, event.event, value), std::move(inputs)));
}

std::unique_ptr<MlxArray> handoff_wait_before(
    const MlxArray& x, const ArrayVec& after, const HandoffEvent& event, uint64_t value,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  const auto stream = gpu_stream(has_target, is_device_only, device_type, stream_index);
  std::vector<mx::array> inputs{x};
  inputs.insert(inputs.end(), after.inner.begin(), after.inner.end());
  return std::make_unique<MlxArray>(mx::array(
      x.shape(), x.dtype(), std::make_shared<HandoffWait>(stream, event.event, value), std::move(inputs)));
}

bool handoff_cpu_wait(const HandoffEvent& event, uint64_t value, uint64_t timeout_ms) {
  return event.event.cast<mx::metal::EventImpl>().mtl_event()->waitUntilSignaledValue(value, timeout_ms);
}

void handoff_cpu_signal(const HandoffEvent& event, uint64_t value) {
  event.event.cast<mx::metal::EventImpl>().signal(value);
}

uint64_t handoff_signaled_value(const HandoffEvent& event) {
  return event.event.cast<mx::metal::EventImpl>().mtl_event()->signaledValue();
}
}  // namespace cxx_mlx
#else
namespace cxx_mlx {
std::unique_ptr<MlxArray> array_from_external_buffer(size_t, size_t, rust::Slice<const int32_t>, uint8_t) {
  throw std::runtime_error("external buffers require Metal");
}
std::unique_ptr<MlxArray> copy_into_external(const MlxArray&, const MlxArray&, const ArrayVec&, bool, bool, uint8_t, int32_t) {
  throw std::runtime_error("external buffers require Metal");
}
std::unique_ptr<HandoffEvent> handoff_event_new() { throw std::runtime_error("handoffs require Metal"); }
std::unique_ptr<MlxArray> handoff_signal_after(const MlxArray&, const ArrayVec&, const HandoffEvent&, uint64_t, bool, bool,
                                               uint8_t, int32_t) {
  throw std::runtime_error("handoffs require Metal");
}
std::unique_ptr<MlxArray> handoff_wait_before(const MlxArray&, const ArrayVec&, const HandoffEvent&, uint64_t, bool, bool, uint8_t,
                                              int32_t) {
  throw std::runtime_error("handoffs require Metal");
}
bool handoff_cpu_wait(const HandoffEvent&, uint64_t, uint64_t) { throw std::runtime_error("handoffs require Metal"); }
void handoff_cpu_signal(const HandoffEvent&, uint64_t) { throw std::runtime_error("handoffs require Metal"); }
uint64_t handoff_signaled_value(const HandoffEvent&) { throw std::runtime_error("handoffs require Metal"); }
}  // namespace cxx_mlx
#endif
