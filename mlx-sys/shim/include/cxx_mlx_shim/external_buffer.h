#pragma once

#include "cxx_mlx_shim/array.h"
#include "cxx_mlx_shim/compile.h"  // for ArrayVec
#include "mlx/event.h"
#include "rust/cxx.h"

#include <cstdint>
#include <memory>

namespace cxx_mlx {

// An MLX array over caller-owned, page-aligned host memory (for example an
// IOSurface shared with the Neural Engine), wrapped as a no-copy shared Metal
// buffer: GPU kernels read it in place. The memory must outlive the array.
std::unique_ptr<MlxArray> array_from_external_buffer(
    size_t address, size_t nbytes, rust::Slice<const int32_t> shape, uint8_t dtype_repr);

// Copy `src` (any strides, same shape and dtype) into the memory of the
// external array `dst` on the GPU, after every array of `after`; the result
// aliases `dst`'s memory and is complete once it is evaluated.
std::unique_ptr<MlxArray> copy_into_external(
    const MlxArray& src, const MlxArray& dst, const ArrayVec& after,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

// A Metal shared event that hands work between the GPU stream and another
// processor: the GPU signals or waits on it inside its command buffers, a CPU
// thread waits on or signals it.
struct HandoffEvent {
  mlx::core::Event event;
};

std::unique_ptr<HandoffEvent> handoff_event_new();

// `x` (aliased, no copy), once every array of `after` is complete and the GPU
// has signaled `value` on `event`.
std::unique_ptr<MlxArray> handoff_signal_after(
    const MlxArray& x, const ArrayVec& after, const HandoffEvent& event, uint64_t value,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

// `x` (aliased, no copy), read by the GPU only after every array of `after`
// and once `event` reaches `value`.
std::unique_ptr<MlxArray> handoff_wait_before(
    const MlxArray& x, const ArrayVec& after, const HandoffEvent& event, uint64_t value,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

// Blocks the calling thread until `event` reaches `value` or `timeout_ms`
// passes; whether it was reached.
bool handoff_cpu_wait(const HandoffEvent& event, uint64_t value, uint64_t timeout_ms);
void handoff_cpu_signal(const HandoffEvent& event, uint64_t value);
uint64_t handoff_signaled_value(const HandoffEvent& event);

}  // namespace cxx_mlx
