#include "cxx_mlx_shim/stream.h"

// The cxx-generated header is what defines Device/Stream as concrete types.
// Including it here is what makes the conversion functions have access to
// full type definitions.
#include "mlx-sys/src/bridge/stream.rs.h"

#include "mlx/memory.h"
#include "mlx/transforms.h"

namespace cxx_mlx {

// === Conversions (cxx_mlx ↔ mlx::core) ===
//
// Both representations have identical layout (same field order, same
// underlying integer types). We do field-by-field copy rather than
// reinterpret_cast — explicit and safe, the compiler optimizes it to
// register-level copy.
//
// Note: Device.device_type is stored as int32_t on the Rust side (wire type
// for the bridge), matching MLX's DeviceType underlying type (int/int32_t).

namespace {

mlx::core::Device::DeviceType to_mlx_dtype(int32_t t) {
  // Layout-compatible: the int32_t wire value matches MLX's enum class
  // underlying type (int). Values: Cpu=0, Gpu=1 match mlx/device.h:14-17.
  return static_cast<mlx::core::Device::DeviceType>(t);
}

int32_t from_mlx_dtype(mlx::core::Device::DeviceType t) {
  return static_cast<int32_t>(t);
}

mlx::core::Device to_mlx(Device d) {
  return mlx::core::Device(to_mlx_dtype(d.device_type), d.index);
}

Device from_mlx(const mlx::core::Device& d) {
  return Device{from_mlx_dtype(d.type), d.index};
}

mlx::core::Stream to_mlx(Stream s) {
  return mlx::core::Stream(s.index, to_mlx(s.device));
}

mlx::core::ThreadLocalStream to_mlx_thread_local(Stream s) {
  return mlx::core::ThreadLocalStream(s.index, to_mlx(s.device));
}

Stream from_mlx(const mlx::core::Stream& s) {
  return Stream{s.index, from_mlx(s.device)};
}

}  // namespace

// === Device API ===

Device default_device() {
  return from_mlx(mlx::core::default_device());
}

void set_default_device(Device d) {
  mlx::core::set_default_device(to_mlx(d));
}

bool is_available(Device d) {
  return mlx::core::is_available(to_mlx(d));
}

int32_t device_count(int32_t t) {
  return mlx::core::device_count(to_mlx_dtype(t));
}

// === Stream API ===

Stream default_stream(Device d) {
  return from_mlx(mlx::core::default_stream(to_mlx(d)));
}

Stream new_stream(Device d) {
  return from_mlx(mlx::core::new_stream(to_mlx(d)));
}

Stream new_thread_local_stream(Device d) {
  return from_mlx(mlx::core::new_thread_local_stream(to_mlx(d)));
}

Stream stream_from_thread_local_stream(Stream s) {
  return from_mlx(mlx::core::stream_from_thread_local_stream(to_mlx_thread_local(s)));
}

void set_default_stream(Stream s) {
  mlx::core::set_default_stream(to_mlx(s));
}

rust::Vec<Stream> get_streams() {
  auto streams = mlx::core::get_streams();
  rust::Vec<Stream> out;
  out.reserve(streams.size());
  for (const auto& s : streams) {
    out.push_back(from_mlx(s));
  }
  return out;
}

void clear_streams() {
  mlx::core::clear_streams();
}

// === Transforms ===

void eval_many(rust::Slice<const MlxArray* const> arrays) {
  std::vector<MlxArray> vec;
  vec.reserve(arrays.size());
  for (size_t i = 0; i < arrays.size(); ++i) {
    vec.push_back(*arrays[i]);  // copy ctor — refcount-shared, cheap
  }
  mlx::core::eval(std::move(vec));
}

void async_eval_many(rust::Slice<const MlxArray* const> arrays) {
  std::vector<MlxArray> vec;
  vec.reserve(arrays.size());
  for (size_t i = 0; i < arrays.size(); ++i) {
    vec.push_back(*arrays[i]);  // copy ctor — refcount-shared, cheap
  }
  mlx::core::async_eval(std::move(vec));
}

void synchronize() {
  mlx::core::synchronize();
}

void synchronize_stream(Stream s) {
  mlx::core::synchronize(to_mlx(s));
}

void synchronize_thread_local_stream(Stream s) {
  mlx::core::synchronize(to_mlx_thread_local(s));
}

void clear_cache() {
  mlx::core::clear_cache();
}

}  // namespace cxx_mlx
