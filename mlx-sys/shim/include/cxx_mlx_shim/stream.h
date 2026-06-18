#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "rust/cxx.h"
#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/stream.h"

namespace cxx_mlx {

// Forward-declare the cxx-generated types. Their definitions are emitted by
// cxx into the bridge's generated header; this header forward-declares to
// avoid an include cycle. The shim cc includes the generated header where
// the conversion functions need full type access.

struct Device;
struct Stream;

// Reuse MlxArray from the array shim.
using MlxArray = mlx::core::array;

// === Device ===
Device default_device();
void set_default_device(Device d);
bool is_available(Device d);
int32_t device_count(int32_t t);

// === Stream ===
Stream default_stream(Device d);
Stream new_stream(Device d);
Stream new_thread_local_stream(Device d);
Stream stream_from_thread_local_stream(Stream s);
void set_default_stream(Stream s);
rust::Vec<Stream> get_streams();
void clear_streams();

// === Transforms ===
void eval_many(rust::Slice<const MlxArray* const> arrays);
void async_eval_many(rust::Slice<const MlxArray* const> arrays);
void synchronize();
void synchronize_stream(Stream s);
void synchronize_thread_local_stream(Stream s);
void clear_cache();

}  // namespace cxx_mlx
