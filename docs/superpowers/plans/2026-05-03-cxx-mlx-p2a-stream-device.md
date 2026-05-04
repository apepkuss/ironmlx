# cxx-mlx P2a (Stream/Device + async transforms) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `Device` / `Stream` types as cxx shared POD structs, expose 9 device/stream lifecycle functions, and provide runtime-agnostic `async_eval` returning `impl Future` (via the `blocking` crate) plus matching synchronous transforms.

**Architecture:** First time the project uses cxx shared structs (vs opaque `MlxArray`/`MlxArrayVec` patterns). `Device` and `Stream` are POD (8 bytes / 12 bytes) with binary-compatible layout to MLX C++ via field-by-field conversion in shim. The async story uses `blocking::unblock` — a thread pool for wrapping blocking syscalls in `Future`s without coupling to any specific async runtime. **Critical correctness detail**: MLX's no-arg `synchronize()` is thread-local; the future captures the submission stream at construction time and uses `synchronize_stream(captured)` to avoid the "wrong default stream on blocking pool worker" bug.

**Tech Stack:** Rust 1.94+, cxx 1.0 (shared structs), MLX C++ 0.32, C++20. New dep: `blocking = "1"`.

**Branch:** Work on `p2-fast-io` (already created off master). MLX install at `$HOME/.local/mlx`; export `MLX_DIR=$HOME/.local/mlx` for every cargo invocation.

---

## File Structure

**New files:**

- `mlx-sys/src/bridge/stream.rs` — cxx::bridge with `DeviceType`/`Device`/`Stream` shared structs + 13 FFI functions
- `mlx-sys/shim/include/cxx_mlx_shim/stream.h` — shim header (bidirectional conversions between cxx shared types and `mlx::core::Device`/`Stream`)
- `mlx-sys/shim/src/stream.cc` — shim implementations
- `mlx/src/device.rs` — `Device` re-export + ergonomic constructors + 4 top-level device fns
- `mlx/src/stream.rs` — `Stream` re-export + 5 top-level stream lifecycle fns
- `mlx/src/transforms.rs` — `async_eval` (Future) + `synchronize` + `synchronize_stream` (sync)
- `mlx/tests/p2a_device.rs` — Device basics (~7 tests)
- `mlx/tests/p2a_stream.rs` — Stream lifecycle (~6 tests)
- `mlx/tests/p2a_async.rs` — async_eval/synchronize integration (~6 tests, runs under both `futures_lite` and `tokio`)

**Modified files:**

- `mlx-sys/src/bridge/mod.rs` — `pub mod stream;`
- `mlx-sys/build.rs` — add `"src/bridge/stream.rs"` to `cxx_build::bridges([...])` and `.file("shim/src/stream.cc")`
- `mlx-sys/tests/sys_smoke.rs` — add 2 link-test smokes
- `mlx/Cargo.toml` — add `blocking = "1"` to `[dependencies]`; add `tokio = { version = "1", features = ["rt", "macros"] }` to `[dev-dependencies]` (for async tests; not a runtime lock-in)
- `mlx/src/lib.rs` — `mod device; mod stream; mod transforms;` + re-exports of `Device`, `DeviceType`, `Stream`
- `README.md` — add "Streams & Devices" section with sync + async examples

---

## Task 1: cxx::bridge for stream module + shared structs + 13 FFI functions

**Files:**
- Create: `mlx-sys/src/bridge/stream.rs`
- Create: `mlx-sys/shim/include/cxx_mlx_shim/stream.h`
- Create: `mlx-sys/shim/src/stream.cc`
- Modify: `mlx-sys/src/bridge/mod.rs`
- Modify: `mlx-sys/build.rs`
- Modify: `mlx-sys/tests/sys_smoke.rs`

**Critical layout note**: MLX's `Device::DeviceType` is `enum class { cpu, gpu }` with no explicit underlying type — that's `int` (32-bit) per C++ rules. Our cxx shared `DeviceType` MUST be `#[repr(i32)]` to match — using `#[repr(u8)]` would create a layout mismatch and `reinterpret_cast` UB. We avoid `reinterpret_cast` entirely and do explicit field-by-field conversion in the shim, but the underlying type still needs to match for cxx-generated C++ to produce identical layout.

- [ ] **Step 1: Write failing sys-side smoke tests**

Append to `mlx-sys/tests/sys_smoke.rs`:

```rust
#[test]
fn device_default_links() {
    let d = mlx_sys::stream::ffi::default_device();
    // On macOS Apple Silicon the default is GPU (DeviceType::Gpu = 1)
    assert_eq!(d.device_type as i32, mlx_sys::stream::ffi::DeviceType::Gpu as i32);
    assert_eq!(d.index, 0);
    assert!(mlx_sys::stream::ffi::is_available(d));
}

#[test]
fn stream_default_and_new_links() {
    let d = mlx_sys::stream::ffi::default_device();
    let default_stream = mlx_sys::stream::ffi::default_stream(d);
    let new_stream = mlx_sys::stream::ffi::new_stream(d).expect("new_stream should succeed");
    assert_ne!(default_stream.index, new_stream.index, "new stream should have different index");
    assert_eq!(new_stream.device.index, d.index);
}
```

- [ ] **Step 2: Verify failure**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx-sys --test sys_smoke device_default_links 2>&1 | tail -10
```
Expected: FAIL with `cannot find function default_device in module mlx_sys::stream::ffi` or `failed to resolve: could not find stream in mlx_sys`.

- [ ] **Step 3: Create `mlx-sys/src/bridge/stream.rs`**

```rust
//! Bridge for MLX Device/Stream types and async transforms.
//!
//! Uses cxx shared structs (Device, Stream, DeviceType) for zero-overhead
//! POD value passing. Layout binary-compatible with mlx::core::Device/Stream
//! (same field order, same underlying types). Conversion to/from MLX native
//! types happens in the C++ shim via field-by-field copy.

#[cxx::bridge(namespace = "cxx_mlx")]
pub mod ffi {
    /// Mirror of `mlx::core::Device::DeviceType`. The underlying type MUST
    /// match MLX's enum class default (int = i32) for layout compatibility.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[repr(i32)]
    pub enum DeviceType {
        Cpu = 0,
        Gpu = 1,
    }

    /// Mirror of `mlx::core::Device`. POD, 8 bytes, layout-compatible with MLX.
    /// Construct via `cxx_mlx::Device { device_type, index }` literal — fields
    /// are pub by cxx convention. Safe-layer crate provides ergonomic
    /// `Device::cpu()` / `Device::gpu(index)` constructors.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Device {
        pub device_type: DeviceType,
        pub index: i32,
    }

    /// Mirror of `mlx::core::Stream`. POD, 12 bytes, layout-compatible.
    /// Streams must be obtained via `default_stream()` / `new_stream()` —
    /// constructing one with arbitrary `index` is undefined behavior in MLX
    /// (the stream worker indexed by it may not exist).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Stream {
        pub index: i32,
        pub device: Device,
    }

    unsafe extern "C++" {
        include!("cxx_mlx_shim/stream.h");

        // Reuse the MlxArray opaque type from the array bridge for async_eval.
        type MlxArray = crate::bridge::array::ffi::MlxArray;

        // === Device queries ===
        fn default_device() -> Device;
        fn set_default_device(d: Device);
        fn is_available(d: Device) -> bool;
        fn device_count(t: DeviceType) -> i32;

        // === Stream lifecycle ===
        fn default_stream(d: Device) -> Stream;
        fn new_stream(d: Device) -> Result<Stream>;
        fn set_default_stream(s: Stream);
        fn get_streams() -> Vec<Stream>;
        fn clear_streams();

        // === Transforms ===
        /// # Safety
        ///
        /// Each pointer in `arrays` must point to a valid `MlxArray` that lives
        /// for the duration of this call. MLX `async_eval` copies arrays
        /// internally (refcount-share), so pointers need not outlive the call.
        unsafe fn async_eval_many(arrays: &[*const MlxArray]) -> Result<()>;
        fn synchronize() -> Result<()>;
        fn synchronize_stream(s: Stream) -> Result<()>;
    }
}
```

(The bridge-level `#[allow(clippy::missing_safety_doc)]` was added at the top of `mlx-sys/src/bridge/array.rs` in P1b2a; for this new bridge file we can either add the same allow or rely on the inline `# Safety` doc on `async_eval_many`. Since cxx may strip the doc inside the macro, add the same `#[allow(clippy::missing_safety_doc)]` at the top of `bridge/stream.rs`.)

Add at the very top of the file before the `#[cxx::bridge(...)]`:

```rust
// cxx::bridge generates `unsafe fn` declarations for our pointer-slice variants
// (async_eval_many). The Safety contract is documented in the safe Rust wrapper
// (`mlx::transforms::async_eval`); cxx doesn't propagate doc comments from
// inside the bridge macro.
#[allow(clippy::missing_safety_doc)]
```

- [ ] **Step 4: Create `mlx-sys/shim/include/cxx_mlx_shim/stream.h`**

```cpp
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
// cxx into the bridge's generated header, which this header is not allowed
// to include (would create a circular dependency). Instead we forward-declare
// and the conversion functions are defined in stream.cc which DOES include
// the cxx-generated header.

enum class DeviceType : int32_t;
struct Device;
struct Stream;

// Reuse MlxArray from the array shim.
using MlxArray = mlx::core::array;

// === Device ===
Device default_device();
void set_default_device(Device d);
bool is_available(Device d);
int32_t device_count(DeviceType t);

// === Stream ===
Stream default_stream(Device d);
Stream new_stream(Device d);
void set_default_stream(Stream s);
rust::Vec<Stream> get_streams();
void clear_streams();

// === Transforms ===
void async_eval_many(rust::Slice<const MlxArray* const> arrays);
void synchronize();
void synchronize_stream(Stream s);

}  // namespace cxx_mlx
```

- [ ] **Step 5: Create `mlx-sys/shim/src/stream.cc`**

```cpp
#include "cxx_mlx_shim/stream.h"

// The cxx-generated header is what defines DeviceType/Device/Stream as
// concrete types (with the layout we declared in the bridge). Including it
// here is what makes the conversion functions complete.
#include "mlx-sys/src/bridge/stream.rs.h"

#include "mlx/transforms.h"

namespace cxx_mlx {

// === Conversions (cxx_mlx ↔ mlx::core) ===
//
// Both representations have identical layout (same field order, same
// underlying integer types). We do field-by-field copy rather than
// reinterpret_cast — explicit and safe, the compiler optimizes it to
// register-level copy.

namespace {

mlx::core::Device::DeviceType to_mlx_dtype(DeviceType t) {
  // Layout-compatible: cxx_mlx::DeviceType is repr(i32), mlx default
  // enum class underlying is int (i32). Values match (Cpu=0, Gpu=1
  // matches cpu=0, gpu=1 — declaration order in mlx/device.h:14-17).
  return static_cast<mlx::core::Device::DeviceType>(static_cast<int32_t>(t));
}

DeviceType from_mlx_dtype(mlx::core::Device::DeviceType t) {
  return static_cast<DeviceType>(static_cast<int32_t>(t));
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

int32_t device_count(DeviceType t) {
  return mlx::core::device_count(to_mlx_dtype(t));
}

// === Stream API ===

Stream default_stream(Device d) {
  return from_mlx(mlx::core::default_stream(to_mlx(d)));
}

Stream new_stream(Device d) {
  return from_mlx(mlx::core::new_stream(to_mlx(d)));
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

}  // namespace cxx_mlx
```

- [ ] **Step 6: Register the bridge module in `mlx-sys/src/bridge/mod.rs`**

Append to `mlx-sys/src/bridge/mod.rs` (which currently has `pub mod array;` and `pub mod transforms;`):

```rust
pub mod stream;
```

Also add a re-export at the top of `mlx-sys/src/lib.rs` (read it first to find the right spot):

```rust
pub use bridge::stream;
```

(Place it next to the existing `pub use bridge::array;` and `pub use bridge::transforms;`.)

- [ ] **Step 7: Wire the new shim into `mlx-sys/build.rs`**

In `mlx-sys/build.rs`, find the existing `cxx_build::bridges([...])` call. Update both the bridges list and the file list to include the new stream files:

```rust
    cxx_build::bridges([
        "src/bridge/array.rs",
        "src/bridge/transforms.rs",
        "src/bridge/stream.rs",
    ])
    .file("shim/src/array.cc")
    .file("shim/src/transforms.cc")
    .file("shim/src/stream.cc")
    .include("shim/include")
    .include(&include_dir)
    .std("c++20")
    .flag_if_supported("-fvisibility=hidden")
    .compile("cxx_mlx_shim");
```

- [ ] **Step 8: Verify the smoke tests pass**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx-sys --test sys_smoke 2>&1 | tail -15
```
Expected: 15 sys tests pass (13 pre-existing + 2 new).

If the C++ build fails:
- `mlx-sys/src/bridge/stream.rs.h` not found: cxx-build emits the header into `OUT_DIR/cxxbridge/...`. Check that `cxx_build::bridges([...])` includes the stream bridge.
- Layout mismatch errors: re-verify `#[repr(i32)]` on `DeviceType` (Step 3).
- Static_cast errors on enum: ensure cxx-generated `cxx_mlx::DeviceType` and `mlx::core::Device::DeviceType` both have underlying int type.

If the Rust bridge fails:
- `cannot find type MlxArray`: ensure `type MlxArray = crate::bridge::array::ffi::MlxArray;` is inside the `extern "C++"` block (P1a established this cross-bridge sharing pattern).
- `Hash` not derivable on cxx shared struct: omit it (we don't need Hash for P2a — see spec A1 fallback note).

- [ ] **Step 9: Commit**

```bash
git add mlx-sys/src/bridge/ mlx-sys/shim/ mlx-sys/build.rs mlx-sys/src/lib.rs mlx-sys/tests/sys_smoke.rs
git commit -m "feat(p2a): add stream bridge with shared Device/Stream POD structs (13 FFI fns)"
```

---

## Task 2: `mlx::device` module

**Files:**
- Create: `mlx/src/device.rs`
- Create: `mlx/tests/p2a_device.rs`
- Modify: `mlx/src/lib.rs`

- [ ] **Step 1: Write failing tests**

Create `mlx/tests/p2a_device.rs`:

```rust
use mlx::{Device, DeviceType};

#[test]
fn cpu_constructor() {
    let d = Device::cpu();
    assert_eq!(d.device_type, DeviceType::Cpu);
    assert_eq!(d.index, 0);
}

#[test]
fn gpu_constructor() {
    let d = Device::gpu(0);
    assert_eq!(d.device_type, DeviceType::Gpu);
    assert_eq!(d.index, 0);

    let d2 = Device::gpu(3);
    assert_eq!(d2.index, 3);
}

#[test]
fn device_equality_and_copy() {
    let a = Device::gpu(0);
    let b = a;  // Copy
    assert_eq!(a, b);
    assert_ne!(a, Device::cpu());
    assert_ne!(Device::gpu(0), Device::gpu(1));
}

#[test]
fn default_device_is_gpu_on_apple_silicon() {
    // On macOS Apple Silicon the default device is the GPU.
    let d = mlx::default_device();
    assert_eq!(d.device_type, DeviceType::Gpu);
}

#[test]
fn cpu_and_gpu_both_available() {
    assert!(mlx::is_available(Device::cpu()));
    assert!(mlx::is_available(Device::gpu(0)));
}

#[test]
fn gpu_device_count_at_least_one() {
    assert!(mlx::device_count(DeviceType::Gpu) >= 1);
    // CPU "count" semantics: MLX returns 1 for CPU (single logical device).
    assert!(mlx::device_count(DeviceType::Cpu) >= 1);
}

#[test]
fn set_default_device_round_trip() {
    let original = mlx::default_device();
    mlx::set_default_device(Device::cpu());
    assert_eq!(mlx::default_device(), Device::cpu());
    // Restore so other tests aren't affected.
    mlx::set_default_device(original);
    assert_eq!(mlx::default_device(), original);
}
```

- [ ] **Step 2: Verify failure**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p2a_device 2>&1 | tail -10
```
Expected: FAIL with `unresolved import mlx::Device` (or similar).

- [ ] **Step 3: Create `mlx/src/device.rs`**

```rust
//! Device types and queries.
//!
//! `Device` is a POD struct (8 bytes) describing where MLX should run
//! computations. On macOS Apple Silicon, the default is the GPU (Metal).
//! `Device::cpu()` and `Device::gpu(index)` are convenience constructors;
//! the type itself is `Copy + PartialEq + Eq + Debug`.

pub use mlx_sys::stream::ffi::{Device, DeviceType};

impl Device {
    /// CPU device (always index 0; MLX exposes only one CPU "device").
    pub const fn cpu() -> Self {
        Device {
            device_type: DeviceType::Cpu,
            index: 0,
        }
    }

    /// GPU device with the given index. On macOS Apple Silicon there is
    /// typically only one GPU (index 0).
    pub const fn gpu(index: i32) -> Self {
        Device {
            device_type: DeviceType::Gpu,
            index,
        }
    }
}

/// Get the current thread's default device (where ops execute by default).
pub fn default_device() -> Device {
    mlx_sys::stream::ffi::default_device()
}

/// Set the current thread's default device. Subsequent ops on this thread
/// will execute on `d` unless an explicit stream/device override is provided.
///
/// This is **thread-local** in MLX — setting on thread A does not affect
/// thread B.
pub fn set_default_device(d: Device) {
    mlx_sys::stream::ffi::set_default_device(d);
}

/// Returns `true` if MLX has the given device available on this system.
pub fn is_available(d: Device) -> bool {
    mlx_sys::stream::ffi::is_available(d)
}

/// Number of devices of the given type available on this system.
pub fn device_count(t: DeviceType) -> i32 {
    mlx_sys::stream::ffi::device_count(t)
}
```

- [ ] **Step 4: Wire `mod device;` in `mlx/src/lib.rs`**

In `mlx/src/lib.rs`, add `mod device;` and re-exports. The full mod/use block should become (preserving everything else):

```rust
mod array;
mod broadcast;
mod device;
mod dtype;
mod element;
mod error;
pub mod ops;
mod ops_impl;

pub use array::Array;
pub use broadcast::broadcast_shape;
pub use device::{default_device, device_count, is_available, set_default_device, Device, DeviceType};
pub use dtype::Dtype;
pub use element::Element;
pub use error::{Error, Result};
pub use ops::All;
```

(Adds `mod device;` and the `pub use device::...` line. Other lines unchanged.)

- [ ] **Step 5: Verify**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p2a_device 2>&1 | grep "test result:"
```
Expected: 7 tests pass.

- [ ] **Step 6: Commit**

```bash
git add mlx/src/device.rs mlx/src/lib.rs mlx/tests/p2a_device.rs
git commit -m "feat(p2a): mlx::device module — Device + queries (7 tests)"
```

---

## Task 3: `mlx::stream` module

**Files:**
- Create: `mlx/src/stream.rs`
- Create: `mlx/tests/p2a_stream.rs`
- Modify: `mlx/src/lib.rs`

- [ ] **Step 1: Write failing tests**

Create `mlx/tests/p2a_stream.rs`:

```rust
use mlx::{Device, Stream};

#[test]
fn default_stream_for_default_device() {
    let d = mlx::default_device();
    let s = mlx::default_stream(d);
    assert_eq!(s.device, d);
}

#[test]
fn new_stream_has_unique_index() {
    let d = mlx::default_device();
    let default_s = mlx::default_stream(d);
    let new_s = mlx::new_stream(d).expect("new_stream");
    assert_ne!(default_s.index, new_s.index, "new stream should have a fresh index");
    assert_eq!(new_s.device, d);
}

#[test]
fn get_streams_includes_default() {
    let d = mlx::default_device();
    let default_s = mlx::default_stream(d);
    let all = mlx::get_streams();
    assert!(
        all.iter().any(|s| s.index == default_s.index && s.device == d),
        "default stream should appear in get_streams()"
    );
}

#[test]
fn set_default_stream_round_trip() {
    let d = mlx::default_device();
    let original = mlx::default_stream(d);
    let new_s = mlx::new_stream(d).expect("new_stream");
    mlx::set_default_stream(new_s);
    assert_eq!(mlx::default_stream(d), new_s);
    // Restore.
    mlx::set_default_stream(original);
    assert_eq!(mlx::default_stream(d), original);
}

#[test]
fn stream_equality_and_copy() {
    let d = mlx::default_device();
    let a = mlx::default_stream(d);
    let b = a;  // Copy
    assert_eq!(a, b);
    let other = mlx::new_stream(d).expect("new_stream");
    assert_ne!(a, other);
}

#[test]
fn clear_streams_then_default_still_works() {
    // clear_streams destroys all streams created on current thread.
    // The default stream is recreated lazily by MLX.
    mlx::clear_streams();
    let d = mlx::default_device();
    let _s = mlx::default_stream(d);  // Must not panic.
}
```

- [ ] **Step 2: Verify failure**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p2a_stream 2>&1 | tail -10
```
Expected: FAIL with `unresolved import mlx::Stream`.

- [ ] **Step 3: Create `mlx/src/stream.rs`**

```rust
//! Stream lifecycle and management.
//!
//! `Stream` is a POD struct (12 bytes) representing an MLX execution stream.
//! Ops queued on the same stream execute in order; ops on different streams
//! may run concurrently. Streams are bound to a specific [`Device`].
//!
//! **Construction**: `Stream` literals (`Stream { index, device }`) are
//! technically possible because cxx shared-struct fields are pub, but
//! constructing arbitrary indices is **not safe** — only indices returned
//! by [`default_stream`] / [`new_stream`] correspond to real MLX stream
//! workers. Always obtain streams via these functions.

pub use mlx_sys::stream::ffi::Stream;

use crate::{Device, Error, Result};

/// Get the default stream for the given device on the current thread.
pub fn default_stream(d: Device) -> Stream {
    mlx_sys::stream::ffi::default_stream(d)
}

/// Create a new stream on the given device. The returned stream has a
/// fresh, unique index.
pub fn new_stream(d: Device) -> Result<Stream> {
    mlx_sys::stream::ffi::new_stream(d).map_err(Error::from)
}

/// Make the stream the default for its device on the current thread.
/// Subsequent ops on this thread that target the stream's device will use
/// `s` unless explicitly overridden.
pub fn set_default_stream(s: Stream) {
    mlx_sys::stream::ffi::set_default_stream(s);
}

/// Return all streams currently registered on this thread (across all devices).
pub fn get_streams() -> Vec<Stream> {
    mlx_sys::stream::ffi::get_streams()
}

/// Destroy all streams created in the current thread. The default stream
/// will be recreated lazily on the next access.
pub fn clear_streams() {
    mlx_sys::stream::ffi::clear_streams();
}
```

- [ ] **Step 4: Wire `mod stream;` in `mlx/src/lib.rs`**

Update the mod/use block. Full version after this task:

```rust
mod array;
mod broadcast;
mod device;
mod dtype;
mod element;
mod error;
pub mod ops;
mod ops_impl;
mod stream;

pub use array::Array;
pub use broadcast::broadcast_shape;
pub use device::{default_device, device_count, is_available, set_default_device, Device, DeviceType};
pub use dtype::Dtype;
pub use element::Element;
pub use error::{Error, Result};
pub use ops::All;
pub use stream::{clear_streams, default_stream, get_streams, new_stream, set_default_stream, Stream};
```

- [ ] **Step 5: Verify**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p2a_stream 2>&1 | grep "test result:"
```
Expected: 6 tests pass.

- [ ] **Step 6: Commit**

```bash
git add mlx/src/stream.rs mlx/src/lib.rs mlx/tests/p2a_stream.rs
git commit -m "feat(p2a): mlx::stream module — Stream + lifecycle (6 tests)"
```

---

## Task 4: `mlx::transforms` module — sync transforms + `blocking` dependency

**Files:**
- Create: `mlx/src/transforms.rs` (sync helpers only — async lands in Task 5)
- Modify: `mlx/Cargo.toml` (add `blocking` dep)
- Modify: `mlx/src/lib.rs`

- [ ] **Step 1: Add `blocking` to `mlx/Cargo.toml`**

In `mlx/Cargo.toml`, add `blocking = "1"` to `[dependencies]`. The full `[dependencies]` section becomes:

```toml
[dependencies]
mlx-sys = { path = "../mlx-sys", version = "0.0.1" }
cxx.workspace = true
thiserror.workspace = true
half = "2"
smallvec = "1"
blocking = "1"
```

- [ ] **Step 2: Verify the workspace still builds with the new dep**

```bash
MLX_DIR=$HOME/.local/mlx cargo check -p mlx 2>&1 | tail -5
```
Expected: PASS (compile clean; `blocking` and its transitive deps download).

- [ ] **Step 3: Create `mlx/src/transforms.rs` (sync transforms only)**

```rust
//! Computation graph transforms (sync + async evaluation).
//!
//! For lazy `Array::eval()` see `mlx/src/array.rs`. This module adds:
//!
//! - [`synchronize`] — block on the current thread's default stream
//! - [`synchronize_stream`] — block on a specific stream
//! - [`async_eval`] — submit + return a runtime-agnostic Future (Task 5)

use crate::{Error, Result, Stream};

/// Block the current thread until all queued work on the current thread's
/// **default stream** completes. To synchronize on a specific stream
/// (regardless of which stream is currently the default), use
/// [`synchronize_stream`].
pub fn synchronize() -> Result<()> {
    mlx_sys::stream::ffi::synchronize().map_err(Error::from)
}

/// Block the current thread until all queued work on the **given stream**
/// completes, regardless of which thread queued the work or which stream
/// is currently the default.
pub fn synchronize_stream(s: Stream) -> Result<()> {
    mlx_sys::stream::ffi::synchronize_stream(s).map_err(Error::from)
}
```

- [ ] **Step 4: Wire `mod transforms;` in `mlx/src/lib.rs`**

Add `mod transforms;` and re-export. After this task:

```rust
mod array;
mod broadcast;
mod device;
mod dtype;
mod element;
mod error;
pub mod ops;
mod ops_impl;
mod stream;
pub mod transforms;

pub use array::Array;
pub use broadcast::broadcast_shape;
pub use device::{default_device, device_count, is_available, set_default_device, Device, DeviceType};
pub use dtype::Dtype;
pub use element::Element;
pub use error::{Error, Result};
pub use ops::All;
pub use stream::{clear_streams, default_stream, get_streams, new_stream, set_default_stream, Stream};
pub use transforms::{synchronize, synchronize_stream};
```

(`async_eval` is added to the re-export in Task 5.)

- [ ] **Step 5: Verify the sync helpers work**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --lib 2>&1 | grep "test result:"
```
Expected: existing tests still pass.

Also run a quick smoke from the workspace:

```bash
MLX_DIR=$HOME/.local/mlx cargo build --workspace 2>&1 | tail -5
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mlx/Cargo.toml mlx/src/transforms.rs mlx/src/lib.rs
git commit -m "feat(p2a): mlx::transforms module + blocking dep (synchronize + synchronize_stream)"
```

---

## Task 5: `async_eval` Future implementation

**Files:**
- Modify: `mlx/src/transforms.rs`
- Modify: `mlx/src/lib.rs`
- (Post-mortem fix below also touches `mlx-sys/shim/{include,src}/cxx_mlx_shim/transforms.{h,cc}` and `mlx-sys/src/bridge/transforms.rs` — see "Correction" note.)

> **CORRECTION (2026-05-04, after Task 6 integration testing)**: the captured-stream design described below was **wrong**. MLX's `get_command_encoder(Stream s)` looks up streams in a `thread_local` map ([mlx/backend/metal/device.cpp:809](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/device.cpp)) — calling `synchronize_stream(captured)` on a `blocking::unblock` worker thread throws `"There is no Stream(gpu, N) in current thread."` because that worker never registered the stream. **All 6 Task 6 tests failed**, not just the multi-thread one.
>
> **Fix shipped** (commit `69762fd`): use per-array `array::wait()` ([mlx/array.cpp:144](https://github.com/ml-explore/mlx/blob/main/mlx/array.cpp)). MLX Events are MTLSharedEvent-backed and waitable from any thread. Sys layer adds `array_wait` FFI; safe layer clones each `&Array` (cheap refcount-share), moves the `Vec<Array>` into the closure, and waits on each array's event in sequence inside `blocking::unblock`. Total wait time is bounded by the slowest array, not the sum.
>
> The implementation block below is preserved as the original spec for historical reference. **The shipped code in `mlx/src/transforms.rs` after `69762fd` is the correct implementation** — see design doc spec A4 (also revised) for the canonical version.

This was originally framed as the most subtle part of P2a. The captured-stream theory was: capture the submission stream at submit time and use `synchronize_stream(captured)` inside the future, so the future can be polled on any thread regardless of which thread's default stream is set there. **In practice this fails** because MLX's per-stream CommandEncoder is thread-local, not just the default-stream pointer. See the correction note above.

- [ ] **Step 1: Append `async_eval` to `mlx/src/transforms.rs`**

```rust
use crate::Array;

/// Asynchronously evaluate one or more arrays.
///
/// Submits the computation graph to MLX's stream worker on the **caller's
/// thread's default stream** (non-blocking, < 1µs), then returns a
/// `Future<Output = Result<()>>` that resolves when the work completes.
///
/// The future is **runtime-agnostic** — `.await` it under tokio,
/// async-std, smol, `futures_lite::future::block_on`, or any executor.
///
/// # Cancellation
///
/// Dropping the returned future without awaiting does **not** cancel the
/// submitted MLX work — MLX has no cancellation primitive. The work runs
/// to completion in the background, consuming GPU time and memory. Any
/// subsequent operation on the same arrays will implicitly synchronize.
///
/// # Implementation note
///
/// The future captures the submission stream at construction time and
/// synchronizes on it explicitly via [`blocking::unblock`]. This works
/// correctly even when the future is polled on a different thread than
/// the submitter (MLX's bare `synchronize()` is thread-local; we use
/// `synchronize_stream(captured)` instead). Scheduling overhead is
/// ~5µs per call from the `blocking` global thread pool, negligible vs
/// typical MLX kernel times (µs–ms).
pub fn async_eval(arrays: &[&Array]) -> impl std::future::Future<Output = Result<()>> + Send + use<> {
    // Capture the submission stream NOW (on the caller's thread, before
    // submission). MLX's async_eval queues work on the caller-thread's
    // default stream; we must wait on that exact stream regardless of
    // which thread polls the returned future.
    let device = mlx_sys::stream::ffi::default_device();
    let captured_stream = mlx_sys::stream::ffi::default_stream(device);

    // Build raw pointer slice + submit (sync, fast).
    let raw: Vec<*const mlx_sys::array::ffi::MlxArray> =
        arrays.iter().map(|a| a.as_inner() as *const _).collect();
    // SAFETY: pointers valid for this fn (we hold &Array refs); MLX
    // async_eval copies arrays internally (refcount-share), so pointers
    // need not outlive THIS function — only the submission.
    let submit_result = unsafe { mlx_sys::stream::ffi::async_eval_many(&raw) };

    // Returned future: synchronize on the captured stream via blocking.
    // Stream is Copy (POD), moves into the closure with no lifetime issues.
    async move {
        submit_result.map_err(Error::from)?;
        blocking::unblock(move || {
            mlx_sys::stream::ffi::synchronize_stream(captured_stream).map_err(Error::from)
        })
        .await
    }
}
```

- [ ] **Step 2: Update re-export in `mlx/src/lib.rs`**

Update the `pub use transforms::...` line to include `async_eval`:

```rust
pub use transforms::{async_eval, synchronize, synchronize_stream};
```

- [ ] **Step 3: Quick sanity build check**

```bash
MLX_DIR=$HOME/.local/mlx cargo build -p mlx 2>&1 | tail -5
```
Expected: PASS.

If `+ use<>` syntax fails (older rustc), check rustc version:
```bash
rustc --version
```
Should be 1.82+ (project minimum is 1.94 per workspace Cargo.toml). If older, fallback to `+ Send + 'static`.

- [ ] **Step 4: Commit (no tests yet — Task 6 covers integration)**

```bash
git add mlx/src/transforms.rs mlx/src/lib.rs
git commit -m "feat(p2a): async_eval Future via blocking crate (captures submission stream for cross-thread correctness)"
```

---

## Task 6: `Array::async_eval` method + p2a_async integration tests

**Files:**
- Modify: `mlx/src/array.rs`
- Modify: `mlx/Cargo.toml` (add tokio as dev-dependency)
- Create: `mlx/tests/p2a_async.rs`

- [ ] **Step 1: Add `tokio` and `futures-lite` to `mlx/Cargo.toml` `[dev-dependencies]`**

In `mlx/Cargo.toml`, the `[dev-dependencies]` section becomes:

```toml
[dev-dependencies]
static_assertions = "1"
tokio = { version = "1", features = ["rt", "macros"] }
futures-lite = "2"
```

(`tokio` and `futures-lite` here are purely test deps — used to verify our Future runs under different executors. The library itself does not depend on either. We pick `futures-lite` over `futures` because it's already transitively brought in by `blocking` and is much smaller; same `block_on` API.)

- [ ] **Step 2: Write failing tests**

Create `mlx/tests/p2a_async.rs`:

```rust
use mlx::{transforms, Array, Dtype};

#[test]
fn async_eval_under_futures_lite() {
    // Verify the Future runs under a minimal executor (no tokio dep).
    let arr = Array::zeros(&[1024], Dtype::Float32).expect("zeros");
    futures_lite::future::block_on(arr.async_eval()).expect("async_eval should complete");
    // After eval, to_vec should not need to re-eval (data is materialized).
    let v: Vec<f32> = arr.to_vec().expect("to_vec");
    assert_eq!(v.len(), 1024);
    assert!(v.iter().all(|x| *x == 0.0));
}

#[test]
fn async_eval_under_tokio_current_thread() {
    // Verify the Future runs under tokio (proves runtime-agnostic).
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("tokio rt");
    rt.block_on(async {
        let arr = Array::zeros(&[256], Dtype::Float32).expect("zeros");
        arr.async_eval().await.expect("async_eval under tokio");
        let v: Vec<f32> = arr.to_vec().expect("to_vec");
        assert_eq!(v.len(), 256);
    });
}

#[test]
fn async_eval_under_tokio_multi_thread() {
    // Multi-threaded runtime: future may be polled on a worker thread
    // different from the submitter. Verifies our captured-stream fix.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("tokio multi-thread rt");
    rt.block_on(async {
        let arr = Array::zeros(&[256], Dtype::Float32).expect("zeros");
        arr.async_eval().await.expect("async_eval under multi-thread tokio");
        let v: Vec<f32> = arr.to_vec().expect("to_vec");
        assert_eq!(v.len(), 256);
    });
}

#[test]
fn async_eval_multiple_arrays() {
    // Submit multiple arrays in one async_eval call.
    let a = Array::zeros(&[64], Dtype::Float32).expect("zeros a");
    let b = Array::zeros(&[64], Dtype::Float32).expect("zeros b");
    futures_lite::future::block_on(transforms::async_eval(&[&a, &b]))
        .expect("async_eval multiple");
    assert_eq!(a.to_vec::<f32>().expect("to_vec a").len(), 64);
    assert_eq!(b.to_vec::<f32>().expect("to_vec b").len(), 64);
}

#[test]
fn synchronize_blocks_until_default_stream_drains() {
    // Submit work, then synchronously block on default stream.
    let arr = Array::zeros(&[128], Dtype::Float32).expect("zeros");
    futures_lite::future::block_on(arr.async_eval()).expect("async_eval");
    transforms::synchronize().expect("synchronize");
    // After explicit sync, the array must be evaluated.
    let v: Vec<f32> = arr.to_vec().expect("to_vec");
    assert_eq!(v.len(), 128);
}

#[test]
fn synchronize_stream_for_explicit_stream() {
    // Get the default stream explicitly and synchronize on it.
    let s = mlx::default_stream(mlx::default_device());
    let arr = Array::zeros(&[32], Dtype::Float32).expect("zeros");
    futures_lite::future::block_on(arr.async_eval()).expect("async_eval");
    transforms::synchronize_stream(s).expect("synchronize_stream");
    let v: Vec<f32> = arr.to_vec().expect("to_vec");
    assert_eq!(v.len(), 32);
}
```

(`futures-lite` is the explicit dev-dep added in Step 1 — it's already transitively brought in by `blocking`, but declaring it explicitly avoids depending on transitive surface.)

- [ ] **Step 3: Verify failure**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p2a_async 2>&1 | tail -15
```
Expected: FAIL with `no method named async_eval found for struct Array` (and possibly `unresolved import futures`).

If `futures-lite` resolution fails, double-check Step 1 Cargo.toml edit. (Removed obsolete `futures = "0.3"` fallback — we use `futures-lite` everywhere in this plan.)

- [ ] **Step 4: Add `Array::async_eval` method in `mlx/src/array.rs`**

In the existing `impl Array { ... }` block, add (suggested location: right next to the existing `eval` method):

```rust
    /// Asynchronously evaluate this array. See [`crate::transforms::async_eval`].
    ///
    /// The returned future does not borrow `self` (the underlying MLX
    /// `async_eval` consumes the array reference at submit time; the future
    /// captures only the owned Stream + submit result).
    pub fn async_eval(&self) -> impl std::future::Future<Output = Result<()>> + Send + use<> {
        crate::transforms::async_eval(&[self])
    }
```

- [ ] **Step 5: Verify all tests pass**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test p2a_async 2>&1 | grep "test result:"
```
Expected: 6 tests pass.

If `async_eval_under_tokio_multi_thread` fails specifically (but the single-thread one works), it's the captured-stream bug surfacing — verify Task 5's implementation captured the stream BEFORE submission.

- [ ] **Step 6: Commit**

```bash
git add mlx/Cargo.toml mlx/src/array.rs mlx/tests/p2a_async.rs
git commit -m "feat(p2a): Array::async_eval method + 6 async integration tests (futures_lite + tokio single + multi-thread)"
```

---

## Task 7: README + final workspace verification

**Files:**
- Modify: `README.md`
- Verify: full workspace test + clippy + doc

- [ ] **Step 1: Update README status line**

In `README.md`, change the existing P1-complete status line to:

```markdown
**Status:** 🚧 **P2a complete** — Stream / Device foundation. Adds `Device::cpu()`/`gpu()`, stream lifecycle, runtime-agnostic `async_eval` returning `impl Future` (works under tokio / async-std / smol / `futures_lite::future::block_on` / any executor), and explicit `synchronize` / `synchronize_stream`. Built on P1 inference primitives. Next: P2b (`fast` ops: rms_norm/layer_norm/rope/sdpa) and P2c (`io`: safetensors/gguf load).
```

- [ ] **Step 2: Add a "Streams & Devices" section to `README.md`**

Add before the "Threading" section:

````markdown
## Streams & Devices

`Device::cpu()` / `Device::gpu(index)` are the supported devices on Apple
Silicon. Streams are MLX's execution queues — work on different streams may
run concurrently. The default stream of the default device is used unless
explicitly overridden:

```rust
use mlx::{Array, Device, Dtype};

fn main() -> mlx::Result<()> {
    println!("default device: {:?}", mlx::default_device());
    println!("gpu count: {}", mlx::device_count(mlx::DeviceType::Gpu));

    let _arr = Array::zeros(&[2, 3], Dtype::Float32)?;

    // Optional: switch streams (thread-local).
    let s = mlx::new_stream(Device::gpu(0))?;
    mlx::set_default_stream(s);

    Ok(())
}
```

### Async evaluation

`async_eval` returns a runtime-agnostic `Future`. It works under any
executor — tokio, async-std, smol, or `futures_lite::future::block_on`:

```rust
use mlx::{Array, Dtype};

# #[tokio::main]
# async fn main() -> mlx::Result<()> {
let a = Array::zeros(&[1024], Dtype::Float32)?;
let b = Array::zeros(&[1024], Dtype::Float32)?;

// Submit one or many arrays; await when ready.
mlx::async_eval(&[&a, &b]).await?;

// Or single-array convenience method:
let c = Array::zeros(&[256], Dtype::Float32)?;
c.async_eval().await?;
# Ok(())
# }
```

**Cancellation note**: dropping a future without awaiting does NOT cancel
the submitted MLX work — MLX has no cancellation primitive. The work runs
to completion in the background. Subsequent ops on the same arrays will
implicitly synchronize.

For sync contexts (no executor), use `mlx::synchronize()` (default stream)
or `mlx::synchronize_stream(s)` (explicit stream) to block.
````

- [ ] **Step 3: Update the Roadmap**

Change:

```markdown
- 🎉 **P1 complete** — full inference primitives ready
```

to:

```markdown
- 🎉 **P1 complete** — full inference primitives ready
- ✅ **P2a** — Stream / Device foundation + runtime-agnostic async_eval
- ⏳ **P2b** — `fast` ops (rms_norm / layer_norm / rope / sdpa)
- ⏳ **P2c** — `io` (safetensors / gguf load)
```

- [ ] **Step 4: Run the full workspace test suite**

```bash
MLX_DIR=$HOME/.local/mlx cargo test --workspace 2>&1 | grep "test result:" | head -20
```
Expected: ≥ 159 tests passing (140 P1-complete + 7 device + 6 stream + 6 async = 159).

- [ ] **Step 5: Run clippy**

```bash
MLX_DIR=$HOME/.local/mlx cargo clippy --workspace --all-targets -- -D warnings 2>&1 | grep -v "^warning: mlx-sys@" | tail -10
```
Expected: clean (only upstream MLX header noise filtered out).

If clippy flags the `unsafe fn async_eval_many` doc:
- The bridge file already has `#[allow(clippy::missing_safety_doc)]` from Task 1 Step 3.
- If it doesn't, add it as Task 1 instructed.

- [ ] **Step 6: Build docs**

```bash
MLX_DIR=$HOME/.local/mlx cargo doc -p mlx --no-deps 2>&1 | grep -i "warning" | head -5
echo "---"
MLX_DIR=$HOME/.local/mlx cargo doc -p mlx --no-deps 2>&1 | tail -3
```
Expected: clean with `Finished` (no rustdoc warnings; only the cargo-emitted upstream MLX C++ warnings).

- [ ] **Step 7: Commit**

```bash
git add README.md
git commit -m "docs(p2a): Streams & Devices section + roadmap (P2a complete; P2b/P2c next)"
```

---

## Acceptance Criteria

P2a is complete when:

1. `cargo test --workspace` reports ≥ 159 tests passing (140 P1-complete + 19 P2a)
2. `cargo clippy --workspace --all-targets -- -D warnings` is clean
3. `cargo doc -p mlx --no-deps` builds with no rustdoc warnings
4. `mlx::Device::cpu()` / `Device::gpu(index)` and 4 device top-level fns work
5. `mlx::Stream` re-export and 5 stream lifecycle fns work
6. `mlx::async_eval(&[&Array]) -> impl Future` is runtime-agnostic — verified passing under both `futures_lite::future::block_on` AND `tokio::runtime` (single + multi-thread)
7. `mlx::synchronize()` and `mlx::synchronize_stream(Stream)` work synchronously
8. The async_eval future correctly captures the submission stream — multi-thread tokio test passes (would fail if the captured-stream bug regressed)
9. `Array::async_eval()` method delegates correctly
10. README documents the API with both sync and async examples

When all 10 hold, P2a is ready for fast-forward to master and P2b brainstorm starts.
