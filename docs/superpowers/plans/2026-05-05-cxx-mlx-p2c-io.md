# cxx-mlx P2c · IO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 MLX IO 子系统（safetensors / gguf / npy 三种格式 × {load, save} × {file path, in-memory stream}）提供完整 Rust 安全绑定，含 B-lite 流接口（`Reader::open_file`/`from_bytes` + `Writer::create_file`/`memory`/`into_bytes`）。

**Architecture:** 三层结构：shim C++ 适配层（opaque LoadResult/SaveBuilder + MlxReader/MlxWriter wrapper + 自定义 MemoryReader/MemoryWriter）→ cxx::bridge 声明 ABI（free functions on opaque types，遵循 P2a/P2b 约定）→ 安全 Rust API（`HashMap<String, Array>` + `enum GGUFMetaData` + `Reader`/`Writer` 类型）。`unordered_map<string, array>` 用 names + take_values 双 getter 拼接成 HashMap；`std::variant<...>` 按子类型拆 3 组 getter；`shared_ptr<abstract base>` 用 wrapper class 持有，Rust 端 `UniquePtr<MlxReader/Writer>`。

**Tech Stack:** Rust 1.82+ (`Pin<&mut T>` for cxx opaque-type !Unpin 引用)，cxx 1.0（含 free function on opaque type、`UniquePtr<T>` 按值传递、`rust::Slice<T>`/`rust::Vec<T>`/`rust::String` 桥接），MLX C++ 共享安装，`tempfile` crate（dev-dep）做集成测试，cargo nightly fmt + clippy + release build。

**Spec reference:** `docs/superpowers/specs/2026-05-05-cxx-mlx-p2c-io-design.md`

---

## 关键背景信息（实施者必读）

### 项目三层结构

- **shim 层**：`mlx-sys/shim/include/cxx_mlx_shim/*.h` + `mlx-sys/shim/src/*.cc` —— 手写 C++，把 cxx 不可表达的 MLX 类型抹平为 cxx 友好的 free function + opaque class
- **桥接层**：`mlx-sys/src/bridge/*.rs` —— `#[cxx::bridge]` 声明 ABI；项目惯例是 **free function**（不用 `self: &T` 方法语法）
- **安全层**：`mlx/src/*.rs` —— 顶层 `mlx::*` re-export

### cxx 类型映射

| MLX C++ 类型 | shim 暴露 | cxx bridge 类型 | Rust 端调用 |
|--------------|-----------|-----------------|-------------|
| `std::string` 入参 | `rust::Str` | `&str` | 直接传 `&str` |
| `std::string` 出参 | `rust::String` | `String`（在 `Vec<String>` 内） | `Vec<String>` |
| `std::vector<uint8_t>` 出参 | `rust::Vec<uint8_t>` | `Vec<u8>` | `Vec<u8>` |
| `std::vector<std::unique_ptr<T>>` 出参 | `rust::Vec<std::unique_ptr<T>>` | `Vec<UniquePtr<T>>` | `Vec<UniquePtr<T>>` |
| 抽象 / 不可拷贝类 | opaque class | bridge `type Foo;` | `cxx::UniquePtr<Foo>` |
| 抽象类的常量方法 | shim free function `foo(const Foo& self, ...)` | `fn foo(self: &Foo, ...)` 或 `fn foo(arg: &Foo, ...)` | `&*unique_ptr_var` 或 `&unique_ptr_var`（deref coercion） |
| 抽象类的状态变更方法 | shim free function `foo(Foo& self, ...)` | `fn foo(self: Pin<&mut Foo>, ...)` 或 `fn foo(arg: Pin<&mut Foo>, ...)` | `unique_ptr_var.pin_mut()` |
| 按值消费 `unique_ptr<Foo>` | shim 形参为 `std::unique_ptr<Foo>` | `fn foo(arg: UniquePtr<Foo>) -> ...` | `unique_ptr_var`（move） |

**项目惯例**：所有 bridge function 用 **free function 形态**（参数加 `arg: &T` / `arg: Pin<&mut T>` / `arg: UniquePtr<T>`），不用 `self: &T` 方法语法。

### 已有 API 引用点

- `Array::from_inner(inner: cxx::UniquePtr<...>) -> Self`（[mlx/src/array.rs:11](mlx/src/array.rs#L11)）
- `Array::as_inner(&self) -> &mlx_sys::array::ffi::MlxArray`（[mlx/src/array.rs:139](mlx/src/array.rs#L139)；通过 deref coercion 把 `&UniquePtr<T>` 转 `&T`）
- 跨桥接共享 `MlxArray`：`type MlxArray = crate::bridge::array::ffi::MlxArray;`（参考 [mlx-sys/src/bridge/transforms.rs:10](mlx-sys/src/bridge/transforms.rs#L10)）

### 强制检查（CLAUDE.md + 项目约定，每次 commit 前必跑）

```bash
export MLX_DIR=/Users/sam/.local/mlx
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app --tests -- -D warnings
cargo build --release
```

`--tests` 检查是 P2b 期间确立的额外约定（避免 test 中 `identity_op` 等漏网）。

### MLX 上游 API（来自 `${MLX_DIR}/include/mlx/io.h` 与 `mlx/io/load.h`）

```cpp
// io.h
namespace mlx::core {

using GGUFMetaData =
    std::variant<std::monostate, array, std::string, std::vector<std::string>>;
using GGUFLoad = std::pair<
    std::unordered_map<std::string, array>,
    std::unordered_map<std::string, GGUFMetaData>>;
using SafetensorsLoad = std::pair<
    std::unordered_map<std::string, array>,
    std::unordered_map<std::string, std::string>>;

// npy
void save(std::shared_ptr<io::Writer>, array);
void save(std::string, array);
array load(std::shared_ptr<io::Reader>, StreamOrDevice s = {});
array load(std::string, StreamOrDevice s = {});

// safetensors
SafetensorsLoad load_safetensors(std::shared_ptr<io::Reader>, StreamOrDevice s = {});
SafetensorsLoad load_safetensors(const std::string&, StreamOrDevice s = {});
void save_safetensors(std::shared_ptr<io::Writer>,
    std::unordered_map<std::string, array>,
    std::unordered_map<std::string, std::string> meta = {});
void save_safetensors(std::string,
    std::unordered_map<std::string, array>,
    std::unordered_map<std::string, std::string> meta = {});

// gguf (file path only)
GGUFLoad load_gguf(const std::string&, StreamOrDevice s = {});
void save_gguf(std::string,
    std::unordered_map<std::string, array>,
    std::unordered_map<std::string, GGUFMetaData> meta = {});

}  // namespace mlx::core

// mlx/io/load.h: io::Reader / io::Writer 抽象基类（pure virtual）
//   + io::ParallelFileReader（具体类，文件路径）
//   + io::FileWriter（具体类，文件路径）
```

`StreamOrDevice s = {}` 即 MLX 默认 stream，shim 全部不传该参数。

---

## 文件清单

### 新建
- `mlx-sys/shim/include/cxx_mlx_shim/io.h` — shim 头（约 90 行）
- `mlx-sys/shim/src/io.cc` — shim 实现 + Memory{Reader,Writer} + LoadResult/SaveBuilder（约 200 行）
- `mlx-sys/src/bridge/io.rs` — cxx 桥接（约 30 个 free FFI）
- `mlx/src/io.rs` — 安全 API（14 公开函数 + Reader/Writer/GGUFMetaData 类型）
- `mlx/tests/p2c_io.rs` — 集成测试（约 15+ 测试）

### 修改
- `mlx-sys/build.rs` — `cxx_build::bridges` 加 `"src/bridge/io.rs"`，`.file()` 加 `"shim/src/io.cc"`
- `mlx-sys/src/bridge/mod.rs` — 加 `pub mod io;`
- `mlx-sys/src/lib.rs` — 加 `pub use bridge::io;`
- `mlx/src/lib.rs` — 加 `pub mod io;` + 顶层 re-exports（在 Task 5）
- `mlx/Cargo.toml` — dev-dep 加 `tempfile = "3"`
- `README.md` — 进度更新（在 Task 5）

---

## Task 1: 框架搭建 + Reader/Writer 基础设施

**目的**：打通 build.rs / mod.rs / lib.rs 的全部接线，定义 `MlxReader`/`MlxWriter` opaque wrapper 类、自定义 `MemoryReader`/`MemoryWriter` 实现、5 个工厂/提取函数（`open_file_reader`、`open_memory_reader`、`create_file_writer`、`create_memory_writer`、`writer_into_bytes`），三层链路打通。

**Files:**
- Create: `mlx-sys/shim/include/cxx_mlx_shim/io.h`
- Create: `mlx-sys/shim/src/io.cc`
- Create: `mlx-sys/src/bridge/io.rs`
- Create: `mlx/src/io.rs`
- Create: `mlx/tests/p2c_io.rs`
- Modify: `mlx-sys/build.rs`
- Modify: `mlx-sys/src/bridge/mod.rs`
- Modify: `mlx-sys/src/lib.rs`
- Modify: `mlx/src/lib.rs`
- Modify: `mlx/Cargo.toml`

- [ ] **Step 1.1: 加 dev-dep `tempfile = "3"` 到 `mlx/Cargo.toml`**

打开 `mlx/Cargo.toml`，在 `[dev-dependencies]` section 中追加 `tempfile = "3"`：

```toml
[dev-dependencies]
static_assertions = "1"
tokio = { version = "1", features = ["rt", "rt-multi-thread", "macros"] }
futures-lite = "2"
tempfile = "3"
```

- [ ] **Step 1.2: 写失败的集成测试（基础设施）**

将以下完整内容写入 `mlx/tests/p2c_io.rs`：

```rust
//! Integration tests for mlx::io — file and stream IO for MLX arrays.

use mlx::io::{Reader, Writer};

#[test]
fn reader_open_file_nonexistent_returns_err() {
    let result = Reader::open_file("/nonexistent/path/should-not-exist.safetensors");
    assert!(result.is_err());
}

#[test]
fn reader_from_bytes_constructs_ok() {
    // 空数据可构造（数据有效性由后续 load 验证）
    let _r = Reader::from_bytes(&[]);
    let _r2 = Reader::from_bytes(&[1, 2, 3]);
}

#[test]
fn writer_memory_into_bytes_empty() {
    let writer = Writer::memory();
    let bytes = writer.into_bytes().expect("memory writer into_bytes");
    assert_eq!(bytes, Vec::<u8>::new());
}

#[test]
fn writer_create_file_invalid_path_returns_err() {
    let result = Writer::create_file("/nonexistent_dir_xyz/should-fail.bin");
    assert!(result.is_err());
}

#[test]
fn writer_file_into_bytes_returns_err() {
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    let writer = Writer::create_file(tmp.path().to_str().unwrap()).expect("create_file");
    let result = writer.into_bytes();
    assert!(result.is_err(), "into_bytes on file writer should err");
}
```

- [ ] **Step 1.3: 运行测试，确认失败（编译错误）**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p2c_io --no-run`
Expected: 编译失败，错误提到 `mlx::io` 不存在。

- [ ] **Step 1.4: 创建 shim 头 `io.h`**

将以下完整内容写入 `mlx-sys/shim/include/cxx_mlx_shim/io.h`：

```cpp
#pragma once

#include <cstdint>
#include <cstring>
#include <ios>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "mlx/io.h"
#include "mlx/io/load.h"
#include "rust/cxx.h"

namespace cxx_mlx {

using MlxArray = mlx::core::array;

// ===== Reader / Writer wrappers (opaque to cxx) =====
// 持有 shared_ptr<io::Reader/Writer>。MLX 的 load_*/save_*(stream) 需要 shared_ptr。
class MlxReader {
 public:
  explicit MlxReader(std::shared_ptr<mlx::core::io::Reader> r) : ptr(std::move(r)) {}
  std::shared_ptr<mlx::core::io::Reader> ptr;
};

class MlxWriter {
 public:
  explicit MlxWriter(std::shared_ptr<mlx::core::io::Writer> w) : ptr(std::move(w)) {}
  std::shared_ptr<mlx::core::io::Writer> ptr;
};

// ===== Memory Reader / Writer (cxx_mlx 自定义) =====
// MLX 没有提供内存版 Reader/Writer，本 shim 自实现以支持 in-memory IO。

class MemoryReader : public mlx::core::io::Reader {
 public:
  explicit MemoryReader(std::vector<uint8_t> bytes);

  bool is_open() const override { return true; }
  bool good() const override { return true; }
  size_t tell() override { return pos_; }
  void seek(int64_t off, std::ios_base::seekdir way = std::ios_base::beg) override;
  void read(char* data, size_t n) override;
  void read(char* data, size_t n, size_t offset) override;
  std::string label() const override { return "memory"; }

 private:
  std::vector<uint8_t> data_;
  size_t pos_ = 0;
};

class MemoryWriter : public mlx::core::io::Writer {
 public:
  MemoryWriter() = default;

  bool is_open() const override { return true; }
  bool good() const override { return true; }
  size_t tell() override { return pos_; }
  void seek(int64_t off, std::ios_base::seekdir way = std::ios_base::beg) override;
  void write(const char* data, size_t n) override;
  std::string label() const override { return "memory"; }

  std::vector<uint8_t> take_bytes() && { return std::move(data_); }

 private:
  std::vector<uint8_t> data_;
  size_t pos_ = 0;
};

// ===== Reader / Writer 工厂 =====
std::unique_ptr<MlxReader> open_file_reader(rust::Str path);
std::unique_ptr<MlxReader> open_memory_reader(rust::Slice<const uint8_t> data);
std::unique_ptr<MlxWriter> create_file_writer(rust::Str path);
std::unique_ptr<MlxWriter> create_memory_writer();
// 仅 MemoryWriter 合法；FileWriter 抛 runtime_error。消费 writer 语义。
rust::Vec<uint8_t> writer_into_bytes(std::unique_ptr<MlxWriter> writer);

}  // namespace cxx_mlx
```

- [ ] **Step 1.5: 创建 shim 实现 `io.cc`**

将以下完整内容写入 `mlx-sys/shim/src/io.cc`：

```cpp
#include "cxx_mlx_shim/io.h"

#include <stdexcept>

namespace cxx_mlx {

// ===== MemoryReader =====

MemoryReader::MemoryReader(std::vector<uint8_t> bytes) : data_(std::move(bytes)) {}

void MemoryReader::seek(int64_t off, std::ios_base::seekdir way) {
  if (way == std::ios_base::beg) {
    pos_ = static_cast<size_t>(off);
  } else if (way == std::ios_base::cur) {
    pos_ = static_cast<size_t>(static_cast<int64_t>(pos_) + off);
  } else if (way == std::ios_base::end) {
    pos_ = static_cast<size_t>(static_cast<int64_t>(data_.size()) + off);
  } else {
    throw std::runtime_error("MemoryReader::seek: invalid seekdir");
  }
}

void MemoryReader::read(char* dst, size_t n) {
  if (pos_ + n > data_.size()) {
    throw std::runtime_error("MemoryReader::read past end");
  }
  std::memcpy(dst, data_.data() + pos_, n);
  pos_ += n;
}

void MemoryReader::read(char* dst, size_t n, size_t offset) {
  if (offset + n > data_.size()) {
    throw std::runtime_error("MemoryReader::read(offset) past end");
  }
  std::memcpy(dst, data_.data() + offset, n);
}

// ===== MemoryWriter =====
// 注意：MLX 的 save_safetensors 会 seek 回头部回填 metadata 偏移，所以
// 必须实现真正的随机写入：tell 返回 pos_，seek 调整 pos_，write 在 pos_ 处
// 覆盖（必要时 resize 扩容）+ 推进 pos_。

void MemoryWriter::seek(int64_t off, std::ios_base::seekdir way) {
  if (way == std::ios_base::beg) {
    pos_ = static_cast<size_t>(off);
  } else if (way == std::ios_base::cur) {
    pos_ = static_cast<size_t>(static_cast<int64_t>(pos_) + off);
  } else if (way == std::ios_base::end) {
    pos_ = static_cast<size_t>(static_cast<int64_t>(data_.size()) + off);
  } else {
    throw std::runtime_error("MemoryWriter::seek: invalid seekdir");
  }
}

void MemoryWriter::write(const char* src, size_t n) {
  if (pos_ + n > data_.size()) {
    data_.resize(pos_ + n);
  }
  std::memcpy(data_.data() + pos_, src, n);
  pos_ += n;
}

// ===== 工厂 =====

std::unique_ptr<MlxReader> open_file_reader(rust::Str path) {
  auto r = std::make_shared<mlx::core::io::ParallelFileReader>(std::string(path));
  if (!r->is_open()) {
    throw std::runtime_error("failed to open file: " + std::string(path));
  }
  return std::make_unique<MlxReader>(std::move(r));
}

std::unique_ptr<MlxReader> open_memory_reader(rust::Slice<const uint8_t> data) {
  auto r = std::make_shared<MemoryReader>(
      std::vector<uint8_t>(data.begin(), data.end()));
  return std::make_unique<MlxReader>(std::move(r));
}

std::unique_ptr<MlxWriter> create_file_writer(rust::Str path) {
  auto w = std::make_shared<mlx::core::io::FileWriter>(std::string(path));
  if (!w->is_open()) {
    throw std::runtime_error("failed to create file: " + std::string(path));
  }
  return std::make_unique<MlxWriter>(std::move(w));
}

std::unique_ptr<MlxWriter> create_memory_writer() {
  return std::make_unique<MlxWriter>(std::make_shared<MemoryWriter>());
}

rust::Vec<uint8_t> writer_into_bytes(std::unique_ptr<MlxWriter> w) {
  auto* mw = dynamic_cast<MemoryWriter*>(w->ptr.get());
  if (!mw) {
    throw std::runtime_error("writer is not a memory writer");
  }
  std::vector<uint8_t> bytes = std::move(*mw).take_bytes();
  rust::Vec<uint8_t> out;
  out.reserve(bytes.size());
  for (uint8_t b : bytes) {
    out.push_back(b);
  }
  return out;
}

}  // namespace cxx_mlx
```

- [ ] **Step 1.6: 创建桥接 `bridge/io.rs`**

将以下完整内容写入 `mlx-sys/src/bridge/io.rs`：

```rust
//! Bridge for MLX IO (load/save: safetensors, gguf, npy + Reader/Writer streams).
//!
//! Map decomposition: shim returns opaque LoadResult types, Rust calls
//! parallel name/value getters and rebuilds HashMap on the safe layer.
//!
//! Save direction: opaque SaveBuilder accumulates entries via add_* calls;
//! single save_*_file/writer call commits.
//!
//! Reader/Writer: opaque MlxReader/MlxWriter wrap shared_ptr<io::Reader/Writer>.
//! B-lite = file + memory backends only; no Rust trait callbacks.

#[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
#[cxx::bridge(namespace = "cxx_mlx")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cxx_mlx_shim/io.h");

        type MlxArray = crate::bridge::array::ffi::MlxArray;
        type MlxReader;
        type MlxWriter;

        // ===== Reader / Writer 工厂 =====
        fn open_file_reader(path: &str) -> Result<UniquePtr<MlxReader>>;
        fn open_memory_reader(data: &[u8]) -> UniquePtr<MlxReader>;
        fn create_file_writer(path: &str) -> Result<UniquePtr<MlxWriter>>;
        fn create_memory_writer() -> UniquePtr<MlxWriter>;
        fn writer_into_bytes(writer: UniquePtr<MlxWriter>) -> Result<Vec<u8>>;
    }
}
```

- [ ] **Step 1.7: 在 `mlx-sys/src/bridge/mod.rs` 末尾增加 `pub mod io;`**

打开 `mlx-sys/src/bridge/mod.rs`，在 `pub mod fast;` 之后增加 `pub mod io;`（保留前面的 module-level 注释不动）。

- [ ] **Step 1.8: 在 `mlx-sys/src/lib.rs` 增加 re-export**

打开 `mlx-sys/src/lib.rs`，在 `pub use bridge::fast;` 之后追加 `pub use bridge::io;`。结果：

```rust
pub use bridge::array;
pub use bridge::fast;
pub use bridge::io;
pub use bridge::stream;
pub use bridge::transforms;
```

（注意：cargo fmt 会按字母排序，提交时以 fmt 后的为准。）

- [ ] **Step 1.9: 在 `mlx-sys/build.rs` 注册 io 桥接 + shim cc**

打开 `mlx-sys/build.rs`，找到 `cxx_build::bridges([...])` 调用块，把它替换为：

```rust
    cxx_build::bridges([
        "src/bridge/array.rs",
        "src/bridge/transforms.rs",
        "src/bridge/stream.rs",
        "src/bridge/fast.rs",
        "src/bridge/io.rs",
    ])
    .file("shim/src/array.cc")
    .file("shim/src/transforms.cc")
    .file("shim/src/stream.cc")
    .file("shim/src/fast.cc")
    .file("shim/src/io.cc")
    .include("shim/include")
    .include(&include_dir)
    .std("c++20")
    .flag_if_supported("-fvisibility=hidden")
    .compile("cxx_mlx_shim");
```

（仅在桥接列表与 `.file()` 列表的末尾追加 io 项）

- [ ] **Step 1.10: 创建安全 API `mlx/src/io.rs`**

将以下完整内容写入 `mlx/src/io.rs`：

```rust
//! File and stream IO for MLX arrays.
//!
//! - safetensors: tensor + string metadata; file path or Reader/Writer
//! - gguf: tensor + variant metadata; file path only (upstream limitation)
//! - npy: single array; file path or Reader/Writer
//!
//! Reader / Writer are opaque handles wrapping MLX io::Reader/Writer.
//! Backends: file path + in-memory (B-lite). No Rust-implemented IO callbacks.

use std::pin::Pin;

use crate::{Error, Result};

/// Opaque IO reader handle. Backed by file (`open_file`) or memory (`from_bytes`).
pub struct Reader(cxx::UniquePtr<mlx_sys::io::ffi::MlxReader>);

/// Opaque IO writer handle. Backed by file (`create_file`) or memory (`memory`).
/// Memory writers can be drained via [`Writer::into_bytes`].
pub struct Writer(cxx::UniquePtr<mlx_sys::io::ffi::MlxWriter>);

impl Reader {
    /// Open a file for reading (uses MLX's parallel file reader internally).
    pub fn open_file(path: &str) -> Result<Self> {
        let inner = mlx_sys::io::ffi::open_file_reader(path).map_err(Error::from)?;
        Ok(Reader(inner))
    }

    /// Construct an in-memory reader from a byte slice (data is copied).
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Reader(mlx_sys::io::ffi::open_memory_reader(bytes))
    }

    #[allow(dead_code)]  // Will be used by load_*_from_reader in Tasks 2/4
    pub(crate) fn pin_mut(&mut self) -> Pin<&mut mlx_sys::io::ffi::MlxReader> {
        self.0.pin_mut()
    }
}

impl Writer {
    /// Open a file for writing (truncates if exists).
    pub fn create_file(path: &str) -> Result<Self> {
        let inner = mlx_sys::io::ffi::create_file_writer(path).map_err(Error::from)?;
        Ok(Writer(inner))
    }

    /// Construct an in-memory writer. Drain via [`Writer::into_bytes`] after writes.
    pub fn memory() -> Self {
        Writer(mlx_sys::io::ffi::create_memory_writer())
    }

    /// Drain the in-memory buffer. Returns `Err` if this is a file writer.
    /// Consumes the writer (memory buffer is moved out).
    pub fn into_bytes(self) -> Result<Vec<u8>> {
        mlx_sys::io::ffi::writer_into_bytes(self.0).map_err(Error::from)
    }

    #[allow(dead_code)]  // Will be used by save_*_to_writer in Tasks 2/4
    pub(crate) fn pin_mut(&mut self) -> Pin<&mut mlx_sys::io::ffi::MlxWriter> {
        self.0.pin_mut()
    }
}
```

- [ ] **Step 1.11: 在 `mlx/src/lib.rs` 增加 `pub mod io;`**

打开 `mlx/src/lib.rs`，在 `pub mod fast;` 那一行之后添加 `pub mod io;`：

```rust
pub mod fast;
pub mod io;
```

（顶层 re-export 在 Task 5 统一加。）

- [ ] **Step 1.12: 编译并运行测试**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p2c_io`
Expected: 5 tests passed（`reader_open_file_nonexistent_returns_err`、`reader_from_bytes_constructs_ok`、`writer_memory_into_bytes_empty`、`writer_create_file_invalid_path_returns_err`、`writer_file_into_bytes_returns_err`）。

- [ ] **Step 1.13: 跑全套 Rust 检查**

```bash
export MLX_DIR=/Users/sam/.local/mlx
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app --tests -- -D warnings
cargo build --release
cargo test --workspace --all-features
```
Expected: 全部通过，无 warning，所有测试 PASS。

- [ ] **Step 1.14: 提交**

```bash
git add mlx-sys/shim/include/cxx_mlx_shim/io.h \
        mlx-sys/shim/src/io.cc \
        mlx-sys/src/bridge/io.rs \
        mlx-sys/src/bridge/mod.rs \
        mlx-sys/src/lib.rs \
        mlx-sys/build.rs \
        mlx/src/io.rs \
        mlx/src/lib.rs \
        mlx/tests/p2c_io.rs \
        mlx/Cargo.toml
git commit -m "feat(p2c): scaffold io module + Reader/Writer infrastructure (3 layers, 5 tests)"
```

---

## Task 2: safetensors load + save（file + stream）

**目的**：在已有 io 框架上追加 safetensors 完整 IO，含 `SafetensorsLoadResult` opaque 类（4 getter）+ `SafetensorsSaveBuilder` opaque 类（2 add 方法）+ 4 个 top-level 函数（load/save × file/stream）。

**Files (all modifications):**
- Modify: `mlx-sys/shim/include/cxx_mlx_shim/io.h`
- Modify: `mlx-sys/shim/src/io.cc`
- Modify: `mlx-sys/src/bridge/io.rs`
- Modify: `mlx/src/io.rs`
- Modify: `mlx/tests/p2c_io.rs`

- [ ] **Step 2.1: 写失败的集成测试**

在 `mlx/tests/p2c_io.rs` 末尾追加：

```rust
use std::collections::HashMap;
use mlx::Array;
use mlx::io;

fn make_test_tensors() -> HashMap<String, Array> {
    let mut tensors = HashMap::new();
    tensors.insert(
        "alpha".to_string(),
        Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]).expect("alpha"),
    );
    tensors.insert(
        "beta".to_string(),
        Array::from_slice(&[10.0_f32, 20.0], &[2]).expect("beta"),
    );
    tensors
}

fn make_test_metadata() -> HashMap<String, String> {
    let mut meta = HashMap::new();
    meta.insert("model".to_string(), "test-model".to_string());
    meta.insert("version".to_string(), "1.0".to_string());
    meta
}

#[test]
fn safetensors_round_trip_file() {
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    let path = tmp.path().to_str().unwrap();
    let tensors = make_test_tensors();
    let metadata = make_test_metadata();

    io::save_safetensors(path, &tensors, &metadata).expect("save");

    let (loaded_tensors, loaded_meta) = io::load_safetensors(path).expect("load");
    assert_eq!(loaded_tensors.len(), tensors.len());
    assert_eq!(loaded_meta, metadata);

    // 数值一致
    let alpha_in: Vec<f32> = tensors["alpha"].to_vec().expect("alpha to_vec");
    let alpha_out: Vec<f32> = loaded_tensors["alpha"].to_vec().expect("alpha out");
    assert_eq!(alpha_in, alpha_out);
    let beta_in: Vec<f32> = tensors["beta"].to_vec().expect("beta to_vec");
    let beta_out: Vec<f32> = loaded_tensors["beta"].to_vec().expect("beta out");
    assert_eq!(beta_in, beta_out);
}

#[test]
fn safetensors_round_trip_memory() {
    let tensors = make_test_tensors();
    let metadata = make_test_metadata();

    let mut writer = io::Writer::memory();
    io::save_safetensors_to_writer(&mut writer, &tensors, &metadata).expect("save to writer");
    let bytes = writer.into_bytes().expect("into_bytes");
    assert!(!bytes.is_empty(), "memory writer should have written bytes");

    let mut reader = io::Reader::from_bytes(&bytes);
    let (loaded_tensors, loaded_meta) =
        io::load_safetensors_from_reader(&mut reader).expect("load from reader");
    assert_eq!(loaded_tensors.len(), tensors.len());
    assert_eq!(loaded_meta, metadata);

    let alpha_in: Vec<f32> = tensors["alpha"].to_vec().expect("alpha");
    let alpha_out: Vec<f32> = loaded_tensors["alpha"].to_vec().expect("alpha out");
    assert_eq!(alpha_in, alpha_out);
}

#[test]
fn safetensors_load_nonexistent_file_returns_err() {
    let result = io::load_safetensors("/nonexistent/path/foo.safetensors");
    assert!(result.is_err());
}

#[test]
fn safetensors_empty_metadata_round_trip() {
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    let path = tmp.path().to_str().unwrap();
    let tensors = make_test_tensors();
    let metadata: HashMap<String, String> = HashMap::new();

    io::save_safetensors(path, &tensors, &metadata).expect("save");
    let (loaded_tensors, loaded_meta) = io::load_safetensors(path).expect("load");
    assert_eq!(loaded_tensors.len(), tensors.len());
    assert!(loaded_meta.is_empty());
}
```

- [ ] **Step 2.2: 运行测试，确认失败**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p2c_io --no-run`
Expected: 编译失败，提示 `io::save_safetensors`、`io::load_safetensors`、`io::save_safetensors_to_writer`、`io::load_safetensors_from_reader` 不存在。

- [ ] **Step 2.3: shim 头追加 SafetensorsLoadResult/SafetensorsSaveBuilder + 4 顶层函数声明**

在 `mlx-sys/shim/include/cxx_mlx_shim/io.h` 末尾的 `}  // namespace cxx_mlx` 之前追加：

```cpp
// ===== SafetensorsLoadResult (opaque) =====

class SafetensorsLoadResult {
 public:
  explicit SafetensorsLoadResult(mlx::core::SafetensorsLoad data)
      : inner_(std::move(data)) {}
  mlx::core::SafetensorsLoad inner_;
};

// 注：take_tensor_values 单次性消费；调后再调 tensor_names 仍可（key 还在），
// 但不要再调 take_tensor_values（array 已 move-from）。Rust 安全层只调一次。
rust::Vec<rust::String> safetensors_tensor_names(const SafetensorsLoadResult& r);
rust::Vec<std::unique_ptr<MlxArray>> safetensors_take_tensor_values(
    SafetensorsLoadResult& r);
rust::Vec<rust::String> safetensors_metadata_names(const SafetensorsLoadResult& r);
rust::Vec<rust::String> safetensors_metadata_values(const SafetensorsLoadResult& r);

// ===== SafetensorsSaveBuilder (opaque) =====

class SafetensorsSaveBuilder {
 public:
  std::unordered_map<std::string, mlx::core::array> tensors;
  std::unordered_map<std::string, std::string> metadata;
};

std::unique_ptr<SafetensorsSaveBuilder> new_safetensors_save_builder();
void safetensors_builder_add_tensor(
    SafetensorsSaveBuilder& b, rust::Str name, const MlxArray& array);
void safetensors_builder_add_metadata(
    SafetensorsSaveBuilder& b, rust::Str key, rust::Str value);

// ===== 顶层 load/save APIs =====

std::unique_ptr<SafetensorsLoadResult> load_safetensors_file(rust::Str path);
std::unique_ptr<SafetensorsLoadResult> load_safetensors_reader(MlxReader& reader);
void save_safetensors_file(rust::Str path, const SafetensorsSaveBuilder& builder);
void save_safetensors_writer(MlxWriter& writer, const SafetensorsSaveBuilder& builder);
```

- [ ] **Step 2.4: shim cc 追加 SafetensorsLoadResult/SafetensorsSaveBuilder 实现 + 顶层函数**

在 `mlx-sys/shim/src/io.cc` 末尾的 `}  // namespace cxx_mlx` 之前追加：

```cpp
// ===== SafetensorsLoadResult getters =====

rust::Vec<rust::String> safetensors_tensor_names(const SafetensorsLoadResult& r) {
  rust::Vec<rust::String> out;
  out.reserve(r.inner_.first.size());
  for (const auto& kv : r.inner_.first) {
    out.push_back(rust::String(kv.first));
  }
  return out;
}

rust::Vec<std::unique_ptr<MlxArray>> safetensors_take_tensor_values(
    SafetensorsLoadResult& r) {
  rust::Vec<std::unique_ptr<MlxArray>> out;
  out.reserve(r.inner_.first.size());
  for (auto& kv : r.inner_.first) {
    out.push_back(std::make_unique<MlxArray>(std::move(kv.second)));
  }
  return out;
}

rust::Vec<rust::String> safetensors_metadata_names(const SafetensorsLoadResult& r) {
  rust::Vec<rust::String> out;
  out.reserve(r.inner_.second.size());
  for (const auto& kv : r.inner_.second) {
    out.push_back(rust::String(kv.first));
  }
  return out;
}

rust::Vec<rust::String> safetensors_metadata_values(const SafetensorsLoadResult& r) {
  rust::Vec<rust::String> out;
  out.reserve(r.inner_.second.size());
  for (const auto& kv : r.inner_.second) {
    out.push_back(rust::String(kv.second));
  }
  return out;
}

// ===== SafetensorsSaveBuilder =====

std::unique_ptr<SafetensorsSaveBuilder> new_safetensors_save_builder() {
  return std::make_unique<SafetensorsSaveBuilder>();
}

void safetensors_builder_add_tensor(
    SafetensorsSaveBuilder& b, rust::Str name, const MlxArray& array) {
  b.tensors.emplace(std::string(name), array);
}

void safetensors_builder_add_metadata(
    SafetensorsSaveBuilder& b, rust::Str key, rust::Str value) {
  b.metadata.emplace(std::string(key), std::string(value));
}

// ===== 顶层 load/save =====

std::unique_ptr<SafetensorsLoadResult> load_safetensors_file(rust::Str path) {
  auto data = mlx::core::load_safetensors(std::string(path));
  return std::make_unique<SafetensorsLoadResult>(std::move(data));
}

std::unique_ptr<SafetensorsLoadResult> load_safetensors_reader(MlxReader& reader) {
  auto data = mlx::core::load_safetensors(reader.ptr);
  return std::make_unique<SafetensorsLoadResult>(std::move(data));
}

void save_safetensors_file(rust::Str path, const SafetensorsSaveBuilder& b) {
  mlx::core::save_safetensors(std::string(path), b.tensors, b.metadata);
}

void save_safetensors_writer(MlxWriter& writer, const SafetensorsSaveBuilder& b) {
  mlx::core::save_safetensors(writer.ptr, b.tensors, b.metadata);
}
```

- [ ] **Step 2.5: 桥接追加 SafetensorsLoadResult/SafetensorsSaveBuilder 类型 + 函数**

在 `mlx-sys/src/bridge/io.rs` 的 `unsafe extern "C++"` 块内（`writer_into_bytes` 之后）追加：

```rust
        type SafetensorsLoadResult;
        type SafetensorsSaveBuilder;

        // ===== safetensors =====
        fn load_safetensors_file(path: &str) -> Result<UniquePtr<SafetensorsLoadResult>>;
        fn load_safetensors_reader(reader: Pin<&mut MlxReader>) -> Result<UniquePtr<SafetensorsLoadResult>>;
        fn safetensors_tensor_names(r: &SafetensorsLoadResult) -> Vec<String>;
        fn safetensors_take_tensor_values(r: Pin<&mut SafetensorsLoadResult>) -> Vec<UniquePtr<MlxArray>>;
        fn safetensors_metadata_names(r: &SafetensorsLoadResult) -> Vec<String>;
        fn safetensors_metadata_values(r: &SafetensorsLoadResult) -> Vec<String>;

        fn new_safetensors_save_builder() -> UniquePtr<SafetensorsSaveBuilder>;
        fn safetensors_builder_add_tensor(b: Pin<&mut SafetensorsSaveBuilder>, name: &str, array: &MlxArray);
        fn safetensors_builder_add_metadata(b: Pin<&mut SafetensorsSaveBuilder>, key: &str, value: &str);
        fn save_safetensors_file(path: &str, builder: &SafetensorsSaveBuilder) -> Result<()>;
        fn save_safetensors_writer(writer: Pin<&mut MlxWriter>, builder: &SafetensorsSaveBuilder) -> Result<()>;
```

- [ ] **Step 2.6: 安全 API 追加 4 个 safetensors 函数 + 解构辅助**

在 `mlx/src/io.rs` 的末尾追加：

```rust
use std::collections::HashMap;

use crate::Array;

// ===== safetensors =====

/// Load tensors + string metadata from a `.safetensors` file.
pub fn load_safetensors(
    path: &str,
) -> Result<(HashMap<String, Array>, HashMap<String, String>)> {
    let mut result = mlx_sys::io::ffi::load_safetensors_file(path).map_err(Error::from)?;
    Ok(safetensors_decompose(&mut result))
}

/// Load tensors + string metadata from a Reader.
pub fn load_safetensors_from_reader(
    reader: &mut Reader,
) -> Result<(HashMap<String, Array>, HashMap<String, String>)> {
    let mut result = mlx_sys::io::ffi::load_safetensors_reader(reader.pin_mut())
        .map_err(Error::from)?;
    Ok(safetensors_decompose(&mut result))
}

fn safetensors_decompose(
    result: &mut cxx::UniquePtr<mlx_sys::io::ffi::SafetensorsLoadResult>,
) -> (HashMap<String, Array>, HashMap<String, String>) {
    let names = mlx_sys::io::ffi::safetensors_tensor_names(result);
    let values = mlx_sys::io::ffi::safetensors_take_tensor_values(result.pin_mut());
    let tensors: HashMap<String, Array> = names
        .into_iter()
        .zip(values.into_iter().map(Array::from_inner))
        .collect();
    let meta_names = mlx_sys::io::ffi::safetensors_metadata_names(result);
    let meta_values = mlx_sys::io::ffi::safetensors_metadata_values(result);
    let metadata: HashMap<String, String> =
        meta_names.into_iter().zip(meta_values).collect();
    (tensors, metadata)
}

/// Save tensors + metadata to a `.safetensors` file.
pub fn save_safetensors(
    path: &str,
    tensors: &HashMap<String, Array>,
    metadata: &HashMap<String, String>,
) -> Result<()> {
    let builder = build_safetensors_builder(tensors, metadata);
    mlx_sys::io::ffi::save_safetensors_file(path, &builder).map_err(Error::from)
}

/// Save tensors + metadata to a Writer.
pub fn save_safetensors_to_writer(
    writer: &mut Writer,
    tensors: &HashMap<String, Array>,
    metadata: &HashMap<String, String>,
) -> Result<()> {
    let builder = build_safetensors_builder(tensors, metadata);
    mlx_sys::io::ffi::save_safetensors_writer(writer.pin_mut(), &builder)
        .map_err(Error::from)
}

fn build_safetensors_builder(
    tensors: &HashMap<String, Array>,
    metadata: &HashMap<String, String>,
) -> cxx::UniquePtr<mlx_sys::io::ffi::SafetensorsSaveBuilder> {
    let mut builder = mlx_sys::io::ffi::new_safetensors_save_builder();
    for (name, array) in tensors {
        mlx_sys::io::ffi::safetensors_builder_add_tensor(
            builder.pin_mut(),
            name,
            array.as_inner(),
        );
    }
    for (key, value) in metadata {
        mlx_sys::io::ffi::safetensors_builder_add_metadata(builder.pin_mut(), key, value);
    }
    builder
}
```

把 Reader 与 Writer 中的 `#[allow(dead_code)]` 注解去掉（pin_mut 现在在被使用）：

打开 `mlx/src/io.rs`，找到 `Reader::pin_mut` 和 `Writer::pin_mut` 上方的 `#[allow(dead_code)]  // Will be used by ...` 注解行，删除这两行。

- [ ] **Step 2.7: 测试通过**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p2c_io`
Expected: 9 tests passed（前 5 + safetensors 4）。

- [ ] **Step 2.8: Rust 检查**

```bash
export MLX_DIR=/Users/sam/.local/mlx
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app --tests -- -D warnings
cargo build --release
```
Expected: 全部通过。

- [ ] **Step 2.9: 提交**

```bash
git add mlx-sys/shim/include/cxx_mlx_shim/io.h \
        mlx-sys/shim/src/io.cc \
        mlx-sys/src/bridge/io.rs \
        mlx/src/io.rs \
        mlx/tests/p2c_io.rs
git commit -m "feat(p2c): safetensors load + save (file + stream, 4 tests)"
```

---

## Task 3: gguf load + save（file only）

**目的**：追加 GGUF 完整 IO，含 `GGUFLoadResult` opaque 类（9 getter，按 variant 类型拆 metadata）+ `GGUFSaveBuilder` opaque 类（6 add/control 方法）+ `GGUFMetaData` Rust 枚举 + 2 个 top-level 函数（load/save，仅 file path）。

**Files (all modifications):**
- Modify: `mlx-sys/shim/include/cxx_mlx_shim/io.h`
- Modify: `mlx-sys/shim/src/io.cc`
- Modify: `mlx-sys/src/bridge/io.rs`
- Modify: `mlx/src/io.rs`
- Modify: `mlx/tests/p2c_io.rs`

- [ ] **Step 3.1: 写失败的集成测试**

在 `mlx/tests/p2c_io.rs` 末尾追加：

```rust
use mlx::io::GGUFMetaData;

#[test]
fn gguf_round_trip_tensor_only() {
    let tmp = tempfile::Builder::new().suffix(".gguf").tempfile().expect("tempfile");
    let path = tmp.path().to_str().unwrap();
    let tensors = make_test_tensors();
    let metadata: HashMap<String, GGUFMetaData> = HashMap::new();

    io::save_gguf(path, &tensors, &metadata).expect("save");
    let (loaded_tensors, loaded_meta) = io::load_gguf(path).expect("load");

    assert_eq!(loaded_tensors.len(), tensors.len());
    assert!(loaded_meta.is_empty());
    let alpha_in: Vec<f32> = tensors["alpha"].to_vec().expect("alpha");
    let alpha_out: Vec<f32> = loaded_tensors["alpha"].to_vec().expect("alpha out");
    assert_eq!(alpha_in, alpha_out);
}

#[test]
fn gguf_round_trip_with_string_meta() {
    let tmp = tempfile::Builder::new().suffix(".gguf").tempfile().expect("tempfile");
    let path = tmp.path().to_str().unwrap();
    let tensors = make_test_tensors();
    let mut metadata: HashMap<String, GGUFMetaData> = HashMap::new();
    metadata.insert("model".to_string(), GGUFMetaData::String("test-model".to_string()));
    metadata.insert("version".to_string(), GGUFMetaData::String("1.0".to_string()));

    io::save_gguf(path, &tensors, &metadata).expect("save");
    let (_loaded_tensors, loaded_meta) = io::load_gguf(path).expect("load");

    assert_eq!(loaded_meta.len(), 2);
    match &loaded_meta["model"] {
        GGUFMetaData::String(s) => assert_eq!(s, "test-model"),
        _ => panic!("expected String variant for 'model'"),
    }
    match &loaded_meta["version"] {
        GGUFMetaData::String(s) => assert_eq!(s, "1.0"),
        _ => panic!("expected String variant for 'version'"),
    }
}

#[test]
fn gguf_round_trip_with_string_list_meta() {
    let tmp = tempfile::Builder::new().suffix(".gguf").tempfile().expect("tempfile");
    let path = tmp.path().to_str().unwrap();
    let tensors = make_test_tensors();
    let mut metadata: HashMap<String, GGUFMetaData> = HashMap::new();
    metadata.insert(
        "tags".to_string(),
        GGUFMetaData::StringList(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
    );

    io::save_gguf(path, &tensors, &metadata).expect("save");
    let (_loaded_tensors, loaded_meta) = io::load_gguf(path).expect("load");

    match &loaded_meta["tags"] {
        GGUFMetaData::StringList(list) => {
            assert_eq!(list, &vec!["a".to_string(), "b".to_string(), "c".to_string()]);
        }
        _ => panic!("expected StringList variant for 'tags'"),
    }
}

#[test]
fn gguf_round_trip_with_array_meta() {
    let tmp = tempfile::Builder::new().suffix(".gguf").tempfile().expect("tempfile");
    let path = tmp.path().to_str().unwrap();
    let tensors = make_test_tensors();
    let mut metadata: HashMap<String, GGUFMetaData> = HashMap::new();
    let scale_array = Array::from_slice(&[2.5_f32, 3.5], &[2]).expect("scale");
    metadata.insert("scale".to_string(), GGUFMetaData::Array(scale_array));

    io::save_gguf(path, &tensors, &metadata).expect("save");
    let (_loaded_tensors, loaded_meta) = io::load_gguf(path).expect("load");

    match &loaded_meta["scale"] {
        GGUFMetaData::Array(arr) => {
            let v: Vec<f32> = arr.to_vec().expect("array meta to_vec");
            assert_eq!(v, vec![2.5_f32, 3.5]);
        }
        _ => panic!("expected Array variant for 'scale'"),
    }
}

#[test]
fn gguf_load_nonexistent_file_returns_err() {
    let result = io::load_gguf("/nonexistent/path/foo.gguf");
    assert!(result.is_err());
}
```

- [ ] **Step 3.2: 运行测试，确认失败**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p2c_io --no-run`
Expected: 编译失败，提示 `io::save_gguf`、`io::load_gguf`、`io::GGUFMetaData` 不存在。

- [ ] **Step 3.3: shim 头追加 GGUFLoadResult/GGUFSaveBuilder + 顶层函数**

在 `mlx-sys/shim/include/cxx_mlx_shim/io.h` 末尾的 `}  // namespace cxx_mlx` 之前追加：

```cpp
// ===== GGUFLoadResult (opaque) =====

class GGUFLoadResult {
 public:
  explicit GGUFLoadResult(mlx::core::GGUFLoad data) : inner_(std::move(data)) {}
  mlx::core::GGUFLoad inner_;
};

rust::Vec<rust::String> gguf_tensor_names(const GGUFLoadResult& r);
rust::Vec<std::unique_ptr<MlxArray>> gguf_take_tensor_values(GGUFLoadResult& r);
// metadata 按 variant 类型拆（monostate 静默丢弃）
rust::Vec<rust::String> gguf_array_meta_names(const GGUFLoadResult& r);
rust::Vec<std::unique_ptr<MlxArray>> gguf_take_array_meta_values(GGUFLoadResult& r);
rust::Vec<rust::String> gguf_string_meta_names(const GGUFLoadResult& r);
rust::Vec<rust::String> gguf_string_meta_values(const GGUFLoadResult& r);
rust::Vec<rust::String> gguf_string_list_meta_names(const GGUFLoadResult& r);
// string list 用 packed (concat) + lengths 表达
rust::Vec<rust::String> gguf_string_list_meta_values_packed(const GGUFLoadResult& r);
rust::Vec<uint64_t> gguf_string_list_meta_lengths(const GGUFLoadResult& r);

// ===== GGUFSaveBuilder (opaque) =====

class GGUFSaveBuilder {
 public:
  std::unordered_map<std::string, mlx::core::array> tensors;
  std::unordered_map<std::string, mlx::core::GGUFMetaData> metadata;
  // string list 用 begin/push/end 三步法
  std::optional<std::pair<std::string, std::vector<std::string>>> pending_list;
};

std::unique_ptr<GGUFSaveBuilder> new_gguf_save_builder();
void gguf_builder_add_tensor(
    GGUFSaveBuilder& b, rust::Str name, const MlxArray& array);
void gguf_builder_add_array_meta(
    GGUFSaveBuilder& b, rust::Str key, const MlxArray& array);
void gguf_builder_add_string_meta(
    GGUFSaveBuilder& b, rust::Str key, rust::Str value);
void gguf_builder_begin_string_list_meta(GGUFSaveBuilder& b, rust::Str key);
void gguf_builder_push_string_list_meta(GGUFSaveBuilder& b, rust::Str value);
void gguf_builder_end_string_list_meta(GGUFSaveBuilder& b);

std::unique_ptr<GGUFLoadResult> load_gguf_file(rust::Str path);
void save_gguf_file(rust::Str path, const GGUFSaveBuilder& builder);
```

- [ ] **Step 3.4: shim cc 追加 GGUFLoadResult/GGUFSaveBuilder 实现**

在 `mlx-sys/shim/src/io.cc` 末尾的 `}  // namespace cxx_mlx` 之前追加：

```cpp
// ===== GGUFLoadResult getters =====

rust::Vec<rust::String> gguf_tensor_names(const GGUFLoadResult& r) {
  rust::Vec<rust::String> out;
  out.reserve(r.inner_.first.size());
  for (const auto& kv : r.inner_.first) {
    out.push_back(rust::String(kv.first));
  }
  return out;
}

rust::Vec<std::unique_ptr<MlxArray>> gguf_take_tensor_values(GGUFLoadResult& r) {
  rust::Vec<std::unique_ptr<MlxArray>> out;
  out.reserve(r.inner_.first.size());
  for (auto& kv : r.inner_.first) {
    out.push_back(std::make_unique<MlxArray>(std::move(kv.second)));
  }
  return out;
}

rust::Vec<rust::String> gguf_array_meta_names(const GGUFLoadResult& r) {
  rust::Vec<rust::String> out;
  for (const auto& kv : r.inner_.second) {
    if (std::holds_alternative<mlx::core::array>(kv.second)) {
      out.push_back(rust::String(kv.first));
    }
  }
  return out;
}

rust::Vec<std::unique_ptr<MlxArray>> gguf_take_array_meta_values(GGUFLoadResult& r) {
  rust::Vec<std::unique_ptr<MlxArray>> out;
  for (auto& kv : r.inner_.second) {
    if (std::holds_alternative<mlx::core::array>(kv.second)) {
      out.push_back(std::make_unique<MlxArray>(
          std::move(std::get<mlx::core::array>(kv.second))));
    }
  }
  return out;
}

rust::Vec<rust::String> gguf_string_meta_names(const GGUFLoadResult& r) {
  rust::Vec<rust::String> out;
  for (const auto& kv : r.inner_.second) {
    if (std::holds_alternative<std::string>(kv.second)) {
      out.push_back(rust::String(kv.first));
    }
  }
  return out;
}

rust::Vec<rust::String> gguf_string_meta_values(const GGUFLoadResult& r) {
  rust::Vec<rust::String> out;
  for (const auto& kv : r.inner_.second) {
    if (std::holds_alternative<std::string>(kv.second)) {
      out.push_back(rust::String(std::get<std::string>(kv.second)));
    }
  }
  return out;
}

rust::Vec<rust::String> gguf_string_list_meta_names(const GGUFLoadResult& r) {
  rust::Vec<rust::String> out;
  for (const auto& kv : r.inner_.second) {
    if (std::holds_alternative<std::vector<std::string>>(kv.second)) {
      out.push_back(rust::String(kv.first));
    }
  }
  return out;
}

rust::Vec<rust::String> gguf_string_list_meta_values_packed(const GGUFLoadResult& r) {
  rust::Vec<rust::String> out;
  for (const auto& kv : r.inner_.second) {
    if (std::holds_alternative<std::vector<std::string>>(kv.second)) {
      const auto& list = std::get<std::vector<std::string>>(kv.second);
      for (const auto& s : list) {
        out.push_back(rust::String(s));
      }
    }
  }
  return out;
}

rust::Vec<uint64_t> gguf_string_list_meta_lengths(const GGUFLoadResult& r) {
  rust::Vec<uint64_t> out;
  for (const auto& kv : r.inner_.second) {
    if (std::holds_alternative<std::vector<std::string>>(kv.second)) {
      out.push_back(static_cast<uint64_t>(
          std::get<std::vector<std::string>>(kv.second).size()));
    }
  }
  return out;
}

// ===== GGUFSaveBuilder =====

std::unique_ptr<GGUFSaveBuilder> new_gguf_save_builder() {
  return std::make_unique<GGUFSaveBuilder>();
}

void gguf_builder_add_tensor(
    GGUFSaveBuilder& b, rust::Str name, const MlxArray& array) {
  b.tensors.emplace(std::string(name), array);
}

void gguf_builder_add_array_meta(
    GGUFSaveBuilder& b, rust::Str key, const MlxArray& array) {
  b.metadata.emplace(std::string(key), mlx::core::GGUFMetaData(array));
}

void gguf_builder_add_string_meta(
    GGUFSaveBuilder& b, rust::Str key, rust::Str value) {
  b.metadata.emplace(std::string(key), mlx::core::GGUFMetaData(std::string(value)));
}

void gguf_builder_begin_string_list_meta(GGUFSaveBuilder& b, rust::Str key) {
  if (b.pending_list.has_value()) {
    throw std::runtime_error(
        "begin_string_list_meta called without end_string_list_meta");
  }
  b.pending_list = std::make_pair(std::string(key), std::vector<std::string>{});
}

void gguf_builder_push_string_list_meta(GGUFSaveBuilder& b, rust::Str value) {
  if (!b.pending_list.has_value()) {
    throw std::runtime_error(
        "push_string_list_meta called without begin_string_list_meta");
  }
  b.pending_list->second.push_back(std::string(value));
}

void gguf_builder_end_string_list_meta(GGUFSaveBuilder& b) {
  if (!b.pending_list.has_value()) {
    throw std::runtime_error(
        "end_string_list_meta called without begin_string_list_meta");
  }
  b.metadata.emplace(
      std::move(b.pending_list->first),
      mlx::core::GGUFMetaData(std::move(b.pending_list->second)));
  b.pending_list.reset();
}

// ===== 顶层 load/save =====

std::unique_ptr<GGUFLoadResult> load_gguf_file(rust::Str path) {
  auto data = mlx::core::load_gguf(std::string(path));
  return std::make_unique<GGUFLoadResult>(std::move(data));
}

void save_gguf_file(rust::Str path, const GGUFSaveBuilder& b) {
  mlx::core::save_gguf(std::string(path), b.tensors, b.metadata);
}
```

- [ ] **Step 3.5: 桥接追加 GGUFLoadResult/GGUFSaveBuilder + 函数**

在 `mlx-sys/src/bridge/io.rs` 的 `unsafe extern "C++"` 块内（safetensors 之后）追加：

```rust
        type GGUFLoadResult;
        type GGUFSaveBuilder;

        // ===== GGUF =====
        fn load_gguf_file(path: &str) -> Result<UniquePtr<GGUFLoadResult>>;
        fn gguf_tensor_names(r: &GGUFLoadResult) -> Vec<String>;
        fn gguf_take_tensor_values(r: Pin<&mut GGUFLoadResult>) -> Vec<UniquePtr<MlxArray>>;
        fn gguf_array_meta_names(r: &GGUFLoadResult) -> Vec<String>;
        fn gguf_take_array_meta_values(r: Pin<&mut GGUFLoadResult>) -> Vec<UniquePtr<MlxArray>>;
        fn gguf_string_meta_names(r: &GGUFLoadResult) -> Vec<String>;
        fn gguf_string_meta_values(r: &GGUFLoadResult) -> Vec<String>;
        fn gguf_string_list_meta_names(r: &GGUFLoadResult) -> Vec<String>;
        fn gguf_string_list_meta_values_packed(r: &GGUFLoadResult) -> Vec<String>;
        fn gguf_string_list_meta_lengths(r: &GGUFLoadResult) -> Vec<u64>;

        fn new_gguf_save_builder() -> UniquePtr<GGUFSaveBuilder>;
        fn gguf_builder_add_tensor(b: Pin<&mut GGUFSaveBuilder>, name: &str, array: &MlxArray);
        fn gguf_builder_add_array_meta(b: Pin<&mut GGUFSaveBuilder>, key: &str, array: &MlxArray);
        fn gguf_builder_add_string_meta(b: Pin<&mut GGUFSaveBuilder>, key: &str, value: &str);
        fn gguf_builder_begin_string_list_meta(b: Pin<&mut GGUFSaveBuilder>, key: &str) -> Result<()>;
        fn gguf_builder_push_string_list_meta(b: Pin<&mut GGUFSaveBuilder>, value: &str) -> Result<()>;
        fn gguf_builder_end_string_list_meta(b: Pin<&mut GGUFSaveBuilder>) -> Result<()>;
        fn save_gguf_file(path: &str, builder: &GGUFSaveBuilder) -> Result<()>;
```

- [ ] **Step 3.6: 安全 API 追加 GGUFMetaData enum + 2 个 gguf 函数**

在 `mlx/src/io.rs` 的末尾追加：

```rust
// ===== GGUF =====

/// GGUF metadata value. Mirrors `mlx::core::GGUFMetaData` minus monostate
/// (the empty variant is silently dropped during load).
#[derive(Debug)]
pub enum GGUFMetaData {
    Array(Array),
    String(String),
    StringList(Vec<String>),
}

/// Load tensors + GGUF metadata from a `.gguf` file.
pub fn load_gguf(
    path: &str,
) -> Result<(HashMap<String, Array>, HashMap<String, GGUFMetaData>)> {
    let mut result = mlx_sys::io::ffi::load_gguf_file(path).map_err(Error::from)?;
    Ok(gguf_decompose(&mut result))
}

fn gguf_decompose(
    result: &mut cxx::UniquePtr<mlx_sys::io::ffi::GGUFLoadResult>,
) -> (HashMap<String, Array>, HashMap<String, GGUFMetaData>) {
    // tensors
    let tensor_names = mlx_sys::io::ffi::gguf_tensor_names(result);
    let tensor_values = mlx_sys::io::ffi::gguf_take_tensor_values(result.pin_mut());
    let tensors: HashMap<String, Array> = tensor_names
        .into_iter()
        .zip(tensor_values.into_iter().map(Array::from_inner))
        .collect();

    // metadata: 三类合并
    let mut metadata: HashMap<String, GGUFMetaData> = HashMap::new();

    // array metadata
    let arr_names = mlx_sys::io::ffi::gguf_array_meta_names(result);
    let arr_values = mlx_sys::io::ffi::gguf_take_array_meta_values(result.pin_mut());
    for (name, arr) in arr_names.into_iter().zip(arr_values) {
        metadata.insert(name, GGUFMetaData::Array(Array::from_inner(arr)));
    }

    // string metadata
    let str_names = mlx_sys::io::ffi::gguf_string_meta_names(result);
    let str_values = mlx_sys::io::ffi::gguf_string_meta_values(result);
    for (name, value) in str_names.into_iter().zip(str_values) {
        metadata.insert(name, GGUFMetaData::String(value));
    }

    // string list metadata: 解 packed
    let list_names = mlx_sys::io::ffi::gguf_string_list_meta_names(result);
    let packed = mlx_sys::io::ffi::gguf_string_list_meta_values_packed(result);
    let lengths = mlx_sys::io::ffi::gguf_string_list_meta_lengths(result);
    let mut idx: usize = 0;
    for (name, len) in list_names.into_iter().zip(lengths) {
        let len = len as usize;
        let strings: Vec<String> = packed[idx..idx + len].to_vec();
        idx += len;
        metadata.insert(name, GGUFMetaData::StringList(strings));
    }

    (tensors, metadata)
}

/// Save tensors + GGUF metadata to a `.gguf` file.
pub fn save_gguf(
    path: &str,
    tensors: &HashMap<String, Array>,
    metadata: &HashMap<String, GGUFMetaData>,
) -> Result<()> {
    let mut builder = mlx_sys::io::ffi::new_gguf_save_builder();
    for (name, array) in tensors {
        mlx_sys::io::ffi::gguf_builder_add_tensor(builder.pin_mut(), name, array.as_inner());
    }
    for (key, value) in metadata {
        match value {
            GGUFMetaData::Array(a) => mlx_sys::io::ffi::gguf_builder_add_array_meta(
                builder.pin_mut(),
                key,
                a.as_inner(),
            ),
            GGUFMetaData::String(s) => mlx_sys::io::ffi::gguf_builder_add_string_meta(
                builder.pin_mut(),
                key,
                s,
            ),
            GGUFMetaData::StringList(items) => {
                mlx_sys::io::ffi::gguf_builder_begin_string_list_meta(builder.pin_mut(), key)
                    .map_err(Error::from)?;
                for item in items {
                    mlx_sys::io::ffi::gguf_builder_push_string_list_meta(
                        builder.pin_mut(),
                        item,
                    )
                    .map_err(Error::from)?;
                }
                mlx_sys::io::ffi::gguf_builder_end_string_list_meta(builder.pin_mut())
                    .map_err(Error::from)?;
            }
        }
    }
    mlx_sys::io::ffi::save_gguf_file(path, &builder).map_err(Error::from)
}
```

- [ ] **Step 3.7: 测试通过**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p2c_io`
Expected: 14 tests passed（前 9 + gguf 5）。

注：如果 GGUF 测试因 MLX 上游对某个 dtype/张量大小限制而失败，停下来报告 BLOCKED 并描述实际行为，不要硬改。

- [ ] **Step 3.8: Rust 检查**

```bash
export MLX_DIR=/Users/sam/.local/mlx
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app --tests -- -D warnings
cargo build --release
```
Expected: 全部通过。

- [ ] **Step 3.9: 提交**

```bash
git add mlx-sys/shim/include/cxx_mlx_shim/io.h \
        mlx-sys/shim/src/io.cc \
        mlx-sys/src/bridge/io.rs \
        mlx/src/io.rs \
        mlx/tests/p2c_io.rs
git commit -m "feat(p2c): gguf load + save with variant metadata (5 tests)"
```

---

## Task 4: npy load + save（file + stream）

**目的**：追加 npy 单数组 IO，4 个顶层函数（load/save × file/stream）。

**Files (all modifications):**
- Modify: `mlx-sys/shim/include/cxx_mlx_shim/io.h`
- Modify: `mlx-sys/shim/src/io.cc`
- Modify: `mlx-sys/src/bridge/io.rs`
- Modify: `mlx/src/io.rs`
- Modify: `mlx/tests/p2c_io.rs`

- [ ] **Step 4.1: 写失败的集成测试**

在 `mlx/tests/p2c_io.rs` 末尾追加：

```rust
#[test]
fn npy_round_trip_file() {
    let tmp = tempfile::Builder::new().suffix(".npy").tempfile().expect("tempfile");
    let path = tmp.path().to_str().unwrap();
    let array = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("a");

    io::save_npy(path, &array).expect("save_npy");
    let loaded = io::load_npy(path).expect("load_npy");

    assert_eq!(loaded.shape().as_slice(), &[2, 3]);
    let v: Vec<f32> = loaded.to_vec().expect("to_vec");
    assert_eq!(v, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn npy_round_trip_memory() {
    let array = Array::from_slice(&[10.0_f32, 20.0, 30.0], &[3]).expect("a");

    let mut writer = io::Writer::memory();
    io::save_npy_to_writer(&mut writer, &array).expect("save");
    let bytes = writer.into_bytes().expect("into_bytes");
    assert!(!bytes.is_empty());

    let mut reader = io::Reader::from_bytes(&bytes);
    let loaded = io::load_npy_from_reader(&mut reader).expect("load");

    assert_eq!(loaded.shape().as_slice(), &[3]);
    let v: Vec<f32> = loaded.to_vec().expect("to_vec");
    assert_eq!(v, vec![10.0_f32, 20.0, 30.0]);
}

#[test]
fn npy_load_nonexistent_file_returns_err() {
    let result = io::load_npy("/nonexistent/path/foo.npy");
    assert!(result.is_err());
}
```

- [ ] **Step 4.2: 运行测试，确认失败**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p2c_io --no-run`
Expected: 编译失败，提示 `io::save_npy`、`io::load_npy`、`io::save_npy_to_writer`、`io::load_npy_from_reader` 不存在。

- [ ] **Step 4.3: shim 头追加 4 个 npy 函数声明**

在 `mlx-sys/shim/include/cxx_mlx_shim/io.h` 末尾的 `}  // namespace cxx_mlx` 之前追加：

```cpp
// ===== npy (single-array) =====

std::unique_ptr<MlxArray> load_npy_file(rust::Str path);
std::unique_ptr<MlxArray> load_npy_reader(MlxReader& reader);
void save_npy_file(rust::Str path, const MlxArray& array);
void save_npy_writer(MlxWriter& writer, const MlxArray& array);
```

- [ ] **Step 4.4: shim cc 追加 4 个 npy 函数实现**

在 `mlx-sys/shim/src/io.cc` 末尾的 `}  // namespace cxx_mlx` 之前追加：

```cpp
// ===== npy =====

std::unique_ptr<MlxArray> load_npy_file(rust::Str path) {
  return std::make_unique<MlxArray>(mlx::core::load(std::string(path)));
}

std::unique_ptr<MlxArray> load_npy_reader(MlxReader& reader) {
  return std::make_unique<MlxArray>(mlx::core::load(reader.ptr));
}

void save_npy_file(rust::Str path, const MlxArray& array) {
  mlx::core::save(std::string(path), array);
}

void save_npy_writer(MlxWriter& writer, const MlxArray& array) {
  mlx::core::save(writer.ptr, array);
}
```

- [ ] **Step 4.5: 桥接追加 4 个 npy 函数**

在 `mlx-sys/src/bridge/io.rs` 的 `unsafe extern "C++"` 块内（gguf 之后）追加：

```rust
        // ===== npy =====
        fn load_npy_file(path: &str) -> Result<UniquePtr<MlxArray>>;
        fn load_npy_reader(reader: Pin<&mut MlxReader>) -> Result<UniquePtr<MlxArray>>;
        fn save_npy_file(path: &str, array: &MlxArray) -> Result<()>;
        fn save_npy_writer(writer: Pin<&mut MlxWriter>, array: &MlxArray) -> Result<()>;
```

- [ ] **Step 4.6: 安全 API 追加 4 个 npy 函数**

在 `mlx/src/io.rs` 的末尾追加：

```rust
// ===== npy =====

/// Load a single array from a `.npy` file.
pub fn load_npy(path: &str) -> Result<Array> {
    let inner = mlx_sys::io::ffi::load_npy_file(path).map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Load a single array from a Reader (`.npy` format).
pub fn load_npy_from_reader(reader: &mut Reader) -> Result<Array> {
    let inner = mlx_sys::io::ffi::load_npy_reader(reader.pin_mut()).map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Save a single array to a `.npy` file.
pub fn save_npy(path: &str, array: &Array) -> Result<()> {
    mlx_sys::io::ffi::save_npy_file(path, array.as_inner()).map_err(Error::from)
}

/// Save a single array to a Writer (`.npy` format).
pub fn save_npy_to_writer(writer: &mut Writer, array: &Array) -> Result<()> {
    mlx_sys::io::ffi::save_npy_writer(writer.pin_mut(), array.as_inner()).map_err(Error::from)
}
```

- [ ] **Step 4.7: 测试通过**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p2c_io`
Expected: 17 tests passed（前 14 + npy 3）。

- [ ] **Step 4.8: Rust 检查**

```bash
export MLX_DIR=/Users/sam/.local/mlx
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app --tests -- -D warnings
cargo build --release
```
Expected: 全部通过。

- [ ] **Step 4.9: 提交**

```bash
git add mlx-sys/shim/include/cxx_mlx_shim/io.h \
        mlx-sys/shim/src/io.cc \
        mlx-sys/src/bridge/io.rs \
        mlx/src/io.rs \
        mlx/tests/p2c_io.rs
git commit -m "feat(p2c): npy load + save (file + stream, 3 tests)"
```

---

## Task 5: 顶层 re-export + README + 全套验证

**目的**：把 P2c 公开 API 暴露到 `mlx::*` 顶层，README 状态升级到 P2c 完成，跑完整 workspace 检查。

**Files:**
- Modify: `mlx/src/lib.rs`
- Modify: `mlx/tests/p2c_io.rs`
- Modify: `README.md`

- [ ] **Step 5.1: 在 `mlx/src/lib.rs` 的 `pub mod io;` 后追加 re-export**

打开 `mlx/src/lib.rs`，找到 `pub mod io;`，在其后追加 re-export。最终该处呈现：

```rust
pub mod io;
pub use io::{
    load_gguf, load_npy, load_npy_from_reader, load_safetensors,
    load_safetensors_from_reader, save_gguf, save_npy, save_npy_to_writer,
    save_safetensors, save_safetensors_to_writer, GGUFMetaData, Reader, Writer,
};
```

（cargo fmt 会按字母排序，提交时以 fmt 后的为准。）

- [ ] **Step 5.2: 在测试文件追加 re-export 验证测试**

在 `mlx/tests/p2c_io.rs` 末尾追加：

```rust
#[test]
fn top_level_re_exports_work() {
    // 验证可以通过 mlx::* 顶层访问 P2c 公开 API
    let _r = mlx::Reader::from_bytes(&[]);
    let _w = mlx::Writer::memory();
    // GGUFMetaData 类型可访问
    let _meta = mlx::GGUFMetaData::String("test".to_string());

    // load_safetensors 函数可达（不需要真正成功调用）
    let result = mlx::load_safetensors("/nonexistent/foo.safetensors");
    assert!(result.is_err());
}
```

- [ ] **Step 5.3: 运行所有 P2c 测试**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --test p2c_io`
Expected: 18 tests passed。

- [ ] **Step 5.4: 跑完整 workspace 测试**

Run: `MLX_DIR=/Users/sam/.local/mlx cargo test --workspace --all-features`
Expected: 所有现有测试 + 18 个新 P2c 测试全部通过。

- [ ] **Step 5.5: 更新 `README.md`**

`README.md` 中已有的 P2b 状态记录与 Roadmap 表格需要升级。已勘察的关键位置：

**位置 A（Status banner，约 line 5）**：当前文本是：
```
**Status:** 🚧 **P2b complete** — Fused inference kernels via `mlx::fast::*` (rms_norm, layer_norm, rope with int and array offset, scaled_dot_product_attention). Builds on P2a's Stream/Device foundation. Next: P2c (`io`: safetensors/gguf load).
```

升级为类似（保留中英文风格）：
```
**Status:** 🚧 **P2c complete** — Full IO subsystem: `mlx::io::{load,save}_{safetensors,gguf,npy}` + B-lite `Reader`/`Writer` (file + in-memory). Builds on P2b's fused kernels. Next: P3 (quantization ops for 4-bit GGUF/MLX inference).
```

**位置 B（Roadmap 表格，约 line 215）**：当前文本是：
```
- ✅ **P2a** — Stream / Device foundation + runtime-agnostic async_eval
- ✅ **P2b** — `fast` ops (rms_norm / layer_norm / rope int+array offset / sdpa) — 12 integration tests
- ⏳ **P2c** — `io` (safetensors / gguf load)
```

升级为：
```
- ✅ **P2a** — Stream / Device foundation + runtime-agnostic async_eval
- ✅ **P2b** — `fast` ops (rms_norm / layer_norm / rope int+array offset / sdpa) — 12 integration tests
- ✅ **P2c** — `io` (safetensors / gguf / npy + Reader/Writer streams) — 18 integration tests
```

如 README 还有其他 fast/io 相关引用（grep `P2c|safetensors|fast` 检查），按现有风格保持一致。

- [ ] **Step 5.6: 跑全套 Rust 检查**

```bash
export MLX_DIR=/Users/sam/.local/mlx
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app --tests -- -D warnings
cargo build --release
cargo test --workspace --all-features
```
Expected: 全部通过，无 warning，所有测试 PASS。

- [ ] **Step 5.7: 提交**

```bash
git add mlx/src/lib.rs mlx/tests/p2c_io.rs README.md
git commit -m "feat(p2c): re-export io API at crate root + README progress"
```

- [ ] **Step 5.8: 最终 git log 与 commit 数核对**

Run: `git log --oneline | head -10`
Expected: 看到 5 个 P2c feat commit（Tasks 1–5）+ docs (spec + plan) commit。

---

## 自检（plan 作者自检结果）

**Spec 覆盖**：
- ✅ Section "范围"：12 个 MLX 公开函数 — Tasks 1（Reader/Writer 5 个）+ 2（safetensors 4 个）+ 3（gguf 2 个）+ 4（npy 4 个）覆盖
- ✅ Section "Shim 层"：所有 opaque 类（MlxReader/MlxWriter、MemoryReader/MemoryWriter、SafetensorsLoadResult、GGUFLoadResult、SafetensorsSaveBuilder、GGUFSaveBuilder）每个 task 各自定义
- ✅ Section "Bridge 层"：每 task 追加对应 FFI 声明
- ✅ Section "安全层"：HashMap-based API + GGUFMetaData enum + Reader/Writer 类型
- ✅ Section "错误处理"：所有可能抛异常的函数返回 `Result<T>`，shim 不 try/catch
- ✅ Section "测试策略"：18 个集成测试（5 + 4 + 5 + 3 + 1）覆盖各路径
- ✅ Section "文件结构"：与 plan 中文件清单一致
- ✅ Section "风险与缓解"：take_* 单次性消费 / dynamic_cast / unordered_map 顺序问题，plan 内通过 helper 函数（safetensors_decompose / gguf_decompose）封装实现

**类型一致性**：
- 所有 task 用 `Array::from_inner(inner)`（项目惯例）
- 所有 task 用 `array.as_inner()`（项目惯例）
- 所有 task 用 `Error::from` + `?`
- bridge 使用 free function（`arg: &T` / `arg: Pin<&mut T>` / `arg: UniquePtr<T>`），不用 `self: &T` 方法语法
- 所有 `MlxArray` 跨桥接通过 `type MlxArray = crate::bridge::array::ffi::MlxArray;` 共享
- `cxx::UniquePtr<T>` 在安全层用作私有字段；`pin_mut()` 私有 helper 暴露 `Pin<&mut T>`
- `HashMap<String, Array>` 与 `HashMap<String, String>` / `HashMap<String, GGUFMetaData>` 类型在所有 task 一致

**已知 placeholder 修正**：
- 无 TBD/TODO/FIXME
- 每个 step 都有完整代码块或具体命令
- 测试代码全部完整可运行
