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

// ===== SafetensorsLoadResult (opaque) =====

class SafetensorsLoadResult {
 public:
  explicit SafetensorsLoadResult(mlx::core::SafetensorsLoad data)
      : inner_(std::move(data)) {}
  mlx::core::SafetensorsLoad inner_;
};

// 注：take_tensor_by_name 单次性消费（同名重复调用会抛异常）。
// Names 应来自 safetensors_tensor_names()；不存在时 shim 抛 runtime_error。
rust::Vec<rust::String> safetensors_tensor_names(const SafetensorsLoadResult& r);
// Take a single tensor by name (single-use; subsequent take with same name throws).
// Names should be obtained from safetensors_tensor_names(); shim throws if not found.
std::unique_ptr<MlxArray> safetensors_take_tensor_by_name(
    SafetensorsLoadResult& r, rust::Str name);
rust::Vec<rust::String> safetensors_metadata_names(const SafetensorsLoadResult& r);
rust::Vec<rust::String> safetensors_metadata_values(const SafetensorsLoadResult& r);

// ===== SafetensorsSaveBuilder (opaque) =====

class SafetensorsSaveBuilder {
 public:
  std::unordered_map<std::string, mlx::core::array> tensors;
  std::unordered_map<std::string, std::string> metadata;
};

std::unique_ptr<SafetensorsSaveBuilder> new_safetensors_save_builder();
// Adds a tensor to the builder. The array is shallow-copied (shared buffer
// via mlx::core::array's refcounted internals), so this is cheap regardless
// of array size.
void safetensors_builder_add_tensor(
    SafetensorsSaveBuilder& b, rust::Str name, const MlxArray& array);
void safetensors_builder_add_metadata(
    SafetensorsSaveBuilder& b, rust::Str key, rust::Str value);

// ===== 顶层 load/save APIs =====

std::unique_ptr<SafetensorsLoadResult> load_safetensors_file(rust::Str path);
std::unique_ptr<SafetensorsLoadResult> load_safetensors_reader(MlxReader& reader);
void save_safetensors_file(rust::Str path, const SafetensorsSaveBuilder& builder);
void save_safetensors_writer(MlxWriter& writer, const SafetensorsSaveBuilder& builder);

// ===== GGUFLoadResult (opaque) =====

class GGUFLoadResult {
 public:
  explicit GGUFLoadResult(mlx::core::GGUFLoad data) : inner_(std::move(data)) {}
  mlx::core::GGUFLoad inner_;
};

// 注：take_*_by_name 单次性消费——成功取出后会从 map erase；同名重复调用抛异常。
rust::Vec<rust::String> gguf_tensor_names(const GGUFLoadResult& r);
std::unique_ptr<MlxArray> gguf_take_tensor_by_name(GGUFLoadResult& r, rust::Str name);

// metadata 按 variant 类型拆（monostate 静默丢弃）
rust::Vec<rust::String> gguf_array_meta_names(const GGUFLoadResult& r);
std::unique_ptr<MlxArray> gguf_take_array_meta_by_name(GGUFLoadResult& r, rust::Str name);

rust::Vec<rust::String> gguf_string_meta_names(const GGUFLoadResult& r);
rust::Vec<rust::String> gguf_string_meta_values(const GGUFLoadResult& r);

rust::Vec<rust::String> gguf_string_list_meta_names(const GGUFLoadResult& r);
// string list 用 packed (concat) + lengths 表达，避免 nested Vec 桥接限制
rust::Vec<rust::String> gguf_string_list_meta_values_packed(const GGUFLoadResult& r);
rust::Vec<uint64_t> gguf_string_list_meta_lengths(const GGUFLoadResult& r);

// ===== GGUFSaveBuilder (opaque) =====
// Public fields: opaque to Rust; shim free funcs are the canonical accessors.

class GGUFSaveBuilder {
 public:
  std::unordered_map<std::string, mlx::core::array> tensors;
  std::unordered_map<std::string, mlx::core::GGUFMetaData> metadata;
  // string list 用 begin/push/end 三步法
  std::optional<std::pair<std::string, std::vector<std::string>>> pending_list;
};

std::unique_ptr<GGUFSaveBuilder> new_gguf_save_builder();
// Adds a tensor to the builder. The array is shallow-copied (shared buffer
// via mlx::core::array's refcounted internals), so this is cheap regardless
// of array size.
void gguf_builder_add_tensor(
    GGUFSaveBuilder& b, rust::Str name, const MlxArray& array);
// Adds an array-typed metadata entry. Same shallow-copy semantics as above.
void gguf_builder_add_array_meta(
    GGUFSaveBuilder& b, rust::Str key, const MlxArray& array);
void gguf_builder_add_string_meta(
    GGUFSaveBuilder& b, rust::Str key, rust::Str value);
void gguf_builder_begin_string_list_meta(GGUFSaveBuilder& b, rust::Str key);
void gguf_builder_push_string_list_meta(GGUFSaveBuilder& b, rust::Str value);
void gguf_builder_end_string_list_meta(GGUFSaveBuilder& b);

std::unique_ptr<GGUFLoadResult> load_gguf_file(rust::Str path);
void save_gguf_file(rust::Str path, const GGUFSaveBuilder& builder);

}  // namespace cxx_mlx
