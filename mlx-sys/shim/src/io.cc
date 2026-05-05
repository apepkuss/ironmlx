#include "cxx_mlx_shim/io.h"

#include <stdexcept>

namespace cxx_mlx {

// ===== MemoryReader =====

MemoryReader::MemoryReader(std::vector<uint8_t> bytes) : data_(std::move(bytes)) {}

void MemoryReader::seek(int64_t off, std::ios_base::seekdir way) {
  int64_t new_pos;
  if (way == std::ios_base::beg) {
    new_pos = off;
  } else if (way == std::ios_base::cur) {
    new_pos = static_cast<int64_t>(pos_) + off;
  } else if (way == std::ios_base::end) {
    new_pos = static_cast<int64_t>(data_.size()) + off;
  } else {
    throw std::runtime_error("MemoryReader::seek: invalid seekdir");
  }
  if (new_pos < 0) {
    throw std::runtime_error("MemoryReader::seek: position would be negative");
  }
  if (static_cast<size_t>(new_pos) > data_.size()) {
    throw std::runtime_error("MemoryReader::seek: position past end of buffer");
  }
  pos_ = static_cast<size_t>(new_pos);
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
  int64_t new_pos;
  if (way == std::ios_base::beg) {
    new_pos = off;
  } else if (way == std::ios_base::cur) {
    new_pos = static_cast<int64_t>(pos_) + off;
  } else if (way == std::ios_base::end) {
    new_pos = static_cast<int64_t>(data_.size()) + off;
  } else {
    throw std::runtime_error("MemoryWriter::seek: invalid seekdir");
  }
  if (new_pos < 0) {
    throw std::runtime_error("MemoryWriter::seek: position would be negative");
  }
  // Allow seeking past end; write() will resize.
  pos_ = static_cast<size_t>(new_pos);
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

// ===== SafetensorsLoadResult getters =====

rust::Vec<rust::String> safetensors_tensor_names(const SafetensorsLoadResult& r) {
  rust::Vec<rust::String> out;
  out.reserve(r.inner_.first.size());
  for (const auto& kv : r.inner_.first) {
    out.push_back(rust::String(kv.first));
  }
  return out;
}

std::unique_ptr<MlxArray> safetensors_take_tensor_by_name(
    SafetensorsLoadResult& r, rust::Str name) {
  auto it = r.inner_.first.find(std::string(name));
  if (it == r.inner_.first.end()) {
    throw std::runtime_error("safetensors tensor not found: " + std::string(name));
  }
  // Move array out, then erase entry so subsequent take with same name throws
  // (matches the "single-use" contract documented in the header).
  auto array_out = std::make_unique<MlxArray>(std::move(it->second));
  r.inner_.first.erase(it);
  return array_out;
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

// ===== GGUFLoadResult getters =====

rust::Vec<rust::String> gguf_tensor_names(const GGUFLoadResult& r) {
  rust::Vec<rust::String> out;
  out.reserve(r.inner_.first.size());
  for (const auto& kv : r.inner_.first) {
    out.push_back(rust::String(kv.first));
  }
  return out;
}

std::unique_ptr<MlxArray> gguf_take_tensor_by_name(GGUFLoadResult& r, rust::Str name) {
  auto it = r.inner_.first.find(std::string(name));
  if (it == r.inner_.first.end()) {
    throw std::runtime_error("gguf tensor not found: " + std::string(name));
  }
  // Move array out, then erase entry so subsequent take with same name throws.
  auto array_out = std::make_unique<MlxArray>(std::move(it->second));
  r.inner_.first.erase(it);
  return array_out;
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

std::unique_ptr<MlxArray> gguf_take_array_meta_by_name(GGUFLoadResult& r, rust::Str name) {
  auto it = r.inner_.second.find(std::string(name));
  if (it == r.inner_.second.end()) {
    throw std::runtime_error("gguf array meta not found: " + std::string(name));
  }
  if (!std::holds_alternative<mlx::core::array>(it->second)) {
    throw std::runtime_error("gguf metadata is not an array variant: " + std::string(name));
  }
  // Move array out, then erase entry so subsequent take with same name throws.
  auto array_out = std::make_unique<MlxArray>(
      std::move(std::get<mlx::core::array>(it->second)));
  r.inner_.second.erase(it);
  return array_out;
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

}  // namespace cxx_mlx
