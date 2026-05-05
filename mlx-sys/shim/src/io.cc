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

}  // namespace cxx_mlx
