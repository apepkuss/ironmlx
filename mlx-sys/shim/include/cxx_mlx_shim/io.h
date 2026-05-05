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
