# MLX 构建、安装与环境配置（ironmlx）

本文档是 ironmlx 依赖的 MLX C++ 库的**安装真相来源**：如何构建、安装、补齐传递静态库，以及编译期/运行期所需的环境变量。`mlx-sys/build.rs` 通过 `MLX_DIR` 定位 MLX 安装前缀并静态链接。

适用：Apple Silicon macOS。MLX 源码约定见仓库内存 `reference_mlx_source`（本机为 `/Users/xin/workspace/iron-rivals/mlx`）。

---

## 1. 构建 MLX（静态库）

```bash
MLX_SRC=/Users/xin/workspace/iron-rivals/mlx     # MLX 源码
MLX_PREFIX="$HOME/.local/mlx"                     # 安装前缀（可自定；下文环境变量指向它）

cd "$MLX_SRC"
mkdir -p build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DMLX_BUILD_METAL=ON \
  -DMLX_METAL_JIT=OFF \
  -DMLX_BUILD_TESTS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF \
  -DCMAKE_INSTALL_PREFIX="$MLX_PREFIX"
make -j"$(sysctl -n hw.ncpu)"
make install
```

## 2. 补齐 `make install` 漏装的传递静态库（**关键**）

MLX 把 GGUF 支持编进一个**独立的私有静态库 `libgguflib.a`**（FetchContent 的 gguflib），在构建期链接进 MLX 目标，但 **`make install` 不导出它**（只导出 `libmlx.a`、`libjaccl.a`、`mlx.metallib` 和头文件，含 `mlx/io/gguf.h`）。`libmlx.a` 内含约 10 个未定义的 `_gguf_*` 引用，靠 `libgguflib.a` 提供。

`mlx-sys/build.rs` 会自动链接 `$MLX_DIR/lib` 下**所有** `lib*.a`（`libmlx.a` + 任意传递库）。因此必须把 `libgguflib.a` 补进安装前缀的 `lib/`，否则任何引用 GGUF 的代码会链接失败：

```bash
cp "$MLX_SRC/build/mlx/io/libgguflib.a" "$MLX_PREFIX/lib/"
```

> 不补会怎样：`cargo build --release`（ironmlx 生产二进制走 safetensors、不调用 gguf，`-dead_strip` 剔除 gguf 代码）**不受影响**；但 `mlx` crate 的 `p2c_io` 测试（`gguf_round_trip_*`）会链接失败：`Undefined symbols: _gguf_open / _gguf_close / _gguf_append_kv ...`。本机当前 3 个静态库齐全：`libmlx.a`、`libjaccl.a`、`libgguflib.a`（`libjaccl.a` 与 `mlx.metallib` 由 `make install` 正常导出，仅 `libgguflib.a` 需手动补）。

## 3. 环境变量

**编译期**（`cargo build` / `cargo test`）——`mlx-sys/build.rs` 需要 `$MLX_DIR/include` 与 `$MLX_DIR/lib`：

```bash
export MLX_ROOT="$MLX_PREFIX"          # 例：/tmp/ironmlx-perf-mlx-install-XXXX 或 $HOME/.local/mlx
export MLX_DIR="$MLX_ROOT"
```

**运行期**（CLI / serve / 跑测试）——MLX 本体是静态 `.a` 链接，但 **`mlx.metallib` 仍需运行期可被找到**（位于 `$MLX_PREFIX/lib/mlx.metallib`）：

```bash
export MLX_METAL_PATH="$MLX_ROOT/lib"
export DYLD_LIBRARY_PATH="$MLX_ROOT/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```

> 这些变量**不在** shell profile 里——每个新 shell（含 CI / 工具调用）都要显式 export，否则 `mlx-sys/build.rs` 在 `MLX_DIR` 未设时会 panic。

## 4. 健全性检查

```bash
ls "$MLX_DIR/include/mlx/array.h"      # 头文件
ls "$MLX_DIR/lib/libmlx.a"             # 主静态库
ls "$MLX_DIR/lib/libgguflib.a"         # 传递库（第 2 步补齐）
ls "$MLX_DIR/lib/mlx.metallib"         # 运行期 Metal kernel 库
```
以上全部存在，方可正确链接 + 运行。

## 5. 构建 / 测试 / 运行 ironmlx

```bash
# 一次性导出（编译 + 运行都需要）
export MLX_ROOT="$MLX_PREFIX"
export MLX_DIR="$MLX_ROOT"
export MLX_METAL_PATH="$MLX_ROOT/lib"
export DYLD_LIBRARY_PATH="$MLX_ROOT/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

cargo build --release
cargo test --all-features --workspace          # 全套测试（须先完成第 2 步补 libgguflib.a）
```

CLI 生成（text）：
```bash
MODEL="$HOME/.ironmlx/models/<org>/<model>"
./target/release/ironmlx generate \
  --model "$MODEL" --prompt "请用一句话介绍 MoE 架构。" \
  --max-tokens 128 --temperature 0 --prefill-chunk-size 2048
```

Serve：
```bash
./target/release/ironmlx serve \
  --model "$MODEL" --host 127.0.0.1 --port 8080 \
  --prefill-chunk-size 2048 --b-max 1 --max-cache-cap 32768
```

## 6. 故障排查

| 症状 | 原因 | 处理 |
|---|---|---|
| `mlx-sys/build.rs` panic：`MLX_DIR is not set` | 当前 shell 未 export | 见第 3 节 |
| `MLX_DIR=… does not look like an MLX install prefix (missing include/ or lib/)` | `MLX_DIR` 指错目录（应为安装前缀，非 build 目录） | 指向第 1 节的 `$MLX_PREFIX` |
| 链接报 `Undefined symbols: _gguf_*` | `libgguflib.a` 未补进 `$MLX_DIR/lib` | 见第 2 节 |
| 运行期找不到 `mlx.metallib` | `MLX_METAL_PATH` 未设或指错 | 见第 3 节 |
