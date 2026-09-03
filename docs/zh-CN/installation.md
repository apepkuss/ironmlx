# 安装与构建

## 支持平台

IronMLX 0.1.0 仅支持 Apple Silicon arm64 与 macOS 26.2 或更高版本。
Intel Mac 和更早的 macOS 版本不在支持范围内。

## 当前分发状态

第三方依赖清单、Notices 与许可证原文已由 P0-8A 生成并纳入 App，但尚未完成
P0-8B 法律复核、CycloneDX SBOM 与分发授权，因此 GitHub public binary 发布仍被
硬门禁阻止。当前只支持从受信任的源码 checkout 构建用于本机开发验证。

## 模型权利边界

IronMLX 可以搜索和下载模型，但不拥有或重新授权模型权利。用户必须在上游模型
页面查阅许可证、gated access、用途和再分发限制，并自行确保使用合规。App、DMG
和 ZIP 不包含模型权重；完整边界说明见[模型权利边界](model-license-boundary.md)。

## 从源码构建 App

构建机需要完整 Xcode、CMake、Rust 1.94、`cargo-about 0.9.1`，以及可用的
macOS 26.2 SDK/Metal 工具链。以下命令会检出项目锁定的 MLX commit，并生成
自包含 Release App：

```bash
cargo install --locked --features cli --version 0.9.1 cargo-about
scripts/checkout-release-mlx.sh /tmp/ironmlx-mlx-source
MLX_SRC=/tmp/ironmlx-mlx-source scripts/build-app-bundle.sh
```

构建器会拒绝 dirty 或 commit 不匹配的 MLX checkout。成功后运行：

```bash
scripts/verify-app-bundle.sh dist/IronMLX.app
open dist/IronMLX.app
```

本地构建使用 ad-hoc 签名，未经 Developer ID 签名与 Apple 公证，不能作为正式
安装包对外分发。不要绕过 macOS 安全机制运行来源不明的构建。

## CLI 开发构建

若只调试后端，需要先准备项目锁定且启用 NAX Metal kernels 的 MLX Release
安装，然后导出 `MLX_DIR` 与 `MLX_METAL_PATH`：

```bash
export MLX_DIR=/path/to/validated/mlx-install
export MLX_METAL_PATH="$MLX_DIR/lib"
cargo build --release --bin ironmlx --bin iron-bench
target/release/ironmlx --version
```

## 构建并安装 MLX 依赖

MLX C++ 依赖必须以静态 arm64 库构建，并启用 Metal kernels。仓库提供的
辅助脚本会完成构建、安装、补齐传递库，并生成可 `source` 的环境文件：

```bash
MLX_SRC=/path/to/mlx-source \
MLX_PREFIX="$HOME/.local/mlx" \
scripts/setup-mlx.sh
source "$HOME/.local/mlx/mlx-env.sh"
```

如需手动构建，请使用相同的部署目标和静态库选项：

```bash
MLX_SRC=/path/to/mlx-source
MLX_PREFIX="$HOME/.local/mlx"

cd "$MLX_SRC"
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DMLX_BUILD_METAL=ON \
  -DMLX_METAL_JIT=OFF \
  -DMLX_BUILD_TESTS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=26.2 \
  -DCMAKE_INSTALL_PREFIX="$MLX_PREFIX"
cmake --build build --parallel "$(sysctl -n hw.ncpu)"
cmake --install build
```

MLX 的安装步骤不会导出私有的 GGUF 传递库。必须将它复制到安装前缀，
这样 `mlx` crate 的 GGUF 测试及其他 GGUF 使用方才能正确链接：

```bash
cp "$MLX_SRC/build/mlx/io/libgguflib.a" "$MLX_PREFIX/lib/"
```

生产 `ironmlx` 二进制不使用 GGUF 权重，但缺少该库会导致 GGUF 相关测试
因 `_gguf_*` 符号未定义而失败。`mlx-sys/build.rs` 会自动链接
`MLX_DIR/lib` 下的所有 `lib*.a`，无需额外链接器参数。

## MLX 编译期与运行期环境

每个编译或运行 IronMLX 的 shell、CI 作业和工具调用都必须显式设置 MLX
路径：

```bash
export MLX_ROOT="$HOME/.local/mlx"
export MLX_DIR="$MLX_ROOT"
export MLX_METAL_PATH="$MLX_ROOT/lib"
export DYLD_LIBRARY_PATH="$MLX_ROOT/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
export MACOSX_DEPLOYMENT_TARGET=26.2
export CMAKE_OSX_DEPLOYMENT_TARGET=26.2
```

虽然 MLX 以静态方式链接，但运行期仍会加载 `mlx.metallib`，因此
`MLX_METAL_PATH` 必须指向该文件所在目录。

## MLX 安装完整性检查

构建 IronMLX 前，请确认头文件、静态库和 Metal kernel 库都存在：

```bash
test -f "$MLX_DIR/include/mlx/array.h"
test -f "$MLX_DIR/lib/libmlx.a"
test -f "$MLX_DIR/lib/libgguflib.a"
test -f "$MLX_DIR/lib/mlx.metallib"
```

## 后端测试与本地服务

导出上述环境后，可运行完整 workspace 测试：

```bash
cargo build --release
cargo test --all-features --workspace
```

运行本地文本生成 smoke test：

```bash
MODEL="$HOME/.ironmlx/models/<org>/<model>"
./target/release/ironmlx generate \
  --model "$MODEL" \
  --prompt "请用一句话介绍 MoE 架构。" \
  --max-tokens 128 \
  --temperature 0 \
  --prefill-chunk-size 2048
```

启动本地服务：

```bash
./target/release/ironmlx serve \
  --model "$MODEL" \
  --host 127.0.0.1 \
  --port 8080 \
  --prefill-chunk-size 2048 \
  --b-max 1 \
  --max-cache-cap 32768
```

## MLX 故障排查

| 症状 | 原因 | 处理 |
|---|---|---|
| `MLX_DIR is not set` | 当前 shell 未导出构建路径。 | `source mlx-env.sh`，或设置上面的环境变量。 |
| `missing include/ or lib/` | `MLX_DIR` 指向 MLX build 目录，而非安装前缀。 | 将其指向 `MLX_PREFIX`。 |
| `Undefined symbols: _gguf_*` | `libgguflib.a` 未复制到安装前缀。 | 重复上面的复制步骤，或重新运行 `scripts/setup-mlx.sh`。 |
| `Failed to load the default metallib` | `MLX_METAL_PATH` 未设置或目录错误。 | 将其设置为包含 `mlx.metallib` 的目录。 |
