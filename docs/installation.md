# 安装与构建

## 支持平台

IronMLX 0.1.0 仅支持 Apple Silicon arm64 与 macOS 26.2 或更高版本。
Intel Mac 和更早的 macOS 版本不在支持范围内。

## 当前分发状态

第三方依赖清单、Notices 与许可证原文已由 P0-8A 生成并纳入 App，但尚未完成
P0-8B 法律复核、CycloneDX SBOM 与分发授权，因此 GitHub public binary 发布仍被
硬门禁阻止。当前只支持从受信任的源码 checkout 构建用于本机开发验证。

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
