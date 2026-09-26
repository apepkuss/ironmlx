# IronMLX 开发者指南

[English](../developer-guide.md)

本指南组织 IronMLX 的源码开发和发布工作。普通用户的安装与 API 使用请从[用户指南](user-guide.md)开始。

## 从源码构建

先阅读[从源码构建](building-from-source.md)。其中说明受支持的 Apple Silicon 平台、锁定的 MLX
源码检出、Rust 与 Xcode 要求、自包含 Release App 构建、运行时环境和本地服务冒烟测试。

## 理解项目结构

- `ironmlx-core` 与 `ironmlx-lm`：模型和 tensor 逻辑；
- `ironmlx-runtime`：执行、调度、生命周期和资源所有权；
- `ironmlx`：HTTP API 与 CLI 层；
- `ironmlx-app`：macOS App 与 Dashboard。

修改模型族或协议时，应追踪从 App 或 HTTP 请求，经运行时调度、模型执行到响应流式输出的完整链路。

## 验证修改

先运行与改动相关的检查；提交 Pull Request 前运行完整 workspace 门禁：

```bash
cargo fmt --all -- --check
cargo +nightly fmt --all -- --check
cargo +nightly clippy --locked --all-features --workspace -- -D warnings
cargo build --locked --release
cargo test --locked --all-features --workspace -- --test-threads=1
swift test --package-path ironmlx-app --configuration release --no-parallel
```

构建 App Bundle 后还应运行：

```bash
scripts/verify-app-bundle.sh dist/IronMLX.app
scripts/verify-model-distribution-boundary.sh dist/IronMLX.app
```

如果改动影响真实模型、协议、流式输出或 App 运行时，fixture、ignored test 或仅源码检查不能替代真实验收。

## 扩展 IronMLX

- 模型工作：阅读[支持模型矩阵](supported-models.md)，并检查对应模型加载器、原生模板、tokenizer、权重布局和量化元数据；
- API 工作：阅读 [API 参考](api-reference.md) 和 [API 兼容矩阵](api-compatibility-matrix.md)，包括 typed streaming（类型化流式事件）及客户端工具执行边界；
- 运行时工作：阅读 [Engine Pool（引擎池）](engine-pool.md)、[Scheduler Profile（调度器配置）](scheduler-profile-v5.md) 以及对应缓存或加速文档。

## 贡献与发布

- [参与开发](contributing.md)
- [用户支持](support.md)
- [安全漏洞报告](security.md)
- [版本与发布流程](versioning-and-releases.md)
- [稳定版发布流水线](stable-release-pipeline.md)
- [0.2.0 发布说明](release-notes/0.2.0.md)
- [0.1.0 发布说明](release-notes/0.1.0.md)

发布验收必须分别记录源码测试、Bundle 静态检查、签名与公证产物、Gatekeeper 检查和公开分发状态。
