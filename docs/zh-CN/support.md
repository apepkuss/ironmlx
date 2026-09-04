# 支持

[English](../../SUPPORT.md)

## 支持平台

IronMLX 0.1.0 面向 Apple Silicon（`arm64`）和 macOS 26.2 或更高版本。Intel Mac、更早的 macOS，以及未经修改的第三方模型运行时不在支持范围内。

## 获取帮助

可在 [GitHub issue](https://github.com/apepkuss/ironmlx/issues) 中提供：IronMLX 版本或不可变 commit、macOS 与 Apple Silicon 型号、模型仓库和不可变 revision（不要上传权重）、具体操作、安全的错误文本、最小复现，以及问题来自源码构建还是开发预览。

不要公开凭据、prompt、工具参数、模型权重、私有 URL 或未脱敏日志。安全问题应按 [SECURITY.md](../../SECURITY.md) 私密报告，不要发布到公开 issue。

## 诊断与隐私

维护者需要运行时信息时，可以使用 Dashboard 的“导出诊断信息”。归档在本地生成，不上传数据，并具有限制和脱敏。分享前必须审阅 ZIP；不要替代性地上传原始日志或原始配置。详见[诊断信息导出](diagnostic-bundle.md)和[隐私边界](privacy.md)。

## 支持范围与预期

支持范围包括受支持平台上的文档化 App、本地运行时、loopback API、认证 LAN 模式、模型管理流程和发布产物。实验性模型、不支持的架构、上游服务中断和本地修改可能会协助调查，但不保证可用。不承诺固定响应或解决 SLA；可复现的安全问题通过 [SECURITY.md](../../SECURITY.md) 处理。
