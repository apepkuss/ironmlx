# 安全漏洞报告

[English](../../SECURITY.md)

## 报告漏洞

请通过 [GitHub 私密安全公告](https://github.com/apepkuss/ironmlx/security/advisories/new)
报告疑似漏洞。不要创建公开 issue，也不要在 Pull Request 中公开利用细节。

在安全允许时提供：受影响版本或不可变 commit、macOS 与 Apple Silicon 型号、最小复现和影响说明，以及相关 endpoint、配置或发布产物信息。

绝不要提交模型权重、prompt、工具参数、API Key、HF token、Keychain 数据、Authorization 请求头、私有证书或未脱敏日志。使用 Dashboard 的“导出诊断信息”前必须先审阅归档；该功能仅本地生成，并排除请求正文和凭据。详见[诊断信息导出](diagnostic-bundle.md)。

## 范围与响应

报告范围包括 IronMLX App、Rust/MLX 运行时集成、HTTP API、模型下载完整性校验、发布脚本和随 App 打包的资源。上游模型仓库、模型权重、Hugging Face/ModelScope 服务、macOS 或第三方依赖的问题，也应提交给相应上游项目。

在 0.1.0 尚未正式发布期间，当前 `dev` 和最新发布候选版本是安全支持基线。维护者会在条件允许时确认并分级处理，优先处理高影响问题，并与报告者协商修复和披露时间；不承诺固定响应或修复 SLA。

安全边界、LAN 认证、图片限制和稳定错误码见[网络与图片安全边界](security-boundary.md)。
