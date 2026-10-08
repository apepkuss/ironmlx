# 开发者指南

[English](../developer-guide.md)

从这里选择源码开发、API 接入或 CLI 服务配置入口。App 安装与模型管理见[用户指南](user-guide.md)。

[源码开发](#源码开发) · [API 接入](#api-接入) · [CLI 与高级配置](#cli-与高级配置) · [贡献与维护](#贡献与维护)

## 源码开发

先阅读[从源码构建](building-from-source.md)，了解平台要求、MLX 依赖、App 与 CLI
构建及本地运行环境。新增或修改模型时，对照[支持的模型](supported-models.md)。

### 项目结构与 crate 参考

| Crate | 职责 / 参考 |
| --- | --- |
| [`ironmlx-core`](https://github.com/apepkuss/ironmlx/tree/dev/ironmlx-core) | 共享 tensor 与权重基础能力 |
| [`ironmlx-lm`](https://github.com/apepkuss/ironmlx/tree/dev/ironmlx-lm) | 语言与视觉模型，包括[多模态向量编码器](text-embeddings.md) |
| [`ironmlx-image`](https://github.com/apepkuss/ironmlx/tree/dev/ironmlx-image) | 图片生成模型；见[图片 API](image-generation-api.md) |
| [`ironmlx-audio`](https://github.com/apepkuss/ironmlx/tree/dev/ironmlx-audio) | 音频模型；见[开发参考](../../ironmlx-audio/README.zh-CN.md) |
| [`ironmlx-decision`](https://github.com/apepkuss/ironmlx/tree/dev/ironmlx-decision) | 决策模型；见 [System One API](laya-systemone-api.md) |
| [`ironmlx-runtime`](https://github.com/apepkuss/ironmlx/tree/dev/ironmlx-runtime) | 执行、调度、生命周期和资源管理 |
| [`ironmlx`](https://github.com/apepkuss/ironmlx/tree/dev/ironmlx) | HTTP API 与 CLI |
| [`ironmlx-app`](https://github.com/apepkuss/ironmlx/tree/dev/ironmlx-app) | macOS App 与 Dashboard |
| [`iron-bench`](https://github.com/apepkuss/ironmlx/tree/dev/iron-bench) | 基准测试工具 |

Crate 名称链接指向源码目录。修改模型或协议时，追踪从 App 或 HTTP 请求，经
运行时执行到响应输出的完整链路。

## API 接入

从 [API 参考](api-reference.md)查看全部已公开文档的端点，并进入服务与管理、文本与视觉、向量、语音合成、图片生成或 System One 专题。

### API 快速开始

以下以 Responses 文本请求为例，需加载文本或视觉模型；其他能力使用API 参考中对应专题的请求示例。

先启动 App 并加载兼容模型。App 默认地址为 `http://127.0.0.1:9068`；独立 CLI
默认端口为 8080，以实际配置为准。同一 macOS 用户只能运行一个后端，启动独立
CLI 前应退出已有 App 后端。

检查服务并获取模型 ID：

```bash
curl http://127.0.0.1:9068/health
curl http://127.0.0.1:9068/healthz
curl http://127.0.0.1:9068/v1/models
```

`/health` 仅检查 HTTP 服务是否响应；`/healthz` 提供运行状态。App 的模型列表也
包含已注册但未加载的模型。选择可用模型，将下方 `your-model-id` 替换为实际 ID：

```bash
curl http://127.0.0.1:9068/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model": "your-model-id", "input": "Hello", "store": false, "max_output_tokens": 128, "stream": false}'
```

Responses 为无状态接口，每次请求发送完整历史。IronMLX 不保存对话，也不执行
工具；工具调用由客户端执行。

各协议字段与限制、请求示例、SSE 事件、工具、结构化输出及错误处理见
[文本与视觉 API](text-vision-api.md)。服务地址、局域网认证、健康检查、模型发现和管理接口见
[服务与管理 API](service-api.md)；网络配置与资源限制见[安全边界](security-boundary.md)。

## CLI 与高级配置

| 任务 | 指南 |
| --- | --- |
| 配置 DFlash2 | [CLI 生成、HTTP 服务及 App 设置](dflash2-server-api.md) |
| 配置 Qwen MTP | [匹配权重、启动参数与请求限制](mtp-server-api.md) |
| 提供多模型服务 | [CLI manifest、路由、加载与卸载](engine-pool.md) |
| 校准调度性能 | [可选的离线性能校准](scheduler-profile-v5.md) |

## 贡献与维护

| 任务 | 信息入口 |
| --- | --- |
| 验证修改 | [Rust、Swift 与 App Bundle 检查](building-from-source.md#验证修改) |
| 提交贡献 | [贡献条款与要求](contributing.md) |
| 排查问题 | [故障排查](troubleshooting.md)与[诊断导出](diagnostic-bundle.md) |
| 获取支持 | [问题反馈说明](support.md) |
| 报告漏洞 | [私密报告安全漏洞](security.md) |
| 了解发布 | [版本与渠道](versioning-and-releases.md) |
| 查看版本变更 | [0.2.0 发布说明](release-notes/0.2.0.md)及[0.1.0 发布说明](release-notes/0.1.0.md) |
