# API 参考

[English](../api-reference.md) · [开发者指南](developer-guide.md#api-接入)

从这里按能力选择 API 专题，或通过端点索引查找具体接口。各专题提供请求字段、响应、错误与限制；首次接入示例见[开发者指南](developer-guide.md#api-快速开始)。

## API 专题

| 专题 | 内容 |
| --- | --- |
| [服务与管理 API](service-api.md) | 公共访问约定、健康检查、模型发现与管理接口 |
| [文本与视觉 API](text-vision-api.md) | Responses、Chat Completions、Messages；文本与图片理解、思考、工具、结构化输出 |
| [文本、图片与音频向量 API](text-embeddings.md) | EmbeddingGemma 2 向量、组合输入、输出维度与编码 |
| [语音合成 API](audio-speech-api.md) | 语音输出、声音配置与参考音频 |
| [图片生成 API](image-generation-api.md) | 文生图与单图条件编辑 |
| [System One API](laya-systemone-api.md) | Laya 的选择、评分与真假概率请求 |

## 端点索引

可用范围取决于服务模式及已配置的兼容模型。表格汇总公开文档中的端点；App 模型管理接口与 EnginePool 模型控制接口使用不同路径。

| 方法 | 端点 | 用途 | 可用范围 | 详细文档 |
| --- | --- | --- | --- | --- |
| `GET` | `/health` | HTTP 服务响应 | App / EnginePool / 文本 CLI / DFlash2 | [服务与管理 API](service-api.md#get-health) |
| `GET` | `/healthz` | 运行状态 | App / EnginePool / CLI | [服务与管理 API](service-api.md#get-healthz) |
| `GET` | `/v1/models` | 模型发现 | App / EnginePool / DFlash2 / System One CLI | [服务与管理 API](service-api.md#get-v1models) |
| `POST` | `/v1/responses` | Responses 推理 | App / EnginePool / CLI; 兼容模型 | [文本与视觉 API](text-vision-api.md#openai-responses-api) |
| `POST` | `/v1/chat/completions` | Chat Completions 推理 | App / EnginePool / CLI; 兼容模型 | [文本与视觉 API](text-vision-api.md#openai-chat-completions) |
| `POST` | `/v1/messages` | Messages 推理 | App / EnginePool / CLI; 兼容模型 | [文本与视觉 API](text-vision-api.md#anthropic-messages) |
| `POST` | `/v1/embeddings` | 文本、图片、音频与组合向量 | App / EnginePool; EmbeddingGemma 2 | [向量 API](text-embeddings.md#调用) |
| `POST` | `/v1/audio/speech` | 语音合成 | App / EnginePool; 音频模型 | [语音合成 API](audio-speech-api.md#请求) |
| `GET` | `/v1/audio/voices` | 查询可用声音 | App / EnginePool; 配置声音服务 | [语音合成 API](audio-speech-api.md) |
| `GET` | `/admin/api/audio/voices` | 查询全部声音（含禁用项） | App / EnginePool; 配置声音服务 | [语音合成 API](audio-speech-api.md) |
| `POST` | `/v1/audio/voices` | 创建声音 | App / EnginePool; 配置声音服务 | [语音合成 API](audio-speech-api.md) |
| `PATCH` | `/v1/audio/voices/{id}` | 更新声音 | App / EnginePool; 配置声音服务 | [语音合成 API](audio-speech-api.md) |
| `DELETE` | `/v1/audio/voices/{id}` | 删除声音 | App / EnginePool; 配置声音服务 | [语音合成 API](audio-speech-api.md) |
| `GET` | `/v1/audio/voices/{id}/preview` | 参考音频试听 | App / EnginePool; 配置声音服务 | [语音合成 API](audio-speech-api.md) |
| `POST` | `/v1/images/generations` | 文生图 | App / EnginePool; 图片模型 | [图片生成 API](image-generation-api.md) |
| `POST` | `/v1/images/edits` | 单图条件编辑 | App / EnginePool; 图片模型 | [图片生成 API](image-generation-api.md) |
| `POST` | `/v1/systemone` | 结构化决策 | App / EnginePool / System One CLI; 决策模型 | [System One API](laya-systemone-api.md#api-端点) |
| `POST` | `/v1/models/{model_id}/load` | 加载模型 | EnginePool | [EnginePool 控制接口](engine-pool.md) |
| `POST` | `/v1/models/{model_id}/unload` | 卸载模型 | EnginePool | [EnginePool 控制接口](engine-pool.md) |
| `GET` | `/admin/api/models/loaded` | 已加载模型与指标 | App | [服务与管理 API](service-api.md#管理接口) |
| `GET` | `/admin/api/log-level` | 查询日志级别 | 仅本机回环 | [服务与管理 API](service-api.md#日志级别管理仅本机) |
| `POST` | `/admin/api/log-level` | 修改日志级别 | 仅本机回环 | [服务与管理 API](service-api.md#日志级别管理仅本机) |
| `POST` | `/admin/api/models/register` | 注册本地模型资源 | App 模型管理 daemon | [音频资源注册示例](audio-speech-api.md#注册本地资源) |
| `POST` | `/admin/api/models/load` | 加载本地模型 | App 模型管理 daemon | [音频资源注册示例](audio-speech-api.md#注册本地资源) |

局域网认证见[服务与管理 API](service-api.md#服务地址认证与约定)。独立 System One 的认证及模型发现形状见其[独立 CLI 说明](laya-systemone-api.md#可选的独立-cli)。

## Agent 接入

按应用选择 [Hermes Agent](hermes-agent.md)、[oh-my-pi](oh-my-pi.md) 或 [DeepSeek Harness（DSH）](dsh.md) 配置指南。
