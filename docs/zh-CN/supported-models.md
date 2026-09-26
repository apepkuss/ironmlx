# 支持的模型

[English](../supported-models.md)

IronMLX.app 支持经过验证的 `mlx-community/IndexTTS-2.5-fp16` 资源配置，包括自动准备资源、加载模型、完整 WAV 与 PCM 流式输出。详见 [TTS 模型下载与使用](tts-model-download.md)及[语音 API](audio-speech-api.md)。

IronMLX App 还支持 `aac6fef/laya-multilingual-mlx` 决策模型：下载、加载与卸载、重启恢复，以及通过 App 服务端口调用兼容 TypeSafe 的 System One API。详见 [Laya 使用指南](laya-systemone-api.md)。

IronMLX 0.2.0 支持以下文本和视觉语言模型。请先确认模型族和所需功能，再选择适合设备内存的具体版本。
同一模型族的不同版本、量化文件和模板可能存在兼容性差异；下表不是所有同名模型的验证清单。

## 模型支持概览

“有条件”表示需要兼容的视觉配置和权重。“推理内容”和“工具调用”均要求模型使用受支持的原生模板，不能仅凭模型名称判断。

| 模型族 | 文本 | 图片 | 推理内容 | 工具调用 |
| --- | :---: | :---: | --- | --- |
| Qwen 3.5 / 3.6 Dense | 支持 | 有条件 | 支持 | 支持 |
| Qwen 3.8 Dense | 支持 | 支持 | 支持，默认开启 | 支持 |
| Qwen 3.5 / 3.6 MoE | 支持 | 有条件 | 支持 | 支持 |
| Gemma 4 / Gemma 4 Unified | 支持 | 有条件 | 支持 | 支持 |
| GLM-4 MoE Lite | 支持 | 不支持 | 支持 | 支持 |
| Llama GQA Dense（含兼容的 MiniCPM5-1B） | 支持 | 不支持 | 仅 MiniCPM5 原生模板 | Llama 3.1/3.2 或 MiniCPM5 原生工具模板 |
| MiniCPM-V 4.6 | 支持 | 支持 | 支持 | 支持 |
| DiffusionGemma | 支持 | 支持 | 支持 | 支持 |

表中的文本和视觉生成运行时均支持 HTTP 流式响应。图片能力指理解图片，不代表生成图片或支持视频。
模型列表中出现 embedding、reranker 或 ASR 信息不代表运行时受支持。TTS 仅支持已验证的 IndexTTS 2.5 配置及其独立音频运行时。

## 已验证的具体模型

以下为本页明确记录的 Qwen3.8 验证范围，不是完整模型推荐列表，也不代表同系列其他版本已通过验证。

| 模型 | 已记录的验证范围 |
| --- | --- |
| `mlx-community/Qwen3.8-27B-4bit` | 文本、图片、推理内容、工具调用；匹配的 4-bit MTP 和 DFlash2 文本路径 |
| `mlx-community/Qwen3.8-27B-8bit` | 文本、图片、推理内容、工具调用；匹配的 8-bit MTP 和 DFlash2 文本路径 |

上述 DFlash2 验证使用 `z-lab/Qwen3.8-27B-DFlash2`。模型仓库后续更新的快照仍需兼容性检查，不能直接沿用历史验证结论。

## 生成加速与缓存

| 功能 | 适用范围 | 主要条件 |
| --- | --- | --- |
| MTP | 兼容的 Qwen Dense / MoE | 需要匹配的 MTP 模型；请求模式限制见 [MTP 说明](mtp-server-api.md) |
| 辅助 drafter | 兼容的 Gemma 4 | 需要匹配的 assistant 模型，不能任意搭配 |
| DFlash2 | Qwen3.8 affine 4-bit / 8-bit target | 需要匹配的 draft；仅文本 |
| Prompt Lookup | 表中除 DiffusionGemma 外的模型族 | 取决于运行模式，不能与 DFlash2 同时使用 |
| KV cache | 表中除 DiffusionGemma 外的模型族 | 不代表每种缓存与加速组合均可用 |

DFlash2 支持贪心和精确采样、请求并发及有宽度上限的批处理。它不能与 MTP、Prompt Lookup、KV 量化、分页/SSD 前缀缓存或 Active KV offload 混用；独立路径可使用自己的内存前缀缓存。
DFlash2 draft 不能作为主模型独立加载。完整配置与限制见 [DFlash2 说明](dflash2-server-api.md)。

## 量化与设备内存

量化支持取决于具体模型加载器，不代表每个模型都支持下表中的所有格式。

在 App 的“模型管理”列表中，量化检查点显示检测到的量化格式或位数；未量化检查点
显示检测到的权重数据类型，例如 `FP16` 或 `BF16`，Tooltip 会说明“未量化”并给出
规范化后的 `dtype`。如果模型元数据和 safetensors 头都无法提供有效值，则显示
“未知”。这些标签描述的是存储权重，不是运行时计算精度。

| 权重格式 | 支持范围 |
| --- | --- |
| 未量化 | 数据类型必须受对应模型加载器支持 |
| Affine | 2/4/5/6/8-bit；group size 32/64/128 |
| OptiQ mixed-bit | 2/4/8-bit；group size 64；需要有效的 `optiq_metadata.json` |
| MXFP4 | 4-bit；group size 32 |
| MXFP8 | 8-bit；group size 32 |

内存需求还受到上下文长度、并发请求、缓存和辅助模型影响。架构受支持不代表当前设备一定能够加载；下载前应核对模型文件与可用内存。

## 兼容性与使用限制

- 推理内容依赖精确匹配的模型类型和原生模板。Qwen/GLM/MiniCPM 使用各自受支持的思考模板，Gemma 使用原生 thought channel；相似标记不会自动获得支持。
- 工具调用表示生成、验证调用参数和接收客户端返回的结果；IronMLX 不执行工具。Llama 3.1/3.2 每轮仅支持一个原生自定义工具调用，其 built-in tool 协议不在此范围内。
- Responses 为无状态接口，不保存 response 或 conversation。没有独立的推理摘要、拒绝、音频输出或图片输出通道。
- DiffusionGemma 使用独立生成路径，仅支持 `max_tokens`、`temperature` 和 `seed`，不支持 KV cache、Prompt Lookup 或 MTP。
- HTTP 可用参数由协议决定，不能将内部采样参数直接作为请求字段；模型 profile 可提供的参数也不等同于公开 API 字段。

接口字段、结构化输出和各执行模式的边界见 [API 兼容矩阵](api-compatibility-matrix.md)及 [API 文档](api.md)。模型授权条件见[模型许可边界](model-license-boundary.md)。

## 架构标识参考

排查模型兼容性时，可核对配置中的 `model_type`。标识匹配只是必要条件；配置、tokenizer、模板、权重布局和量化元数据也必须兼容。下载预检或加载校验仍可能拒绝不兼容文件。

| 模型族 | `model_type` |
| --- | --- |
| Qwen 3.5 Dense；采用相同架构的 Qwen 3.6 / 3.8 Dense | `qwen3_5` |
| Qwen 3.5 / 3.6 MoE | `qwen3_5_moe` |
| Gemma 4 / Gemma 4 Unified | `gemma4`、`gemma4_unified` |
| GLM-4 MoE Lite | `glm4_moe_lite` |
| Llama GQA Dense / 兼容的 MiniCPM5-1B | `llama` |
| MiniCPM-V 4.6 | `minicpmv4_6` |
| DiffusionGemma | `diffusion_gemma` |
