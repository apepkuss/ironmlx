# 支持模型矩阵

以下矩阵来自 0.1.0 的生产加载分派。`model_type` 匹配只是必要条件；模型还必须
包含兼容的 `config.json`、tokenizer、chat template、权重布局和量化元数据。
下载前预检与加载时完整性校验仍可能拒绝不兼容 checkpoint。

| 模型族 | `model_type` | 文本 | 图片 | Chat/Responses/Messages tools | MTP/辅助 drafter | Prompt Lookup | KV cache |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen 3.5 Dense / 声明相同类型的 Qwen 3.6 Dense | `qwen3_5` | 是 | 否 | 是，需原生工具模板 | 是 | 是 | 是 |
| Qwen 3.5/3.6 MoE | `qwen3_5_moe` | 是 | 否 | 是，需原生工具模板 | 是 | 是 | 是 |
| Gemma 4 / Gemma 4 Unified | `gemma4`, `gemma4_unified` | 是 | checkpoint 含 `vision_config` 时支持 | 是，需原生工具模板 | 是 | 是 | 是 |
| GLM-4 MoE Lite | `glm4_moe_lite` | 是 | 否 | 是，需原生工具模板 | 否 | 是 | 是 |
| Llama GQA Dense（含兼容的 MiniCPM5-1B） | `llama` | 是 | 否 | 是，需 Llama 3.1/3.2 或 MiniCPM5 原生工具模板 | 否 | 是 | 是 |
| MiniCPM-V 4.6 | `minicpmv4_6` | 是 | 是 | 是，需 MiniCPM-V 4.6 原生工具模板 | 否 | 是 | 是 |
| DiffusionGemma | `diffusion_gemma` | 是 | 是 | 是，需原生 Gemma 工具模板 | 否 | 否 | 否 |

所有运行时均支持流式 HTTP 响应。DiffusionGemma 使用 block-diffusion 生成路径，
只支持 `max_tokens`、`temperature` 与 `seed`；其他 causal 模型还支持 `top_p`、
`top_k` 与 `repetition_penalty`。

Chat/Responses/Messages tools 仅表示 OpenAI 或 Anthropic 协议的结构化函数调用生成与
历史回灌；IronMLX 不执行工具。Responses API 为无状态接口，不持久化 response 或
conversation。
即使其他模型的模板包含相似标记，也不会被推断为支持。
Llama 3.1/3.2 的原生自定义函数协议只允许每个 assistant turn 产生一个工具调用；
其独立的 built-in tool / `<|python_tag|>` 协议不属于 OpenAI `tools` 支持范围。
MiniCPM-V 4.6 与 MiniCPM5 使用不同的原生 XML 工具协议；两者均支持多调用，
MiniCPM5 对含 `<`、`&` 或换行的字符串参数使用 CDATA。

## 权重量化

| 模式 | 支持参数 |
| --- | --- |
| 未量化权重 | checkpoint dtype 必须能被对应模型加载器处理 |
| Affine | 2/4/5/6/8 bit；group size 32/64/128 |
| OptiQ mixed-bit | 2/4/8 bit；group size 64；需要有效 `optiq_metadata.json` |
| MXFP4 | 4 bit；group size 32 |
| MXFP8 | 8 bit；group size 32 |

App 的模型列表可能展示 embedding、reranker、ASR 或 TTS 元数据，但 0.1.0 的
服务加载路径只面向上表中的 LLM/VLM 生成模型。
