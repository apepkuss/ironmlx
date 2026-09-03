# 支持模型矩阵

以下矩阵来自 0.1.0 的生产加载分派。`model_type` 匹配只是必要条件；模型还必须
包含兼容的 `config.json`、tokenizer、chat template、权重布局和量化元数据。
下载前预检与加载时完整性校验仍可能拒绝不兼容 checkpoint。

| 模型族 | `model_type` | 文本 | 图片 | Responses/Messages reasoning | Chat/Responses/Messages tools | MTP/DFlash2/辅助 drafter | Prompt Lookup | KV cache |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen 3.5 Dense / 声明相同类型的 Qwen 3.6 Dense | `qwen3_5` | 是 | 否 | 是，需原生 `<think>` 模板 | 是，需原生工具模板 | 是 | 是 | 是 |
| Qwen 3.8 Dense（已验收 `mlx-community/Qwen3.8-27B-4bit` 与 `mlx-community/Qwen3.8-27B-8bit`） | `qwen3_5` | 是 | 是（图片；不含视频） | 是，默认开启；支持 `none`/`minimal`/`low`/`medium`/`high`/`xhigh`/`max` | 是，需 Qwen3.8 原生工具模板 | 是，已验收匹配的 4-bit/8-bit MTP 与 `z-lab/Qwen3.8-27B-DFlash2`；DFlash2 支持 affine 4-bit/8-bit target，且仅文本 | 是 | 是 |
| Qwen 3.5/3.6 MoE | `qwen3_5_moe` | 是 | 否 | 是，需原生 `<think>` 模板 | 是，需原生工具模板 | 是 | 是 | 是 |
| Gemma 4 / Gemma 4 Unified | `gemma4`, `gemma4_unified` | 是 | checkpoint 含 `vision_config` 时支持 | 是，需原生 `thought` channel | 是，需原生工具模板 | 是 | 是 | 是 |
| GLM-4 MoE Lite | `glm4_moe_lite` | 是 | 否 | 是，需原生 `<think>` 模板 | 是，需原生工具模板 | 否 | 是 | 是 |
| Llama GQA Dense（含兼容的 MiniCPM5-1B） | `llama` | 是 | 否 | 仅 MiniCPM5 原生模板 | 是，需 Llama 3.1/3.2 或 MiniCPM5 原生工具模板 | 否 | 是 | 是 |
| MiniCPM-V 4.6 | `minicpmv4_6` | 是 | 是 | 是，需原生 `<think>` 模板 | 是，需 MiniCPM-V 4.6 原生工具模板 | 否 | 是 | 是 |
| DiffusionGemma | `diffusion_gemma` | 是 | 是 | 是，需原生 `thought` channel | 是，需原生 Gemma 工具模板 | 否 | 否 | 否 |

所有运行时均支持流式 HTTP 响应。DiffusionGemma 使用 block-diffusion 生成路径，
只支持 `max_tokens`、`temperature` 与 `seed`；其他 causal 模型的内部 sampler 与
模型 profile 还支持 `top_p`、`top_k` 与 `repetition_penalty`。公开 HTTP 请求字段
按协议收口：Chat Completions 和 Responses 接受 `temperature`、`top_p`，Anthropic
Messages 额外接受 `top_k`；`repetition_penalty` 不属于这三套公开协议字段，
Chat/Responses 也不接受 `top_k`。内部模型 profile 仍可提供 `top_k` 与
`repetition_penalty` 默认值。

Chat/Responses/Messages tools 仅表示 OpenAI 或 Anthropic 协议的结构化函数调用生成与
历史回灌；IronMLX 不执行工具。Responses API 为无状态接口，不持久化 response 或
conversation。
Responses/Messages reasoning 仅在模型类型和 chat template 同时匹配精确原生契约时启用。
Responses 输出独立 typed item；Messages 输出原生 `thinking` block；两者均支持明文
历史回灌。Qwen3.8 默认保留历史 `reasoning_content`；Chat Completions 可用顶层
`reasoning_effort` 选择 `low`、`medium` 或 `xhigh`，也可通过
`chat_template_kwargs.preserve_thinking=false` 关闭历史思考保留。Responses 的
`minimal`/`low` 映射到 `low`，`medium` 映射到 `medium`，`high`/`xhigh`/`max`
映射到 `xhigh`；Anthropic `output_config.effort` 使用同样的三档收敛。
当前模型没有独立 reasoning summary、
refusal、音频输出或图片输出通道；这些能力不会从普通文本推断。
即使其他模型的模板包含相似标记，也不会被推断为支持。
Qwen3.8 DFlash2 使用独立 CLI/Server actor，支持 Greedy、精确 sampling、
`max-sequences>1` 的请求级并发和有安全宽度上限的 `B=N` tensor batching；它不与
MTP、Prompt Lookup、KV quantization、paged/SSD prefix cache 或 active KV offload
混用。DFlash2 独立路径可使用自己的内存 prefix cache。App 通过结构兼容 matcher、
独立 actor 重启、失败回滚、Tensor Batch 上限配置、Dashboard 运行指标和诊断快照
提供同一执行路径；draft 不作为 base model 独立加载。
完整边界见 [`dflash2-server-api.md`](../dflash2-server-api.md)。
Llama 3.1/3.2 的原生自定义函数协议只允许每个 assistant turn 产生一个工具调用；
其独立的 built-in tool / `<|python_tag|>` 协议不属于 OpenAI `tools` 支持范围。
MiniCPM-V 4.6 与 MiniCPM5 使用不同的原生 XML 工具协议；两者均支持多调用，
MiniCPM5 对含 `<`、`&` 或换行的字符串参数使用 CDATA。
全部工具方言均支持非 strict 工具中嵌套的动态 object 属性（`additionalProperties`
为 `true` 或受支持 Schema），以及字符串、数组和数值边界关键字；顶层工具参数对象
仍必须封闭，`strict:true` 仍要求递归 `additionalProperties:false` 和完整
`required`。Gemma 会在提示词、约束解码和历史调用中透明地把动态 object 投影为确定性
键值条目，并在响应阶段恢复为原始 object；HTTP 请求与响应始终保持客户端提交的 Schema
和参数形状。IronMLX 只生成并验证函数调用，不执行函数或外部工具。

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
