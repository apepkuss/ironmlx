# Supported model matrix

This matrix reflects the 0.1.0 production loading dispatch. Matching
`model_type` is necessary but not sufficient: a checkpoint must also contain a
compatible `config.json`, tokenizer, chat template, weight layout, and
quantization metadata. Download preflight and load-time integrity checks may
still reject an incompatible checkpoint.

| Model family | `model_type` | Text | Images | Responses/Messages reasoning | Chat/Responses/Messages tools | MTP/DFlash2/drafter | Prompt Lookup | KV cache |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen 3.5 Dense / same-type Qwen 3.6 Dense | `qwen3_5` | Yes | No | Yes, native `<think>` template required | Yes, native tool template required | Yes | Yes | Yes |
| Qwen 3.8 Dense (`mlx-community/Qwen3.8-27B-4bit` and `mlx-community/Qwen3.8-27B-8bit` accepted) | `qwen3_5` | Yes | Yes (images; no video) | Yes, enabled by default; `none` through `max` | Yes, native Qwen3.8 tool template required | Yes; accepted matching 4/8-bit MTP and `z-lab/Qwen3.8-27B-DFlash2`; affine 4/8-bit target, text only | Yes | Yes |
| Qwen 3.5/3.6 MoE | `qwen3_5_moe` | Yes | No | Yes, native `<think>` template required | Yes, native tool template required | Yes | Yes | Yes |
| Gemma 4 / Gemma 4 Unified | `gemma4`, `gemma4_unified` | Yes | Supported when checkpoint includes `vision_config` | Yes, native `thought` channel required | Yes, native tool template required | Yes | Yes | Yes |
| GLM-4 MoE Lite | `glm4_moe_lite` | Yes | No | Yes, native `<think>` template required | Yes, native tool template required | No | Yes | Yes |
| Llama GQA Dense (including compatible MiniCPM5-1B) | `llama` | Yes | No | MiniCPM5 native template only | Yes, native Llama 3.1/3.2 or MiniCPM5 tool template required | No | Yes | Yes |
| MiniCPM-V 4.6 | `minicpmv4_6` | Yes | Yes | Yes, native `<think>` template required | Yes, native MiniCPM-V 4.6 tool template required | No | Yes | Yes |
| DiffusionGemma | `diffusion_gemma` | Yes | Yes | Yes, native `thought` channel required | Yes, native Gemma tool template required | No | No | No |

All runtimes support streaming HTTP responses. DiffusionGemma uses block
diffusion and supports only `max_tokens`, `temperature`, and `seed`. Other
causal models also support `top_p`, `top_k`, and `repetition_penalty` in their
internal sampler and model profile. Public protocol fields are narrower:
Chat Completions and Responses accept `temperature` and `top_p`, while
Anthropic Messages additionally accepts `top_k`; `repetition_penalty` is not a
public field and Chat/Responses do not accept `top_k`.

Chat/Responses/Messages tools mean structured function-call generation and
history replay. IronMLX generates and validates calls but never executes tools.
Responses is stateless and does not persist responses or conversations.
Reasoning is enabled only when both model type and chat template match the exact
native contract. There is no independent reasoning summary, refusal, audio
output, or image output channel.

Qwen3.8 DFlash2 runs in a separate CLI/server actor. It supports greedy and
exact sampling, request concurrency with `max-sequences > 1`, and bounded
`B=N` tensor batching. It cannot be combined with MTP, Prompt Lookup, KV
quantization, paged/SSD prefix cache, or active-KV offload. See the [DFlash2
server API](dflash2-server-api.md) for the complete boundary.

Llama 3.1/3.2 native custom function protocol permits one tool call per
assistant turn; its built-in-tool / `<|python_tag|>` dialect is outside OpenAI
`tools`. MiniCPM-V 4.6 and MiniCPM5 use different native XML tool dialects.
Tool schemas remain closed at the top level; `strict:true` requires recursive
`additionalProperties:false` and complete `required` fields.

## Weight quantization

| Mode | Supported parameters |
| --- | --- |
| Unquantized | Checkpoint dtype must be supported by the model loader |
| Affine | 2/4/5/6/8-bit; group size 32/64/128 |
| OptiQ mixed-bit | 2/4/8-bit; group size 64; valid `optiq_metadata.json` required |
| MXFP4 | 4-bit; group size 32 |
| MXFP8 | 8-bit; group size 32 |

The App model list may show embedding, reranker, ASR, or TTS metadata, but the
0.1.0 service loading path targets only the LLM/VLM generation models above.
