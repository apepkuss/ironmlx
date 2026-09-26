# Supported models

[简体中文](zh-CN/supported-models.md)

IronMLX.app supports speech synthesis with the verified `mlx-community/IndexTTS-2.5-fp16` profile: automatic resource preparation, model loading, complete WAV responses and PCM streaming. See the [speech API](audio-speech-api.md) and [TTS model download and usage](tts-model-download.md).

IronMLX App also supports the typed decision model `aac6fef/laya-multilingual-mlx`: download, load/unload, restart recovery and TypeSafe-compatible System One API calls on the App service port. See the [Laya guide](laya-systemone-api.md).

IronMLX 0.2.0 supports the following text and vision-language models. Check the model family and features you need, then choose a version that fits your device's memory.
Versions, quantized files and templates within a family can differ in compatibility. This table is not a validation list for every model with a matching name.

## Model overview

“Conditional” requires compatible vision configuration and weights. Reasoning output and tool calling both require a supported native model template; the model name alone is insufficient.

| Model family | Text | Images | Reasoning output | Tool calling |
| --- | :---: | :---: | --- | --- |
| Qwen 3.5 / 3.6 Dense | Yes | Conditional | Yes | Yes |
| Qwen 3.8 Dense | Yes | Yes | Yes, enabled by default | Yes |
| Qwen 3.5 / 3.6 MoE | Yes | Conditional | Yes | Yes |
| Gemma 4 / Gemma 4 Unified | Yes | Conditional | Yes | Yes |
| GLM-4 MoE Lite | Yes | No | Yes | Yes |
| Llama GQA Dense (including compatible MiniCPM5-1B) | Yes | No | MiniCPM5 native template only | Native Llama 3.1/3.2 or MiniCPM5 tool template |
| MiniCPM-V 4.6 | Yes | Yes | Yes | Yes |
| DiffusionGemma | Yes | Yes | Yes | Yes |

All text and vision generation runtimes in the table support streaming HTTP responses. Image support means understanding images, not generating images or supporting video.
Embedding, reranker or ASR metadata in the model list does not imply runtime support. TTS support is limited to the verified IndexTTS 2.5 profile and its separate audio runtime.

## Validated model versions

The following Qwen3.8 validation scope is explicitly recorded on this page. It is not a complete recommendation list and does not establish validation of other versions in the family.

| Model | Recorded validation scope |
| --- | --- |
| `mlx-community/Qwen3.8-27B-4bit` | Text, images, reasoning output and tools; matching 4-bit MTP and DFlash2 text path |
| `mlx-community/Qwen3.8-27B-8bit` | Text, images, reasoning output and tools; matching 8-bit MTP and DFlash2 text path |

DFlash2 validation used `z-lab/Qwen3.8-27B-DFlash2`. Later repository snapshots still require compatibility checks; historical validation does not automatically apply to updated files.

## Generation acceleration and caching

| Feature | Applicable models | Main conditions |
| --- | --- | --- |
| MTP | Compatible Qwen Dense / MoE | Requires a matching MTP model; see [MTP documentation](mtp-server-api.md) for request-mode limits |
| Auxiliary drafter | Compatible Gemma 4 | Requires a matching assistant model; arbitrary pairings are not supported |
| DFlash2 | Qwen3.8 affine 4-bit / 8-bit targets | Requires a matching draft; text only |
| Prompt Lookup | Listed families except DiffusionGemma | Depends on runtime mode; cannot be combined with DFlash2 |
| KV cache | Listed families except DiffusionGemma | Does not imply support for every cache and acceleration combination |

DFlash2 supports greedy and exact sampling, concurrent requests and bounded batch widths. It cannot be combined with MTP, Prompt Lookup, KV quantization, paged/SSD prefix caches or Active KV offload. Its separate path can use its own in-memory prefix cache.
The DFlash2 draft cannot be loaded independently as a base model. See [DFlash2 documentation](dflash2-server-api.md) for configuration and full limitations.

## Quantization and device memory

Quantization support depends on the model loader. Not every model supports every format below.

In the App's **Model Management** list, quantized checkpoints display their
detected quantization format or bit width. Unquantized checkpoints display the
detected weight data type, such as `FP16` or `BF16`; the tooltip identifies the
checkpoint as unquantized and includes the normalized `dtype`. If neither model
metadata nor safetensors headers provide a usable value, the column displays
**Unknown**. These labels describe stored weights, not runtime compute precision.

| Weight format | Supported range |
| --- | --- |
| Unquantized | Data type must be supported by the model loader |
| Affine | 2/4/5/6/8-bit; group size 32/64/128 |
| OptiQ mixed-bit | 2/4/8-bit; group size 64; valid `optiq_metadata.json` required |
| MXFP4 | 4-bit; group size 32 |
| MXFP8 | 8-bit; group size 32 |

Context length, concurrent requests, caches and auxiliary models also affect memory use. Architecture support does not guarantee that a model fits your device. Check model files and available memory before downloading.

## Compatibility and usage limits

- Reasoning output requires an exact match between model type and native template. Qwen/GLM/MiniCPM use their supported thinking templates; Gemma uses its native thought channel. Similar markers do not automatically qualify.
- Tool calling means generating and validating calls and accepting results returned by the client. IronMLX does not execute tools. Llama 3.1/3.2 allows one native custom tool call per turn; its built-in-tool dialect is outside this scope.
- Responses is stateless and does not store responses or conversations. There are no independent reasoning-summary, refusal, audio-output or image-output channels.
- DiffusionGemma uses a separate generation path supporting only `max_tokens`, `temperature` and `seed`, without KV cache, Prompt Lookup or MTP.
- Public HTTP parameters depend on the protocol. Internal sampling parameters and model-profile settings are not automatically accepted as API request fields.

See the [API compatibility matrix](api-compatibility-matrix.md) and [API documentation](api.md) for request fields, structured outputs and runtime boundaries. See [model license boundaries](model-license-boundary.md) for model usage terms.

## Architecture identifiers

For compatibility troubleshooting, check `model_type` in the model configuration. Matching identifiers are necessary but insufficient: configuration, tokenizer, template, weight layout and quantization metadata must also be compatible. Download preflight or load-time checks may still reject incompatible files.

| Model family | `model_type` |
| --- | --- |
| Qwen 3.5 Dense; Qwen 3.6 / 3.8 Dense using the same architecture | `qwen3_5` |
| Qwen 3.5 / 3.6 MoE | `qwen3_5_moe` |
| Gemma 4 / Gemma 4 Unified | `gemma4`, `gemma4_unified` |
| GLM-4 MoE Lite | `glm4_moe_lite` |
| Llama GQA Dense / compatible MiniCPM5-1B | `llama` |
| MiniCPM-V 4.6 | `minicpmv4_6` |
| DiffusionGemma | `diffusion_gemma` |
