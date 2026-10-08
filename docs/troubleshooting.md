# Troubleshooting

[简体中文](zh-CN/troubleshooting.md)

## Start with the symptom

| Symptom | What to check |
| --- | --- |
| App will not start | Confirm Apple Silicon and macOS 26.4 or later. Record the macOS alert and App version. For a source build, use [the build checks](building-from-source.md). |
| Download interrupted | Retry the same model in Dashboard. Do not move unfinished `.partial` files manually. |
| Model refuses to load | Read the Dashboard error. Check [model loading](#model-loading), free disk space, available memory and download integrity. |
| API connection fails | Use the endpoint and port shown in Dashboard. Local mode accepts connections only from this Mac. See [API quick start](developer-guide.md#api-quick-start). |
| LAN returns 401 | Copy the current API key again and send it as a Bearer token. |
| LAN certificate error | Install or specify the exported CA and use the IP shown in the App. Follow [LAN setup](security-boundary.md). |
| Image rejected | Use JPEG/PNG/WebP base64 input and check [image limits](security-boundary.md). |
| Slow response or high memory use | Reduce context length and concurrency; unload unused models. Compare with the same model and settings. |
| Cache uses too much space | Adjust cache capacity in Dashboard. Stop the backend before manually removing cache files; see [data locations](storage-and-uninstall.md). |
| Backend keeps exiting | Open **Logs → Incident history** to inspect the cause and recovery result. Check memory, missing model files and configuration errors. |
| Update check fails | Check the network and retry later. See [Automatic updates](automatic-updates.md). |

## Model loading

Check the [supported models](supported-models.md) list and the exact Dashboard
error first. Confirm that the snapshot is complete and that disk space and
memory are sufficient. Context length, concurrency, caches and auxiliary models
increase memory use; reduce them or select a smaller model if needed.

### Weight formats

Support depends on the model loader; not every model supports every format
below. Use the formats listed for the specific model in the catalogue.

| Weight format | Supported range |
| --- | --- |
| Unquantized | Data type must be supported by the model loader |
| Affine | 2/4/5/6/8-bit; group size 32/64/128 |
| OptiQ mixed-bit | 2/4/8-bit; group size 64; valid `optiq_metadata.json` required |
| MXFP4 | 4-bit; group size 32 |
| MXFP8 | 8-bit; group size 32 |

### Architecture identifiers

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
| Qwen Image 2.1 | `qwen_image_2_1` |

## Logs and diagnostics

Change the log level in Settings when more detail is needed. The change is saved immediately without restarting the backend or saving other pending settings.
The level filter on the Logs page only changes which existing records are displayed. It does not generate extra logs.
INFO is the default; All includes TRACE. More detail cannot recover messages that were filtered out earlier.
If applying a level fails or the backend is busy, follow the displayed message and retry.

Use **Export diagnostic information** to save a local redacted ZIP. Review it before sharing it with a support request.
See [Diagnostic export](diagnostic-bundle.md) for included information.

## Recovery and memory pressure

Incident history supports filtering, viewing, clearing and JSON export. Clearing it does not stop the backend or delete ordinary logs and model settings.
A deliberate stop is not recorded as a crash. Repeated crashes can pause automatic recovery.
Memory protection can reject new requests, reclaim caches or unload unpinned idle models. Reduce context or concurrency before retrying; a pressure rejection does not by itself mean the model is unsupported.

## Report a reproducible problem

Include the App version, macOS and chip, model ID and revision, quantization, the operation and safe error text.
For performance problems, include context/output lengths, concurrency and whether the cache was cold or warm. Use several runs.
Do not publish private prompts or credentials. See [Support](../SUPPORT.md).

CLI users can use `RUST_LOG`; App backend launches use the saved App log level. API-level control is documented in [Log-level management](service-api.md#log-level-management-loopback-only).
