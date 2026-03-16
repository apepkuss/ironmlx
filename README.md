# ironmlx

A Rust implementation of local LLM inference on Apple Silicon, powered by [MLX](https://github.com/ml-explore/mlx).

ironmlx is the Rust counterpart of [omlx](https://github.com/nicekid1/omlx), aiming to provide fast, native local inference with an OpenAI-compatible API.

[中文版](README_zh.md)

## Features

- MLX-native inference via `mlx-c` C API bindings
- 4-bit / 8-bit quantized model support (affine quantization)
- Jinja2-based chat template rendering (from `tokenizer_config.json`)
- OpenAI-compatible HTTP API (`/v1/chat/completions`, `/v1/completions`)
- SSE streaming responses
- Continuous batching engine with async request queue
- Automatic model download from HuggingFace

## Supported Model Architectures

ironmlx targets full compatibility with [mlx-lm](https://github.com/ml-explore/mlx-lm) (117 architectures). Current status:

| Category | Architecture | `model_type` | Status | Notes |
|----------|-------------|-------------|--------|-------|
| **Llama** | Llama 2/3 | `llama` | :white_check_mark: | Verified (SmolLM-135M-4bit) |
| | Llama 4 | `llama4` | :x: | |
| **Qwen** | Qwen3 | `qwen3` | :white_check_mark: | Verified (Qwen3-0.6B-4bit, 19.3 tok/s) |
| | Qwen3.5 | `qwen3_5` | :white_check_mark: | Verified (Qwen3.5-4B-4bit, 6.4 tok/s, text-only) |
| | Qwen2 | `qwen2` | :x: | |
| | Qwen3 MoE | `qwen3_moe` | :x: | |
| **Gemma** | Gemma 2 | `gemma2` | :x: | |
| | Gemma 3 | `gemma3` | :x: | |
| **DeepSeek** | DeepSeek V3 | `deepseek_v3` | :x: | |
| | DeepSeek V2 | `deepseek_v2` | :x: | |
| **Phi** | Phi 3 | `phi3` | :x: | |
| | Phi MoE | `phimoe` | :x: | |
| **Mistral** | Mixtral | `mixtral` | :x: | Remapped from `mistral` -> `llama` |

<details>
<summary><b>Full architecture list (117 total)</b></summary>

| Category | Architecture | `model_type` | Status |
|----------|-------------|-------------|--------|
| **Llama** | Llama 2/3 | `llama` | :white_check_mark: |
| | Llama 4 | `llama4` | :x: |
| | Mistral 3 | `mistral3` | :x: |
| **Qwen** | Qwen | `qwen` | :x: |
| | Qwen2 | `qwen2` | :x: |
| | Qwen2 MoE | `qwen2_moe` | :x: |
| | Qwen2 VL | `qwen2_vl` | :x: |
| | Qwen3 | `qwen3` | :white_check_mark: |
| | Qwen3.5 | `qwen3_5` | :white_check_mark: |
| | Qwen3.5 MoE | `qwen3_5_moe` | :x: |
| | Qwen3 MoE | `qwen3_moe` | :x: |
| | Qwen3 Next | `qwen3_next` | :x: |
| | Qwen3 VL | `qwen3_vl` | :x: |
| | Qwen3 VL MoE | `qwen3_vl_moe` | :x: |
| **Gemma** | Gemma | `gemma` | :x: |
| | Gemma 2 | `gemma2` | :x: |
| | Gemma 3 | `gemma3` | :x: |
| | Gemma 3 Text | `gemma3_text` | :x: |
| | Gemma 3n | `gemma3n` | :x: |
| | Recurrent Gemma | `recurrent_gemma` | :x: |
| **DeepSeek** | DeepSeek | `deepseek` | :x: |
| | DeepSeek V2 | `deepseek_v2` | :x: |
| | DeepSeek V3 | `deepseek_v3` | :x: |
| | DeepSeek V32 | `deepseek_v32` | :x: |
| **Phi** | Phi | `phi` | :x: |
| | Phi 3 | `phi3` | :x: |
| | Phi 3 Small | `phi3small` | :x: |
| | PhiXtral | `phixtral` | :x: |
| | Phi MoE | `phimoe` | :x: |
| **GLM** | GLM | `glm` | :x: |
| | GLM 4 | `glm4` | :x: |
| | GLM 4 MoE | `glm4_moe` | :x: |
| | GLM 4 MoE Lite | `glm4_moe_lite` | :x: |
| | GLM MoE DSA | `glm_moe_dsa` | :x: |
| **Mistral** | Mixtral | `mixtral` | :x: |
| | Ministral 3 | `ministral3` | :x: |
| | Pixtral | `pixtral` | :x: |
| **SSM/Recurrent** | Mamba | `mamba` | :x: |
| | Mamba 2 | `mamba2` | :x: |
| | RWKV 7 | `rwkv7` | :x: |
| | SSM | `ssm` | :x: |
| **Cohere** | Cohere | `cohere` | :x: |
| | Cohere 2 | `cohere2` | :x: |
| **IBM** | Granite | `granite` | :x: |
| | Granite MoE | `granitemoe` | :x: |
| | Granite MoE Hybrid | `granitemoehybrid` | :x: |
| **OLMo** | OLMo | `olmo` | :x: |
| | OLMo 2 | `olmo2` | :x: |
| | OLMo 3 | `olmo3` | :x: |
| | OLMoE | `olmoe` | :x: |
| **InternLM** | InternLM 2 | `internlm2` | :x: |
| | InternLM 3 | `internlm3` | :x: |
| **MiniCPM** | MiniCPM | `minicpm` | :x: |
| | MiniCPM 3 | `minicpm3` | :x: |
| **NVIDIA** | Nemotron | `nemotron` | :x: |
| | Nemotron NAS | `nemotron-nas` | :x: |
| | Nemotron H | `nemotron_h` | :x: |
| **GPT** | GPT-2 | `gpt2` | :x: |
| | GPT-NeoX | `gpt_neox` | :x: |
| | GPT BigCode | `gpt_bigcode` | :x: |
| | GPT OSS | `gpt_oss` | :x: |
| **StarCoder** | StarCoder 2 | `starcoder2` | :x: |
| **EXaONE** | EXaONE | `exaone` | :x: |
| | EXaONE 4 | `exaone4` | :x: |
| | EXaONE MoE | `exaone_moe` | :x: |
| **Hunyuan** | Hunyuan | `hunyuan` | :x: |
| | Hunyuan V1 Dense | `hunyuan_v1_dense` | :x: |
| **ERNIE** | ERNIE 4.5 | `ernie4_5` | :x: |
| | ERNIE 4.5 MoE | `ernie4_5_moe` | :x: |
| **Plamo** | Plamo | `plamo` | :x: |
| | Plamo 2 | `plamo2` | :x: |
| **Kimi** | Kimi K2.5 | `kimi_k25` | :x: |
| | Kimi Linear | `kimi_linear` | :x: |
| | Kimi VL | `kimi_vl` | :x: |
| **LFM** | LFM 2 | `lfm2` | :x: |
| | LFM 2 MoE | `lfm2_moe` | :x: |
| | LFM 2 VL | `lfm2-vl` | :x: |
| **Other** | AFM 7 | `afm7` | :x: |
| | AFMoE | `afmoe` | :x: |
| | Apertus | `apertus` | :x: |
| | Baichuan M1 | `baichuan_m1` | :x: |
| | Bailing MoE | `bailing_moe` | :x: |
| | Bailing MoE Linear | `bailing_moe_linear` | :x: |
| | BitNet | `bitnet` | :x: |
| | DBRX | `dbrx` | :x: |
| | DOTS 1 | `dots1` | :x: |
| | Falcon H1 | `falcon_h1` | :x: |
| | Helium | `helium` | :x: |
| | Jamba | `jamba` | :x: |
| | Klear | `Klear` | :x: |
| | Lille 130M | `lille-130m` | :x: |
| | Longcat Flash | `longcat_flash` | :x: |
| | Longcat Flash N-gram | `longcat_flash_ngram` | :x: |
| | MIMO | `mimo` | :x: |
| | MIMO V2 Flash | `mimo_v2_flash` | :x: |
| | Minimax | `minimax` | :x: |
| | MLA | `mla` | :x: |
| | NanoChat | `nanochat` | :x: |
| | OpenELM | `openelm` | :x: |
| | SEED OSS | `seed_oss` | :x: |
| | SmolLM 3 | `smollm3` | :x: |
| | Solar Open | `solar_open` | :x: |
| | StableLM | `stablelm` | :x: |
| | Step 3.5 | `step3p5` | :x: |
| | TeleChat 3 | `telechat3` | :x: |
| | YoutuAI LLM | `youtu_llm` | :x: |

</details>

> Models from the `mlx-community` HuggingFace organization in SafeTensors format are recommended.

## Workspace Structure

```
ironmlx/
├── mlx-sys/      # FFI bindings (auto-generated by bindgen from mlx-c headers)
├── mlx/          # ironmlx-core — safe Rust API (ops, nn, model, generate)
├── ironmlx/      # Binary crate — CLI + OpenAI-compatible HTTP server
└── vendor/mlx-c/ # MLX C API (git submodule)
```

## Quick Start

### Build

```bash
cargo build --release
```

### End-to-End Verification

Downloads a small quantized model and runs inference:

```bash
cargo run --release --example verify_e2e
# Default: Qwen3-0.6B-4bit (~320MB, auto-downloaded from HuggingFace)

# Or specify a model:
cargo run --release --example verify_e2e -- mlx-community/SmolLM-135M-Instruct-4bit
```

### Start Inference Server

```bash
cargo run --release --bin ironmlx -- --model /path/to/model --port 8080
```

### API Usage

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100,
    "stream": true
  }'
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Rust 1.85+ (edition 2024)
- CMake (for building MLX)

## License

Apache-2.0
