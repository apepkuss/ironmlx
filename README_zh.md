# ironmlx

基于 [MLX](https://github.com/ml-explore/mlx) 的 Rust 本地 LLM 推理引擎，专为 Apple Silicon 设计。

ironmlx 是 [omlx](https://github.com/nicekid1/omlx) 的 Rust 版本，目标是提供快速、原生的本地推理服务，兼容 OpenAI API。

[English](README.md)

## 特性

- 通过 `mlx-c` C API 绑定实现 MLX 原生推理
- 支持 4-bit / 8-bit 量化模型（affine 量化）
- 基于 Jinja2 的聊天模板渲染（从 `tokenizer_config.json` 加载）
- OpenAI 兼容 HTTP API（`/v1/chat/completions`、`/v1/completions`）
- SSE 流式响应
- 连续批处理引擎 + 异步请求队列
- 自动从 HuggingFace 下载模型

## 支持的模型架构

ironmlx 目标是全面兼容 [mlx-lm](https://github.com/ml-explore/mlx-lm)（共 117 种架构）。当前状态：

| 分类 | 架构 | `model_type` | 状态 | 备注 |
|------|------|-------------|------|------|
| **Llama** | Llama 2/3 | `llama` | :white_check_mark: | 已验证（SmolLM-135M-4bit） |
| | Llama 4 | `llama4` | :x: | |
| **Qwen** | Qwen3 | `qwen3` | :white_check_mark: | 已验证（Qwen3-0.6B-4bit，19.3 tok/s） |
| | Qwen3.5 | `qwen3_5` | :x: | |
| | Qwen2 | `qwen2` | :x: | |
| | Qwen3 MoE | `qwen3_moe` | :x: | |
| **Gemma** | Gemma 2 | `gemma2` | :x: | |
| | Gemma 3 | `gemma3` | :x: | |
| **DeepSeek** | DeepSeek V3 | `deepseek_v3` | :x: | |
| | DeepSeek V2 | `deepseek_v2` | :x: | |
| **Phi** | Phi 3 | `phi3` | :x: | |
| | Phi MoE | `phimoe` | :x: | |
| **Mistral** | Mixtral | `mixtral` | :x: | `mistral` 重映射至 `llama` |

<details>
<summary><b>完整架构列表（共 117 种）</b></summary>

| 分类 | 架构 | `model_type` | 状态 |
|------|------|-------------|------|
| **Llama** | Llama 2/3 | `llama` | :white_check_mark: |
| | Llama 4 | `llama4` | :x: |
| | Mistral 3 | `mistral3` | :x: |
| **Qwen** | Qwen | `qwen` | :x: |
| | Qwen2 | `qwen2` | :x: |
| | Qwen2 MoE | `qwen2_moe` | :x: |
| | Qwen2 VL | `qwen2_vl` | :x: |
| | Qwen3 | `qwen3` | :white_check_mark: |
| | Qwen3.5 | `qwen3_5` | :x: |
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
| **SSM/循环** | Mamba | `mamba` | :x: |
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
| **混元** | Hunyuan | `hunyuan` | :x: |
| | Hunyuan V1 Dense | `hunyuan_v1_dense` | :x: |
| **文心** | ERNIE 4.5 | `ernie4_5` | :x: |
| | ERNIE 4.5 MoE | `ernie4_5_moe` | :x: |
| **Plamo** | Plamo | `plamo` | :x: |
| | Plamo 2 | `plamo2` | :x: |
| **Kimi** | Kimi K2.5 | `kimi_k25` | :x: |
| | Kimi Linear | `kimi_linear` | :x: |
| | Kimi VL | `kimi_vl` | :x: |
| **LFM** | LFM 2 | `lfm2` | :x: |
| | LFM 2 MoE | `lfm2_moe` | :x: |
| | LFM 2 VL | `lfm2-vl` | :x: |
| **其他** | AFM 7 | `afm7` | :x: |
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

> 推荐使用 `mlx-community` HuggingFace 组织中 SafeTensors 格式的模型。

## 工作区结构

```
ironmlx/
├── mlx-sys/      # FFI 绑定层（bindgen 从 mlx-c 头文件自动生成）
├── mlx/          # ironmlx-core — 安全 Rust API（ops、nn、model、generate）
├── ironmlx/      # 二进制 crate — CLI + OpenAI 兼容 HTTP 服务
└── vendor/mlx-c/ # MLX C API（git 子模块）
```

## 快速开始

### 编译

```bash
cargo build --release
```

### 端到端验证

下载小型量化模型并运行推理：

```bash
cargo run --release --example verify_e2e
# 默认：Qwen3-0.6B-4bit（~320MB，自动从 HuggingFace 下载）

# 或指定模型：
cargo run --release --example verify_e2e -- mlx-community/SmolLM-135M-Instruct-4bit
```

### 启动推理服务

```bash
cargo run --release --bin ironmlx -- --model /path/to/model --port 8080
```

### API 调用

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "你好！"}],
    "max_tokens": 100,
    "stream": true
  }'
```

## 系统要求

- macOS + Apple Silicon（M1/M2/M3/M4）
- Rust 1.85+（edition 2024）
- CMake（用于编译 MLX）

## 许可证

Apache-2.0
