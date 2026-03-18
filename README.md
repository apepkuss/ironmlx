<p align="center">
  <img src="assets/logo.png" width="260" alt="ironmlx logo"/>
</p>

<h1 align="center">ironmlx</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Built_with-Rust-B7410E?logo=rust" alt="Rust"/>
  <img src="https://img.shields.io/badge/Platform-Apple_Silicon-000000?logo=apple" alt="Apple Silicon"/>
  <img src="https://img.shields.io/badge/License-Apache_2.0-blue" alt="License"/>

</p>

<p align="center">
  <strong>Your Mac has a GPU. Use it.</strong>
</p>

<p align="center">
  A local AI inference service for Apple Silicon.<br/>
  Run LLMs and vision models locally on your Mac — no cloud, no subscription, your data stays on your device.
</p>

<p align="center">
  <code>brew install ironmlx → launch → chat</code>
</p>

<p align="center">
  <a href="README.md">🇺🇸 English</a> · <a href="README_zh.md">🇨🇳 中文</a>
</p>

## Features

- **Multimodal (VLM)** — image + video understanding (Qwen3.5-VL)
- **Web Admin Panel** — Status, Models, Chat, Settings, Logs, Benchmark
- **macOS Menubar App** — native AppKit, welcome wizard, preferences
- **Automatic model download** from HuggingFace (repo ID as model path)
- **OpenAI-compatible API** — `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`
- **Anthropic-compatible API** — `/v1/messages` with streaming
- **Tool Calling** — function definitions, JSON tool_calls parsing
- **Rerank API** — `/v1/rerank` (Cohere-compatible)
- **Multi-model serving** — load/unload models at runtime via EnginePool
- **MLX-native inference**
- **Quantized models** — affine (2/3/4/6/8 bit), mxfp8, nvfp4

## Supported Models

ironmlx targets compatibility with [mlx-lm](https://github.com/ml-explore/mlx-lm) model architectures. Current status:

**Recent (last 6 months)**

| Architecture | `model_type` | Released | Status | Notes |
| ------------ | ------------ | -------- | ------ | ----- |
| Qwen3.5 | `qwen3_5` | 2026.02 | :white_check_mark: | Text + VLM (image/video) |
| Qwen3.5 MoE | `qwen3_5_moe` | 2026.02 | :white_check_mark: | 256 experts, top-8 routing |
| Mistral Small 4 | `mistral3` | 2026.03 | :x: | |
| Nemotron 3 Super | `nemotron_h` | 2026.03 | :x: | |
| DeepSeek V32 | `deepseek_v32` | 2026.01 | :x: | |
| Step 3.5 | `step3p5` | 2026.01 | :x: | |
| SmolLM 3 | `smollm3` | 2026.01 | :x: | |
| OLMo 3 | `olmo3` | 2025.12 | :x: | |
| Qwen3 | `qwen3` | 2025.10 | :white_check_mark: | Text + thinking mode |
| Qwen3 MoE | `qwen3_moe` | 2025.10 | :x: | |
| RWKV 7 | `rwkv7` | 2025.10 | :x: | |

<details>
<summary><b>Older architectures (6+ months)</b></summary>

| Architecture | `model_type` | Released | Status |
| ------------ | ------------ | -------- | ------ |
| Falcon H1 | `falcon_h1` | 2025.07 | :x: |
| Llama 4 | `llama4` | 2025.04 | :x: |
| Gemma 3 | `gemma3` | 2025.03 | :x: |
| Gemma 3n | `gemma3n` | 2025.03 | :x: |
| Phi 4 mini | `phi3` | 2025.02 | :x: |
| DeepSeek V3 | `deepseek_v3` | 2024.12 | :x: |
| Llama 2/3 | `llama` | 2023-2024 | :white_check_mark: |
| Gemma 2 | `gemma2` | 2024 | :x: |
| Mixtral | `mixtral` | 2023.12 | :x: |
| Mamba 2 | `mamba2` | 2024 | :x: |
| Cohere 2 | `cohere2` | 2024 | :x: |
| GLM 4 | `glm4` | 2024 | :x: |
| InternLM 3 | `internlm3` | 2024 | :x: |
| Granite | `granite` | 2024 | :x: |
| GPT-NeoX | `gpt_neox` | 2023 | :x: |
| StarCoder 2 | `starcoder2` | 2024 | :x: |

</details>

**Embedding & Reranker Models**

| Architecture | Type | Status | Representative Models |
| ------------ | ---- | ------ | --------------------- |
| BERT | Embedding / Reranker | :white_check_mark: | all-MiniLM-L6-v2, ms-marco-MiniLM |
| XLM-RoBERTa | Embedding / Reranker | :white_check_mark: | bge-m3, bge-reranker-v2-m3 |
| ModernBERT | Embedding | :white_check_mark: | ModernBERT-base |
| GTE | Embedding | :white_check_mark: | gte-large-en-v1.5 |
| Jina | Embedding / Reranker | :white_check_mark: | jina-embeddings-v5, jina-reranker-v2 |
| E5-Mistral | Embedding | :white_check_mark: | e5-mistral-7b-instruct |

> Recommended: models from `mlx-community` on HuggingFace in SafeTensors format.

## Install

**Homebrew (recommended):**

```bash
brew install ironmlx
```

**Build from source:**

```bash
git clone https://github.com/apepkuss/ironmlx.git
cd ironmlx
cargo build --release
```

## Getting Started

### Desktop App

Launch the menubar app — it handles everything automatically:

```bash
ironmlx-app
```

1. First launch opens a welcome wizard — enter a model ID (e.g. `mlx-community/Qwen3-0.6B-4bit`) and it downloads automatically
2. The ⚡ icon appears in your menubar
3. Click **Dashboard** to open the Web Admin Panel in your browser

That's it. Your local inference server is running.

### Web Admin Panel

Access at `http://localhost:8080/admin` — no command line needed:

- **Chat** — Talk to your model directly in the browser (streaming, Markdown, code highlighting)
- **Models** — Search HuggingFace, download, load/unload models with one click
- **Status** — Real-time memory usage and server health
- **Benchmark** — Measure inference speed (TTFT, tokens/sec)
- **Settings** — Configure port, sampling parameters, API keys
- **Logs** — View server activity

Supports dark/light/system theme and 4 languages (EN / 中文 / 日本語 / 한국어).

### Background Service

Run ironmlx as a background service that starts automatically:

```bash
brew services start ironmlx
```

## Using the Inference API

ironmlx exposes OpenAI-compatible and Anthropic-compatible APIs. Any app or tool that supports these APIs works out of the box — point it to `http://localhost:8080`.

**Works with:** ChatGPT clients, Cursor, Continue, Open Interpreter, LangChain, LlamaIndex, and any OpenAI SDK.

**Example — Python (OpenAI SDK):**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

**Example — JavaScript:**

```javascript
const response = await fetch("http://localhost:8080/v1/chat/completions", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    messages: [{ role: "user", content: "Hello!" }],
    max_tokens: 100,
    stream: false,
  }),
});
const data = await response.json();
console.log(data.choices[0].message.content);
```

### API Endpoints

| Endpoint | Description |
| -------- | ----------- |
| `POST /v1/chat/completions` | Chat completion (text, multimodal, streaming) |
| `POST /v1/completions` | Text completion |
| `POST /v1/messages` | Anthropic-compatible messages API |
| `POST /v1/embeddings` | Text embeddings (BERT, XLM-RoBERTa, etc.) |
| `POST /v1/rerank` | Document reranking (Cohere-compatible) |
| `GET /v1/models` | List loaded models |
| `POST /v1/models/load` | Load a model |
| `POST /v1/models/unload` | Unload a model |
| `GET /health` | Server health + memory stats |

### Multimodal (Images)

Send images as base64, URL, or local file path:

```python
response = client.chat.completions.create(
    model="default",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
            {"type": "text", "text": "What's in this image?"}
        ]
    }],
)
```

### Tool Calling

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
        }
    }],
)
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- For Homebrew install: just `brew install ironmlx`
- For building from source: Rust 1.85+, CMake
- Optional: ffmpeg (for video inference)

## License

Apache-2.0
