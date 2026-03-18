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
  <strong>你的 Mac，你的 AI。</strong>
</p>

<p align="center">
  Apple Silicon 本地 AI 推理服务。<br/>
  在你的 Mac 上本地运行大语言模型和视觉模型 — 无需云端、无需订阅、数据留在你的设备上，只需"安装、启动、开始对话"。
</p>

<p align="center">
  <a href="README.md">English</a> · <a href="README_zh.md">中文</a>
</p>

## 特性

- **多模态（VLM）** — 图片 + 视频理解（Qwen3.5-VL）
- **Web 管理面板** — 状态监控、模型管理、对话测试、性能基准
- **macOS 菜单栏应用** — 原生 AppKit，引导向导、偏好设置
- **自动模型下载** — 从 HuggingFace 自动下载（支持 repo ID 作为模型路径）
- **OpenAI 兼容 API** — `/v1/chat/completions`、`/v1/completions`、`/v1/models`
- **Anthropic 兼容 API** — `/v1/messages`，支持流式响应
- **Tool Calling** — 函数定义、JSON tool_calls 解析
- **多模型服务** — 运行时动态加载/卸载模型（EnginePool）
- **MLX 原生推理**
- **4-bit / 8-bit 量化模型** 支持（affine 量化）

## 支持的模型架构

ironmlx 目标兼容 [mlx-lm](https://github.com/ml-explore/mlx-lm) 模型架构。当前状态：

**近期（最近 6 个月）**

| 架构 | `model_type` | 发布时间 | 状态 | 备注 |
| ---- | ------------ | -------- | ---- | ---- |
| Qwen3.5 | `qwen3_5` | 2026.02 | :white_check_mark: | 文本 + VLM（图片/视频） |
| Qwen3.5 MoE | `qwen3_5_moe` | 2026.02 | :x: | |
| Mistral Small 4 | `mistral3` | 2026.03 | :x: | |
| Nemotron 3 Super | `nemotron_h` | 2026.03 | :x: | |
| DeepSeek V32 | `deepseek_v32` | 2026.01 | :x: | |
| Step 3.5 | `step3p5` | 2026.01 | :x: | |
| SmolLM 3 | `smollm3` | 2026.01 | :x: | |
| OLMo 3 | `olmo3` | 2025.12 | :x: | |
| Qwen3 | `qwen3` | 2025.10 | :white_check_mark: | 文本 + thinking 模式 |
| Qwen3 MoE | `qwen3_moe` | 2025.10 | :x: | |
| RWKV 7 | `rwkv7` | 2025.10 | :x: | |

<details>
<summary><b>更早的架构（6 个月以上）</b></summary>

| 架构 | `model_type` | 发布时间 | 状态 |
| ---- | ------------ | -------- | ---- |
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

> 推荐使用 `mlx-community` HuggingFace 组织中 SafeTensors 格式的模型。

## 安装

**Homebrew（推荐）：**

```bash
brew install ironmlx
```

**从源码构建：**

```bash
git clone https://github.com/apepkuss/ironmlx.git
cd ironmlx
cargo build --release
```

## 快速开始

### 桌面应用

启动菜单栏应用 — 一切自动完成：

```bash
ironmlx-app
```

1. 首次启动弹出引导向导 — 输入模型 ID（如 `mlx-community/Qwen3-0.6B-4bit`），自动下载
2. 菜单栏出现 ⚡ 图标
3. 点击 **Dashboard** 在浏览器中打开 Web 管理面板

完成。本地推理服务已在运行。

### Web 管理面板

访问 `http://localhost:8080/admin` — 无需命令行：

- **Chat** — 在浏览器中直接对话（流式响应、Markdown、代码高亮）
- **Models** — 搜索 HuggingFace、一键下载、加载/卸载模型
- **Status** — 实时内存使用和服务器状态
- **Benchmark** — 测量推理速度（TTFT、tokens/sec）
- **Settings** — 配置端口、采样参数、API 密钥
- **Logs** — 查看服务器日志

支持暗色/亮色/跟随系统主题，4 种语言（EN / 中文 / 日本語 / 한국어）。

### 后台服务

将 ironmlx 作为后台服务运行，开机自动启动：

```bash
brew services start ironmlx
```

## 使用推理 API

ironmlx 提供 OpenAI 兼容和 Anthropic 兼容的 API。任何支持这些 API 的应用或工具都可以直接使用 — 将地址指向 `http://localhost:8080`。

**兼容的工具：** ChatGPT 客户端、Cursor、Continue、Open Interpreter、LangChain、LlamaIndex，以及任何 OpenAI SDK。

**示例 — Python (OpenAI SDK)：**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "你好！"}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

**示例 — JavaScript：**

```javascript
const response = await fetch("http://localhost:8080/v1/chat/completions", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    messages: [{ role: "user", content: "你好！" }],
    max_tokens: 100,
    stream: false,
  }),
});
const data = await response.json();
console.log(data.choices[0].message.content);
```

### API 端点

| 端点 | 说明 |
| ---- | ---- |
| `POST /v1/chat/completions` | 对话补全（文本、多模态、流式） |
| `POST /v1/completions` | 文本补全 |
| `POST /v1/messages` | Anthropic 兼容消息 API |
| `GET /v1/models` | 列出已加载模型 |
| `POST /v1/models/load` | 加载模型 |
| `POST /v1/models/unload` | 卸载模型 |
| `GET /health` | 服务器状态 + 内存统计 |

### 多模态（图片）

支持 base64、URL 或本地文件路径：

```python
response = client.chat.completions.create(
    model="default",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
            {"type": "text", "text": "这张图片里是什么？"}
        ]
    }],
)
```

### Tool Calling

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "东京天气怎么样？"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取城市天气",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
        }
    }],
)
```

## 系统要求

- macOS + Apple Silicon（M1/M2/M3/M4）
- Homebrew 安装：只需 `brew install ironmlx`
- 从源码构建：Rust 1.85+、CMake
- 可选：ffmpeg（用于视频推理）

## 许可证

Apache-2.0
