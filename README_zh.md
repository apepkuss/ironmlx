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

| 架构 | `model_type` | QK Norm | 示例模型 | 已验证 |
|-----|-------------|---------|---------|--------|
| Llama | `llama` | 否 | Llama 3、SmolLM、Mistral、Yi | 是（SmolLM-135M-4bit） |
| Qwen3 | `qwen3` | 是 | Qwen3-0.6B、Qwen3-1.7B、Qwen3-8B | 是（Qwen3-0.6B-4bit，19.3 tok/s） |

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
