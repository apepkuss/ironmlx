# 图片生成 API

[English](../image-generation-api.md) · [API 参考](api-reference.md) · [服务与管理 API](service-api.md) · [支持的模型](supported-models.md)

IronMLX 通过以下 OpenAI-compatible 接口，为
`mlx-community/Qwen-Image-2.1-MLX-4bit` 提供文生图和单图条件编辑推理：

```text
POST /v1/images/generations
POST /v1/images/edits
```

推理运行时使用原生 Rust/MLX 实现，不依赖 Python 或 `mflux`。App 负责模型的下载、
扫描、加载、管理及模型推理；图片展示、蒙版制作、画布操作和生成后合成由上游应用负责。

## 请求

```bash
curl http://127.0.0.1:9068/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Qwen-Image-2.1-MLX-4bit",
    "prompt": "干净白色背景中央的一个小红色圆形",
    "n": 1,
    "size": "1024x1024",
    "response_format": "b64_json",
    "seed": 7,
    "inference_steps": 40
  }'
```

| 字段 | 支持范围 |
| --- | --- |
| `model` | 服务存在无歧义默认模型时可省略；否则用于指定已加载的图片生成模型 |
| `prompt` | 必填，不能为空 |
| `n` | 可选；仅支持 `1` |
| `size` | 可选，格式为 `WIDTHxHEIGHT`；默认 `1024x1024` |
| `response_format` | 可选；仅支持 `b64_json` |
| `quality`、`style`、`user` | 为兼容 OpenAI 请求形状而接受；不改变本地推理行为 |
| `seed` | IronMLX 扩展；可选无符号整数，用于可复现生成 |
| `inference_steps` | IronMLX 扩展；可选整数，默认 `40` |

宽高均须为 32 的倍数，且分别位于 256–2048；总像素不得超过 1,572,864，
`inference_steps` 范围为 2–200。

## 图片条件编辑请求

`POST /v1/images/edits` 接受 `multipart/form-data`，将一张条件图和提示词交给
Qwen Image 2.1，并返回新生成的图片：

```bash
curl http://127.0.0.1:9068/v1/images/edits \
  -F 'model=mlx-community/Qwen-Image-2.1-MLX-4bit' \
  -F 'prompt=把中央圆形从红色改成蓝色' \
  -F 'image=@condition.png;type=image/png' \
  -F 'size=1024x1024' \
  -F 'response_format=b64_json' \
  -F 'seed=11' \
  -F 'inference_steps=40'
```

编辑接口中的 `model`、`prompt`、`n`、`size`、`response_format`、`seed` 和
`inference_steps` 与文生图接口采用相同约束；`quality`、`user` 可以传入，但不改变
推理行为。必须且只能上传一张 JPEG、PNG 或 WebP 条件图：文件最大 10 MiB，任一边最长
8192 像素，总像素不超过 16,777,216。IronMLX 会校验声明的媒体类型与实际文件内容。
模型侧预处理保持宽高比，以约 1024² 像素为目标，并将宽高对齐到 32 的倍数。

接口会以 `unsupported_image_mask` 明确拒绝 `mask`：这里提供的是整图条件编辑，
不是带蒙版的局部重绘（inpainting）。

## 响应

响应采用 OpenAI Images 结构，每个 `b64_json` 都是完整 PNG 的 Base64 编码：

```json
{
  "created": 1790480000,
  "data": [
    { "b64_json": "iVBORw0KGgo..." }
  ]
}
```

请求进入有界的串行图片生成通道。队列已满时返回 OpenAI 风格的 503 错误，错误码为
`image_queue_full`，上游可稍后重试。提示词、数量、格式、尺寸或步数非法时返回
OpenAI 风格的 400 错误。

## 功能边界

- 不实现蒙版局部重绘和图片变体接口。
- 蒙版创建/管理、画布操作、合成和编辑器 UI 不属于 IronMLX 推理服务范围。
- 不提供 Anthropic 图片生成接口。
- Qwen Image 2.1 使用 Qwen Research License；下载或部署前应确认其非商业/研究用途条件。

## 开发者验证

`ironmlx/tests/qwen_image_http.rs` 是需要显式启用的真实模型验收测试。它以 32 GiB
总内存、16 GiB 单模型的软件限额启动 App 守护进程，动态注册并加载指定模型，通过
公开接口生成 PNG，再以该 PNG 发起图片条件编辑请求，验证返回 PNG 及协议边界，
最后卸载模型。

构建 `dist/IronMLX.app` 后，可先针对该 Bundle 运行 256×256、两步的快速生命周期检查：

```sh
IRONMLX_QWEN_IMAGE_MODEL_DIR=/path/to/Qwen-Image-2.1-MLX-4bit \
IRONMLX_QWEN_IMAGE_HTTP_BUNDLE="$PWD/dist/IronMLX.app" \
  cargo test --locked -p ironmlx --release --test qwen_image_http -- \
  --ignored --nocapture --test-threads=1
```

32 GiB 目标机资格验收使用同一个测试，但执行 API 默认负载（1024×1024、40 步），
并在加载模型前校验机器报告的物理内存：

```sh
IRONMLX_QWEN_IMAGE_MODEL_DIR=/path/to/Qwen-Image-2.1-MLX-4bit \
IRONMLX_QWEN_IMAGE_HTTP_BUNDLE="$PWD/dist/IronMLX.app" \
IRONMLX_QWEN_IMAGE_FULL_ACCEPTANCE=1 \
IRONMLX_QWEN_IMAGE_EXPECTED_MEMORY_GB=32 \
  cargo test --locked -p ironmlx --release --test qwen_image_http -- \
  --ignored --nocapture --test-threads=1
```

测试会输出物理内存、加载耗时、生成/编辑耗时、服务 RSS 和 PNG 大小。Bundle 模式只使用
Bundle 内的 helper 与 metallib，在子进程中清除 `MLX_*`、`DYLD_*` 覆盖，并在工作树
之外运行服务。失败时会保留临时服务日志并打印其目录。在更大内存机器上设置软件限额
只能作为补充证据，不能代替真实 32 GiB 机器资格验收。
