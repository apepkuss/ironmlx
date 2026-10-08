# 文本、图片与音频向量 API

[English](../text-embeddings.md)

支持 `mlx-community/embeddinggemma-2-bf16` 与
`mlx-community/embeddinggemma-2-4bit`（MLX affine、4 bit、group size 64）的文本、视觉和音频编码器。
当前不支持视频输入。App 可以下载、加载和卸载两个版本，已有经过验证的
“仅下载”快照会重新判断兼容性，无需重新下载权重。

编码器在 `ironmlx-lm::models::embedding_gemma2` 中实现，复用现有浮点和量化层组件，
使用独立的双向注意力和 PLE 计算图。affine4 checkpoint 的文本编码器使用 4bit，
视觉编码器、图片投影和音频编码器保留 BF16 权重，音频投影采用 affine4。音频编码器以 FP32 计算中间结果，随后投影到文本编码器的 BF16 激活。`ironmlx-runtime` 管理请求队列、内存与模型租约，
`ironmlx` 在 App 服务和 engine-pool 服务中提供 HTTP 接口。

## 调用

`POST /v1/embeddings` 必须指定已注册的 `model`，`input` 可以是一个非空字符串或
一个包含有序内容的对象，或包含 1–32 个字符串/内容对象的数组。每条最多 8192 个 token，包含特殊 token 和任务前缀；
超过上限会报错，不会自动截断。

`dimensions` 默认 768，可选 128、256、512、768；降维后重新归一化。
`encoding_format` 支持 `float`（默认）和 `base64`（小端 float32），不支持 token ID 和视频输入。内容对象是 IronMLX 扩展；原有字符串形式和返回格式保持 OpenAI-compatible。

```sh
curl http://127.0.0.1:9068/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/embeddinggemma-2-4bit","input":["task: search result | query: 苹果公司的总部在哪里？","title: none | text: 苹果公司总部位于美国加利福尼亚州库比蒂诺。"],"dimensions":256}'
```

## 图片与图文组合

一个 `content` 数组中的内容按顺序编码，生成**一个组合向量**。文本使用
`{"type":"text","text":"..."}`，图片使用
`{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}`。
仅图片输入可以省略文本。`input` 数组中的每条样本生成一个向量，数组可以混合字符串
和内容对象；每条样本可以交错包含多段文字和多张图片。

```python
import base64
import json
import urllib.request
from pathlib import Path

image = base64.b64encode(Path("photo.png").read_bytes()).decode("ascii")
payload = {
    "model": "mlx-community/embeddinggemma-2-4bit",
    "input": {"content": [
        {"type": "text", "text": "这张照片的描述。"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
    ]},
    "dimensions": 768,
}
request = urllib.request.Request(
    "http://127.0.0.1:9068/v1/embeddings",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
print(json.load(urllib.request.urlopen(request)))
```

图片必须通过 JPEG、PNG 或 WebP 数据 URL 提交，不支持远程 URL 和服务端文件路径。
每个请求最多 8 张图片，每张压缩文件最多 10 MiB，base64 解码后的图片文件字节总计
最多 24 MiB，JSON 请求体最多 32 MiB。图片边长最多 8192 像素，每张最多
16,777,216 像素，每个请求累计最多 33,554,432 像素。每条样本最多 64 段内容和
128 KiB 文本。请求体或图片预算超限返回 413，图片格式无效返回 400。

模型适配器负责 RGB 转换、保持宽高比的 Pillow-compatible 双三次缩放、1/255 像素
缩放，以及 patch 补齐和 attention mask。默认图片预算为 280 个 soft token，实际
数量由宽高比决定。适配器插入图片边界 token，将图片占位的 embedding 替换为视觉特征。
用户提交原图和文本即可，无需手动缩放图片或插入媒体 token。8192 token 上限及
`usage` 包含文本、特殊 token 和实际图片 soft token。图片和图文支持全部四种输出维度；
按[官方建议](https://huggingface.co/google/embeddinggemma-2#3-matryoshka-dimension-truncation)，
图片检索优先使用 256–768 维，128 维主要适用于文本。

## 音频与文本／音频组合

在同一有序 `content` 数组中使用
`{"type":"input_audio","input_audio":{"data":"BASE64","format":"wav"}}`。
纯音频可省略文本，也可交错放入文本、音频和图片；每条样本仍生成一个向量。
`data` 是完整文件的标准 base64，不带 data URL 前缀，不接受远程 URL 或服务端文件路径。

```python
from pathlib import Path
import base64
import requests

audio = base64.b64encode(Path("recording.wav").read_bytes()).decode("ascii")
response = requests.post("http://127.0.0.1:9068/v1/embeddings", json={
    "model": "mlx-community/embeddinggemma-2-4bit",
    "input": {"content": [
        {"type": "text", "text": "Recorded speech: "},
        {"type": "input_audio", "input_audio": {"data": audio, "format": "wav"}},
    ]},
    "dimensions": 256,
})
response.raise_for_status()
vector = response.json()["data"][0]["embedding"]
```

支持 PCM16/24/32、float32 WAV，以及 FLAC 和 MP3，采样率 8–96 kHz、单声道或双声道。
浮点 PCM 必须有限且归一化到 [-1, 1]。单段必须大于 10 ms、最多 30 秒；每次请求最多
8 段，单个文件最多 16 MiB，base64 解码后的文件字节合计最多 24 MiB，重采样后的
音频总时长最多 60 秒。每段源 PCM 解码最多 48 MiB。原有 32 MiB JSON 请求体、
32 条样本、每条 64 个内容部分的限制仍适用。格式无效或不匹配返回 400，资源超限
返回 413。CPU 解码最多并发 4 个，忙时返回 429，且在获取 GPU 模型前完成。
超长音频会拒绝，不会静默截断。

IronMLX 将双声道混合为单声道，通过共享 sinc 重采样器转换到 16 kHz。模型适配器
使用 checkpoint 的半因果 320 样本周期 Hann 窗、160 样本步长、512 点 FFT 和
128 个 HTK 幅度 mel 滤波器，计算 `log(mel + 0.001)`。帧 mask 经过两次 stride-2
下采样，由有效输出帧决定实际音频 soft token 数量。适配器插入音频边界 token，
并用音频特征替换占位，不需要用户手工提取 mel 特征或插入媒体 token。
纯音频、文本／音频及混合媒体支持四种输出维度和两种返回编码。`usage` 包含文本、
边界及实际音频／图片 soft token。检索任务前缀仍由调用方显式提供，只作用于文本部分。

返回 OpenAI-compatible 的向量列表：`data` 中各项包含 `object: "embedding"`、
`index` 和浮点 `embedding`，顺序与输入一致。`usage.prompt_tokens` 与
`usage.total_tokens` 统计所有输入 token。

模型对有效 token（包括前缀）求均值并进行 L2 归一化。接口不会自动补任务前缀或
聊天模板，检索时请按[官方说明](https://huggingface.co/google/embeddinggemma-2)显式提供
query/document 前缀，索引和查询使用同一模型版本、维度与前缀规则。

同一模型的请求串行执行，批量输入逐条编码，以控制峰值内存并保持单条结果一致。
模型卸载后可按需自动加载。无效输入返回 400，队列满返回 429，服务不可用返回 503，
超时返回 504。客户端断开后，正在执行的 GPU 工作仍持有模型租约。
该模型不能用于聊天生成，也不接受采样或 KV Cache 参数。

## 运行状态

App【状态】页显示向量模型的队列活动和专用性能指标。完成／失败请求数自模型加载后累计；
延迟、输入吞吐和向量吞吐显示最近 60 秒的 P50，最多保留 4096 个样本。
延迟包含运行时排队与处理时间，不含 HTTP 媒体解码、模型加载和响应序列化。
吞吐按处理时间计算，不含排队；输入 token 包含图片和音频 soft token，批次中每条输入生成一个向量。
尚无样本或近期样本过期时显示横线。卸载再加载模型后，累计计数重置。
失败计数覆盖运行时拒绝或执行失败、队列满、超时和排队期间取消；进入运行时前的 HTTP
校验失败不计入。超时与后台任务结束发生竞争时，只记录一次结果。
`/healthz` 的模型条目及 `/admin/api/models/loaded` 通过 `embedding_metrics` 返回这些指标。

## 验证

两个精度版本分别与上游同精度实现对齐，验证范围是文本、图片及音频数值、输入边界、基础跨模态检索和模型生命周期，
不代表完整检索质量评测。复现命令与环境变量详见[英文验证说明](../text-embeddings.md#qualification)。

音频参考固定 Transformers `cb33194ad6152bd9fad6305378d92db385dd7b32` 与 NumPy 1.26.4，
覆盖特征 mask、语音检索、静音、短音频及有序文本／音频。音频向量要求余弦相似度大于
0.9995、最大分量误差小于 0.005；FP32 编码器的浮点 kernel 舍入可能跨越 BF16 激活边界。
文本和图片的原有数值容差保持不变。
