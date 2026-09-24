# 语音合成 API

[English](../audio-speech-api.md) · [TTS 模型下载与使用](tts-model-download.md)

服务器通过模型池和 App 模型管理 daemon 提供 `POST /v1/audio/speech`。
`ironmlx-audio` 负责原生语音合成与音频 IO；`ironmlx-runtime` 负责加载、调度、
取消和内存准入。

## 通过 IronMLX.app 使用

下载、资源准备、加载、声音管理和故障恢复步骤见
[TTS 模型下载与使用](tts-model-download.md)。请求与响应契约以本文为准。

## 注册本地资源

音频模型需要经过验证的源 snapshot、派生的参考编码器资源、固定的资源锁、
WeText FST 文件和 UniDic-lite 字典。资源 profile 与转换工具见
[音频库说明](../../ironmlx-audio/README.md)。资源路径均为明确的本地路径；加载时
会进行校验，服务器不会自行下载或转换这些资源。

模型池使用 `ironmlx serve --model-manifest models.json`：

```json
{
  "models": [
    {
      "id": "mlx-community/IndexTTS-2.5-fp16",
      "path": "/models/indextts25/snapshots/<revision>",
      "load_policy": "lazy",
      "audio": {
        "derived_resources": "/models/indextts25-derived",
        "resource_lock": "/resources/indextts25/sources.json",
        "wetext_fsts": "/resources/wetext/fsts",
        "unidic_dir": "/resources/unidic_lite/dicdir"
      }
    }
  ]
}
```

对于模型管理 daemon（`ironmlx serve`），现有本地接口
`/admin/api/models/register` 和 `/admin/api/models/load` 接受相同的 `audio`
对象，并同时接收 `model`（公开标识）和 `model_dir`（源 snapshot）。已有的卸载、
固定、TTL 和模型数量策略同样适用。音频模型报告 `runtime_kind: "tts"`、架构
`indextts25` 和调度器 `serial_audio`。因果调度器、采样、MTP 和 PromptLookup
覆盖参数会被拒绝。

可执行文件会在初始化 MLX 或启动 worker 前将 `MLX_ENABLE_TF32` 默认设为 `0`。
用户显式设置的环境变量会保留，但音频加载器会拒绝 `0` 以外的值。嵌入 Rust
runtime 的应用必须在首次调用 MLX 前设置这一进程级精度策略。音频 worker 使用
单独注册的 stream 和请求级随机状态。内存与 allocator cache 限制仍为进程级，
并与语言模型共享。

## 声音配置

声音配置为 OpenAI-compatible 客户端提供稳定标识。App 默认将其保存在
`~/.ironmlx/audio/voices/`；`ironmlx serve` 可通过 `--voice-profile-dir`
指定其他目录。`profiles.json` 保存元数据和相对的受管文件名，参考音频单独保存在
`files/` 下。

| 接口 | 作用 |
| --- | --- |
| `GET /v1/audio/voices` | 为兼容客户端列出已启用声音 |
| `GET /admin/api/audio/voices` | 列出全部声音，包括已禁用配置 |
| `POST /v1/audio/voices` | 使用 `id`、`name`、可选的 `language`/`enabled` 和 Base64 `ref_audio` 创建声音 |
| `PATCH /v1/audio/voices/{id}` | 更新元数据、启用状态或替换 `ref_audio` |
| `DELETE /v1/audio/voices/{id}` | 删除声音配置及其受管参考文件 |
| `GET /v1/audio/voices/{id}/preview` | 返回已保存的参考录音供试听 |

声音 ID 长度为 1–64 个字符，可包含 ASCII 字母、数字、`.`、`_` 或 `-`，并且
必须以字母或数字开头。修改显示名称不会改变 ID。写入操作会先验证完整音频，再以
原子方式发布元数据。公开发现接口只返回 ID、显示名称和可选语言；管理接口还会返回
校验元数据和启用状态。试听接口返回参考录音，并设置
`X-IronMLX-Voice-Preview: reference`；它不会重新合成试听样本。

## 请求

返回完整 WAV：

```json
{
  "model": "mlx-community/IndexTTS-2.5-fp16",
  "input": "这是需要合成的文本。",
  "ref_audio": "<Base64>",
  "response_format": "wav"
}
```

使用受管声音的等价请求：

```json
{
  "model": "mlx-community/IndexTTS-2.5-fp16",
  "input": "这是需要合成的文本。",
  "voice": "speaker_a",
  "response_format": "wav"
}
```

PCM 流：

```json
{
  "model": "mlx-community/IndexTTS-2.5-fp16",
  "input": "这是需要合成的文本。",
  "ref_audio": "<Base64>",
  "response_format": "pcm",
  "stream": true
}
```

`model` 和 `input` 是必填的非空字符串。`ref_audio` 与 `voice` 必须且只能提供
一个。`voice` 接受稳定的字符串 ID 或 `{ "id": "speaker_a" }`。
`response_format` 默认为 `wav`，`stream` 默认为 `false`。支持上面列出的两种输出
组合。未知字段、重复字段和显式 `null` 均会被拒绝。

`ref_audio` 使用带 padding、无空白字符的标准规范 Base64。应将完整音频文件编码
为该字符串；不接受文件路径、URL 或 data URL。音频格式从解码后的字节中识别。
输入支持 WAV（PCM 16/24/32 或 IEEE float32）、FLAC 和 MP3。服务先验证完整输入
文件，再截取开头 15 秒作为参考。

输入是一个完整 JSON 请求。自动语言选择和固定合成 profile 见音频库说明。流式
粒度为完成的文本分段：每个分段必须完成语音生成、声学解码与声码器处理后，才能
输出对应 PCM。

## 响应与客户端处理

两种模式都产生 22050 Hz、单声道、有符号 16 位小端采样。响应包含
`X-Request-Id`、`X-Audio-Sample-Rate: 22050`、`X-Audio-Channels: 1` 和
`X-Audio-Sample-Format: s16le`。

| 模式 | Content-Type | 长度 | Cache-Control |
| --- | --- | --- | --- |
| 完整 WAV | `audio/wav` | 完整文件 `Content-Length` | `no-store` |
| PCM 流 | `audio/pcm` | 无 `Content-Length` | `no-store, no-transform` |

PCM 响应还包含 `X-IronMLX-Streaming-Granularity: segment`。流中只有无 WAV
头、无 JSON 事件的原始采样。网络读取可以在任意字节边界结束，包括奇数字节数；
客户端必须保留最后一个未配对字节，与下一次读取合并后再解释小端采样。传输 chunk
不标识文本分段。WAV 和 PCM 使用相同的舍入与饱和规则。

服务器会等到获得第一个非空且可播放的 chunk 后再发送 HTTP 200。在此之前发生的
错误使用现有 JSON 错误信封；HTTP 200 发出后的失败会中止响应体，不会追加 JSON
或正常完成标记。客户端必须区分正常结束与截断或失败的传输，并且不能在已经播放
部分音频后自动重放请求。

断开连接会取消工作。正在运行的 GPU kernel 无法抢占：worker lease、执行 permit
和计算 reservation 会保留到取消检查点及最后一次 GPU 同步完成。卸载活动模型时
模型进入 draining，不能在 worker 仍在使用时释放模型。

## 限制与调度

| 策略 | 默认值 |
| --- | --- |
| HTTP body / 解码文件 / 解码后的 f32 PCM | 32 MiB / 16 MiB / 48 MiB |
| 参考音频 | 1–60 秒、8–96 kHz、单声道或双声道 |
| 文本 | 65536 UTF-8 字节、16384 个规范 token、256 个分段 |
| 分段 token 预算 | 120 |
| 包含分段间静音的总输出 | 13230000 帧（600 秒） |
| 每模型执行数 / 等待队列 | 1 / 4 |
| 输入准备执行数 / 等待容量 | 1 / 4 |
| 输出 channel | 8 个 chunk，每个最多 8192 个单声道帧 |
| 输入 / 模型加载 deadline | 60 / 300 秒 |
| 排队 / 首段音频 / 总执行 / 消费方阻塞 | 60 / 120 / 900 / 30 秒 |

`audio.execution` 配置可选接受 `queue_timeout_ms`、
`first_audio_timeout_ms`、`execution_timeout_ms`、`slow_consumer_timeout_ms`、
`max_output_frames` 和 `segment_tokens`。deadline 必须为正且不超过 24 小时。
输出容量可以从 13230000 帧向下降低；分段预算可以在 6–120 token 之间设置。
首段音频和总执行 deadline 从请求获得执行 permit 后开始计算，不包含排队和模型加载。
这些是服务策略，不是模型硬限制或延迟保证。

输入存储、常驻模型资源、临时合成 tensor、完整分段和传输 buffer 均纳入共享进程
governor。初始保守准入策略为每个已准入输入预留 256 MiB，为每个执行中的合成任务
预留 32 GiB；输出存储与允许增长的 allocator cache 另行预留。因此在内存有限的
设备上，即使请求很短也可能被拒绝。有界输出 channel 本身并不等同于合成内存预算。
发出字节对应的 reservation 会持续到网络栈消费或丢弃这些字节。

## 错误类别

| 状态码 | 示例 |
| --- | --- |
| 400 | 字段、Base64 或音频无效；输出组合不支持；模型任务不匹配 |
| 401 | 现有 LAN 安全策略下的鉴权失败 |
| 404 | 模型未注册或已禁用 |
| 413 | 请求、解码音频或文本超过容量 |
| 503 | 队列已满、内存不足或资源不可用；包含 `Retry-After` |
| 504 | 在响应 header 发出前达到排队、首段音频或执行 deadline |
| 500 | 权重无效、生成达到限制但未完成、worker 或推理意外失败 |

响应 header 发出后，生成限制、deadline、背压失败和 worker 失败都会以传输错误终止
响应体，不再使用相应的 header 前状态码。

## 开发验证

`ironmlx/tests/audio_http.rs` 包含一个被忽略的真实模型 HTTP 验收测试。通过文档中
说明的环境变量为其提供经过验证的本地资源、参考 WAV 和本地 LLM snapshot。测试
经由真实服务器进程执行原生合成，包括失败与取消路径。普通测试不能替代该验收；
Release App 打包和客户端播放也需要各自的验收检查。

### 独立 macOS 播放客户端

[Swift 示例](../../examples/speech-client.swift) 使用 AVAudioEngine 通过默认音频输出
设备播放。使用 Xcode 命令行工具构建并运行：

```sh
swiftc -parse-as-library -swift-version 6 examples/speech-client.swift -o /tmp/ironmlx-speech-client
/tmp/ironmlx-speech-client --reference reference.wav --text '这是需要合成的文本。' --format wav --output speech.wav
/tmp/ironmlx-speech-client --reference reference.wav --text '这是需要合成的文本。' --format pcm --output speech.pcm
/tmp/ironmlx-speech-client --voice speaker_a --text '这是需要合成的文本。' --format pcm --output voice.pcm
```

必须且只能提供 `--reference FILE` 或 `--voice ID` 其中之一。

默认 endpoint 为 `http://127.0.0.1:9068/v1/audio/speech`；可通过 `--url` 和
`--model` 选择其他 endpoint 或已注册模型。需要认证时设置 `IRONMLX_API_KEY`。
运行客户端前应按上文注册模型资源。Dashboard 资源配置与播放不属于该示例的职责。

PCM 在接收过程中开始播放，音频播放器最多预排两秒。读取结果可能包含奇数字节数，
解码器会保留未配对字节。WAV 会在完整响应及其 header 通过校验后开始播放。两种
模式都验证格式 header，而不是猜测采样率或声道数。JSON 输出记录音频设备报告的
已播放帧数、首次播放时间、下载完成时间与播放完成时间。这些观测结果不评价主观音质。

客户端使用 macOS `/usr/bin/curl` 和 HTTP/1.1 检测被截断的 chunked 响应；
CFNetwork/URLSession 可能会将缺失最终 chunk 的情况视为正常 EOF。播放队列已满时，
操作系统 pipe 会施加背压。示例不会重试合成请求。按 Ctrl-C 或使用
`--cancel-after-ms N` 会停止播放并关闭请求。传输、解码或设备错误会以非零状态退出；
只有传输与播放成功后才会发布可选输出文件。已有输出文件不会被覆盖。临时请求文件
保存在私有目录中，并在退出时删除。

客户端测试将 wire fixture 与真实模型验收分开：

```sh
IRONMLX_SPEECH_CLIENT=/tmp/ironmlx-speech-client \
  IRONMLX_SPEECH_PLAYBACK_TESTS=1 \
  python3 -m unittest discover -s scripts/tests -p test_speech_client.py -v
```

播放测试开关要求可用的音频设备。CI 会运行不依赖该设备的传输检查；这些检查不能
证明声音确实可播放，也不能替代真实模型验收。

### Release Bundle 验收

使用 `scripts/build-app-bundle.sh` 构建，然后运行 Bundle 静态检查和模型分发边界
检查。若要针对该确切 Bundle 重复真实 HTTP 测试，在真实资源测试环境中增加以下变量：

```sh
IRONMLX_AUDIO_HTTP_BUNDLE="$PWD/dist/IronMLX.app" \
  cargo test --locked -p ironmlx --release --test audio_http -- --ignored --nocapture --test-threads=1
```

该模式仅使用 Bundle 中的 helper 和 metallib，在子进程中清除 `MLX_*` 和
`DYLD_*` 覆盖变量，并在 checkout 之外运行。模型、派生权重、FST 和字典数据仍为
明确的外部资源，不随 App 打包。还需针对 Bundle helper 单独重复客户端播放检查。
本地 ad-hoc 签名的 Bundle 不代表 Developer ID 签名、公证或公开分发授权。
