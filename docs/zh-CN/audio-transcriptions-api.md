# 语音识别 API

[English](../audio-transcriptions-api.md)

IronMLX 的 App 后端和模型池服务均提供 OpenAI 兼容的
`POST /v1/audio/transcriptions`，使用原生 Rust/MLX 在本地推理，支持：

- `mlx-community/Qwen3-ASR-0.6B-8bit`
- `mlx-community/Qwen3-ASR-1.7B-8bit`

## App 服务管理

在模型管理中下载或导入模型并加载，App 负责管理模型与推理服务状态。
上游应用通过转写 API 提交音频、语言和可选的背景或词汇提示，并处理返回文本。
Dashboard 不提供转写客户端。已注册的模型也支持通过 API 请求按需加载。

先前仅下载保存的快照会通过原生预检重新确认支持状态，保留已下载的权重文件。
ASR 模型不能设为默认聊天生成模型。

## API

请求使用 `multipart/form-data`，必需字段为 `model` 和二进制文件 `file`。
可选字段为 `language`、`prompt`、`response_format` 和 `timestamp_granularities[]`。
省略 `language` 时自动识别。`response_format` 支持三个值：

| 值 | 返回内容 | 调用模型 |
| --- | --- | --- |
| `json`（默认） | JSON 格式的转写文字 | 仅 ASR |
| `text` | 纯文本 | 仅 ASR |
| `verbose_json` | 转写文字、时长及指定粒度的时间戳 | ASR 加固定 ForcedAligner |

只做单模型转写时，使用 `json` 或 `text`，不传 `timestamp_granularities[]`。
采用贪心解码，接受 `temperature=0`。使用 `stream=true`、`response_format=json` 时返回 SSE，
事件为 `transcript.text.delta` 和 `transcript.text.done`；拼接增量文本即可得到最终文本。

```sh
curl http://127.0.0.1:9068/v1/audio/transcriptions \
  -F model=mlx-community/Qwen3-ASR-0.6B-8bit \
  -F file=@speech.wav \
  -F language=zh
```

```json
{"text":"甚至出现交易几乎停滞的情况。"}
```

OpenAI SDK 可通过 IronMLX 的 `base_url` 调用 `audio.transcriptions.create`；
模型名使用上述本地模型 ID，认证沿用当前网络模式。Anthropic Messages 不承载 ASR。

CLI 也可直接使用本地快照：

```sh
ironmlx transcribe --model /absolute/path/to/snapshot --audio speech.wav --language zh
ironmlx serve --model /absolute/path/to/snapshot --model-id mlx-community/Qwen3-ASR-0.6B-8bit
```

App 后端的 `/admin/api/models/register`、`/admin/api/models/load` 使用现有
`model`、`model_dir` 字段。ASR 不需要 IndexTTS 的 `audio` 资源配置，也不额外下载
音色、词典或辅助模型。模型池清单使用普通 `id`、`path`、`load_policy` 字段。

## 长音频与资源边界

服务端在低能量位置将长录音自动分为连续的 25–30 秒片段，逐段识别并合并文本。
音频不裁剪，分段不重叠、不跳过样本；英文片段之间补空格，中文等文本直接拼接。
每段默认最多生成 1,024 个 token，达到生成上限会报错，不将截断结果当作成功返回。
实时识别的已确认前缀也计入该段预算，连续更新不会绕过设置。

App 的模型参数页为每个 ASR 模型分别保存以下设置，修改用于新请求与新实时会话，
重启后由 App 恢复到后端：

| 模型 | 默认最大音频时长 | 默认每段最大输出 Token |
| --- | --- | --- |
| Qwen3-ASR-0.6B-8bit | 1,200 秒 | 1,024 |
| Qwen3-ASR-1.7B-8bit | 1,200 秒 | 1,024 |

两个规格没有官方给出的不同上限，因此采用相同初始策略，用户可分别调整。
时长参考[官方分段策略](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/utils.py)，
输出预算参考[官方评测配置](https://github.com/QwenLM/Qwen3-ASR#evaluation)；
这些是 IronMLX 的默认策略，不是模型的硬上限。时长可设为 1–7,200 秒，
输出预算可设为 1–8,192 Token。时长限制适用于一次文件请求或一轮显式提交的实时输入，
输出预算适用于每个识别片段，不限制合并后全文的总 Token 数。留空恢复该模型的默认值。
超过 20 分钟的文件应先提高该模型的时长设置，自动分段处理仍保留。

App 通过本机管理端点 `POST /admin/api/models/asr` 应用设置，
正文为 `{"model":"模型 ID","settings":{"max_audio_duration_seconds":1200,"max_output_tokens":1024}}`。
`GET /admin/api/models/asr?model=模型ID` 返回有效设置和模型默认值。
该端点不加载权重、不重载模型，也不修改正在进行的请求；App 将设置持久化在模型参数中。

- 支持 WAV PCM/浮点、FLAC、MP3；最大时长按模型设置，服务允许至多两小时，编码文件最大 128 MiB、解码 PCM 最大 512 MiB。
  高采样率或双声道录音可能先达到解码字节上限。
- 单声道或双声道、8–96 kHz，原生转换为 16 kHz 单声道。
- 背景提示最多 16,384 个 UTF-8 字节。
- 每模型最多四个排队 GPU 任务；长录音在片段间让出工作线程，使实时请求能穿插执行。
- 上传超时 60 秒；处理期限为 180 秒加录音时长的两倍。
- 断开请求会取消计算；已提交的 GPU 工作完成后才释放模型租约和内存预留。

不支持的参数、重复的普通字段、字幕格式和非零温度会明确报错。
说话人分离尚未实现。流式文件转写要求 `response_format=json`。

## 带时间戳转写：固定双模型流程

向现有转写接口传入 `response_format=verbose_json`。`model` 仍选择两款受支持的
Qwen3-ASR 中的一款；IronMLX 先转写，再用固定的
`mlx-community/Qwen3-ForcedAligner-0.6B-8bit` 对齐每段音频和识别文字。
请提前下载、注册该辅助模型，请求不会自动下载权重。当前不提供 `alignment_model`
参数或任意模型组合接口。

```sh
curl http://127.0.0.1:9068/v1/audio/transcriptions \
  -F model=mlx-community/Qwen3-ASR-0.6B-8bit -F file=@speech.wav \
  -F response_format=verbose_json \
  -F 'timestamp_granularities[]=word' -F 'timestamp_granularities[]=segment'
```

响应包含 `task`、`text`、`language`、`duration`（秒），以及请求的
`words: [{word,start,end}]` 和／或 `segments: [{id,text,start,end}]`。
`timestamp_granularities[]` 是数组字段，`[]` 表示可传入多个值。
`word` 请求字词级时间戳，`segment` 请求片段级时间戳；同时请求时，像上例一样
重复传入同名 multipart 字段。此字段仅适用于 `verbose_json`，省略时默认 `segment`；
在 `json` 或 `text` 模式下传入会报错。两种时间戳均来自真实对齐；片段时间覆盖该 ASR 窗口中
首个至最后一个对齐单元，不代表按句分段或 VAD 边界。长音频结果会加上窗口在
原始录音中的时间偏移。普通 `json`／`text` 和 SSE 仍只调用 ASR；时间戳仅支持
非流式 `verbose_json`，本次不为实时 WebSocket 增加时间戳。
字段遵循 [OpenAI 转写契约](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create)，
不伪造 Whisper 专属的 token 评分或置信度字段。

缺少辅助模型、对齐语言不受支持、容量不足或对齐失败时，整个请求报错，不返回
缺失或伪造时间戳的成功结果。进入对齐步骤前释放 ASR 租约，仍遵守原有模型数量、
固定加载与内存策略：App 达到模型数量上限会拒绝加载，通常需要给两款模型留出
加载名额；使用 lazy 模型的模型池服务可以按既有策略卸载空闲且未固定的模型，
因此可以只保留一个加载名额。

## 独立强制对齐（IronMLX 扩展）

`POST /v1/audio/alignments` 使用 multipart，必须提供 `model`、`file`、`text`、
`language`。`model` 指定已注册的 ForcedAligner；此接口仅对齐，不运行 ASR。
可选 `response_format=json`。不接受采样参数、提示词、流式输出或时间戳粒度参数。
这是 IronMLX 扩展接口，不是 OpenAI 或 Anthropic 标准 API。

```sh
curl http://127.0.0.1:9068/v1/audio/alignments \
  -F model=mlx-community/Qwen3-ForcedAligner-0.6B-8bit \
  -F file=@speech.wav -F 'text=甚至出现交易几乎停滞的情况。' -F language=zh
```

返回 `{text,language,duration,words:[{word,start,end}]}`，时间单位为秒。
独立对齐最多接受 300 秒音频、16,384 个 UTF-8 字节的文字和 2,048 个有声文字单元；
音频和文字合计不能超过 8,192 个上下文 token。更长的独立对齐输入应由上游拆成
相互匹配的音频／文字对，不能在不知道文字归属的情况下仅切音频。组合流程则利用
各 ASR 窗口的转写文字自动处理，并沿用对应 ASR 模型的时长设置。
文件、上传、排队、取消和内存边界与转写接口共用。

该对齐模型支持中文、粤语、英语、德语、西班牙语、法语、意大利语、葡萄牙语、俄语、
日语和韩语。其他语言仍可普通转写，但不能使用此固定对齐流程。
中文汉字逐字对齐，空格分隔的文字按清理后的词处理。本原生实现对日语逐字符对齐、
韩语按空格单位对齐，不使用上游 nagisa／soynlp 分词；因此 `words` 不保证在所有语言
中都是语言学意义上的词。标点保留在顶层全文中，没有独立时间戳。
模型预测时间戳的粒度为 80 ms，返回小数秒不表示具有更细精度。
对齐给定文字不等于验证文字识别正确。

App 将该模型标为 `runtime_kind: "alignment"`，支持音频输入，不支持流式输出、
默认聊天模型、生成采样或输出 Token 参数。旧的已校验仅下载快照在加载时通过
原生预检后升级，保留已有权重。App 不增加录音或转写操作页面。

## 实时识别

上游应用连接：

```text
ws://127.0.0.1:9068/v1/realtime?intent=transcription&model=mlx-community/Qwen3-ASR-0.6B-8bit
```

LAN 模式使用现有的认证 `wss` 端点和 Bearer 请求头。App 后端与模型池服务均支持该路径；
App 不提供录音客户端。服务端先发送 `session.created`，随后可配置会话：

```json
{"type":"session.update","session":{"type":"transcription","audio":{"input":{"format":{"type":"audio/pcm","rate":24000},"transcription":{"model":"mlx-community/Qwen3-ASR-0.6B-8bit","language":"zh"},"turn_detection":null}}}}
```

支持可选 `prompt`。省略 `language` 自动识别，也可用单元素 `languages` 数组。
会话配置仅在两次音频轮次之间修改。未知处理选项会被拒绝。
也接受 beta 格式的 `transcription_session.update` 和 PCM16 字段名。

持续发送 **24 kHz、单声道、带符号 PCM16、小端序**样本的 Base64：

```json
{"type":"input_audio_buffer.append","audio":"BASE64_PCM16"}
```

每条追加消息最多四秒，建议按较短音频帧发送。每累积两秒新音频便执行一次识别，
无需先提交音频结束。实现重新编码当前音频窗口，并在初始更新后回退五个 token 构造稳定前缀。
连续输入在不超过 30 秒的窗口内寻找低能量边界，自动确认一个结果项并开始下一项，使会话内存保持有界。
边界不会早于当前已识别音频的位置。

根据 `item_id` 处理以下事件：

| 事件 | 上游应用行为 |
| --- | --- |
| `conversation.item.input_audio_transcription.partial` | IronMLX 扩展：用 `transcript` 替换当前临时文本；`revision` 在当前结果项内递增 |
| `conversation.item.input_audio_transcription.delta` | 追加稳定文本 `delta` |
| `conversation.item.input_audio_transcription.completed` | 用 `transcript` 作为最终确认文本 |
| `conversation.item.input_audio_transcription.failed` | 处理当前结果项的错误，不能视为成功 |

不要将临时全文与稳定增量拼入同一个文本缓冲区。
上游检测语音结束后发送 `input_audio_buffer.commit`，至少需要 100 ms 音频；
服务端发送 `input_audio_buffer.committed` 与最终结果，连接继续接受下一轮音频。
`completed` 增加 IronMLX 扩展字段 `is_final`：自动窗口结果为 `false`，
显式提交轮次的最后一个结果为 `true`。提交时仍有多个待处理窗口，应等待该标记再结束接收。
`input_audio_buffer.clear` 取消并丢弃当前轮次，返回 `input_audio_buffer.cleared`。
断开连接会取消正在执行的工作。

每进程最多八个实时连接；单连接最多缓冲 60 秒音频；WebSocket 消息最大 512 KiB；
空闲超时 90 秒，连接与累计输入音频限制两小时。输入过快或队列满时明确报错。
活跃会话持有模型租约；延迟取决于硬件和并发负载，建立连接本身不代表满足某个延迟指标。

协议实现 OpenAI Realtime 的转写子集，并增加上述可修订临时文本事件。
不提供 WebRTC、降噪、音频生成、时间戳或说话人分离。
麦克风采集和文本展示由上游应用负责；客户端 VAD 可按需使用。

### 服务端 VAD

通过 `session.audio.input.turn_detection` 启用，也接受 beta 顶层
`session.turn_detection`：

```json
{"type":"session.update","session":{"audio":{"input":{"turn_detection":{"type":"server_vad","threshold":0.5,"prefix_padding_ms":300,"silence_duration_ms":500}}}}}
```

省略配置时保留当前模式；初始为手动模式（`null`）。在轮次之间将
`turn_detection` 设为 `null` 可关闭 VAD。阈值范围 0–1，前置保留范围
0–1,000 ms，静音时长范围 100–3,000 ms。
Earshot 1.2.2 使用内置分类器在本地 CPU 上按 16 ms 帧检测，无需额外下载模型。
检测器的重采样不改变交给 ASR 的原始 24 kHz PCM。这是声学 VAD，不是语义 VAD。

服务端发出 `input_audio_buffer.speech_started`，携带 `item_id` 和
`audio_start_ms`；讲话结束后发出同一 ID 的 `input_audio_buffer.speech_stopped`
及 `audio_end_ms`。时间相对于已接受音频的时间轴，包含前置保留与尾部静音；
这些是语音边界，不是字词时间戳。上游必须持续发送静音音频才能检测静音结束，
网络停止发包不等于静音。结束后自动发送 `input_audio_buffer.committed`，
并以 `is_final: true` 完成转写，无需客户端提交。
后续讲话可以在前一段推理期间排队，总待处理 PCM 仍限制为 60 秒。
连续讲话即使没有静音，也会在约 28 秒时自动分段，保证 ASR 窗口有界。

仍可用 `input_audio_buffer.commit` 主动结束当前讲话，不完整检测帧也会保留。
没有活动讲话时提交会报错。清空会取消当前及排队轮次并重置检测器；断连会取消所有工作。
每模型的音频时长和输出预算仍有效；VAD 轮次超过时长限制时明确报错并关闭会话、取消待处理任务。
不支持降噪、`semantic_vad` 或对话生成/打断选项。这些是 API 会话参数，App 不增加任务操作界面。

可运行的上游客户端见 [`examples/transcription-client.swift`](../../examples/transcription-client.swift)。
客户端按录音速率发送原始 PCM，同时接收临时结果，将确认结果以 JSON 行写入标准输出。
输入必须是无 WAV 文件头的 24 kHz 单声道 PCM16 小端序；可先在上游转换格式：

```sh
ffmpeg -i speech.wav -ar 24000 -ac 1 -f s16le speech.pcm
swiftc -parse-as-library examples/transcription-client.swift -o /tmp/ironmlx-transcription-client
/tmp/ironmlx-transcription-client --audio speech.pcm --language zh
```

使用 `--model` 指定 1.7B 模型，`--url` 指定实时端点。LAN 模式的 Bearer 密钥通过
`IRONMLX_API_KEY` 环境变量传入，不写入命令行参数。空缓冲区提交仍会返回参数错误。

模型能力返回 `runtime_kind: "asr"`，声明音频输入与流式支持。卸载、TTL、模型数量和
进程内存策略继续生效；ASR 不使用通用生成调度器、KV 策略、MTP 或 PromptLookup 控件。
