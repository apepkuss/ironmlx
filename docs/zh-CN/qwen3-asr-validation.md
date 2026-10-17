# Qwen3-ASR 验证记录

验证环境：Apple M1 Pro、32 GiB 内存，macOS 26.6。两个模型均使用本地已验证的快照；生产推理不依赖 Python。

| 模型 | Hugging Face revision |
| --- | --- |
| `mlx-community/Qwen3-ASR-0.6B-8bit` | `89e96d92ba34aca20b3e29fb10cc284097d1219f` |
| `mlx-community/Qwen3-ASR-1.7B-8bit` | `a8379a2e2f9e313c9292cdf1af4055ab56d50d55` |

## 数值与转写结果

参考实现为 MLX Audio 的 `qwen3_asr`，commit `dad2087f790dfe96b07d5aac1405db4b578b411f`。
音频来自官方 Qwen3-ASR 仓库 commit `7c6daf77a2421100f5fb066495372c00129d39ff` 的中英文示例。
使用相同的 16 kHz 输入，0.6B、1.7B 的中英文共四组转写文本和输出 token 数均与参考实现一致。
128 维 Whisper 特征的完整测试张量与 Transformers 参考误差小于 `2e-4`。

两个模型均通过 App 管理后端的注册、加载、转写、卸载测试；JSON 与纯文本结果一致。
Dashboard 仅负责模型和服务管理；转写请求由上游 API 客户端发起，不提供音频上传与转写结果控件。
中文样本输出：`甚至出现交易几乎停滞的情况。`

最终 App Bundle 的静态检查和模型分发边界检查通过；独立 `serve --model` 服务也返回正确中文结果。

真实 App 后端还验证了两种模型的 FLAC、MP3 中文转写，以及 48 kHz WAV 英文转写。
未知模型返回 404；非法语言、重复字段、非零温度、字幕格式，以及流式配合纯文本格式的请求返回 400。
ASR 模型用于聊天或 TTS 时返回 400。断开一次转写请求后，后续请求正常完成，活动数和排队数恢复为零。

旧的 `download_only` 快照在 App 启动注册时通过原生预检并迁移为 `asr`，保留已有权重。
真实 App 下载服务也完成了两种模型的完整下载、校验、发布和可运行状态验收。

## 长音频与实时识别

文件接口已经移除原先的 30 秒整段输入限制：服务端自动在低能量位置分段，
每段不超过 30 秒，连续覆盖输入并合并文本。每模型可配置时长和单段输出预算，默认 1,200 秒和 1,024 token。
资源边界为最长两小时、128 MiB 编码文件和 512 MiB 解码 PCM；这些是服务约束，
不代表已经对两小时所有格式、语言与音质组合做过准确率评估。

两个模型均通过以下真实推理测试，使用 App 管理后端的注册、加载和卸载流程：

- 将官方中文样本重复 300 次，生成 **1,263 秒（21 分 3 秒）**录音。
  测试将对应模型的时长设置提高到 7,200 秒，结束后恢复 1,200 秒默认值。
  两个模型的完整 JSON 结果均含 300 次目标语句，没有在原先 30 秒或共享重采样器 60 秒处截断。
- 与上述文件请求并发，通过实际 TCP WebSocket 以音频速率持续发送 **37.89 秒** PCM16。
  两个模型均在 `commit` 前返回临时文本，自动完成首个滚动窗口；提交后完成第二个窗口，
  9 句目标语句均保留，稳定增量拼接与确认全文一致。
- 对 **42.1 秒**文件启用 `stream=true`，返回 SSE 的增量和完成事件。
- 实时会话清空返回确认；发送新音频后断开连接，后续卸载正常完成。

上述长录音由短样本重复构成，用于验证覆盖、调度和接口生命周期，不能代替自然长录音的准确率评测。
实时实现重新编码累计音频窗口并采用五 token 回退前缀，采用有界滚动窗口；
可按实时会话启用 CPU 服务端 VAD；不提供 WebRTC、说话人分离或时间戳。上游应用负责采集音频和展示结果。

服务端 VAD 使用 Earshot 1.2.2，在两款模型的真实 TCP WebSocket 测试中验证：
纯静音不产生语音事件；连续两段讲话无需客户端提交即可正确转写；起止、提交和完成事件
保持相同 item ID，排队轮次的前驱 ID 正确；启用 VAD 时仍可主动提交，关闭 VAD 后
手动提交继续工作。清空会取消推理及排队内容，确认后不再出现旧结果。
检测器单元测试覆盖分包无关性、前置保留、静音结束、连续讲话约 28 秒分段的完整覆盖，
以及主动提交保留不完整帧。这些测试不等于已经完成复杂噪声场景下的 VAD 准确率评测。


分段单元测试覆盖 95 秒输入和不整齐尾部的完整样本覆盖、低能量边界选择，
以及实时边界不能退到已识别音频之前。重采样测试确认 ASR 可处理超过 60 秒的输入，
同时 TTS 参考音频保留原有 60 秒限制。实时参数测试覆盖格式、VAD、语言字段和缓冲区上限。

最终 Release 后端已装入签名后的 App Bundle，并通过实际运行的 `127.0.0.1:9068` 验收：
两个模型的 42.1 秒文件各返回 10 句，37.89 秒实时流各返回 9 句、两个确认项和 17 个临时事件。
0.6B 文件 SSE 的增量拼接与最终全文一致；完成后两个模型的活跃数和排队数均为零。
独立 `serve --model` 的 EnginePool 路由也通过 42.1 秒文件及短音频实时识别，随后退出测试服务并恢复 App。
App 中的后端与本次 Release 产物具有相同 Mach-O UUID；Bundle 静态检查和模型分发边界检查通过。
新增 WebSocket 依赖的第三方材料已重建并通过可重复性检查，SBOM 验证包含 324 个组件。

## 可重复的检查

Rust 音频单元测试：

```sh
MLX_DIR=/absolute/path/to/mlx-install MLX_ENABLE_TF32=0 \
  cargo test -p ironmlx-audio qwen3_asr
```

真实模型 API 测试：

```sh
MLX_DIR=/absolute/path/to/mlx-install \
IRONMLX_ASR_SMALL_DIR=/absolute/path/to/small/snapshot \
IRONMLX_ASR_LARGE_DIR=/absolute/path/to/large/snapshot \
IRONMLX_ASR_TEST_AUDIO=/absolute/path/to/asr_zh_16k.wav \
MLX_ENABLE_TF32=0 \
  cargo test -p ironmlx --lib qwen3_asr_real_app_lifecycle_and_api -- --ignored --nocapture
```

下载验收测试使用独立的 App 数据目录，避免修改用户的下载任务：

```sh
IRONMLX_QWEN3_ASR_LIVE_ROOT=/absolute/path/to/temporary/app-data \
IRONMLX_QWEN3_ASR_LIVE_HELPER=/absolute/path/to/ironmlx \
  swift test --package-path ironmlx-app --filter qwen3ASRLiveDownloadUsingAppService
```

Rust 格式检查、nightly Clippy、release 编译、App 相关的 48 项 Swift 测试均通过。
本次长音频和实时扩展另通过 7 项 Swift 相关测试与 11 项服务安全回归测试；Swift Release 构建通过。
最终重启验收发现并修正 ASR 恢复时被选为默认聊天模型的问题；4 项针对性 Swift 测试通过，
覆盖 ASR/Embedding 单独恢复、与生成模型混合恢复、旧默认标记及原有生成模型默认选择。
修正后的最终 App 实际重启后，两种 ASR 模型均成功恢复，均未设为生成默认模型；
页面不再显示恢复失败，重启后的自动语言识别转写结果正确。
全量及扩大范围的 Swift 测试进程以 signal 6 退出，诊断为 `freed pointer was not the last allocation`；扩大范围运行还出现菜单重启测试超时，均不能记为通过。

当前支持范围和接口边界见[语音识别 API](audio-transcriptions-api.md)。
上游客户端可按该文档编译 [`examples/transcription-client.swift`](../../examples/transcription-client.swift)，
直接调用 App 启动的服务，App 本身不加入音频采集或转写客户端控件。

## 每模型 ASR 参数

两个模型的默认最大音频时长为 1,200 秒，每段输出预算为 1,024 Token；
App 参数页独立保存两个模型的设置，留空恢复模型默认值。时长允许 1–7,200 秒，
输出预算允许 1–8,192 Token。实时识别的前缀计入本段总输出预算。

新增存储与注册测试覆盖按模型隔离、保存后重新读取、非法值不覆盖旧值、清空恢复默认，
以及启动注册将持久设置恢复到后端。真实模型 API 测试对两种模型分别验证：
设置 3 秒时拒绝 4.21 秒输入，设置 1 Token 时返回 `asr_output_limit_exceeded`，
提高时长为 7,200 秒后保留 21 分 3 秒转写与持续实时识别，测试结束后恢复默认。

最终 App 页面验收确认两个模型均显示 1,200 秒与 1,024 Token。通过 0.6B 参数页保存 3 秒后，
实际文件与 WebSocket 请求均被时长限制拒绝，1.7B 的同一文件请求仍成功；
保存 6 秒与 1 Token 后，实际文件与 WebSocket 返回 `asr_output_limit_exceeded`。
App 重启后后端恢复 0.6B 的 6 秒/1 Token，1.7B 保持默认值。随后通过页面清空测试值，
两个模型均恢复默认，文件转写再次正确完成。Rust 格式检查、nightly Clippy、Release 构建、
17 项 Swift 存储回归、真实模型回归和最终 Bundle 检查通过。
