# TTS 模型下载与使用

[English](../tts-model-download.md) · [语音 API](audio-speech-api.md)

IronMLX.app 支持经过验证的 `mlx-community/IndexTTS-2.5-fp16` 资源配置。下载时自动准备推理所需资源，无需安装 Python、PyTorch 或手动填写依赖路径。其他 TTS 布局和模型版本需要单独适配。

## 使用步骤

1. 在 Dashboard 中搜索并下载 `mlx-community/IndexTTS-2.5-fp16`。应用选择已支持的固定版本。
2. 等待主模型下载、辅助资源准备和完整性校验完成，然后在模型列表点击加载。
3. 在 Dashboard 的“声音”页面导入参考音频并分配稳定的声音 ID，或在请求中直接提供参考音频的 Base64 字符串。
4. 通过 `/v1/audio/speech` 使用 `voice` 或 `ref_audio` 合成语音；可返回完整 WAV，或使用 `stream: true` 与 `response_format: pcm` 接收流式音频。

应用保存资源配置，并在重启时恢复此前加载或固定的模型。声音配置默认保存到 `~/.ironmlx/audio/voices/`，其中配置文件只引用受控目录内的音频文件。声音 ID 可供 OpenAI-compatible 前端使用；`GET /v1/audio/voices` 返回已启用声音。首次准备需要联网，完整资源准备后可以离线运行。合成音频播放由调用方或[独立示例客户端](../../examples/speech-client.swift)完成；Dashboard 可以试听声音配置中的参考录音。请求参数及完整示例见[语音 API](audio-speech-api.md)。

此前只下载主模型的用户，可在模型列表点击“准备语音资源”。准备失败后可重试，已经验证的模型文件和下载缓存会复用。

## IndexTTS 2.5 仓库契约

以 [mlx-community/IndexTTS-2.5-fp16](https://huggingface.co/mlx-community/IndexTTS-2.5-fp16/tree/65644cd70da15309ffeb74aa03f3686bb04e3eb1) 为参考：

| 文件 | 用途 |
| --- | --- |
| `config.json`, `config.yaml` | 模型配置 |
| `model_manifest.json`, `conversion_report.json` | 组件清单及转换记录 |
| `gpt.safetensors` | UnifiedVoice GPT |
| `codec.safetensors` | 语音 codec |
| `s2mel.safetensors` | S2Mel |
| `bigvgan.safetensors` | 声码器 |
| `model.safetensors` | w2v-BERT 语音前端，不是 GPT 主权重 |
| `multilingual_zh_ja_yue_char_del.tiktoken` | tiktoken 词表，无 `tokenizer.json` |
| `feat1.pt`, `feat2.pt`, `wav2vec2bert_stats.pt` | 音色、情绪矩阵及特征统计；由固定存储映射提取，不执行 pickle |
| `README.md`, `LICENSE*` | 仓库说明及许可证 |

识别依据为 manifest 的 `model_family=IndexTTS`、`model_version=2.5`、`format_version=1`，并交叉检查 config 版本和词表路径。四个组件的文件名由 manifest 读取，其大小须与固定 revision 的远端清单一致。缺组件、词表或辅助文件时，在下载权重前拒绝，不发布不完整 snapshot。

复用 App 的固定 commit、SHA-256 / Git blob 校验、断点续传、磁盘预留、下载队列、journal 和原子发布流程。文件保存到：

```text
~/.ironmlx/models/huggingface/<owner>--<repo>/snapshots/<commit>/
```

## 自动资源准备

`.ironmlx-snapshot.json` 记录 `model_type=indextts2_5`、`artifact_role=tts` 和完整下载清单。推理资源配置按内置版本和 SHA-256 校验，包括：

- 主模型中的五个 safetensors 组件、配置与词表。
- CAMPPlus 辅助权重及 w2v-BERT 配置；复用主模型已有的 w2v-BERT 权重。
- 原生重建的音色、情绪、统计与 CAMPPlus safetensors。
- WeText 0.1.2 FST 与 UniDic-lite 1.0.8 字典，以及来源说明和许可证。

辅助下载缓存与准备好的资源保存到 `~/.ironmlx/audio/indextts25/`。应用校验固定输入和输出，先在临时目录准备，再发布完整目录；不会改写模型 snapshot，也不会执行下载包中的 Python 或 pickle。断点续传缓存保留供重试，取消或失败时不会发布就绪状态。

模型文件完整性和运行资源就绪状态分别检查。资源缺失或变动时显示未就绪；准备成功后可加载。原生加载器在使用资源时再次校验。

## 验证

普通回归无需模型文件：

```sh
swift test -c release --package-path ironmlx-app
node scripts/tests/test-dashboard-tts.mjs
```

真实下载验收显式指定隔离的数据目录和 Release 下载 helper：

```sh
IRONMLX_TEST_DOWNLOAD_INDEXTTS=1 \
IRONMLX_TEST_DOWNLOAD_ROOT=/absolute/path/to/isolated-data \
IRONMLX_TEST_DOWNLOAD_BACKEND=/absolute/path/to/IronMLX.app/Contents/Helpers/ironmlx \
swift test -c release --package-path ironmlx-app --filter indexTTSLiveDownloadUsingAppService
```

随后使用同一隔离目录验证应用配置保存、模型加载、后端重启恢复和 WAV/PCM 接口：

```sh
IRONMLX_TEST_DOWNLOAD_ROOT=/absolute/path/to/isolated-data \
IRONMLX_AUDIO_APP_BUNDLE=/absolute/path/to/IronMLX.app \
IRONMLX_AUDIO_APP_REFERENCE=/absolute/path/to/reference.wav \
swift test -c release --package-path ironmlx-app --filter audioAppLoadsAndRestoresFromSavedConfigurationWithBundle
```
