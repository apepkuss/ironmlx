# TTS 模型下载

下载契约与推理支持分开。App 下载器目前识别 IndexTTS 2.5 的 MLX 仓库布局；其他 TTS 布局需要各自的下载契约，不按文件后缀直接放行。

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
| `feat1.pt`, `feat2.pt`, `wav2vec2bert_stats.pt` | 情绪矩阵及特征统计；仅下载，不反序列化执行 |
| `README.md`, `LICENSE*` | 仓库说明及许可证 |

识别依据为 manifest 的 `model_family=IndexTTS`、`model_version=2.5`、`format_version=1`，并交叉检查 config 版本和词表路径。四个组件的文件名由 manifest 读取，其大小须与固定 revision 的远端清单一致。缺组件、词表或辅助文件时，在下载权重前拒绝，不发布不完整 snapshot。

复用 App 的固定 commit、SHA-256 / Git blob 校验、断点续传、磁盘预留、下载队列、journal 和原子发布流程。文件保存到：

```text
~/.ironmlx/models/huggingface/<owner>--<repo>/snapshots/<commit>/
```

`.ironmlx-snapshot.json` 记录 `model_type=indextts2_5`、`artifact_role=tts`，以及上游 `required_auxiliary_resources` 到 `compatibility.external_resources`。上游声明 `facebook/w2v-bert-2.0`、`funasr/campplus`；它们是依赖记录，本次下载不递归下载外部仓库，也不宣称已解决运行时依赖。上游 config 中的原始 `.pth` 路径属于转换前配置，以转换 manifest 指定的 safetensors 组件为准。

下载完成后本地模型显示为 TTS，完整性状态可验证，但推理 readiness 为 `unsupported_model_type`。本次不新增 TTS 推理、音频接口或语音克隆支持。

## 验证

```sh
swift test --package-path ironmlx-app --filter 'indexTTS|huggingFaceDownload|dflash2DraftDownload'
```

真实下载测试默认跳过，显式启用后调用生产 `ModelDownloadService` 和 `ProviderModelFileDownloader`，写入当前用户的 App 模型目录：

```sh
IRONMLX_TEST_DOWNLOAD_INDEXTTS=1 \
IRONMLX_TEST_DOWNLOAD_BACKEND=/absolute/path/to/release/ironmlx \
swift test -c release --package-path ironmlx-app --filter indexTTSLiveDownloadUsingAppService
```

### 2026-09-09 本机验证

在 `feat/indextts25`（基于 `dev@e1d1f99ad3926b7c17bd62ac38db170e8d28da11`）上，Release AppCore 真实下载测试完成。固定远端 revision 为 `65644cd70da15309ffeb74aa03f3686bb04e3eb1`，发布 15 个文件，共 4,481,042,335 字节；发布后 `ModelSnapshotVerifier.verify` 再次对全部文件进行 SHA-256 校验并通过。下载约 320 秒，含发布后复核的测试约 322 秒。

初次 Rust 直连遇到 `tls handshake eof`，未发布快照。重试时向进程传入本机已有的 HTTP/HTTPS 代理配置后成功，未更改系统代理设置。使用现有 Release `ironmlx` 下载 helper，其 SHA-256 为 `2b15e6d53c97adaf4cb283bb7f09c46c0429445fca56300aac9fd948b27a3de1`。本次验证的是 Release AppCore 下载服务加生产下载 helper，不是重新打包启动 App 的 UI 验收。

新增 TTS 测试覆盖完整组件发布与不可加载状态、5 种缺资源情况、4 种元数据或辅助文件损坏情况；既有 LLM/DFlash2 下载测试通过。另有 27 项本地扫描、下载队列、预检、断点续传和完整性回归测试通过。
