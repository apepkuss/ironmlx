# IronMLX 用户指南

[English](../user-guide.md)

本指南围绕最新公开版本组织用户使用路径。各章节链接到现有专题文档，不重复复制详细内容。

![IronMLX Dashboard 中文界面展示 Qwen3.8-27B-4bit 与 DFlash2](../images/dashboard-qwen38-dflash2-zh-CN.png)

状态页展示了运行中的本机 endpoint、已加载的 `mlx-community/Qwen3.8-27B-4bit` 模型及其匹配的 DFlash2 draft。

## 安装与启动

1. 从 [IronMLX GitHub Releases 页面](https://github.com/apepkuss/ironmlx/releases/latest)
   下载最新的 Apple Silicon 版 DMG 或 ZIP；
2. 可选：使用同一 Release 页面提供的 `RELEASE-SHA256SUMS` 校验下载的归档文件；
3. 如果下载的是 DMG：打开 DMG，将 `IronMLX.app` 拖到 `Applications` 文件夹图标，
   等待复制完成；关闭 DMG 窗口并推出磁盘映像；
4. 如果下载的是 ZIP：将其解压到本地文件夹；
5. 使用 DMG 时，在 Finder 的“应用程序”中找到并双击 IronMLX；使用 ZIP 时，打开
   解压出的 App。如果还没有可用模型，App 会先打开一个窗口，可在其中下载推荐的[对话](#菜单栏对话)模型；
6. 使用 Dashboard 从 Hugging Face 或 ModelScope 搜索、下载或加载兼容模型；
7. 访问 `http://127.0.0.1:9068/healthz` 检查服务状态。

下载模型前请先阅读[支持的模型](supported-models.md)。架构受支持不代表某个具体快照一定
适合当前设备内存，也不代表其上游许可证条款允许你的使用方式。安装包不包含模型权重，模型仍受
上游模型仓库的许可证和访问条件约束。

如果健康检查失败，请参阅[故障排查](troubleshooting.md)。

## 菜单栏对话

在菜单栏的 IronMLX 图标中选择**对话**，即可在一个小窗口里与本机模型对话。对话使用这台
Mac 上的本机服务；服务已停止时会自动启动。

如果还没有下载可以对话的模型，会改为打开**还没有可以对话的模型**窗口。支持的模型目录中有
适合这台 Mac 内存的入门模型时，窗口会推荐它。点击**下载推荐模型**会从 huggingface.co
下载，连不上 huggingface.co 时改用 hf-mirror.com，下载后加载为默认模型。窗口一直开着
时，模型准备好后会直接进入对话；关闭窗口或点击**在后台继续**后，下载继续进行，菜单栏会在
**对话**下方显示进度，IronMLX 也会请求通知权限，以便模型准备好后提醒你。点击
**自己选择模型**会打开 Dashboard 的**模型 → 模型下载**。

对话优先使用已加载的默认模型，其次是其他已加载的对话模型。没有已加载的对话模型时，会从已
下载的模型中选一个加载（依次为默认模型、入门模型、体积最小的模型），并设为默认模型。窗口
顶部的模型菜单列出已下载的对话模型；切换前会先确认，然后卸载当前默认模型（已固定的不卸载），
再把新模型加载为默认模型；**管理模型…** 会打开**模型 → 模型管理**。对话不会加载或切换 DFlash2 模型：
有模型以 DFlash2 运行时，对话使用该模型，切换需在 Dashboard 中进行。

消息为文本，最长 10 万字符。当前模型能看图时，每条消息可通过**添加图片**按钮、拖入或粘贴
附加最多 4 张图片，每张小于 20 MB。模型支持思考时，**思考**可开启或关闭思考；模型提供
思考档位时，还可选择**低**、**中**（默认）或**高**。

**历史**列出之前的对话，**新对话**开始一个新对话。只保留最近 20 个对话，保存在这台
Mac 上，见[数据位置与卸载](storage-and-uninstall.md)。历史面板中的
**接入 DeepSeek Harness 等外部 Agent →** 会打开 Dashboard 的 **Agent** 页面，并定位到
[DeepSeek Harness](dsh.md) 指南。

## 下载与更新模型

在 **模型 → 模型下载 → 支持的模型** 中按类型浏览，先选择量化版本，再点击 **HuggingFace** 下载。同一模型的量化版本合并展示；选项包含格式和位数，选择后会显示仓库、预计大小、内存提示与验证状态。

支持的模型目录下载公开仓库，不使用 HF Token。ModelScope 按钮在仓库与镜像内容完成验证前隐藏；需要自行指定仓库或凭据时，使用单独的 HuggingFace / ModelScope Tab。

**HuggingFace** Tab 中的 **使用 hf-mirror 下载** 默认关闭。开启后，通过 `https://hf-mirror.com` 搜索仓库并下载模型；鼠标悬停或键盘聚焦 `?` 图标可查看说明。新建 HF 下载任务使用当前选择的来源，排队及暂停任务保留原来源，重启后仍然有效。下载失败不会自动切换来源。将 HF Token 发送给第三方镜像前需要确认；是否经过代理仍取决于网络配置。

已安装版本显示 **已下载 · 检查更新**。检查只比较仓库版本；发现更新后，点击 **下载更新** 才会加入下载队列。IndexTTS 使用已验证资源配置固定的版本。

在 **下载任务** 中，**暂停** 保留已下载数据，重启后仍保持暂停；**继续下载** 使用记录的快照与保留文件。需要凭据的任务可能要求重新输入 Token，Token 不保存到磁盘。**删除任务** 会在确认后删除未完成任务及临时文件；已完成任务的 **清除记录** 保留已安装模型。

下载会按各组件自己的索引选择权重，并纳入主索引之外的独立组件。Breeze 和 Qwen3-TTS 即使选择“仅下载”，也必须包含音频 codec 权重；这不会启用推理支持。仓库声明的可选 MTP 文件会在存在时一起下载。缺少必需组件的旧快照会标记为不完整，重新下载同一版本即可修复；支持文件克隆的文件系统会校验并复用已有主权重，修复通过验证前保留原快照。`Verified` 表示所选文件和组件检查通过，不代表运行时支持该模型。

## 导入已有模型

如果模型已在这台 Mac 上下载，进入 Dashboard 的**模型 → 模型管理**，点击右上角
**导入本地模型**。选择含有 `config.json`、权重和 tokenizer 的模型目录；使用 Hugging
Face 缓存时，应选择 `snapshots/<commit>` 目录，而不是缓存仓库的上层目录。确认页会
显示来源、目标位置和复制大小。IronMLX 会复制文件、校验快照，再将模型加入管理列表；
原目录不会被删除。

有可验证 IronMLX 清单的快照会保留原仓库身份。其他本地目录进入 IronMLX 管理的
`~/.ironmlx/models/standalone/`。导入需要足够空间存放完整副本；成功后，列表中的
模型可按其显示的 ID 加载和管理。特殊模型可能还需要准备额外运行资源。

## 加载进度与取消

在 Dashboard 中加载模型时会打开进度面板，列出各加载阶段（如**读取权重**、**预热**），
并显示已读取字节、已用时间和预计剩余时间。点击**后台继续**可关闭面板；切换页面或关闭
后重新打开 Dashboard 时，加载不会中断，侧边栏会显示进度（如**正在加载 · 42%**）。
重新加载等其他加载也会显示在侧边栏；点击侧边栏的进度即可打开面板。

点击**取消加载**可停止加载。取消新加载会释放其占用的内存；取消重新加载会继续使用
当前已加载的版本；取消切换到 DFlash2 会恢复之前的配置，并重新加载原来的模型。如果
取消 App 启动时正在恢复的 DFlash2 模型，后端会以普通模式重启，不加载模型；该模型的
DFlash2 设置会保留，之后可以再次以 DFlash2 加载。

## 权重格式标签

在 App 的“模型管理”列表中，量化检查点显示检测到的量化格式或位数；未量化检查点
显示检测到的权重数据类型，例如 `FP16` 或 `BF16`，Tooltip 会说明“未量化”并给出
规范化后的 `dtype`。如果模型元数据和 safetensors 头都无法提供有效值，则显示
“未知”。这些标签描述的是存储权重，不是运行时计算精度。

## 首个 API 请求

加载 `mlx-community/Qwen3.8-27B-4bit` 后，可使用以下命令验证完整的本地 Responses 请求：

```bash
curl -fsS http://127.0.0.1:9068/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/Qwen3.8-27B-4bit","input":"Reply with exactly IRONMLX_OK.","store":false,"max_output_tokens":16,"temperature":0,"stream":false}'
```

完成的响应中应包含 `IRONMLX_OK`。使用其他已加载模型时，将 model ID 替换为
`GET /v1/models` 返回的精确值。当前 RC 对该模型还显示 DFlash2，使用匹配的
`z-lab/Qwen3.8-27B-DFlash2` draft；加速细节和限制见[支持的模型](supported-models.md)。

## Agent 配置

将 IronMLX 作为 Agent 的本地推理后端，按应用选择配置指南。

| Agent | 配置方式 |
| --- | --- |
| [Hermes Agent](hermes-agent.md) | 本地 Responses 服务 |
| [OpenCode](opencode.md) | CLI / Desktop 一键配置，使用原生 Chat Completions 和已识别模型的思考档位 |
| [oh-my-pi](oh-my-pi.md) | 本地 OpenAI 兼容服务 |
| [DeepSeek Harness（DSH）](dsh.md) | [Desktop App](dsh.md#dsh-desktop-app-配置) 或 [CLI](dsh.md#dsh-cli-配置) |

## 语音合成

### 声音配置

加载语音模型后，打开 Dashboard 的“声音”页面，导入参考音频并分配稳定的声音 ID。
调用[语音 API](audio-speech-api.md)时使用 `voice` 指定该 ID，也可以在请求中通过
`ref_audio` 直接提供参考音频。OpenAI-compatible 客户端可通过
`GET /v1/audio/voices` 获取已启用的声音。

Dashboard 试听的是声音配置中保存的参考录音。合成音频由调用方或
[独立示例客户端](../../examples/speech-client.swift)播放。声音配置保存在
`~/.ironmlx/audio/voices/`；修改显示名称不会改变声音 ID。
请求参数及 WAV/PCM 示例见[语音 API](audio-speech-api.md)。

### 资源就绪

App 自动准备语音资源。如果模型列表提示资源缺失，点击“准备语音资源”。
中断后可以重试，已验证的模型文件和资源缓存会复用。首次准备需要联网；
完整资源就绪后可以离线合成。App 保存资源配置，并在重启时恢复此前加载或固定的模型。

### 模型设置

在“模型管理”中点击语音模型对应的齿轮按钮。模型基础信息为只读；
只有模型文件提供有效值时，才显示对应的配置字段。

同一窗口还提供六项可编辑的运行策略，默认值和允许范围以设置窗口显示为准。

| 设置项 | 默认值 | 允许范围与作用 |
| --- | --- | --- |
| 队列超时 | 60 秒 | 正数，最长 24 小时；限制请求等待执行许可的时间 |
| 首段音频超时 | 120 秒 | 正数，最长 24 小时；从请求获得执行许可后开始计时 |
| 执行超时 | 900 秒 | 正数，最长 24 小时；从请求获得执行许可后开始计时 |
| 慢速消费者超时 | 30 秒 | 正数，最长 24 小时；限制输出被消费方阻塞的时间 |
| 最大输出时长 | 600 秒 | 正数，不超过运行时配置上限；包含分段之间的静音 |
| 分段 Token 数 | 120 | 6–120 的整数；每个语音合成分段的 Token 预算 |

如果执行配置不可用，上述控件和保存按钮保持禁用。保存后，策略按模型持久化；
如果模型已加载，App 会在当前任务空闲后重载模型以应用新值，后续加载和重启恢复
也会继续使用这些设置。这些是服务运行策略，不是模型硬限制或延迟保证。对应的
`audio.execution` 字段和调度语义见[语音 API](audio-speech-api.md)。

## 按任务选择文档

- [Embedding API](text-embeddings.md)：EmbeddingGemma 2 BF16／affine4 的输入格式、限制与 Dashboard 向量指标。
- [Laya 决策 API](laya-systemone-api.md)：在 App 中加载 Laya，并供外部应用调用。
- [OpenAI Decisions API](openai-decisions-api.md)：用官方 OpenAI SDK 调用 Laya。
- [支持的模型](supported-models.md)：具体模型、类型、权重格式、支持能力与加速类型；
- [开发者指南](developer-guide.md)：API 快速开始、CLI 配置、源码开发与贡献入口；
- [文本与视觉 API](text-vision-api.md)：请求字段、流式输出、结构化输出、推理、图片和错误；

## 隐私、安全与本地数据

- [隐私与网络边界](privacy.md)
- [安全边界](security-boundary.md)
- [数据位置与卸载](storage-and-uninstall.md)
- [诊断信息导出](diagnostic-bundle.md)
- [自动更新](automatic-updates.md)
- [模型权利边界](model-license-boundary.md)

## CLI 与高级配置

- [DFlash2 配置与使用](dflash2-server-api.md)
- [Qwen MTP 配置与使用](mtp-server-api.md)
- [CLI 多模型服务（EnginePool）](engine-pool.md)
- [调度性能校准](scheduler-profile-v5.md)

## 故障排查与版本信息

- [故障排查](troubleshooting.md)
- [Known Issues](known-issues.md)
- [0.3.0 发布说明](release-notes/0.3.0.md)
- [0.2.0 发布说明](release-notes/0.2.0.md)
- [0.1.0 发布说明](release-notes/0.1.0.md)

源码构建、测试、贡献和发布工程请从[从源码构建](building-from-source.md)及[参与开发](contributing.md)开始。
