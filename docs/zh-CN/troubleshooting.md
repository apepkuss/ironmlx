# 故障排查

[English](../troubleshooting.md)

## 从现象开始检查

| 现象 | 检查与处理 |
| --- | --- |
| App 无法启动 | 确认是 Apple Silicon、macOS 26.4 及以上，记录系统提示和 App 版本。源码构建请执行[构建检查](building-from-source.md)。 |
| 模型下载中断 | 在 Dashboard 重试同一模型，不要手动移动未完成的 `.partial` 文件。 |
| 模型被拒绝加载 | 阅读 Dashboard 错误提示，检查[模型加载](#模型加载)、可用磁盘、内存和下载完整性。 |
| API 连接失败 | 使用 Dashboard 显示的地址和端口；本机模式只能从当前 Mac 访问。参见 [API 快速开始](developer-guide.md#api-快速开始)。 |
| LAN 返回 401 | 重新复制当前 API Key，并作为 Bearer token 发送。 |
| LAN 证书错误 | 安装或指定 App 导出的 CA，连接 App 显示的 IP。参见 [LAN 配置](security-boundary.md)。 |
| 图片被拒绝 | 使用 JPEG/PNG/WebP base64 内容，并检查[图片限制](security-boundary.md)。 |
| 响应慢或内存占用高 | 缩短上下文、减少并发并卸载不用的模型；比较时保持模型和设置一致。 |
| 缓存占用过大 | 在 Dashboard 调整缓存容量；手动删除缓存前先停止后端，路径见[数据位置](storage-and-uninstall.md)。 |
| 后端反复退出 | 在“日志 → 故障历史”查看原因和恢复结果，检查内存、模型文件缺失和配置错误。 |
| 检查更新失败 | 检查网络后稍后重试，参见[自动更新](automatic-updates.md)。 |

## 模型加载

先对照[支持的模型](supported-models.md)，查看 Dashboard 的具体错误提示，确认
模型快照完整，并且磁盘空间和内存充足。上下文长度、并发、缓存和辅助模型都会
增加内存需求；内存不足时，减少这些设置或选择更小的模型。

### 权重格式

支持范围取决于具体模型加载器，不代表每个模型都支持下表中的所有格式。
优先使用模型清单中为该模型列出的权重格式。

| 权重格式 | 支持范围 |
| --- | --- |
| 未量化 | 数据类型必须受对应模型加载器支持 |
| Affine | 2/4/5/6/8-bit；group size 32/64/128 |
| OptiQ mixed-bit | 2/4/8-bit；group size 64；需要有效的 `optiq_metadata.json` |
| MXFP4 | 4-bit；group size 32 |
| MXFP8 | 8-bit；group size 32 |

### 架构标识参考

排查模型兼容性时，可核对配置中的 `model_type`。标识匹配只是必要条件；配置、tokenizer、模板、权重布局和量化元数据也必须兼容。下载预检或加载校验仍可能拒绝不兼容文件。

| 模型族 | `model_type` |
| --- | --- |
| Qwen 3.5 Dense；采用相同架构的 Qwen 3.6 / 3.8 Dense | `qwen3_5` |
| Qwen 3.5 / 3.6 MoE | `qwen3_5_moe` |
| Gemma 4 / Gemma 4 Unified | `gemma4`、`gemma4_unified` |
| GLM-4 MoE Lite | `glm4_moe_lite` |
| Llama GQA Dense / 兼容的 MiniCPM5-1B | `llama` |
| MiniCPM-V 4.6 | `minicpmv4_6` |
| DiffusionGemma | `diffusion_gemma` |
| Qwen Image 2.1 | `qwen_image_2_1` |

## 日志与诊断

需要更多信息时，在设置中调整日志级别。修改会立即保存，不重启后端，也不保存其他尚未提交的设置。
日志页的级别筛选只改变已有记录的显示，不会生成更多日志。
默认级别为 INFO，“全部”包含 TRACE；提高详细程度无法补回此前被过滤的记录。
应用级别失败或后端繁忙时，请按界面提示稍后重试。

使用“导出诊断信息”保存本地脱敏 ZIP，分享给维护者前先自行检查。
包含哪些信息见[诊断导出](diagnostic-bundle.md)。

## 恢复与内存压力

故障历史支持筛选、查看、清除和 JSON 导出。清除历史不会停止后端，也不会删除普通日志和模型设置。
主动停止不会被记录为崩溃；连续崩溃可能暂停自动恢复。
内存保护可能拒绝新请求、回收缓存或卸载未固定的空闲模型。请先缩短上下文或减少并发再重试，压力拒绝本身不代表模型不受支持。

## 提交可复现的问题

提供 App 版本、macOS 和芯片型号、模型 ID 与 revision、量化、具体操作和安全的错误文本。
性能问题还应说明上下文/输出长度、并发及缓存冷热状态，并提供多轮结果。
不要公开私有提示词或凭据，详见[支持](support.md)。

独立 CLI 可使用 `RUST_LOG`；App 启动的后端使用已保存的 App 日志级别。API 控制接口见[日志级别管理](service-api.md#日志级别管理仅本机)。
