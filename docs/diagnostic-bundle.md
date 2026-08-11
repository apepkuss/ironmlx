# 隐私安全的诊断信息导出

Dashboard 的“日志”页标题栏提供“导出诊断信息”。该操作通过 App 原生消息在本机收集数据，并只将 ZIP 保存到用户在 macOS 保存面板中明确选择的位置；它不经过 backend HTTP API，不会被 LAN 客户端触发，也不会上传网络或遥测。

## 固定格式

诊断格式 schema version 为 `1`，ZIP 条目顺序固定：

1. `manifest.json`
2. `system.json`
3. `runtime-health.json`
4. `models.json`
5. `incidents.json`
6. `logs/app.log`
7. `logs/backend.log`

`manifest.json` 记录 App/backend/MLX 构建身份、分发通道、backend 在线状态、签名和公证状态，以及每个条目的生成状态、字节数和截断标记。backend 离线或 health 请求在 2 秒内失败时，归档仍会生成，`runtime-health.json` 使用稳定错误码报告失败；收集过程不会启动、重启或停止 backend。

## 数据白名单

- `system.json`：macOS 版本和 build、Apple 芯片、物理内存、App 架构、签名有效性、Developer ID 与 stapled ticket 状态。
- `runtime-health.json`：backend 版本和模式、设备类别、调度器、内存、Active KV 统计及已加载模型的运行状态。原始模型路径和 Active KV 存储目录不进入 DTO。
- `models.json`：provider、repo ID、immutable commit、requested revision、量化、模型类别/能力、loaded/active revision、完整性状态和验证时间。
- `incidents.json`：只复用 `BackendIncidentStore` 规范化后的结构化故障数据。
- `logs/*`：只读取两个当前日志文件的受限尾部。

不读取或复制配置原文件、环境变量全集、Keychain、请求正文、权重、tokenizer、完整模型配置或 sidecar 内容，也不收集用户名、主机名、序列号、Apple ID、MAC 地址等稳定身份信息。

## 双层隐私边界

第一层使用结构化白名单决定可进入归档的字段；第二层由 `DiagnosticPrivacy` 对所有文本和最终 JSON 数据执行统一脱敏。该组件同时供故障记录使用，覆盖 prompt/messages/input/body、function/tool 参数、Authorization/Bearer、Cookie、HF token、LAN API Key、密码/secret、用户名和用户主目录路径。

## 容量与安全写入

- models：512 KiB，最多 256 个版本记录。
- incidents：512 KiB，最多保留层允许的 20 条。
- App 日志：512 KiB / 4,000 行。
- backend 日志：1.5 MiB / 10,000 行。
- 全部未压缩条目：3.5 MiB；最终 ZIP：4 MiB。

日志通过 `O_NOFOLLOW` 打开，仅接受普通文件，并直接从尾部做有界读取。ZIP 从已脱敏的内存数据按固定顺序生成，不遍历诊断数据中的目录或 symlink。最终文件先在目标目录创建权限为 `0600` 的隐藏临时文件，完成写入和 `fsync` 后原子发布；取消或错误会删除临时文件。

“故障历史”中的“导出故障记录”仍按当前筛选条件导出 JSON，与完整诊断 ZIP 是两条独立链路。
