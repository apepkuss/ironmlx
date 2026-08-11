# 故障历史与恢复解释

IronMLX App 将后端异常退出及其自动恢复生命周期保存为结构化事故历史。该功能复用 `BackendIncidentRecord`，不会替代普通运行日志，也不会改变实时 `onServerCrash` 故障通知。

## 数据与保留边界

- 默认文件：`~/.ironmlx/incidents/backend-incidents.json`
- 默认保留：最近 20 条事故
- 存储上限：1 MiB；达到上限时优先丢弃最旧事故
- 单条日志尾部：最多 32 KiB
- 单次诊断导出：最多 512 KiB
- 写入与清除均使用原子文件替换
- 文件损坏或异常超限时，读取结果为空，不阻塞 App 启动

事故记录不保存 prompt、完整请求体或凭据。日志尾部在持久化前执行脱敏；查询和导出使用持久化层再次规范化后的数据。Authorization、API key、token、prompt/request body 字段及用户主目录前缀会被移除或替换。

## DashboardBridge 本地接口

以下接口只在 App 内嵌 DashboardBridge 中使用，不暴露为后端或 LAN HTTP API：

- `GET /admin/api/incidents`
- `GET /admin/api/incidents/{id}`
- `GET /admin/api/incidents/export`
- `POST /admin/api/incidents/clear`

列表和导出支持 `status`、`model`、`reason`、`from`、`to`、`limit`。未知状态、原因、日期或非正整数上限会返回稳定错误结构。导出必须由用户点击触发，并通过 macOS 保存面板选择本地目标；不会自动上传。

`clear` 只将事故历史原子替换为空数组，不会停止后端、修改自动恢复状态、删除普通日志或变更模型配置。

## 恢复解释

恢复解释优先使用 `BackendModelRecoveryFailure.reason`，不解析任意日志文本。稳定原因包括：

- `memory_insufficient`
- `model_limit_reached`
- `model_files_missing`
- `model_snapshot_invalid`
- `incompatible_configuration`
- `unknown_model_load_failure`
- `crash_loop_breaker`

每个模型失败仍保留后端结构化错误码、恢复阶段、是否可重试和建议动作。Dashboard 根据当前界面语言提供用户说明和建议动作。

## UI 行为

日志页默认打开“运行日志”，原有日志来源、级别筛选、搜索、自动刷新和日志导出保持不变。“故障历史”提供筛选、详情、清除及“导出故障记录”；该 JSON 导出继续携带当前筛选条件。页面标题栏的“导出诊断信息”是独立的完整 ZIP 导出，不受故障筛选影响。未查看事故通过本地 `localStorage` 水位计算徽标；切换到故障历史并成功读取后标记为已查看。

实时故障和恢复通知仍由全局 `onServerCrash` 横幅立即展示，不依赖当前日志子 Tab。

## 已知限制

- 未读状态属于当前 Dashboard WebKit 数据存储，不在多设备或多用户间同步。
- 事故历史只记录 IronMLX App 监管到的后端异常退出；用户主动停止不会生成事故。
- 故障信息 JSON 只包含受容量限制的结构化事故和脱敏日志尾部；完整诊断 ZIP 的独立白名单见 [隐私安全的诊断信息导出](diagnostic-bundle.md)。
