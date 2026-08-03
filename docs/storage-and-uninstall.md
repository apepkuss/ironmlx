# 数据位置与卸载

## 默认数据位置

| 内容 | 路径 |
| --- | --- |
| App 配置 | `~/.ironmlx/config/app_config.json` |
| Hugging Face / ModelScope 模型快照 | `~/.ironmlx/models/` |
| 分页 SSD 前缀缓存 | `~/.ironmlx/cache/paged_prefix_cache/` |
| App 与后端日志 | `~/.ironmlx/logs/` |
| 模型参数 | `~/.ironmlx/model_params.json` |
| 后端故障记录 | `~/.ironmlx/incidents/backend-incidents.json` |
| 调度器 profile store | `~/.ironmlx/scheduler-profiles/` |
| 调度器校准报告 | `~/.ironmlx/reports/scheduler-autotune/` |

用户在 Dashboard 中配置自定义 cache directory 后，缓存会写到该目录而不是默认
路径。LAN API Key、CA 与 TLS 私钥由 macOS Keychain 管理，service 标识为
`com.ironmlx.lan-security.v1`。

## 卸载 App

1. 退出 IronMLX，确认后端进程已停止；
2. 删除 `IronMLX.app`；
3. 若不保留模型与配置，删除 `~/.ironmlx`；
4. 若使用过自定义 cache directory，单独删除该目录；
5. 若启用过 LAN 模式，在 Keychain Access 中删除 IronMLX LAN security 条目。

删除 `~/.ironmlx` 会永久移除已下载模型、未完成下载、配置、日志、缓存和报告。
如需保留模型，先备份 `~/.ironmlx/models`，不要直接删除整个目录。
