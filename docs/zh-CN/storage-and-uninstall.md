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

## 选择清理范围

- **只删除 App**：退出 IronMLX，确认后端已停止，再删除 `IronMLX.app`。模型和配置保留。
- **保留数据重装**：按上一步删除 App 后安装替换版本，保留 `~/.ironmlx` 和 Keychain 条目；这不等同于干净安装。
- **彻底清理**：退出 App 后删除 App、`~/.ironmlx` 及单独配置的缓存目录；如启用过 LAN，在“钥匙串访问”中删除 IronMLX LAN security 条目。

删除 `~/.ironmlx` 会永久移除其中的模型、未完成下载、配置、日志、缓存和报告。需要保留时先备份；自定义目录需单独核对。
这里列出 App 管理的数据位置，不承诺清除 macOS 的历史信任记录或全部系统偏好。

## 参考：故障历史保留

`~/.ironmlx/incidents/backend-incidents.json` 默认保留最近 20 条事故，总量上限 1 MiB，单条日志尾部最多 32 KiB；单次故障 JSON 导出最多 512 KiB。损坏或异常超限的历史按空记录处理，不阻塞启动。Dashboard 未读标记属于 WebKit 本地存储。
