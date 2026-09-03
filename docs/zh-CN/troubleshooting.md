# 故障排查

| 现象 | 检查与处理 |
| --- | --- |
| App 无法启动 | 确认是 Apple Silicon、macOS 26.2+；本地构建先运行 `scripts/verify-app-bundle.sh dist/IronMLX.app` |
| 模型下载中断 | 在 Dashboard 重试同一模型；下载器会基于不可变 commit 与 Range/ETag 恢复，避免手动移动 `.partial` 文件 |
| 模型被拒绝加载 | 查看 Dashboard readiness 与 `~/.ironmlx/logs/backend.log`；确认架构、量化元数据、磁盘、内存和快照完整性 |
| API 连接失败 | 确认 App endpoint 和端口；local 模式只能从本机访问；先请求 `/health` 再请求 `/healthz` |
| LAN 返回 401 | 使用 `Authorization: Bearer ...`，重新复制或轮换 API Key；所有 LAN 路由都需要认证 |
| LAN TLS 失败 | 导入/指定 App 导出的 CA，并连接证书包含的具体 IP；不要关闭证书校验 |
| 图片请求被拒绝 | 只使用 JPEG/PNG/WebP base64；不要传 HTTP/HTTPS URL；检查请求体、图片字节和像素限制 |
| 首 token 很慢或吞吐异常 | 确认使用 Release 构建和 NAX-enabled MLX；检查内存压力、并发、prefill chunk、KV/前缀缓存及 scheduler profile |
| 缓存占用过大 | 在 Dashboard 设置容量或清理 `~/.ironmlx/cache/paged_prefix_cache`；清理前先停止后端 |
| 后端反复退出 | 在 Dashboard 的“日志 → 故障历史”查看结构化原因、恢复动作和脱敏日志尾部；“导出故障记录”保存当前筛选 JSON，“导出诊断信息”保存完整诊断 ZIP |

报告问题时至少附带 App/CLI 版本、macOS 与芯片型号、模型不可变 commit、量化
信息、复现请求和已脱敏日志。性能问题还应固定并发、prompt/output tokens、缓存
冷热状态，并提供多轮结果。
