# 动态日志级别

[English](../runtime-log-level.md)

设置页的日志级别会单独即时应用并保存，不重启后端，也不保存页面上其他尚未提交的设置。它控制 App 与 IronMLX 后端实际生成的诊断日志；日志页的级别选择器仅筛选已有记录的显示。

选项保留全部、DEBUG、INFO（默认）、WARNING、ERROR。全部包含 TRACE；旧配置中的 `TRACE`、`WARN` 分别兼容为 `ALL`、`WARNING`。后端第三方依赖日志最多开放到 WARNING（选择 ERROR 时为 ERROR），避免打开项目详细诊断时同时启用依赖的网络内容转储。

App 启动时应用已保存级别，并通过 `IRONMLX_LOG_LEVEL` 传给每次启动的 helper。独立 CLI 未指定 App 级别时仍沿用 `RUST_LOG`。Swift DEBUG 日志覆盖后端状态转换、下载队列调度和级别应用；开启级别不会自动为没有日志调用的路径生成诊断信息，也无法补回此前被过滤的日志。

## 应用流程

App 先查询后端级别，以进程 ID 和版本号为前提更新过滤器，再仅保存 `log_level`，最后调整自身过滤器。后端停止时不发送 HTTP 请求，保存值在下次启动时应用。启动、恢复、停止或其他设置应用过程中拒绝变更，供用户稍后重试。

应用期间禁用控件。失败后恢复已保存的选项并说明原因。响应丢失或配置写入失败时，App 查询实际状态，并在条件仍匹配时恢复后端原级别。若进程或版本已经变化，或回退无法确认，界面明确报告恢复未确认，不将其显示为应用成功。

## 本机管理接口

`GET /admin/api/log-level` 返回 `level`、`revision`、`process_id`。独立 CLI 的自定义 `RUST_LOG` 过滤器以空 `level` 表示，直到被规范级别替换；App 要求初始级别为规范值。

`POST /admin/api/log-level` 接收：

```json
{"level":"DEBUG","expected_revision":0,"expected_process_id":12345}
```

成功后返回新状态。版本号或进程 ID 过期返回 409；无效 JSON 和不支持的级别被拒绝；运行时控制不可用返回 503。接口仅挂在回环监听端，LAN 监听端不提供该路由，即使携带有效 LAN Key 也不能访问。

## 验证

定向测试覆盖实际文件日志、动态过滤输出、配置隔离、保存失败、响应丢失、回退失败、后端停止和 Dashboard 消息链路。显式启用的 `runtimeLogLevelLiveHelperAndStreamingInference` 测试使用构建后的 helper 和已有模型快照，检查 SSE 推理中五档切换、PID 不变、流正常结束，以及计划重启后恢复已保存级别。由于生产 helper 限制每个 macOS 用户只有一个实例，执行前需要退出当前 App。
