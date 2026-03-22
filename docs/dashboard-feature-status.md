# Dashboard 功能实现状态

> 最后更新：2026-03-22

## 状态页 (Status)

- [x] 服务器状态、运行时间、模型信息 — 从 `/health` API 实时获取
- [x] GPU 内存实时曲线图 — Canvas 绘制，定时轮询
- [ ] GPU 提示文字中的内存大小（如 "24GB"）— 当前硬编码，未读取设备实际内存

## 模型管理页 (Model Manager)

- [x] 本地模型列表扫描（~/.ironmlx/models/）
- [x] 模型加载 / 卸载（通过后端 API）
- [x] 模型删除（单个 + 批量）
- [x] 默认模型切换（radio 按钮 → AppConfig）
- [x] 模型参数设置弹窗 UI
- [ ] 模型参数设置弹窗"保存"按钮 — 未持久化参数到后端
- [ ] 模型参数值（Temperature、Top P 等）— 修改后不会应用到推理请求

## 模型下载页 (Model Download)

- [ ] HuggingFace 搜索框 — 纯 UI，无搜索功能
- [ ] 手动下载（Model ID + Token 输入框）— 纯 UI
- [ ] 下载按钮 — 未触发实际下载逻辑

## 基准测试页 (Benchmark)

- [ ] 运行基准测试按钮 — 纯 UI，无功能
- [ ] 测试指标选择（pp1024、pp4096 等）— 纯 UI
- [ ] 模型选择下拉框 — 已同步模型列表，但无法触发测试
- [ ] 测试结果展示区 — 纯 UI

## 日志页 (Logs)

- [x] 日志内容读取 — 通过 bridge 从后端获取
- [x] 日志文件切换（下拉框）
- [x] 刷新按钮 + 自动滚动开关
- [ ] 日志级别过滤下拉框 — 纯 UI，无过滤功能
- [ ] 搜索框 — 纯 UI，无搜索功能

## 设置页 (Settings)

### 已实现
- [x] Port — 保存到 AppConfig，重启生效
- [x] Language — 保存到 AppConfig，实时切换菜单栏 + Dashboard 语言
- [x] Theme — 保存到 AppConfig，实时切换深色/浅色模式
- [x] 保存设置按钮 — 收集所有设置值，写入 app_config.json
- [x] 重启确认 — 检测到需要重启的设置变更后弹出确认并重启服务

### 未实现
- [ ] Host（下拉框）— 保存了但 AppConfig 中无 host 字段，后端未使用
- [ ] Log Level — 保存了但后端不读取该配置
- [ ] Memory Limit (Total) 滑块 — 纯 UI
- [ ] Memory Limit (Model Only) 滑块 — 纯 UI
- [ ] Hot Cache Limit 滑块 — 纯 UI
- [ ] Cold Cache Limit 滑块 — 纯 UI
- [ ] Enable Cache 开关 — 纯 UI
- [ ] SSD Cache Directory 输入框 — 纯 UI
- [ ] Max Sequences 输入框 — 纯 UI
- [ ] Completion Batch Size 输入框 — 纯 UI
- [ ] Initial Cache Blocks 输入框 — 纯 UI

## 对话页 (Chat)

- [x] Moss Desktop 检测（/Applications/Moss.app）
- [x] Moss Desktop 启动（open -a Moss）
- [x] 配置指南展示（Chat URL + 一键复制）
- [x] 未安装时显示下载链接（GitHub Releases）
