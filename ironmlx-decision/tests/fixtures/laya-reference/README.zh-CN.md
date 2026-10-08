# Laya 参考验证

[English](README.md)

本文保留原生 `aac6fef/laya-multilingual-mlx` 运行时的开发验收要求与历史测量。
面向用户的请求与响应字段统一维护在 [System One API 指南](../../../../docs/zh-CN/laya-systemone-api.md#请求)中。

## 参考检查点

已验证的 Hugging Face revision 为
`f2b4faf51023039425946074e2cf1361d2db11d5`；其 `model.safetensors`
SHA-256 为 `7fc5834af4d8fdfb268d272a9d1a66e5819a0daac98241651c4c888cc43adff1`。

## 验收边界

原生 API 必须解析未经修改的固定版本检查点，保留 tokenizer 与
问题格式，并在同一组固定输入上，与参考 `laya-mlx` 运行时的所选选项、
score、noul、概率向量和输入 token 数保持一致。测试应覆盖单一及混合问题
批次、结构化 state/criteria、非英文文本、空或长 state，以及选项预算超限拒绝。
同步的端到端延迟和峰值内存必须单独测量，不能引用参考项目发布的数值代替。
App 必须将该产物识别、下载并校验为决策模型。可通过
`ironmlx decide --model-dir <snapshot> --request <request.json>` 执行原生推理；
App 管理的 API 和独立服务见[本地 System One API](../../../../docs/zh-CN/laya-systemone-api.md)。

在仓库根目录，可使用以下命令重新运行[真实模型测试](../../laya_reference_parity.rs)：
`LAYA_MODEL_DIR=<snapshot> LAYA_METALLIB=<mlx.metallib> cargo test -p ironmlx-decision --test laya_reference_parity`

该测试现已覆盖 FP16 和 FP32、批大小 1/2/16，以及前缀缓存开启
和关闭。FP32 参考输出使用相同的上游 revision
`0a859518634112655cb97c745dbf04f5191aaf13` 与检查点，通过
`Agent(model_path, dtype="float32", batch_size=16)` 生成。

## 历史测量

在开发用 Apple Silicon 主机上（2026-09-24），三问题示例的三次串行冷进程
运行中，`ironmlx decide` 的墙钟时间中位数为 1.93 s、峰值 RSS 为 1,305 MB；
使用 `/usr/bin/time -l` 测得的 `laya-mlx` 对应数值为 0.58 s 和 1,240 MB。
这些测量包含启动与模型加载，不是预热后的吞吐量比较。测量使用最初的
顺序原生路径；当前原生运行时会批量处理问题，默认批大小为 16，
这些历史数据不代表当前性能。
