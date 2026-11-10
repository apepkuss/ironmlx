# M1 Pro/Max GPU 推理通道

`ironmlx serve` 与 `ironmlx generate` 新增 `--m1-profile auto|off`，默认
`auto`，标准 App 启动也自动采用此默认值，无需额外开关或启动参数；可显式使用
`off` 关闭专用通道。`auto` 只为真实 M1 Pro/Max 上的 Qwen3.8-27B affine4/group64
激活逐模型执行计划，其他设备、模型与格式沿用原路径。M5 的开关、默认参数、专用内核及全局
Metal 设置保持原样。

首版使用已有原生 GPU 算子：单行投影走 product-stable QMV；2–16 行走
`qmv_fast_wide`，Pro 每组最多 2 行、Max 最多 4 行；更长 prefill 保留原生 MLX
affine4 QMM。GatedDeltaNet 复用带 shape/dtype 检查的融合 q/k norm 与 gate
算子。CPU stream 与不兼容操作数回退原路径，不增加常驻权重副本或整层解量化空间。
已有精确投机验算及 batch/position/product isolation 的投影作用域优先，宽 verify
不会因为行数超过 16 就误入普通 prefill QMM。

普通、MTP 和 DFlash2 target 共享同一模型持有的计划；DFlash2 draft 保持原样。
计划只在文本 forward 和 target logit projection 的作用域内生效，嵌套的 inactive
模型会屏蔽外层状态，返回、错误和 unwind 后恢复。M1 不写全局环境变量，也不修改
M5 profile 的 `OnceLock`。

`healthz.m1_profiles` 只在真实 M1 Pro/Max 上报告各因果模型的状态、设备、指纹和
narrow/prefill/fallback 投影调用次数，包含 `off` 和 `unsupported_model` 状态。
其他设备完全省略该字段，聚合及嵌套健康响应也遵守此规则。硬件识别独立于开关并缓存结果；
非 M1 设备不创建 M1 计划，也不准备 M1 权重、内核或 workspace。计数反映构图时的路由选择，不是 GPU
耗时。DFlash2 指纹和普通/MTP 的 SSD 前缀缓存命名空间包含 M1 配方，内存前缀缓存
仍由各 actor 独立持有。

现有 load、cold materialization、prefill、KV、workspace 内存保护均保留，不改变
内存上限、常驻策略、chunk 大小或投机预算。ANE 尚未接入，M1 报告
`ane: not_implemented`；现有 M5 的 `--ane` 行为不受影响。

Pro/Max 行分组是按硬件与模型资格自动选择的配方，不代表已验证性能提升。专用 M1 tiled prefill
GEMM、attention 内核与进一步调参留待后续。
