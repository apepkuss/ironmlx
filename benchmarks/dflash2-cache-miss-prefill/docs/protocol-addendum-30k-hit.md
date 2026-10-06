# 补充协议：30K 命中 TTFT 双峰诊断与文档分叉 token 证据

2026-10-06 写定，在下列运行之前。`docs/protocol.md` 与 `results/timing-v1` 保持不变，原判据不放宽、不重判。

## 1. 文档分叉 token 证据（`scenarios.py --set docfork`，标签 `scenarios-docfork-v1`）

原因：`scenarios-main-v1` 中 30K 文档分叉的 token 记录被截断（服务关闭时诊断尚未写完），只验证了文本与结束原因。截断记录保留。

**会话：**
- B-on、C-on、B-off、C-off，二进制同主场景（b9c7b4fb / f007c386），各一个新进程；
- 每个会话两个预热，然后与主场景相同的 30K code-1 链：零命中 → 完全重复 → 追加 → 分叉 → 文档分叉；
- 关闭服务前等待 token 记录全部写完。

**要求：**
- C-on 文档分叉与 B-on、B-off、C-off 逐 token 相同，结束状态相同；
- 两侧命中 token 相同；
- 链上其他步骤与主场景的同侧结果一致（文本哈希）。

## 2. 30K 命中双峰诊断（`diagnose_30k_hit.py`，标签 `diag30k-hit-v1`）

**二进制（仅诊断）：**
- `ironmlx-baseline-diag2-f2fc817d`：候选源码将 `cold_single_prefill` 恒置为 false，指纹恢复 v1，行为等同基线；
- `ironmlx-candidate-diag2-c40265c9`：候选源码；
- 两者都带细分诊断：命中路径的 load / restore_graph / materialize / first_logits 计时，每个标记点记录 MLX active、MLX cache 和进程 footprint，均为计数器读取，不加 eval；另记录 actor 接纳时的压力等级。

**环境：** App 参数与 App 环境，另加 `IRONMLX_DIAGNOSTIC_DFLASH2_PREFILL_PHASES=1`。

**顺序：** 8 个会话，Bd Cd Cd Bd Bd Cd Cd Bd。

**每个会话：**
1. 与 `run_timing.py` 完全相同的步骤：首请求、两个预热、5 个短零命中、2 个短重复、追加、30K 零命中、30K 重复（记为 R1）；
2. 再做 3 次 30K 完全重复（R2–R4，间隔 1 s）；
3. 等 10 s 后再做 1 次（R5）。

**读法（事先固定，不做通过判断）：**
1. 每个 30K 重复请求按服务端 prefill 是否 > 60 ms 分为慢、快两类，统计每臂、每个位置（R1–R5）的慢请求数。
2. 比较慢、快两类在各阶段（load、restore_graph、materialize、first_logits）的耗时，找出差异集中的阶段。
3. 检查慢请求与接纳压力等级、MLX cache 字节和 footprint 的对应关系。
4. 比较 Bd 与 Cd 在 R1 上慢请求的比例；样本少（每臂 4 个 R1），只作描述。

**后续：**
- 只有当诊断显示候选引入了差异（例如某阶段只在 Cd 变慢，或慢请求比例明显偏向 Cd，并有机制解释）时，才做最小修复。
- 修复后用重新固定的协议验证，原 `timing-v1` 结果保留。
- 若诊断显示两臂机制相同、与改动无关，则如实报告原判据仍未满足。是否接受由 Boss 决定，不自行重判。
