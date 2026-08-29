# Qwen3.8-27B affine8 DFlash2 性能归档

日期：2026-08-29

状态：Greedy 汇总结果已更新。本记录只覆盖当前分支的
Qwen3.8-27B-8bit target 与匹配 DFlash2 draft；B1/B2/B4 的 64 和 256
tokens 场景均为正收益。数据执行版本为 `dd37fde`；采样路径、KV profile
和完整逐组原始报告不包含在本归档中。

## 环境与测量契约

| 项目 | 配置 |
|---|---|
| 当前分支 | `feat/qwen3-8-27b-8bit@60f02b65ab0f3b0179adddf80f958944d6ff8655` |
| 数据执行 commit | `dd37fde67af113501f40ce893b55e7a5609907e1` |
| MLX | `fix/nax-qmm-product-shape-main@73ad5df20cb30be4192e5c4d0ae8130674773427` |
| 硬件 | Apple M5 Max，128 GB 统一内存 |
| Target | `mlx-community/Qwen3.8-27B-8bit@815b83c0df8ffd1d1b5244cf75fd6ef14fca9ef9` |
| DFlash2 | `z-lab/Qwen3.8-27B-DFlash2@50307d4c4cde6860d4eee73e2547cd786fe8e8a4` |
| DFlash2 配置 | block size 4，draft runtime 4-bit，Greedy，B1/B2/B4 |
| 输入/输出 | 固定短 prompt（50 tokens）；输出 64 或 256 tokens |
| 统计方式 | aggregate generation TPS；另记录完整 HTTP wall aggregate TPS |

普通 decode 与强制 DFlash2 使用相同模型、prompt、cache 状态、warmup、cooldown
和执行顺序。输出 token 必须逐行完全一致；本表只保留整理后的汇总结果，不包含
原始 JSON 或生成正文。

## Dense Greedy 汇总

| 输出长度 | Batch | 普通 decode（TPS） | DFlash2（TPS） | 变化 | 结论 |
|---:|---:|---:|---:|---:|---|
| 64 | B1 | 15.086 | 30.744 | +103.78% | 正收益 |
| 64 | B2 | 28.161 | 34.146 | +21.26% | 正收益 |
| 64 | B4 | 29.115 | 34.985 | +20.16% | 正收益 |
| 256 | B1 | 16.170 | 34.460 | +113.11% | 正收益 |
| 256 | B2 | 31.034 | 36.714 | +18.30% | 正收益 |
| 256 | B4 | 28.758 | 34.029 | +18.33% | 正收益 |

### 门禁解释

- B1/B2/B4 在 64 和 256 tokens 场景均为正收益，且均超过 5% 的提升目标。
- 本轮 B4/256 的 `28.758 → 34.029 TPS` 结果替代此前的负收益记录。
- 表中为汇总 TPS；在缺少逐组原始报告时，不将其扩展为完整统计稳定性承诺。
- 本归档不覆盖 MTP、Prompt Lookup、KV quantization、Paged/SSD prefix cache 或
  Active KV Offload 组合；DFlash2 仍是文本专用路径。

## 相关记录

- 8-bit 非 DFlash2 MTP 的 Dense/B4 KV/B1-B2 回归数据见
  [`qwen38-affine8-mtp/2026-08-26/summary.md`](../../qwen38-affine8-mtp/2026-08-26/summary.md)。
- 4-bit DFlash2 历史数据见
  [`dflash2-final-validation/2026-08-23/summary.md`](../../dflash2-final-validation/2026-08-23/summary.md)，
  不与本归档合并解读。

## 证据边界

结果仅适用于上述 commit、MLX、checkpoint、硬件和请求条件；不能外推为所有
Apple Silicon 设备或所有输出长度的 TPS 承诺。此前的
`63.888 → 58.604 TPS` 属于旧轮次结果，不应与本表合并解读。
