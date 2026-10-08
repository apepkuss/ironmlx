# 勘误：B1 短输入 TTFT 专项确认协议

2026-10-07 补记。被勘误的文件 `docs/protocol-b1-ttft-confirmation.md` 保持原文不变：
- sha256 34f4bf589b61e38e81f8bff14ad098f26447d61fceddb20afd1a4a67aadb9d48；
- 与 `results/b1-ttft-confirmation-v1/run.json` 中记录的 `protocol_sha256` 一致。

**问题：**
- Boss 的要求是固定 24 个配对轮次，并且要完整执行。
- 协议原文自行加了一条规则：“有效轮次少于 20 时，结论记为无法判定”；通过条件也写成“新 24 轮完整（有效轮次 ≥ 20）”。这等于允许最少 20 对有效轮次即可判定通过，比 Boss 的要求宽松。
- 分析脚本 `scripts/analyze_b1_confirmation.py` 按这条规则实现（`MIN_ROUNDS = 20`）。

**按 Boss 要求应为：** 24 对必须全部有效，才能给出通过或未通过的结论；有任何无效轮次，结论都是“无效”。

**对结论的影响：** 没有影响。实际运行中 48 个会话全部有效，24 对全部计入（`evidence/b1-ttft-confirmation-v1-analysis.json` 中 `valid_rounds = 24`，`invalid_sessions = []`）。按 Boss 的原要求，结论同样是“通过”，第 1 项的结论不变。

冻结的协议、脚本和结果都不修改，本勘误单独记录。
