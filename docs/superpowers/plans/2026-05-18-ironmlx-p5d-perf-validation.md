# P5d — Perf Gate + omlx Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完成 P5 端到端验证：iron-bench 加 35B-A3B-4bit profile；与 omlx CLI 进行 ≥50 prompt × ≥200 token 抽样对齐验证（greedy argmax 100% 一致 + top-K logits max_abs_diff < 1e-3）；性能基线录入历史；close-out 报告 commit。

**Architecture:** 仅验证 + 测量，不引入新功能。本 phase 是 P5 整体验收守门人。

**Tech Stack:** Rust 1.94 / iron-bench (cargo workspace member) / omlx Python CLI / Bash 集成脚本。

**Spec reference:** [docs/superpowers/specs/2026-05-18-ironmlx-p5-qwen35-moe-design.md](../specs/2026-05-18-ironmlx-p5-qwen35-moe-design.md) §4 / §7

---

## Pre-flight

### Step 0.1: P5c 闭环条件确认

- [ ] 在 `ironmlx-p5-moe` 分支 + P5c 全部 commit

```
git log --oneline -8
```
Expected: 看到 `p5c-*` commits 含 close-out

- [ ] working tree clean

```
git status --short
```
Expected: 空

### Step 0.2: P5c smoke + batched 集成测试基线

- [ ] 重跑 P5c 三个集成 test 确认环境 OK

```
export IRONMLX_MOE_MODEL_DIR=~/.ironmlx/models/.../snapshots/<sha>
cargo test -p ironmlx --release --test p5_qwen35_moe_smoke -- --ignored
cargo test -p ironmlx --release --test p5_qwen35_moe_batched -- --ignored
cargo test -p ironmlx --release --test p5_qwen35_moe_http_smoke -- --ignored
```
Expected: 全 PASS

### Step 0.3: omlx + mlx-lm 环境就绪

- [ ] omlx CLI 可加载 35B-A3B-4bit（P5b T0 Step 0.3 已验证；重做一次确认）

```
cd /Users/xin/workspace/iron-rivals/omlx
python -m omlx.generate --model ~/.ironmlx/models/.../snapshots/<sha> \
  --prompt "Hello world" --max-tokens 10 --temp 0 2>&1 | tail -5
```
Expected: 输出 10 token，无 OOM

- [ ] mlx-lm 作为辅 baseline 可加载（如需）

```
python -c "from mlx_lm import load, generate; m, t = load('~/.ironmlx/models/.../snapshots/<sha>'); print(generate(m, t, 'Hello', max_tokens=10))"
```
Expected: 输出有效 token，或 surface 给 Boss（若 mlx-lm 未支持 Qwen3.5-MoE）

---

## Task 1: iron-bench `qwen3.5-moe` profile

**Files:**
- Modify: `iron-bench/src/profiles.rs` 或对应 profile 配置文件（按 iron-bench 实际结构调整）
- Modify: `iron-bench/configs/qwen3.5-moe.toml`（新建，或直接编辑 profiles.rs 表）

- [ ] **Step 1.1: 探查 iron-bench profile 注册位置**

Run:
```
grep -rn "profile\|qwen3\|qwen2" /Users/xin/workspace/ironmlx-backend/iron-bench/src/ | head -30
```

记下 profile 定义位置（推测在 profiles.rs 或 main.rs）。

- [ ] **Step 1.2: 新增 qwen3.5-moe profile**

按现有 profile 风格添加 entry（具体语法依实际 iron-bench struct 而定），示例骨架：
```rust
// iron-bench/src/profiles.rs（或对应位置）
ProfileEntry {
    name: "qwen3.5-moe",
    huggingface_id: "mlx-community/Qwen3.5-35B-A3B-4bit",
    local_path_hint: "~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit",
    model_type: "qwen3_5_moe",
    prefill_tokens: vec![128, 512, 2048],
    decode_warmup: 5,
    decode_steady: 50,
    // 其他 fields...
},
```

- [ ] **Step 1.3: 验证 iron-bench 能识别新 profile**

```
cargo run --release -p iron-bench -- --list-profiles
```
Expected: 输出列表中包含 `qwen3.5-moe`

- [ ] **Step 1.4: Commit T1**

```
git add iron-bench/
git commit -m "$(cat <<'EOF'
feat(p5d-t1): iron-bench qwen3.5-moe profile entry

Adds mlx-community/Qwen3.5-35B-A3B-4bit to iron-bench profile
registry. Default prefill sweep [128, 512, 2048] + decode steady
50 tokens with 5-warmup, matching the dense Qwen3.5 profile
convention.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: 跑 iron-bench `qwen3.5-moe` 基线落地

**Files:**
- Create: `docs/superpowers/plans/p5d-perf-baseline.md`（基线数字记录，gitignore exempt）

- [ ] **Step 2.1: 串行跑 ironmlx + omlx 两侧 iron-bench**

按 [feedback_serial_perf_experiments](../../../.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/feedback_serial_perf_experiments.md) 串行规则，**一次只跑一个 server，避免 GPU/swap 互相污染**。

ironmlx 侧：
```
cd /Users/xin/workspace/ironmlx-backend
cargo run --release -p iron-bench -- \
  --profile qwen3.5-moe \
  --backend ironmlx \
  --prefill 128,512,2048 \
  --decode-steady 50 \
  --output reports/p5d-ironmlx-moe.json 2>&1 | tee reports/p5d-ironmlx-moe.log
```

omlx 侧（停掉 ironmlx server 后跑）：
```
cargo run --release -p iron-bench -- \
  --profile qwen3.5-moe \
  --backend omlx \
  --prefill 128,512,2048 \
  --decode-steady 50 \
  --output reports/p5d-omlx-moe.json 2>&1 | tee reports/p5d-omlx-moe.log
```

Expected: 两份 JSON + 两份 log 报告，跑完无 OOM/crash。

- [ ] **Step 2.2: 整理基线表格写入 plan inline**

把数据回填到本任务 step 2.3 表格：

| 指标 | ironmlx | omlx CLI | 相对差 |
|---|---|---|---|
| prefill PP=128 (tok/s) | ___ | ___ | ___% |
| prefill PP=512 | ___ | ___ | ___% |
| prefill PP=2048 | ___ | ___ | ___% |
| decode steady ITL (ms) | ___ | ___ | ___% |
| decode steady tok/s | ___ | ___ | ___% |
| peak memory (GB) | ___ | ___ | — |

- [ ] **Step 2.3: 性能 gate 判定**

按 spec §4.3 "perf gate 阈值由 T1 实测数据落定后定" 原则：
- **可接受**：ironmlx 相对 omlx 在所有 prefill/decode 指标上相对差 < 30%
- **退化** (> 30%)：surface 给 Boss，分析根因（可能 SparseMoeBlock G2 fallback 性能问题）；如 T0 决定走 G1 但 mlx::gather_qmm 仍不优，留 P5e 优化 phase

若可接受 → 写入 close-out 报告（T5）；若退化 → Boss 决定是否阻塞 P5 整体闭环。

- [ ] **Step 2.4: Commit T2**

```
git add reports/p5d-*.json reports/p5d-*.log
git commit -m "$(cat <<'EOF'
test(p5d-t2): MoE perf baseline — iron-bench serial run vs omlx

Serial benchmark per memory[feedback_serial_perf_experiments].
Records prefill PP=128/512/2048 + decode steady ITL on
Qwen3.5-35B-A3B-4bit. Both ironmlx and omlx backends profiled
under identical hardware + prompt set.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: omlx greedy argmax 跨 prompt 对齐验证

**Files:**
- Create: `scripts/p5d_omlx_argmax_align.sh`
- Create: `scripts/p5d_compare_argmax.py`

- [ ] **Step 3.1: 准备 50-prompt 对齐集**

Create `scripts/p5d_prompts.txt`：
```
Once upon a time, in a small village,
The quick brown fox jumps over
def fibonacci(n):
    if n < 2:
List three reasons why exercise is important:
Translate to French: Good morning.
Write a haiku about autumn.
... (50 行 prompt，覆盖 code/Math/QA/翻译/续写/创作 6 类)
```

完整 50 prompt 可由 Boss 提供或参考 iron-bench 现有 prompt 集合。

- [ ] **Step 3.2: 写跑对齐脚本**

Create `scripts/p5d_omlx_argmax_align.sh`:
```bash
#!/bin/bash
# 串行跑 ironmlx + omlx，对每个 prompt 拿 first 200 token greedy 输出，
# 输出 JSON 文件供 Python 脚本比对。
set -euo pipefail

MODEL_DIR=${IRONMLX_MOE_MODEL_DIR:?set IRONMLX_MOE_MODEL_DIR}
PROMPTS=scripts/p5d_prompts.txt
OUT_DIR=reports/p5d-argmax

mkdir -p "$OUT_DIR"

# (1) ironmlx CLI 生成
i=0
while IFS= read -r prompt; do
  out=$(cargo run --release -p ironmlx --quiet -- generate \
    --model "$MODEL_DIR" --prompt "$prompt" --max-tokens 200 --temperature 0)
  echo "{\"idx\":$i,\"prompt\":$(printf %s "$prompt" | jq -Rs .),\"output\":$(printf %s "$out" | jq -Rs .)}" \
    >> "$OUT_DIR/ironmlx.jsonl"
  i=$((i+1))
done < "$PROMPTS"

# (2) omlx CLI 生成（停掉 ironmlx 后串行）
cd /Users/xin/workspace/iron-rivals/omlx
i=0
while IFS= read -r prompt; do
  out=$(python -m omlx.generate --model "$MODEL_DIR" --prompt "$prompt" --max-tokens 200 --temp 0)
  echo "{\"idx\":$i,\"prompt\":$(printf %s "$prompt" | jq -Rs .),\"output\":$(printf %s "$out" | jq -Rs .)}" \
    >> "$OLDPWD/$OUT_DIR/omlx.jsonl"
  i=$((i+1))
done < "$OLDPWD/$PROMPTS"
cd - >/dev/null
```

加可执行权限：
```
chmod +x scripts/p5d_omlx_argmax_align.sh
```

- [ ] **Step 3.3: 写比对 Python 脚本**

Create `scripts/p5d_compare_argmax.py`:
```python
#!/usr/bin/env python3
"""Compare ironmlx vs omlx greedy outputs per prompt. Exit 0 if all match."""
import json, sys

ironmlx = [json.loads(l) for l in open("reports/p5d-argmax/ironmlx.jsonl")]
omlx   = [json.loads(l) for l in open("reports/p5d-argmax/omlx.jsonl")]
assert len(ironmlx) == len(omlx), f"length mismatch: {len(ironmlx)} vs {len(omlx)}"

mismatches = []
for a, b in zip(ironmlx, omlx):
    assert a["idx"] == b["idx"]
    if a["output"] != b["output"]:
        # 找第一个差异位置
        ao, bo = a["output"], b["output"]
        diff_at = next(
            (i for i, (x, y) in enumerate(zip(ao, bo)) if x != y),
            min(len(ao), len(bo))
        )
        mismatches.append((a["idx"], diff_at, ao[:diff_at + 20], bo[:diff_at + 20]))

if mismatches:
    print(f"❌ MISMATCH: {len(mismatches)}/{len(ironmlx)} prompts diverged")
    for idx, at, ai, bi in mismatches[:5]:
        print(f"\n  prompt {idx} diverges at char {at}")
        print(f"    ironmlx: {ai!r}")
        print(f"    omlx:    {bi!r}")
    sys.exit(1)
print(f"✓ All {len(ironmlx)} prompts: greedy output identical")
```

- [ ] **Step 3.4: 跑对齐**

```
chmod +x scripts/p5d_compare_argmax.py
bash scripts/p5d_omlx_argmax_align.sh
python3 scripts/p5d_compare_argmax.py
```

Expected: `✓ All 50 prompts: greedy output identical`

如有 mismatch：分析第一个 divergence 位置的 routing / topk renorm / softmax 顺序是否与 mlx-lm 算法 reference 一致。

- [ ] **Step 3.5: Commit T3**

```
git add scripts/p5d_omlx_argmax_align.sh scripts/p5d_compare_argmax.py scripts/p5d_prompts.txt
git commit -m "$(cat <<'EOF'
test(p5d-t3): cross-prompt omlx argmax alignment (50 prompts × 200 token)

Serial harness + comparator. Validates ironmlx greedy output
byte-identical to omlx CLI on 50-prompt fixture set under
mlx-community/Qwen3.5-35B-A3B-4bit. Strict pass criteria —
any divergence surfaces with first-mismatch context.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: top-K logits max_abs_diff 验证

**Files:**
- Create: `scripts/p5d_logits_align.py`

- [ ] **Step 4.1: 写 logits 抽样比对脚本**

Create `scripts/p5d_logits_align.py`:
```python
#!/usr/bin/env python3
"""Compare top-K logits per first decode step on 5 prompts.

Requires patched omlx + ironmlx to dump logits to file:
  - omlx: --dump-logits=<path>
  - ironmlx: cargo bin with `generate --dump-logits=<path>` (P5d T4 增量)

或临时方案：直接在 omlx Python 内 mokey-patch forward 抓 logits，
ironmlx 用 cargo test 集成测试导出 logits 到文件。
"""
import json, sys
import numpy as np

K = 100  # top-K 容差范围
TOL = 1e-3

PROMPTS_DUMPS = [
    ("ironmlx_logits_p0.npy", "omlx_logits_p0.npy"),
    # 5 prompts → 5 (a, b) pair
]

max_global = 0.0
for ironmlx_f, omlx_f in PROMPTS_DUMPS:
    a = np.load(f"reports/p5d-argmax/{ironmlx_f}")  # [vocab]
    b = np.load(f"reports/p5d-argmax/{omlx_f}")
    # top-K argsort omlx
    topk = np.argsort(b)[-K:]
    diff = np.abs(a[topk] - b[topk]).max()
    max_global = max(max_global, diff)
    if diff >= TOL:
        print(f"❌ {ironmlx_f}: top-{K} max_abs_diff = {diff} >= {TOL}")

if max_global < TOL:
    print(f"✓ top-{K} logits max_abs_diff = {max_global} < {TOL}")
    sys.exit(0)
sys.exit(1)
```

- [ ] **Step 4.2: 修 ironmlx + omlx 支持 dump-logits（如未支持）**

仅当 4.1 脚本需要 numpy logits dump 时执行。否则 4.1 可以改为：
- 在 ironmlx `tests/p5_qwen35_moe_logits_dump.rs` 内调用 forward + `numpy_save` 落盘
- 用 omlx Python 写一个小脚本 dump 同 prompt 的 first-step logits

具体实现路径取决于 omlx CLI 是否提供 logits hook —— 如未提供，**改用 mlx_lm Python 直接 dump** 作为 baseline（mlx-lm 提供 forward 即返回 logits Array）。

- [ ] **Step 4.3: 跑 logits 对齐**

```
# 串行：ironmlx → omlx/mlx-lm → 比对
python3 scripts/p5d_logits_align.py
```
Expected: `✓ top-100 logits max_abs_diff < 1e-3`

- [ ] **Step 4.4: Commit T4**

```
git add scripts/p5d_logits_align.py
git commit -m "$(cat <<'EOF'
test(p5d-t4): top-K logits max_abs_diff vs omlx/mlx-lm baseline

Validates ironmlx forward output top-100 logits within 1e-3 of
omlx (primary) and mlx-lm (auxiliary) on 5-prompt sample. Tighter
than argmax-only criterion: catches sub-decision-boundary
numerical drift that would surface as routing tiebreak
divergence on adversarial prompts.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Close-out 报告 + P5 整体闭环验收

**Files:**
- Create: `docs/superpowers/plans/2026-05-18-ironmlx-p5-closeout.md`

- [ ] **Step 5.1: 写 close-out 报告**

Create `docs/superpowers/plans/2026-05-18-ironmlx-p5-closeout.md`:
```markdown
# P5 — Qwen3.5 MoE Foundation Close-out

| 字段 | 值 |
|---|---|
| 日期 | 2026-05-XX（实际完成时填） |
| 分支 | ironmlx-p5-moe |
| Spec | docs/superpowers/specs/2026-05-18-ironmlx-p5-qwen35-moe-design.md |
| 4 sub-phase plans | p5a-trait-refactor / p5b-moe-forward / p5c-scheduler-integration / p5d-perf-validation |

## 实测产出

### MoE forward 路径
- T0 调研结论：mlx::gather_qmm 暴露情况 ___；选定 kernel 路径 G1/G2 ___
- HF expert key 命名实际：___（与预期一致 / 有差异：___）
- norm_topk_prob 默认值：___

### 性能基线（M1 Pro 32GB 实测）
| 指标 | ironmlx | omlx | 相对差 |
|---|---|---|---|
| prefill PP=128 (tok/s) | ___ | ___ | ___% |
| prefill PP=512 | ___ | ___ | ___% |
| prefill PP=2048 | ___ | ___ | ___% |
| decode steady tok/s | ___ | ___ | ___% |
| peak RAM | ___ GB | ___ GB | — |

### 数值对齐验收
- 50 prompt × 200 token greedy 输出：___/50 PASS（≥1 fail 为不闭环）
- top-100 logits max_abs_diff：___（要求 < 1e-3）

### Dense 路径回归
- p4_http_smoke：PASS / FAIL
- b1_p2_3b_3_concurrent_gs：PASS / FAIL
- lib unit test 数量：___ (P5 起点 ___)

## 已知问题 + 后续 phase 候选

1. ___（如 G2 fallback 性能差于 dense → P5e G1 优化）
2. ___（如 16GB Mac 兼容性 → P6 / P5.x expert preload-with-smelt_mask 探索）
3. ___（VL path / MTP path — 明确属于 P6 / P7 未启动）

## P5 → 后续 phase 移交

- 现有分支 `ironmlx-p5-moe` 可合并回 `ironmlx` 主开发分支后启动 P6 VL
- close-out 数据写入 iron-bench 历史基线表（reports/）
- memory 更新（dmlx 对照、device-aware tile 等 reference）

🤖 Generated by Claude Code per superpowers workflow.
```

- [ ] **Step 5.2: 工具链 hygiene 终检**

```
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings
cargo build --release
cargo test -p ironmlx --lib --release
```
Expected: 全 PASS，clippy 零 warning。

- [ ] **Step 5.3: 集成测试终检**

```
export IRONMLX_MODEL_DIR=~/.ironmlx/models/.../Qwen3.5-4B-MLX-4bit/snapshots/<sha>
export IRONMLX_MOE_MODEL_DIR=~/.ironmlx/models/.../Qwen3.5-35B-A3B-4bit/snapshots/<sha>
cargo test -p ironmlx --release --test p4_http_smoke -- --ignored
cargo test -p ironmlx --release --test p5_qwen35_moe_smoke -- --ignored
cargo test -p ironmlx --release --test p5_qwen35_moe_batched -- --ignored
cargo test -p ironmlx --release --test p5_qwen35_moe_http_smoke -- --ignored
bash scripts/p5d_omlx_argmax_align.sh
python3 scripts/p5d_compare_argmax.py
python3 scripts/p5d_logits_align.py
```
Expected: 全 PASS

- [ ] **Step 5.4: Final commit**

```
git add docs/superpowers/plans/2026-05-18-ironmlx-p5-closeout.md
git commit -m "$(cat <<'EOF'
docs(p5): close-out report — Qwen3.5 MoE foundation complete

End-of-phase report covering: actual MoE forward path decisions
(T0 research outputs), perf baseline (M1 Pro 32GB), numerical
alignment with omlx (50 prompts × 200 token greedy + top-100
logits), dense regression status, known issues queued for
follow-up phases.

P5 sub-phase chain complete: P5a (trait) → P5b (MoE forward)
→ P5c (integration) → P5d (validation). Branch ironmlx-p5-moe
ready for merge consideration after Boss review.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## P5d 闭环条件 = P5 整体闭环条件

- [ ] iron-bench `qwen3.5-moe` profile 跑通且数据落地（reports/p5d-*.json）
- [ ] 50 prompt greedy 输出与 omlx 100% 一致
- [ ] top-100 logits max_abs_diff < 1e-3 vs omlx（或 mlx-lm 退路）
- [ ] dense 路径 p4_http_smoke 不退化
- [ ] clippy / fmt / release build / lib unit test 全 PASS
- [ ] close-out 报告 commit

满足全部 → P5 整体完成，可向 Boss 申请 review + merge `ironmlx-p5-moe` 回 `ironmlx`。

---

## Self-Review Notes

- ✓ Spec coverage：§4.1 单测 (前 phase 已实现) / §4.2 集成测试 / §4.3 perf gate 全部覆盖
- ✓ omlx 主 baseline + mlx-lm 辅 baseline，与 spec §1.2 Q4 一致（不对齐实现，只对齐输出）
- ✓ Task 数 = 5 + Pre-flight，符合 5-7 范围
- ✓ Close-out 报告留实测填空槽位，避免 plan 阶段虚构数字
