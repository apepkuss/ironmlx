# P5d — Perf Gate + mlx-vlm Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完成 P5 端到端验证：iron-bench 加 35B-A3B-4bit profile；与 mlx-vlm reference 进行 ≥50 prompt × ≥200 token 抽样对齐验证（greedy argmax 100% 一致 + top-K logits max_abs_diff < 1e-3）；性能基线录入历史；close-out 报告 commit。

**Architecture:** 仅验证 + 测量，不引入新功能。本 phase 是 P5 整体验收守门人。

**Tech Stack:** Rust 1.94 / iron-bench (cargo workspace member) / mlx-vlm Python (via uv run --with-editable) / Bash 集成脚本。

**Spec reference:** [docs/superpowers/specs/2026-05-18-ironmlx-p5-qwen35-moe-design.md](../specs/2026-05-18-ironmlx-p5-qwen35-moe-design.md) §4 / §7

---

## Pre-flight

### Step 0.1: P5c 闭环条件确认

- [ ] 在 `ironmlx-p5-moe` 分支 + P5c 全部 commit

```bash
git log --oneline -8
```

Expected: 看到 `p5c-*` commits 含 close-out

- [ ] working tree clean

```bash
git status --short
```

Expected: 空

### Step 0.2: P5c smoke + batched 集成测试基线

- [ ] 重跑 P5c 三个集成 test 确认环境 OK

```bash
export IRONMLX_MOE_MODEL_DIR=~/.ironmlx/models/.../snapshots/<sha>
cargo test -p ironmlx --release --test p5_qwen35_moe_smoke -- --ignored
cargo test -p ironmlx --release --test p5_qwen35_moe_batched -- --ignored
cargo test -p ironmlx --release --test p5_qwen35_moe_http_smoke -- --ignored
```

Expected: 全 PASS

### Step 0.3: mlx-vlm 环境就绪 (P5b T0 已验证)

- [ ] mlx-vlm 可加载 35B-A3B-4bit + generate 1 token

Run:

```
cd /Users/xin/workspace/iron-rivals/mlx-vlm
uv run --with-editable . python -c "
from mlx_vlm import load, generate
model, processor = load('~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/<sha>')
out = generate(model, processor, prompt='Hello', max_tokens=5, temperature=0.0, verbose=False)
print(out.text)
"
```

Expected: 5 tokens 输出无 OOM/crash (P5b T0 already verified — peak ~20.4GB, gen ~105 tok/s).

---

## Task 1: iron-bench `qwen3.5-moe` profile

**Files:**
- Modify: `iron-bench/src/profiles.rs` 或对应 profile 配置文件（按 iron-bench 实际结构调整）
- Modify: `iron-bench/configs/qwen3.5-moe.toml`（新建，或直接编辑 profiles.rs 表）

- [ ] **Step 1.1: 探查 iron-bench profile 注册位置**

Run:

```bash
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

```bash
cargo run --release -p iron-bench -- --list-profiles
```

Expected: 输出列表中包含 `qwen3.5-moe`

- [ ] **Step 1.4: Commit T1**

```bash
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

- [ ] **Step 2.1: 串行跑 ironmlx + mlx-vlm 两侧 iron-bench**

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

mlx-vlm 侧（停掉 ironmlx server 后跑）：

```
cargo run --release -p iron-bench -- \
  --profile qwen3.5-moe \
  --backend mlx-vlm \
  --prefill 128,512,2048 \
  --decode-steady 50 \
  --output reports/p5d-mlxvlm-moe.json 2>&1 | tee reports/p5d-mlxvlm-moe.log
```

Expected: 两份 JSON + 两份 log 报告，跑完无 OOM/crash。

- [ ] **Step 2.2: 整理基线表格写入 plan inline**

把数据回填到本任务 step 2.3 表格：

| 指标 | ironmlx | mlx-vlm | 相对差 |
|---|---|---|---|
| prefill PP=128 (tok/s) | ___ | ___ | ___% |
| prefill PP=512 | ___ | ___ | ___% |
| prefill PP=2048 | ___ | ___ | ___% |
| decode steady ITL (ms) | ___ | ___ | ___% |
| decode steady tok/s | ___ | ___ | ___% |
| peak memory (GB) | ___ | ___ | — |

- [ ] **Step 2.3: 性能 gate 判定**

按 spec §4.3 "perf gate 阈值由 T1 实测数据落定后定" 原则：

- **可接受**：ironmlx 相对 mlx-vlm 在所有 prefill/decode 指标上相对差 < 20% (mlx-vlm 是 Python overhead 但 backend 同 mlx Metal; ironmlx Rust 应该相当或更优)
- **退化** (> 20%)：surface 给 Boss，分析根因（可能 SparseMoeBlock G2 fallback 性能问题）；如 T0 决定走 G1 但 mlx::gather_qmm 仍不优，留 P5e 优化 phase

若可接受 → 写入 close-out 报告（T5）；若退化 → Boss 决定是否阻塞 P5 整体闭环。

- [ ] **Step 2.4: Commit T2**

```bash
git add reports/p5d-*.json reports/p5d-*.log
git commit -m "$(cat <<'EOF'
test(p5d-t2): MoE perf baseline — iron-bench serial run vs mlx-vlm

Serial benchmark per memory[feedback_serial_perf_experiments].
Records prefill PP=128/512/2048 + decode steady ITL on
Qwen3.5-35B-A3B-4bit. Both ironmlx and mlx-vlm backends profiled
under identical hardware + prompt set.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: mlx-vlm greedy argmax 跨 prompt 对齐验证

**Files:**

- Create: `scripts/p5d_mlxvlm_argmax_align.sh`
- Create: `scripts/p5d_compare_argmax.py`

- [ ] **Step 3.1: 准备 50-prompt 对齐集**

Create `scripts/p5d_prompts.txt`：

```text
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

Create `scripts/p5d_mlxvlm_argmax_align.sh`:

```bash
#!/bin/bash
# 串行跑 ironmlx + mlx-vlm，对每个 prompt 拿 first 200 token greedy 输出，
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

# (2) mlx-vlm 生成（停掉 ironmlx 后串行）
cd /Users/xin/workspace/iron-rivals/mlx-vlm
i=0
while IFS= read -r prompt; do
  out=$(uv run --with-editable . python -c "
from mlx_vlm import load, generate
model, processor = load('$MODEL_DIR')
out = generate(model, processor, prompt='$prompt', max_tokens=200, temperature=0.0, verbose=False)
print(out.text)
")
  echo "{\"idx\":$i,\"prompt\":$(printf %s "$prompt" | jq -Rs .),\"output\":$(printf %s "$out" | jq -Rs .)}" \
    >> "$OLDPWD/$OUT_DIR/mlx-vlm.jsonl"
  i=$((i+1))
done < "$OLDPWD/$PROMPTS"
cd - >/dev/null
```

加可执行权限：

```bash
chmod +x scripts/p5d_mlxvlm_argmax_align.sh
```

- [ ] **Step 3.3: 写比对 Python 脚本**

Create `scripts/p5d_compare_argmax.py`:

```python
#!/usr/bin/env python3
"""Compare ironmlx vs mlx-vlm greedy outputs per prompt. Exit 0 if all match."""
import json, sys

ironmlx = [json.loads(l) for l in open("reports/p5d-argmax/ironmlx.jsonl")]
ref     = [json.loads(l) for l in open("reports/p5d-argmax/mlx-vlm.jsonl")]
assert len(ironmlx) == len(ref), f"length mismatch: {len(ironmlx)} vs {len(ref)}"

mismatches = []
for a, b in zip(ironmlx, ref):
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
    print(f"MISMATCH: {len(mismatches)}/{len(ironmlx)} prompts diverged")
    for idx, at, ai, bi in mismatches[:5]:
        print(f"\n  prompt {idx} diverges at char {at}")
        print(f"    ironmlx: {ai!r}")
        print(f"    mlx-vlm: {bi!r}")
    sys.exit(1)
print(f"OK All {len(ironmlx)} prompts: greedy output identical")
```

- [ ] **Step 3.4: 跑对齐**

```bash
chmod +x scripts/p5d_compare_argmax.py
bash scripts/p5d_mlxvlm_argmax_align.sh
python3 scripts/p5d_compare_argmax.py
```

Expected: `OK All 50 prompts: greedy output identical`

如有 mismatch：分析第一个 divergence 位置的 routing / topk renorm / softmax 顺序是否与 mlx-lm 算法 reference 一致。

- [ ] **Step 3.5: Commit T3**

```bash
git add scripts/p5d_mlxvlm_argmax_align.sh scripts/p5d_compare_argmax.py scripts/p5d_prompts.txt
git commit -m "$(cat <<'EOF'
test(p5d-t3): cross-prompt mlx-vlm argmax alignment (50 prompts × 200 token)

Serial harness + comparator. Validates ironmlx greedy output
byte-identical to mlx-vlm reference on 50-prompt fixture set under
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

Requires ironmlx + mlx-vlm to dump logits to file:
  - mlx-vlm: monkey-patch forward in uv run script to save logits via numpy
  - ironmlx: cargo bin with `generate --dump-logits=<path>` (P5d T4 增量)

或临时方案：直接在 mlx-vlm Python 内 monkey-patch forward 抓 logits，
ironmlx 用 cargo test 集成测试导出 logits 到文件。
"""
import json, sys
import numpy as np

K = 100  # top-K 容差范围
TOL = 1e-3

PROMPTS_DUMPS = [
    ("ironmlx_logits_p0.npy", "mlxvlm_logits_p0.npy"),
    # 5 prompts → 5 (a, b) pair
]

max_global = 0.0
for ironmlx_f, ref_f in PROMPTS_DUMPS:
    a = np.load(f"reports/p5d-argmax/{ironmlx_f}")  # [vocab]
    b = np.load(f"reports/p5d-argmax/{ref_f}")
    # top-K argsort mlx-vlm reference
    topk = np.argsort(b)[-K:]
    diff = np.abs(a[topk] - b[topk]).max()
    max_global = max(max_global, diff)
    if diff >= TOL:
        print(f"FAIL {ironmlx_f}: top-{K} max_abs_diff = {diff} >= {TOL}")

if max_global < TOL:
    print(f"OK top-{K} logits max_abs_diff = {max_global} < {TOL}")
    sys.exit(0)
sys.exit(1)
```

- [ ] **Step 4.2: 修 ironmlx + mlx-vlm 支持 dump-logits（如未支持）**

仅当 4.1 脚本需要 numpy logits dump 时执行。否则 4.1 可以改为：

- 在 ironmlx `tests/p5_qwen35_moe_logits_dump.rs` 内调用 forward + `numpy_save` 落盘
- 用 mlx-vlm Python (`uv run --with-editable`) 写一个小脚本 monkey-patch forward dump 同 prompt 的 first-step logits

- [ ] **Step 4.3: 跑 logits 对齐**

```bash
# 串行：ironmlx → mlx-vlm → 比对
python3 scripts/p5d_logits_align.py
```

Expected: `OK top-100 logits max_abs_diff < 1e-3`

- [ ] **Step 4.4: Commit T4**

```bash
git add scripts/p5d_logits_align.py
git commit -m "$(cat <<'EOF'
test(p5d-t4): top-K logits max_abs_diff vs mlx-vlm baseline

Validates ironmlx forward output top-100 logits within 1e-3 of
mlx-vlm reference on 5-prompt sample. Tighter than argmax-only
criterion: catches sub-decision-boundary numerical drift that
would surface as routing tiebreak divergence on adversarial
prompts.

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
| 指标 | ironmlx | mlx-vlm | 相对差 |
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

```bash
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings
cargo build --release
cargo test -p ironmlx --lib --release
```

Expected: 全 PASS，clippy 零 warning。

- [ ] **Step 5.3: 集成测试终检**

```bash
export IRONMLX_MODEL_DIR=~/.ironmlx/models/.../Qwen3.5-4B-MLX-4bit/snapshots/<sha>
export IRONMLX_MOE_MODEL_DIR=~/.ironmlx/models/.../Qwen3.5-35B-A3B-4bit/snapshots/<sha>
cargo test -p ironmlx --release --test p4_http_smoke -- --ignored
cargo test -p ironmlx --release --test p5_qwen35_moe_smoke -- --ignored
cargo test -p ironmlx --release --test p5_qwen35_moe_batched -- --ignored
cargo test -p ironmlx --release --test p5_qwen35_moe_http_smoke -- --ignored
bash scripts/p5d_mlxvlm_argmax_align.sh
python3 scripts/p5d_compare_argmax.py
python3 scripts/p5d_logits_align.py
```

Expected: 全 PASS

### Step 5.4: regression sweep_full gate (per feedback_regression_sweep_at_closeout)

Before close-out commit, MUST run full regression sweep. **Must achieve 19/19 PASS** (or document any pre-existing flake explicitly).

Prereq: 4B snapshot at `~/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/<sha>` (download if absent via `HF_HUB_CACHE=$HOME/.ironmlx/models hf download mlx-community/Qwen3.5-4B-MLX-4bit`).

```bash
export QWEN35_MODEL=$(ls -d ~/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/*/ | head -1)
MLX_DIR=$HOME/.local/mlx ./scripts/sweep/sweep_full.sh 2>&1 | tail -5
```

Expected: `19/19 PASS in <NN>m <SS>s`.

If a suite fails: investigate root cause (P5d regression vs pre-existing) and either fix in-place or document as known-flaky in this Task's outputs.

- [ ] **Step 5.5: Final commit**

```bash
git add docs/superpowers/plans/2026-05-18-ironmlx-p5-closeout.md
git commit -m "$(cat <<'EOF'
docs(p5): close-out report — Qwen3.5 MoE foundation complete

End-of-phase report covering: actual MoE forward path decisions
(T0 research outputs), perf baseline (M1 Pro 32GB), numerical
alignment with mlx-vlm (50 prompts × 200 token greedy + top-100
logits), dense regression status, known issues queued for
follow-up phases.

Verification: sweep_full 19/19 PASS, clippy zero warnings,
release build clean.

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
- [ ] 50 prompt greedy 输出与 mlx-vlm 100% 一致
- [ ] top-100 logits max_abs_diff < 1e-3 vs mlx-vlm
- [ ] dense 路径 p4_http_smoke 不退化
- [ ] clippy / fmt / release build / lib unit test 全 PASS
- [ ] sweep_full 19/19 PASS
- [ ] close-out 报告 commit

满足全部 → P5 整体完成，可向 Boss 申请 review + merge `ironmlx-p5-moe` 回 `ironmlx`。

---

## Self-Review Notes

- ✓ Spec coverage：§4.1 单测 (前 phase 已实现) / §4.2 集成测试 / §4.3 perf gate 全部覆盖
- ✓ mlx-vlm 主 baseline (P5b T0 验证)，与 spec §1.2 Q4 一致（不对齐实现，只对齐输出）
- ✓ Task 数 = 5 + Pre-flight，符合 5-7 范围
- ✓ Close-out 报告留实测填空槽位，避免 plan 阶段虚构数字
