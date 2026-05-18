# P5b — Qwen3.5 MoE Forward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `models/qwen3_5_moe/` 新模块内实现 `Qwen35MoeModel` 完整 forward path（包含 `SparseMoeBlock` routed + shared expert）；`impl Model` 闭合 trait 契约；与 omlx CLI 在单 prompt 上 greedy argmax 对齐验证通过。

**Architecture:** Hybrid expert 布局：shared per-Linear (`nn::Mlp`) + routed fused stacked Array；Router = `softmax → topk → [renorm]`；expert quantized matmul 路径 G1 (`mlx::gather_qmm` 单 fused op) 或 G2 (per-expert scatter+qmm fallback)，由 T0 调研决定。

**Tech Stack:** Rust 1.94 / mlx (cxx-mlx wrapper) / Apple Silicon Metal / safetensors mmap。

**Spec reference:** [docs/superpowers/specs/2026-05-18-ironmlx-p5-qwen35-moe-design.md](../specs/2026-05-18-ironmlx-p5-qwen35-moe-design.md) §2.2 / §3.2-3.7 / §5

---

## Pre-flight

### Step 0.1: P5a 闭环条件确认

- [ ] 确认在 `ironmlx-p5-moe` 分支 + P5a 已 commit

Run: `git -C /Users/xin/workspace/ironmlx-backend log --oneline -10`
Expected: 看到 7 个 `p5a-*` commits + `docs(p5): ...` spec commit

- [ ] 确认 working tree clean

Run: `git -C /Users/xin/workspace/ironmlx-backend status --short`
Expected: 空

### Step 0.2: 基线确认

- [ ] dense regression base

Run:
```
cargo test -p ironmlx --lib --release
```
Expected: 全 PASS（与 P5a close-out 一致）

---

## Task 0: 研究 — snapshot keys + mlx gather op + omlx baseline

**Files:** N/A（research only）

- [ ] **Step 0.0: 拉 35B-A3B-4bit snapshot 到本地**

Run:
```
huggingface-cli download mlx-community/Qwen3.5-35B-A3B-4bit \
  --local-dir ~/.ironmlx/models/mlx-community--Qwen3.5-35B-A3B-4bit
```

或如果按 HF cache layout：
```
HF_HOME=~/.ironmlx/models huggingface-cli download mlx-community/Qwen3.5-35B-A3B-4bit
```

Expected: ~17.5GB 下载，根据网络可能 5-30 分钟。

- [ ] **Step 0.1: 列 expert / router / shared 相关 tensor key**

Run（在 ironmlx 仓库内）:
```
python3 - <<'PY'
import os, json, glob
root = os.path.expanduser("~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots")
snap = sorted(glob.glob(os.path.join(root, "*")))[0]
idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))
keys = sorted(idx["weight_map"].keys())
# 抽样 layer 0 的 MoE 相关 keys
sample = [k for k in keys if k.startswith("model.layers.0.")]
for k in sample:
    print(k)
PY
```

Expected: 看到例如：
- `model.layers.0.input_layernorm.weight`
- `model.layers.0.linear_attn.*` 或 `model.layers.0.self_attn.*`
- `model.layers.0.mlp.experts.gate_proj.weight` / `.scales` / `.biases`
- `model.layers.0.mlp.experts.up_proj.weight` / ...
- `model.layers.0.mlp.experts.down_proj.weight` / ...
- `model.layers.0.mlp.shared_expert.gate_proj.weight` / ...
- `model.layers.0.mlp.gate.weight`（router）

**记录**：把准确 key 列表写到 `docs/superpowers/plans/p5b-snapshot-keys.txt`（gitignore），供 T2/T3 编码引用。

- [ ] **Step 0.2: 读 mlx C++ headers 确认 gather_qmm 是否暴露**

Run:
```
grep -rn "gather_qmm\|gather_mm\|quantized.*gather" /Users/xin/workspace/iron-rivals/mlx --include='*.h' --include='*.hpp' 2>/dev/null | head -30
```

如有命中：进一步在 `mlx/src/mlx/quantization.rs`（cxx-mlx wrapper）确认是否已暴露同名函数。如未暴露但 mlx C++ 已有 → 需在 `mlx-sys` 加 cxx bridge + `mlx/` 加 safe wrapper（小型 wrapper 任务，作为 G1 路径前置）。

- [ ] **Step 0.3: 确认 omlx CLI 可加载 35B-A3B-4bit**

Run（顺序串行，遵守 memory 串行 perf 规则）:
```
cd /Users/xin/workspace/iron-rivals/omlx
python -m omlx.generate \
  --model ~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/$(ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots | head -1) \
  --prompt "Once upon a time" \
  --max-tokens 20 \
  --temp 0 2>&1 | tail -5
```

Expected: omlx 输出 20 token，无 OOM/crash。**如果 omlx 加载失败**：surface 给 Boss，决定退路用 mlx-lm 作为唯一 baseline。

- [ ] **Step 0.4: 读 mlx-lm Qwen3MoeSparseMoeBlock 源码作为算法 reference（仅读不抄）**

Run:
```
find ~/.venv ~/workspace/iron-rivals -name 'qwen3_moe.py' 2>/dev/null | head -3
```

如已在 omlx repo 内：阅读其 SparseMoeBlock 实现确认：
- softmax 顺序（是否在 topk 前/后）
- `norm_topk_prob` 默认值
- shared_expert 加权方式（与 routed 同权重 / 不同权重 / 全权重）

**记录结论**到 `docs/superpowers/plans/p5b-algorithm-reference.txt`（gitignore），供 T2 实现引用。**禁止 copy 代码，只摘要算法步骤**。

- [ ] **Step 0.5: 决策 + 记录到 plan inline**

回填本任务的"研究产出"小节，写在本文件 Task 0 末尾：
```markdown
**T0 产出（实测填入）：**
- 实际 expert key 命名前缀：______（fused stacked / per-expert 等）
- mlx::gather_qmm 是否暴露：✓/✗
- 选定 kernel 路径：G1 / G2
- omlx 在 35B-A3B-4bit 是否可加载：✓/✗
- norm_topk_prob 默认：true / false
- softmax 在 topk 之前 / 之后
```

- [ ] **Step 0.6: Commit T0**

```
git add docs/superpowers/plans/2026-05-18-ironmlx-p5b-moe-forward.md
git commit -m "$(cat <<'EOF'
research(p5b-t0): MoE snapshot keys / mlx gather op / omlx baseline

Pulled mlx-community/Qwen3.5-35B-A3B-4bit snapshot. Confirmed
expert key naming, mlx::gather_qmm availability, and omlx baseline
loading. Algorithm reference notes captured locally (not checked in
per design philosophy: omlx is observation only).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 1: `Qwen35MoeConfig` + 模块骨架

**Files:**
- Create: `ironmlx/src/models/qwen3_5_moe/mod.rs`
- Create: `ironmlx/src/models/qwen3_5_moe/config.rs`
- Modify: `ironmlx/src/models/mod.rs`
- Test: `ironmlx/src/models/qwen3_5_moe/config.rs`（tests 模块）

- [ ] **Step 1.1: 写 config.rs**

Create `ironmlx/src/models/qwen3_5_moe/config.rs`:
```rust
//! Qwen3.5 MoE text-config parsing.

use anyhow::{anyhow, Context};
use serde::Deserialize;

use crate::core::Loader;
use crate::nn::AttnKind;
use crate::Result;

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParams {
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub mrope_section: Vec<i32>,
}

fn default_partial_rotary_factor() -> f32 { 0.25 }
fn default_rope_theta() -> f32 { 10_000_000.0 }
fn default_max_position_embeddings() -> i32 { 32768 }

impl Default for RopeParams {
    fn default() -> Self {
        Self {
            partial_rotary_factor: default_partial_rotary_factor(),
            rope_theta: default_rope_theta(),
            mrope_section: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen35MoeConfig {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    #[serde(default)]
    pub head_dim: Option<i32>,
    pub vocab_size: i32,
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub full_attention_interval: i32,
    #[serde(default)]
    pub linear_num_value_heads: i32,
    #[serde(default)]
    pub linear_num_key_heads: i32,
    #[serde(default)]
    pub linear_key_head_dim: i32,
    #[serde(default)]
    pub linear_value_head_dim: i32,
    #[serde(default)]
    pub linear_conv_kernel_dim: i32,
    #[serde(default)]
    pub rope_parameters: RopeParams,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,

    // MoE-specific fields
    pub num_experts: i32,
    pub num_experts_per_tok: i32,
    pub moe_intermediate_size: i32,
    pub shared_expert_intermediate_size: i32,
    #[serde(default)]
    pub mlp_only_layers: Vec<i32>,
    #[serde(default)]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub router_aux_loss_coef: f32,
}

impl Qwen35MoeConfig {
    pub fn from_loader(loader: &Loader) -> Result<Self> {
        let raw = loader.config_raw_value();
        let text_config = raw
            .get("text_config")
            .ok_or_else(|| anyhow!("config.json missing text_config field"))?;
        let cfg: Qwen35MoeConfig = serde_json::from_value(text_config.clone())
            .context("failed to deserialize Qwen35MoeConfig from text_config")?;
        Ok(cfg)
    }

    pub fn effective_head_dim(&self) -> i32 {
        self.head_dim.unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn layer_kind(&self, layer_idx: i32) -> AttnKind {
        if (layer_idx + 1) % self.full_attention_interval == 0 {
            AttnKind::Full
        } else {
            AttnKind::Linear
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn realistic_text_config_json() -> serde_json::Value {
        // Subset of mlx-community/Qwen3.5-35B-A3B-4bit text_config
        serde_json::json!({
            "attention_bias": false,
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_size": 2048,
            "intermediate_size": 512,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "max_position_embeddings": 262144,
            "mlp_only_layers": [],
            "moe_intermediate_size": 512,
            "num_attention_heads": 16,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 40,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-06,
            "rope_parameters": {
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000.0
            },
            "shared_expert_intermediate_size": 512,
            "vocab_size": 248320
        })
    }

    #[test]
    fn parses_35b_a3b_text_config() {
        let v = realistic_text_config_json();
        let cfg: Qwen35MoeConfig = serde_json::from_value(v).expect("parse");
        assert_eq!(cfg.num_experts, 256);
        assert_eq!(cfg.num_experts_per_tok, 8);
        assert_eq!(cfg.moe_intermediate_size, 512);
        assert_eq!(cfg.shared_expert_intermediate_size, 512);
        assert_eq!(cfg.num_hidden_layers, 40);
        assert!(cfg.mlp_only_layers.is_empty());
    }

    #[test]
    fn layer_kind_partition_full_attention_interval_4() {
        let cfg: Qwen35MoeConfig =
            serde_json::from_value(realistic_text_config_json()).unwrap();
        // 40 layers, interval=4 → Full at {3,7,...,39}
        let full_count = (0..cfg.num_hidden_layers)
            .filter(|i| matches!(cfg.layer_kind(*i), AttnKind::Full))
            .count();
        assert_eq!(full_count, 10);
    }
}
```

- [ ] **Step 1.2: 写模块 mod.rs（skeleton, 其他类型 T2/T3 添加）**

Create `ironmlx/src/models/qwen3_5_moe/mod.rs`:
```rust
//! Qwen3.5 MoE model (text-only). See spec
//! docs/superpowers/specs/2026-05-18-ironmlx-p5-qwen35-moe-design.md.

pub mod config;
// 后续 task 解开：
// pub mod sparse_moe;
// pub mod decoder_layer;
// pub mod text_model;
// pub mod model;

pub use config::{Qwen35MoeConfig, RopeParams};
// pub use model::Qwen35MoeModel;
// pub use text_model::Qwen35MoeTextModel;
```

- [ ] **Step 1.3: models/mod.rs 解开 qwen3_5_moe**

Modify `ironmlx/src/models/mod.rs`:
```rust
pub mod qwen3_5;
pub mod qwen3_5_moe;   // ← 解开（原是注释）

pub use qwen3_5::{Qwen35Config, Qwen35Model, Qwen35TextModel, RopeParams};
pub use qwen3_5_moe::{Qwen35MoeConfig, RopeParams as MoeRopeParams};
```

- [ ] **Step 1.4: 验证 build + test**

Run:
```
cargo build -p ironmlx
cargo test -p ironmlx --lib --release qwen3_5_moe::config::tests
```
Expected: 2 tests PASS。

- [ ] **Step 1.5: Commit T1**

```
git add ironmlx/src/models/qwen3_5_moe/ ironmlx/src/models/mod.rs
git commit -m "$(cat <<'EOF'
feat(p5b-t1): Qwen35MoeConfig + module skeleton

Parses text_config for mlx-community/Qwen3.5-35B-A3B-4bit including
MoE-specific fields (num_experts, num_experts_per_tok,
moe_intermediate_size, shared_expert_intermediate_size,
mlp_only_layers, norm_topk_prob). Layer kind partition reuses
full_attention_interval pattern from dense.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `Router` + `SparseMoeBlock` forward

**Files:**
- Create: `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs`
- Modify: `ironmlx/src/models/qwen3_5_moe/mod.rs`（解开 sparse_moe）

- [ ] **Step 2.1: 写 sparse_moe.rs（含 Router + RoutedExperts + SparseMoeBlock 三结构）**

按 spec §3.6 创建。**关键 forward_on 实现** 由 T0 决策的 G1/G2 路径决定。**先写 G2 fallback**（不依赖 mlx::gather_qmm，确保功能闭环；若 T0 确认 G1 可用则在 T6 切换）。

Create `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs`:
```rust
//! SparseMoeBlock: routed top-k experts + shared expert.
//! Algorithm reference (read-only): mlx-lm qwen3_moe.py::Qwen3MoeSparseMoeBlock.
//! See spec §2.2 / §3.6 for data flow.

use anyhow::{anyhow, Context};
use mlx::{Array, Dtype, StreamOrDevice};

use crate::core::Loader;
use crate::nn::{Linear, Mlp};
use crate::Result;

pub struct Router {
    gate: Linear,                // [num_experts, hidden] (router weights)
    num_experts: i32,
}

impl Router {
    pub fn from_loader(loader: &Loader, prefix: &str, num_experts: i32) -> Result<Self> {
        let gate = Linear::from_loader(loader, &format!("{prefix}.gate"))?;
        Ok(Self { gate, num_experts })
    }

    /// Returns (topk_idx [B*S, K] uint32, topk_probs [B*S, K] fp).
    pub fn route_on(
        &self,
        flat_x: &Array,
        num_experts_per_tok: i32,
        norm_topk_prob: bool,
        target: StreamOrDevice,
    ) -> Result<(Array, Array)> {
        // (1) gate logits
        let logits = self.gate.forward_on(flat_x, target)?;       // [B*S, E]

        // (2) softmax along last axis
        let probs = mlx::ops::softmax_on(&logits, &[-1_i32][..], false, target)?;

        // (3) topk along last axis: returns (values, indices)
        let (topk_vals, topk_idx) = mlx::ops::topk_on(
            &probs, num_experts_per_tok, -1, target,
        )?;

        // (4) optional renormalize
        let topk_probs = if norm_topk_prob {
            let sum = mlx::ops::sum_on(&topk_vals, &[-1_i32][..], true, target)?;
            (&topk_vals / &sum)?
        } else {
            topk_vals
        };

        Ok((topk_idx, topk_probs))
    }
}

pub struct RoutedExperts {
    pub gate_weight: Array, pub gate_scales: Array, pub gate_biases: Option<Array>,
    pub up_weight:   Array, pub up_scales:   Array, pub up_biases:   Option<Array>,
    pub down_weight: Array, pub down_scales: Array, pub down_biases: Option<Array>,
    pub group_size:  i32,
    pub bits:        i32,
    pub num_experts: i32,
}

impl RoutedExperts {
    pub fn from_loader(loader: &Loader, prefix: &str) -> Result<Self> {
        let qmeta = loader.quant_meta().ok_or_else(|| anyhow!(
            "RoutedExperts requires quantized checkpoint; loader has no QuantMeta"
        ))?;
        let gate_weight = loader.tensor(&format!("{prefix}.gate_proj.weight"))?.clone();
        let gate_scales = loader.tensor(&format!("{prefix}.gate_proj.scales"))?.clone();
        let gate_biases = loader.tensor_opt(&format!("{prefix}.gate_proj.biases")).cloned();
        let up_weight = loader.tensor(&format!("{prefix}.up_proj.weight"))?.clone();
        let up_scales = loader.tensor(&format!("{prefix}.up_proj.scales"))?.clone();
        let up_biases = loader.tensor_opt(&format!("{prefix}.up_proj.biases")).cloned();
        let down_weight = loader.tensor(&format!("{prefix}.down_proj.weight"))?.clone();
        let down_scales = loader.tensor(&format!("{prefix}.down_proj.scales"))?.clone();
        let down_biases = loader.tensor_opt(&format!("{prefix}.down_proj.biases")).cloned();
        let num_experts = gate_weight.shape().as_slice()[0];
        Ok(Self {
            gate_weight, gate_scales, gate_biases,
            up_weight, up_scales, up_biases,
            down_weight, down_scales, down_biases,
            group_size: qmeta.group_size,
            bits: qmeta.bits,
            num_experts,
        })
    }
}

pub struct SparseMoeBlock {
    router: Router,
    routed: RoutedExperts,
    shared: Mlp,
    num_experts_per_tok: i32,
    norm_topk_prob: bool,
}

impl SparseMoeBlock {
    pub fn from_loader(
        loader: &Loader,
        prefix: &str,
        num_experts: i32,
        num_experts_per_tok: i32,
        norm_topk_prob: bool,
    ) -> Result<Self> {
        let router = Router::from_loader(loader, prefix, num_experts)?;
        let routed = RoutedExperts::from_loader(loader, &format!("{prefix}.experts"))?;
        let shared = Mlp::from_loader(loader, &format!("{prefix}.shared_expert"))?;
        Ok(Self { router, routed, shared, num_experts_per_tok, norm_topk_prob })
    }

    pub fn forward_on(&self, x: &Array, target: StreamOrDevice) -> Result<Array> {
        // (1) flatten [B, S, H] → [BS, H]
        let dims = x.shape();
        let dvec = dims.as_slice();
        let (b, s, h) = (dvec[0], dvec[1], dvec[2]);
        let flat_x = mlx::ops::shape::reshape(x, &[b * s, h][..])
            .context("SparseMoeBlock: reshape input to [BS, H]")?;

        // (2) router → (topk_idx [BS, K], topk_probs [BS, K])
        let (topk_idx, topk_probs) = self.router.route_on(
            &flat_x, self.num_experts_per_tok, self.norm_topk_prob, target,
        )?;

        // (3) routed expert MLP — G2 fallback (per-expert scatter+qmm).
        //     G1 (mlx::gather_qmm) path branch lands in T6 if T0 confirmed
        //     availability.
        let routed_out = self.routed_g2_forward(&flat_x, &topk_idx, &topk_probs, target)?;

        // (4) shared expert (parallel, all tokens)
        let shared_out = self.shared.forward_on(&flat_x, target)?;

        // (5) sum + reshape back
        let sum = (&routed_out + &shared_out)?;
        Ok(mlx::ops::shape::reshape(&sum, &[b, s, h][..])?)
    }

    /// G2 path: for each of K topk positions, gather expert id per token,
    /// run per-expert quantized_matmul on subset, weighted sum.
    /// 性能 fallback；G1 在 T6 替换。
    fn routed_g2_forward(
        &self,
        flat_x: &Array,           // [BS, H]
        topk_idx: &Array,         // [BS, K]
        topk_probs: &Array,       // [BS, K]
        target: StreamOrDevice,
    ) -> Result<Array> {
        // 对每个 expert id e ∈ [0, num_experts):
        //   mask_e [BS, K] = (topk_idx == e)
        //   weights_e [BS] = (mask_e * topk_probs).sum(-1)
        //   if any(weights_e > 0):
        //     x_e = flat_x  // 所有 token 都过这个 expert（mask 后求和）
        //     gate_e = flat_x @ routed.gate_weight[e].T  (quantized_matmul)
        //     up_e   = flat_x @ routed.up_weight[e].T
        //     act_e  = silu(gate_e) * up_e
        //     down_e = act_e @ routed.down_weight[e].T
        //     out  += down_e * weights_e[:, None]
        //
        // 注：256 expert 全跑是最朴素 G2；正确但慢。后续 G1 替换。
        // 详细代码块 ~80 行，省略此处 — 见 sparse_moe.rs 实际实现。
        // 关键 helper: per-expert slice routed.gate_weight[e:e+1, ...]
        // + 单独 quantized_matmul + scales/biases slicing。
        todo!("G2 expert scatter+qmm — 详细实现在 T2.4 step")
    }
}
```

> **NOTE for executor**: `routed_g2_forward` 完整代码体在下一 step 实现。当前 step 仅放骨架。

- [ ] **Step 2.2: 实现 routed_g2_forward 内部 (per-expert qmm loop)**

替换 `routed_g2_forward` 的 todo 实现为：
```rust
fn routed_g2_forward(
    &self,
    flat_x: &Array,
    topk_idx: &Array,
    topk_probs: &Array,
    target: StreamOrDevice,
) -> Result<Array> {
    let bs = flat_x.shape().as_slice()[0];
    let h = flat_x.shape().as_slice()[1];

    // 初始化输出 [BS, H] = 0
    let mut acc = Array::zeros((bs, h), flat_x.dtype())?;

    for e in 0..self.routed.num_experts {
        // (a) per-token weight: weights_e [BS, 1] = ((topk_idx == e) * topk_probs).sum(-1, keepdim=true)
        let e_scalar = Array::from_iter(std::iter::once(e), &[1_i32][..]);
        let mask = mlx::ops::equal_on(topk_idx, &e_scalar, target)?;
        let mask_f = mlx::ops::cast::astype(&mask, flat_x.dtype())?;
        let weighted_mask = (&mask_f * topk_probs)?;
        let weights_e = mlx::ops::sum_on(&weighted_mask, &[-1_i32][..], true, target)?; // [BS, 1]

        // 短路：若全 token 都未路由到 e，跳过（仍需要 eval 才知道；G2 性能差是已知）。
        // 为保正确性这里不做 host-side 短路 — 让 mlx 自己处理 0 权重 mask。

        // (b) per-expert slice quantized matmul
        //     gate_w_e [moe_inter, H_packed] = routed.gate_weight[e:e+1, :, :].squeeze(0)
        let gate_w_e = mlx::ops::indexing::slice_strided(
            &self.routed.gate_weight,
            &[e, 0, 0][..],
            &[e+1, self.routed.gate_weight.shape().as_slice()[1], self.routed.gate_weight.shape().as_slice()[2]][..],
            &[1_i32, 1, 1][..],
        )?;
        let gate_w_e = mlx::ops::shape::squeeze_on(&gate_w_e, Some(&[0_i32][..]), target)?;
        // 同步 slice scales/biases
        let gate_s_e = slice_axis0(&self.routed.gate_scales, e, target)?;
        let gate_b_e = self.routed.gate_biases.as_ref().map(|a| slice_axis0(a, e, target)).transpose()?;
        let up_w_e   = slice_axis0_squeeze(&self.routed.up_weight, e, target)?;
        let up_s_e   = slice_axis0(&self.routed.up_scales, e, target)?;
        let up_b_e   = self.routed.up_biases.as_ref().map(|a| slice_axis0(a, e, target)).transpose()?;
        let down_w_e = slice_axis0_squeeze(&self.routed.down_weight, e, target)?;
        let down_s_e = slice_axis0(&self.routed.down_scales, e, target)?;
        let down_b_e = self.routed.down_biases.as_ref().map(|a| slice_axis0(a, e, target)).transpose()?;

        let gate_out = mlx::quantization::quantized_matmul_on(
            flat_x, &gate_w_e, &gate_s_e, gate_b_e.as_ref(),
            /* transpose */ true, Some(self.routed.group_size), Some(self.routed.bits),
            "affine", target,
        )?;
        let up_out = mlx::quantization::quantized_matmul_on(
            flat_x, &up_w_e, &up_s_e, up_b_e.as_ref(),
            true, Some(self.routed.group_size), Some(self.routed.bits),
            "affine", target,
        )?;
        let silu_gate = (&gate_out * &gate_out.sigmoid_on(target)?)?;
        let act = (&silu_gate * &up_out)?;
        let down_out = mlx::quantization::quantized_matmul_on(
            &act, &down_w_e, &down_s_e, down_b_e.as_ref(),
            true, Some(self.routed.group_size), Some(self.routed.bits),
            "affine", target,
        )?;
        // weighted accumulate
        let weighted = (&down_out * &weights_e)?;  // broadcast [BS,H] × [BS,1]
        acc = (&acc + &weighted)?;
    }
    Ok(acc)
}

#[inline]
fn slice_axis0(arr: &Array, idx: i32, target: StreamOrDevice) -> Result<Array> {
    let dim0_full = arr.shape().as_slice()[0];
    let dim1_full = arr.shape().as_slice()[1];
    let _ = dim0_full;
    Ok(mlx::ops::indexing::slice_strided(
        arr,
        &[idx, 0][..],
        &[idx+1, dim1_full][..],
        &[1_i32, 1][..],
    )?)
}

#[inline]
fn slice_axis0_squeeze(arr: &Array, idx: i32, target: StreamOrDevice) -> Result<Array> {
    let s = arr.shape();
    let sv = s.as_slice();
    let raw = mlx::ops::indexing::slice_strided(
        arr,
        &[idx, 0, 0][..],
        &[idx+1, sv[1], sv[2]][..],
        &[1_i32, 1, 1][..],
    )?;
    Ok(mlx::ops::shape::squeeze_on(&raw, Some(&[0_i32][..]), target)?)
}
```

- [ ] **Step 2.3: 解开 sparse_moe 模块导出**

Modify `ironmlx/src/models/qwen3_5_moe/mod.rs`:
```rust
pub mod config;
pub mod sparse_moe;

pub use config::{Qwen35MoeConfig, RopeParams};
pub use sparse_moe::{Router, RoutedExperts, SparseMoeBlock};
```

- [ ] **Step 2.4: 单元测试**

Append to `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use mlx::Dtype;

    #[test]
    fn router_topk_shape_and_dtype() {
        // 构造 stub Router — 用 fp Linear 而非 quantized (单测便利)
        let n_experts = 8_i32;
        let h = 16_i32;
        let weight: Array = (
            (0..n_experts*h).map(|i| (i as f32) * 0.01).collect::<Vec<_>>().as_slice(),
            &[n_experts, h][..],
        ).try_into().unwrap();
        let gate = Linear::new_fp(weight, None);
        let router = Router { gate, num_experts: n_experts };

        let bs = 4_i32;
        let flat_x: Array = (
            (0..bs*h).map(|i| (i as f32) * 0.1).collect::<Vec<_>>().as_slice(),
            &[bs, h][..],
        ).try_into().unwrap();

        let (idx, probs) = router.route_on(&flat_x, 3, true, ()).unwrap();
        assert_eq!(idx.shape().as_slice(), &[bs, 3]);
        assert_eq!(probs.shape().as_slice(), &[bs, 3]);
        // After renorm, sum of probs per row ≈ 1.0
        let row_sum = mlx::ops::sum_on(&probs, &[-1_i32][..], false, ()).unwrap();
        let v: Vec<f32> = mlx::ops::cast::astype(&row_sum, Dtype::Float32).unwrap().to_vec().unwrap();
        for s in v { assert!((s - 1.0).abs() < 1e-5, "row sum {s} != 1.0"); }
    }
}
```

- [ ] **Step 2.5: 验证 build + test**

Run:
```
cargo build -p ironmlx
cargo test -p ironmlx --lib --release qwen3_5_moe::sparse_moe::tests
```
Expected: 1 test PASS。

- [ ] **Step 2.6: Commit T2**

```
git add ironmlx/src/models/qwen3_5_moe/sparse_moe.rs ironmlx/src/models/qwen3_5_moe/mod.rs
git commit -m "$(cat <<'EOF'
feat(p5b-t2): SparseMoeBlock + Router (G2 per-expert qmm path)

Router: softmax + topk + optional renormalize. RoutedExperts holds
fused stacked quantized weights [E, out, in_packed]. SparseMoeBlock
forward uses G2 fallback (per-expert slice + quantized_matmul +
weighted accumulate). G1 (mlx::gather_qmm) substitution deferred
to T6 pending T0 availability confirmation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `DecoderLayerMoe` + `Qwen35MoeTextModel`

**Files:**
- Create: `ironmlx/src/models/qwen3_5_moe/decoder_layer.rs`
- Create: `ironmlx/src/models/qwen3_5_moe/text_model.rs`
- Modify: `ironmlx/src/models/qwen3_5_moe/mod.rs`

- [ ] **Step 3.1: 写 decoder_layer.rs**

Create `ironmlx/src/models/qwen3_5_moe/decoder_layer.rs`:
```rust
//! Decoder layer for Qwen3.5 MoE — same hybrid attention as dense
//! but FFN is SparseMoeBlock instead of nn::Mlp. See spec §3.5.

use anyhow::anyhow;
use mlx::{Array, StreamOrDevice};

use crate::core::cache::{GatedDeltaCache, KVCache};
use crate::core::Loader;
use crate::nn::{
    AttnKind, AttnPath, GatedAttention, GatedAttentionConfig, GatedDeltaNet, GatedDeltaNetConfig,
    LayerCache, Mrope, RmsNorm,
};
use crate::Result;

use super::sparse_moe::SparseMoeBlock;

#[derive(Debug, Clone, Copy)]
pub struct DecoderLayerMoeConfig {
    pub hidden_size: i32,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
    pub attention_bias: bool,
    pub linear_num_value_heads: i32,
    pub linear_num_key_heads: i32,
    pub linear_key_head_dim: i32,
    pub linear_value_head_dim: i32,
    pub linear_conv_kernel_dim: i32,
    pub num_experts: i32,
    pub num_experts_per_tok: i32,
    pub norm_topk_prob: bool,
}

pub struct DecoderLayerMoe {
    input_layernorm: RmsNorm,
    attn: AttnPath,
    post_attention_layernorm: RmsNorm,
    ffn: SparseMoeBlock,
    cfg: DecoderLayerMoeConfig,
}

impl DecoderLayerMoe {
    pub fn from_loader(
        loader: &Loader,
        prefix: &str,
        cfg: DecoderLayerMoeConfig,
        kind: AttnKind,
    ) -> Result<Self> {
        let input_layernorm = RmsNorm::from_loader(
            loader, &format!("{prefix}.input_layernorm"), cfg.rms_norm_eps,
        )?;
        let attn = match kind {
            AttnKind::Full => {
                let ga = GatedAttention::from_loader(
                    loader, &format!("{prefix}.self_attn"),
                    GatedAttentionConfig {
                        num_heads: cfg.num_heads,
                        num_kv_heads: cfg.num_kv_heads,
                        head_dim: cfg.head_dim,
                        rms_norm_eps: cfg.rms_norm_eps,
                        attention_bias: cfg.attention_bias,
                    },
                )?;
                AttnPath::Full(ga)
            }
            AttnKind::Linear => {
                let gdn = GatedDeltaNet::from_loader(
                    loader, &format!("{prefix}.linear_attn"),
                    GatedDeltaNetConfig {
                        hidden_size: cfg.hidden_size,
                        num_v_heads: cfg.linear_num_value_heads,
                        num_k_heads: cfg.linear_num_key_heads,
                        head_k_dim: cfg.linear_key_head_dim,
                        head_v_dim: cfg.linear_value_head_dim,
                        conv_kernel_size: cfg.linear_conv_kernel_dim,
                        rms_norm_eps: cfg.rms_norm_eps,
                    },
                )?;
                AttnPath::Linear(gdn)
            }
        };
        let post_attention_layernorm = RmsNorm::from_loader(
            loader, &format!("{prefix}.post_attention_layernorm"), cfg.rms_norm_eps,
        )?;
        let ffn = SparseMoeBlock::from_loader(
            loader, &format!("{prefix}.mlp"),
            cfg.num_experts, cfg.num_experts_per_tok, cfg.norm_topk_prob,
        )?;
        Ok(Self { input_layernorm, attn, post_attention_layernorm, ffn, cfg })
    }

    pub fn kind(&self) -> AttnKind {
        match &self.attn {
            AttnPath::Full(_) => AttnKind::Full,
            AttnPath::Linear(_) => AttnKind::Linear,
        }
    }

    /// 数据流与 dense DecoderLayer::forward_on 完全相同，仅 ffn 不同。
    #[allow(clippy::too_many_arguments)]
    pub fn forward_on(
        &self,
        x: &Array,
        mrope: &Mrope,
        cos: &Array,
        sin: &Array,
        full_attn_mask: Option<&Array>,
        linear_attn_mask: Option<&Array>,
        per_row_lens: Option<&[i32]>,
        cache: Option<&mut LayerCache>,
        target: StreamOrDevice,
    ) -> Result<Array> {
        if x.ndim() != 3 {
            return Err(anyhow!(
                "DecoderLayerMoe::forward_on: x must be rank-3, got {}",
                x.ndim()
            ));
        }
        let normed_in = self.input_layernorm.forward_on(x, target)?;
        let attn = match (&self.attn, cache) {
            (AttnPath::Full(a), Some(LayerCache::Full(kv))) => a.forward_on(
                &normed_in, mrope, cos, sin,
                full_attn_mask, linear_attn_mask, per_row_lens,
                Some(kv), target,
            )?,
            (AttnPath::Full(a), None) => a.forward_on(
                &normed_in, mrope, cos, sin,
                full_attn_mask, linear_attn_mask, per_row_lens,
                None, target,
            )?,
            (AttnPath::Linear(a), Some(LayerCache::Linear(gdc))) => a.forward_on(
                &normed_in, linear_attn_mask, per_row_lens, Some(gdc), target,
            )?,
            (AttnPath::Linear(a), None) => a.forward_on(
                &normed_in, linear_attn_mask, per_row_lens, None, target,
            )?,
            _ => return Err(anyhow!(
                "DecoderLayerMoe::forward_on: attention kind / cache kind mismatch"
            )),
        };
        let h = (x + &attn)?;
        let normed_post = self.post_attention_layernorm.forward_on(&h, target)?;
        let ffn_out = self.ffn.forward_on(&normed_post, target)?;
        Ok((&h + &ffn_out)?)
    }
}
```

- [ ] **Step 3.2: 写 text_model.rs**

Create `ironmlx/src/models/qwen3_5_moe/text_model.rs`:
```rust
//! Qwen3.5 MoE text model — embed + N×DecoderLayerMoe + RmsNorm.

use anyhow::anyhow;
use mlx::{Array, StreamOrDevice};

use crate::core::Loader;
use crate::nn::{Embedding, LayerCache, Mrope, RmsNorm};
use crate::Result;

use super::config::Qwen35MoeConfig;
use super::decoder_layer::{DecoderLayerMoe, DecoderLayerMoeConfig};

pub struct Qwen35MoeTextModel {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayerMoe>,
    norm: RmsNorm,
    mrope: Mrope,
    cfg: Qwen35MoeConfig,
}

impl Qwen35MoeTextModel {
    pub fn from_loader(loader: &Loader, cfg: Qwen35MoeConfig) -> Result<Self> {
        let embed_tokens = Embedding::from_loader(loader, "model.embed_tokens")?;
        let head_dim = cfg.effective_head_dim();
        if cfg.rope_parameters.mrope_section.is_empty() {
            return Err(anyhow!("rope_parameters.mrope_section must be non-empty"));
        }
        let mrope = Mrope::new(
            head_dim,
            cfg.rope_parameters.rope_theta,
            cfg.rope_parameters.partial_rotary_factor,
            &cfg.rope_parameters.mrope_section,
            true,
        )?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers as usize);
        for i in 0..cfg.num_hidden_layers {
            let layer_cfg = DecoderLayerMoeConfig {
                hidden_size: cfg.hidden_size,
                num_heads: cfg.num_attention_heads,
                num_kv_heads: cfg.num_key_value_heads,
                head_dim,
                rms_norm_eps: cfg.rms_norm_eps,
                attention_bias: cfg.attention_bias,
                linear_num_value_heads: cfg.linear_num_value_heads,
                linear_num_key_heads: cfg.linear_num_key_heads,
                linear_key_head_dim: cfg.linear_key_head_dim,
                linear_value_head_dim: cfg.linear_value_head_dim,
                linear_conv_kernel_dim: cfg.linear_conv_kernel_dim,
                num_experts: cfg.num_experts,
                num_experts_per_tok: cfg.num_experts_per_tok,
                norm_topk_prob: cfg.norm_topk_prob,
            };
            let kind = cfg.layer_kind(i);
            layers.push(DecoderLayerMoe::from_loader(
                loader, &format!("model.layers.{i}"), layer_cfg, kind,
            )?);
        }
        let norm = RmsNorm::from_loader(loader, "model.norm", cfg.rms_norm_eps)?;
        Ok(Self { embed_tokens, layers, norm, mrope, cfg })
    }

    pub fn config(&self) -> &Qwen35MoeConfig { &self.cfg }
    pub fn num_layers(&self) -> usize { self.layers.len() }

    pub fn embed_on(&self, input_ids: &Array, target: StreamOrDevice) -> Result<Array> {
        self.embed_tokens.forward_on(input_ids, target)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_post_embedding_on(
        &self,
        hidden: &Array,
        position_ids: &Array,
        cache: Option<&mut [LayerCache]>,
        attention_mask: Option<&Array>,
        linear_attention_mask: Option<&Array>,
        per_row_lens: Option<&[i32]>,
        target: StreamOrDevice,
    ) -> Result<Array> {
        if let Some(c) = cache.as_deref() {
            if c.len() != self.layers.len() {
                return Err(anyhow!(
                    "Qwen35MoeTextModel::forward_post_embedding_on: cache.len()={} != num_layers={}",
                    c.len(), self.layers.len()
                ));
            }
        }
        let (cos, sin) = self.mrope.cos_sin(position_ids)?;
        let mut x = hidden.clone();
        match cache {
            Some(c) => {
                for (layer, cell) in self.layers.iter().zip(c.iter_mut()) {
                    x = layer.forward_on(
                        &x, &self.mrope, &cos, &sin,
                        attention_mask, linear_attention_mask, per_row_lens,
                        Some(cell), target,
                    )?;
                }
            }
            None => {
                for layer in &self.layers {
                    x = layer.forward_on(
                        &x, &self.mrope, &cos, &sin,
                        attention_mask, linear_attention_mask, per_row_lens,
                        None, target,
                    )?;
                }
            }
        }
        self.norm.forward_on(&x, target)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_on(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let hidden = self.embed_on(input_ids, target)?;
        self.forward_post_embedding_on(
            &hidden, position_ids, cache, decode_mask, None, per_row_lens, target,
        )
    }
}
```

- [ ] **Step 3.3: 解开 mod 导出**

Modify `ironmlx/src/models/qwen3_5_moe/mod.rs`:
```rust
pub mod config;
pub mod decoder_layer;
pub mod sparse_moe;
pub mod text_model;

pub use config::{Qwen35MoeConfig, RopeParams};
pub use decoder_layer::{DecoderLayerMoe, DecoderLayerMoeConfig};
pub use sparse_moe::{Router, RoutedExperts, SparseMoeBlock};
pub use text_model::Qwen35MoeTextModel;
```

- [ ] **Step 3.4: 编译 + 单测验证**

Run:
```
cargo build -p ironmlx
cargo test -p ironmlx --lib --release qwen3_5_moe::
```
Expected: 全 PASS（仅 config + sparse_moe::tests::router_topk）

- [ ] **Step 3.5: Commit T3**

```
git add ironmlx/src/models/qwen3_5_moe/
git commit -m "$(cat <<'EOF'
feat(p5b-t3): DecoderLayerMoe + Qwen35MoeTextModel

DecoderLayerMoe shares attention path with dense (GatedAttention /
GatedDeltaNet from nn::), FFN replaced by SparseMoeBlock.
Qwen35MoeTextModel mirrors Qwen35TextModel API (from_loader / embed_on /
forward_post_embedding_on / forward_on).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `Qwen35MoeModel` + `impl Model` + Loader sanitize MoE

**Files:**
- Create: `ironmlx/src/models/qwen3_5_moe/model.rs`
- Modify: `ironmlx/src/core/loader.rs`（sanitize 增量）
- Modify: `ironmlx/src/models/qwen3_5_moe/mod.rs`

- [ ] **Step 4.1: 写 model.rs**

Create `ironmlx/src/models/qwen3_5_moe/model.rs`:
```rust
//! Top-level Qwen3.5 MoE model. See spec §3.4.

use anyhow::Context;
use mlx::{Array, Dtype, StreamOrDevice};

use crate::core::cache::{GatedDeltaCache, KVCache};
use crate::core::memory_budget::ModelMeta;
use crate::core::{Loader, Model};
use crate::nn::{AttnKind, LayerCache, Linear};
use crate::Result;

use super::config::Qwen35MoeConfig;
use super::text_model::Qwen35MoeTextModel;

pub struct Qwen35MoeModel {
    text: Qwen35MoeTextModel,
    lm_head: Linear,
}

impl Qwen35MoeModel {
    pub fn from_loader(loader: &Loader) -> Result<Self> {
        let cfg = Qwen35MoeConfig::from_loader(loader)
            .context("parsing Qwen35MoeConfig from loader")?;
        Self::from_loader_with_config(loader, cfg)
    }

    pub fn from_loader_with_config(loader: &Loader, cfg: Qwen35MoeConfig) -> Result<Self> {
        if cfg.tie_word_embeddings {
            return Err(anyhow::anyhow!(
                "Qwen35MoeModel: tie_word_embeddings expected false for A3B; got true"
            ));
        }
        let lm_head = Linear::from_loader(loader, "lm_head")?;
        let text = Qwen35MoeTextModel::from_loader(loader, cfg)?;
        Ok(Self { text, lm_head })
    }

    pub fn config(&self) -> &Qwen35MoeConfig { self.text.config() }
    pub fn text(&self) -> &Qwen35MoeTextModel { &self.text }

    pub fn approx_weight_bytes(&self) -> usize {
        let cfg = self.config();
        let h = cfg.hidden_size as usize;
        let l = cfg.num_hidden_layers as usize;
        let e = cfg.num_experts as usize;
        let me = cfg.moe_intermediate_size as usize;
        let se = cfg.shared_expert_intermediate_size as usize;
        let attn = 4 * h * h * l / 2;
        let routed = 3 * e * h * me * l / 2;
        let shared = 3 * h * se * l / 2;
        let embed_head = 2 * (cfg.vocab_size as usize) * h / 2;
        attn + routed + shared + embed_head
    }

    pub fn forward_on(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let hidden = self.text.forward_on(
            input_ids, position_ids, per_row_lens, decode_mask, cache, target,
        )?;
        // 切最后位置 + lm_head 投影（参考 dense Qwen35Model::slice_last_and_project）
        let dims = hidden.shape();
        let dvec = dims.as_slice();
        let (b, s, h) = (dvec[0], dvec[1], dvec[2]);
        let last_hidden = if s > 1 {
            mlx::ops::indexing::slice_strided(
                &hidden, &[0_i32, s-1, 0][..], &[b, s, h][..], &[1_i32, 1, 1][..],
            )?
        } else {
            hidden
        };
        self.lm_head.forward_on(&last_hidden, target)
    }

    pub fn batched_prefill(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        attention_mask: &Array,
        linear_attention_mask: &Array,
        per_row_lens: &[i32],
        cache: Option<&mut [LayerCache]>,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let hidden = self.text.embed_on(input_ids, target)?;
        let hidden = self.text.forward_post_embedding_on(
            &hidden, position_ids, cache,
            Some(attention_mask), Some(linear_attention_mask),
            Some(per_row_lens), target,
        )?;
        // per-row 切 last + lm_head（参考 dense per_row_slice_last）
        let dims = hidden.shape();
        let dvec = dims.as_slice();
        let (b, s, h) = (dvec[0], dvec[1], dvec[2]);
        let mut rows: Vec<Array> = Vec::with_capacity(b as usize);
        for (i, &l) in per_row_lens.iter().enumerate() {
            let pos = l - 1;
            let row = mlx::ops::indexing::slice_strided_on(
                &hidden,
                &[i as i32, pos, 0][..],
                &[i as i32 + 1, pos + 1, h][..],
                &[1_i32, 1, 1][..],
                target,
            )?;
            rows.push(row);
        }
        let row_refs: Vec<&Array> = rows.iter().collect();
        let last_hidden = mlx::ops::shape::concatenate_on(&row_refs[..], 0, target)?;
        let _ = s;
        self.lm_head.forward_on(&last_hidden, target)
    }

    pub fn make_cache(&self, batch: i32, cap: i32, dtype: Dtype) -> Result<Vec<LayerCache>> {
        let cfg = self.config();
        let head_dim = cfg.effective_head_dim();
        let mut out = Vec::with_capacity(cfg.num_hidden_layers as usize);
        for i in 0..cfg.num_hidden_layers {
            match cfg.layer_kind(i) {
                AttnKind::Full => {
                    out.push(LayerCache::Full(
                        KVCache::new(batch, cfg.num_key_value_heads, head_dim, head_dim, dtype, cap)
                            .with_step(cap),
                    ));
                }
                AttnKind::Linear => {
                    let conv_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads * 2
                        + cfg.linear_value_head_dim * cfg.linear_num_value_heads;
                    out.push(LayerCache::Linear(GatedDeltaCache::new_with_cap(
                        batch, cfg.linear_conv_kernel_dim, conv_dim,
                        cfg.linear_num_value_heads, cfg.linear_value_head_dim,
                        cfg.linear_key_head_dim, dtype, cap,
                    )?));
                }
            }
        }
        Ok(out)
    }

    pub fn model_meta(&self) -> ModelMeta {
        let cfg = self.config();
        ModelMeta {
            num_hidden_layers: cfg.num_hidden_layers,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            hidden_size: cfg.hidden_size,
            head_dim: cfg.head_dim,
            weight_bytes: self.approx_weight_bytes(),
        }
    }
}

impl Model for Qwen35MoeModel {
    fn make_cache(&self, batch: i32, cap: i32, dtype: Dtype) -> Result<Vec<LayerCache>> {
        Qwen35MoeModel::make_cache(self, batch, cap, dtype)
    }
    fn forward_on(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        target: StreamOrDevice,
    ) -> Result<Array> {
        Qwen35MoeModel::forward_on(self, input_ids, position_ids, per_row_lens, decode_mask, cache, target)
    }
    fn batched_prefill(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        attention_mask: &Array,
        linear_attention_mask: &Array,
        per_row_lens: &[i32],
        cache: Option<&mut [LayerCache]>,
        target: StreamOrDevice,
    ) -> Result<Array> {
        Qwen35MoeModel::batched_prefill(
            self, input_ids, position_ids, attention_mask, linear_attention_mask,
            per_row_lens, cache, target,
        )
    }
    fn model_meta(&self) -> ModelMeta { Qwen35MoeModel::model_meta(self) }
    fn num_hidden_layers(&self) -> usize {
        self.config().num_hidden_layers as usize
    }
}
```

- [ ] **Step 4.2: Loader sanitize MoE 兼容**

按 T0 实测 key 命名调整 `ironmlx/src/core/loader.rs::sanitize` —— 若 MoE expert key 命名符合预期 `model.layers.{i}.mlp.experts.{gate|up|down}_proj.weight`，无需改 sanitize。

如果发现 MoE 的 `mtp.*` 不应触发 norm shift（dense 的旧行为），按 T0 调研结果加 model_type 条件分支：
```rust
// 在 sanitize 内 has_mtp 检测后追加
let is_moe = config_raw.get("model_type")
    .and_then(|v| v.as_str()) == Some("qwen3_5_moe");
if is_moe {
    // MoE 不做 +1 shift（除非 T0 验证表明也需要）
    // 具体决策在 T0 Step 0.4 算法 reference 阅读时确定
}
```

> **NOTE for executor**: 此分支具体生效与否取决于 T0 实测。如 T0 显示 MoE 也需要 +1 shift，本步骤不做改动。

- [ ] **Step 4.3: 解开 mod 导出**

Modify `ironmlx/src/models/qwen3_5_moe/mod.rs`:
```rust
pub mod config;
pub mod decoder_layer;
pub mod model;
pub mod sparse_moe;
pub mod text_model;

pub use config::{Qwen35MoeConfig, RopeParams};
pub use decoder_layer::{DecoderLayerMoe, DecoderLayerMoeConfig};
pub use model::Qwen35MoeModel;
pub use sparse_moe::{Router, RoutedExperts, SparseMoeBlock};
pub use text_model::Qwen35MoeTextModel;
```

Modify `ironmlx/src/models/mod.rs`:
```rust
pub use qwen3_5_moe::{Qwen35MoeConfig, Qwen35MoeModel, Qwen35MoeTextModel};
```

- [ ] **Step 4.4: build 验证**

Run: `cargo build -p ironmlx`
Expected: 成功

- [ ] **Step 4.5: Commit T4**

```
git add ironmlx/src/models/qwen3_5_moe/ ironmlx/src/models/mod.rs ironmlx/src/core/loader.rs
git commit -m "$(cat <<'EOF'
feat(p5b-t4): Qwen35MoeModel + impl Model + Loader sanitize compat

Qwen35MoeModel composes Qwen35MoeTextModel + lm_head (untied);
implements core::Model trait. approx_weight_bytes formula accounts
for 256 routed experts + shared_expert per layer + attention + embed.
Loader sanitize tolerates MoE expert key naming (no changes if
naming matches expected scheme).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: 单 prompt omlx argmax 对齐验证

**Files:**
- Create: `ironmlx/tests/p5_qwen35_moe_smoke.rs`

- [ ] **Step 5.1: 写 smoke 集成测试**

Create `ironmlx/tests/p5_qwen35_moe_smoke.rs`:
```rust
//! P5b smoke: load Qwen35MoeModel from real snapshot, run forward
//! on a 4-token prompt, validate output shape + sanity values.
//!
//! Run with:
//!   IRONMLX_MOE_MODEL_DIR=<path-to-snapshot> \
//!     cargo test -p ironmlx --release --test p5_qwen35_moe_smoke -- --ignored --nocapture

use mlx::Dtype;

use ironmlx::core::{Loader, Model};
use ironmlx::models::Qwen35MoeModel;

#[test]
#[ignore]
fn p5b_smoke_load_and_forward() {
    let dir = std::env::var("IRONMLX_MOE_MODEL_DIR")
        .or_else(|_| {
            let home = std::env::var("HOME").unwrap();
            let glob = format!(
                "{home}/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots"
            );
            let entries = std::fs::read_dir(&glob).expect("snapshots dir missing");
            let first = entries
                .filter_map(|e| e.ok())
                .next()
                .expect("at least one snapshot");
            Ok::<String, std::env::VarError>(first.path().to_string_lossy().into_owned())
        })
        .expect("locate snapshot");

    let loader = Loader::open(std::path::Path::new(&dir)).expect("Loader::open");
    let model = Qwen35MoeModel::from_loader(&loader).expect("Qwen35MoeModel::from_loader");

    let input_ids: mlx::Array = (&[100_i32, 200, 300, 400][..], &[1_i32, 4][..])
        .try_into().unwrap();
    let pos = ironmlx::core::generate::build_position_ids(0, 4).expect("build_position_ids");

    let mut cache = Model::make_cache(&model, 1, 16, Dtype::Bfloat16).expect("make_cache");
    let logits = Model::forward_on(
        &model, &input_ids, &pos, None, None, Some(&mut cache), ().into(),
    ).expect("forward_on");

    // shape [1, 1, vocab=248320]
    assert_eq!(logits.shape().as_slice()[0], 1);
    assert_eq!(logits.shape().as_slice()[1], 1);
    assert_eq!(logits.shape().as_slice()[2], 248320);

    // sanity: 有限值
    let v: Vec<f32> = mlx::ops::cast::astype(&logits, Dtype::Float32)
        .unwrap().to_vec().unwrap();
    assert!(v.iter().all(|x| x.is_finite()), "non-finite logits present");
}
```

- [ ] **Step 5.2: 本地手测 + 拿 argmax 对照 omlx**

跑：
```
IRONMLX_MOE_MODEL_DIR=~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/<sha> \
  cargo test -p ironmlx --release --test p5_qwen35_moe_smoke -- --ignored --nocapture
```

Expected: PASS（forward 完成，shape + finite check 通过）

然后拿同 input_ids 在 omlx CLI 跑 forward / generate，比对第一个 token argmax 是否一致：
```
python -m omlx.generate \
  --model ~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/<sha> \
  --prompt-tokens "100 200 300 400" \
  --max-tokens 1 \
  --temp 0
```

若 omlx 不支持 --prompt-tokens flag，手工 prompt encode tokens 100/200/300/400 在 omlx 内部 trace logits → argmax；ironmlx 也 trace 比对。

**对齐验证标准**：第一个 token argmax 100% 一致。若不一致，inline 分析 routing/norm_topk_prob/softmax 顺序是否与 mlx-lm 实现匹配（不抄代码，但行为对齐）。

- [ ] **Step 5.3: Commit T5**

```
git add ironmlx/tests/p5_qwen35_moe_smoke.rs
git commit -m "$(cat <<'EOF'
test(p5b-t5): MoE smoke load+forward + first-token omlx argmax align

#[ignore] integration test loads Qwen35MoeModel from real snapshot
and runs forward_on on a 4-token prompt. First-token argmax
matched against omlx CLI on the same input (out-of-band verification
during local execution).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: G1 升级（如 T0 确认 mlx::gather_qmm 可用） + close-out

**Files:**
- Modify: `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs`（若 G1 可行）
- Modify: 可能 `mlx-sys/` + `mlx/` 添加 gather_qmm cxx bridge + wrapper（如未暴露）

- [ ] **Step 6.1: 条件 G1 实现**

仅当 T0 Step 0.5 决策为 G1 时执行。否则跳过此 step，G2 是最终路径。

如执行 G1：替换 `SparseMoeBlock::routed_g2_forward` 调用为 `routed_g1_forward`，使用 `mlx::quantization::gather_qmm_on(...)`（cxx-mlx wrapper 调用 mlx C++ `gather_qmm` 单 fused op）。具体 wrapper 签名按 T0 确认后填入。

- [ ] **Step 6.2: 跑全 lib + smoke**

Run:
```
cargo test -p ironmlx --lib --release
IRONMLX_MOE_MODEL_DIR=... cargo test -p ironmlx --release --test p5_qwen35_moe_smoke -- --ignored
```
Expected: 全 PASS

- [ ] **Step 6.3: 工具链 hygiene**

```
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings
cargo build --release
```
Expected: 零 warning，build OK

- [ ] **Step 6.4: close-out commit**

```
git add -A
git commit -m "$(cat <<'EOF'
chore(p5b): close-out — MoE forward path closed, omlx aligned

SparseMoeBlock G1/G2 path selected (see commit body). First-token
argmax aligned with omlx CLI baseline on Qwen3.5-35B-A3B-4bit.
Lib unit tests pass; smoke integration test passes locally.
clippy / fmt / release build clean.

P5b sub-phase complete. P5c can proceed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## P5b 闭环条件

- [ ] T0 研究产出在 plan 文件 Step 0.5 处填实
- [ ] `cargo test -p ironmlx --lib --release` 含 qwen3_5_moe::config::tests + qwen3_5_moe::sparse_moe::tests 全 PASS
- [ ] `IRONMLX_MOE_MODEL_DIR=... cargo test -p ironmlx --release --test p5_qwen35_moe_smoke -- --ignored` PASS
- [ ] ironmlx 与 omlx 在 ≥1 prompt 上首 token argmax 一致
- [ ] `cargo +nightly clippy --all-features --workspace -- -D warnings` 零 warning

满足全部 → P5c 启动。

---

## Self-Review Notes

- ✓ Spec coverage：Qwen35MoeConfig (§3.2) / Qwen35MoeTextModel (§3.3) / Qwen35MoeModel + impl Model (§3.4) / DecoderLayerMoe (§3.5) / SparseMoeBlock+Router (§3.6) / Loader sanitize (§3.7)
- ✓ T0 研究覆盖所有 known unknowns（snapshot keys / gather op / norm_topk_prob / softmax 顺序 / omlx baseline）
- ✓ G1/G2 决策路径明确，G2 作为基线，G1 作为优化
- ✓ Task 数 = 6 + Pre-flight，符合 [feedback_task_breakdown_bounded](../../../.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/feedback_task_breakdown_bounded.md) 5-7 范围
- ✓ "不对齐实现"哲学：mlx-lm 仅作算法 reference，omlx 作数值 baseline，独立实现
