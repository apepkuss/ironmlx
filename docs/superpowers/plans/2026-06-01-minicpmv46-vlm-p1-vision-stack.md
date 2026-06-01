# MiniCPM-V-4.6 VLM — P1 Vision Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the MiniCPM-V-4.6 SigLIP vision stack (embeddings + 27-layer encoder + mid-encoder VitMerger + Merger) and verify its merged vision embeddings match mlx-vlm `get_vision_embedding` on a fixed preprocessed-pixel fixture.

**Architecture:** New `ironmlx/src/models/minicpmv4_6/vision/` module. Vision is bf16 (unquantized). Forward: `SiglipEmbeddings → encoder L0..6 → VitMerger (grid÷2×2) → encoder L7..26 → post_layernorm → Merger (grid÷2×2) → [N, 1024]`. Reuses `nn::LayerNorm`, the `gelu_tanh` numeric (lifted to shared), `mlx::fast::scaled_dot_product_attention`, raw-Array weight-loading pattern from `models/vision/`.

**Tech Stack:** Rust, cxx-mlx (`mlx` crate), MLX Metal. Reference (observation only): `/Users/xin/workspace/iron-rivals/mlx-vlm/mlx_vlm/models/minicpmv4_6/{vision,minicpmv4_6}.py`.

**Scope:** This is **P1 of 3** (per spec `docs/superpowers/specs/2026-06-01-minicpmv46-vlm-design.md` §9). P1 = vision stack with fixture-fed parity. P2 (single-image e2e) and P3 (multi-slice + multi-image) are separate plans authored after P1 lands — P1's fixtures resolve open question §11.1 (patch Conv vs packed).

**Environment (every cargo/test command):** `source ~/.local/mlx/mlx-env.sh` first (sets MLX_DIR / MLX_METAL_PATH / DYLD_LIBRARY_PATH).

**Authoritative dims** (spec §1.2-1.3): vision hidden=1152, heads=16 (head_dim=72), 27 layers, patch=14, image_size=980 (pos table 70×70=4900), intermediate=4304, eps=1e-6, gelu_pytorch_tanh. group_hidden=4608=1152·4, window_intermediate=17216=4304·4, merger out=1024 (LM hidden). insert_layer_id=6, merge_group=(2,2).

---

## File Structure

- Create `ironmlx/src/models/minicpmv4_6/vision/mod.rs` — `MiniCpmV46Vision` (owns sub-modules; `from_loader` + `compute_vision_embeds`).
- Create `ironmlx/src/models/minicpmv4_6/vision/embeddings.rs` — `SiglipEmbeddings` (patch embed + pos-bucket).
- Create `ironmlx/src/models/minicpmv4_6/vision/encoder.rs` — `SiglipEncoderLayer`, `SiglipEncoder`.
- Create `ironmlx/src/models/minicpmv4_6/vision/merger.rs` — `VitMerger`, `Merger`.
- Modify `ironmlx/src/models/minicpmv4_6/config.rs` — add `MiniCpmV46VisionConfig` + parse it (do NOT change the existing text-only path).
- Modify `ironmlx/src/models/minicpmv4_6/mod.rs` — `pub mod vision;` + re-exports.
- Modify `ironmlx/src/nn/mod.rs` + `ironmlx/src/models/vision/block.rs` — lift `gelu_tanh` to `pub(crate)` shared use (see Task 3).
- Modify `ironmlx/src/core/loader.rs` — `open_multimodal` must also retain `vit_merger.*` and `merger.*` keys (spec §8.4).
- Create `ironmlx/src/models/minicpmv4_6/vision/tests fixtures` driver `ironmlx/tests/fixtures/minicpmv46_vl/gen_vision_embeds.py`.
- Create `ironmlx/tests/minicpmv46_vision_parity.rs` — `#[ignore]` fixture-fed parity test.

---

### Task 1: `MiniCpmV46VisionConfig` parsing

**Files:**
- Modify: `ironmlx/src/models/minicpmv4_6/config.rs`
- Test: same file `#[cfg(test)]`

- [ ] **Step 1: Write the failing test**

Add to the `tests` module in `config.rs`:

```rust
#[test]
fn parses_vision_config_and_merge_params() {
    let raw = raw_minicpmv46_config(); // existing helper; ensure it has vision_config + insert_layer_id
    let vc = MiniCpmV46VisionConfig::from_raw(&raw).expect("parse");
    assert_eq!(vc.hidden_size, 1152);
    assert_eq!(vc.num_hidden_layers, 27);
    assert_eq!(vc.num_attention_heads, 16);
    assert_eq!(vc.head_dim(), 72);
    assert_eq!(vc.patch_size, 14);
    assert_eq!(vc.image_size, 980);
    assert_eq!(vc.pos_grid_side, 70); // 980 / 14
    assert_eq!(vc.insert_layer_id, 6);
    assert_eq!(vc.merge_group, (2, 2));
    assert_eq!(vc.image_token_id, 248056);
    assert!((vc.layer_norm_eps - 1e-6).abs() < 1e-9);
}
```

Ensure `raw_minicpmv46_config()` (added in commit 236db39) includes the top-level `insert_layer_id: 6` and `image_token_id: 248056` and the SigLIP `vision_config` block (it already does — verify).

- [ ] **Step 2: Run test to verify it fails**

Run: `source ~/.local/mlx/mlx-env.sh && cargo test --release -p ironmlx --lib minicpmv4_6::config::tests::parses_vision_config -- --nocapture`
Expected: FAIL — `MiniCpmV46VisionConfig` not defined.

- [ ] **Step 3: Implement `MiniCpmV46VisionConfig`**

Add to `config.rs`:

```rust
use serde::Deserialize;

/// SigLIP vision config for MiniCPM-V-4.6, plus the top-level merge params the
/// vision stack needs. Parsed separately from the text Qwen35Config.
#[derive(Debug, Clone)]
pub struct MiniCpmV46VisionConfig {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub patch_size: i32,
    pub image_size: i32,
    pub layer_norm_eps: f32,
    /// sqrt of position_embedding table rows = image_size / patch_size (70).
    pub pos_grid_side: i32,
    /// Top-level config.json fields the vision forward needs.
    pub insert_layer_id: i32,
    pub merge_group: (i32, i32),
    pub image_token_id: i32,
}

impl MiniCpmV46VisionConfig {
    pub fn head_dim(&self) -> i32 {
        self.hidden_size / self.num_attention_heads
    }

    pub fn from_loader(loader: &crate::core::Loader) -> crate::Result<Self> {
        Self::from_raw(loader.config_raw_value())
    }

    pub fn from_raw(raw: &serde_json::Value) -> crate::Result<Self> {
        #[derive(Deserialize)]
        struct VisionRaw {
            hidden_size: i32,
            intermediate_size: i32,
            num_hidden_layers: i32,
            num_attention_heads: i32,
            patch_size: i32,
            image_size: i32,
            #[serde(default = "default_vis_eps")]
            layer_norm_eps: f32,
        }
        fn default_vis_eps() -> f32 { 1e-6 }

        let vraw = raw
            .get("vision_config")
            .ok_or_else(|| anyhow!("MiniCPM-V-4.6 config missing vision_config"))?;
        let v: VisionRaw = serde_json::from_value(vraw.clone())
            .context("deserialize MiniCpmV46VisionConfig")?;
        let insert_layer_id = raw
            .get("insert_layer_id")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(6) as i32;
        let image_token_id = raw
            .get("image_token_id")
            .and_then(serde_json::Value::as_i64)
            .ok_or_else(|| anyhow!("MiniCPM-V-4.6 config missing image_token_id"))? as i32;
        let pos_grid_side = v.image_size / v.patch_size;
        Ok(Self {
            hidden_size: v.hidden_size,
            intermediate_size: v.intermediate_size,
            num_hidden_layers: v.num_hidden_layers,
            num_attention_heads: v.num_attention_heads,
            patch_size: v.patch_size,
            image_size: v.image_size,
            layer_norm_eps: v.layer_norm_eps,
            pos_grid_side,
            insert_layer_id,
            merge_group: (2, 2), // downsample_mode "16x" → window_kernel_size (2,2)
            image_token_id,
        })
    }
}
```

(`anyhow`/`Context` are already imported at top of `config.rs`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `source ~/.local/mlx/mlx-env.sh && cargo test --release -p ironmlx --lib minicpmv4_6::config -- --nocapture`
Expected: PASS (all config tests, including the new one + the 6 existing).

- [ ] **Step 5: Commit**

```bash
git add ironmlx/src/models/minicpmv4_6/config.rs
git commit -m "feat(minicpmv4_6): parse SigLIP vision config + merge params"
```

---

### Task 2: `SiglipEmbeddings` (patch embed + position-bucket interpolation)

**Files:**
- Create: `ironmlx/src/models/minicpmv4_6/vision/embeddings.rs`
- Modify: `ironmlx/src/models/minicpmv4_6/mod.rs` (`pub mod vision;`), `vision/mod.rs` (`pub mod embeddings;`)
- Test: in `embeddings.rs` `#[cfg(test)]` (shape-only; numeric parity deferred to Task 7)

**Weight prefixes:** `vision_tower.embeddings.patch_embedding.{weight,bias}` (weight `[1152,14,14,3]`), `vision_tower.embeddings.position_embedding.weight` (`[4900,1152]`). bf16, load as raw Arrays (pattern: `models/vision/patch_embed.rs`).

**Patch embed decision (open question §11.1):** Implement the **packed-matmul** path first (P1 fixture from mlx-vlm uses patch-packed layout; `_packed_patch_embedding`): reshape conv weight `[1152,14,14,3] → [1152, 588]`, and pixel input `[1, 14, n·14, 3] → [1, n, 588]`, then `patches @ weightᵀ + bias`. The fixture (Task 7) carries the exact pixel layout; if it is CHW/Conv instead, adapt here and re-run Task 7.

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use mlx::{Array, Dtype};

    #[test]
    fn position_bucket_ids_match_grid() {
        // grid 4x4, pos_grid_side=70 → buckets in [0,70); ids = bh*70 + bw.
        let ids = position_bucket_ids(4, 4, 70);
        assert_eq!(ids.len(), 16);
        // first patch bucket (frac=0) maps to 0; ids monotonic non-decreasing in row-major.
        assert_eq!(ids[0], 0);
        assert!(ids.iter().all(|&v| v >= 0 && v < 70 * 70));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source ~/.local/mlx/mlx-env.sh && cargo test --release -p ironmlx --lib minicpmv4_6::vision::embeddings -- --nocapture`
Expected: FAIL — module/function not defined.

- [ ] **Step 3: Implement `SiglipEmbeddings`**

```rust
//! SigLIP patch embedding + NaViT-style position-bucket interpolation.

use anyhow::Result;
use mlx::{ops, Array, StreamOrDevice};

use crate::core::Loader;
use super::super::config::MiniCpmV46VisionConfig;

pub struct SiglipEmbeddings {
    /// Conv weight reshaped to [hidden, patch*patch*channels] = [1152, 588].
    patch_w_2d: Array,
    patch_b: Array,
    /// [pos_grid_side^2, hidden] = [4900, 1152].
    pos_embed: Array,
    hidden: i32,
    patch: i32,
    pos_grid_side: i32,
}

/// Map each patch of a (grid_h, grid_w) image to a learned-position-table id
/// via fractional bucketing against `pos_grid_side` boundaries (mlx-vlm
/// `_build_position_buckets`). Row-major over (h, w).
pub fn position_bucket_ids(grid_h: i32, grid_w: i32, side: i32) -> Vec<i32> {
    let bucket = |n: i32| -> Vec<i32> {
        let n = n.max(1);
        (0..n)
            .map(|i| {
                let frac = ((i as f32) / (n as f32)).min(1.0 - 1e-6);
                // count boundaries (k/side for k=1..side-1) that frac >= .
                let mut b = 0;
                for k in 1..side {
                    if frac >= (k as f32) / (side as f32) {
                        b += 1;
                    }
                }
                b
            })
            .collect()
    };
    let bh = bucket(grid_h);
    let bw = bucket(grid_w);
    let mut ids = Vec::with_capacity((grid_h * grid_w) as usize);
    for &h in &bh {
        for &w in &bw {
            ids.push(h * side + w);
        }
    }
    ids
}

impl SiglipEmbeddings {
    pub fn from_loader(loader: &Loader, cfg: &MiniCpmV46VisionConfig) -> Result<Self> {
        let w = loader
            .tensor("vision_tower.embeddings.patch_embedding.weight")?
            .clone();
        let patch_elems = cfg.patch_size * cfg.patch_size * 3;
        let patch_w_2d = w.reshape(&[cfg.hidden_size, patch_elems][..])?;
        let patch_b = loader
            .tensor("vision_tower.embeddings.patch_embedding.bias")?
            .clone();
        let pos_embed = loader
            .tensor("vision_tower.embeddings.position_embedding.weight")?
            .clone();
        Ok(Self {
            patch_w_2d,
            patch_b,
            pos_embed,
            hidden: cfg.hidden_size,
            patch: cfg.patch_size,
            pos_grid_side: cfg.pos_grid_side,
        })
    }

    /// `pixel_values`: patch-packed `[1, patch, n*patch, 3]`. Returns `[1, grid_h*grid_w, hidden]`.
    pub fn forward_on(
        &self,
        pixel_values: &Array,
        grid_h: i32,
        grid_w: i32,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        let dims = pixel_values.shape();
        let d = dims.as_slice();
        let (p, total_w, c) = (d[1], d[2], d[3]);
        let n = total_w / p;
        // [1, p, n*p, c] → [1, p, n, p, c] → [1, n, p, p, c] → [1, n, p*p*c]
        let x = pixel_values.reshape_on(&[1, p, n, p, c][..], target)?;
        let x = x.transpose_axes_on(&[0_i32, 2, 1, 3, 4][..], target)?;
        let x = x.reshape_on(&[1, n, p * p * c][..], target)?;
        let wt = self.patch_w_2d.transpose_on(target)?; // [588, 1152]
        let mut embeds = ops::matmul(&x, &wt)?; // [1, n, 1152]
        embeds = &embeds + &self.patch_b;
        // position embeddings
        let ids = position_bucket_ids(grid_h, grid_w, self.pos_grid_side);
        let id_arr: Array = (ids.as_slice(), &[ids.len() as i32][..]).try_into()?;
        let pos = ops::indexing::take(&self.pos_embed, &id_arr, 0)?; // [n, 1152]
        let pos = pos.reshape_on(&[1, ids.len() as i32, self.hidden][..], target)?;
        Ok(&embeds + &pos)
    }

    pub(crate) fn collect_weights<'a>(&'a self, out: &mut Vec<&'a Array>) {
        out.push(&self.patch_w_2d);
        out.push(&self.patch_b);
        out.push(&self.pos_embed);
    }
}
```

Add `pub mod embeddings;` to `vision/mod.rs` and `pub mod vision;` to `minicpmv4_6/mod.rs`.

- [ ] **Step 4: Run test to verify it passes**

Run: `source ~/.local/mlx/mlx-env.sh && cargo test --release -p ironmlx --lib minicpmv4_6::vision::embeddings -- --nocapture`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ironmlx/src/models/minicpmv4_6/vision/ ironmlx/src/models/minicpmv4_6/mod.rs
git commit -m "feat(minicpmv4_6): SigLIP embeddings + position-bucket interpolation"
```

---

### Task 3: Lift `gelu_tanh` to shared + `SiglipEncoder`

**Files:**
- Modify: `ironmlx/src/models/vision/block.rs` (make `gelu_tanh` `pub(crate)`), or move to `ironmlx/src/nn/mod.rs` as `pub(crate) fn gelu_tanh`. Prefer lifting to `nn` (used by two vision modules now). Update `block.rs` to import it.
- Create: `ironmlx/src/models/minicpmv4_6/vision/encoder.rs`
- Test: `encoder.rs` `#[cfg(test)]` shape test (random weights).

**Weight prefixes** (per layer `vision_tower.encoder.layers.{i}.`): `layer_norm1.{weight,bias}`, `self_attn.{q,k,v,out}_proj.{weight,bias}` (all `[1152,1152]`/`[1152]`), `layer_norm2.{weight,bias}`, `mlp.fc1.{weight,bias}` (`[4304,1152]`/`[4304]`), `mlp.fc2.{weight,bias}` (`[1152,4304]`/`[1152]`). All bf16.

**Reuse:** `nn::LayerNorm::from_loader(loader, "{prefix}.layer_norm1", eps)`; the lifted `gelu_tanh`. Attention is standard MHA, **no RoPE** (unlike `models/vision/block.rs::VitAttention`), separate q/k/v/out (not packed qkv).

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use mlx::{Array, Dtype};

    #[test]
    fn encoder_layer_preserves_shape() {
        let h = 1152; let heads = 16;
        let layer = SiglipEncoderLayer::new_for_test(h, heads, 4304);
        let x = Array::zeros(&[1, 9, h][..], Dtype::Bfloat16).unwrap();
        let y = layer.forward_on(&x, ()).unwrap();
        assert_eq!(y.shape().as_slice(), &[1, 9, h]);
    }
}
```

(`new_for_test` builds the layer from zero/identity Arrays — add a `#[cfg(test)] pub fn new_for_test`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `source ~/.local/mlx/mlx-env.sh && cargo test --release -p ironmlx --lib minicpmv4_6::vision::encoder -- --nocapture`
Expected: FAIL — not defined.

- [ ] **Step 3: Implement `SiglipEncoderLayer` + `SiglipEncoder`**

```rust
//! SigLIP encoder: 27 × pre-norm MHA(+bias) / GELU-tanh MLP layers. No RoPE.

use anyhow::Result;
use mlx::fast::scaled_dot_product_attention;
use mlx::{ops, Array, StreamOrDevice};

use crate::core::Loader;
use crate::nn::{gelu_tanh, LayerNorm}; // gelu_tanh lifted from models/vision/block.rs

struct Mha { qw: Array, qb: Array, kw: Array, kb: Array, vw: Array, vb: Array, ow: Array, ob: Array, heads: i32, head_dim: i32 }

impl Mha {
    fn from_loader(loader: &Loader, prefix: &str, hidden: i32, heads: i32) -> Result<Self> {
        let g = |n: &str| loader.tensor(&format!("{prefix}.{n}")).map(Clone::clone);
        Ok(Self {
            qw: g("q_proj.weight")?, qb: g("q_proj.bias")?,
            kw: g("k_proj.weight")?, kb: g("k_proj.bias")?,
            vw: g("v_proj.weight")?, vb: g("v_proj.bias")?,
            ow: g("out_proj.weight")?, ob: g("out_proj.bias")?,
            heads, head_dim: hidden / heads,
        })
    }
    fn proj(x: &Array, w: &Array, b: &Array, t: StreamOrDevice) -> Result<Array> {
        Ok(&ops::matmul(x, &w.transpose_on(t)?)? + b)
    }
    fn forward_on(&self, x: &Array, t: StreamOrDevice) -> Result<Array> {
        let d = x.shape(); let (bsz, s) = (d.as_slice()[0], d.as_slice()[1]);
        let to_heads = |a: Array| -> Result<Array> {
            Ok(a.reshape_on(&[bsz, s, self.heads, self.head_dim][..], t)?
                .transpose_axes_on(&[0_i32, 2, 1, 3][..], t)?)
        };
        let q = to_heads(Self::proj(x, &self.qw, &self.qb, t)?)?;
        let k = to_heads(Self::proj(x, &self.kw, &self.kb, t)?)?;
        let v = to_heads(Self::proj(x, &self.vw, &self.vb, t)?)?;
        let scale = (self.head_dim as f32).powf(-0.5);
        let o = scaled_dot_product_attention(&q, &k, &v, scale, None)?; // no mask
        let o = o.transpose_axes_on(&[0_i32, 2, 1, 3][..], t)?
            .reshape_on(&[bsz, s, self.heads * self.head_dim][..], t)?;
        Self::proj(&o, &self.ow, &self.ob, t)
    }
}

pub struct SiglipEncoderLayer { ln1: LayerNorm, attn: Mha, ln2: LayerNorm, fc1w: Array, fc1b: Array, fc2w: Array, fc2b: Array }

impl SiglipEncoderLayer {
    pub fn from_loader(loader: &Loader, prefix: &str, hidden: i32, heads: i32, eps: f32) -> Result<Self> {
        let g = |n: &str| loader.tensor(&format!("{prefix}.{n}")).map(Clone::clone);
        Ok(Self {
            ln1: LayerNorm::from_loader(loader, &format!("{prefix}.layer_norm1"), eps)?,
            attn: Mha::from_loader(loader, &format!("{prefix}.self_attn"), hidden, heads)?,
            ln2: LayerNorm::from_loader(loader, &format!("{prefix}.layer_norm2"), eps)?,
            fc1w: g("mlp.fc1.weight")?, fc1b: g("mlp.fc1.bias")?,
            fc2w: g("mlp.fc2.weight")?, fc2b: g("mlp.fc2.bias")?,
        })
    }
    pub fn forward_on(&self, x: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
        let t = target.into();
        let h = &self.attn.forward_on(&self.ln1.forward_on(x, t)?, t)? + x;
        let n = self.ln2.forward_on(&h, t)?;
        let mlp = &ops::matmul(&n, &self.fc1w.transpose_on(t)?)? + &self.fc1b;
        let mlp = gelu_tanh(&mlp, t)?;
        let mlp = &ops::matmul(&mlp, &self.fc2w.transpose_on(t)?)? + &self.fc2b;
        Ok(&h + &mlp)
    }
}

pub struct SiglipEncoder { pub layers: Vec<SiglipEncoderLayer> }

impl SiglipEncoder {
    pub fn from_loader(loader: &Loader, cfg: &super::super::config::MiniCpmV46VisionConfig) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers as usize);
        for i in 0..cfg.num_hidden_layers {
            layers.push(SiglipEncoderLayer::from_loader(
                loader, &format!("vision_tower.encoder.layers.{i}"),
                cfg.hidden_size, cfg.num_attention_heads, cfg.layer_norm_eps)?);
        }
        Ok(Self { layers })
    }
}
```

Lift `gelu_tanh`: cut the `fn gelu_tanh(...)` body from `models/vision/block.rs` into `nn/mod.rs` as `pub(crate) fn gelu_tanh`, and in `block.rs` replace the local def with `use crate::nn::gelu_tanh;`. Verify `block.rs` still compiles. Add `new_for_test` constructor behind `#[cfg(test)]`.

> Confirm against the compiler: `scaled_dot_product_attention` signature (arg order, mask type `Option<&Array>`), `transpose_axes_on`/`reshape_on` names — these match `models/vision/block.rs` usage; adapt if the binding differs.

- [ ] **Step 4: Run test to verify it passes**

Run: `source ~/.local/mlx/mlx-env.sh && cargo test --release -p ironmlx --lib minicpmv4_6::vision::encoder -- --nocapture` then `cargo test --release -p ironmlx --lib vision::block` (regression for the lift).
Expected: PASS both.

- [ ] **Step 5: Commit**

```bash
git add ironmlx/src/nn/mod.rs ironmlx/src/models/vision/block.rs ironmlx/src/models/minicpmv4_6/vision/encoder.rs ironmlx/src/models/minicpmv4_6/vision/mod.rs
git commit -m "feat(minicpmv4_6): SigLIP encoder layer/stack; lift gelu_tanh to nn"
```

---

### Task 4: `VitMerger` (mid-encoder window resampler)

**Files:**
- Create/extend: `ironmlx/src/models/minicpmv4_6/vision/merger.rs`
- Test: shape + divisibility-Err unit test.

**Weight prefixes** (`vit_merger.`): `layer_norm1.{weight,bias}` (`[1152]`), `self_attn.{q,k,v,out}_proj.{weight,bias}` (`[1152,1152]`), `pre_norm.{weight,bias}` (`[4608]`), `linear_1.{weight,bias}` (`[17216,4608]`), `linear_2.{weight,bias}` (`[1152,17216]`).

**Forward** (mlx-vlm `VitMerger.__call__`, observation): input `x [grid_h*grid_w, 1152]`, grid.
1. require `grid_h % 2 == 0 && grid_w % 2 == 0` else `Err`.
2. `merged_h=grid_h/2, merged_w=grid_w/2`. reshape `[grid_h, grid_w, 1152]` → `[merged_h, 2, merged_w, 2, 1152]` → transpose `(0,2,1,3,4)` → `[merged_h*merged_w, 4, 1152]` (windows).
3. `normed = layer_norm1(windows)`; `attn = self_attn(normed, normed, normed)` (CrossAttention over the 4 group tokens, 16 heads); `windows = windows + attn`.
4. `residual = mean(windows, axis=1)` → `[M, 1152]`.
5. `merged = windows.reshape([M, 4608])`; `merged = pre_norm(merged)`; `linear_1` → GELU(precise) → `linear_2` → `[M, 1152]`.
6. return `merged + residual`, `merged_h`, `merged_w`.

> **GELU note:** mlx-vlm uses `nn.GELU(approx="precise")` here (NOT tanh). "precise" = erf-based exact GELU. Use mlx's exact gelu (e.g. `ops::gelu` if available, else `0.5*x*(1+erf(x/sqrt2))`). Confirm the right binding; this differs from the encoder's tanh GELU.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn vit_merger_halves_grid_and_errors_on_odd() {
    let m = VitMerger::new_for_test(1152, 16, 17216);
    let x = Array::zeros(&[6 * 6, 1152][..], Dtype::Bfloat16).unwrap();
    let (out, h, w) = m.forward_on(&x, 6, 6, ()).unwrap();
    assert_eq!((h, w), (3, 3));
    assert_eq!(out.shape().as_slice(), &[9, 1152]);
    assert!(m.forward_on(&x, 5, 6, ()).is_err()); // odd grid_h
}
```

- [ ] **Step 2: Run to verify it fails** — Run: `cargo test --release -p ironmlx --lib minicpmv4_6::vision::merger::tests::vit_merger -- --nocapture`. Expected FAIL.

- [ ] **Step 3: Implement `VitMerger`** — `CrossAttention` is the same MHA shape as Task 3's `Mha` but query/key/value are the same tensor; reuse a small SDPA helper. Implement per the 6 steps above with exact reshapes. (Full code authored at execution time against the compiler, using Task 3's `Mha` SDPA pattern verbatim for the attention sub-step and the reshape math specified above.)

- [ ] **Step 4: Run to verify it passes** — Expected PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(minicpmv4_6): VitMerger window resampler"`

---

### Task 5: `Merger` (2×2 → LM-hidden projection)

**Files:** `ironmlx/src/models/minicpmv4_6/vision/merger.rs` (extend). Test: shape (out=1024).

**Weight prefixes** (`merger.mlp.0.`): `pre_norm.{weight,bias}` (`[4608]`), `linear_1.{weight,bias}` (`[4608,4608]`), `linear_2.{weight,bias}` (`[1024,4608]`). merger_times=1 → single block.

**Forward** (mlx-vlm `Merger.__call__`): input `[grid_h*grid_w, 1152]`, grid. reshape `[grid_h,grid_w,1152]` → `[mh,2,mw,2,1152]` → transpose `(0,2,1,3,4)` → `[mh*mw, 4608]`; then `pre_norm → linear_1 → GELU(precise) → linear_2` → `[mh*mw, 1024]`. Require grid even else `Err`.

- [ ] **Step 1: failing test** — `merger_outputs_lm_hidden`: 6×6 input → out `[9, 1024]`.
- [ ] **Step 2: run fail.**
- [ ] **Step 3: implement** per the forward above (reshape identical to VitMerger step 2; then MergerBlock = pre_norm/linear_1/gelu_precise/linear_2).
- [ ] **Step 4: run pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat(minicpmv4_6): Merger projection to LM hidden"`

---

### Task 6: `MiniCpmV46Vision` orchestration + `Loader` retention

**Files:**
- Create: `ironmlx/src/models/minicpmv4_6/vision/mod.rs` (`MiniCpmV46Vision`).
- Modify: `ironmlx/src/core/loader.rs` — `open_multimodal` retains `vit_merger.*` + `merger.*` (spec §8.4).
- Test: `loader.rs` unit test that `open_multimodal` keeps those keys; orchestration shape test deferred to Task 7 (needs real weights).

- [ ] **Step 1: Write the failing test (loader retention)**

In `loader.rs` `#[cfg(test)]`, extend the existing `sanitize_drops_vision_tower_keys` style test:

```rust
#[test]
fn sanitize_keeps_resampler_keys_when_multimodal() {
    let arr = Array::zeros(&[2,2][..], Dtype::Bfloat16).unwrap();
    let mut w = HashMap::new();
    w.insert("vit_merger.linear_1.weight".into(), arr.clone());
    w.insert("merger.mlp.0.linear_2.weight".into(), arr.clone());
    w.insert("vision_tower.embeddings.patch_embedding.weight".into(), arr.clone());
    Loader::sanitize(&mut w, &empty_text_config(), /* keep_vision_tower */ true).unwrap();
    assert!(w.contains_key("vit_merger.linear_1.weight"));
    assert!(w.contains_key("merger.mlp.0.linear_2.weight"));
}
```

- [ ] **Step 2: Run to verify it fails** — Run: `cargo test --release -p ironmlx --lib loader::tests::sanitize_keeps_resampler -- --nocapture`. Expected FAIL (keys dropped — current retain only spares `vision_tower.`).

- [ ] **Step 3: Implement loader retention + `MiniCpmV46Vision`**

In `loader.rs` sanitize, the `keep_vision_tower` branch currently `retain`s by dropping `audio_*`. Ensure the drop-list for the `else` (text-only) branch also drops `vit_merger.` / `merger.`, and the keep branch retains them. Concretely, in the text-only `retain`, add `&& !k.starts_with("vit_merger.") && !k.starts_with("merger.")`; the multimodal branch already keeps everything except audio, so `vit_merger.`/`merger.` are retained — **verify** the text-only path drops them (so `Loader::open` stays lean) and multimodal keeps them.

`vision/mod.rs`:

```rust
pub mod embeddings;
pub mod encoder;
pub mod merger;

use anyhow::Result;
use mlx::{Array, StreamOrDevice};
use crate::core::Loader;
use super::config::MiniCpmV46VisionConfig;
use self::{embeddings::SiglipEmbeddings, encoder::SiglipEncoder, merger::{VitMerger, Merger}};

pub struct MiniCpmV46Vision {
    embeddings: SiglipEmbeddings,
    encoder: SiglipEncoder,
    vit_merger: VitMerger,
    merger: Merger,
    post_ln: crate::nn::LayerNorm,
    insert_layer_id: i32,
}

impl MiniCpmV46Vision {
    pub fn from_loader(loader: &Loader, cfg: &MiniCpmV46VisionConfig) -> Result<Self> {
        Ok(Self {
            embeddings: SiglipEmbeddings::from_loader(loader, cfg)?,
            encoder: SiglipEncoder::from_loader(loader, cfg)?,
            vit_merger: VitMerger::from_loader(loader, cfg)?,
            merger: Merger::from_loader(loader, cfg)?,
            post_ln: crate::nn::LayerNorm::from_loader(loader, "vision_tower.post_layernorm", cfg.layer_norm_eps)?,
            insert_layer_id: cfg.insert_layer_id,
        })
    }

    /// Single image: `pixel_values` patch-packed, `(grid_h, grid_w)`.
    /// Returns merged vision embeddings `[N, lm_hidden=1024]`.
    pub fn compute_vision_embeds(
        &self, pixel_values: &Array, grid_h: i32, grid_w: i32,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let t = target.into();
        let mut h = self.embeddings.forward_on(pixel_values, grid_h, grid_w, t)?; // [1, G, 1152]
        let (mut gh, mut gw) = (grid_h, grid_w);
        for (i, layer) in self.encoder.layers.iter().enumerate() {
            h = layer.forward_on(&h, t)?;
            if i as i32 == self.insert_layer_id {
                let row = h.reshape_on(&[gh * gw, h.shape().as_slice()[2]][..], t)?;
                let (merged, nh, nw) = self.vit_merger.forward_on(&row, gh, gw, t)?;
                gh = nh; gw = nw;
                h = merged.reshape_on(&[1, gh * gw, merged.shape().as_slice()[1]][..], t)?;
            }
        }
        let h = self.post_ln.forward_on(&h, t)?;
        let row = h.reshape_on(&[gh * gw, h.shape().as_slice()[2]][..], t)?;
        let (merged, _, _) = self.merger.forward_on(&row, gh, gw, t)?;
        Ok(merged)
    }
}
```

- [ ] **Step 4: Run to verify it passes** — Run: `cargo test --release -p ironmlx --lib loader -- --nocapture` (retention test) + `cargo build --release -p ironmlx` (orchestration compiles).
Expected: retention PASS, build OK.

- [ ] **Step 5: Commit** — `git commit -m "feat(minicpmv4_6): vision orchestration + retain resampler weights in loader"`

---

### Task 7: P1 acceptance — vision-embeds parity vs mlx-vlm

**Files:**
- Create: `ironmlx/tests/fixtures/minicpmv46_vl/gen_vision_embeds.py` (mlx-vlm driver), `.gitignore` (`expected_*.npy`, `input_*.npy`).
- Create: `ironmlx/tests/minicpmv46_vision_parity.rs` (`#[ignore]`).

- [ ] **Step 1: Write the fixture generator**

```python
"""Dump MiniCPM-V-4.6 vision-stack inputs + merged vision embeds via mlx-vlm.
Run from the mlx-vlm checkout:
  cd /Users/xin/workspace/iron-rivals/mlx-vlm
  MINICPMV46_MODEL=<snap> uv run --with-editable . python \
    /Users/xin/workspace/ironmlx-backend-minicpmv46/ironmlx/tests/fixtures/minicpmv46_vl/gen_vision_embeds.py
"""
import os; from pathlib import Path
import numpy as np, mlx.core as mx
from mlx_vlm import load
OUT = Path(__file__).parent
model, processor = load(os.environ["MINICPMV46_MODEL"])
# Build a single preprocessed image (slice_mode off / single slice) via the processor.
# Use a fixed fixture image; capture the EXACT pixel_values + tgt_sizes the model consumes.
img = OUT.parent / "p6_qwen35_vl" / "coco_sample.jpg"
inputs = processor.image_processor([str(img)], max_slice_nums=1)  # adapt to processor API
pix = inputs["pixel_values"][0][0]           # the single slice's pixel array
tgt = inputs["tgt_sizes"][0][0]              # (grid_h, grid_w)
mx.save(str(OUT/"input_pixel_values.npy"), mx.array(np.array(pix), dtype=mx.bfloat16).astype(mx.float32))
np.save(str(OUT/"input_grid.npy"), np.array(tgt, dtype=np.int32))
emb = model.get_vision_embedding([[mx.array(np.array(pix))]], [[np.array(tgt)]])[0]  # [N,1024]
mx.eval(emb)
mx.save(str(OUT/"expected_vision_embeds.npy"), emb.astype(mx.float32))
print("grid", np.array(tgt), "emb", emb.shape)
```

> The exact processor entry-point + key names (`pixel_values`/`tgt_sizes`) must be confirmed against `processing_minicpmv4_6.py` at execution time; the script captures whatever layout the model actually consumes so the Rust side feeds identical bytes.

- [ ] **Step 2: Generate the fixture**

Run the command in the docstring. Expected: prints grid + emb shape `[N, 1024]`; writes `input_pixel_values.npy`, `input_grid.npy`, `expected_vision_embeds.npy`.

- [ ] **Step 3: Write the parity test**

```rust
//! P1 acceptance: MiniCPM-V-4.6 vision stack vs mlx-vlm get_vision_embedding.
use mlx::{Array, Dtype};
use ironmlx::core::Loader;
use ironmlx::models::minicpmv4_6::{config::MiniCpmV46VisionConfig, vision::MiniCpmV46Vision};

const DIR: &str = "tests/fixtures/minicpmv46_vl";
fn npy(n: &str) -> Array { mlx::io::load_npy(&format!("{DIR}/{n}")).expect(n) }

#[test]
#[ignore = "requires MINICPMV46_MODEL + generated fixtures"]
fn minicpmv46_vision_embeds_match_mlxvlm() {
    let dir = std::env::var("MINICPMV46_MODEL").expect("MINICPMV46_MODEL");
    let loader = Loader::open_multimodal(std::path::Path::new(&dir)).expect("open_multimodal");
    let cfg = MiniCpmV46VisionConfig::from_loader(&loader).expect("cfg");
    let vision = MiniCpmV46Vision::from_loader(&loader, &cfg).expect("vision");

    let grid: Vec<i32> = npy("input_grid.npy").to_vec().expect("grid");
    let (gh, gw) = (grid[0], grid[1]);
    let pix = mlx::ops::cast::astype(&npy("input_pixel_values.npy"), Dtype::Bfloat16).unwrap();
    // pix must be reshaped to the packed [1, patch, n*patch, 3] layout the embeddings expect;
    // adapt from the captured layout (Step 1) — assert the reshape matches gh*gw patches.
    let got = vision.compute_vision_embeds(&pix, gh, gw, ()).expect("embeds");

    let exp = npy("expected_vision_embeds.npy");
    let a: Vec<f32> = mlx::ops::cast::astype(&got, Dtype::Float32).unwrap().to_vec().unwrap();
    let b: Vec<f32> = exp.to_vec().unwrap();
    assert_eq!(a.len(), b.len(), "shape mismatch");
    let max_abs = a.iter().zip(&b).map(|(x,y)| (x-y).abs()).fold(0.0f32, f32::max);
    let dot: f32 = a.iter().zip(&b).map(|(x,y)| x*y).sum();
    let na: f32 = a.iter().map(|x| x*x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x*x).sum::<f32>().sqrt();
    let cos = dot/(na*nb);
    println!("vision-embeds: max_abs={max_abs:.4} cos={cos:.6}");
    assert!(cos > 0.999, "cosine {cos} < 0.999 (structural bug)");
    // max_abs threshold: set from this first measurement (bf16 noise floor),
    // documented like the text-only test. Provisional gate: max_abs < 1.0.
    assert!(max_abs < 1.0, "max_abs {max_abs} too large");
}
```

- [ ] **Step 4: Run the parity test**

Run: `source ~/.local/mlx/mlx-env.sh && MINICPMV46_MODEL=<snap> cargo test --release -p ironmlx --test minicpmv46_vision_parity -- --ignored --nocapture`
Expected: `cos > 0.999`; record `max_abs`. If `cos` is low → debug the divergent stage (add per-stage dumps: embeddings out, post-insert, post-merger) against mlx-vlm intermediate captures. Lock the `max_abs` threshold to the observed bf16 floor with a documented first-principles rationale (per [[feedback_first_principles_feasibility_gate]]); do not loosen post-hoc.

- [ ] **Step 5: Commit**

```bash
git add ironmlx/tests/fixtures/minicpmv46_vl/ ironmlx/tests/minicpmv46_vision_parity.rs
git commit -m "test(minicpmv4_6): P1 vision-embeds parity vs mlx-vlm"
```

---

## Final Gate (P1 done)

`source ~/.local/mlx/mlx-env.sh` then:
- `cargo +nightly fmt --all -- --check` ✓
- `cargo +nightly clippy --all-features --workspace -- -D warnings` ✓ (canonical gate)
- `cargo build --release` ✓
- `cargo test --release -p ironmlx --lib` ✓ (no regression; new vision unit tests pass)
- parity test `cos > 0.999` ✓

On green: P1 lands; author the P2 plan (single-image e2e: `MiniCpmV46Model` + cross-modal wiring + single-image no-slice preprocessing + dispatch/CLI/serve), using P1's resolved patch-embed layout.

## Self-Review notes
- Spec coverage: P1 covers spec §3,§4,§5,§6,§8.4 + §9-P1 + §10 (vision-embeds fixture). §7(preprocess) only minimally (fixture from mlx-vlm); §8.1-8.3(MiniCpmV46Model+dispatch) = P2. Out of P1 scope by design.
- Open question §11.1 (Conv vs packed) resolved empirically in Task 2/7.
- §11.3 (`nn::LayerNorm` exists) → confirmed; lift `gelu_tanh` instead (Task 3).
- Reuse-not-replicate: `gelu_tanh`, `LayerNorm`, SDPA pattern, raw-Array loading, `PatchEmbed`-style reshape.
- GELU divergence flagged: encoder = tanh GELU (reuse `gelu_tanh`); VitMerger/Merger = **precise/erf** GELU (Task 4 note).
