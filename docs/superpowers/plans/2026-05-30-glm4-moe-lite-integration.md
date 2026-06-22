# GLM-4.7-Flash (`glm4_moe_lite`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `model_type=glm4_moe_lite` (GLM-4.7-Flash) inference to ironmlx, matching/beating omlx on tok/s and aligning logits with mlx-vlm.

**Architecture:** New `glm4_moe_lite` model module: DeepSeek-style absorbed Multi-head Latent Attention with two SDPA shape regimes (decode L==1: query absorbed to latent, K=V=cached latent; prefill L>1: latent un-folded per-head), a noaux_tc sigmoid router (`e_score_correction_bias`) + ungated shared expert, layer-0 dense FFN + layers 1–46 MoE, interleaved RoPE. Reuses `RoutedExperts` (extended with one public method), `Mlp`, `Linear`, `RmsNorm`, SwiGLU, MLX op bindings; writes fresh `PerHeadQuantLinear`, `MlaAttention`, `MlaLatentCache`, `Glm4MoeBlock`, `Glm4DecoderLayer`, model + config.

**Tech Stack:** Rust, MLX (`mlx` crate safe wrapper), 4-bit affine group_size=64, Metal.

**Companion spec:** `docs/superpowers/specs/2026-05-30-glm4-moe-lite-integration-design.md`.

**Authoritative consumption reference [OBS]:** `/Users/xin/workspace/iron-rivals/omlx/.venv/lib/python3.14/site-packages/mlx_lm/models/glm4_moe_lite.py` + `.../models/mla.py` (`MultiLinear`/`QuantizedMultiLinear`) + `.../models/switch_layers.py` (`SwitchGLU`). This is the de-facto consumer of the mlx-community 4-bit weights and the omlx perf baseline; use it to triangulate exact tensor mechanics (the math spec remains the DeepSeek papers). The MLA forward we mirror is `glm4_moe_lite.py:124-174`; the router is `group_expert_select` `:197-228`.

---

## Verified API reference (use these EXACT signatures)

**Model trait** (`ironmlx/src/core/model.rs:13`):
```rust
pub trait Model {
    fn make_cache(&self, batch: i32, cap: i32, dtype: Dtype) -> Result<Vec<LayerCache>>;
    fn forward_on(&self, input_ids: &Array, position_ids: &Array, per_row_lens: Option<&[i32]>,
                  decode_mask: Option<&Array>, cache: Option<&mut [LayerCache]>, target: StreamOrDevice) -> Result<Array>;
    fn batched_prefill(&self, input_ids: &Array, position_ids: &Array, attention_mask: &Array,
                  linear_attention_mask: &Array, per_row_lens: &[i32], cache: Option<&mut [LayerCache]>,
                  target: StreamOrDevice) -> Result<Array>;
    fn forward_text_hidden(&self, input_ids: &Array, position_ids: &Array, per_row_lens: Option<&[i32]>,
                  decode_mask: Option<&Array>, cache: Option<&mut [LayerCache]>, target: StreamOrDevice) -> Result<Array>;
    fn requires_position_ids(&self) -> bool { true }
    fn model_meta(&self) -> ModelMeta;
    fn num_hidden_layers(&self) -> usize;
}
```
Imports: `use mlx::{Array, Dtype, StreamOrDevice};` `use crate::core::memory_budget::ModelMeta;` `use crate::nn::LayerCache;` `use crate::Result;`

**DenseVlMethods** (`ironmlx/src/core/scheduler.rs:98`): GLM is the FIRST text-only model; no stub exists. Step 1 of Task 1 copies the exact 4 method signatures (`batched_prefill_vl`, `compute_vision_embeds`, `forward_vl_chunk`, `forward_vl_hidden`) from `scheduler.rs:98-146` and returns `Err(anyhow!("Glm4MoeLiteModel is text-only: VL methods unsupported"))` from each.

**MLX bindings (verified real names):**
```rust
mlx::fast::rope_with_array_offset_on(x:&Array, dims:i32, traditional:bool, base:Option<f32>, scale:f32, offset:&Array/*[B] i32*/, freqs:Option<&Array>, target) -> Result<Array>
mlx::fast::rope_on(x:&Array, dims:i32, traditional:bool, base:Option<f32>, scale:f32, offset:i32, freqs:Option<&Array>, target) -> Result<Array>
mlx::fast::scaled_dot_product_attention_on(q:&Array,k:&Array,v:&Array, scale:f32, mask_mode:&str, mask_arr:Option<&Array>, sinks:Option<&Array>, target) -> Result<Array>
mlx::quantization::quantized_matmul_on(x:&Array, w:&Array, scales:&Array, biases:Option<&Array>, transpose:bool, group_size:Option<i32>, bits:Option<i32>, mode:&str, target) -> Result<Array>
mlx::quantization::gather_quantized_matmul_on(x,w,scales, biases:Option<&Array>, lhs_indices:Option<&Array>, rhs_indices:Option<&Array>, transpose:bool, group_size:Option<i32>, bits:Option<i32>, mode:&str, sorted_indices:bool, target) -> Result<Array>
mlx::ops::softmax_on(a,axes,precise:bool,target) ; mlx::ops::sum_on(a,axis,keepdims,target) ; mlx::ops::sort::argpartition_on(a,kth,axis,target)->u32 ; mlx::ops::sort::argsort_on(a,axis,target)
mlx::ops::take_along_axis_on / slice_strided_on / slice_update_on / concatenate_on / expand_dims_on / reshape_on / broadcast_to_on / swapaxes_on / matmul_on / where_on
array.sigmoid_on(target) -> Result<Array>
```

**Reusable ironmlx components:**
```rust
crate::nn::RmsNorm::{from_loader(loader,prefix,eps)->Result<Self>, new(weight,eps), forward_on(x,target)}   // plain weight, NO offset (GLM safe)
crate::nn::Linear::{from_loader(loader,prefix)->Result<Self>, forward_on(x,target)}        // 2-D only; auto quant via {prefix}.scales
crate::nn::Mlp::{from_loader(loader,prefix)->Result<Self>, forward_on(x,target)}           // SwiGLU dense (layer-0 + shared expert)
crate::models::qwen3_5_moe::sparse_moe::RoutedExperts::from_loader(loader,prefix)->Result<Self>  // switch_mlp; extended in Task 5
crate::core::loader::Loader::{tensor(key)->Result<&Array>, tensor_opt(key)->Option<&Array>, quant_meta_for(prefix)->Option<QuantMeta>, config_raw_value()->&serde_json::Value, open(path)->Result<Loader>}
```

**Reference for the per-head quantized projection** (`mla.py` `QuantizedMultiLinear`): weight `[num_heads, out, in_packed]`, `__call__(x, transpose)` = `quantized_matmul(x, w, scales, biases, transpose=transpose, group_size, bits, mode)`. transpose=True → in→out (decode); transpose=False → out→in (prefill un-fold). The single-kv-head input `[B,1,L,512]` broadcasts across the weight's `num_heads=20` automatically via quantized_matmul batch broadcasting → `[B,20,L,...]` (NO manual head-broadcast needed). ironmlx's `Linear` is 2-D only → Task 4a adds `PerHeadQuantLinear`.

**Resolved design decisions (baked in):**
- `requires_position_ids() -> false`; RoPE offset = per-row cache length from `MlaLatentCache::offsets()`; the `position_ids` arg is ignored.
- RoPE = `rope_with_array_offset_on` (per-row offset; tolerates non-uniform cache lengths); B=1 prefill may use scalar `rope_on(offset=0)`.
- MlaAttention dispatches decode vs prefill on query seq len `L` (scheduler keeps phases separate — decode batch is `[B,1]` uniform, prefill `[B,T_max]`, chunked prefill `[1,chunk]`; **no per-row mixed regimes**). Task 6 asserts uniform `per_row_lens` defensively.
- `LayerCache::Mla(MlaLatentCache)` IS added (REJECTING review M6): `Model::make_cache` returns the concrete `Vec<LayerCache>`, so the cache must be a `LayerCache` variant — there is no generic alternative. Boss also chose a dedicated cache type.
- Router math in float32 (sigmoid), `+1e-20` normalization epsilon INCLUDED (matches omlx reference for parity), `routed_scaling_factor` loaded from config (never hardcoded).
- RMSNorm plain-weight (GLM unaffected by Qwen `(1+w)`/MTP offset); mtp.* strip no-op when absent.

**Build/lint gates (CLAUDE.md, before each Rust commit):**
```bash
cargo +nightly fmt --all && cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings
cargo build --release
```
**Test Array API (verified pattern, NOT `Array::from_slice`):** construct via `(&data[..], &shape[..]).try_into().unwrap()`; read via `.to_vec().unwrap()`. Grep `ironmlx/tests/` / `core/cache/kv_cache.rs` tests for the canonical form before writing tests.

---

## Task 1: Module scaffold + config + registry wiring

**Files:** Create `glm4_moe_lite/{mod.rs,config.rs,model.rs}`; Modify `models/mod.rs`, `cli/serve.rs:181`, `bin/ironmlx-core-bench.rs:139`. Test: `config.rs` tests.

- [ ] **Step 1: Read patterns to mirror**

`qwen3_5_moe/config.rs` (serde + `from_loader` via `loader.config_raw_value()`); `scheduler.rs:98-146` (the 4 `DenseVlMethods` signatures); `qwen3_5_moe/model.rs` `model_meta()` (the `ModelMeta` struct literal — copy its field set). Confirm `qwen3_6_moe/model.rs` exists for cross-reference (it does).

- [ ] **Step 2: Write failing config tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    const RAW: &str = r#"{ "model_type":"glm4_moe_lite","hidden_size":2048,"num_hidden_layers":47,
      "first_k_dense_replace":1,"num_attention_heads":20,"num_key_value_heads":20,"q_lora_rank":768,
      "kv_lora_rank":512,"qk_nope_head_dim":192,"qk_rope_head_dim":64,"v_head_dim":256,
      "n_routed_experts":64,"num_experts_per_tok":4,"n_shared_experts":1,"moe_intermediate_size":1536,
      "intermediate_size":10240,"norm_topk_prob":true,"routed_scaling_factor":1.8,"topk_method":"noaux_tc",
      "n_group":1,"topk_group":1,"rope_theta":1000000,"partial_rotary_factor":1.0,"rms_norm_eps":1e-5,
      "vocab_size":154880,"tie_word_embeddings":false,"max_position_embeddings":202752,"rope_scaling":null }"#;
    #[test] fn parses_and_validates() {
        let c = Glm4MoeLiteConfig::from_json_str(RAW).unwrap();
        assert_eq!((c.hidden_size,c.num_hidden_layers,c.first_k_dense_replace),(2048,47,1));
        assert_eq!((c.q_lora_rank,c.kv_lora_rank,c.qk_nope_head_dim,c.qk_rope_head_dim,c.v_head_dim),(768,512,192,64,256));
        assert_eq!(c.q_head_dim(),256); assert_eq!(c.softmax_scale(),1.0/16.0);
        assert_eq!((c.n_routed_experts,c.num_experts_per_tok),(64,4));
        assert!(c.norm_topk_prob); assert_eq!(c.routed_scaling_factor,1.8);
        assert!(c.is_moe_layer(1) && !c.is_moe_layer(0));
    }
    #[test] fn rejects_grouped_routing() { assert!(Glm4MoeLiteConfig::from_json_str(&RAW.replace("\"n_group\":1","\"n_group\":8")).is_err()); }
    #[test] fn rejects_non_noaux_tc()  { assert!(Glm4MoeLiteConfig::from_json_str(&RAW.replace("noaux_tc","greedy")).is_err()); }
    #[test] fn rejects_rope_scaling()  { assert!(Glm4MoeLiteConfig::from_json_str(&RAW.replace("\"rope_scaling\":null","\"rope_scaling\":{\"factor\":2.0}")).is_err()); }
}
```

- [ ] **Step 3: Run → FAIL** `cargo test -p ironmlx glm4_moe_lite::config 2>&1 | tail -20`

- [ ] **Step 4: Implement `config.rs`** (struct with all fields incl. `routed_scaling_factor: f32`, `n_shared_experts`, `moe_intermediate_size`, `intermediate_size`; `from_json_str` + `from_loader` (deserialize `loader.config_raw_value().clone()`); `validate()` rejecting `topk_method!="noaux_tc"`, `n_group!=1||topk_group!=1` with message `"glm4_moe_lite: only n_group=1 and topk_group=1 supported; got n_group={n_group}, topk_group={topk_group}"`, and `rope_scaling.is_some()`; helpers `q_head_dim()`, `softmax_scale()=1/sqrt(q_head_dim)`, `is_moe_layer(i)= i>=first_k_dense_replace`). Comment on `from_loader`: `// GLM uses a FLAT top-level config (unlike Qwen's config[text_config]) — deserialize the whole value`.

- [ ] **Step 5: Run → PASS** (4 tests).

- [ ] **Step 6: `mod.rs` + model skeleton + registry**

`mod.rs`: `pub mod config; pub mod model; pub use config::Glm4MoeLiteConfig; pub use model::Glm4MoeLiteModel;`

`model.rs` skeleton: struct holding `cfg`; `from_loader` parses config; `impl Model` with all 5 methods returning `Err(anyhow!("...not yet implemented (Task 6)"))` except `requires_position_ids()->false`, `num_hidden_layers()->cfg.num_hidden_layers as usize`, and `model_meta()` returning the real `ModelMeta` literal (copy field set from `qwen3_5_moe/model.rs`; `head_dim: Some(256)` per v_head_dim, `num_key_value_heads: 20`). `impl crate::core::scheduler::DenseVlMethods` with the 4 signatures copied verbatim from `scheduler.rs:98-146`, each body `Err(anyhow!("Glm4MoeLiteModel is text-only: VL methods unsupported"))`, with required `#[allow(clippy::too_many_arguments)]`.

`models/mod.rs`: `pub mod glm4_moe_lite; pub use glm4_moe_lite::Glm4MoeLiteModel;`

`serve.rs` (before `other =>` ~`:204`):
```rust
        "glm4_moe_lite" => {
            let model = crate::models::Glm4MoeLiteModel::from_loader(&loader).context("Glm4MoeLiteModel::from_loader")?;
            serve_with_model(model, tokenizer, &args, None)
        }
```
`ironmlx-core-bench.rs` (before `other =>` ~`:161`):
```rust
        "glm4_moe_lite" => { let model = crate::models::Glm4MoeLiteModel::from_loader(&loader)?; run_for_model(&model, &tokenizer, &args, load_ms) }
```

- [ ] **Step 7: Build → compiles.** `cargo build -p ironmlx 2>&1 | tail -10`

- [ ] **Step 8: Commit** `git commit -m "feat(glm4_moe_lite): config + module scaffold + registry wiring"`

---

## Task 2: `Glm4Rope` — interleaved RoPE wrapper

**Files:** Create `glm4_moe_lite/rope.rs`; Modify `mod.rs`. Test: same file.

- [ ] **Step 1: Failing test** (uses a SMALL test base for hand-computability; the real model passes `cfg.rope_theta=1e6` — assert `traditional=true` interleaved, per-row offset). Build arrays via `(&data[..], &shape[..]).try_into().unwrap()`. Compute expected from `θ=p·base^(-2i/dims)`, pair `(x0,x1)→(x0cosθ−x1sinθ, x0sinθ+x1cosθ)`. Two tests: (a) `interleaved_matches_hand_computed` (dims=4, base=10000, pos=1, x=[1,0,0,1]) tol 1e-4; (b) `per_row_offset_differs` (offsets [0,3] → row0 identity at pe pair0, row1 rotated).

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement `rope.rs`**
```rust
use anyhow::Result;
use mlx::{Array, StreamOrDevice};
pub struct Glm4Rope { dims: i32, base: f32 }
impl Glm4Rope {
    pub fn new(dims: i32, base: f32) -> Self { Self { dims, base } }
    /// x: [B,H,S,dims]; offset: [B] i32 per-row start position. Interleaved (traditional=true).
    pub fn apply(&self, x: &Array, offset: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
        mlx::fast::rope_with_array_offset_on(x, self.dims, true, Some(self.base), 1.0, offset, None, target)
    }
    pub fn apply_scalar(&self, x: &Array, offset: i32, target: impl Into<StreamOrDevice>) -> Result<Array> {
        mlx::fast::rope_on(x, self.dims, true, Some(self.base), 1.0, offset, None, target)
    }
}
```

- [ ] **Step 4: Run → PASS** (use the project's real `Array` ctor/`to_vec` if names differ — grep `nn/mrope.rs` tests).

- [ ] **Step 5: Commit** `git commit -m "feat(glm4_moe_lite): interleaved RoPE wrapper"`

---

## Task 3: `MlaLatentCache` + `LayerCache::Mla`

**Files:** Create `glm4_moe_lite/mla_cache.rs`; Modify `nn/decoder_layer.rs:65` (+ reset arm at `:70`); Modify `mod.rs`. Test: same file.

- [ ] **Step 1: Read `KVCache::update_and_fetch_on`** (`core/cache/kv_cache.rs:111-231`) fully — copy its lazy-alloc + step-grow-via-concatenate + per-row `slice_update_on` loop + fetch-`[0..max_offset]` + bounds-check structure. `MlaLatentCache` applies the SAME structure to TWO buffers of differing last-dim width.

- [ ] **Step 2: Failing test** (`append_and_fetch_two_steps`: batch1, kv_lora512, rope64, cap8, step8; append 2 tokens then 1; assert fetched shapes `[1,1,2/3,512]`+`[1,1,2/3,64]` and `offsets()==[2]` then `[3]`. `rejects_wrong_per_row_lens_len`).

- [ ] **Step 3: Run → FAIL.**

- [ ] **Step 4: Implement `mla_cache.rs`** — full body (NOT a skeleton):
```rust
//! Latent MQA cache for GLM absorbed-MLA: normalized c_kv[kv_lora] + post-rope k_pe[rope], single kv head.
//! Mirrors KVCache (two buffers, differing widths). step default 256 + with_step(cap) one-shot prealloc
//! avoids the cache-update Metal slow path (KV-floor is about cap/step, not head-dim width).
use anyhow::{anyhow, Result};
use mlx::{Array, Dtype, StreamOrDevice};
use mlx::ops::shape::concatenate_on;
use mlx::ops::indexing::{slice_strided_on, slice_update_on};

pub struct MlaLatentCache { c_kv: Option<Array>, k_pe: Option<Array>, offsets: Vec<i32>,
    cap: i32, step: i32, batch: i32, kv_lora: i32, rope: i32, dtype: Dtype }

impl MlaLatentCache {
    pub fn new(batch:i32, kv_lora:i32, rope:i32, dtype:Dtype, cap:i32) -> Self {
        Self{ c_kv:None, k_pe:None, offsets:vec![0;batch as usize], cap, step:256, batch, kv_lora, rope, dtype } }
    pub fn with_step(mut self, step:i32)->Self{ assert!(step>0,"step must be positive (got {step})"); self.step=step; self }
    pub fn offsets(&self)->&[i32]{ &self.offsets }
    pub fn cap(&self)->i32{ self.cap }
    pub fn dtype(&self)->Dtype{ self.dtype }
    pub fn grow_cap(&mut self, new_cap:i32){ if new_cap>self.cap { self.cap=new_cap; } }   // mirror KVCache::grow_cap
    pub fn reset(&mut self)->Result<()>{ self.c_kv=None; self.k_pe=None; for o in &mut self.offsets {*o=0;} Ok(()) }

    /// c_kv_new [B,1,S,kv_lora], k_pe_new [B,1,S,rope]; returns full-history [B,1,L,*].
    pub fn update_and_fetch_on(&mut self, c_kv_new:&Array, k_pe_new:&Array, per_row_lens:&[i32],
                               target: impl Into<StreamOrDevice>) -> Result<(Array,Array)> {
        let target = target.into();
        if per_row_lens.len()!=self.batch as usize { return Err(anyhow!("MlaLatentCache: per_row_lens.len()={} != batch={}", per_row_lens.len(), self.batch)); }
        for (i,&l) in per_row_lens.iter().enumerate() {
            if self.offsets[i]+l > self.cap { return Err(anyhow!("MlaLatentCache: row {i} offset {}+{l} exceeds cap {}", self.offsets[i], self.cap)); }
        }
        // Replicate KVCache::update_and_fetch_on EXACTLY for BOTH buffers, with widths (kv_lora, rope):
        //  - lazy alloc / step-grow each buffer to [batch,1,grown_cap,width] via concatenate_on of zeros
        //  - per-row: slice_update_on(buffer, src_row, [i,0,offsets[i],0]..) for rows with per_row_lens[i]>0
        //  - advance offsets[i] += per_row_lens[i]
        //  - fetch: slice_strided_on(buffer, [..,:max(offsets),:]) for both buffers
        // (copy the Step-1 KVCache body line-for-line, duplicated for c_kv + k_pe)
        unimplemented!("REPLACE: paste KVCache::update_and_fetch_on body, applied to self.c_kv (width kv_lora) and self.k_pe (width rope)")
    }
}
```
> The `unimplemented!` is a directed copy of a SPECIFIC verified function (KVCache::update_and_fetch_on, read in Step 1) duplicated across two buffers — replace it with that body before running the test.

`nn/decoder_layer.rs:65`: add `Mla(crate::models::glm4_moe_lite::mla_cache::MlaLatentCache),` to `enum LayerCache`. In `impl LayerCache::reset(&mut self) -> anyhow::Result<()>` (`:70`) add arm `LayerCache::Mla(c) => c.reset(),`.

- [ ] **Step 5: Run → PASS** (2 tests).

- [ ] **Step 6: Build + lint + commit** `git commit -m "feat(glm4_moe_lite): MlaLatentCache + LayerCache::Mla variant"`

---

## Task 4: `MlaAttention` — two-regime absorbed-MLA forward

> Mirrors `glm4_moe_lite.py:124-174`. 4a = per-head quant linear + shared prefix + cache; 4b = two regimes + mask + o_proj. Key gate: decode and prefill must agree on a single token.

**Files:** Create `glm4_moe_lite/mla_attention.rs`; Modify `mod.rs`. Test: same file.

### Task 4a: `PerHeadQuantLinear` + projections + RoPE + cache write

- [ ] **Step 1: Failing test for `PerHeadQuantLinear`** — synthetic: quantize a known `[H=2,out=3,in=8]` fp weight via the project's quantize op, build `PerHeadQuantLinear`, apply `transpose=true` to `x[1,2,1,8]` → `[1,2,1,3]`, compare to a dequantized reference matmul (tol 1e-2 for 4-bit). Add `transpose=false` on `x[1,2,1,3]` → `[1,2,1,8]`. Add a broadcast test: `x[1,1,1,8]` with `H=2` weight → `[1,2,1,3]` (single input-head broadcasts across 2 weight-heads).

- [ ] **Step 2: Implement `PerHeadQuantLinear`** (mirrors `mla.py` `QuantizedMultiLinear`):
```rust
pub struct PerHeadQuantLinear { weight: Array, scales: Array, biases: Option<Array>, group_size: i32, bits: i32 }
impl PerHeadQuantLinear {
    /// Loads {prefix}.weight [H,out,in/8] + .scales + optional .biases. group_size/bits from quant_meta.
    pub fn from_loader(loader:&crate::core::Loader, prefix:&str) -> anyhow::Result<Self> {
        let qm = loader.quant_meta_for(prefix).ok_or_else(|| anyhow::anyhow!("{prefix}: expected quantized per-head weight"))?;
        Ok(Self{ weight: loader.tensor(&format!("{prefix}.weight"))?.clone(),
                 scales: loader.tensor(&format!("{prefix}.scales"))?.clone(),
                 biases: loader.tensor_opt(&format!("{prefix}.biases")).cloned(),
                 group_size: qm.group_size, bits: qm.bits })   // adapt field names to QuantMeta
    }
    /// transpose=true: in->out (decode). transpose=false: out->in (prefill un-fold).
    /// Single-kv-head x [B,1,L,*] broadcasts across the weight's H heads automatically.
    pub fn apply(&self, x:&Array, transpose:bool, target: impl Into<mlx::StreamOrDevice>) -> anyhow::Result<Array> {
        mlx::quantization::quantized_matmul_on(x, &self.weight, &self.scales, self.biases.as_ref(),
            transpose, Some(self.group_size), Some(self.bits), "affine", target)
    }
}
```
> If the checkpoint's `embed_q`/`unembed_out` are NOT quantized (no `.scales`), fall back to a plain per-head `matmul_on` variant. Probe in Step 1.

- [ ] **Step 3: `MlaAttention::from_loader` + `project_qkv`**

Struct fields: `q_a_proj/q_b_proj/kv_a_proj_with_mqa/o_proj: Linear`, `q_a_layernorm/kv_a_layernorm: RmsNorm`, `embed_q/unembed_out: PerHeadQuantLinear`, `rope: Glm4Rope`, dims (`n_heads,qk_nope,qk_rope,kv_lora,v_head`), `scale: f32`. `from_loader(loader,prefix,cfg)` loads each at `{prefix}.{name}`; `rope=Glm4Rope::new(cfg.qk_rope_head_dim, cfg.rope_theta)`; `scale=cfg.softmax_scale()`.

`project_qkv(&self, x, offset, target) -> Result<(Array,Array,Array,Array)>` returns `(q_nope[B,H,S,192], q_pe[B,H,S,64] rope'd, c_kv_n[B,1,S,512] NORMALIZED, k_pe[B,1,S,64] rope'd)`, mirroring `glm4_moe_lite.py:132-148`:
1. `q = q_b_proj(q_a_layernorm(q_a_proj(x)))`; reshape `[B,S,H,256]`; transpose `[B,H,S,256]`; split `q_nope=[..,:192]`,`q_pe=[..,192:]`.
2. `kv = kv_a_proj_with_mqa(x)`; split `c_kv=[..,:512]`,`k_pe=[..,512:]`; `c_kv_n = kv_a_layernorm(c_kv)`; reshape c_kv_n `[B,1,S,512]`, k_pe `[B,1,S,64]`.
3. `q_pe = rope.apply(q_pe, offset)`; `k_pe = rope.apply(k_pe, offset)`.

Add `#[cfg(test)] from_parts(...)` ctor (tiny dims H=2,nope=4,rope=2,kv_lora=6,v=4) for Task 4b. Add a shared-prefix shape test.

- [ ] **Step 4: Run + commit** `git commit -m "feat(glm4_moe_lite): per-head quant linear + MLA shared prefix"`

### Task 4b: Two regimes + mask + o_proj

- [ ] **Step 1: Decode==prefill equivalence test** (the correctness gate): build `from_parts` MlaAttention; fill a cache with L=3 tokens; take ONE query token; run it through the DECODE regime and (separately) the PREFILL regime; assert `max|decode_out - prefill_out| < 1e-4`. (Same scale, same pe-mask; the two regimes are algebraically identical attention.) Plus a `decode_respects_causality` test (no future leakage).

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement `forward_on`** (mirrors `glm4_moe_lite.py:124-174`):
```text
forward_on(&self, x, offset:&Array, cache:&mut MlaLatentCache, per_row_lens:&[i32], mask:Option<&Array>, target) -> Result<Array>
  L = x.shape()[1]
  (q_nope, q_pe, c_kv_n, k_pe) = project_qkv(x, offset, target)
  (kv_latent, k_pe_all) = cache.update_and_fetch_on(&c_kv_n, &k_pe, per_row_lens, target)   // [B,1,Lc,512],[B,1,Lc,64]
  pe_scores = matmul_on(&(q_pe * scale), &swapaxes_on(&k_pe_all,-1,-2,target), target)        // [B,H,L,Lc]
  if let Some(m) = mask { pe_scores = apply_mask(&pe_scores, m, target) }                     // see note
  if L == 1 {                                  // DECODE
      let q_lat = self.embed_q.apply(&q_nope, /*transpose=*/true, target)?;                   // [B,H,1,512]
      let o = sdpa(&q_lat, &kv_latent, &kv_latent, scale, "array", Some(&pe_scores))?;          // [B,H,1,512]
      out = self.unembed_out.apply(&o, true, target)?;                                          // [B,H,1,256]
  } else {                                     // PREFILL
      let k = self.embed_q.apply(&kv_latent, /*transpose=*/false, target)?;                    // [B,H,Lc,192]
      let v = self.unembed_out.apply(&kv_latent, true, target)?;                               // [B,H,Lc,256]
      out = sdpa(&q_nope, &k, &v, scale, "array", Some(&pe_scores))?;                           // [B,H,L,256]
  }
  out = reshape_on(&out.transpose([0,2,1,3]), [B,L, H*256])
  self.o_proj.forward_on(&out, target)
```
- **`sdpa` call:** `mlx::fast::scaled_dot_product_attention_on(q, k, v, scale, "array", Some(&pe_scores), None, target)` — `mask_mode="array"` because we pass an additive float mask; SDPA computes `softmax(scale·q@kᵀ + pe_scores)@v`. (pe_scores already includes `*scale` on the rope term.)
- **`apply_mask` note (B3/B4/M5):** GLM uses the engine-provided mask (`attention_mask` in batched_prefill, `decode_mask` in forward_on) — do NOT build a fresh causal mask. Mirror how `qwen3_5_moe` full attention consumes the mask: if it is a BOOLEAN mask, fold into pe_scores via `where_on(mask, pe_scores, dtype_min)` (as `glm4_moe_lite.py:154-159`); if it is already an ADDITIVE float mask, add it. Determine which by reading how `GatedAttention`/the scheduler builds it (grep `create_attention_mask`/`build_batch_attention_mask`). For decode (L==1) the engine typically passes no causal mask (single token sees all valid cache) → `mask=None`; pass `pe_scores` directly. The engine's mask already encodes the `Lc>L` offset (lower-right alignment) and per-row padding — reuse it, don't recompute.

- [ ] **Step 4: Run → PASS** (equivalence < 1e-4 + causality).

- [ ] **Step 5: Build + clippy + commit** `git commit -m "feat(glm4_moe_lite): two-regime absorbed-MLA attention"`

---

## Task 5: `Glm4MoeBlock` — noaux_tc router + ungated shared + RoutedExperts combine

**Files:** Create `glm4_moe_lite/moe.rs`; Modify `qwen3_5_moe/sparse_moe.rs` (add ONE public method to `RoutedExperts`); Modify `mod.rs`. Test: `moe.rs`.

- [ ] **Step 1: Read `SparseMoeBlock::forward_on`** (`sparse_moe.rs:508`) + the SwitchGLU-style expert path; extract the gather_qmm sorted/broadcast dispatch (threshold `SORTED_ROUTING_MIN_BS_K=64`, argsort reorder, `gather_quantized_matmul_on(...,lhs_indices=idx,sorted_indices=...)`, fused gate/up, `invoke_swiglu`, down_proj, weighted-sum). This becomes the new public method.

- [ ] **Step 2: Add `RoutedExperts::apply_experts`** to `sparse_moe.rs` (extract existing logic; `pub`):
```rust
/// SwitchGLU-style: route flat tokens x[BS,H] through top-k experts `inds`[BS,k] (u32),
/// combine by `weights`[BS,k], return [BS,H]. Mirrors mlx_lm SwitchGLU + the existing
/// sorted/broadcast dispatch in SparseMoeBlock.
pub fn apply_experts(&self, x:&Array, inds:&Array, weights:&Array, target:StreamOrDevice) -> Result<Array> { /* extracted */ }
```

- [ ] **Step 3: Failing router test** (`noaux_tc_route`):
```rust
#[test] fn router_selects_with_bias_weights_from_raw_sigmoid() {
    // 1 token, 4 experts, k=2, norm=true, scale=1.8. logits=[0,0,0,0]->s=0.5 each.
    // bias=[-9,9,9,-9] -> selection picks {1,2}; weights from RAW s -> [0.5,0.5] norm->[0.5,0.5] *1.8 -> [0.9,0.9].
    let (idx,w) = noaux_tc_route(&arr(&[0.,0.,0.,0.],&[1,4]), &arr(&[-9.,9.,9.,-9.],&[4]), 2, true, 1.8, t).unwrap();
    let mut iv:Vec<u32>=idx.to_vec().unwrap(); iv.sort(); assert_eq!(iv, vec![1,2]);
    for x in w.to_vec().unwrap() { assert!((x-0.9).abs()<1e-5, "got {x}"); }
}
```

- [ ] **Step 4: Run → FAIL.**

- [ ] **Step 5: Implement `noaux_tc_route` + `Glm4MoeBlock`** (mirrors `group_expert_select` + `Glm4MoeLiteMoE`):
```rust
// router (float32 math, +1e-20 included for omlx parity)
fn noaux_tc_route(logits:&Array, bias:&Array, k:i32, norm_topk_prob:bool, scale:f32, target:StreamOrDevice) -> Result<(Array,Array)> {
    let s = logits.astype(Dtype::Float32).sigmoid_on(target)?;                 // [BS,E]
    let s_choice = &s + bias;                                                  // bias [E] broadcasts
    let e = s.shape()[s.ndim()-1];
    let part = argpartition_on(&s_choice, e-k, -1, target)?;                   // u32
    let idx = slice_strided_on(&part, /*last axis [e-k .. e]*/ ...)?;          // [BS,k]
    let mut w = take_along_axis_on(&s, &idx, -1, target)?;                     // RAW sigmoid (NOT s_choice)
    if k>1 && norm_topk_prob { w = &w / &(sum_on(&w,-1,true,target)? + 1e-20); }
    w = &w * scale;
    Ok((idx, w))
}
```
`Glm4MoeBlock::from_loader(loader,prefix,cfg)`: `gate = Linear::from_loader("{prefix}.gate")` (plain), `bias = loader.tensor("{prefix}.gate.e_score_correction_bias")?.clone()`, `experts = RoutedExperts::from_loader("{prefix}.switch_mlp")`, `shared = Mlp::from_loader("{prefix}.shared_experts")`, store `k=cfg.num_experts_per_tok`, `norm=cfg.norm_topk_prob`, `scale=cfg.routed_scaling_factor` (from config — NOT hardcoded). **Do NOT load `shared_expert_gate`.**
`forward_on(x, target, layer_idx)`:
```text
flat = reshape x [B,S,H] -> [BS,H]
logits = gate.forward_on(flat)                         // [BS,64] plain float
(idx, w) = noaux_tc_route(&logits, &self.bias, self.k, self.norm, self.scale, target)
routed = self.experts.apply_experts(&flat, &idx, &w, target)   // [BS,H]
shared = self.shared.forward_on(&flat, target)                 // UNGATED
reshape (&routed + &shared) back to [B,S,H]
```

- [ ] **Step 6: Full-block numeric test** (tiny weights vs hand-rolled reference), run → PASS.

- [ ] **Step 7: Build + clippy + commit** `git commit -m "feat(glm4_moe_lite): noaux_tc router + ungated shared MoE block + RoutedExperts::apply_experts"`

---

## Task 6: `Glm4DecoderLayer` + `Glm4MoeLiteModel` assembly

**Files:** Create `glm4_moe_lite/decoder_layer.rs`; complete `glm4_moe_lite/model.rs`; Modify `mod.rs`. Test: `ironmlx/tests/glm4_moe_lite_smoke.rs`.

- [ ] **Step 1: `Glm4DecoderLayer`** — fields `input_layernorm/post_attention_layernorm: RmsNorm`, `attn: MlaAttention`, `ffn: Ffn` where `enum Ffn { Dense(Mlp), Moe(Glm4MoeBlock) }`. `from_loader(loader,layer_idx,cfg)`: `ffn = if cfg.is_moe_layer(layer_idx) { Moe(Glm4MoeBlock::from_loader("{p}.mlp",cfg)) } else { Dense(Mlp::from_loader("{p}.mlp")) }`. `forward_on(x, offset, cache:&mut MlaLatentCache, per_row_lens, mask:Option<&Array>, target, layer_idx)`:
```text
h = x + attn.forward_on(input_layernorm(x), offset, cache, per_row_lens, mask, target)
out = h + match ffn { Dense(m)=>m.forward_on(post_attention_layernorm(h),target), Moe(b)=>b.forward_on(post_attention_layernorm(h),target,layer_idx) }
```

- [ ] **Step 2: Complete `Glm4MoeLiteModel`** — fields `embed_tokens: Embedding`, `layers: Vec<Glm4DecoderLayer>`, `norm: RmsNorm`, `lm_head: Linear` (separate; `tie_word_embeddings=false`), `cfg`. Methods:
- `make_cache(batch,cap,dtype)`: `Ok((0..n).map(|_| LayerCache::Mla(MlaLatentCache::new(batch, cfg.kv_lora_rank, cfg.qk_rope_head_dim, dtype, cap).with_step(cap))).collect())`.
- shared `run_layers(input_ids, per_row_lens, mask, cache, target) -> Result<Array>` (hidden states):
```text
let batch = input_ids.shape()[0]; let seq_len = input_ids.shape()[1];
let caches = cache.ok_or_else(|| anyhow!("glm4_moe_lite requires a cache"))?;
let prl: Vec<i32> = per_row_lens.map(|s| s.to_vec()).unwrap_or_else(|| vec![seq_len; batch as usize]);
// regime uniformity (scheduler guarantees this; assert defensively — REJECTS mixed prefill/decode)
if !prl.iter().all(|&l| l == prl[0]) { return Err(anyhow!("glm4_moe_lite: non-uniform per_row_lens {:?} (mixed prefill/decode in one forward unsupported)", prl)); }
// rope offset = pre-update per-row cache length (uniform across layers); read layer 0
let offsets_vec = match &caches[0] { LayerCache::Mla(c) => c.offsets().to_vec(), _ => return Err(anyhow!("glm4_moe_lite: expected LayerCache::Mla")) };
let offset = Array::try_from((&offsets_vec[..], &[batch][..]))?;            // i32 [B]
let mut h = self.embed_tokens.forward_on(input_ids, target)?;
for (i, layer) in self.layers.iter().enumerate() {
    let LayerCache::Mla(c) = &mut caches[i] else { return Err(anyhow!("glm4_moe_lite: expected LayerCache::Mla at layer {i}")) };
    h = layer.forward_on(&h, &offset, c, &prl, mask, target, i as i32)?;
}
self.norm.forward_on(&h, target)
```
- `forward_on(input_ids, _position_ids, per_row_lens, decode_mask, cache, target)`: `let h = self.run_layers(input_ids, per_row_lens, decode_mask, cache, target)?; self.lm_head.forward_on(&h, target)`. (decode_mask is the engine's decode mask; GLM passes it through to attention; GLM does NOT use `linear_attention_mask`.)
- `batched_prefill(input_ids, _pos, attention_mask, _linear, per_row_lens, cache, target)`: `let h = self.run_layers(input_ids, Some(per_row_lens), Some(attention_mask), cache, target)?; self.lm_head.forward_on(&h, target)`.
- `forward_text_hidden(...)`: `run_layers(...)` returning `h` (no lm_head).
- `requires_position_ids()->false`, `num_hidden_layers()`, `model_meta()` (real `ModelMeta` literal).

Document at the top of model.rs: GLM ignores `position_ids` (RoPE offset comes from cache) and `linear_attention_mask` (no linear attention); regime is per-call-uniform (asserted).

- [ ] **Step 3: Integration test** `ironmlx/tests/glm4_moe_lite_smoke.rs` (env-gated; skip if no weights). Define helpers inline; use `Loader::open(Path::new(&dir))`:
```rust
fn arr(d:&[f32], s:&[i32]) -> Array { (d, s).try_into().unwrap() }
fn ids(d:&[i32], s:&[i32]) -> Array { (d, s).try_into().unwrap() }
#[test] fn glm_loads_and_prefill_is_finite() {
    let Some(dir) = glm_snapshot_dir() else { eprintln!("skip: no GLM weights"); return; };
    let loader = Loader::open(Path::new(&dir)).unwrap();
    let model = Glm4MoeLiteModel::from_loader(&loader).unwrap();
    let mut cache = model.make_cache(1, 16, Dtype::Bfloat16).unwrap();
    let input = ids(&[1,2,3,4], &[1,4]);
    let pos = ids(&[0], &[1]);                          // dummy (requires_position_ids=false)
    let mask = build_causal_mask(4);                    // use the project's mask builder
    let logits = model.batched_prefill(&input, &pos, &mask, &mask, &[4], Some(&mut cache), StreamOrDevice::default()).unwrap();
    assert_eq!(logits.shape().last(), Some(&154880));
    assert!(logits.astype(Dtype::Float32).to_vec::<f32>().unwrap().iter().all(|x| x.is_finite()));
}
```
> Use the project's real causal-mask builder (grep `qwen3_5_moe` integration tests). Confirm `Array::try_from((&[i32], &[i32]))` is the real ctor; adapt if different.

- [ ] **Step 4: Run with weights present:**
`GLM47_MODEL_DIR=$(echo ~/.ironmlx/models/models--mlx-community--GLM-4.7-Flash-4bit/snapshots/*) cargo test -p ironmlx glm_loads_and_prefill 2>&1 | tail -30` → PASS (finite, last dim 154880).

- [ ] **Step 5: Full build + lint + commit**
```bash
cargo +nightly fmt --all && cargo +nightly clippy --all-features --workspace -- -D warnings 2>&1 | tail -10 && cargo build --release 2>&1 | tail -5
git commit -m "feat(glm4_moe_lite): decoder layer + model assembly; loads + forward runs"
```

---

## Task 7: Correctness vs mlx-vlm + perf vs omlx + feasibility gate

**Files:** Create `ironmlx/tests/glm4_moe_lite_parity.rs`. Perf numbers → working report (reports/, gitignored).

- [ ] **Step 1: Hard-assert config constants vs runtime config.json** (B7):
`python3 -c "import json;c=json.load(open('$GLM_DIR/config.json'));assert c['routed_scaling_factor']==1.8 and c['topk_method']=='noaux_tc' and c.get('rope_scaling') is None, c"` → must pass loudly. If `rope_scaling` non-null → STOP, reopen design (1/16 scale needs mscale, spec §10.8).

- [ ] **Step 2: Correctness — logits parity vs mlx-vlm.** Reference logits from mlx-vlm (`/Users/xin/workspace/iron-rivals/mlx-vlm`, `uv run --with-editable`) for a fixed prompt. Assert: top-5 token IDs match exactly; max abs logit diff on top-50 < tol (start 0.5 for 4-bit); 32-token greedy continuation identical. On mismatch use systematic-debugging (likely suspects: RoPE interleave, router weight-vs-bias, prefill un-fold transpose, mask polarity).

- [ ] **Step 3: Perf — omlx baseline then iron-bench** (serial, one server at a time):
confirm omlx runs glm4_moe_lite (`cd iron-rivals/omlx`, run its CLI smoke); then `iron-bench` omlx baseline + ironmlx at PP=512. **Acceptance: ironmlx prefill+decode tok/s ≥ omlx.** If omlx can't run it → record + fall back to Boss-agreed target (spec §9 precondition).

- [ ] **Step 4: If perf FAILS — feasibility gate BEFORE kernel work** (spec §9): measure the up-project-to-256-MHA lever's Amdahl ceiling at PP=128 AND PP=512; only if dual-point projected e2e ≥5% AND MLX-saturation pre-screen passes, open a kernel task. Do NOT implement on intuition (Stage β precedent). Record the decision.

- [ ] **Step 5: Commit the parity harness only** `git commit -m "test(glm4_moe_lite): logits parity harness vs mlx-vlm"`. Perf numbers + gate decision → gitignored report or close-out commit body.

---

## Self-Review (plan author) — post-adversarial-review revision

This revision applied all 8 BLOCKERs + MAJORs from the plan-review workflow:
- **B1/B2/M9** (per-head quant embed_q/unembed_out + head-broadcast + transpose direction): RESOLVED via `PerHeadQuantLinear` (Task 4a) mirroring omlx `QuantizedMultiLinear`; transpose=true (decode in→out) / false (prefill out→in) is the omlx production path; head-broadcast is automatic via quantized_matmul batch broadcasting.
- **B3/B4/M5** (causal mask + SDPA mode): GLM consumes the engine-provided mask folded into pe_scores; SDPA called with `mask_mode="array"`; decode_mask/attention_mask sourced from the trait args (Task 4b note, Task 6).
- **B5** (cache body): full `MlaLatentCache::update_and_fetch_on` directed to copy the verified `KVCache` body across two buffers (Task 3).
- **B6** (expert combine): add `RoutedExperts::apply_experts` public method (Task 5).
- **B7** (routed_scaling): in config + loaded into block + hard-asserted (Task 5 + 7.1).
- **B8** (regime uniformity): asserted in `run_layers` (Task 6).
- **M1/M10** (test API + helpers): `(&data,&shape).try_into()` / `.to_vec().unwrap()` / `Loader::open` / inline helpers.
- **M2/M4** (DenseVlMethods + ModelMeta): exact signatures from scheduler.rs:98 + real ModelMeta literal.
- **M3/m5** (router op names + index assert): verified `_on` names; assert `idx==[1,2]`.
- **M6** REJECTED with reasoning: `Model::make_cache->Vec<LayerCache>` forces a `LayerCache::Mla` variant; no generic alternative + Boss chose a dedicated cache type.
- **M7/M8** (equivalence test + c_kv_n): fleshed-out decode==prefill gate + `from_parts` ctor; `project_qkv` returns normalized c_kv_n.
- **M11** (spec cliff rationale): corrected in spec §3.
- **Spec coverage:** §1→T1; §3→T3; §4→T4; §5→T2; §6→T5; §7→T1+T6; §8→T7.2; §9→T7.3/7.4; §10 open items resolved (regime/RMSNorm/mtp/position_ids/rope-offset) or carried (routed_scaling 7.1, YaRN 7.1, +1e-20 included, 256-MHA gate 7.4, grouped-routing rejected in config).
- **Remaining directed-mirrors (intentional, point at one verified function):** MlaLatentCache body→KVCache::update_and_fetch_on; apply_experts→SparseMoeBlock expert path; ModelMeta/DenseVlMethods→named source files. All novel logic (PerHeadQuantLinear, two-regime forward, router, rope) has complete code.
