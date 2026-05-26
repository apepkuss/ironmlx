# P6 MoE-VL Research

| Field | Value |
| --- | --- |
| Date | 2026-05-26 |
| Branch | `ironmlx-p6-moe-vl` |
| Worktree | `/Users/xin/workspace/ironmlx-backend-moe-vl` |
| Scope | Qwen3.5 MoE VL feasibility, reuse points, risks, and remaining decisions |

## Summary

Qwen3.5 MoE VL support is feasible in a separate worktree, but it is not a small
runtime flag. The current `Qwen35MoeModel` deliberately implements MoE as
text-only and satisfies `DenseVlMethods` through panic stubs. The implementation
work is therefore to replace those stubs with a real multimodal path and extend
MoE config/model metadata to carry `vision_config`.

The high-confidence approach is to reuse the existing dense VL pipeline shape:

1. Load `vision_tower.*` through `Loader::open_multimodal`.
2. Compute vision embeddings with the shared `models::vision::VisionTower` implementation.
3. Embed MoE text tokens through `Qwen35MoeTextModel::embed_on`.
4. Scatter vision embeddings via `qwen3_5::cross_modal::replace_image_tokens`.
5. Continue through `Qwen35MoeTextModel::forward_post_embedding_on`.
6. Slice/project with MoE's untied `lm_head`.

## Local Checkpoint Facts

Local MoE snapshot:

`~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec`

Observed config:

| Field | MoE 35B-A3B | Dense 4B |
| --- | ---: | ---: |
| `model_type` | `qwen3_5_moe` | `qwen3_5` |
| `architectures[0]` | `Qwen3_5MoeForConditionalGeneration` | `Qwen3_5ForConditionalGeneration` |
| text hidden | 2048 | 2560 |
| text layers | 40 | 32 |
| vocab | 248320 | 248320 |
| vision depth | 27 | 24 |
| vision hidden | 1152 | 1024 |
| vision heads | 16 | 16 |
| vision intermediate | 4304 | 4096 |
| vision out hidden | 2048 | 2560 |
| vision position embeddings | 2304 | 2304 |
| vision weight keys | 333 | 297 |

Observed tensor metadata from safetensors headers:

| Tensor | MoE shape/dtype | Dense shape/dtype |
| --- | --- | --- |
| `vision_tower.patch_embed.proj.weight` | `[1152, 2, 16, 16, 3] BF16` | `[1024, 2, 16, 16, 3] BF16` |
| `vision_tower.pos_embed.weight` | `[2304, 1152] BF16` | `[2304, 1024] BF16` |
| `vision_tower.blocks.0.attn.qkv.weight` | `[3456, 1152] BF16` | `[3072, 1024] BF16` |
| `vision_tower.blocks.0.mlp.linear_fc1.weight` | `[4304, 1152] BF16` | `[4096, 1024] BF16` |
| `vision_tower.merger.linear_fc1.weight` | `[4608, 4608] BF16` | `[4096, 4096] BF16` |
| `vision_tower.merger.linear_fc2.weight` | `[2048, 4608] BF16` | `[2560, 4096] BF16` |
| `language_model.model.embed_tokens.weight` | `[248320, 256] U32` | `[248320, 320] U32` |
| `language_model.lm_head.weight` | `[248320, 256] U32` | missing, tied embeddings |

Special token ids match between MoE and dense:

| Token | Id |
| --- | ---: |
| `<|vision_start|>` | 248053 |
| `<|vision_end|>` | 248054 |
| `<|vision_pad|>` | 248055 |
| `<|image_pad|>` | 248056 |
| `<|video_pad|>` | 248057 |

## Code Findings

### Existing reusable pieces

- `Loader::open_multimodal` already retains `vision_tower.*` keys and strips
  `language_model.` from text weights.
- Dense `Qwen35Model` already owns `vision: Option<VisionTower>` and implements
  `compute_vision_embeds`, `forward_vl_chunk`, `forward_vl`, and
  `batched_prefill_vl`; `VisionTower` now lives under `models::vision` so MoE
  can use it without importing from the dense model directory.
- `Qwen35MoeTextModel` already exposes the same key hooks required for VL:
  `embed_on` and `forward_post_embedding_on`.
- `cross_modal::replace_image_tokens` is already batch-capable and validates
  hidden-size equality.
- `GenerateRequest`, `GenerationStream::new`, `Scheduler::prefill_admitted`,
  and `Scheduler::admit_mid_*` already carry and use image fields through the
  `DenseVlMethods` extension trait.

### Gaps in MoE path

- `Qwen35MoeConfig` currently parses only `text_config`; it ignores top-level
  `vision_config`.
- `Qwen35MoeModel` has no `vision: Option<VisionTower>` field.
- `Qwen35MoeModel::model_meta` hardcodes `spatial_merge_size: 2` and documents
  that MoE has no VL endpoint.
- `Qwen35MoeModel` implements `DenseVlMethods` with panic stubs. This is the
  current runtime blocker.
- Existing dense vision code is mostly dimension-parametric, but comments and
  some tests assume 1024/2560. Implementation should verify the MoE 1152/2048
  path explicitly.

## Feasibility Assessment

Feasible with moderate risk.

The strongest signal is that the MoE checkpoint uses the same `vision_tower.*`
key namespace and the same tokenizer image token ids as dense. Its vision tower
is larger but structurally compatible: hidden/out dimensions differ, while
patch size, merge size, position embedding count, and heads remain compatible
with the existing code shape.

The primary implementation risk is not architecture mismatch; it is accidental
coupling to dense-only assumptions in tests and comments.

## Recommended Work Breakdown

1. Add `VisionConfig` parsing to `Qwen35MoeConfig`.
2. Add `vision: Option<VisionTower>` to `Qwen35MoeModel` and load it when
   `vision_config` and `vision_tower.patch_embed.proj.weight` are present.
3. Replace MoE `DenseVlMethods` panic stubs with real implementations mirroring
   dense logic but using MoE text backbone and untied `lm_head`.
4. Add cheap unit tests for config parsing and fake-shape VL scatter/project
   path where possible.
5. Add ignored real-checkpoint tests:
   - `compute_vision_embeds` shape for MoE: output hidden must be 2048.
   - single-image MoE `forward_vl` smoke: finite logits, shape `[1, 1, 248320]`.
   - optional first-token sentinel once an external/reference baseline is chosen.
6. Only after shape/smoke passes, wire an HTTP MoE image request through
   existing OpenAI path and ensure it no longer hits the panic stub.

## Decisions And Remaining Questions For Boss

1. Module ownership: Boss accepted extracting the existing vision implementation
   first. `VisionTower` is now exposed through `crate::models::vision`, while
   `VisionConfig` remains in `qwen3_5::config` and is re-exported from the
   shared module for the first extraction step.
2. Verification baseline: use mlx-vlm for MoE-VL first-token/logit comparison,
   or start with ironmlx shape/finite smoke only and defer numerical baseline.
3. Server behavior: should MoE image requests be enabled immediately once
   methods are real, or hidden behind an explicit feature/CLI flag until the
   ignored real-checkpoint tests pass.

## Recommended Decision

The shared-module extraction has been accepted and implemented as the first
step. Continue by wiring MoE-VL against `crate::models::vision::VisionTower`,
with dense and MoE models each owning only their top-level text-backbone-specific
routing.
