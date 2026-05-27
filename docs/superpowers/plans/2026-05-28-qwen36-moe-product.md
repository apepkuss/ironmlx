# Qwen3.6 MoE Product Implementation Plan

## Scope

Implement product-grade Qwen3.6 MoE support for the real
`mlx-community/Qwen3.6-35B-A3B-4bit` checkpoint in the sibling worktree branch
`ironmlx-qwen36-moe`.

The final result must expose Qwen3.6 MoE through:

- core model API
- `ironmlx generate`
- `ironmlx serve`

All three surfaces must support text, single-image, and multi-image workflows
where their API contract admits image input.

## Design Review Result

The real checkpoint declares the same Hugging Face architecture and tensor key
layout as Qwen3.5 MoE-VL, while adding Qwen3.6-specific per-module quantization
metadata. The reliable implementation is therefore:

- keep the existing MoE-VL execution kernel shared
- add a named `qwen3_6_moe` architecture package
- validate Qwen3.6 by checkpoint structure
- route Qwen3.6 checkpoints explicitly through the new public model type
- fix loader quantization metadata so Qwen3.6 router gates load correctly

This avoids duplicated numeric kernels while still giving IronMLX a dedicated,
discoverable Qwen3.6 product API.

## Tasks

1. Add `ironmlx/src/models/qwen3_6_moe/`.
   - Implement `Qwen36MoeConfig`.
   - Implement structural Qwen3.6 detector.
   - Implement `Qwen36MoeModel` wrapper.
   - Implement `Model` and `DenseVlMethods`.
   - Add unit tests for config detection and trait surface.

2. Wire model exports and dispatch.
   - Export `qwen3_6_moe` from `ironmlx/src/models/mod.rs`.
   - Update `ironmlx generate` dispatch to prefer Qwen3.6 when detected.
   - Update `ironmlx serve` dispatch to prefer Qwen3.6 when detected.

3. Complete CLI image support.
   - Add repeated `--image <PATH>` arguments.
   - Load text-only checkpoints without vision when no images are provided.
   - Load multimodal checkpoints when images are provided.
   - Preprocess one or more local images.
   - Concatenate image patch arrays and pass image grids to `GenerationStream`.
   - Generate the same image placeholder string shape used by the server.

4. Preserve and test Qwen3.6 quantization.
   - Keep per-prefix quantization override parsing in the loader.
   - Ensure `Linear::from_loader` and `Embedding::from_loader` query prefix
     overrides before global quantization metadata.
   - Keep targeted tests covering normalized `language_model.` prefixes.

5. Add real-checkpoint verification hooks.
   - Add ignored tests that run against `QWEN36_MOE_MODEL`.
   - Cover text, single-image, and multi-image core generation smoke.
   - Reuse existing fixture images from `ironmlx/tests/fixtures/p6_qwen35_vl/`.

6. Run verification.
   - `MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib core::loader::tests`
   - targeted qwen3_6_moe tests
   - real checkpoint CLI text smoke
   - real checkpoint CLI single-image smoke
   - real checkpoint CLI multi-image smoke
   - serve text/single-image/multi-image HTTP smoke
   - required Rust formatting, clippy, and release build commands

## Execution Notes

- Use local paths for CLI image input.
- Use OpenAI chat content parts for serve image input.
- Do not add fallback compatibility paths for unrelated model families.
- Do not copy the MoE execution kernel into Qwen3.6 while the checkpoint
  architecture and tensor graph remain identical.
