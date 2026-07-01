# Gemma4 Long-Context Reference Fixtures

This directory contains tooling for real-checkpoint Gemma4 12B long-context
parity diagnostics. Generated `.npy` and `.json` fixture files are intentionally
not committed because they are large and tied to the local checkpoint revision.

The generator uses mlx-vlm as the reference implementation and writes one case
per requested token length:

- `case_<tokens>_input_ids.npy`
- `case_<tokens>_expected.json`
- `case_<tokens>_expected_after_append_logits.npy`
- `case_<tokens>_expected_layer_last_hiddens.npy`
- `case_<tokens>_expected_layer0_stage_last_hiddens.npy`
- `case_<tokens>_expected_drafter_first_logits.npy` when `--drafter` is set
- `case_<tokens>_expected_drafter_round.json` when `--drafter` is set

Run from the repository root with a local Gemma4 12B MLX snapshot:

```bash
GEMMA4_LONG_CONTEXT_MODEL=$HOME/.ironmlx/models/models--mlx-community--gemma-4-12B-it-4bit/snapshots/<snapshot> \
  uv run --with mlx-vlm --with mlx-lm \
    python ironmlx/tests/fixtures/gemma4_long_context/gen_reference.py \
      --tokens 18000 --tokens 19900 --tokens 20000 --tokens 24000
```

Use `--prefill-step-size 2048` unless the reference implementation changes its
default chunking policy. The generated traces represent the state after the
prompt has been prefetched into cache and the first reference-generated token
has been appended as a single decode step.

To also generate Gemma4 assistant-drafter first-round references:

```bash
GEMMA4_LONG_CONTEXT_MODEL=$HOME/.ironmlx/models/models--mlx-community--gemma-4-12B-it-4bit/snapshots/<snapshot> \
  uv run --with mlx-vlm --with mlx-lm \
    python ironmlx/tests/fixtures/gemma4_long_context/gen_reference.py \
      --tokens 18000 --tokens 24000 \
      --drafter $HOME/.ironmlx/models/models--mlx-community--gemma-4-12B-it-assistant-4bit/snapshots/<snapshot>
```

The drafter round fixture records the first sampled target token, the assistant
draft tokens, and the target verifier tokens for `--draft-tokens 2`.
