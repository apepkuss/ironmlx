# Gemma4 Unified Parity Fixtures

This directory contains scripts for Gemma4 unified (`mlx-community/gemma-4-12B-it-4bit`)
processor and logits parity fixtures. Generated `.npy` and `.txt` files are
gitignored because they depend on a local 12B checkpoint and are large.

Reference implementation: `mlx-vlm` `gemma4_unified`, the authoritative
consumer of the `mlx-community` 4-bit weights.

Generate fixtures:

```text
cd /path/to/mlx-vlm
GEMMA4_UNIFIED_MODEL=$HOME/.ironmlx/models/models--mlx-community--gemma-4-12B-it-4bit/snapshots/<sha> \
  uv run --with-editable . python \
  /Users/xin/workspace/ironmlx-backend-gemma4-unified/ironmlx/tests/fixtures/gemma4_unified/gen_fixture.py
```

Run parity tests:

```text
MLX_DIR=$HOME/.local/mlx \
GEMMA4_UNIFIED_MODEL=$HOME/.ironmlx/models/models--mlx-community--gemma-4-12B-it-4bit/snapshots/<sha> \
  cargo test --release -p ironmlx --test gemma4_unified_parity -- --ignored --nocapture --test-threads=1
```
