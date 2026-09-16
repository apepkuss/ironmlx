# Logits argmax validation record

The supplementary Qwen3.5 MoE logits probe compares five first-token outputs
from IronMLX with an mlx-vlm reference. The recorded result was:

- argmax match: 5/5;
- top-100 maximum absolute difference below `1.0` for bf16: 5/5;
- fp32 threshold below `0.001`: 0/5, which is expected because the model uses
  bf16 inference.

This is an observational comparison, not a release gate or a complete model
acceptance result. Reproduce it with the ignored test
`qwen35_moe_logits_dump` and record the exact model snapshot, MLX revision,
IronMLX commit, and runtime environment with any new result. Generated `.npy`
files are written below the ignored `reports/logits-argmax/` directory and are
not committed.
