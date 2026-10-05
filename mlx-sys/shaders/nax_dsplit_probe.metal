// Diagnostic instantiation of Apple's installed MLX 0.32.3 D256 NAX kernel.
// Upstream implementation copyright Apple Inc.; MLX is MIT licensed.
// No installed MLX library or rival application is modified.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h"

instantiate_kernel("ironmlx_probe_dsplit_bf16", attention_nax_dsplit,
                   bfloat, 64, 32, 256, 4, 2, bfloat, float)
