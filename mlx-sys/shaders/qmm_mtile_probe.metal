// Apple MLX templates (MIT) with private M-tile instantiations only.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/loader.h"
#include "mlx/backend/metal/kernels/quantized_nax.h"

instantiate_kernel("ironmlx_qmm_bm64_aligned", affine_qmm_t_nax,
                   bfloat16_t, 64, 4, true, false, 64, 64, 64, 2, 2)
instantiate_kernel("ironmlx_qmm_bm64_ragged", affine_qmm_t_nax,
                   bfloat16_t, 64, 4, false, false, 64, 64, 64, 2, 2)
instantiate_kernel("ironmlx_qmm_bm128_aligned", affine_qmm_t_nax,
                   bfloat16_t, 64, 4, true, false, 128, 64, 64, 4, 2)
instantiate_kernel("ironmlx_qmm_bm128_ragged", affine_qmm_t_nax,
                   bfloat16_t, 64, 4, false, false, 128, 64, 64, 4, 2)
