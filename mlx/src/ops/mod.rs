//! Free-function form of MLX ops. Operator overloads (`Add`, `Sub`, etc.)
//! and `Array` methods (`a.exp()`, `a.matmul()`) all delegate here.
//!
//! Every op returns `Result<Array>` because broadcasting validation, dtype
//! mismatch, or MLX-side errors all surface as recoverable Rust errors.
//!
//! Each binary/unary op exposes a default variant (current default stream)
//! and a `*_on` variant taking `impl Into<StreamOrDevice>` (P5.7). Both
//! are emitted from a single declaration via the
//! [`op_with_stream!`](crate::op_with_stream) macro.

#[macro_use]
mod macros;

pub mod binary;
pub mod cast;
pub mod constructors;
pub mod conv;
pub mod cumulative;
pub mod indexing;
pub mod matmul;
pub mod reduction;
pub mod shape;
pub mod sort;
pub mod unary;

pub use binary::{
    add,
    add_on,
    // P5.5 additions
    clip,
    clip_on,
    divide,
    divide_on,
    equal,
    equal_on,
    greater,
    greater_equal,
    greater_equal_on,
    greater_on,
    less,
    less_equal,
    less_equal_on,
    less_on,
    // P5.6 二元补完
    logaddexp,
    logaddexp_on,
    maximum,
    maximum_on,
    minimum,
    minimum_on,
    multiply,
    multiply_on,
    negative,
    negative_on,
    not_equal,
    not_equal_on,
    power,
    power_on,
    remainder,
    remainder_on,
    subtract,
    subtract_on,
};
pub use cast::{astype, astype_on};
pub use constructors::{
    arange, arange_on, eye, eye_on, full, full_like, full_like_on, full_on, identity, identity_on,
    linspace, linspace_on, ones, ones_like, ones_like_on, ones_on, tri, tri_on, tril, tril_on,
    triu, triu_on, zeros_like, zeros_like_on,
};
pub use cumulative::{cumprod, cumprod_on, cumsum, cumsum_on};
pub use indexing::{
    gather, gather_on, slice, slice_on, slice_strided, slice_strided_on, slice_update,
    slice_update_on, take, take_along_axis, take_along_axis_on, take_on, where_, where_on,
};
pub use matmul::{
    addmm, addmm_on, block_masked_matmul, block_masked_matmul_on, gather_matmul, gather_matmul_on,
    inner_product, inner_product_on, matmul, matmul_on, outer, outer_on, segmented_matmul,
    segmented_matmul_on, tensordot, tensordot_axes, tensordot_axes_on, tensordot_on,
};
pub use reduction::{
    all, all_on, any, any_on, argmax, argmax_on, argmin, argmin_on, logsumexp, logsumexp_on, max,
    max_on, mean, mean_on, min, min_on, prod, prod_on, sum, sum_on, All, IntoAxes,
};
pub use shape::{
    broadcast_to, broadcast_to_on, concatenate, concatenate_on, expand_dims, expand_dims_on,
    flatten, flatten_on, repeat, repeat_on, reshape, reshape_on, split_at, split_at_on, split_n,
    split_n_on, squeeze, squeeze_on, stack, stack_on, transpose, transpose_axes, transpose_axes_on,
    transpose_on,
};
// `sort::sort` is intentionally NOT re-exported at this flat level: it
// would shadow the `sort` module name (`mlx::ops::sort` resolves to the
// module, and `pub use sort::sort` introduces an item with the same path
// segment, producing E0255). Users access the free fn as
// `mlx::ops::sort::sort` (or via `Array::sort`). The other four fns have
// no such conflict and are re-exported flat for convenience.
pub use sort::{
    argpartition, argpartition_on, argsort, argsort_on, partition, partition_on, topk, topk_on,
};
pub use unary::{
    abs, abs_on, ceil, ceil_on, cos, cos_on, erf, erf_on, exp, exp_on, expm1, expm1_on, floor,
    floor_on, isfinite, isfinite_on, isinf, isinf_on, isnan, isnan_on, log, log_on, logical_not,
    logical_not_on, nan_to_num, nan_to_num_on, reciprocal, reciprocal_on, round, round_on, rsqrt,
    rsqrt_on, sigmoid, sigmoid_on, sign, sign_on, sin, sin_on, softmax, softmax_on, sqrt, sqrt_on,
    square, square_on, tan, tan_on, tanh, tanh_on,
};
