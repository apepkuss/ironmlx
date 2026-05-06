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
pub mod indexing;
pub mod matmul;
pub mod reduction;
pub mod shape;
pub mod unary;

pub use binary::{
    add, add_on, divide, divide_on, multiply, multiply_on, negative, negative_on, subtract,
    subtract_on,
};
pub use indexing::{
    gather, gather_on, slice, slice_on, slice_strided, slice_strided_on, take, take_along_axis,
    take_along_axis_on, take_on, where_, where_on,
};
pub use matmul::{
    addmm, block_masked_matmul, gather_matmul, inner_product, matmul, outer, segmented_matmul,
    tensordot, tensordot_axes,
};
pub use reduction::{
    argmax, argmax_on, max, max_on, mean, mean_on, min, min_on, sum, sum_on, All, IntoAxes,
};
pub use shape::{
    broadcast_to, broadcast_to_on, concatenate, concatenate_on, reshape, reshape_on, split_at,
    split_at_on, split_n, split_n_on, stack, stack_on, transpose, transpose_axes,
    transpose_axes_on, transpose_on,
};
pub use unary::{
    erf, erf_on, exp, exp_on, log, log_on, reciprocal, reciprocal_on, rsqrt, rsqrt_on, sigmoid,
    sigmoid_on, sqrt, sqrt_on, square, square_on, tanh, tanh_on,
};
