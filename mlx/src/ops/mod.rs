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
pub use indexing::{gather, slice, slice_strided, take, take_along_axis, where_};
pub use matmul::{
    addmm, block_masked_matmul, gather_matmul, inner_product, matmul, outer, segmented_matmul,
    tensordot, tensordot_axes,
};
pub use reduction::{argmax, max, mean, min, sum, All, IntoAxes};
pub use shape::{
    broadcast_to, concatenate, reshape, split_at, split_n, stack, transpose, transpose_axes,
};
pub use unary::{
    erf, erf_on, exp, exp_on, log, log_on, reciprocal, reciprocal_on, rsqrt, rsqrt_on, sigmoid,
    sigmoid_on, sqrt, sqrt_on, square, square_on, tanh, tanh_on,
};
