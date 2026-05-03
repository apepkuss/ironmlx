//! Free-function form of MLX ops. Operator overloads (`Add`, `Sub`, etc.)
//! and `Array` methods (`a.exp()`, `a.matmul()`) all delegate here.
//!
//! Every op returns `Result<Array>` because broadcasting validation, dtype
//! mismatch, or MLX-side errors all surface as recoverable Rust errors.

pub mod binary;
pub mod reduction;
pub mod shape;
pub mod unary;

pub use binary::{add, divide, multiply, negative, subtract};
pub use reduction::{All, IntoAxes, argmax, max, mean, min, sum};
pub use shape::{broadcast_to, reshape, transpose, transpose_axes};
pub use unary::{erf, exp, log, reciprocal, rsqrt, sigmoid, sqrt, square, tanh};
