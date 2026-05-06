//! Sorting and partitioning ops.
//!
//! All five fns operate along a single `axis` (negative indexing supported);
//! `sort` returns the sorted values, `argsort` the indices that would sort
//! `a`. `partition` / `argpartition` perform a partial sort positioning the
//! `kth`-smallest element in its sorted slot. `topk` returns the top-k
//! values along `axis` (note: only values — for indices, compose with
//! `argpartition` + `take_along_axis`).
//!
//! Each op exposes a default + `_on` variant (P5.7).
//!
//! ## Naming note (`sort` vs `sort` module)
//!
//! The free function `mlx::ops::sort::sort` lives in the same module path as
//! the module name. To avoid a name collision in a flat re-export, the
//! `sort` fn is **not** re-exported at `mlx::ops` level; reach it as
//! `mlx::ops::sort::sort`. The other four fns *are* re-exported flat.

use crate::{Array, Result};

op_with_stream! {
    /// Sort along `axis` (ascending). Stable; NaNs go to the end (MLX semantics).
    pub fn sort(a: &Array, axis: i32) -> Result<Array>
        => mlx_sys::array::ffi::sort(a.as_inner(), axis);
}

op_with_stream! {
    /// Indices that would sort `a` along `axis` (ascending). Returns `Uint32`.
    pub fn argsort(a: &Array, axis: i32) -> Result<Array>
        => mlx_sys::array::ffi::argsort(a.as_inner(), axis);
}

op_with_stream! {
    /// Partial sort along `axis`: place the `kth`-smallest element at
    /// position `kth`, with smaller elements before and larger after
    /// (within-group ordering is unspecified).
    pub fn partition(a: &Array, kth: i32, axis: i32) -> Result<Array>
        => mlx_sys::array::ffi::partition(a.as_inner(), kth, axis);
}

op_with_stream! {
    /// Indices form of [`partition`]. Returns `Uint32`.
    pub fn argpartition(a: &Array, kth: i32, axis: i32) -> Result<Array>
        => mlx_sys::array::ffi::argpartition(a.as_inner(), kth, axis);
}

op_with_stream! {
    /// Top-k values along `axis` (the largest k). MLX returns **values
    /// only**; for indices compose `argpartition` + `take_along_axis`.
    /// The returned values are not guaranteed to be in any specific order
    /// among themselves — sort if you need ordered top-k.
    pub fn topk(a: &Array, k: i32, axis: i32) -> Result<Array>
        => mlx_sys::array::ffi::topk(a.as_inner(), k, axis);
}
