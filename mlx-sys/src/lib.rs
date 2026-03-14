//! Raw unsafe FFI bindings to MLX via the mlx-c C API.
//!
//! These bindings are generated at build time by bindgen from the mlx-c
//! headers. Do not use this crate directly – use the `ironmlx` crate instead.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(clippy::all)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
