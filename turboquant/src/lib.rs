//! TurboQuant: KV cache compression via Walsh-Hadamard rotation + Lloyd-Max scalar quantization.
//!
//! Implements Algorithm 1 (TurboQuantMSE) from "TurboQuant: Online Vector Quantization
//! with Near-optimal Distortion Rate" (arXiv 2504.19874).
//!
//! # Architecture
//!
//! Quantize (per head_dim vector):
//!   1. Extract norm: gamma = ||x||_2, x_hat = x / gamma
//!   2. WHT rotation: y = WHT(signs * x_hat)
//!   3. Scalar quantize: indices[j] = nearest_centroid(y[j])
//!   4. Store: (packed_indices, gamma)
//!
//! Dequantize:
//!   1. Lookup: y_hat[j] = codebook[indices[j]]
//!   2. Norm correction: y_hat = y_hat * (gamma / ||y_hat||_2)
//!   3. Inverse rotation: x_hat = WHT(signs * y_hat) (WHT is self-inverse)

pub mod codebook;
pub mod pack;
pub mod quantize;
pub mod wht;
