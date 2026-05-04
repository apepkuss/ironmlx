//! Bridge for MLX fast ops (fused Transformer kernels).
//!
//! Optional `array` arguments use raw `*const MlxArray` — same convention
//! as `async_eval_many` in the stream bridge. nullptr maps to MLX's
//! `std::optional<array>{std::nullopt}`. The Rust-side safe wrapper
//! converts `Option<&Array>` to a raw pointer at call time.
//!
//! Optional `float` arguments (rope's `base`) are encoded as a
//! `bool has_base` + `f32 base` pair to avoid raw float pointers across
//! cxx (which doesn't model `Option<f32>` directly).

#[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
#[cxx::bridge(namespace = "cxx_mlx")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cxx_mlx_shim/fast.h");

        type MlxArray = crate::bridge::array::ffi::MlxArray;

        unsafe fn fast_rms_norm(
            x: &MlxArray,
            weight: *const MlxArray,
            eps: f32,
        ) -> Result<UniquePtr<MlxArray>>;

        unsafe fn fast_layer_norm(
            x: &MlxArray,
            weight: *const MlxArray,
            bias: *const MlxArray,
            eps: f32,
        ) -> Result<UniquePtr<MlxArray>>;

        unsafe fn fast_rope(
            x: &MlxArray,
            dims: i32,
            traditional: bool,
            has_base: bool,
            base: f32,
            scale: f32,
            offset: i32,
            freqs: *const MlxArray,
        ) -> Result<UniquePtr<MlxArray>>;
    }
}
