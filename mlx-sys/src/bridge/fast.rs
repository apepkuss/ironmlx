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
//!
//! Each fn carries 4 trailing `StreamOrDevice` args (P5.7) — same encoding
//! as the array bridge: `(has_target, is_device_only, device_type, stream_index)`.

#[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
#[cxx::bridge(namespace = "cxx_mlx")]
pub mod ffi {
    /// cxx-friendly encoding of `mlx::core::fast::TemplateArg`
    /// (`std::variant<int, bool, Dtype>`).
    /// `kind`: 0 = Int (`int_val`), 1 = Bool (`bool_val`), 2 = Dtype
    /// (`int_val` carries the dtype repr).
    struct TemplateArgC {
        name: String,
        kind: u8,
        int_val: i32,
        bool_val: bool,
    }

    unsafe extern "C++" {
        include!("cxx_mlx_shim/fast.h");

        type MlxArray = crate::bridge::array::ffi::MlxArray;
        type MetalKernelInner;
        type ShapesVec;

        // === P3a ShapesVec ===
        fn shapes_vec_new() -> UniquePtr<ShapesVec>;
        fn shapes_vec_push(v: Pin<&mut ShapesVec>, shape: &[i32]);
        fn shapes_vec_count(v: &ShapesVec) -> usize;

        unsafe fn fast_rms_norm(
            x: &MlxArray,
            weight: *const MlxArray,
            eps: f32,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;

        unsafe fn fast_layer_norm(
            x: &MlxArray,
            weight: *const MlxArray,
            bias: *const MlxArray,
            eps: f32,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
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
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;

        unsafe fn fast_rope_with_array_offset(
            x: &MlxArray,
            dims: i32,
            traditional: bool,
            has_base: bool,
            base: f32,
            scale: f32,
            offset: &MlxArray,
            freqs: *const MlxArray,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;

        unsafe fn fast_scaled_dot_product_attention(
            queries: &MlxArray,
            keys: &MlxArray,
            values: &MlxArray,
            scale: f32,
            mask_mode: &str,
            mask_arr: *const MlxArray,
            sinks: *const MlxArray,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
    }
}
