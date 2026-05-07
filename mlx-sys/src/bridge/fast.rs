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
    /// (`std::variant<int, bool, Dtype>`). One field per variant arm:
    /// `kind=0` → `int_val` (i32), `kind=1` → `bool_val` (bool),
    /// `kind=2` → `dtype_val` (u8 — same convention as the rest of the
    /// cxx-mlx FFI surface, e.g. `array_zeros`'s `dtype: u8`).
    /// Unused fields are zero-initialized by the Rust producer.
    struct TemplateArgC {
        name: String,
        kind: u8,
        int_val: i32,
        bool_val: bool,
        dtype_val: u8,
    }

    unsafe extern "C++" {
        include!("cxx_mlx_shim/fast.h");

        type MlxArray = crate::bridge::array::ffi::MlxArray;
        type MetalKernelInner;
        type ShapesVec;

        type ArrayVec = crate::bridge::compile::ffi::ArrayVec;

        // === P3a ShapesVec ===
        fn shapes_vec_new() -> UniquePtr<ShapesVec>;
        fn shapes_vec_push(v: Pin<&mut ShapesVec>, shape: &[i32]);
        fn shapes_vec_count(v: &ShapesVec) -> usize;

        // === P3a metal_kernel_build ===
        fn metal_kernel_build(
            name: &str,
            input_names: &[String],
            output_names: &[String],
            source: &str,
            header: &str,
            ensure_row_contiguous: bool,
            atomic_outputs: bool,
        ) -> Result<UniquePtr<MetalKernelInner>>;

        // === P3a metal_kernel_dispatch ===
        fn metal_kernel_dispatch(
            kernel: &MetalKernelInner,
            inputs: &ArrayVec,
            output_shapes: &ShapesVec,
            output_dtypes: &[u8],
            gx: i32,
            gy: i32,
            gz: i32,
            tx: i32,
            ty: i32,
            tz: i32,
            template_args: &[TemplateArgC],
            has_init: bool,
            init_value: f32,
            verbose: bool,
            has_stream: bool,
            dev_only: bool,
            dev_type: u8,
            stream_idx: i32,
        ) -> Result<UniquePtr<ArrayVec>>;

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
