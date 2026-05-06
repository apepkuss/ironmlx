// cxx::bridge generates `unsafe fn` declarations for our pointer-slice variants
// (array_concatenate, array_stack). The Safety contracts are documented in the
// safe Rust wrappers (`mlx::ops::concatenate`, `mlx::ops::stack`); cxx doesn't
// propagate doc comments from inside the bridge macro.
//
// `clippy::too_many_arguments` is suppressed because P5.7 adds 4 trailing
// stream-encoding params (has_target/is_device_only/device_type/stream_index)
// to many ops, pushing several past clippy's default 7-arg threshold. The
// safe API in `mlx::ops::*` collapses these back into a single
// `impl Into<StreamOrDevice>` argument.
#[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
#[cxx::bridge(namespace = "cxx_mlx")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cxx_mlx_shim/array.h");

        /// Opaque holder for `mlx::core::array`. Internally refcounted by MLX.
        type MlxArray;

        fn array_zeros(shape: &[i32], dtype: u8) -> Result<UniquePtr<MlxArray>>;
        fn array_shape(a: &MlxArray) -> Vec<i32>;
        fn array_ndim(a: &MlxArray) -> usize;
        fn array_size(a: &MlxArray) -> usize;
        fn array_dtype(a: &MlxArray) -> u8;
        fn array_clone(a: &MlxArray) -> UniquePtr<MlxArray>;
        fn array_is_available(a: &MlxArray) -> bool;

        // from_slice family — Result-wrapped per the shim throw rule
        // (MLX may throw on shape×dtype size mismatch).
        fn array_from_bool(data: &[u8], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;
        fn array_from_u8(data: &[u8], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;
        fn array_from_i8(data: &[i8], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;
        fn array_from_i16(data: &[i16], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;
        fn array_from_i32(data: &[i32], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;
        fn array_from_i64(data: &[i64], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;
        fn array_from_f16(data: &[u16], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;
        fn array_from_bf16(data: &[u16], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;
        fn array_from_f32(data: &[f32], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;
        fn array_from_f64(data: &[f64], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;

        // item family — Result-wrapped (MLX item<T>() may throw if dtype mismatches
        // or eval fails for any reason).
        fn array_item_bool(a: &MlxArray) -> Result<bool>;
        fn array_item_u8(a: &MlxArray) -> Result<u8>;
        fn array_item_i8(a: &MlxArray) -> Result<i8>;
        fn array_item_i16(a: &MlxArray) -> Result<i16>;
        fn array_item_i32(a: &MlxArray) -> Result<i32>;
        fn array_item_i64(a: &MlxArray) -> Result<i64>;
        fn array_item_f16(a: &MlxArray) -> Result<u16>;
        fn array_item_bf16(a: &MlxArray) -> Result<u16>;
        fn array_item_f32(a: &MlxArray) -> Result<f32>;
        fn array_item_f64(a: &MlxArray) -> Result<f64>;

        // to_vec family — Result-wrapped (data() can throw if storage isn't
        // available, e.g. eval hasn't been called).
        fn array_to_vec_bool(a: &MlxArray) -> Result<Vec<u8>>;
        fn array_to_vec_u8(a: &MlxArray) -> Result<Vec<u8>>;
        fn array_to_vec_i8(a: &MlxArray) -> Result<Vec<i8>>;
        fn array_to_vec_i16(a: &MlxArray) -> Result<Vec<i16>>;
        fn array_to_vec_i32(a: &MlxArray) -> Result<Vec<i32>>;
        fn array_to_vec_i64(a: &MlxArray) -> Result<Vec<i64>>;
        fn array_to_vec_f16(a: &MlxArray) -> Result<Vec<u16>>;
        fn array_to_vec_bf16(a: &MlxArray) -> Result<Vec<u16>>;
        fn array_to_vec_f32(a: &MlxArray) -> Result<Vec<f32>>;
        fn array_to_vec_f64(a: &MlxArray) -> Result<Vec<f64>>;

        // Binary ops (P1b1) — Result-wrapped per the shim throw rule.
        // 4 trailing stream params encode StreamOrDevice (P5.7).
        fn array_add(
            a: &MlxArray,
            b: &MlxArray,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
        fn array_subtract(
            a: &MlxArray,
            b: &MlxArray,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
        fn array_multiply(
            a: &MlxArray,
            b: &MlxArray,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
        fn array_divide(
            a: &MlxArray,
            b: &MlxArray,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;

        // Unary ops (P1b1) — Result-wrapped (MLX may throw on dtype not supported,
        // e.g. sqrt on integer types).
        fn array_negative(
            a: &MlxArray,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
        fn array_exp(
            a: &MlxArray,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
        fn array_log(
            a: &MlxArray,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
        fn array_sqrt(
            a: &MlxArray,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
        fn array_tanh(
            a: &MlxArray,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
        fn array_sigmoid(
            a: &MlxArray,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
        fn array_square(
            a: &MlxArray,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
        fn array_rsqrt(
            a: &MlxArray,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
        fn array_erf(
            a: &MlxArray,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
        fn array_reciprocal(
            a: &MlxArray,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;

        // === P1b2a opaque type for std::vector<array> returns ===
        type MlxArrayVec;

        // === P1b2a reductions (5 ops × {all, axis, axes}) ===
        // 4 trailing stream params encode StreamOrDevice (P5.7).
        fn array_sum_all(
            a: &MlxArray,
            keepdims: bool,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
        fn array_sum_axis(
            a: &MlxArray,
            axis: i32,
            keepdims: bool,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
        fn array_sum_axes(
            a: &MlxArray,
            axes: &[i32],
            keepdims: bool,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;

        fn array_mean_all(
            a: &MlxArray,
            keepdims: bool,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
        fn array_mean_axis(
            a: &MlxArray,
            axis: i32,
            keepdims: bool,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
        fn array_mean_axes(
            a: &MlxArray,
            axes: &[i32],
            keepdims: bool,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;

        fn array_max_all(
            a: &MlxArray,
            keepdims: bool,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
        fn array_max_axis(
            a: &MlxArray,
            axis: i32,
            keepdims: bool,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
        fn array_max_axes(
            a: &MlxArray,
            axes: &[i32],
            keepdims: bool,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;

        fn array_min_all(
            a: &MlxArray,
            keepdims: bool,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
        fn array_min_axis(
            a: &MlxArray,
            axis: i32,
            keepdims: bool,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
        fn array_min_axes(
            a: &MlxArray,
            axes: &[i32],
            keepdims: bool,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;

        fn array_argmax_all(
            a: &MlxArray,
            keepdims: bool,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
        fn array_argmax_axis(
            a: &MlxArray,
            axis: i32,
            keepdims: bool,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;

        // === P1b2a shape ops (P5.7: + 4 trailing stream params) ===
        fn array_reshape(
            a: &MlxArray,
            shape: &[i32],
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
        fn array_transpose(
            a: &MlxArray,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
        fn array_transpose_axes(
            a: &MlxArray,
            axes: &[i32],
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
        fn array_broadcast_to(
            a: &MlxArray,
            shape: &[i32],
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;

        // Safety contract: each pointer in `arrays` must point to a valid
        // MlxArray that lives for the duration of the call. The safe wrappers
        // `mlx::ops::concatenate` and `mlx::ops::stack` satisfy this.
        unsafe fn array_concatenate(
            arrays: &[*const MlxArray],
            axis: i32,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
        unsafe fn array_stack(
            arrays: &[*const MlxArray],
            axis: i32,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;

        fn array_split_n(
            a: &MlxArray,
            num_splits: i32,
            axis: i32,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArrayVec>>;
        fn array_split_at(
            a: &MlxArray,
            indices: &[i32],
            axis: i32,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArrayVec>>;

        fn split_result_len(v: &MlxArrayVec) -> usize;
        fn split_result_at(v: &MlxArrayVec, i: usize) -> Result<UniquePtr<MlxArray>>;

        // === P1b2a matmul (P5.7: + 4 trailing stream params) ===
        fn array_matmul(
            a: &MlxArray,
            b: &MlxArray,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;

        // === P1b2b dtype extension ===
        fn array_from_u16(data: &[u16], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;
        fn array_from_u32(data: &[u32], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;
        fn array_from_u64(data: &[u64], shape: &[i32]) -> Result<UniquePtr<MlxArray>>;

        fn array_item_u16(a: &MlxArray) -> Result<u16>;
        fn array_item_u32(a: &MlxArray) -> Result<u32>;
        fn array_item_u64(a: &MlxArray) -> Result<u64>;

        fn array_to_vec_u16(a: &MlxArray) -> Result<Vec<u16>>;
        fn array_to_vec_u32(a: &MlxArray) -> Result<Vec<u32>>;
        fn array_to_vec_u64(a: &MlxArray) -> Result<Vec<u64>>;

        // === P1b2b indexing ops (P5.7: + 4 trailing stream params) ===
        fn array_where(
            cond: &MlxArray,
            x: &MlxArray,
            y: &MlxArray,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
        fn array_take(
            a: &MlxArray,
            indices: &MlxArray,
            axis: i32,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
        fn array_take_along_axis(
            a: &MlxArray,
            indices: &MlxArray,
            axis: i32,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
        fn array_slice_strided(
            a: &MlxArray,
            start: &[i32],
            stop: &[i32],
            strides: &[i32],
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
        unsafe fn array_gather(
            a: &MlxArray,
            indices: &[*const MlxArray],
            axes: &[i32],
            slice_sizes: &[i32],
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;

        // === P5 ops extensions (P5.7: + 4 trailing stream params) ===
        fn tensordot_axis(
            a: &MlxArray,
            b: &MlxArray,
            axis: i32,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;

        fn tensordot_axes(
            a: &MlxArray,
            b: &MlxArray,
            axes_a: &[i32],
            axes_b: &[i32],
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;

        fn outer(
            a: &MlxArray,
            b: &MlxArray,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;

        fn inner(
            a: &MlxArray,
            b: &MlxArray,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;

        fn addmm(
            c: &MlxArray,
            a: &MlxArray,
            b: &MlxArray,
            alpha: f32,
            beta: f32,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;

        unsafe fn block_masked_mm(
            a: &MlxArray,
            b: &MlxArray,
            block_size: i32,
            mask_out: *const MlxArray,
            mask_lhs: *const MlxArray,
            mask_rhs: *const MlxArray,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;

        unsafe fn gather_mm(
            a: &MlxArray,
            b: &MlxArray,
            lhs_indices: *const MlxArray,
            rhs_indices: *const MlxArray,
            sorted_indices: bool,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;

        fn segmented_mm(
            a: &MlxArray,
            b: &MlxArray,
            segments: &MlxArray,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;

        // === P5.5 comparison + element-wise binary ===
        fn equal(
            a: &MlxArray,
            b: &MlxArray,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
        fn not_equal(
            a: &MlxArray,
            b: &MlxArray,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
        fn less(
            a: &MlxArray,
            b: &MlxArray,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
        fn less_equal(
            a: &MlxArray,
            b: &MlxArray,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
        fn greater(
            a: &MlxArray,
            b: &MlxArray,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
        fn greater_equal(
            a: &MlxArray,
            b: &MlxArray,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
        fn maximum(
            a: &MlxArray,
            b: &MlxArray,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
        fn minimum(
            a: &MlxArray,
            b: &MlxArray,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;

        // === P5.5 clip ===
        // a_min / a_max are nullable: pass null pointer for "no bound on that side".
        // unsafe because raw pointers cross FFI; safe wrapper enforces the lifetime contract.
        unsafe fn clip(
            a: &MlxArray,
            a_min: *const MlxArray,
            a_max: *const MlxArray,
            has_target: bool,
            is_device_only: bool,
            device_type: u8,
            stream_index: i32,
        ) -> Result<UniquePtr<MlxArray>>;
    }
}
