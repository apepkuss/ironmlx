//! Bridge for MLX random subsystem.
//!
//! KeyPair opaque wraps std::pair<array, array> from split(key). Single-use
//! semantics: take_first / take_second each callable once (taken_ bool flag).
//!
//! Optional encodings:
//! - Option<&Array> → *const MlxArray (nullptr = None)
//! - Dtype → u8 dtype_repr (shim uses dtype_from_repr from shim_helpers.h)

#[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
#[cxx::bridge(namespace = "cxx_mlx")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cxx_mlx_shim/random.h");

        type MlxArray = crate::bridge::array::ffi::MlxArray;
        type KeyPair;

        // ===== KeyPair accessors =====
        fn key_pair_take_first(p: Pin<&mut KeyPair>) -> Result<UniquePtr<MlxArray>>;
        fn key_pair_take_second(p: Pin<&mut KeyPair>) -> Result<UniquePtr<MlxArray>>;

        // ===== State =====
        fn key(seed: u64) -> Result<UniquePtr<MlxArray>>;
        fn seed(seed: u64);
        fn split(key: &MlxArray) -> Result<UniquePtr<KeyPair>>;
        fn split_n(key: &MlxArray, num: i32) -> Result<UniquePtr<MlxArray>>;

        // ===== Basic distributions =====
        unsafe fn bits(
            shape: &[i32],
            width: i32,
            key: *const MlxArray,
        ) -> Result<UniquePtr<MlxArray>>;

        unsafe fn uniform(
            low: &MlxArray,
            high: &MlxArray,
            shape: &[i32],
            dtype_repr: u8,
            key: *const MlxArray,
        ) -> Result<UniquePtr<MlxArray>>;

        unsafe fn uniform_default(
            shape: &[i32],
            dtype_repr: u8,
            key: *const MlxArray,
        ) -> Result<UniquePtr<MlxArray>>;

        unsafe fn normal(
            shape: &[i32],
            dtype_repr: u8,
            loc: *const MlxArray,
            scale: *const MlxArray,
            key: *const MlxArray,
        ) -> Result<UniquePtr<MlxArray>>;

        unsafe fn randint(
            low: &MlxArray,
            high: &MlxArray,
            shape: &[i32],
            dtype_repr: u8,
            key: *const MlxArray,
        ) -> Result<UniquePtr<MlxArray>>;

        // ===== Discrete distributions =====
        unsafe fn bernoulli(
            p: &MlxArray,
            shape: &[i32],
            key: *const MlxArray,
        ) -> Result<UniquePtr<MlxArray>>;

        unsafe fn bernoulli_default(
            p: &MlxArray,
            key: *const MlxArray,
        ) -> Result<UniquePtr<MlxArray>>;

        unsafe fn categorical(
            logits: &MlxArray,
            axis: i32,
            key: *const MlxArray,
        ) -> Result<UniquePtr<MlxArray>>;

        unsafe fn categorical_n(
            logits: &MlxArray,
            axis: i32,
            num_samples: i32,
            key: *const MlxArray,
        ) -> Result<UniquePtr<MlxArray>>;

        unsafe fn categorical_shaped(
            logits: &MlxArray,
            axis: i32,
            shape: &[i32],
            key: *const MlxArray,
        ) -> Result<UniquePtr<MlxArray>>;
    }
}
