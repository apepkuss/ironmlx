#[cxx::bridge(namespace = "cxx_mlx")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cxx_mlx_shim/transforms.h");

        // Cross-bridge opaque type alias — both bridges refer to the same
        // C++ type cxx_mlx::MlxArray (= mlx::core::array). cxx 1.0 supports
        // sharing opaque types this way as long as the namespace and
        // underlying C++ type match across both bridges.
        type MlxArray = crate::bridge::array::ffi::MlxArray;

        fn eval_one(a: &MlxArray) -> Result<()>;
    }
}
