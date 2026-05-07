//! trybuild compile-fail tests. Each `ui/*.rs` should fail to compile,
//! with the expected error in `ui/*.stderr`. The point: verify the
//! `DispatchBuilder` typestate makes `.dispatch()` callable only when all
//! 5 mandatory fields (inputs / output_shapes / output_dtypes / grid /
//! threadgroup) have been set.

#[test]
fn metal_kernel_typestate_compile_fails() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/trybuild/ui/metal_kernel_missing_*.rs");
}
