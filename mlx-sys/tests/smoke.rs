/// Verify that the FFI layer links correctly and basic lifecycle calls work.
#[test]
fn array_new_free() {
    unsafe {
        let arr = mlx_sys::mlx_array_new();
        // ctx must be non-null – mlx always allocates
        assert!(!arr.ctx.is_null());
        mlx_sys::mlx_array_free(arr);
    }
}

#[test]
fn scalar_float_roundtrip() {
    unsafe {
        let arr = mlx_sys::mlx_array_new_float(3.14_f32);
        assert!(!arr.ctx.is_null());

        let rc = mlx_sys::mlx_array_eval(arr);
        assert_eq!(rc, 0, "eval failed");

        let mut val: f32 = 0.0;
        let rc = mlx_sys::mlx_array_item_float32(&mut val, arr);
        assert_eq!(rc, 0, "item_float32 failed");
        assert!((val - 3.14_f32).abs() < 1e-5, "value mismatch: {}", val);

        mlx_sys::mlx_array_free(arr);
    }
}
