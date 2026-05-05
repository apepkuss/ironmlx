use mlx::{Array, Dtype};

#[test]
fn clone_shares_storage() {
    let a = Array::zeros(&[2, 3], Dtype::Float32).expect("zeros");
    let b = a.clone();
    // Both arrays should report the same shape — they share underlying storage.
    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.size(), b.size());
    assert_eq!(a.dtype(), b.dtype());
}

#[test]
fn original_can_be_dropped_clone_still_usable() {
    let b = {
        let a = Array::zeros(&[5], Dtype::Int32).expect("zeros");
        a.clone()
    };
    // a is dropped; b still works because MLX refcount kept the storage alive.
    assert_eq!(b.size(), 5);
    b.eval().expect("eval after drop should succeed");
}

#[test]
fn debug_does_not_trigger_eval() {
    // Force-eval first to compare; then create a fresh lazy and verify Debug doesn't eval.
    let arr = Array::zeros(&[2, 3], Dtype::Float32).expect("zeros");
    let lazy = Array::zeros(&[4, 5], Dtype::Float32).expect("zeros");
    let was_available_before = mlx_sys::array::ffi::array_is_available(lazy.as_inner());
    let _ = format!("{:?}", lazy);
    let was_available_after = mlx_sys::array::ffi::array_is_available(lazy.as_inner());
    assert_eq!(
        was_available_before, was_available_after,
        "Debug must not trigger eval"
    );
    // Sanity: after explicit eval, is_available should flip to true.
    lazy.eval().expect("eval");
    let was_available_after_eval = mlx_sys::array::ffi::array_is_available(lazy.as_inner());
    assert!(
        was_available_after_eval,
        "after eval, is_available should be true"
    );
    let _ = arr;
}

#[test]
fn debug_format_includes_shape_and_dtype() {
    let arr = Array::zeros(&[2, 3], Dtype::Float32).expect("zeros");
    let s = format!("{:?}", arr);
    assert!(s.contains("shape"), "Debug output missing 'shape': {}", s);
    assert!(
        s.contains("Float32"),
        "Debug output missing 'Float32': {}",
        s
    );
    assert!(s.contains("2"), "Debug output missing dim '2': {}", s);
    assert!(s.contains("3"), "Debug output missing dim '3': {}", s);
}

#[test]
fn shape_returns_smallvec_compatible_slice() {
    use smallvec::SmallVec;
    let arr = Array::zeros(&[2, 3, 4], Dtype::Float32).expect("zeros");
    let s = arr.shape();
    // Verify the public type really is a SmallVec; this would not compile if
    // shape() returned Vec<i32>.
    let _: &SmallVec<[i32; 8]> = &s;
    assert_eq!(s.as_slice(), &[2, 3, 4]);
    assert_eq!(s.len(), 3);
}

#[test]
fn shape_at_supports_negative_indexing() {
    let arr = Array::zeros(&[2, 3, 4], Dtype::Float32).expect("zeros");
    assert_eq!(arr.shape_at(0), 2);
    assert_eq!(arr.shape_at(2), 4);
    assert_eq!(arr.shape_at(-1), 4);
    assert_eq!(arr.shape_at(-3), 2);
}
