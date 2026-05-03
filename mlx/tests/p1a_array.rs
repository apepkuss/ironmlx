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
