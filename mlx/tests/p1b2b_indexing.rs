use mlx::{ops, Array, Error};

#[test]
fn where_basic() {
    // cond = [[1, 0], [0, 1]] (bridged through u8 — non-zero is true),
    // x = [[1, 2], [3, 4]], y = [[10, 20], [30, 40]]
    // result = [[1, 20], [30, 4]]
    let cond_data: Vec<u8> = vec![1, 0, 0, 1];
    let cond = Array::from_slice(&cond_data, &[2, 2]).expect("from_slice cond");
    let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]).expect("from_slice x");
    let y = Array::from_slice(&[10.0_f32, 20.0, 30.0, 40.0], &[2, 2]).expect("from_slice y");
    let r = ops::where_(&cond, &x, &y).expect("where_");
    assert_eq!(r.shape().as_slice(), &[2, 2]);
    assert_eq!(r.to_vec::<f32>().expect("to_vec"), vec![1.0, 20.0, 30.0, 4.0]);
}

#[test]
fn where_with_broadcasting() {
    // cond [2, 1], x [2, 3], y [3] (broadcast across all)
    let cond_data: Vec<u8> = vec![1, 0];
    let cond = Array::from_slice(&cond_data, &[2, 1]).expect("from_slice");
    let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("from_slice");
    let y = Array::from_slice(&[100.0_f32, 200.0, 300.0], &[3]).expect("from_slice");
    let r = ops::where_(&cond, &x, &y).expect("where_");
    assert_eq!(r.shape().as_slice(), &[2, 3]);
    // Row 0: cond=1 (broadcasted) → x; Row 1: cond=0 → y
    assert_eq!(
        r.to_vec::<f32>().expect("to_vec"),
        vec![1.0, 2.0, 3.0, 100.0, 200.0, 300.0]
    );
}

#[test]
fn where_broadcast_mismatch_errors() {
    let cond_data: Vec<u8> = vec![1, 0];
    let cond = Array::from_slice(&cond_data, &[2]).expect("from_slice");
    let x = Array::from_slice(&[1.0_f32; 6], &[2, 3]).expect("from_slice");
    let y = Array::from_slice(&[1.0_f32; 8], &[2, 4]).expect("from_slice");
    let result = ops::where_(&cond, &x, &y);
    assert!(matches!(result, Err(Error::BroadcastMismatch { .. })), "got {result:?}");
}

#[test]
fn where_method_form() {
    // cond.where_(&x, &y) — self is the condition
    let cond_data: Vec<u8> = vec![1, 0];
    let cond = Array::from_slice(&cond_data, &[2]).expect("from_slice");
    let x = Array::from_slice(&[1.0_f32, 2.0], &[2]).expect("from_slice");
    let y = Array::from_slice(&[10.0_f32, 20.0], &[2]).expect("from_slice");
    let r = cond.where_(&x, &y).expect("method form");
    assert_eq!(r.to_vec::<f32>().expect("to_vec"), vec![1.0, 20.0]);
}
