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

#[test]
fn take_along_axis_0() {
    // a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]], indices = [0, 2], axis = 0
    // result = [[1, 2, 3], [7, 8, 9]]
    let a = Array::from_slice(
        &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[3, 3],
    )
    .expect("from_slice");
    let indices = Array::from_slice(&[0_u32, 2], &[2]).expect("from_slice");
    let r = ops::take(&a, &indices, 0).expect("take");
    assert_eq!(r.shape().as_slice(), &[2, 3]);
    assert_eq!(
        r.to_vec::<f32>().expect("to_vec"),
        vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0]
    );
}

#[test]
fn take_along_axis_1() {
    // Same a, indices = [0, 2], axis = 1 → pick cols 0 and 2 → [[1, 3], [4, 6], [7, 9]]
    let a = Array::from_slice(
        &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[3, 3],
    )
    .expect("from_slice");
    let indices = Array::from_slice(&[0_u32, 2], &[2]).expect("from_slice");
    let r = ops::take(&a, &indices, 1).expect("take");
    assert_eq!(r.shape().as_slice(), &[3, 2]);
    assert_eq!(
        r.to_vec::<f32>().expect("to_vec"),
        vec![1.0, 3.0, 4.0, 6.0, 7.0, 9.0]
    );
}

#[test]
fn take_along_axis_pytorch_gather_semantics() {
    // a = [[10, 20, 30], [40, 50, 60]], indices same shape, axis = 1
    // indices = [[0, 2, 1], [1, 0, 2]] → result = [[10, 30, 20], [50, 40, 60]]
    let a = Array::from_slice(&[10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3]).expect("from_slice");
    let indices_data: Vec<u32> = vec![0, 2, 1, 1, 0, 2];
    let indices = Array::from_slice(&indices_data, &[2, 3]).expect("from_slice");
    let r = ops::take_along_axis(&a, &indices, 1).expect("take_along_axis");
    assert_eq!(r.shape().as_slice(), &[2, 3]);
    assert_eq!(
        r.to_vec::<f32>().expect("to_vec"),
        vec![10.0, 30.0, 20.0, 50.0, 40.0, 60.0]
    );
}

#[test]
fn take_method_form() {
    let a = Array::from_slice(&[10.0_f32, 20.0, 30.0], &[3]).expect("from_slice");
    let indices = Array::from_slice(&[2_u32, 0], &[2]).expect("from_slice");
    let r = a.take(&indices, 0).expect("method take");
    assert_eq!(r.to_vec::<f32>().expect("to_vec"), vec![30.0, 10.0]);
}
