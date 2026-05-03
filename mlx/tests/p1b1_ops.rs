use mlx::{Array, Dtype, Error};

#[test]
fn add_same_shape() {
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]).expect("from_slice");
    let b = Array::from_slice(&[10.0_f32, 20.0, 30.0], &[3]).expect("from_slice");
    let c = mlx::ops::add(&a, &b).expect("add");
    let v: Vec<f32> = c.to_vec().expect("to_vec");
    assert_eq!(v, vec![11.0, 22.0, 33.0]);
}

#[test]
fn add_broadcast_scalar_shape() {
    // [2, 3] + [3] should broadcast to [2, 3]
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("from_slice");
    let b = Array::from_slice(&[10.0_f32, 20.0, 30.0], &[3]).expect("from_slice");
    let c = mlx::ops::add(&a, &b).expect("add");
    assert_eq!(c.shape().as_slice(), &[2, 3]);
    let v: Vec<f32> = c.to_vec().expect("to_vec");
    assert_eq!(v, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
}

#[test]
fn add_broadcast_mismatch_err() {
    let a = Array::from_slice(&[1.0_f32; 6], &[2, 3]).expect("from_slice");
    let b = Array::from_slice(&[1.0_f32; 8], &[2, 4]).expect("from_slice");
    let result = mlx::ops::add(&a, &b);
    match result {
        Err(Error::BroadcastMismatch { lhs, rhs }) => {
            assert_eq!(lhs, vec![2, 3]);
            assert_eq!(rhs, vec![2, 4]);
        }
        other => panic!("expected BroadcastMismatch, got {other:?}"),
    }
}

#[test]
fn add_operator_all_ref_combos() {
    let a = Array::from_slice(&[1.0_f32, 2.0], &[2]).expect("from_slice");
    let b = Array::from_slice(&[10.0_f32, 20.0], &[2]).expect("from_slice");

    // All four reference combinations should compile and produce same result.
    let r1 = (&a + &b).expect("&a + &b");
    let r2 = (a.clone() + &b).expect("a + &b");
    let r3 = (&a + b.clone()).expect("&a + b");
    let r4 = (a.clone() + b.clone()).expect("a + b");

    let expected = vec![11.0_f32, 22.0];
    assert_eq!(r1.to_vec::<f32>().expect("to_vec"), expected);
    assert_eq!(r2.to_vec::<f32>().expect("to_vec"), expected);
    assert_eq!(r3.to_vec::<f32>().expect("to_vec"), expected);
    assert_eq!(r4.to_vec::<f32>().expect("to_vec"), expected);
}
