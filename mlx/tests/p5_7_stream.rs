//! Integration tests for P5.7 — Stream/Device + batch eval + compile cache.

use mlx::transforms::{async_eval, eval};
use mlx::Array;

#[test]
fn eval_many_arrays() {
    let a: Array = (&[1.0_f32, 2.0][..], (2,)).try_into().expect("a");
    let b: Array = (&[3.0_f32, 4.0][..], (2,)).try_into().expect("b");
    let c = &a + &b;
    let d = &a * &b;
    eval(&[&c, &d]).expect("eval many");
    assert_eq!(c.to_vec::<f32>().unwrap(), vec![4.0, 6.0]);
    assert_eq!(d.to_vec::<f32>().unwrap(), vec![3.0, 8.0]);
}

#[test]
fn eval_empty_list_is_noop() {
    eval(&[]).expect("eval empty");
}

#[test]
fn async_eval_many_arrays() {
    let a: Array = (&[10.0_f32, 20.0][..], (2,)).try_into().unwrap();
    let b = &a + &a;
    let c = &a * &a;
    async_eval(&[&b, &c]).expect("async_eval many");
    // Both should be available — subsequent to_vec implicitly waits for the
    // submitted async work to complete.
    assert_eq!(b.to_vec::<f32>().unwrap(), vec![20.0, 40.0]);
    assert_eq!(c.to_vec::<f32>().unwrap(), vec![100.0, 400.0]);
}
