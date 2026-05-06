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

#[test]
fn compile_clear_cache_is_callable() {
    use mlx::compile::{clear_cache, compile, ShapeMode};
    use mlx::Array;

    // Build and call a compiled fn so the cache has at least one entry.
    let f = compile(
        |inputs: &[&Array]| -> mlx::Result<Vec<Array>> {
            let one: Array = (&[1.0_f32][..], (1,)).try_into()?;
            Ok(vec![inputs[0].try_add(&one)?])
        },
        ShapeMode::Fixed,
    )
    .expect("compile");

    let x: Array = (&[5.0_f32][..], (1,)).try_into().unwrap();
    let _ = f.invoke(&[&x]).expect("invoke");

    // Now clear — must not panic.
    clear_cache();
}

// === Task 4: op_with_stream! macro pilot — ops::binary _on variants ===

use mlx::ops::binary as ops_binary;
use mlx::{default_stream, Device};

#[test]
fn add_on_default_matches_add() {
    let a: Array = (&[1.0_f32, 2.0][..], (2,)).try_into().unwrap();
    let b: Array = (&[10.0_f32, 20.0][..], (2,)).try_into().unwrap();
    let c1 = ops_binary::add(&a, &b).expect("add");
    let c2 = ops_binary::add_on(&a, &b, ()).expect("add_on default");
    assert_eq!(c1.to_vec::<f32>().unwrap(), c2.to_vec::<f32>().unwrap());
}

#[test]
fn add_on_explicit_stream() {
    let a: Array = (&[1.0_f32, 2.0][..], (2,)).try_into().unwrap();
    let b: Array = (&[10.0_f32, 20.0][..], (2,)).try_into().unwrap();
    let s = default_stream(Device::cpu());
    let c = ops_binary::add_on(&a, &b, s).expect("add_on stream");
    assert_eq!(c.to_vec::<f32>().unwrap(), vec![11.0, 22.0]);
}

#[test]
fn add_on_explicit_device() {
    let a: Array = (&[1.0_f32, 2.0][..], (2,)).try_into().unwrap();
    let b: Array = (&[10.0_f32, 20.0][..], (2,)).try_into().unwrap();
    let c = ops_binary::add_on(&a, &b, Device::cpu()).expect("add_on device");
    assert_eq!(c.to_vec::<f32>().unwrap(), vec![11.0, 22.0]);
}

// === Task 5: ops sweep — unary _on variants ===

#[test]
fn unary_sqrt_on_default_matches_sqrt() {
    use mlx::ops::unary as ops_unary;
    let a: Array = (&[1.0_f32, 4.0, 9.0][..], (3,)).try_into().unwrap();
    let r1 = ops_unary::sqrt(&a).expect("sqrt");
    let r2 = ops_unary::sqrt_on(&a, ()).expect("sqrt_on default");
    assert_eq!(r1.to_vec::<f32>().unwrap(), r2.to_vec::<f32>().unwrap());
}

#[test]
fn unary_exp_on_explicit_device() {
    use mlx::ops::unary as ops_unary;
    let a: Array = (&[0.0_f32, 1.0][..], (2,)).try_into().unwrap();
    let r = ops_unary::exp_on(&a, Device::cpu()).expect("exp_on device");
    let v = r.to_vec::<f32>().unwrap();
    assert!((v[0] - 1.0).abs() < 1e-5);
    assert!((v[1] - std::f32::consts::E).abs() < 1e-5);
}
