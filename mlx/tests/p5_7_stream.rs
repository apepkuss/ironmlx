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

// === Task 5: ops sweep — reduction _on variants ===

#[test]
fn reduction_sum_on_all_axes() {
    use mlx::ops::reduction as ops_reduction;
    use mlx::ops::All;
    let a: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (4,)).try_into().unwrap();
    let r = ops_reduction::sum_on(&a, All, false, Device::cpu()).expect("sum_on All");
    assert!((r.item::<f32>().unwrap() - 10.0).abs() < 1e-5);
}

#[test]
fn reduction_mean_on_axis() {
    use mlx::ops::reduction as ops_reduction;
    let a: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (2, 2)).try_into().unwrap();
    let r = ops_reduction::mean_on(&a, 0_i32, false, Device::cpu()).expect("mean_on axis");
    assert_eq!(r.to_vec::<f32>().unwrap(), vec![2.0, 3.0]);
}

#[test]
fn reduction_max_on_axes_default_matches() {
    use mlx::ops::reduction as ops_reduction;
    let a: Array = (&[1.0_f32, 5.0, 2.0, 8.0][..], (2, 2)).try_into().unwrap();
    let r1 = ops_reduction::max(&a, vec![0_i32, 1], false).expect("max");
    let r2 = ops_reduction::max_on(&a, vec![0_i32, 1], false, ()).expect("max_on default");
    assert_eq!(r1.to_vec::<f32>().unwrap(), r2.to_vec::<f32>().unwrap());
}

// === Task 5: ops sweep — shape _on variants ===

#[test]
fn shape_reshape_on_explicit_device() {
    use mlx::ops::shape as ops_shape;
    let a: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (4,)).try_into().unwrap();
    let r = ops_shape::reshape_on(&a, (2, 2), Device::cpu()).expect("reshape_on");
    assert_eq!(r.shape().as_slice(), &[2, 2]);
}

#[test]
fn shape_transpose_on_default_matches() {
    use mlx::ops::shape as ops_shape;
    let a: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (2, 2)).try_into().unwrap();
    let r1 = ops_shape::transpose(&a).expect("transpose");
    let r2 = ops_shape::transpose_on(&a, ()).expect("transpose_on default");
    assert_eq!(r1.to_vec::<f32>().unwrap(), r2.to_vec::<f32>().unwrap());
}

#[test]
fn shape_concatenate_on_explicit_device() {
    use mlx::ops::shape as ops_shape;
    let a: Array = (&[1.0_f32, 2.0][..], (2,)).try_into().unwrap();
    let b: Array = (&[3.0_f32, 4.0][..], (2,)).try_into().unwrap();
    let r = ops_shape::concatenate_on(&[&a, &b], 0, Device::cpu()).expect("concatenate_on");
    assert_eq!(r.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn shape_split_n_on_explicit_device() {
    use mlx::ops::shape as ops_shape;
    let a: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (4,)).try_into().unwrap();
    let parts = ops_shape::split_n_on(&a, 2, 0, Device::cpu()).expect("split_n_on");
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0].to_vec::<f32>().unwrap(), vec![1.0, 2.0]);
    assert_eq!(parts[1].to_vec::<f32>().unwrap(), vec![3.0, 4.0]);
}
