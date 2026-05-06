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

// === Task 5: ops sweep — indexing _on variants ===

#[test]
fn indexing_take_on_explicit_device() {
    use mlx::ops::indexing as ops_indexing;
    let a: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (4,)).try_into().unwrap();
    let idx: Array = (&[0_u32, 2][..], (2,)).try_into().unwrap();
    let r = ops_indexing::take_on(&a, &idx, 0, Device::cpu()).expect("take_on");
    assert_eq!(r.to_vec::<f32>().unwrap(), vec![1.0, 3.0]);
}

#[test]
fn indexing_slice_on_default_matches() {
    use mlx::ops::indexing as ops_indexing;
    let a: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (4,)).try_into().unwrap();
    let r1 = ops_indexing::slice(&a, [1_i32], [3_i32]).expect("slice");
    let r2 = ops_indexing::slice_on(&a, [1_i32], [3_i32], ()).expect("slice_on default");
    assert_eq!(r1.to_vec::<f32>().unwrap(), r2.to_vec::<f32>().unwrap());
}

#[test]
fn indexing_where_on_explicit_device() {
    use mlx::ops::indexing as ops_indexing;
    let cond: Array = (&[1_u8, 0, 1][..], (3,)).try_into().unwrap();
    let x: Array = (&[1.0_f32, 2.0, 3.0][..], (3,)).try_into().unwrap();
    let y: Array = (&[10.0_f32, 20.0, 30.0][..], (3,)).try_into().unwrap();
    let r = ops_indexing::where_on(&cond, &x, &y, Device::cpu()).expect("where_on");
    assert_eq!(r.to_vec::<f32>().unwrap(), vec![1.0, 20.0, 3.0]);
}

// === Task 5: ops sweep — matmul _on variants ===

#[test]
fn matmul_on_explicit_device() {
    use mlx::ops::matmul as ops_matmul;
    let a: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (2, 2)).try_into().unwrap();
    let b: Array = (&[1.0_f32, 0.0, 0.0, 1.0][..], (2, 2)).try_into().unwrap();
    let c = ops_matmul::matmul_on(&a, &b, Device::cpu()).expect("matmul_on");
    assert_eq!(c.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn matmul_addmm_on_default_matches() {
    use mlx::ops::matmul as ops_matmul;
    let c: Array = (&[1.0_f32, 1.0, 1.0, 1.0][..], (2, 2)).try_into().unwrap();
    let a: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (2, 2)).try_into().unwrap();
    let b: Array = (&[1.0_f32, 0.0, 0.0, 1.0][..], (2, 2)).try_into().unwrap();
    let r1 = ops_matmul::addmm(&c, &a, &b, 1.0, 1.0).expect("addmm");
    let r2 = ops_matmul::addmm_on(&c, &a, &b, 1.0, 1.0, ()).expect("addmm_on default");
    assert_eq!(r1.to_vec::<f32>().unwrap(), r2.to_vec::<f32>().unwrap());
}

// === Task 5.F: cross-module smoke test for ops _on variants ===

#[test]
fn ops_smoke_stream_routing() {
    let a: Array = (&[1.0_f32, 4.0, 9.0][..], (3,)).try_into().unwrap();

    // unary on CPU device
    let s = mlx::ops::unary::sqrt_on(&a, Device::cpu()).expect("sqrt_on");
    assert_eq!(s.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);

    // reduction
    let m = mlx::ops::reduction::sum_on(&a, mlx::ops::All, false, Device::cpu()).expect("sum_on");
    assert!((m.item::<f32>().unwrap() - 14.0).abs() < 1e-5);

    // shape
    let r = mlx::ops::shape::reshape_on(&a, (3,), Device::cpu()).expect("reshape_on");
    assert_eq!(r.shape().as_slice(), &[3]);

    // indexing
    let idx: Array = (&[0_u32, 2][..], (2,)).try_into().unwrap();
    let t = mlx::ops::indexing::take_on(&a, &idx, 0, Device::cpu()).expect("take_on");
    assert_eq!(t.to_vec::<f32>().unwrap(), vec![1.0, 9.0]);

    // matmul (use 2x2 example)
    let m1: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (2, 2)).try_into().unwrap();
    let m2: Array = (&[1.0_f32, 0.0, 0.0, 1.0][..], (2, 2)).try_into().unwrap();
    let p = mlx::ops::matmul::matmul_on(&m1, &m2, Device::cpu()).expect("matmul_on");
    assert_eq!(p.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

// === Task 7: fast / quantization _on variants ===

#[test]
fn fast_ops_on_variants() {
    let x: Array = (&[1.0_f32; 16][..], (4, 4)).try_into().unwrap();
    let w: Array = (&[1.0_f32; 4][..], (4,)).try_into().unwrap();
    let r = mlx::fast::rms_norm_on(&x, Some(&w), 1e-5, Device::cpu()).expect("rms_norm_on");
    assert_eq!(r.shape().as_slice(), &[4, 4]);
}

#[test]
fn quantization_ops_on_variants() {
    // group_size=64 requires last dim divisible by 64; use [2, 64].
    let x: Array = (&[1.0_f32; 128][..], (2, 64)).try_into().unwrap();
    // quantize returns Vec<Array> — affine mode produces [packed, scales, biases].
    let parts =
        mlx::quantization::quantize_on(&x, Some(64), Some(4), "affine", None, Device::cpu())
            .expect("quantize_on");
    assert_eq!(parts.len(), 3);
    let q = &parts[0];
    let scales = &parts[1];
    let biases = &parts[2];
    let r = mlx::quantization::dequantize_on(
        q,
        scales,
        Some(biases),
        Some(64),
        Some(4),
        "affine",
        None,
        None,
        Device::cpu(),
    )
    .expect("dequantize_on");
    assert_eq!(r.shape().as_slice(), &[2, 64]);
}

// === Task 6: Array methods *_on variants ===

#[test]
fn array_methods_on_variants() {
    let a: Array = (&[1.0_f32, 4.0][..], (2,)).try_into().unwrap();
    let b: Array = (&[2.0_f32, 8.0][..], (2,)).try_into().unwrap();

    // Each *_on variant produces same result as default when target is Device::cpu()
    assert_eq!(
        a.try_add_on(&b, Device::cpu())
            .unwrap()
            .to_vec::<f32>()
            .unwrap(),
        a.try_add(&b).unwrap().to_vec::<f32>().unwrap()
    );
    assert_eq!(
        a.sqrt_on(Device::cpu()).unwrap().to_vec::<f32>().unwrap(),
        a.sqrt().unwrap().to_vec::<f32>().unwrap()
    );
    assert_eq!(
        a.reshape_on((2,), Device::cpu())
            .unwrap()
            .shape()
            .as_slice(),
        &[2]
    );
}
