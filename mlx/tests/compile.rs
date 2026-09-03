//! Integration tests for mlx::compile — JIT compilation of Rust closures.

use mlx::compile::{
    compile, disable_compile, enable_compile, set_compile_mode, CompileMode, ShapeMode,
};
use mlx::Array;
use mlx_sys::compile::ffi::{
    array_vec_count, array_vec_get_at, array_vec_new, array_vec_push, array_vec_take_at,
};

#[test]
fn compile_mode_setters() {
    // Round-trip every variant; the calls must not panic and must leave
    // compile in a usable (enabled) state for subsequent tests.
    set_compile_mode(CompileMode::Disabled);
    set_compile_mode(CompileMode::NoSimplify);
    set_compile_mode(CompileMode::NoFuse);
    set_compile_mode(CompileMode::Enabled);
    disable_compile();
    enable_compile();
}

#[test]
fn array_vec_round_trip() {
    // We don't expose ArrayVec to end users, but we exercise it via the
    // mlx-sys bridge to lock in count/push/get_at/take_at semantics.
    let a = Array::try_from((&[1.0_f32, 2.0][..], &[2][..])).expect("a");
    let b = Array::try_from((&[3.0_f32, 4.0, 5.0][..], &[3][..])).expect("b");
    let c = Array::try_from((&[6.0_f32][..], &[1][..])).expect("c");

    let mut v = array_vec_new();
    assert_eq!(array_vec_count(&v), 0);

    array_vec_push(v.pin_mut(), a.as_inner());
    array_vec_push(v.pin_mut(), b.as_inner());
    array_vec_push(v.pin_mut(), c.as_inner());
    assert_eq!(array_vec_count(&v), 3);

    // get_at clones (shared buffer). Count is unchanged.
    let got1 = array_vec_get_at(&v, 1).expect("get_at 1");
    let got1 = Array::from_inner(got1);
    let got1_vec: Vec<f32> = got1.to_vec().expect("got1 to_vec");
    assert_eq!(got1_vec, vec![3.0, 4.0, 5.0]);
    assert_eq!(array_vec_count(&v), 3);

    // take_at removes the element. Count drops.
    let taken0 = array_vec_take_at(v.pin_mut(), 0).expect("take 0");
    let taken0 = Array::from_inner(taken0);
    let taken0_vec: Vec<f32> = taken0.to_vec().expect("taken0 to_vec");
    assert_eq!(taken0_vec, vec![1.0, 2.0]);
    assert_eq!(array_vec_count(&v), 2);

    // After taking index 0, the previous index 1 (b) is now at index 0.
    let taken_b = array_vec_take_at(v.pin_mut(), 0).expect("take new 0");
    let taken_b = Array::from_inner(taken_b);
    let taken_b_vec: Vec<f32> = taken_b.to_vec().expect("taken_b");
    assert_eq!(taken_b_vec, vec![3.0, 4.0, 5.0]);
    assert_eq!(array_vec_count(&v), 1);

    // Out-of-range take_at returns Err, not UB.
    assert!(array_vec_take_at(v.pin_mut(), 99).is_err());
    // get_at OOB also returns Err.
    assert!(array_vec_get_at(&v, 99).is_err());
}

#[test]
fn compile_simple_unary() {
    enable_compile();
    let f = compile(
        |inputs: &[&Array]| -> mlx::Result<Vec<Array>> {
            let x = inputs[0];
            let one = Array::try_from((&[1.0_f32][..], &[1][..]))?;
            let y = mlx::ops::add(x, &one)?;
            Ok(vec![y])
        },
        ShapeMode::Fixed,
    )
    .expect("compile");

    let x = Array::try_from((&[1.0_f32, 2.0, 3.0][..], &[3][..])).expect("x");
    let outs = f.invoke(&[&x]).expect("invoke");
    assert_eq!(outs.len(), 1);
    let y: Vec<f32> = outs[0].to_vec().expect("to_vec");
    assert_eq!(y, vec![2.0, 3.0, 4.0]);
}

#[test]
fn compile_two_input() {
    let f = compile(
        |inputs: &[&Array]| -> mlx::Result<Vec<Array>> {
            let a = inputs[0];
            let b = inputs[1];
            let prod = mlx::ops::multiply(a, b)?;
            let out = mlx::ops::add(&prod, a)?;
            Ok(vec![out])
        },
        ShapeMode::Fixed,
    )
    .expect("compile");

    let a = Array::try_from((&[2.0_f32, 3.0][..], &[2][..])).expect("a");
    let b = Array::try_from((&[10.0_f32, 100.0][..], &[2][..])).expect("b");
    let outs = f.invoke(&[&a, &b]).expect("invoke");
    let v: Vec<f32> = outs[0].to_vec().expect("v");
    assert_eq!(v, vec![22.0, 303.0]);
}

#[test]
fn compile_captures_weight() {
    let w = Array::try_from((&[10.0_f32, 20.0, 30.0][..], &[3][..])).expect("w");
    let w_for_closure = w.clone();

    let f = compile(
        move |inputs: &[&Array]| -> mlx::Result<Vec<Array>> {
            let x = inputs[0];
            let y = mlx::ops::multiply(x, &w_for_closure)?;
            Ok(vec![y])
        },
        ShapeMode::Fixed,
    )
    .expect("compile");

    let x1 = Array::try_from((&[1.0_f32, 1.0, 1.0][..], &[3][..])).expect("x1");
    let y1: Vec<f32> = f.invoke(&[&x1]).expect("y1")[0].to_vec().expect("v");
    assert_eq!(y1, vec![10.0, 20.0, 30.0]);

    let x2 = Array::try_from((&[2.0_f32, 2.0, 2.0][..], &[3][..])).expect("x2");
    let y2: Vec<f32> = f.invoke(&[&x2]).expect("y2")[0].to_vec().expect("v");
    assert_eq!(y2, vec![20.0, 40.0, 60.0]);
}

#[test]
fn compile_shapeless_reuse() {
    let f = compile(
        |inputs: &[&Array]| -> mlx::Result<Vec<Array>> {
            let two = Array::try_from((&[2.0_f32][..], &[1][..]))?;
            Ok(vec![mlx::ops::multiply(inputs[0], &two)?])
        },
        ShapeMode::Shapeless,
    )
    .expect("compile");

    let x1 = Array::try_from((&[1.0_f32, 2.0][..], &[2][..])).expect("x1");
    let y1: Vec<f32> = f.invoke(&[&x1]).expect("y1")[0].to_vec().expect("v");
    assert_eq!(y1, vec![2.0, 4.0]);

    let x2 = Array::try_from((&[3.0_f32, 4.0, 5.0][..], &[3][..])).expect("x2");
    let y2: Vec<f32> = f.invoke(&[&x2]).expect("y2")[0].to_vec().expect("v");
    assert_eq!(y2, vec![6.0, 8.0, 10.0]);
}

#[test]
fn compile_callback_error_propagates() {
    let f = compile(
        |_inputs: &[&Array]| -> mlx::Result<Vec<Array>> {
            Err(mlx::Error::Mlx("intentional callback failure".into()))
        },
        ShapeMode::Fixed,
    );

    let saw_err = match f {
        Err(_) => true,
        Ok(cf) => {
            let x = Array::try_from((&[1.0_f32][..], &[1][..])).expect("x");
            cf.invoke(&[&x]).is_err()
        }
    };
    assert!(saw_err, "callback Err must propagate as Rust Err");
}

#[test]
fn compile_callback_panic_caught() {
    let f = compile(
        |_inputs: &[&Array]| -> mlx::Result<Vec<Array>> {
            panic!("intentional callback panic");
        },
        ShapeMode::Fixed,
    );

    let saw_err = match f {
        Err(_) => true,
        Ok(cf) => {
            let x = Array::try_from((&[1.0_f32][..], &[1][..])).expect("x");
            cf.invoke(&[&x]).is_err()
        }
    };
    assert!(saw_err, "callback panic must be caught and surfaced as Err");
}

#[test]
fn submodule_path_works() {
    use mlx::compile::{
        compile, disable_compile, enable_compile, set_compile_mode, CompileMode, CompiledFn,
    };

    // Exercise every global control via the mlx::compile::* submodule path.
    set_compile_mode(CompileMode::Enabled);
    disable_compile();
    enable_compile();

    let f: CompiledFn = compile(
        |inputs: &[&Array]| -> mlx::Result<Vec<Array>> {
            let one = Array::try_from((&[1.0_f32][..], &[1][..]))?;
            Ok(vec![mlx::ops::add(inputs[0], &one)?])
        },
        ShapeMode::Fixed,
    )
    .expect("compile via submodule");

    let x = Array::try_from((&[10.0_f32][..], &[1][..])).expect("x");
    let v: Vec<f32> = f.invoke(&[&x]).expect("invoke")[0].to_vec().expect("v");
    assert_eq!(v, vec![11.0]);
}
