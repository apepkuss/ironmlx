//! Integration tests for mlx::compile — JIT compilation of Rust closures.

use mlx::compile::{disable_compile, enable_compile, set_compile_mode, CompileMode};

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

use mlx::Array;

#[test]
fn array_vec_round_trip() {
    // We don't expose ArrayVec to end users, but we exercise it via the
    // mlx-sys bridge to lock in count/push/get_at/take_at semantics.
    use mlx_sys::compile::ffi::{
        array_vec_count, array_vec_get_at, array_vec_new, array_vec_push, array_vec_take_at,
    };

    let a = Array::from_slice(&[1.0_f32, 2.0], &[2]).expect("a");
    let b = Array::from_slice(&[3.0_f32, 4.0, 5.0], &[3]).expect("b");
    let c = Array::from_slice(&[6.0_f32], &[1]).expect("c");

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
