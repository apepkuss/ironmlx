use mlx::{Array, Error};

#[test]
fn add_same_shape() {
    let a = Array::try_from((&[1.0_f32, 2.0, 3.0][..], &[3][..])).expect("try_from");
    let b = Array::try_from((&[10.0_f32, 20.0, 30.0][..], &[3][..])).expect("try_from");
    let c = mlx::ops::add(&a, &b).expect("add");
    let v: Vec<f32> = c.to_vec().expect("to_vec");
    assert_eq!(v, vec![11.0, 22.0, 33.0]);
}

#[test]
fn add_broadcast_scalar_shape() {
    // [2, 3] + [3] should broadcast to [2, 3]
    let a =
        Array::try_from((&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0][..], &[2, 3][..])).expect("try_from");
    let b = Array::try_from((&[10.0_f32, 20.0, 30.0][..], &[3][..])).expect("try_from");
    let c = mlx::ops::add(&a, &b).expect("add");
    assert_eq!(c.shape().as_slice(), &[2, 3]);
    let v: Vec<f32> = c.to_vec().expect("to_vec");
    assert_eq!(v, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
}

#[test]
fn add_broadcast_mismatch_err() {
    let a = Array::try_from((&[1.0_f32; 6][..], &[2, 3][..])).expect("try_from");
    let b = Array::try_from((&[1.0_f32; 8][..], &[2, 4][..])).expect("try_from");
    let result = mlx::ops::add(&a, &b);
    match result {
        Err(Error::BroadcastMismatch { lhs, rhs }) => {
            assert_eq!(lhs.as_slice(), &[2, 3]);
            assert_eq!(rhs.as_slice(), &[2, 4]);
        }
        other => panic!("expected BroadcastMismatch, got {other:?}"),
    }
}

#[test]
fn add_operator_all_ref_combos() {
    let a = Array::try_from((&[1.0_f32, 2.0][..], &[2][..])).expect("try_from");
    let b = Array::try_from((&[10.0_f32, 20.0][..], &[2][..])).expect("try_from");

    // All four reference combinations should compile and produce same result.
    let r1 = &a + &b;
    let r2 = a.clone() + &b;
    let r3 = &a + b.clone();
    let r4 = a.clone() + b.clone();

    let expected = vec![11.0_f32, 22.0];
    assert_eq!(r1.to_vec::<f32>().expect("to_vec"), expected);
    assert_eq!(r2.to_vec::<f32>().expect("to_vec"), expected);
    assert_eq!(r3.to_vec::<f32>().expect("to_vec"), expected);
    assert_eq!(r4.to_vec::<f32>().expect("to_vec"), expected);
}

#[test]
fn sub_mul_div_basic() {
    let a = Array::try_from((&[10.0_f32, 20.0, 30.0][..], &[3][..])).expect("try_from");
    let b = Array::try_from((&[1.0_f32, 2.0, 3.0][..], &[3][..])).expect("try_from");

    let s = &a - &b;
    assert_eq!(s.to_vec::<f32>().expect("to_vec"), vec![9.0, 18.0, 27.0]);

    let m = &a * &b;
    assert_eq!(m.to_vec::<f32>().expect("to_vec"), vec![10.0, 40.0, 90.0]);

    let d = &a / &b;
    assert_eq!(d.to_vec::<f32>().expect("to_vec"), vec![10.0, 10.0, 10.0]);
}

#[test]
fn neg_basic() {
    let a = Array::try_from((&[1.0_f32, -2.0, 3.0][..], &[3][..])).expect("try_from");
    let n = -&a;
    assert_eq!(n.to_vec::<f32>().expect("to_vec"), vec![-1.0, 2.0, -3.0]);
    let n2 = -a;
    assert_eq!(n2.to_vec::<f32>().expect("to_vec"), vec![-1.0, 2.0, -3.0]);
}

#[test]
fn neg_on_unsigned_wraps() {
    // MLX permits negation on u8 (wraps two's-complement style).
    // This test documents the actual runtime behaviour: 1u8 → 255u8, 2u8 → 254u8.
    // (The op succeeds — MLX does not pre-validate dtype for neg.)
    let a = Array::try_from((&[1_u8, 2, 3][..], &[3][..])).expect("try_from");
    let result = -&a;
    let v = result.to_vec::<u8>().expect("to_vec");
    assert_eq!(v, vec![255_u8, 254, 253]);
}

#[test]
fn scalar_rhs_f32() {
    let a = Array::try_from((&[1.0_f32, 2.0, 3.0][..], &[3][..])).expect("try_from");
    let r = &a + 10.0_f32;
    assert_eq!(r.to_vec::<f32>().expect("to_vec"), vec![11.0, 12.0, 13.0]);

    let r2 = &a * 2.0_f32;
    assert_eq!(r2.to_vec::<f32>().expect("to_vec"), vec![2.0, 4.0, 6.0]);
}

#[test]
fn unary_numerical_correctness() {
    let zero = Array::try_from((&[0.0_f32][..], &[][..])).expect("try_from");
    assert!((zero.exp().expect("exp").item::<f32>().expect("item") - 1.0).abs() < 1e-6);
    assert!((zero.erf().expect("erf").item::<f32>().expect("item") - 0.0).abs() < 1e-5);

    let one = Array::try_from((&[1.0_f32][..], &[][..])).expect("try_from");
    assert!((one.log().expect("log").item::<f32>().expect("item") - 0.0).abs() < 1e-6);
    assert!((one.sqrt().expect("sqrt").item::<f32>().expect("item") - 1.0).abs() < 1e-6);
    // Transcendentals at 1e-5 to tolerate kernel/approx changes across MLX versions
    // (tanh/sigmoid implementations can drift in the last few f32 bits).
    assert!((one.tanh().expect("tanh").item::<f32>().expect("item") - 0.7615942).abs() < 1e-5);
    assert!(
        (one.sigmoid().expect("sigmoid").item::<f32>().expect("item") - 0.7310586).abs() < 1e-5
    );
    assert!(
        (one.reciprocal()
            .expect("reciprocal")
            .item::<f32>()
            .expect("item")
            - 1.0)
            .abs()
            < 1e-6
    );

    let three = Array::try_from((&[3.0_f32][..], &[][..])).expect("try_from");
    assert!((three.square().expect("square").item::<f32>().expect("item") - 9.0).abs() < 1e-6);

    let four = Array::try_from((&[4.0_f32][..], &[][..])).expect("try_from");
    assert!((four.rsqrt().expect("rsqrt").item::<f32>().expect("item") - 0.5).abs() < 1e-6);
}

#[test]
fn unary_method_matches_free_fn() {
    let a = Array::try_from((&[1.0_f32, 2.0, 3.0][..], &[3][..])).expect("try_from");
    let by_method = a.exp().expect("method");
    let by_freefn = mlx::ops::exp(&a).expect("free fn");
    assert_eq!(
        by_method.to_vec::<f32>().expect("method to_vec"),
        by_freefn.to_vec::<f32>().expect("freefn to_vec")
    );
}

#[test]
fn unary_chain_composes() {
    // Compute (exp(x) - 1) / 2  for x = [0.0, 1.0]; expected ≈ [0.0, 0.859]
    let x = Array::try_from((&[0.0_f32, 1.0][..], &[2][..])).expect("try_from");
    let r = (&x.exp().expect("exp") - 1.0_f32) / 2.0_f32;
    let v = r.to_vec::<f32>().expect("to_vec");
    assert!((v[0] - 0.0).abs() < 1e-6);
    assert!((v[1] - 0.85914).abs() < 1e-3);
}

#[test]
fn scalar_rhs_i32_on_owned() {
    let a = Array::try_from((&[1_i32, 2, 3][..], &[3][..])).expect("try_from");
    let r = a - 1_i32;
    assert_eq!(r.to_vec::<i32>().expect("to_vec"), vec![0, 1, 2]);
}

#[test]
fn scalar_rhs_half_f16() {
    let a = Array::try_from((
        &[half::f16::from_f32(1.0), half::f16::from_f32(2.0)][..],
        &[2][..],
    ))
    .expect("try_from");
    let r = &a + half::f16::from_f32(0.5);
    let v = r.to_vec::<half::f16>().expect("to_vec");
    assert!((v[0].to_f32() - 1.5).abs() < 1e-3);
    assert!((v[1].to_f32() - 2.5).abs() < 1e-3);
}
