//! Integration tests for P5.6 — Tier 2 inference ops.

use mlx::Array;

#[test]
fn abs_negates_negatives() {
    let a: Array = (&[-1.0_f32, 2.0, -3.0][..], (3,)).try_into().unwrap();
    let r = mlx::ops::unary::abs(&a).expect("abs");
    assert_eq!(r.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn sign_returns_minus_one_zero_one() {
    let a: Array = (&[-2.0_f32, 0.0, 3.0][..], (3,)).try_into().unwrap();
    let r = mlx::ops::unary::sign(&a).expect("sign");
    assert_eq!(r.to_vec::<f32>().unwrap(), vec![-1.0, 0.0, 1.0]);
}

#[test]
fn floor_ceil_round() {
    let a: Array = (&[1.4_f32, 1.5, 1.6, -1.4, -1.5, -1.6][..], (6,))
        .try_into()
        .unwrap();
    let f = mlx::ops::unary::floor(&a).expect("floor");
    let c = mlx::ops::unary::ceil(&a).expect("ceil");
    let r = mlx::ops::unary::round(&a, 0).expect("round");
    assert_eq!(
        f.to_vec::<f32>().unwrap(),
        vec![1.0, 1.0, 1.0, -2.0, -2.0, -2.0]
    );
    assert_eq!(
        c.to_vec::<f32>().unwrap(),
        vec![2.0, 2.0, 2.0, -1.0, -1.0, -1.0]
    );
    let rv = r.to_vec::<f32>().unwrap();
    assert_eq!(rv[0], 1.0);
    assert_eq!(rv[5], -2.0);
}

#[test]
fn sin_cos_tan_basic() {
    let a: Array = (&[0.0_f32, std::f32::consts::FRAC_PI_2][..], (2,))
        .try_into()
        .unwrap();
    let s = mlx::ops::unary::sin(&a).expect("sin");
    let c = mlx::ops::unary::cos(&a).expect("cos");
    let sv: Vec<f32> = s.to_vec().unwrap();
    let cv: Vec<f32> = c.to_vec().unwrap();
    assert!((sv[0] - 0.0).abs() < 1e-5);
    assert!((sv[1] - 1.0).abs() < 1e-5);
    assert!((cv[0] - 1.0).abs() < 1e-5);
    assert!(cv[1].abs() < 1e-5);
}

#[test]
fn tan_at_zero() {
    let a: Array = (&[0.0_f32][..], (1,)).try_into().unwrap();
    let t = mlx::ops::unary::tan(&a).expect("tan");
    assert!(t.to_vec::<f32>().unwrap()[0].abs() < 1e-5);
}

#[test]
fn expm1_subtracts_one() {
    let a: Array = (&[0.0_f32, 1.0][..], (2,)).try_into().unwrap();
    let r = mlx::ops::unary::expm1(&a).expect("expm1");
    let v: Vec<f32> = r.to_vec().unwrap();
    assert!(v[0].abs() < 1e-5); // expm1(0) == 0
    assert!((v[1] - (std::f32::consts::E - 1.0)).abs() < 1e-4);
}

#[test]
fn array_methods_for_unary() {
    let a: Array = (&[-2.0_f32, 1.5][..], (2,)).try_into().unwrap();
    assert_eq!(a.abs().unwrap().to_vec::<f32>().unwrap(), vec![2.0, 1.5]);
    assert_eq!(a.sign().unwrap().to_vec::<f32>().unwrap(), vec![-1.0, 1.0]);
    assert_eq!(a.floor().unwrap().to_vec::<f32>().unwrap(), vec![-2.0, 1.0]);
}

#[test]
fn stream_routing_for_abs() {
    use mlx::Device;
    let a: Array = (&[-1.0_f32, 2.0][..], (2,)).try_into().unwrap();
    let r = mlx::ops::unary::abs_on(&a, Device::cpu()).expect("abs_on");
    assert_eq!(r.to_vec::<f32>().unwrap(), vec![1.0, 2.0]);
}
