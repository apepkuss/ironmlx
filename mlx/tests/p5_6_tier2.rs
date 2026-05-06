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

// === Task 2: 数值卫生 + logical_not ===

#[test]
fn isnan_isinf_isfinite_classify() {
    let a: Array = (
        &[1.0_f32, f32::NAN, f32::INFINITY, -f32::INFINITY, 0.0][..],
        (5,),
    )
        .try_into()
        .unwrap();
    let nan = mlx::ops::unary::isnan(&a).expect("isnan");
    let inf = mlx::ops::unary::isinf(&a).expect("isinf");
    let fin = mlx::ops::unary::isfinite(&a).expect("isfinite");
    assert_eq!(
        nan.to_vec::<bool>().unwrap(),
        vec![false, true, false, false, false]
    );
    assert_eq!(
        inf.to_vec::<bool>().unwrap(),
        vec![false, false, true, true, false]
    );
    assert_eq!(
        fin.to_vec::<bool>().unwrap(),
        vec![true, false, false, false, true]
    );
}

#[test]
fn nan_to_num_replaces_nonfinite() {
    let a: Array = (
        &[1.0_f32, f32::NAN, f32::INFINITY, -f32::INFINITY][..],
        (4,),
    )
        .try_into()
        .unwrap();
    let r = mlx::ops::unary::nan_to_num(&a, 0.0, Some(1e30), Some(-1e30)).expect("nan_to_num");
    let v: Vec<f32> = r.to_vec().unwrap();
    assert_eq!(v[0], 1.0);
    assert_eq!(v[1], 0.0);
    assert!(v[2] >= 1e29);
    assert!(v[3] <= -1e29);
}

#[test]
fn logical_not_inverts_bool() {
    let a: Array = (&[1.0_f32, 0.0, 2.0][..], (3,)).try_into().unwrap();
    let r = mlx::ops::unary::logical_not(&a).expect("logical_not");
    assert_eq!(r.to_vec::<bool>().unwrap(), vec![false, true, false]);
}

// === Task 3: 二元补完 (power / logaddexp / remainder) ===

#[test]
fn power_element_wise() {
    let a: Array = (&[2.0_f32, 3.0][..], (2,)).try_into().unwrap();
    let b: Array = (&[3.0_f32, 2.0][..], (2,)).try_into().unwrap();
    let r = mlx::ops::binary::power(&a, &b).expect("power");
    assert_eq!(r.to_vec::<f32>().unwrap(), vec![8.0, 9.0]);
}

#[test]
fn logaddexp_numerically_stable() {
    let a: Array = (&[0.0_f32, 1.0][..], (2,)).try_into().unwrap();
    let b: Array = (&[0.0_f32, 1.0][..], (2,)).try_into().unwrap();
    let r = mlx::ops::binary::logaddexp(&a, &b).expect("logaddexp");
    let v: Vec<f32> = r.to_vec().unwrap();
    // log(2*e^0) = log(2) ≈ 0.693
    // log(2*e^1) = 1 + log(2) ≈ 1.693
    assert!((v[0] - 0.6931).abs() < 1e-3);
    assert!((v[1] - 1.6931).abs() < 1e-3);
}

#[test]
fn remainder_modulo_like() {
    let a: Array = (&[7.0_f32, -7.0][..], (2,)).try_into().unwrap();
    let b: Array = (&[3.0_f32, 3.0][..], (2,)).try_into().unwrap();
    let r = mlx::ops::binary::remainder(&a, &b).expect("remainder");
    let v: Vec<f32> = r.to_vec().unwrap();
    assert_eq!(v[0], 1.0); // 7 % 3 = 1
    assert!((v[1] - 2.0).abs() < 1e-5); // MLX: positive remainder
}

// === Task 4: reduction 补完 (argmin / all / any / prod / logsumexp) ===

#[test]
fn argmin_finds_min_index() {
    use mlx::ops::All;
    let a: Array = (&[3.0_f32, 1.0, 2.0][..], (3,)).try_into().unwrap();
    let r = mlx::ops::reduction::argmin(&a, All, false).expect("argmin");
    assert_eq!(r.item::<u32>().unwrap(), 1);
}

#[test]
fn prod_multiplies_all() {
    use mlx::ops::All;
    let a: Array = (&[2.0_f32, 3.0, 4.0][..], (3,)).try_into().unwrap();
    let r = mlx::ops::reduction::prod(&a, All, false).expect("prod");
    assert!((r.item::<f32>().unwrap() - 24.0).abs() < 1e-5);
}

#[test]
fn all_any_bool() {
    use mlx::ops::All;
    let mixed: Array = (&[1.0_f32, 1.0, 0.0][..], (3,)).try_into().unwrap();
    let false_arr: Array = (&[0.0_f32, 0.0, 0.0][..], (3,)).try_into().unwrap();
    assert!(!mlx::ops::reduction::all(&mixed, All, false)
        .unwrap()
        .item::<bool>()
        .unwrap());
    assert!(mlx::ops::reduction::any(&mixed, All, false)
        .unwrap()
        .item::<bool>()
        .unwrap());
    assert!(!mlx::ops::reduction::any(&false_arr, All, false)
        .unwrap()
        .item::<bool>()
        .unwrap());
}

#[test]
fn logsumexp_numerically_stable() {
    use mlx::ops::All;
    let a: Array = (&[0.0_f32, 0.0][..], (2,)).try_into().unwrap();
    let r = mlx::ops::reduction::logsumexp(&a, All, false).expect("logsumexp");
    // log(2) ≈ 0.6931
    assert!((r.item::<f32>().unwrap() - 0.6931).abs() < 1e-3);
}

// === Task 5: 累积归约 (cumsum/cumprod) + shape 补完 (flatten/repeat) ===

#[test]
fn cumsum_along_axis() {
    let a: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (4,)).try_into().unwrap();
    let r = mlx::ops::cumulative::cumsum(&a, 0, false, true).expect("cumsum");
    assert_eq!(r.to_vec::<f32>().unwrap(), vec![1.0, 3.0, 6.0, 10.0]);
}

#[test]
fn cumprod_inclusive() {
    let a: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (4,)).try_into().unwrap();
    let r = mlx::ops::cumulative::cumprod(&a, 0, false, true).expect("cumprod");
    assert_eq!(r.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 6.0, 24.0]);
}

#[test]
fn flatten_collapses_dims() {
    let a: Array = (&[1.0_f32; 24][..], (2, 3, 4)).try_into().unwrap();
    let r = mlx::ops::shape::flatten(&a, 0, -1).expect("flatten");
    assert_eq!(r.shape().as_slice(), &[24]);
}

#[test]
fn repeat_along_axis() {
    let a: Array = (&[1.0_f32, 2.0][..], (2,)).try_into().unwrap();
    let r = mlx::ops::shape::repeat(&a, 3, 0).expect("repeat");
    assert_eq!(r.shape().as_slice(), &[6]);
    // Each element repeats `repeats` times consecutively along the axis.
    assert_eq!(
        r.to_vec::<f32>().unwrap(),
        vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    );
}
