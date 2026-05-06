//! Integration tests for P5.5 — Tier 1 inference ops.

use mlx::ops::binary as bin;
use mlx::{Array, Dtype};

#[test]
fn equal_returns_bool_array() {
    let a: Array = (&[1.0_f32, 2.0, 3.0][..], (3,)).try_into().unwrap();
    let b: Array = (&[1.0_f32, 5.0, 3.0][..], (3,)).try_into().unwrap();
    let r = bin::equal(&a, &b).expect("equal");
    assert_eq!(r.dtype(), Dtype::Bool);
    let v: Vec<bool> = r.to_vec().unwrap();
    assert_eq!(v, vec![true, false, true]);
}

#[test]
fn not_equal_complements_equal() {
    let a: Array = (&[1.0_f32, 2.0][..], (2,)).try_into().unwrap();
    let b: Array = (&[1.0_f32, 5.0][..], (2,)).try_into().unwrap();
    let r = bin::not_equal(&a, &b).unwrap();
    assert_eq!(r.to_vec::<bool>().unwrap(), vec![false, true]);
}

#[test]
fn less_constructs_mask_for_where() {
    let a: Array = (&[1.0_f32, 5.0, 3.0][..], (3,)).try_into().unwrap();
    let thresh: Array = (&[2.0_f32][..], (1,)).try_into().unwrap();
    let mask = bin::less(&a, &thresh).expect("less");
    let v: Vec<bool> = mask.to_vec().unwrap();
    assert_eq!(v, vec![true, false, false]);
}

#[test]
fn less_equal_greater_greater_equal() {
    let a: Array = (&[1.0_f32, 2.0, 3.0][..], (3,)).try_into().unwrap();
    let b: Array = (&[2.0_f32; 3][..], (3,)).try_into().unwrap();
    assert_eq!(
        bin::less_equal(&a, &b).unwrap().to_vec::<bool>().unwrap(),
        vec![true, true, false]
    );
    assert_eq!(
        bin::greater(&a, &b).unwrap().to_vec::<bool>().unwrap(),
        vec![false, false, true]
    );
    assert_eq!(
        bin::greater_equal(&a, &b)
            .unwrap()
            .to_vec::<bool>()
            .unwrap(),
        vec![false, true, true]
    );
}

#[test]
fn maximum_minimum_element_wise() {
    let a: Array = (&[1.0_f32, 5.0, 3.0][..], (3,)).try_into().unwrap();
    let b: Array = (&[2.0_f32, 4.0, 3.0][..], (3,)).try_into().unwrap();
    assert_eq!(
        bin::maximum(&a, &b).unwrap().to_vec::<f32>().unwrap(),
        vec![2.0, 5.0, 3.0]
    );
    assert_eq!(
        bin::minimum(&a, &b).unwrap().to_vec::<f32>().unwrap(),
        vec![1.0, 4.0, 3.0]
    );
}

#[test]
fn clip_with_both_bounds() {
    let x: Array = (&[-1.0_f32, 0.5, 5.0][..], (3,)).try_into().unwrap();
    let lo: Array = (&[0.0_f32][..], (1,)).try_into().unwrap();
    let hi: Array = (&[1.0_f32][..], (1,)).try_into().unwrap();
    let c = bin::clip(&x, Some(&lo), Some(&hi)).expect("clip");
    assert_eq!(c.to_vec::<f32>().unwrap(), vec![0.0, 0.5, 1.0]);
}

#[test]
fn clip_with_only_min() {
    let x: Array = (&[-1.0_f32, 0.5, 5.0][..], (3,)).try_into().unwrap();
    let lo: Array = (&[0.0_f32][..], (1,)).try_into().unwrap();
    let c = bin::clip(&x, Some(&lo), None).expect("clip");
    assert_eq!(c.to_vec::<f32>().unwrap(), vec![0.0, 0.5, 5.0]);
}

#[test]
fn array_methods_for_comparisons() {
    let a: Array = (&[1.0_f32, 2.0][..], (2,)).try_into().unwrap();
    let b: Array = (&[1.0_f32, 5.0][..], (2,)).try_into().unwrap();
    assert_eq!(
        a.equal(&b).unwrap().to_vec::<bool>().unwrap(),
        vec![true, false]
    );
    assert_eq!(
        a.maximum(&b).unwrap().to_vec::<f32>().unwrap(),
        vec![1.0, 5.0]
    );
}

#[test]
fn stream_routing_for_comparisons() {
    use mlx::Device;
    let a: Array = (&[1.0_f32, 2.0][..], (2,)).try_into().unwrap();
    let b: Array = (&[1.0_f32, 5.0][..], (2,)).try_into().unwrap();
    let r = bin::equal_on(&a, &b, Device::cpu()).expect("equal_on");
    assert_eq!(r.to_vec::<bool>().unwrap(), vec![true, false]);
}
