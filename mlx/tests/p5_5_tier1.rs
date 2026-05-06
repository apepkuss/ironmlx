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

// === P5.5 Task 2: softmax + sort family ===

#[test]
fn softmax_all_axes_normalizes_full_tensor() {
    use mlx::ops::All;
    let logits: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (2, 2)).try_into().unwrap();
    let p = mlx::ops::unary::softmax(&logits, All, false).expect("softmax");
    // With All, the sum across the full tensor is 1.0 (every element shares
    // the global denominator). For per-row normalization, pass `-1` instead.
    let v: Vec<f32> = p.to_vec().unwrap();
    let total: f32 = v.iter().sum();
    assert!((total - 1.0).abs() < 1e-5, "total={total}");
}

#[test]
fn softmax_specific_axis() {
    let logits: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (2, 2)).try_into().unwrap();
    let p = mlx::ops::unary::softmax(&logits, -1, false).expect("softmax axis -1");
    let v: Vec<f32> = p.to_vec().unwrap();
    assert!((v[0] + v[1] - 1.0).abs() < 1e-5);
    assert!((v[2] + v[3] - 1.0).abs() < 1e-5);
}

#[test]
fn sort_ascending_by_default() {
    let a: Array = (&[3.0_f32, 1.0, 2.0][..], (3,)).try_into().unwrap();
    let s = mlx::ops::sort::sort(&a, -1).expect("sort");
    assert_eq!(s.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn argsort_returns_indices() {
    let a: Array = (&[3.0_f32, 1.0, 2.0][..], (3,)).try_into().unwrap();
    let idx = mlx::ops::sort::argsort(&a, -1).expect("argsort");
    let v: Vec<u32> = idx.to_vec().unwrap();
    assert_eq!(v, vec![1, 2, 0]);
}

#[test]
fn topk_returns_largest_k() {
    let a: Array = (&[3.0_f32, 1.0, 5.0, 2.0, 4.0][..], (5,))
        .try_into()
        .unwrap();
    let r = mlx::ops::sort::topk(&a, 3, -1).expect("topk");
    let mut v: Vec<f32> = r.to_vec().unwrap();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(v, vec![3.0, 4.0, 5.0]);
}

#[test]
fn partition_places_kth_in_position() {
    let a: Array = (&[3.0_f32, 1.0, 5.0, 2.0, 4.0][..], (5,))
        .try_into()
        .unwrap();
    let p = mlx::ops::sort::partition(&a, 2, -1).expect("partition");
    let v: Vec<f32> = p.to_vec().unwrap();
    // Element at position 2 must be the 3rd-smallest (== 3.0).
    assert_eq!(v[2], 3.0);
}

#[test]
fn argpartition_partitions_at_kth() {
    let a: Array = (&[3.0_f32, 1.0, 5.0, 2.0, 4.0][..], (5,))
        .try_into()
        .unwrap();
    let idx = mlx::ops::sort::argpartition(&a, 2, -1).expect("argpartition");
    let v: Vec<u32> = idx.to_vec().unwrap();
    assert_eq!(v.len(), 5);
    // Element at idx[2] is the 3rd-smallest (value 3.0 at original index 0).
    assert_eq!(v[2], 0);
}

#[test]
fn array_method_softmax_works() {
    use mlx::ops::All;
    let x: Array = (&[1.0_f32, 2.0][..], (2,)).try_into().unwrap();
    let p = x.softmax(All, false).expect("softmax method");
    let v: Vec<f32> = p.to_vec().unwrap();
    assert!((v[0] + v[1] - 1.0).abs() < 1e-5);
}

#[test]
fn array_methods_for_sort_family() {
    let a: Array = (&[3.0_f32, 1.0, 2.0][..], (3,)).try_into().unwrap();
    assert_eq!(
        a.sort(-1).unwrap().to_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0]
    );
    assert_eq!(
        a.argsort(-1).unwrap().to_vec::<u32>().unwrap(),
        vec![1, 2, 0]
    );
    let mut tk: Vec<f32> = a.topk(2, -1).unwrap().to_vec().unwrap();
    tk.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(tk, vec![2.0, 3.0]);
}

#[test]
fn stream_routing_for_sort() {
    use mlx::Device;
    let a: Array = (&[3.0_f32, 1.0, 2.0][..], (3,)).try_into().unwrap();
    let s = mlx::ops::sort::sort_on(&a, -1, Device::cpu()).expect("sort_on");
    assert_eq!(s.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

// === Task 3: astype ===

#[test]
fn astype_f32_to_f16() {
    let a: Array = (&[1.0_f32, 2.0, 3.0][..], (3,)).try_into().unwrap();
    let b = mlx::ops::cast::astype(&a, Dtype::Float16).expect("astype");
    assert_eq!(b.dtype(), Dtype::Float16);
}

#[test]
fn astype_int_to_float() {
    let a: Array = (&[1_i32, 2, 3][..], (3,)).try_into().unwrap();
    let b = mlx::ops::cast::astype(&a, Dtype::Float32).expect("astype");
    assert_eq!(b.dtype(), Dtype::Float32);
    assert_eq!(b.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn array_method_astype_works() {
    let a: Array = (&[1_i32, 2][..], (2,)).try_into().unwrap();
    let b = a.astype(Dtype::Float32).expect("astype method");
    assert_eq!(b.dtype(), Dtype::Float32);
}

// === P5.5 Task 4: array constructors ===

#[test]
fn arange_with_step() {
    let r = mlx::ops::constructors::arange(0.0, 5.0, 1.0, Dtype::Int32).expect("arange");
    let v: Vec<i32> = r.to_vec().unwrap();
    assert_eq!(v, vec![0, 1, 2, 3, 4]);
}

#[test]
fn linspace_endpoints() {
    let r = mlx::ops::constructors::linspace(0.0, 1.0, 5, Dtype::Float32).expect("linspace");
    let v: Vec<f32> = r.to_vec().unwrap();
    assert!((v[0] - 0.0).abs() < 1e-5);
    assert!((v[4] - 1.0).abs() < 1e-5);
}

#[test]
fn ones_correct_value() {
    let o = mlx::ops::constructors::ones((2, 3), Dtype::Float32).expect("ones");
    assert_eq!(o.to_vec::<f32>().unwrap(), vec![1.0; 6]);
}

#[test]
fn ones_like_zeros_like() {
    let a: Array = (&[1.0_f32; 6][..], (2, 3)).try_into().unwrap();
    let o = mlx::ops::constructors::ones_like(&a).unwrap();
    assert_eq!(o.shape().as_slice(), &[2, 3]);
    let z = mlx::ops::constructors::zeros_like(&a).unwrap();
    assert_eq!(z.to_vec::<f32>().unwrap(), vec![0.0; 6]);
}

#[test]
fn full_with_value() {
    let v: Array = (&[7.0_f32][..], (1,)).try_into().unwrap();
    let f = mlx::ops::constructors::full((2, 2), &v, Dtype::Float32).expect("full");
    assert_eq!(f.to_vec::<f32>().unwrap(), vec![7.0; 4]);
}

#[test]
fn eye_identity_correct() {
    let i = mlx::ops::constructors::identity(3, Dtype::Float32).expect("identity");
    let v: Vec<f32> = i.to_vec().unwrap();
    assert_eq!(v, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,]);
}

#[test]
fn tri_lower_triangular() {
    let t = mlx::ops::constructors::tri(3, 3, 0, Dtype::Float32).expect("tri");
    let v: Vec<f32> = t.to_vec().unwrap();
    assert_eq!(v, vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0,]);
}

#[test]
fn tril_triu_masks() {
    let a: Array = (&[1.0_f32; 9][..], (3, 3)).try_into().unwrap();
    let lower = mlx::ops::constructors::tril(&a, 0).expect("tril");
    let upper = mlx::ops::constructors::triu(&a, 0).expect("triu");
    let lv: Vec<f32> = lower.to_vec().unwrap();
    let uv: Vec<f32> = upper.to_vec().unwrap();
    // Lower triangular zeros above diagonal
    assert_eq!(lv[1], 0.0);
    assert_eq!(lv[2], 0.0);
    assert_eq!(lv[5], 0.0);
    // Upper triangular zeros below diagonal
    assert_eq!(uv[3], 0.0);
    assert_eq!(uv[6], 0.0);
    assert_eq!(uv[7], 0.0);
}
