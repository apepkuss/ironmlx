//! Integration tests for mlx::random — PRNG + distributions.

use mlx::random::{key, seed, split, split_n};
use mlx::{random, Array, Dtype};

#[test]
fn key_is_deterministic_for_same_seed() {
    let k1 = key(42).expect("key 42");
    let k2 = key(42).expect("key 42 again");
    let v1: Vec<u32> = k1.to_vec().expect("k1 to_vec");
    let v2: Vec<u32> = k2.to_vec().expect("k2 to_vec");
    assert_eq!(v1, v2, "key(42) must be deterministic");
}

#[test]
fn key_differs_for_different_seeds() {
    let k1 = key(42).expect("key 42");
    let k2 = key(43).expect("key 43");
    let v1: Vec<u32> = k1.to_vec().expect("k1");
    let v2: Vec<u32> = k2.to_vec().expect("k2");
    assert_ne!(v1, v2, "different seeds should produce different keys");
}

#[test]
fn split_returns_two_distinct_subkeys() {
    let k = key(42).expect("key");
    let (a, b) = split(&k).expect("split");
    let va: Vec<u32> = a.to_vec().expect("a");
    let vb: Vec<u32> = b.to_vec().expect("b");
    let vk: Vec<u32> = k.to_vec().expect("k");
    assert_ne!(va, vb, "split sub-keys must differ");
    assert_ne!(va, vk, "sub-key 0 must differ from parent");
    assert_ne!(vb, vk, "sub-key 1 must differ from parent");
}

#[test]
fn split_n_returns_n_keys() {
    let k = key(42).expect("key");
    let keys = split_n(&k, 5).expect("split_n");
    assert_eq!(keys.shape().as_slice()[0], 5, "first dim must be num=5");
}

#[test]
fn seed_global_is_callable() {
    seed(123);
    seed(456);
}

#[test]
fn bits_returns_uint32_with_shape() {
    let k = key(42).expect("key");
    let b = random::bits()
        .shape(10)
        .width(4)
        .key(&k)
        .sample()
        .expect("bits");
    assert_eq!(b.shape().as_slice(), &[10]);
    let v: Vec<u32> = b.to_vec().expect("to_vec");
    assert_eq!(v.len(), 10);
}

#[test]
fn uniform_default_in_zero_to_one() {
    let k = key(42).expect("key");
    let u = random::uniform()
        .shape(100)
        .dtype(Dtype::Float32)
        .key(&k)
        .sample()
        .expect("uniform");
    assert_eq!(u.shape().as_slice(), &[100]);
    let v: Vec<f32> = u.to_vec().expect("to_vec");
    for x in &v {
        assert!(*x >= 0.0 && *x < 1.0, "uniform value {x} not in [0, 1)");
    }
}

#[test]
fn uniform_with_low_high_in_range() {
    let k = key(42).expect("key");
    let u = random::uniform()
        .low(2.0)
        .high(5.0)
        .shape(100)
        .dtype(Dtype::Float32)
        .key(&k)
        .sample()
        .expect("uniform");
    let v: Vec<f32> = u.to_vec().expect("to_vec");
    for x in &v {
        assert!(*x >= 2.0 && *x < 5.0, "uniform value {x} not in [2, 5)");
    }
}

#[test]
fn normal_finite_and_centered() {
    let k = key(42).expect("key");
    let n = random::normal()
        .shape(1000)
        .dtype(Dtype::Float32)
        .key(&k)
        .sample()
        .expect("normal");
    assert_eq!(n.shape().as_slice(), &[1000]);
    let v: Vec<f32> = n.to_vec().expect("to_vec");
    for x in &v {
        assert!(x.is_finite(), "non-finite value: {x}");
    }
    let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
    assert!(
        mean.abs() < 0.2,
        "normal mean {mean} not near 0 (loose tolerance for N=1000)"
    );
}

#[test]
fn randint_in_range_and_int32() {
    let k = key(42).expect("key");
    let r = random::randint()
        .low(0)
        .high(10)
        .shape(100)
        .dtype(Dtype::Int32)
        .key(&k)
        .sample()
        .expect("randint");
    let v: Vec<i32> = r.to_vec().expect("to_vec");
    for x in &v {
        assert!(*x >= 0 && *x < 10, "randint value {x} not in [0, 10)");
    }
}

#[test]
fn bernoulli_only_zero_or_one() {
    let k = key(42).expect("key");
    let p = Array::try_from((&[0.5_f32][..], &[][..])).expect("p");
    let b = random::bernoulli(&p)
        .shape(100)
        .key(&k)
        .sample()
        .expect("bernoulli");
    assert_eq!(b.shape().as_slice(), &[100]);
    let v: Vec<bool> = b.to_vec().expect("to_vec");
    assert_eq!(v.len(), 100);
}

#[test]
fn bernoulli_default_shape_from_p() {
    let k = key(42).expect("key");
    let p = Array::try_from((&[0.7_f32][..], &[][..])).expect("p");
    let b = random::bernoulli(&p)
        .key(&k)
        .sample()
        .expect("bernoulli_default");
    assert_eq!(b.shape().as_slice(), &[] as &[i32]);
}

#[test]
fn categorical_index_in_vocab() {
    let k = key(42).expect("key");
    let logits_data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
    let logits = Array::try_from((&logits_data[..], &[4, 8][..])).expect("logits");

    let out = random::categorical(&logits)
        .axis(-1)
        .key(&k)
        .sample()
        .expect("categorical");
    assert_eq!(out.shape().as_slice(), &[4]);
    let v: Vec<u32> = out.to_vec().expect("to_vec");
    for idx in &v {
        assert!(*idx < 8, "categorical idx {idx} out of vocab=[0, 8)");
    }
}

#[test]
fn categorical_n_returns_n_samples() {
    let k = key(42).expect("key");
    let logits_data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
    let logits = Array::try_from((&logits_data[..], &[4, 8][..])).expect("logits");

    let out = random::categorical(&logits)
        .axis(-1)
        .num_samples(3)
        .key(&k)
        .sample()
        .expect("categorical_n");
    assert_eq!(out.shape().as_slice(), &[4, 3]);
    let v: Vec<u32> = out.to_vec().expect("to_vec");
    for idx in &v {
        assert!(*idx < 8, "categorical_n idx {idx} out of vocab");
    }
}

#[test]
fn categorical_shaped_returns_explicit_shape() {
    let k = key(42).expect("key");
    let logits_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
    let logits = Array::try_from((&logits_data[..], &[2, 8][..])).expect("logits");

    let out = random::categorical(&logits)
        .axis(-1)
        .shape((5, 2))
        .key(&k)
        .sample()
        .expect("categorical_shaped");
    assert_eq!(out.shape().as_slice(), &[5, 2]);
}

#[test]
fn truncated_normal_in_bounds() {
    let k = key(42).expect("key");
    let lower = Array::try_from((&[-1.0_f32][..], &[][..])).expect("lower");
    let upper = Array::try_from((&[1.0_f32][..], &[][..])).expect("upper");
    let t = random::truncated_normal(&lower, &upper)
        .shape(100)
        .dtype(Dtype::Float32)
        .key(&k)
        .sample()
        .expect("truncated_normal");
    let v: Vec<f32> = t.to_vec().expect("to_vec");
    for x in &v {
        assert!(
            *x >= -1.0 && *x <= 1.0,
            "truncated value {x} out of [-1, 1]"
        );
    }
}

#[test]
fn truncated_normal_default_broadcast_shape() {
    let k = key(42).expect("key");
    let lower = Array::try_from((&[-1.0_f32, -2.0][..], &[2][..])).expect("lower");
    let upper = Array::try_from((&[1.0_f32, 2.0][..], &[2][..])).expect("upper");
    let t = random::truncated_normal(&lower, &upper)
        .dtype(Dtype::Float32)
        .key(&k)
        .sample()
        .expect("truncated_normal_default");
    assert_eq!(t.shape().as_slice(), &[2]);
}

#[test]
fn gumbel_finite() {
    let k = key(42).expect("key");
    let g = random::gumbel()
        .shape(100)
        .dtype(Dtype::Float32)
        .key(&k)
        .sample()
        .expect("gumbel");
    assert_eq!(g.shape().as_slice(), &[100]);
    let v: Vec<f32> = g.to_vec().expect("to_vec");
    for x in &v {
        assert!(x.is_finite(), "non-finite gumbel value: {x}");
    }
}

#[test]
fn laplace_finite() {
    let k = key(42).expect("key");
    let l = random::laplace()
        .shape(100)
        .dtype(Dtype::Float32)
        .loc(0.0)
        .scale(1.0)
        .key(&k)
        .sample()
        .expect("laplace");
    assert_eq!(l.shape().as_slice(), &[100]);
    let v: Vec<f32> = l.to_vec().expect("to_vec");
    for x in &v {
        assert!(x.is_finite(), "non-finite laplace value: {x}");
    }
}

#[test]
fn multivariate_normal_binding_smoke() {
    let k = key(42).expect("key");
    let mean = Array::try_from((&[0.0_f32, 0.0][..], &[2][..])).expect("mean");
    let cov = Array::try_from((&[1.0_f32, 0.0, 0.0, 1.0][..], &[2, 2][..])).expect("cov");

    let result = random::multivariate_normal(&mean, &cov)
        .shape(10)
        .dtype(Dtype::Float32)
        .key(&k)
        .sample();

    match result {
        Ok(mvn) => {
            assert_eq!(mvn.shape().as_slice(), &[10, 2]);
            match mvn.to_vec::<f32>() {
                Ok(v) => {
                    for x in &v {
                        assert!(x.is_finite(), "non-finite mvn value: {x}");
                    }
                }
                Err(e) => {
                    let msg = format!("{e:?}");
                    assert!(
                        msg.contains("not yet supported") || msg.contains("svd"),
                        "multivariate_normal eval failed with non-NYI error: {msg}"
                    );
                }
            }
        }
        Err(e) => {
            let msg = format!("{e:?}");
            assert!(
                msg.contains("not yet supported") || msg.contains("svd"),
                "multivariate_normal construction failed with non-NYI error: {msg}"
            );
        }
    }
}

#[test]
fn permutation_arange_is_valid_perm() {
    let k = key(42).expect("key");
    let p = random::permutation_range(10)
        .key(&k)
        .sample()
        .expect("permutation_range");
    assert_eq!(p.shape().as_slice(), &[10]);

    let mut v: Vec<u32> = p.to_vec().expect("to_vec");
    v.sort();
    assert_eq!(
        v,
        (0..10).collect::<Vec<u32>>(),
        "permutation must be a re-ordering of 0..n"
    );
}

#[test]
fn permutation_array_preserves_elements() {
    let k = key(42).expect("key");
    let x = Array::try_from((&[1.0_f32, 2.0, 3.0, 4.0, 5.0][..], &[5][..])).expect("x");
    let p = random::permutation(&x)
        .axis(0)
        .key(&k)
        .sample()
        .expect("permutation");
    assert_eq!(p.shape().as_slice(), &[5]);

    let mut v: Vec<f32> = p.to_vec().expect("to_vec");
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(
        v,
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0],
        "permutation must preserve the multiset"
    );
}

#[test]
fn submodule_path_works() {
    // 所有 random API 只通过 mlx::random 子模块路径访问。
    let k = mlx::random::key(42).expect("key via mlx::random");
    let u = mlx::random::uniform()
        .shape(10)
        .dtype(mlx::Dtype::Float32)
        .key(&k)
        .sample()
        .expect("uniform via mlx::random");
    assert_eq!(u.shape().as_slice(), &[10]);
}
