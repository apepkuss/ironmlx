//! Integration tests for P5 ops extensions (matmul family).

use mlx::ops::{inner_product, outer, tensordot, tensordot_axes};
use mlx::Array;

#[test]
fn tensordot_axis_matches_matmul_for_2d() {
    // 2D tensordot(a, b, 1) 等价于 matmul(a, b)
    // a: [2, 3], b: [3, 4] → tensordot=matmul=[2, 4]
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("a");
    let b = Array::from_slice(
        &[
            7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        ],
        &[3, 4],
    )
    .expect("b");
    let td = tensordot(&a, &b, 1).expect("tensordot");
    assert_eq!(td.shape().as_slice(), &[2, 4]);
    let mm = a.matmul(&b).expect("matmul");
    let v_td: Vec<f32> = td.to_vec().expect("td vec");
    let v_mm: Vec<f32> = mm.to_vec().expect("mm vec");
    for (t, m) in v_td.iter().zip(&v_mm) {
        assert!((t - m).abs() < 1e-4, "tensordot {t} != matmul {m}");
    }
}

#[test]
fn tensordot_axes_explicit_contraction() {
    // a: [2, 3], b: [3, 4], 收缩 a 的 axis 1 与 b 的 axis 0 → [2, 4]
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("a");
    let b = Array::from_slice(
        &[
            1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ],
        &[3, 4],
    )
    .expect("b");
    let td = tensordot_axes(&a, &b, &[1], &[0]).expect("tensordot_axes");
    assert_eq!(td.shape().as_slice(), &[2, 4]);
}

#[test]
fn outer_product_shape_and_values() {
    // outer([a0,a1,a2], [b0,b1]) → [[a0*b0, a0*b1], [a1*b0, a1*b1], [a2*b0, a2*b1]]
    let a = Array::from_slice(&[2.0_f32, 3.0, 5.0], &[3]).expect("a");
    let b = Array::from_slice(&[7.0_f32, 11.0], &[2]).expect("b");
    let o = outer(&a, &b).expect("outer");
    assert_eq!(o.shape().as_slice(), &[3, 2]);
    let v: Vec<f32> = o.to_vec().expect("vec");
    let expected = [
        2.0_f32 * 7.0,
        2.0 * 11.0,
        3.0 * 7.0,
        3.0 * 11.0,
        5.0 * 7.0,
        5.0 * 11.0,
    ];
    for (got, want) in v.iter().zip(expected.iter()) {
        assert!((got - want).abs() < 1e-4, "outer: got {got}, want {want}");
    }
}

#[test]
fn inner_product_dot_scalar() {
    // inner_product([1,2,3], [4,5,6]) = 1*4 + 2*5 + 3*6 = 32
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]).expect("a");
    let b = Array::from_slice(&[4.0_f32, 5.0, 6.0], &[3]).expect("b");
    let dot = inner_product(&a, &b).expect("inner");
    let v: Vec<f32> = dot.to_vec().expect("vec");
    assert_eq!(v.len(), 1);
    assert!(
        (v[0] - 32.0).abs() < 1e-4,
        "inner_product = {}, want 32",
        v[0]
    );
}

use mlx::ops::addmm;

#[test]
fn addmm_alpha_beta_formula() {
    // D = β*C + α*(A @ B)
    // A: [2, 3], B: [3, 2], C: [2, 2]
    // 设 α=2.0, β=3.0
    // A @ B 的第 [i,j] 元素 = sum_k A[i,k] * B[k,j]
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("a");
    let b = Array::from_slice(&[1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2]).expect("b");
    let c = Array::from_slice(&[10.0_f32, 20.0, 30.0, 40.0], &[2, 2]).expect("c");

    let d = addmm(&c, &a, &b, 2.0, 3.0).expect("addmm");
    assert_eq!(d.shape().as_slice(), &[2, 2]);

    // 手算参考:
    // A @ B = [[1*1+2*0+3*1, 1*0+2*1+3*1], [4*1+5*0+6*1, 4*0+5*1+6*1]]
    //       = [[4, 5], [10, 11]]
    // D = 3*C + 2*(A@B)
    //   = [[3*10+2*4, 3*20+2*5], [3*30+2*10, 3*40+2*11]]
    //   = [[38, 70], [110, 142]]
    let v: Vec<f32> = d.to_vec().expect("vec");
    let expected = [38.0_f32, 70.0, 110.0, 142.0];
    for (got, want) in v.iter().zip(expected.iter()) {
        assert!((got - want).abs() < 1e-4, "addmm: got {got}, want {want}");
    }
}

use mlx::ops::{block_masked_mm, gather_mm, segmented_mm};

#[test]
fn block_masked_mm_smoke_no_masks() {
    // 不传 mask 时退化为常规 matmul（块大小不影响结果）
    // A: [4, 4], B: [4, 4], block_size=2
    let a_data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
    let a = Array::from_slice(&a_data, &[4, 4]).expect("a");
    let b = Array::from_slice(&b_data, &[4, 4]).expect("b");

    let result = block_masked_mm(&a, &b, 2, None, None, None);

    match result {
        Ok(out) => {
            assert_eq!(out.shape().as_slice(), &[4, 4]);
            match out.to_vec::<f32>() {
                Ok(v) => {
                    for x in &v {
                        assert!(x.is_finite());
                    }
                }
                Err(e) => {
                    let msg = format!("{e:?}");
                    assert!(
                        msg.contains("not yet supported") || msg.contains("NYI"),
                        "block_masked_mm eval non-NYI error: {msg}"
                    );
                }
            }
        }
        Err(e) => {
            let msg = format!("{e:?}");
            assert!(
                msg.contains("not yet supported") || msg.contains("NYI"),
                "block_masked_mm construction non-NYI error: {msg}"
            );
        }
    }
}

#[test]
fn gather_mm_no_indices_smoke() {
    // gather_mm 不传 indices 时退化为常规 batched matmul
    // A: [2, 3, 4], B: [2, 4, 5] → [2, 3, 5]
    let a_data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.01).collect();
    let b_data: Vec<f32> = (0..40).map(|i| (i as f32) * 0.005).collect();
    let a = Array::from_slice(&a_data, &[2, 3, 4]).expect("a");
    let b = Array::from_slice(&b_data, &[2, 4, 5]).expect("b");

    let out = gather_mm(&a, &b, None, None, false).expect("gather_mm");
    assert_eq!(out.shape().as_slice(), &[2, 3, 5]);
    let v: Vec<f32> = out.to_vec().expect("vec");
    for x in &v {
        assert!(x.is_finite(), "gather_mm: non-finite {x}");
    }
}

#[test]
fn segmented_mm_smoke() {
    // segmented_mm: A: [M, K], B: [K, N] (必须 2D, 不支持 batched),
    // segments: i32 array, shape (..., 2), 每行 [start, end] 描述 K 维上的 segment.
    // 构造 1-segment 覆盖全部 K=3: segments = [[0, 3]], shape [1, 2]
    // 期望输出 shape [1, 2, 4] (segments shape 去掉末维 + [M, N])
    let a_data: Vec<f32> = (0..6).map(|i| (i as f32) * 0.1).collect();
    let b_data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.05).collect();
    let a = Array::from_slice(&a_data, &[2, 3]).expect("a");
    let b = Array::from_slice(&b_data, &[3, 4]).expect("b");
    let segments = Array::from_slice(&[0_i32, 3], &[1, 2]).expect("segments");

    let result = segmented_mm(&a, &b, &segments);

    match result {
        Ok(out) => {
            // shape 通常 [B, M, num_segments, N] = [1, 2, 1, 4] 或类似
            let v: Vec<f32> = match out.to_vec::<f32>() {
                Ok(v) => v,
                Err(e) => {
                    let msg = format!("{e:?}");
                    assert!(
                        msg.contains("not yet supported") || msg.contains("NYI"),
                        "segmented_mm eval non-NYI: {msg}"
                    );
                    return;
                }
            };
            for x in &v {
                assert!(x.is_finite(), "segmented_mm: non-finite {x}");
            }
        }
        Err(e) => {
            let msg = format!("{e:?}");
            assert!(
                msg.contains("not yet supported") || msg.contains("NYI"),
                "segmented_mm construction non-NYI error: {msg}"
            );
        }
    }
}

#[test]
fn top_level_re_exports_work() {
    // 验证 P5 公开 API 通过 mlx::ops::* 模块路径访问
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]).expect("a");
    let b = Array::from_slice(&[4.0_f32, 5.0, 6.0], &[3]).expect("b");
    // 确认 ops/mod.rs re-export 链路通过到 P5 新增项
    let dot = mlx::ops::inner_product(&a, &b).expect("inner_product via mlx::ops");
    let v: Vec<f32> = dot.to_vec().expect("vec");
    assert!((v[0] - 32.0).abs() < 1e-4);
}
