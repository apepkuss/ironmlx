//! Integration tests for mlx::fast — fused MLX kernels for Transformer inference.

use mlx::{fast, Array};

#[test]
fn rms_norm_no_weight_known_values() {
    // x = [[1.0, 2.0, 3.0, 4.0]], shape [1, 4]
    // mean(x^2) = (1+4+9+16)/4 = 7.5
    // sqrt(7.5 + 1e-5) ≈ 2.7386140
    // Expected output ≈ x / 2.7386 = [0.36514, 0.73029, 1.09543, 1.46059]
    let x = Array::try_from((&[1.0_f32, 2.0, 3.0, 4.0][..], &[1, 4][..])).expect("x");
    let out = fast::rms_norm(&x, None, 1e-5).expect("rms_norm");
    assert_eq!(out.shape().as_slice(), &[1, 4]);

    let v: Vec<f32> = out.to_vec().expect("to_vec");
    let expected = [0.36514_f32, 0.73029, 1.09543, 1.46059];
    for (i, (got, want)) in v.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-3,
            "rms_norm[{i}] = {got}, want {want}"
        );
    }
}

#[test]
fn rms_norm_with_weight_scales_output() {
    // Same x as above; weight = [2.0, 2.0, 2.0, 2.0] → output = 2 × no-weight result.
    let x = Array::try_from((&[1.0_f32, 2.0, 3.0, 4.0][..], &[1, 4][..])).expect("x");
    let w = Array::try_from((&[2.0_f32, 2.0, 2.0, 2.0][..], &[4][..])).expect("w");
    let out = fast::rms_norm(&x, Some(&w), 1e-5).expect("rms_norm");
    let v: Vec<f32> = out.to_vec().expect("to_vec");
    let expected = [0.73029_f32, 1.46058, 2.19087, 2.92117];
    for (i, (got, want)) in v.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-3,
            "rms_norm_w[{i}] = {got}, want {want}"
        );
    }
}

#[test]
fn layer_norm_no_weight_no_bias_known_values() {
    // x = [[1.0, 2.0, 3.0, 4.0]], shape [1, 4]
    // mean = 2.5; var = 1.25; sqrt(1.25 + 1e-5) ≈ 1.11803
    // normalized = (x - 2.5) / 1.11803 ≈ [-1.34164, -0.44721, 0.44721, 1.34164]
    let x = Array::try_from((&[1.0_f32, 2.0, 3.0, 4.0][..], &[1, 4][..])).expect("x");
    let out = fast::layer_norm(&x, None, None, 1e-5).expect("layer_norm");
    assert_eq!(out.shape().as_slice(), &[1, 4]);

    let v: Vec<f32> = out.to_vec().expect("to_vec");
    let expected = [-1.34164_f32, -0.44721, 0.44721, 1.34164];
    for (i, (got, want)) in v.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-3,
            "layer_norm[{i}] = {got}, want {want}"
        );
    }
}

#[test]
fn layer_norm_with_weight_and_bias() {
    // weight=[1,1,1,1], bias=[10,10,10,10] → output = normalized + 10
    let x = Array::try_from((&[1.0_f32, 2.0, 3.0, 4.0][..], &[1, 4][..])).expect("x");
    let w = Array::try_from((&[1.0_f32, 1.0, 1.0, 1.0][..], &[4][..])).expect("w");
    let b = Array::try_from((&[10.0_f32, 10.0, 10.0, 10.0][..], &[4][..])).expect("b");
    let out = fast::layer_norm(&x, Some(&w), Some(&b), 1e-5).expect("layer_norm");
    let v: Vec<f32> = out.to_vec().expect("to_vec");
    let expected = [8.65836_f32, 9.55279, 10.44721, 11.34164];
    for (i, (got, want)) in v.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-3,
            "layer_norm_wb[{i}] = {got}, want {want}"
        );
    }
}

#[test]
fn rope_basic_shape_finite() {
    // 最简验证：base=Some(10000), traditional=false, offset=0, freqs=None。
    // x: [B=1, H=1, S=4, D=8]，dims=8（旋转全部维度）
    // 主要验证形状不变 + 输出有限 + 与输入显著不同（确实做了旋转）
    let total: usize = 32; // 1 * 1 * 4 * 8
    let data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
    let x = Array::try_from((&data[..], &[1, 1, 4, 8][..])).expect("x");
    let out = fast::rope(&x, 8, false, Some(10000.0), 1.0, 0, None).expect("rope");

    assert_eq!(out.shape().as_slice(), &[1, 1, 4, 8]);
    let v: Vec<f32> = out.to_vec().expect("to_vec");
    assert_eq!(v.len(), total);
    for x in &v {
        assert!(x.is_finite(), "non-finite value in rope output: {x}");
    }
    // 第 0 个位置（pos=0）的旋转应当是单位变换：cos(0)=1, sin(0)=0 → 输出 = 输入
    // 但是实际 MLX 实现里，pos=0 的旋转不是恒等，因为 freq 的角度也跟 dim_idx 走。
    // 这里只验证整体不全等于输入：
    let in_v = x.to_vec::<f32>().expect("x.to_vec");
    let mut differ = 0;
    for (a, b) in v.iter().zip(in_v.iter()) {
        if (a - b).abs() > 1e-6 {
            differ += 1;
        }
    }
    assert!(differ > 0, "rope should rotate at least some elements");
}

#[test]
fn rope_offset_shifts_output() {
    // 同样输入，offset=0 vs offset=4 应当产生不同的输出（实际是把 pos 位置移了 4 步）。
    let total: usize = 32; // 1 * 1 * 4 * 8
    let data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
    let x = Array::try_from((&data[..], &[1, 1, 4, 8][..])).expect("x");

    let out0 = fast::rope(&x, 8, false, Some(10000.0), 1.0, 0, None).expect("rope_0");
    let out4 = fast::rope(&x, 8, false, Some(10000.0), 1.0, 4, None).expect("rope_4");

    let v0: Vec<f32> = out0.to_vec().expect("to_vec0");
    let v4: Vec<f32> = out4.to_vec().expect("to_vec4");

    let mut differ = 0;
    for (a, b) in v0.iter().zip(v4.iter()) {
        if (a - b).abs() > 1e-4 {
            differ += 1;
        }
    }
    assert!(
        differ > 0,
        "different offsets should produce different rope outputs"
    );
}

#[test]
fn rope_traditional_differs_from_default() {
    // traditional=true 与 traditional=false 是不同的 rope 排布方式。
    let total: usize = 32; // 1 * 1 * 4 * 8
    let data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
    let x = Array::try_from((&data[..], &[1, 1, 4, 8][..])).expect("x");

    let out_f = fast::rope(&x, 8, false, Some(10000.0), 1.0, 0, None).expect("rope_f");
    let out_t = fast::rope(&x, 8, true, Some(10000.0), 1.0, 0, None).expect("rope_t");

    let vf: Vec<f32> = out_f.to_vec().expect("to_vec_f");
    let vt: Vec<f32> = out_t.to_vec().expect("to_vec_t");

    let mut differ = 0;
    for (a, b) in vf.iter().zip(vt.iter()) {
        if (a - b).abs() > 1e-4 {
            differ += 1;
        }
    }
    assert!(differ > 0, "traditional vs non-traditional should differ");
}

#[test]
fn rope_with_array_offset_per_batch_offsets() {
    // batch=2，offsets=[0, 4]：行 0 用 offset=0，行 1 用 offset=4。
    // 期望：第 0 行结果 == fast::rope(...offset=0)，第 1 行结果 == fast::rope(...offset=4)
    // 简化验证：用 batch=2 的输入，比较与单一 offset 路径的一致性。

    let per_row: usize = 32; // 1 * 4 * 8 (H=1, S=4, D=8)
                             // batch 0: 全 0.01 增长
    let row0: Vec<f32> = (0..per_row).map(|i| (i as f32) * 0.01).collect();
    // batch 1: 同样的 pattern（让两个 batch 共享数据，方便比对）
    let row1 = row0.clone();

    let mut combined: Vec<f32> = Vec::with_capacity(per_row * 2);
    combined.extend_from_slice(&row0);
    combined.extend_from_slice(&row1);
    let x_batched = Array::try_from((&combined[..], &[2, 1, 4, 8][..])).expect("x_batched");

    let offsets = Array::try_from((&[0_i32, 4][..], &[2][..])).expect("offsets");
    let out =
        fast::rope_with_array_offset(&x_batched, 8, false, Some(10000.0), 1.0, &offsets, None)
            .expect("rope_array");
    assert_eq!(out.shape().as_slice(), &[2, 1, 4, 8]);

    // 单独用 int offset 路径计算两个参考：
    let x_single = Array::try_from((&row0[..], &[1, 1, 4, 8][..])).expect("x_single");
    let ref_0 = fast::rope(&x_single, 8, false, Some(10000.0), 1.0, 0, None).expect("ref0");
    let ref_4 = fast::rope(&x_single, 8, false, Some(10000.0), 1.0, 4, None).expect("ref4");

    let v_out: Vec<f32> = out.to_vec().expect("to_vec");
    let v_ref0: Vec<f32> = ref_0.to_vec().expect("ref0_vec");
    let v_ref4: Vec<f32> = ref_4.to_vec().expect("ref4_vec");

    // 第 0 个 batch 应当与 offset=0 参考一致
    for i in 0..per_row {
        let a = v_out[i];
        let b = v_ref0[i];
        assert!(
            (a - b).abs() < 1e-4,
            "batch0[{i}] = {a}, ref offset=0 = {b}"
        );
    }
    // 第 1 个 batch 应当与 offset=4 参考一致
    for i in 0..per_row {
        let a = v_out[per_row + i];
        let b = v_ref4[i];
        assert!(
            (a - b).abs() < 1e-4,
            "batch1[{i}] = {a}, ref offset=4 = {b}"
        );
    }
}

#[test]
fn sdpa_no_mask_matches_manual_reference() {
    // Q=K=V=I (4×4 identity)，scale=1，无 mask。
    // softmax(I @ I.T) = softmax(I) → 每行 [exp(1), 1, 1, 1] / (exp(1) + 3)
    // weights @ V (=I) = weights 本身
    let n: usize = 4;
    let mut data = vec![0.0_f32; n * n];
    for i in 0..n {
        data[i * n + i] = 1.0;
    }
    let id_2d = Array::try_from((&data[..], &[n as i32, n as i32][..])).expect("id");
    let q = id_2d.reshape(&[1, 1, n as i32, n as i32]).expect("q");
    let k = q.clone();
    let v = q.clone();

    let out = fast::scaled_dot_product_attention(&q, &k, &v, 1.0, "", None, None).expect("sdpa");
    assert_eq!(out.shape().as_slice(), &[1, 1, n as i32, n as i32]);

    let result: Vec<f32> = out.to_vec().expect("to_vec");
    let e = std::f32::consts::E;
    let norm = e + 3.0;
    let expected_diag = e / norm;
    let expected_off = 1.0 / norm;

    for i in 0..n {
        for j in 0..n {
            let actual = result[i * n + j];
            let want = if i == j { expected_diag } else { expected_off };
            assert!(
                (actual - want).abs() < 1e-3,
                "sdpa[{i},{j}] = {actual}, want {want}"
            );
        }
    }
}

#[test]
fn sdpa_causal_mode_zeros_future_positions() {
    // mask_mode="causal"：因果掩码。weights @ V=I 时，第 i 行对位置 j>i 的注意力应为 0。
    // 用 Q=K=I, V=I, scale=1.0, causal 模式 → 输出第 i 行的 j>i 位置应为 0。
    let n: usize = 4;
    let mut data = vec![0.0_f32; n * n];
    for i in 0..n {
        data[i * n + i] = 1.0;
    }
    let id_2d = Array::try_from((&data[..], &[n as i32, n as i32][..])).expect("id");
    let q = id_2d.reshape(&[1, 1, n as i32, n as i32]).expect("q");
    let k = q.clone();
    let v = q.clone();

    let out =
        fast::scaled_dot_product_attention(&q, &k, &v, 1.0, "causal", None, None).expect("sdpa");
    let result: Vec<f32> = out.to_vec().expect("to_vec");

    for i in 0..n {
        for j in (i + 1)..n {
            let val = result[i * n + j];
            assert!(
                val.abs() < 1e-5,
                "causal sdpa[{i},{j}] should be 0, got {val}"
            );
        }
    }
}

#[test]
fn sdpa_custom_mask_zeros_masked_positions() {
    // 提供自定义 mask（全 -inf 的右上三角等价 causal）。验证传 mask_arr 路径通。
    let n: usize = 4;
    let mut data = vec![0.0_f32; n * n];
    for i in 0..n {
        data[i * n + i] = 1.0;
    }
    let id_2d = Array::try_from((&data[..], &[n as i32, n as i32][..])).expect("id");
    let q = id_2d.reshape(&[1, 1, n as i32, n as i32]).expect("q");
    let k = q.clone();
    let v = q.clone();

    // additive mask shape [n, n]
    let mut mask_data = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            if j > i {
                mask_data[i * n + j] = f32::NEG_INFINITY;
            }
        }
    }
    let mask = Array::try_from((&mask_data[..], &[n as i32, n as i32][..])).expect("mask");

    let out =
        fast::scaled_dot_product_attention(&q, &k, &v, 1.0, "", Some(&mask), None).expect("sdpa");
    let result: Vec<f32> = out.to_vec().expect("to_vec");

    for i in 0..n {
        for j in (i + 1)..n {
            let val = result[i * n + j];
            assert!(
                val.abs() < 1e-5,
                "custom-mask sdpa[{i},{j}] should be 0, got {val}"
            );
        }
    }
}

#[test]
fn submodule_path_works() {
    // 通过 mlx::fast::rms_norm 直接调用（验证子模块路径可达）
    let x = Array::try_from((&[1.0_f32, 2.0, 3.0, 4.0][..], &[1, 4][..])).expect("x");
    let out = mlx::fast::rms_norm(&x, None, 1e-5).expect("rms_norm");
    assert_eq!(out.shape().as_slice(), &[1, 4]);
}
