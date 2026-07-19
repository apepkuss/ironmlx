//! TurboQuant GPU primitive tests.

use std::sync::{Mutex, MutexGuard, OnceLock};

use mlx::{Array, Dtype};

fn turboquant_test_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
}

fn test_values(len: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(len);
    let mut state = 0x1234_5678_9abc_def0_u64;
    for _ in 0..len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let u = (state & 0xffff) as f32 / 65_535.0;
        out.push((u - 0.5) * 1.7);
    }
    out
}

fn wht_inplace(x: &mut [f32]) {
    let n = x.len();
    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let a = x[j];
                let b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
            i += h * 2;
        }
        h *= 2;
    }
    let scale = 1.0 / (n as f32).sqrt();
    for v in x {
        *v *= scale;
    }
}

fn reference_query_rotate(input: &[f32], head_dim: usize, signs: &[f32]) -> Vec<f32> {
    let vector_count = input.len() / head_dim;
    let mut out = vec![0.0_f32; input.len()];
    for vec_idx in 0..vector_count {
        let start = vec_idx * head_dim;
        let mut values: Vec<f32> = input[start..start + head_dim]
            .iter()
            .zip(signs.iter())
            .map(|(&x, &sign)| x * sign)
            .collect();
        wht_inplace(&mut values);
        out[start..start + head_dim].copy_from_slice(&values);
    }
    out
}

fn reference_quantize(input: &[f32], head_dim: usize, bits: u8) -> (Vec<u32>, Vec<f32>, usize) {
    let seed = 0x5455_5242_4f51_5541_u64;
    let signs = turboquant::wht::generate_signs(head_dim, seed);
    let codebook = turboquant::codebook::Codebook::new(bits, head_dim);
    let values_per_word = match bits {
        3 => 10,
        4 => 8,
        _ => unreachable!(),
    };
    let packed_dim = head_dim.div_ceil(values_per_word);
    let vector_count = input.len() / head_dim;

    let mut packed = vec![0_u32; vector_count * packed_dim];
    let mut adjusted_norms = vec![0.0_f32; vector_count];

    for vec_idx in 0..vector_count {
        let start = vec_idx * head_dim;
        let x = &input[start..start + head_dim];
        let norm = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let safe_norm = if norm > 1.0e-10 { norm } else { 1.0 };

        let mut rotated: Vec<f32> = x
            .iter()
            .zip(signs.iter())
            .map(|(&x, &sign)| (x / safe_norm) * sign)
            .collect();
        wht_inplace(&mut rotated);

        let indices: Vec<u32> = rotated
            .iter()
            .map(|&v| u32::from(codebook.nearest_index(v)))
            .collect();
        let recon_norm = indices
            .iter()
            .map(|&idx| {
                let c = codebook.centroid(idx as u8);
                c * c
            })
            .sum::<f32>()
            .sqrt();
        adjusted_norms[vec_idx] = norm / recon_norm.max(1.0e-10);

        for word_idx in 0..packed_dim {
            let mut word = 0_u32;
            for i in 0..values_per_word {
                let src = word_idx * values_per_word + i;
                if src < head_dim {
                    word |= indices[src] << (i * usize::from(bits));
                }
            }
            packed[vec_idx * packed_dim + word_idx] = word;
        }
    }

    (packed, adjusted_norms, packed_dim)
}

fn reference_dequantize(
    packed: &[u32],
    norms: &[f32],
    head_dim: usize,
    bits: u8,
    packed_dim: usize,
) -> Vec<f32> {
    let seed = 0x5455_5242_4f51_5541_u64;
    let signs = turboquant::wht::generate_signs(head_dim, seed);
    let codebook = turboquant::codebook::Codebook::new(bits, head_dim);
    let values_per_word = match bits {
        3 => 10,
        4 => 8,
        _ => unreachable!(),
    };
    let vector_count = norms.len();
    let mut out = vec![0.0_f32; vector_count * head_dim];

    for vec_idx in 0..vector_count {
        let mut rotated = vec![0.0_f32; head_dim];
        for (d, slot) in rotated.iter_mut().enumerate() {
            let word = packed[vec_idx * packed_dim + d / values_per_word];
            let shift = (d % values_per_word) * usize::from(bits);
            let idx = ((word >> shift) & ((1_u32 << bits) - 1)) as u8;
            *slot = codebook.centroid(idx) * norms[vec_idx];
        }
        wht_inplace(&mut rotated);
        for d in 0..head_dim {
            out[vec_idx * head_dim + d] = rotated[d] * signs[d];
        }
    }

    out
}

#[test]
fn turbo_quantize_packs_3bit_indices_and_adjusted_norms() {
    let _guard = turboquant_test_lock();
    let b = 1_i32;
    let h = 2_i32;
    let s = 3_i32;
    let d = 64_i32;
    let bits = 3_u8;
    let data = test_values((b * h * s * d) as usize);
    let (expected_packed, expected_norms, packed_dim) = reference_quantize(&data, d as usize, bits);

    let input: Array = (data.as_slice(), &[b, h, s, d][..]).try_into().unwrap();
    let signs = turboquant::wht::generate_signs(d as usize, 0x5455_5242_4f51_5541_u64);
    let signs: Array = (signs.as_slice(), &[d][..]).try_into().unwrap();
    let codebook = turboquant::codebook::Codebook::new(bits, d as usize);
    let codebook: Array = (
        codebook.centroids.as_slice(),
        &[codebook.centroids.len() as i32][..],
    )
        .try_into()
        .unwrap();

    let (packed, norms) =
        mlx::fast::turbo_quantize(&input, &signs, &codebook, bits).expect("turbo quantize");

    assert_eq!(packed.dtype(), Dtype::Uint32);
    assert_eq!(norms.dtype(), Dtype::Float32);
    assert_eq!(packed.shape().as_slice(), &[b, h, s, packed_dim as i32]);
    assert_eq!(norms.shape().as_slice(), &[b, h, s]);
    assert_eq!(packed.to_vec::<u32>().unwrap(), expected_packed);

    let actual_norms = norms.to_vec::<f32>().unwrap();
    for (actual, expected) in actual_norms.iter().zip(expected_norms.iter()) {
        assert!(
            (actual - expected).abs() < 1.0e-5,
            "actual={actual} expected={expected}"
        );
    }
}

#[test]
fn turbo_quantize_packs_4bit_indices() {
    let _guard = turboquant_test_lock();
    let b = 1_i32;
    let h = 1_i32;
    let s = 2_i32;
    let d = 128_i32;
    let bits = 4_u8;
    let data = test_values((b * h * s * d) as usize);
    let (expected_packed, _expected_norms, packed_dim) =
        reference_quantize(&data, d as usize, bits);

    let input: Array = (data.as_slice(), &[b, h, s, d][..]).try_into().unwrap();
    let signs = turboquant::wht::generate_signs(d as usize, 0x5455_5242_4f51_5541_u64);
    let signs: Array = (signs.as_slice(), &[d][..]).try_into().unwrap();
    let codebook = turboquant::codebook::Codebook::new(bits, d as usize);
    let codebook: Array = (
        codebook.centroids.as_slice(),
        &[codebook.centroids.len() as i32][..],
    )
        .try_into()
        .unwrap();

    let (packed, norms) =
        mlx::fast::turbo_quantize(&input, &signs, &codebook, bits).expect("turbo quantize");

    assert_eq!(packed.shape().as_slice(), &[b, h, s, packed_dim as i32]);
    assert_eq!(norms.shape().as_slice(), &[b, h, s]);
    assert_eq!(packed.to_vec::<u32>().unwrap(), expected_packed);
}

#[test]
fn turbo_dequantize_reconstructs_3bit_vectors_from_packed_words() {
    let _guard = turboquant_test_lock();
    let b = 1_i32;
    let h = 2_i32;
    let s = 3_i32;
    let d = 64_i32;
    let bits = 3_u8;
    let data = test_values((b * h * s * d) as usize);
    let (expected_packed, expected_norms, packed_dim) = reference_quantize(&data, d as usize, bits);
    let expected = reference_dequantize(
        &expected_packed,
        &expected_norms,
        d as usize,
        bits,
        packed_dim,
    );

    let packed: Array = (
        expected_packed.as_slice(),
        &[b, h, s, packed_dim as i32][..],
    )
        .try_into()
        .unwrap();
    let norms: Array = (expected_norms.as_slice(), &[b, h, s][..])
        .try_into()
        .unwrap();
    let signs = turboquant::wht::generate_signs(d as usize, 0x5455_5242_4f51_5541_u64);
    let signs: Array = (signs.as_slice(), &[d][..]).try_into().unwrap();
    let codebook = turboquant::codebook::Codebook::new(bits, d as usize);
    let codebook: Array = (
        codebook.centroids.as_slice(),
        &[codebook.centroids.len() as i32][..],
    )
        .try_into()
        .unwrap();

    let actual =
        mlx::fast::turbo_dequantize(&packed, &norms, &signs, &codebook, bits, d, Dtype::Float32)
            .expect("turbo dequantize");

    assert_eq!(actual.dtype(), Dtype::Float32);
    assert_eq!(actual.shape().as_slice(), &[b, h, s, d]);
    let actual = actual.to_vec::<f32>().unwrap();
    for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1.0e-5,
            "idx={idx} actual={actual} expected={expected}"
        );
    }
}

#[test]
fn turbo_dequantize_supports_bfloat16_output_dtype() {
    let _guard = turboquant_test_lock();
    let b = 1_i32;
    let h = 1_i32;
    let s = 2_i32;
    let d = 64_i32;
    let bits = 3_u8;
    let data = test_values((b * h * s * d) as usize);
    let (expected_packed, expected_norms, packed_dim) = reference_quantize(&data, d as usize, bits);

    let packed: Array = (
        expected_packed.as_slice(),
        &[b, h, s, packed_dim as i32][..],
    )
        .try_into()
        .unwrap();
    let norms: Array = (expected_norms.as_slice(), &[b, h, s][..])
        .try_into()
        .unwrap();
    let signs = turboquant::wht::generate_signs(d as usize, 0x5455_5242_4f51_5541_u64);
    let signs: Array = (signs.as_slice(), &[d][..]).try_into().unwrap();
    let codebook = turboquant::codebook::Codebook::new(bits, d as usize);
    let codebook: Array = (
        codebook.centroids.as_slice(),
        &[codebook.centroids.len() as i32][..],
    )
        .try_into()
        .unwrap();

    let actual =
        mlx::fast::turbo_dequantize(&packed, &norms, &signs, &codebook, bits, d, Dtype::Bfloat16)
            .expect("turbo dequantize bf16");

    assert_eq!(actual.dtype(), Dtype::Bfloat16);
    assert_eq!(actual.shape().as_slice(), &[b, h, s, d]);
    let values = actual.to_vec::<half::bf16>().unwrap();
    assert_eq!(values.len(), (b * h * s * d) as usize);
}

#[test]
fn turboquant_sdpa_decode_matches_dense_materialized_reference() {
    let _guard = turboquant_test_lock();
    let b = 1_i32;
    let h_q = 2_i32;
    let h_kv = 1_i32;
    let s = 5_i32;
    let d = 64_i32;
    let k_bits = 3_u8;
    let v_bits = 4_u8;
    let scale = (d as f32).sqrt().recip();

    let q_data = test_values((b * h_q * d) as usize);
    let k_data = test_values((b * h_kv * s * d) as usize)
        .into_iter()
        .map(|v| v * 0.8)
        .collect::<Vec<_>>();
    let v_data = test_values((b * h_kv * s * d) as usize)
        .into_iter()
        .map(|v| v * 1.1 + 0.05)
        .collect::<Vec<_>>();

    let q: Array = (q_data.as_slice(), &[b, h_q, 1_i32, d][..])
        .try_into()
        .unwrap();
    let k: Array = (k_data.as_slice(), &[b, h_kv, s, d][..])
        .try_into()
        .unwrap();
    let v: Array = (v_data.as_slice(), &[b, h_kv, s, d][..])
        .try_into()
        .unwrap();

    let k_signs = turboquant::wht::generate_signs(d as usize, 0x5455_5242_4f51_5541_u64);
    let k_signs: Array = (k_signs.as_slice(), &[d][..]).try_into().unwrap();
    let v_signs = turboquant::wht::generate_signs(d as usize, 0x5455_5242_4f51_5541_u64);
    let v_signs: Array = (v_signs.as_slice(), &[d][..]).try_into().unwrap();
    let k_codebook = turboquant::codebook::Codebook::new(k_bits, d as usize);
    let k_codebook: Array = (
        k_codebook.centroids.as_slice(),
        &[k_codebook.centroids.len() as i32][..],
    )
        .try_into()
        .unwrap();
    let v_codebook = turboquant::codebook::Codebook::new(v_bits, d as usize);
    let v_codebook: Array = (
        v_codebook.centroids.as_slice(),
        &[v_codebook.centroids.len() as i32][..],
    )
        .try_into()
        .unwrap();

    let (k_packed, k_norms) =
        mlx::fast::turbo_quantize(&k, &k_signs, &k_codebook, k_bits).expect("quantize k");
    let (v_packed, v_norms) =
        mlx::fast::turbo_quantize(&v, &v_signs, &v_codebook, v_bits).expect("quantize v");
    let mask_data = vec![0.0_f32; (b * s) as usize];
    let mask: Array = (mask_data.as_slice(), &[b, 1_i32, 1_i32, s][..])
        .try_into()
        .unwrap();

    let actual = mlx::fast::turboquant_sdpa_decode(
        &q,
        &k_packed,
        &k_norms,
        &v_packed,
        &v_norms,
        &k_signs,
        &k_codebook,
        &v_signs,
        &v_codebook,
        scale,
        k_bits,
        v_bits,
        Some(&mask),
        Dtype::Float32,
    )
    .expect("turboquant sdpa decode");

    let k_dense = mlx::fast::turbo_dequantize(
        &k_packed,
        &k_norms,
        &k_signs,
        &k_codebook,
        k_bits,
        d,
        Dtype::Float32,
    )
    .expect("dequantize k");
    let v_dense = mlx::fast::turbo_dequantize(
        &v_packed,
        &v_norms,
        &v_signs,
        &v_codebook,
        v_bits,
        d,
        Dtype::Float32,
    )
    .expect("dequantize v");
    let expected = mlx::fast::scaled_dot_product_attention(
        &q,
        &k_dense,
        &v_dense,
        scale,
        "",
        Some(&mask),
        None,
    )
    .expect("dense sdpa reference");

    assert_eq!(actual.dtype(), Dtype::Float32);
    assert_eq!(actual.shape().as_slice(), &[b, h_q, 1_i32, d]);
    let actual = actual.to_vec::<f32>().unwrap();
    let expected = expected.to_vec::<f32>().unwrap();
    for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1.0e-3,
            "idx={idx} actual={actual} expected={expected}"
        );
    }
}

#[test]
fn turboquant_sdpa_decode_parallel_matches_dense_materialized_reference() {
    let _guard = turboquant_test_lock();
    let b = 1_i32;
    let h_q = 2_i32;
    let h_kv = 1_i32;
    let s = 513_i32;
    let d = 64_i32;
    let k_bits = 3_u8;
    let v_bits = 4_u8;
    let scale = (d as f32).sqrt().recip();

    let q_data = test_values((b * h_q * d) as usize);
    let k_data = test_values((b * h_kv * s * d) as usize)
        .into_iter()
        .map(|v| v * 0.8)
        .collect::<Vec<_>>();
    let v_data = test_values((b * h_kv * s * d) as usize)
        .into_iter()
        .map(|v| v * 1.1 + 0.05)
        .collect::<Vec<_>>();

    let q: Array = (q_data.as_slice(), &[b, h_q, 1_i32, d][..])
        .try_into()
        .unwrap();
    let k: Array = (k_data.as_slice(), &[b, h_kv, s, d][..])
        .try_into()
        .unwrap();
    let v: Array = (v_data.as_slice(), &[b, h_kv, s, d][..])
        .try_into()
        .unwrap();

    let k_signs = turboquant::wht::generate_signs(d as usize, 0x5455_5242_4f51_5541_u64);
    let k_signs: Array = (k_signs.as_slice(), &[d][..]).try_into().unwrap();
    let v_signs = turboquant::wht::generate_signs(d as usize, 0x5455_5242_4f51_5541_u64);
    let v_signs: Array = (v_signs.as_slice(), &[d][..]).try_into().unwrap();
    let k_codebook = turboquant::codebook::Codebook::new(k_bits, d as usize);
    let k_codebook: Array = (
        k_codebook.centroids.as_slice(),
        &[k_codebook.centroids.len() as i32][..],
    )
        .try_into()
        .unwrap();
    let v_codebook = turboquant::codebook::Codebook::new(v_bits, d as usize);
    let v_codebook: Array = (
        v_codebook.centroids.as_slice(),
        &[v_codebook.centroids.len() as i32][..],
    )
        .try_into()
        .unwrap();

    let (k_packed, k_norms) =
        mlx::fast::turbo_quantize(&k, &k_signs, &k_codebook, k_bits).expect("quantize k");
    let (v_packed, v_norms) =
        mlx::fast::turbo_quantize(&v, &v_signs, &v_codebook, v_bits).expect("quantize v");
    let mask_data = vec![0.0_f32; (b * s) as usize];
    let mask: Array = (mask_data.as_slice(), &[b, 1_i32, 1_i32, s][..])
        .try_into()
        .unwrap();

    let actual = mlx::fast::turboquant_sdpa_decode_parallel(
        &q,
        &k_packed,
        &k_norms,
        &v_packed,
        &v_norms,
        &k_signs,
        &k_codebook,
        &v_signs,
        &v_codebook,
        scale,
        k_bits,
        v_bits,
        Some(&mask),
        Dtype::Float32,
    )
    .expect("parallel turboquant sdpa decode");

    let k_dense = mlx::fast::turbo_dequantize(
        &k_packed,
        &k_norms,
        &k_signs,
        &k_codebook,
        k_bits,
        d,
        Dtype::Float32,
    )
    .expect("dequantize k");
    let v_dense = mlx::fast::turbo_dequantize(
        &v_packed,
        &v_norms,
        &v_signs,
        &v_codebook,
        v_bits,
        d,
        Dtype::Float32,
    )
    .expect("dequantize v");
    let expected = mlx::fast::scaled_dot_product_attention(
        &q,
        &k_dense,
        &v_dense,
        scale,
        "",
        Some(&mask),
        None,
    )
    .expect("dense sdpa reference");

    let actual_values = actual.to_vec::<f32>().unwrap();
    let expected_values = expected.to_vec::<f32>().unwrap();
    assert_eq!(actual_values.len(), expected_values.len());
    for (i, (got, want)) in actual_values.iter().zip(expected_values.iter()).enumerate() {
        let diff = (got - want).abs();
        assert!(
            diff < 2.5e-2,
            "parallel sdpa mismatch at {i}: got {got}, want {want}, diff {diff}"
        );
    }
}

#[test]
fn turboquant_sdpa_decode_serial_preserves_identical_batch_rows() {
    let _guard = turboquant_test_lock();
    let b = 4_i32;
    let h_q = 8_i32;
    let h_kv = 2_i32;
    let s = 121_i32;
    let d = 512_i32;
    let k_bits = 3_u8;
    let v_bits = 4_u8;
    let scale = (d as f32).sqrt().recip();

    let q_row = test_values((h_q * d) as usize);
    let k_row = test_values((h_kv * s * d) as usize);
    let v_row = test_values((h_kv * s * d) as usize);
    let q_data = q_row.repeat(b as usize);
    let k_data = k_row.repeat(b as usize);
    let v_data = v_row.repeat(b as usize);
    let q: Array = (q_data.as_slice(), &[b, h_q, 1_i32, d][..])
        .try_into()
        .unwrap();
    let k: Array = (k_data.as_slice(), &[b, h_kv, s, d][..])
        .try_into()
        .unwrap();
    let v: Array = (v_data.as_slice(), &[b, h_kv, s, d][..])
        .try_into()
        .unwrap();

    let seed = 0x5455_5242_4f51_5541_u64;
    let k_signs = turboquant::wht::generate_signs(d as usize, seed);
    let k_signs: Array = (k_signs.as_slice(), &[d][..]).try_into().unwrap();
    let v_signs = turboquant::wht::generate_signs(d as usize, seed);
    let v_signs: Array = (v_signs.as_slice(), &[d][..]).try_into().unwrap();
    let k_codebook = turboquant::codebook::Codebook::new(k_bits, d as usize);
    let k_codebook: Array = (
        k_codebook.centroids.as_slice(),
        &[k_codebook.centroids.len() as i32][..],
    )
        .try_into()
        .unwrap();
    let v_codebook = turboquant::codebook::Codebook::new(v_bits, d as usize);
    let v_codebook: Array = (
        v_codebook.centroids.as_slice(),
        &[v_codebook.centroids.len() as i32][..],
    )
        .try_into()
        .unwrap();
    let (k_packed, k_norms) =
        mlx::fast::turbo_quantize(&k, &k_signs, &k_codebook, k_bits).expect("quantize k");
    let (v_packed, v_norms) =
        mlx::fast::turbo_quantize(&v, &v_signs, &v_codebook, v_bits).expect("quantize v");
    let mask_data = vec![0.0_f32; (b * s) as usize];
    let mask: Array = (mask_data.as_slice(), &[b, 1_i32, 1_i32, s][..])
        .try_into()
        .unwrap();

    let actual = mlx::fast::turboquant_sdpa_decode(
        &q,
        &k_packed,
        &k_norms,
        &v_packed,
        &v_norms,
        &k_signs,
        &k_codebook,
        &v_signs,
        &v_codebook,
        scale,
        k_bits,
        v_bits,
        Some(&mask),
        Dtype::Float32,
    )
    .expect("serial turboquant sdpa decode");

    let values = actual.to_vec::<f32>().unwrap();
    let row_size = (h_q * d) as usize;
    for row in 1..b as usize {
        assert_eq!(
            &values[row * row_size..(row + 1) * row_size],
            &values[..row_size],
            "serial batch row {row} differs"
        );
    }
}

#[test]
fn turboquant_sdpa_decode_parallel_preserves_identical_batch_rows() {
    let _guard = turboquant_test_lock();
    let b = 4_i32;
    let h_q = 8_i32;
    let h_kv = 2_i32;
    let s = 193_i32;
    let d = 512_i32;
    let k_bits = 3_u8;
    let v_bits = 4_u8;
    let scale = (d as f32).sqrt().recip();

    let q_row = test_values((h_q * d) as usize);
    let k_row = test_values((h_kv * s * d) as usize)
        .into_iter()
        .map(|value| value * 0.8)
        .collect::<Vec<_>>();
    let v_row = test_values((h_kv * s * d) as usize)
        .into_iter()
        .map(|value| value * 1.1 + 0.05)
        .collect::<Vec<_>>();
    let q_data = q_row.repeat(b as usize);
    let k_data = k_row.repeat(b as usize);
    let v_data = v_row.repeat(b as usize);

    let q: Array = (q_data.as_slice(), &[b, h_q, 1_i32, d][..])
        .try_into()
        .unwrap();
    let k: Array = (k_data.as_slice(), &[b, h_kv, s, d][..])
        .try_into()
        .unwrap();
    let v: Array = (v_data.as_slice(), &[b, h_kv, s, d][..])
        .try_into()
        .unwrap();

    let seed = 0x5455_5242_4f51_5541_u64;
    let k_signs = turboquant::wht::generate_signs(d as usize, seed);
    let k_signs: Array = (k_signs.as_slice(), &[d][..]).try_into().unwrap();
    let v_signs = turboquant::wht::generate_signs(d as usize, seed);
    let v_signs: Array = (v_signs.as_slice(), &[d][..]).try_into().unwrap();
    let k_codebook = turboquant::codebook::Codebook::new(k_bits, d as usize);
    let k_codebook: Array = (
        k_codebook.centroids.as_slice(),
        &[k_codebook.centroids.len() as i32][..],
    )
        .try_into()
        .unwrap();
    let v_codebook = turboquant::codebook::Codebook::new(v_bits, d as usize);
    let v_codebook: Array = (
        v_codebook.centroids.as_slice(),
        &[v_codebook.centroids.len() as i32][..],
    )
        .try_into()
        .unwrap();

    let (k_packed, k_norms) =
        mlx::fast::turbo_quantize(&k, &k_signs, &k_codebook, k_bits).expect("quantize k");
    let (v_packed, v_norms) =
        mlx::fast::turbo_quantize(&v, &v_signs, &v_codebook, v_bits).expect("quantize v");
    let mask_data = vec![0.0_f32; (b * s) as usize];
    let mask: Array = (mask_data.as_slice(), &[b, 1_i32, 1_i32, s][..])
        .try_into()
        .unwrap();

    let actual = mlx::fast::turboquant_sdpa_decode_parallel(
        &q,
        &k_packed,
        &k_norms,
        &v_packed,
        &v_norms,
        &k_signs,
        &k_codebook,
        &v_signs,
        &v_codebook,
        scale,
        k_bits,
        v_bits,
        Some(&mask),
        Dtype::Float32,
    )
    .expect("parallel turboquant sdpa decode");

    let q_rot_data = reference_query_rotate(&q_data, d as usize, &k_signs.to_vec().unwrap());
    let q_rot: Array = (q_rot_data.as_slice(), &[b, h_q, d][..])
        .try_into()
        .unwrap();
    let pre_rotated = mlx::fast::turboquant_sdpa_decode_parallel_pre_rotated(
        &q_rot,
        &k_packed,
        &k_norms,
        &v_packed,
        &v_norms,
        &k_codebook,
        &v_signs,
        &v_codebook,
        scale,
        k_bits,
        v_bits,
        Some(&mask),
        Dtype::Float32,
    )
    .expect("pre-rotated parallel turboquant sdpa decode");

    let values = actual.to_vec::<f32>().unwrap();
    let pre_rotated_values = pre_rotated.to_vec::<f32>().unwrap();
    let row_size = (h_q * d) as usize;
    let first = &values[..row_size];
    let pre_rotated_first = &pre_rotated_values[..row_size];
    for row in 1..b as usize {
        let row_values = &values[row * row_size..(row + 1) * row_size];
        let pre_rotated_row = &pre_rotated_values[row * row_size..(row + 1) * row_size];
        for (idx, (got, want)) in pre_rotated_row
            .iter()
            .zip(pre_rotated_first.iter())
            .enumerate()
        {
            assert_eq!(
                got, want,
                "pre-rotated batch row {row} differs at output index {idx}"
            );
        }
        for (idx, (got, want)) in row_values.iter().zip(first.iter()).enumerate() {
            assert_eq!(got, want, "batch row {row} differs at output index {idx}");
        }
    }
}

#[test]
fn turbo_quantize_preserves_identical_large_batch_rows() {
    let _guard = turboquant_test_lock();
    let b = 4_i32;
    let h = 2_i32;
    let s = 193_i32;
    let d = 512_i32;
    let row = test_values((h * s * d) as usize);
    let data = row.repeat(b as usize);
    let input: Array = (data.as_slice(), &[b, h, s, d][..]).try_into().unwrap();

    let seed = 0x5455_5242_4f51_5541_u64;
    let signs = turboquant::wht::generate_signs(d as usize, seed);
    let signs: Array = (signs.as_slice(), &[d][..]).try_into().unwrap();

    for bits in [3_u8, 4_u8] {
        let (expected_packed, expected_norms, _) = reference_quantize(&row, d as usize, bits);
        let codebook = turboquant::codebook::Codebook::new(bits, d as usize);
        let codebook: Array = (
            codebook.centroids.as_slice(),
            &[codebook.centroids.len() as i32][..],
        )
            .try_into()
            .unwrap();
        let (packed, norms) =
            mlx::fast::turbo_quantize(&input, &signs, &codebook, bits).expect("turbo quantize");
        let packed = packed.to_vec::<u32>().unwrap();
        let norms = norms.to_vec::<f32>().unwrap();

        let packed_row_size = packed.len() / b as usize;
        let norm_row_size = norms.len() / b as usize;
        assert_eq!(&packed[..packed_row_size], expected_packed);
        for (actual, expected) in norms[..norm_row_size].iter().zip(expected_norms.iter()) {
            assert!(
                (actual - expected).abs() < 5.0e-5,
                "{bits}-bit norm mismatch: actual={actual}, expected={expected}"
            );
        }
        for batch in 1..b as usize {
            assert_eq!(
                &packed[batch * packed_row_size..(batch + 1) * packed_row_size],
                &packed[..packed_row_size],
                "{bits}-bit packed row {batch} differs"
            );
            assert_eq!(
                &norms[batch * norm_row_size..(batch + 1) * norm_row_size],
                &norms[..norm_row_size],
                "{bits}-bit norm row {batch} differs"
            );
        }
    }
}

#[test]
fn turboquant_sdpa_decode_parallel_pre_rotated_matches_regular_parallel() {
    let _guard = turboquant_test_lock();
    let b = 1_i32;
    let h_q = 2_i32;
    let h_kv = 1_i32;
    let s = 513_i32;
    let d = 64_i32;
    let k_bits = 3_u8;
    let v_bits = 4_u8;
    let scale = (d as f32).sqrt().recip();

    let q_data = test_values((b * h_q * d) as usize);
    let k_data = test_values((b * h_kv * s * d) as usize)
        .into_iter()
        .map(|v| v * 0.8)
        .collect::<Vec<_>>();
    let v_data = test_values((b * h_kv * s * d) as usize)
        .into_iter()
        .map(|v| v * 1.1 + 0.05)
        .collect::<Vec<_>>();

    let q: Array = (q_data.as_slice(), &[b, h_q, 1_i32, d][..])
        .try_into()
        .unwrap();
    let k: Array = (k_data.as_slice(), &[b, h_kv, s, d][..])
        .try_into()
        .unwrap();
    let v: Array = (v_data.as_slice(), &[b, h_kv, s, d][..])
        .try_into()
        .unwrap();

    let k_signs = turboquant::wht::generate_signs(d as usize, 0x5455_5242_4f51_5541_u64);
    let q_rot_data = reference_query_rotate(&q_data, d as usize, &k_signs);
    let q_rot: Array = (q_rot_data.as_slice(), &[b, h_q, d][..])
        .try_into()
        .unwrap();
    let k_signs: Array = (k_signs.as_slice(), &[d][..]).try_into().unwrap();
    let v_signs = turboquant::wht::generate_signs(d as usize, 0x5455_5242_4f51_5541_u64);
    let v_signs: Array = (v_signs.as_slice(), &[d][..]).try_into().unwrap();
    let k_codebook = turboquant::codebook::Codebook::new(k_bits, d as usize);
    let k_codebook: Array = (
        k_codebook.centroids.as_slice(),
        &[k_codebook.centroids.len() as i32][..],
    )
        .try_into()
        .unwrap();
    let v_codebook = turboquant::codebook::Codebook::new(v_bits, d as usize);
    let v_codebook: Array = (
        v_codebook.centroids.as_slice(),
        &[v_codebook.centroids.len() as i32][..],
    )
        .try_into()
        .unwrap();

    let (k_packed, k_norms) =
        mlx::fast::turbo_quantize(&k, &k_signs, &k_codebook, k_bits).expect("quantize k");
    let (v_packed, v_norms) =
        mlx::fast::turbo_quantize(&v, &v_signs, &v_codebook, v_bits).expect("quantize v");
    let mask_data = vec![0.0_f32; (b * s) as usize];
    let mask: Array = (mask_data.as_slice(), &[b, 1_i32, 1_i32, s][..])
        .try_into()
        .unwrap();

    let expected = mlx::fast::turboquant_sdpa_decode_parallel(
        &q,
        &k_packed,
        &k_norms,
        &v_packed,
        &v_norms,
        &k_signs,
        &k_codebook,
        &v_signs,
        &v_codebook,
        scale,
        k_bits,
        v_bits,
        Some(&mask),
        Dtype::Float32,
    )
    .expect("regular parallel turboquant sdpa decode");
    let actual = mlx::fast::turboquant_sdpa_decode_parallel_pre_rotated(
        &q_rot,
        &k_packed,
        &k_norms,
        &v_packed,
        &v_norms,
        &k_codebook,
        &v_signs,
        &v_codebook,
        scale,
        k_bits,
        v_bits,
        Some(&mask),
        Dtype::Float32,
    )
    .expect("pre-rotated parallel turboquant sdpa decode");

    let actual_values = actual.to_vec::<f32>().unwrap();
    let expected_values = expected.to_vec::<f32>().unwrap();
    assert_eq!(actual_values.len(), expected_values.len());
    for (i, (got, want)) in actual_values.iter().zip(expected_values.iter()).enumerate() {
        let diff = (got - want).abs();
        assert!(
            diff < 1.0e-5,
            "pre-rotated sdpa mismatch at {i}: got {got}, want {want}, diff {diff}"
        );
    }
}

#[test]
fn turboquant_sdpa_multirow_matches_dense_causal_reference() {
    let _guard = turboquant_test_lock();
    let b = 1_i32;
    let h_q = 4_i32;
    let h_kv = 1_i32;
    let q_rows = 3_i32;
    let s = 67_i32;
    let d = 64_i32;
    let k_bits = 3_u8;
    let v_bits = 4_u8;
    let scale = (d as f32).sqrt().recip();

    let q_data = test_values((b * h_q * q_rows * d) as usize);
    let k_data = test_values((b * h_kv * s * d) as usize)
        .into_iter()
        .map(|value| value * 0.8)
        .collect::<Vec<_>>();
    let v_data = test_values((b * h_kv * s * d) as usize)
        .into_iter()
        .map(|value| value * 1.1 + 0.05)
        .collect::<Vec<_>>();
    let q: Array = (q_data.as_slice(), &[b, h_q, q_rows, d][..])
        .try_into()
        .unwrap();
    let k: Array = (k_data.as_slice(), &[b, h_kv, s, d][..])
        .try_into()
        .unwrap();
    let v: Array = (v_data.as_slice(), &[b, h_kv, s, d][..])
        .try_into()
        .unwrap();

    let seed = 0x5455_5242_4f51_5541_u64;
    let k_signs = turboquant::wht::generate_signs(d as usize, seed);
    let k_signs: Array = (k_signs.as_slice(), &[d][..]).try_into().unwrap();
    let v_signs = turboquant::wht::generate_signs(d as usize, seed);
    let v_signs: Array = (v_signs.as_slice(), &[d][..]).try_into().unwrap();
    let k_codebook = turboquant::codebook::Codebook::new(k_bits, d as usize);
    let k_codebook: Array = (
        k_codebook.centroids.as_slice(),
        &[k_codebook.centroids.len() as i32][..],
    )
        .try_into()
        .unwrap();
    let v_codebook = turboquant::codebook::Codebook::new(v_bits, d as usize);
    let v_codebook: Array = (
        v_codebook.centroids.as_slice(),
        &[v_codebook.centroids.len() as i32][..],
    )
        .try_into()
        .unwrap();

    let (k_packed, k_norms) =
        mlx::fast::turbo_quantize(&k, &k_signs, &k_codebook, k_bits).expect("quantize k");
    let (v_packed, v_norms) =
        mlx::fast::turbo_quantize(&v, &v_signs, &v_codebook, v_bits).expect("quantize v");
    let query_lens: Array = (&[q_rows][..], &[b][..]).try_into().unwrap();
    let kv_lens: Array = (&[s][..], &[b][..]).try_into().unwrap();
    let actual = mlx::fast::turboquant_sdpa_multirow(
        &q,
        &k_packed,
        &k_norms,
        &v_packed,
        &v_norms,
        &k_signs,
        &k_codebook,
        &v_signs,
        &v_codebook,
        scale,
        k_bits,
        v_bits,
        &query_lens,
        &kv_lens,
        None,
        Dtype::Float32,
    )
    .expect("multi-row turboquant sdpa");
    let mask_data = vec![0.0_f32; (b * s) as usize];
    let broadcast_mask: Array = (mask_data.as_slice(), &[b, 1_i32, 1_i32, s][..])
        .try_into()
        .unwrap();
    let masked_actual = mlx::fast::turboquant_sdpa_multirow(
        &q,
        &k_packed,
        &k_norms,
        &v_packed,
        &v_norms,
        &k_signs,
        &k_codebook,
        &v_signs,
        &v_codebook,
        scale,
        k_bits,
        v_bits,
        &query_lens,
        &kv_lens,
        Some(&broadcast_mask),
        Dtype::Float32,
    )
    .expect("masked multi-row turboquant sdpa");

    let k_dense = mlx::fast::turbo_dequantize(
        &k_packed,
        &k_norms,
        &k_signs,
        &k_codebook,
        k_bits,
        d,
        Dtype::Float32,
    )
    .expect("dequantize k");
    let v_dense = mlx::fast::turbo_dequantize(
        &v_packed,
        &v_norms,
        &v_signs,
        &v_codebook,
        v_bits,
        d,
        Dtype::Float32,
    )
    .expect("dequantize v");
    let expected = mlx::fast::scaled_dot_product_attention(
        &q, &k_dense, &v_dense, scale, "causal", None, None,
    )
    .expect("dense causal sdpa reference");

    assert_eq!(actual.shape().as_slice(), &[b, h_q, q_rows, d]);
    let actual_values = actual.to_vec::<f32>().unwrap();
    let masked_actual_values = masked_actual.to_vec::<f32>().unwrap();
    let expected_values = expected.to_vec::<f32>().unwrap();
    for (idx, (actual, expected)) in actual_values.iter().zip(expected_values.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff < 2.5e-2,
            "multi-row sdpa mismatch at {idx}: got {actual}, want {expected}, diff {diff}"
        );
    }
    for (idx, (actual, expected)) in masked_actual_values
        .iter()
        .zip(expected_values.iter())
        .enumerate()
    {
        let diff = (actual - expected).abs();
        assert!(
            diff < 2.5e-2,
            "masked multi-row sdpa mismatch at {idx}: got {actual}, want {expected}, diff {diff}"
        );
    }
}
