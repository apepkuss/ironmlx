use std::time::Instant;

use mlx::transforms::eval;
use mlx::{fast, random, Array, Dtype};

fn reference_decode(
    q: &[f32],
    k_pages: &[f32],
    v_pages: &[f32],
    block_table: &[i32],
    lengths: &[i32],
    shape: RefShape,
) -> Vec<f32> {
    let RefShape {
        batch,
        q_heads,
        kv_heads,
        head_dim,
        block_size,
        max_blocks,
        ..
    } = shape;
    let pages_stride = kv_heads * block_size * head_dim;
    let head_stride = block_size * head_dim;
    let q_group = q_heads / kv_heads;
    let mut out = vec![0.0_f32; batch * q_heads * head_dim];

    for b in 0..batch {
        let len = lengths[b] as usize;
        for qh in 0..q_heads {
            let kvh = qh / q_group;
            let mut scores = vec![0.0_f32; len];
            let mut max_score = f32::NEG_INFINITY;
            for t in 0..len {
                let page_col = t / block_size;
                let in_page = t % block_size;
                let page_id = block_table[b * max_blocks + page_col] as usize;
                let mut dot = 0.0_f32;
                for d in 0..head_dim {
                    let q_idx = ((b * q_heads + qh) * head_dim) + d;
                    let k_idx = page_id * pages_stride + kvh * head_stride + in_page * head_dim + d;
                    dot += q[q_idx] * k_pages[k_idx];
                }
                scores[t] = dot * shape.scale;
                max_score = max_score.max(scores[t]);
            }

            let mut denom = 0.0_f32;
            for score in &mut scores {
                *score = (*score - max_score).exp();
                denom += *score;
            }

            for d in 0..head_dim {
                let mut acc = 0.0_f32;
                for t in 0..len {
                    let page_col = t / block_size;
                    let in_page = t % block_size;
                    let page_id = block_table[b * max_blocks + page_col] as usize;
                    let v_idx = page_id * pages_stride + kvh * head_stride + in_page * head_dim + d;
                    acc += (scores[t] / denom) * v_pages[v_idx];
                }
                out[(b * q_heads + qh) * head_dim + d] = acc;
            }
        }
    }

    out
}

#[derive(Clone, Copy)]
struct RefShape {
    batch: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    max_blocks: usize,
    scale: f32,
}

fn assert_close(got: &[f32], expected: &[f32]) {
    assert_eq!(got.len(), expected.len());
    for (idx, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-4,
            "idx {idx}: got {g}, expected {e}, diff {}",
            (g - e).abs()
        );
    }
}

#[test]
fn paged_decode_b1_matches_reference() {
    let shape = RefShape {
        batch: 1,
        q_heads: 2,
        kv_heads: 1,
        head_dim: 4,
        block_size: 2,
        max_blocks: 3,
        scale: 0.5,
    };
    let q_data: Vec<f32> = (0..shape.batch * shape.q_heads * shape.head_dim)
        .map(|i| (i as f32 + 1.0) * 0.02)
        .collect();
    let k_pages_data: Vec<f32> =
        (0..shape.max_blocks * shape.kv_heads * shape.block_size * shape.head_dim)
            .map(|i| (i as f32 + 1.0) * 0.03)
            .collect();
    let v_pages_data: Vec<f32> =
        (0..shape.max_blocks * shape.kv_heads * shape.block_size * shape.head_dim)
            .map(|i| 1.0 + (i as f32) * 0.01)
            .collect();
    let block_table_data = vec![0_i32, 1, 2];
    let lengths_data = vec![5_i32];

    let q: Array = (&q_data[..], &[1_i32, 2, 1, 4][..]).try_into().unwrap();
    let k_pages: Array = (&k_pages_data[..], &[3_i32, 1, 2, 4][..])
        .try_into()
        .unwrap();
    let v_pages: Array = (&v_pages_data[..], &[3_i32, 1, 2, 4][..])
        .try_into()
        .unwrap();
    let block_table: Array = (&block_table_data[..], &[1_i32, 3][..]).try_into().unwrap();
    let lengths: Array = (&lengths_data[..], &[1_i32][..]).try_into().unwrap();

    let out = fast::paged_scaled_dot_product_attention_decode(
        &q,
        &k_pages,
        &v_pages,
        &block_table,
        &lengths,
        shape.scale,
        shape.block_size as i32,
    )
    .expect("paged decode");
    assert_eq!(out.shape().as_slice(), &[1, 2, 1, 4]);

    let expected = reference_decode(
        &q_data,
        &k_pages_data,
        &v_pages_data,
        &block_table_data,
        &lengths_data,
        shape,
    );
    let got = out.to_vec::<f32>().unwrap();
    assert_close(&got, &expected);
}

#[test]
fn paged_decode_b2_ragged_matches_reference() {
    let shape = RefShape {
        batch: 2,
        q_heads: 4,
        kv_heads: 2,
        head_dim: 4,
        block_size: 2,
        max_blocks: 3,
        scale: 0.5,
    };
    let page_count = 6;
    let q_data: Vec<f32> = (0..shape.batch * shape.q_heads * shape.head_dim)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.015)
        .collect();
    let k_pages_data: Vec<f32> =
        (0..page_count * shape.kv_heads * shape.block_size * shape.head_dim)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.02)
            .collect();
    let v_pages_data: Vec<f32> =
        (0..page_count * shape.kv_heads * shape.block_size * shape.head_dim)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.025)
            .collect();
    let block_table_data = vec![0_i32, 1, 2, 3, 4, 5];
    let lengths_data = vec![5_i32, 3];

    let q: Array = (&q_data[..], &[2_i32, 4, 1, 4][..]).try_into().unwrap();
    let k_pages: Array = (&k_pages_data[..], &[6_i32, 2, 2, 4][..])
        .try_into()
        .unwrap();
    let v_pages: Array = (&v_pages_data[..], &[6_i32, 2, 2, 4][..])
        .try_into()
        .unwrap();
    let block_table: Array = (&block_table_data[..], &[2_i32, 3][..]).try_into().unwrap();
    let lengths: Array = (&lengths_data[..], &[2_i32][..]).try_into().unwrap();

    let out = fast::paged_scaled_dot_product_attention_decode(
        &q,
        &k_pages,
        &v_pages,
        &block_table,
        &lengths,
        shape.scale,
        shape.block_size as i32,
    )
    .expect("paged decode");
    assert_eq!(out.shape().as_slice(), &[2, 4, 1, 4]);

    let expected = reference_decode(
        &q_data,
        &k_pages_data,
        &v_pages_data,
        &block_table_data,
        &lengths_data,
        shape,
    );
    let got = out.to_vec::<f32>().unwrap();
    assert_close(&got, &expected);
}

#[test]
fn paged_decode_parallel_b2_ragged_matches_reference() {
    let shape = RefShape {
        batch: 2,
        q_heads: 4,
        kv_heads: 2,
        head_dim: 8,
        block_size: 16,
        max_blocks: 9,
        scale: 0.25,
    };
    let page_count = shape.batch * shape.max_blocks;
    let q_data: Vec<f32> = (0..shape.batch * shape.q_heads * shape.head_dim)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.006)
        .collect();
    let k_pages_data: Vec<f32> =
        (0..page_count * shape.kv_heads * shape.block_size * shape.head_dim)
            .map(|i| ((i % 37) as f32 - 18.0) * 0.004)
            .collect();
    let v_pages_data: Vec<f32> =
        (0..page_count * shape.kv_heads * shape.block_size * shape.head_dim)
            .map(|i| ((i % 41) as f32 - 20.0) * 0.005)
            .collect();
    let block_table_data: Vec<i32> = (0..page_count as i32).collect();
    let lengths_data = vec![129_i32, 97];

    let q: Array = (&q_data[..], &[2_i32, 4, 1, 8][..]).try_into().unwrap();
    let k_pages: Array = (&k_pages_data[..], &[18_i32, 2, 16, 8][..])
        .try_into()
        .unwrap();
    let v_pages: Array = (&v_pages_data[..], &[18_i32, 2, 16, 8][..])
        .try_into()
        .unwrap();
    let block_table: Array = (&block_table_data[..], &[2_i32, 9][..]).try_into().unwrap();
    let lengths: Array = (&lengths_data[..], &[2_i32][..]).try_into().unwrap();

    let out = fast::paged_scaled_dot_product_attention_decode(
        &q,
        &k_pages,
        &v_pages,
        &block_table,
        &lengths,
        shape.scale,
        shape.block_size as i32,
    )
    .expect("paged decode");
    assert_eq!(out.shape().as_slice(), &[2, 4, 1, 8]);

    let expected = reference_decode(
        &q_data,
        &k_pages_data,
        &v_pages_data,
        &block_table_data,
        &lengths_data,
        shape,
    );
    let got = out.to_vec::<f32>().unwrap();
    assert_close(&got, &expected);
}

#[test]
#[ignore = "release-mode performance regression test; requires MLX_DIR and local Metal device"]
fn paged_decode_b1_qwen35_shape_stays_under_latency_budget() {
    fn median(mut values: Vec<f64>) -> f64 {
        values.sort_by(|a, b| a.partial_cmp(b).expect("finite timing"));
        values[values.len() / 2]
    }

    fn sample(shape: [i32; 4]) -> Array {
        random::normal()
            .shape(shape)
            .dtype(Dtype::Float16)
            .sample()
            .expect("normal")
    }

    let batch = 1;
    let q_heads = 16;
    let kv_heads = 4;
    let head_dim = 256;
    let seq_len = 8192;
    let block_size = 16;
    let max_blocks = seq_len / block_size;
    let page_count = batch * max_blocks;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q = sample([batch, q_heads, 1, head_dim]);
    let k_pages = sample([page_count, kv_heads, block_size, head_dim]);
    let v_pages = sample([page_count, kv_heads, block_size, head_dim]);
    let block_table_data: Vec<i32> = (0..max_blocks).collect();
    let lengths_data = vec![seq_len];
    let block_table: Array = (&block_table_data[..], &[batch, max_blocks][..])
        .try_into()
        .unwrap();
    let lengths: Array = (&lengths_data[..], &[batch][..]).try_into().unwrap();

    for _ in 0..2 {
        let out = fast::paged_scaled_dot_product_attention_decode(
            &q,
            &k_pages,
            &v_pages,
            &block_table,
            &lengths,
            scale,
            block_size,
        )
        .expect("paged decode warmup");
        eval(&[&out]).expect("eval warmup");
    }

    let mut latencies_ms = Vec::new();
    for _ in 0..7 {
        let start = Instant::now();
        let out = fast::paged_scaled_dot_product_attention_decode(
            &q,
            &k_pages,
            &v_pages,
            &block_table,
            &lengths,
            scale,
            block_size,
        )
        .expect("paged decode");
        eval(&[&out]).expect("eval paged decode");
        latencies_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    let median_ms = median(latencies_ms);
    println!("paged_decode_b1_qwen35_shape_median_ms={median_ms:.3}");
    assert!(
        median_ms < 1.0,
        "B=1 Qwen3.5-shaped paged decode should stay below 1ms per full-attention layer, got {median_ms:.3}ms"
    );
}
