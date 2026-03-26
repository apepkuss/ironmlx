//! Fused Paged Attention — single GPU dispatch for KV write + attention.
//!
//! One Metal kernel per layer that:
//! 1. Copies pool + writes new KV at target positions
//! 2. Computes paged attention via page table lookup
//! 3. Returns attention output + updated pool arrays

use crate::array::Array;
use crate::cache::paged_cache::{PageTable, PAGE_SIZE};
use crate::dtype::Dtype;
use crate::error::Result;
use crate::metal_kernel::{MetalKernel, MetalKernelConfig};
use crate::ops;
use crate::stream::Stream;

const FUSED_KERNEL_SOURCE: &str = include_str!("fused_paged_attention_kernel.metal");

/// Execute fused KV-write + paged attention for one layer.
///
/// Returns `(attn_out, k_pool_updated, v_pool_updated)`.
/// - `attn_out`: `[B, n_heads, 1, head_dim]`
/// - `k_pool_updated` / `v_pool_updated`: same shape as pool arrays with new KV written
#[allow(clippy::too_many_arguments)]
pub fn fused_paged_attention(
    q: &Array,
    new_k: &Array,
    new_v: &Array,
    k_pool: &Array,
    v_pool: &Array,
    page_tables: &[&PageTable],
    seq_lens: &[i32],
    _layer_idx: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    num_pages: usize,
    dtype: Dtype,
    stream: &Stream,
) -> Result<(Array, Array, Array)> {
    let batch_size = page_tables.len();
    let max_pages = page_tables.iter().map(|pt| pt.num_pages()).max().unwrap_or(0);
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Build padded page table: [B * max_pages]
    let mut pt_data: Vec<i32> = Vec::with_capacity(batch_size * max_pages);
    for pt in page_tables {
        let pages: Vec<i32> = pt.pages.iter().map(|&p| p as i32).collect();
        pt_data.extend_from_slice(&pages);
        pt_data.extend(std::iter::repeat_n(0i32, max_pages - pages.len()));
    }
    let page_table_arr = Array::from_slice_i32(&pt_data);

    let seq_lens_arr = Array::from_slice_i32(seq_lens);

    // Compute global write positions for each batch element
    let mut write_positions: Vec<i32> = Vec::with_capacity(batch_size);
    for b in 0..batch_size {
        let pos = seq_lens[b] as usize;
        let page_logical = pos / PAGE_SIZE;
        let page_offset = pos % PAGE_SIZE;
        let physical_page = page_tables[b].pages[page_logical];
        write_positions.push((physical_page * PAGE_SIZE + page_offset) as i32);
    }
    let write_pos_arr = Array::from_slice_i32(&write_positions);
    let max_pages_arr = Array::from_int(max_pages as i32);

    // Flatten inputs for kernel
    // q: [B, n_heads, 1, head_dim] → [B * n_heads * head_dim]
    let q_flat = ops::reshape(q, &[(batch_size * n_heads * head_dim) as i32], stream)?;
    // new_k/v: [B, n_kv_heads, 1, head_dim] → [B * n_kv_heads * head_dim]
    let k_flat = ops::reshape(new_k, &[(batch_size * n_kv_heads * head_dim) as i32], stream)?;
    let v_flat = ops::reshape(new_v, &[(batch_size * n_kv_heads * head_dim) as i32], stream)?;
    // pools: [num_pages, n_kv_heads, PAGE_SIZE, head_dim] → flatten
    let pool_elements = num_pages * n_kv_heads * PAGE_SIZE * head_dim;
    let k_pool_flat = ops::reshape(k_pool, &[pool_elements as i32], stream)?;
    let v_pool_flat = ops::reshape(v_pool, &[pool_elements as i32], stream)?;

    // Embed constants into shader source via unique placeholders
    let source = FUSED_KERNEL_SOURCE
        .replace("/*##N_HEADS##*/1", &n_heads.to_string())
        .replace("/*##N_KV_HEADS##*/1", &n_kv_heads.to_string())
        .replace("/*##HEAD_DIM##*/128", &head_dim.to_string())
        .replace("/*##PAGE_SIZE##*/256", &PAGE_SIZE.to_string())
        .replace("/*##NUM_PAGES##*/8", &num_pages.to_string())
        .replace("/*##SCALE##*/0.08838835", &format!("{:.8}", scale));

    let kernel_name = format!(
        "fused_paged_h{}_kv{}_hd{}_ps{}_np{}",
        n_heads, n_kv_heads, head_dim, PAGE_SIZE, num_pages
    );

    let kernel = MetalKernel::new(
        &kernel_name,
        &[
            "q",
            "new_k",
            "new_v",
            "k_pool",
            "v_pool",
            "page_table",
            "seq_lens",
            "write_pos",
            "max_pages_c",
        ],
        &["attn_out", "k_pool_out", "v_pool_out"],
        &source,
        "",
        true,
        false,
    );

    let config = MetalKernelConfig::new();

    // Output shapes (flattened)
    let attn_elements = batch_size * n_heads * head_dim;
    config.add_output(&[attn_elements as i32], dtype);
    config.add_output(&[pool_elements as i32], dtype);
    config.add_output(&[pool_elements as i32], dtype);

    // Grid: one thread per (batch, head)
    config.set_grid((batch_size * n_heads) as i32, 1, 1);
    config.set_thread_group(1, 1, 1);
    config.add_template_dtype("T", dtype);
    config.set_init_value(0.0);

    let outputs = kernel.apply(
        &[
            &q_flat,
            &k_flat,
            &v_flat,
            &k_pool_flat,
            &v_pool_flat,
            &page_table_arr,
            &seq_lens_arr,
            &write_pos_arr,
            &max_pages_arr,
        ],
        &config,
        stream,
    )?;

    if outputs.len() < 3 {
        return Err(crate::error::Error::Mlx(
            "fused_paged_attention: expected 3 outputs".to_string(),
        ));
    }

    // Reshape outputs
    let attn_out = ops::reshape(
        &outputs[0],
        &[batch_size as i32, n_heads as i32, 1, head_dim as i32],
        stream,
    )?;
    let k_pool_out = ops::reshape(
        &outputs[1],
        &[num_pages as i32, n_kv_heads as i32, PAGE_SIZE as i32, head_dim as i32],
        stream,
    )?;
    let v_pool_out = ops::reshape(
        &outputs[2],
        &[num_pages as i32, n_kv_heads as i32, PAGE_SIZE as i32, head_dim as i32],
        stream,
    )?;

    Ok((attn_out, k_pool_out, v_pool_out))
}
