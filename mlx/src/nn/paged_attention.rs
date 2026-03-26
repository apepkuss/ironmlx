//! Paged Attention — custom Metal kernel for page-table-based KV cache access.

use crate::array::Array;
use crate::cache::paged_cache::{PagePool, PageTable};
use crate::dtype::Dtype;
use crate::error::Result;
use crate::metal_kernel::{MetalKernel, MetalKernelConfig};
use crate::ops;
use crate::stream::Stream;

/// Metal shader source for paged attention (embedded at compile time).
const PAGED_ATTN_SOURCE: &str = include_str!("paged_attention_kernel.metal");

/// Execute paged attention for batched decode.
///
/// # Arguments
/// * `q` - Query tensor `[B, n_heads, 1, head_dim]`
/// * `pool` - Page pool containing all KV pages
/// * `page_tables` - Per-sequence page tables
/// * `seq_lens` - Actual sequence length per batch element
/// * `layer_idx` - Which layer's pages to use
/// * `n_heads` - Number of query heads
/// * `head_dim` - Dimension per head
/// * `stream` - MLX stream
///
/// # Returns
/// Attention output `[B, n_heads, 1, head_dim]`
#[allow(clippy::too_many_arguments)]
pub fn paged_attention(
    q: &Array,
    pool: &PagePool,
    page_tables: &[&PageTable],
    seq_lens: &[i32],
    layer_idx: usize,
    n_heads: usize,
    head_dim: usize,
    dtype: Dtype,
    stream: &Stream,
) -> Result<Array> {
    let batch_size = page_tables.len();
    let n_kv_heads = pool.n_kv_heads;
    let max_pages = page_tables
        .iter()
        .map(|pt| pt.num_pages())
        .max()
        .unwrap_or(0);
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Build padded page table: [B, max_pages]
    let mut pt_data: Vec<i32> = Vec::with_capacity(batch_size * max_pages);
    for pt in page_tables {
        let pages: Vec<i32> = pt.pages.iter().map(|&p| p as i32).collect();
        pt_data.extend_from_slice(&pages);
        // Pad with 0 (page 0 will be masked by seq_len)
        pt_data.extend(std::iter::repeat_n(0i32, max_pages - pages.len()));
    }
    let page_table_arr = Array::from_slice_i32(&pt_data);
    let page_table_2d = ops::reshape(
        &page_table_arr,
        &[batch_size as i32, max_pages as i32],
        stream,
    )?;

    let seq_lens_arr = Array::from_slice_i32(seq_lens);

    // Embed scalar constants directly into shader source via string replacement.
    // This avoids Metal address space issues with scalar array inputs.
    let source = PAGED_ATTN_SOURCE
        .replace("N_HEADS_VAL", &n_heads.to_string())
        .replace("N_KV_HEADS_VAL", &n_kv_heads.to_string())
        .replace("MAX_PAGES_VAL", &max_pages.to_string())
        .replace("NUM_PAGES_VAL", &pool.num_pages.to_string())
        .replace("SCALE_VAL", &format!("{:.8}", scale))
        .replace("HEAD_DIM_VAL", &head_dim.to_string())
        .replace("PAGE_SIZE_VAL", &crate::cache::paged_cache::PAGE_SIZE.to_string());

    // Unique kernel name per configuration (MLX caches compiled kernels by name)
    let kernel_name = format!(
        "paged_attn_h{}_kv{}_hd{}_p{}",
        n_heads, n_kv_heads, head_dim, pool.num_pages
    );

    let kernel = MetalKernel::new(
        &kernel_name,
        &["q", "k_pages", "v_pages", "page_table", "seq_lens"],
        &["out"],
        &source,
        "",
        true,
        false,
    );

    let config = MetalKernelConfig::new();

    let out_elements = batch_size * n_heads * head_dim;
    config.add_output(&[out_elements as i32], dtype);

    config.set_grid((batch_size * n_heads) as i32, 1, 1);
    config.set_thread_group(1, 1, 1);

    config.add_template_dtype("T", dtype);
    config.set_init_value(0.0);

    let outputs = kernel.apply(
        &[
            q,
            &pool.k_pages[layer_idx],
            &pool.v_pages[layer_idx],
            &page_table_2d,
            &seq_lens_arr,
        ],
        &config,
        stream,
    )?;

    // Reshape output from [B * n_heads * head_dim] to [B, n_heads, 1, head_dim]
    let flat_out = outputs
        .into_iter()
        .next()
        .ok_or_else(|| crate::error::Error::Mlx("paged_attention returned no output".to_string()))?;

    ops::reshape(
        &flat_out,
        &[batch_size as i32, n_heads as i32, 1, head_dim as i32],
        stream,
    )
}

// KV write is now handled by PagePool::write_kv() in paged_cache.rs
