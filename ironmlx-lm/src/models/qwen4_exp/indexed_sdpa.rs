//! Preserve pinned MLX two-pass decode arithmetic while fetching only selected
//! KV pages. Compaction must not renumber tokens into different BF16 partials.
use anyhow::{ensure, Context, Result};
use mlx::{Array, Dtype, MetalKernel, Shape, StreamOrDevice};
use std::sync::OnceLock;

fn blocks(n: usize) -> Result<usize> {
    static ARCH: OnceLock<std::result::Result<String, String>> = OnceLock::new();
    let arch = ARCH
        .get_or_init(|| mlx::metal::architecture().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| anyhow::anyhow!(e.clone()))?;
    let value = match arch.chars().last() {
        Some('s') => {
            if n <= 8192 {
                128
            } else if n <= 32768 {
                256
            } else if n <= 65536 {
                512
            } else {
                1024
            }
        }
        Some('d') => {
            if n < 16384 {
                128
            } else if n < 65536 {
                512
            } else {
                1024
            }
        }
        _ => 64,
    };
    match std::env::var("MLX_SDPA_BLOCKS") {
        Ok(v) => {
            let n = v.parse::<i32>().context("invalid MLX_SDPA_BLOCKS")?;
            if n > 0 {
                let n = (n as usize).div_ceil(32) * 32;
                ensure!(n <= 4096, "MLX_SDPA_BLOCKS exceeds bounded Flash workspace");
                Ok(n)
            } else {
                Ok(value)
            }
        }
        Err(_) => Ok(value),
    }
}
fn kernel(first: bool) -> Result<&'static MetalKernel> {
    static FIRST: OnceLock<MetalKernel> = OnceLock::new();
    static SECOND: OnceLock<MetalKernel> = OnceLock::new();
    let slot = if first { &FIRST } else { &SECOND };
    if let Some(kernel) = slot.get() {
        return Ok(kernel);
    }
    let (one, two) = include_str!("indexed_sdpa.metal")
        .split_once("// PASS TWO")
        .context("indexed SDPA source split missing")?;
    let (name, inputs, outputs, body) = if first {
        (
            "ironmlx_flash_indexed_sdpa_first",
            vec![
                "queries",
                "keys",
                "values",
                "page_map",
                "n_keys",
                "packed_size",
                "scale",
            ],
            vec!["partials", "sums", "maxs"],
            one,
        )
    } else {
        (
            "ironmlx_flash_indexed_sdpa_second",
            vec!["partials", "sums", "maxs", "n_blocks"],
            vec!["out"],
            two,
        )
    };
    let built = MetalKernel::builder(name)
        .inputs(&inputs)
        .outputs(&outputs)
        .header("#include <metal_simdgroup>\nusing namespace metal;\n")
        .source(body)
        .build()?;
    let _ = slot.set(built);
    Ok(slot.get().unwrap())
}
#[allow(clippy::too_many_arguments)]
pub(super) fn decode(
    q: &Array,
    k: &Array,
    v: &Array,
    pages: &[usize],
    logical: usize,
    ratio: usize,
    scale: f32,
    target: StreamOrDevice,
) -> Result<Array> {
    let qs = q.shape();
    let kd = k.shape();
    let heads = qs.as_slice()[1];
    let kv = kd.as_slice()[1];
    let d = qs.as_slice()[3];
    ensure!(
        logical >= 4096
            && d == 256
            && heads == 24
            && kv == 2
            && qs.as_slice()[2] == 1
            && q.dtype() == Dtype::Bfloat16,
        "unsupported indexed Flash decode geometry"
    );
    let blocks = blocks(logical)?;
    let mut map = vec![-1i32; logical.div_ceil(ratio)];
    for (local, &global) in pages.iter().enumerate() {
        ensure!(global < map.len(), "invalid indexed KV page");
        map[global] = local as i32;
    }
    let q = mlx::ops::shape::contiguous_on(q, false, target)?;
    let k = mlx::ops::shape::contiguous_on(k, false, target)?;
    let v = mlx::ops::shape::contiguous_on(v, false, target)?;
    let map = Array::try_from((map.as_slice(), [map.len() as i32]))?;
    let n = Array::try_from((&[logical as i32][..], [1]))?;
    let packed = Array::try_from((&[kd.as_slice()[2]][..], [1]))?;
    let scale = Array::try_from((&[scale][..], [1]))?;
    let mut partials = kernel(true)?
        .dispatch_builder()
        .inputs(&[&q, &k, &v, &map, &n, &packed, &scale])
        .output_shapes(&[
            Shape::from([heads, blocks as i32, d]),
            Shape::from([heads, blocks as i32]),
            Shape::from([heads, blocks as i32]),
        ])
        .output_dtypes(&[q.dtype(), Dtype::Float32, Dtype::Float32])
        .template_dtype("T", q.dtype())
        .template_int("D", d)
        .template_int("GQA", heads / kv)
        .template_int("BLOCKS", blocks as i32)
        .template_int("RATIO", ratio as i32)
        .grid(32 * heads, 1, blocks as i32)
        .threadgroup(32, 1, 1)
        .stream(target)
        .dispatch()?;
    let part = partials.take_at(0)?;
    let sums = partials.take_at(0)?;
    let maxs = partials.take_at(0)?;
    let nblocks = Array::try_from((&[blocks as i32][..], [1]))?;
    let mut out = kernel(false)?
        .dispatch_builder()
        .inputs(&[&part, &sums, &maxs, &nblocks])
        .output_shapes(&[Shape::from([1, heads, 1, d])])
        .output_dtypes(&[q.dtype()])
        .template_dtype("T", q.dtype())
        .template_int("D", d)
        .grid(1024 * heads, 1, 1)
        .threadgroup(1024, 1, 1)
        .stream(target)
        .dispatch()?;
    Ok(out.take_at(0)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires actual Apple GPU; checks native reduction boundaries to 256K"]
    fn indexed_decode_preserves_native_long_context_arithmetic() {
        for logical in [4096usize, 8193, 16384, 32769, 65537, 262144] {
            let sequence =
                mlx::ops::constructors::arange(0., (2 * logical * 256) as f64, 1., Dtype::Float32)
                    .unwrap();
            let k = mlx::ops::unary::sin(&sequence)
                .unwrap()
                .reshape([1, 2, logical as i32, 256])
                .unwrap()
                .astype(Dtype::Bfloat16)
                .unwrap();
            let v = mlx::ops::unary::cos(&sequence)
                .unwrap()
                .reshape([1, 2, logical as i32, 256])
                .unwrap()
                .astype(Dtype::Bfloat16)
                .unwrap();
            let q =
                mlx::ops::constructors::arange(0., (24 * 256) as f64, 1., Dtype::Float32).unwrap();
            let q = mlx::ops::unary::sin(&q)
                .unwrap()
                .reshape([1, 24, 1, 256])
                .unwrap()
                .astype(Dtype::Bfloat16)
                .unwrap();
            let mut pages: Vec<usize> = (0..logical.div_ceil(4))
                .filter(|p| p % 31 == 0)
                .take(511)
                .collect();
            pages.push((logical - 1) / 4);
            pages.sort_unstable();
            pages.dedup();
            let indices: Vec<i32> = pages
                .iter()
                .flat_map(|p| (0..4).map(move |i| (p * 4 + i).min(logical - 1) as i32))
                .collect();
            let indices = Array::try_from((indices.as_slice(), [indices.len() as i32])).unwrap();
            let packed_k = mlx::ops::indexing::take(&k, &indices, 2).unwrap();
            let packed_v = mlx::ops::indexing::take(&v, &indices, 2).unwrap();
            let mask: Vec<bool> = (0..logical)
                .map(|i| pages.binary_search(&(i / 4)).is_ok())
                .collect();
            let mask = Array::try_from((mask.as_slice(), [1, 1, 1, logical as i32])).unwrap();
            let expected = mlx::fast::scaled_dot_product_attention(
                &q,
                &k,
                &v,
                1. / 16.,
                "",
                Some(&mask),
                None,
            )
            .unwrap();
            let actual = decode(
                &q,
                &packed_k,
                &packed_v,
                &pages,
                logical,
                4,
                1. / 16.,
                ().into(),
            )
            .unwrap();
            assert_eq!(
                expected.to_vec::<half::bf16>().unwrap(),
                actual.to_vec::<half::bf16>().unwrap(),
                "native reduction changed at N={logical}"
            );
            eprintln!(
                "indexed/native exact at N={logical} blocks={}",
                blocks(logical).unwrap()
            );
        }
    }
}
