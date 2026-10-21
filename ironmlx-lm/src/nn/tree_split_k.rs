//! Long-context flat-tree attention on the split-K D256 NAX kernels: each
//! tree node's row equals the row of a linear verify of its own
//! root-to-node path (same key blocks, chunk columns and accumulation), so
//! a tree window commits exactly what linear windows would.
use crate::Result;
use mlx::{Array, StreamOrDevice};

/// Keys per row-part tail: the prompt keys from the last 32-key chunk start
/// (at most 31) plus a root-to-leaf path (at most 16 nodes).
const TAIL_KEYS: i32 = 48;
/// Query rows per kernel row part.
const PART_ROWS: i32 = 16;

/// Node slots and part paths of a tree: node-major rows, two slots per part,
/// both on the part's root-to-leaf path (a lone node repeats itself).
struct Layout {
    slots: Vec<i32>,
    part_paths: Vec<usize>,
    first_slot: Vec<i32>,
}

fn layout(paths: &[Vec<i32>], slots_per_part: usize) -> Layout {
    let n = paths.len();
    let mut is_parent = vec![false; n];
    for path in paths {
        if path.len() >= 2 {
            is_parent[path[path.len() - 2] as usize] = true;
        }
    }
    let mut assigned = vec![false; n];
    let mut slots = Vec::new();
    let mut part_paths = Vec::new();
    for leaf in (0..n).filter(|&leaf| !is_parent[leaf]) {
        let fresh: Vec<i32> = paths[leaf]
            .iter()
            .copied()
            .filter(|&node| !assigned[node as usize])
            .collect();
        for &node in &fresh {
            assigned[node as usize] = true;
        }
        for chunk in fresh.chunks(slots_per_part) {
            for i in 0..slots_per_part {
                slots.push(chunk[i.min(chunk.len() - 1)]);
            }
            part_paths.push(leaf);
        }
    }
    let mut first_slot = vec![-1; n];
    for (slot, &node) in slots.iter().enumerate() {
        if first_slot[node as usize] < 0 {
            first_slot[node as usize] = slot as i32;
        }
    }
    Layout {
        slots,
        part_paths,
        first_slot,
    }
}

/// Host-built inputs of one window's split-K tree attention.
pub(crate) struct Prepared {
    gqa: i32,
    slot_count: i32,
    parts: i32,
    key_count: i32,
    tail_from: i32,
    tail: Array,
    limits: Array,
    slots: Array,
    first: Array,
}

fn prepare(paths: &[Vec<i32>], base: i32, heads: i32, kv_heads: i32) -> Result<Option<Prepared>> {
    if kv_heads <= 0 || heads % kv_heads != 0 {
        return Ok(None);
    }
    let gqa = heads / kv_heads;
    let min_keys = mlx::metal::decode_d256_nax_active_min_keys();
    let longest = paths.iter().map(Vec::len).max().unwrap_or(0) as i32;
    let key_count = base + longest;
    // A linear verify of any of these paths sees the same key blocks.
    if PART_ROWS % gqa != 0
        || min_keys <= 0
        || base + 1 < min_keys
        || longest > TAIL_KEYS - 31
        || mlx::fast::decode_block_keys(base + 1) != mlx::fast::decode_block_keys(key_count)
    {
        return Ok(None);
    }
    let layout = layout(paths, (PART_ROWS / gqa) as usize);
    let slot_count = layout.slots.len() as i32;
    let parts = layout.part_paths.len() as i32;
    let tail_from = base / 32 * 32;
    let mut tail = Vec::with_capacity((parts * TAIL_KEYS) as usize);
    for &leaf in &layout.part_paths {
        let start = tail.len();
        tail.extend(tail_from..base);
        tail.extend(paths[leaf].iter().map(|&node| base + node));
        tail.resize(start + TAIL_KEYS as usize, base);
    }
    let limits: Vec<i32> = layout
        .slots
        .iter()
        .map(|&node| base + paths[node as usize].len() as i32)
        .collect();
    Ok(Some(Prepared {
        gqa,
        slot_count,
        parts,
        key_count,
        tail_from,
        tail: (tail.as_slice(), &[tail.len() as i32][..]).try_into()?,
        limits: (limits.as_slice(), &[slot_count][..]).try_into()?,
        slots: (layout.slots.as_slice(), &[slot_count][..]).try_into()?,
        first: (layout.first_slot.as_slice(), &[paths.len() as i32][..]).try_into()?,
    }))
}

/// Tree attention of `queries` `[1, Hq, R, 256]` (node order, rotated) over
/// `keys` / `values` `[1, Hkv, base + R, 256]` (the prompt, then the tree's
/// nodes in node order); `plan.paths[n]` is node `n`'s root-to-node path.
/// `Ok(None)` when the split-K tree kernel does not apply.
pub(crate) fn attention_on(
    queries: &Array,
    keys: &Array,
    values: &Array,
    base: i32,
    plan: &super::dflash_tree::Plan,
    scale: f32,
    target: StreamOrDevice,
) -> Result<Option<Array>> {
    let q_dims = queries.shape();
    let q_dims = q_dims.as_slice();
    let k_dims = keys.shape();
    let k_dims = k_dims.as_slice();
    let (heads, nodes, dim) = (q_dims[1], q_dims[2], q_dims[3]);
    let kv_heads = k_dims[1];
    if q_dims[0] != 1
        || dim != 256
        || nodes as usize != plan.paths.len()
        || k_dims[2] < base + nodes
    {
        return Ok(None);
    }
    if plan.split_k.get().is_none() {
        let _ = plan
            .split_k
            .set((base, prepare(&plan.paths, base, heads, kv_heads)?));
    }
    let (cached_base, prepared) = plan.split_k.get().unwrap();
    anyhow::ensure!(
        *cached_base == base,
        "tree attention offsets differ across layers"
    );
    let Some(p) = prepared.as_ref() else {
        return Ok(None);
    };
    let gather_tail = |array: &Array| -> Result<Array> {
        Ok(mlx::ops::indexing::take_on(array, &p.tail, 2, target)?
            .reshape_on(&[1, kv_heads, p.parts, TAIL_KEYS, dim][..], target)?)
    };
    let tail_keys = gather_tail(keys)?;
    let tail_values = gather_tail(values)?;
    let node_major = mlx::ops::indexing::take_on(queries, &p.slots, 2, target)?
        .reshape_on(&[1, kv_heads, p.gqa, p.slot_count, dim][..], target)?
        .transpose_axes_on(&[0_i32, 1, 3, 2, 4][..], target)?
        .reshape_on((1, kv_heads, p.slot_count * p.gqa, dim), target)?;
    let Some(output) = mlx::fast::tree_attention_on(
        &node_major,
        keys,
        values,
        &tail_keys,
        &tail_values,
        &p.limits,
        scale,
        p.gqa,
        p.key_count,
        p.tail_from,
        target,
    )?
    else {
        return Ok(None);
    };
    let head_major = output
        .reshape_on(&[1, kv_heads, p.slot_count, p.gqa, dim][..], target)?
        .transpose_axes_on(&[0_i32, 1, 3, 2, 4][..], target)?
        .reshape_on((1, heads, p.slot_count, dim), target)?;
    Ok(Some(mlx::ops::indexing::take_on(
        &head_major,
        &p.first,
        2,
        target,
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::Dtype;
    use serial_test::serial;

    #[test]
    fn layout_pairs_nodes_on_one_path() {
        // root 0 -> 1 -> 2 -> 3, root -> 4 -> 5, 1 -> 6
        let parents = [-1, 0, 1, 2, 0, 4, 1];
        let mut paths: Vec<Vec<i32>> = Vec::new();
        for (i, &p) in parents.iter().enumerate() {
            let mut path = if p < 0 {
                Vec::new()
            } else {
                paths[p as usize].clone()
            };
            path.push(i as i32);
            paths.push(path);
        }
        let layout = layout(&paths, 2);
        assert_eq!(layout.slots.len(), layout.part_paths.len() * 2);
        for (part, &leaf) in layout.part_paths.iter().enumerate() {
            for &node in &layout.slots[part * 2..part * 2 + 2] {
                assert!(
                    paths[leaf].contains(&node),
                    "part {part} node {node} off path"
                );
            }
        }
        assert!(layout.first_slot.iter().all(|&slot| slot >= 0));
    }

    /// Every node's row equals the row of the linear split-K verify of its
    /// root-to-node path, bit for bit.
    #[test]
    #[serial(mlx_metal)]
    #[ignore = "requires an M5+ GPU (NAX) and MLX_DIR"]
    fn tree_rows_equal_linear_path_verify_rows() -> Result<()> {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        }
        let library =
            std::env::temp_dir().join(format!("ironmlx-tree-nax-{}.metallib", std::process::id()));
        std::fs::write(
            &library,
            mlx::metal::prefill_shaders::PREFILL_D256_NAX_METALLIB,
        )?;
        assert!(mlx::metal::set_prefill_d256_nax_library(
            library.to_str().unwrap()
        ));
        assert!(mlx::metal::set_decode_d256_nax_min_keys(8193));
        let (hq, hkv, d) = (16, 2, 256);
        let scale = 1.0 / 16.0;
        let trees: [&[i32]; 4] = [
            &[-1, 0, 1, 2],
            &[-1, 0, 1, 0, 3, 1],
            &[-1, 0, 0, 1, 2, 3, 3, 4],
            &[-1, 0, 1, 2, 0, 4, 1, 6, 2],
        ];
        for (case, base) in [
            (0, 9000),
            (1, 12345),
            (2, 16380),
            (3, 31000),
            (0, 40191),
            (2, 16300),
        ] {
            let parents = trees[case];
            let r = parents.len() as i32;
            let mut paths: Vec<Vec<i32>> = Vec::new();
            for (i, &p) in parents.iter().enumerate() {
                let mut path = if p < 0 {
                    Vec::new()
                } else {
                    paths[p as usize].clone()
                };
                path.push(i as i32);
                paths.push(path);
            }
            let key = mlx::random::key(base as u64)?;
            let sample = |shape: &[i32], k: &Array| -> Result<Array> {
                Ok(mlx::random::normal()
                    .shape(shape)
                    .dtype(Dtype::Bfloat16)
                    .key(k)
                    .sample()?)
            };
            let keys_all = sample(&[1, hkv, base + r, d], &key)?;
            let values_all = sample(&[1, hkv, base + r, d], &mlx::random::key(base as u64 + 1)?)?;
            let queries = sample(&[1, hq, r, d], &mlx::random::key(base as u64 + 2)?)?;
            let target = StreamOrDevice::default();
            let plan = crate::nn::dflash_tree::Plan::new(parents)?;
            let Some(tree) =
                attention_on(&queries, &keys_all, &values_all, base, &plan, scale, target)?
            else {
                // A block-size boundary between the root and the deepest node.
                assert_ne!(
                    mlx::fast::decode_block_keys(base + 1),
                    mlx::fast::decode_block_keys(
                        base + paths.iter().map(Vec::len).max().unwrap() as i32
                    )
                );
                eprintln!(
                    "case {case} base {base}: tree straddles a block-size boundary (falls back)"
                );
                continue;
            };
            let tree_rows: Vec<f32> = mlx::ops::astype(&tree, Dtype::Float32)?.to_vec()?;
            let mut checked = 0;
            for (node, path) in paths.iter().enumerate() {
                let idx: Vec<i32> = (0..base).chain(path.iter().map(|&n| base + n)).collect();
                let idx: Array = (idx.as_slice(), &[idx.len() as i32][..]).try_into()?;
                let k_lin = mlx::ops::indexing::take(&keys_all, &idx, 2)?;
                let v_lin = mlx::ops::indexing::take(&values_all, &idx, 2)?;
                let qi: Array = (path.as_slice(), &[path.len() as i32][..]).try_into()?;
                let q_lin = mlx::ops::indexing::take(&queries, &qi, 2)?;
                let lin = mlx::fast::scaled_dot_product_attention(
                    &q_lin, &k_lin, &v_lin, scale, "causal", None, None,
                )?;
                let lin: Vec<f32> = mlx::ops::astype(&lin, Dtype::Float32)?.to_vec()?;
                let depth = path.len() - 1;
                for h in 0..hq as usize {
                    for x in 0..d as usize {
                        let a = tree_rows[(h * r as usize + node) * d as usize + x];
                        let b = lin[(h * path.len() + depth) * d as usize + x];
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "case {case} base {base} node {node} head {h} dim {x}"
                        );
                    }
                }
                checked += 1;
            }
            eprintln!("case {case} base {base}: {checked} nodes bit-identical to their linear verify rows");
        }
        Ok(())
    }
}
