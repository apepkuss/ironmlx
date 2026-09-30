//! Shared committed-prefix tiles plus small path-gathered tails, B1 only.
use crate::Result;
use mlx::{Array, Dtype, MetalKernel, Shape, StreamOrDevice};
use std::sync::OnceLock;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[serial_test::serial(mlx_metal)]
    fn branching_attention_matches_serial_across_tiles() -> Result<()> {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        }
        if !mlx::metal::architecture()?.starts_with("applegpu_g17") {
            return Ok(());
        }
        use mlx::ops::{
            cast::astype,
            indexing::{slice_strided_on, take_on},
        };
        let target = StreamOrDevice::default();
        let paths = vec![
            vec![0],
            vec![0, 1],
            vec![0, 2],
            vec![0, 1, 3],
            vec![0, 2, 4],
            vec![0, 1, 3, 5],
            vec![0, 2, 4, 6],
            vec![0, 7],
        ];
        for prefix in [0, 1, 62, 63, 64, 65, 510, 511, 512, 513, 1023, 2048] {
            let make = |h: i32, n: i32, salt: i32| -> Result<Array> {
                let data = (0..h * n * 256)
                    .map(|i| (((i * 13 + salt) % 113) as f32 - 56.) * 0.009)
                    .collect::<Vec<_>>();
                Ok(astype(
                    &Array::try_from((data.as_slice(), &[1, h, n, 256][..]))?,
                    Dtype::Bfloat16,
                )?)
            };
            let q = make(24, 8, 3)?;
            let k = make(4, prefix + 8, 11)?;
            let v = make(4, prefix + 8, 17)?;
            let wide = attend(&q, &k, &v, 0.0625, &Layout::new(&paths, prefix)?, target)?;
            for (row, path) in paths.iter().enumerate() {
                let row = row as i32;
                let slice = |a: &Array| {
                    slice_strided_on(
                        a,
                        &[0, 0, row, 0][..],
                        &[1, 24, row + 1, 256][..],
                        &[1, 1, 1, 1][..],
                        target,
                    )
                };
                let ids = (0..prefix)
                    .chain(path.iter().map(|r| prefix + r))
                    .collect::<Vec<_>>();
                let ids: Array = (ids.as_slice(), &[ids.len() as i32][..]).try_into()?;
                let qr = slice(&q)?;
                let kr = take_on(&k, &ids, 2, target)?;
                let vr = take_on(&v, &ids, 2, target)?;
                let one = astype(
                    &super::super::m5_attention::attend(&qr, &kr, &vr, 0.0625, target)?,
                    Dtype::Float32,
                )?
                .to_vec::<f32>()?;
                let got = astype(&slice(&wide)?, Dtype::Float32)?.to_vec::<f32>()?;
                assert_eq!(one, got, "prefix={prefix} row={row}");
            }
            eprintln!("tree prefix={prefix}: all branches match");
        }
        Ok(())
    }
}

pub(crate) struct Layout {
    base: i32,
    rows: i32,
    ca: i32,
    ncb: i32,
    tiles: i32,
    meta: Array,
    nodes: Array,
    paths: Array,
    tile_stream: Array,
}
impl Layout {
    pub fn new(paths: &[Vec<i32>], base: i32) -> Result<Self> {
        let rows = paths.len() as i32;
        let pt = base / 64 * 64;
        let ca = (pt + 511) / 512;
        let depth = paths.iter().map(Vec::len).max().unwrap() as i32 - 1;
        let ncb = (base + depth) / 512 - pt / 512 + 1;
        let tiles = (rows * 6 + 15) / 16;
        let mut meta = vec![0; 20];
        meta[..7].copy_from_slice(&[1, ca, tiles, ncb, rows, tiles * 16, ca]);
        meta[8..15].copy_from_slice(&[pt, ca, rows, 0, base, ncb, 0]);
        let nodes = paths
            .iter()
            .flat_map(|p| [p.len() as i32 - 1, 0])
            .collect::<Vec<_>>();
        let paths = paths
            .iter()
            .flat_map(|p| {
                let mut r = p.clone();
                r.resize(16, 0);
                r
            })
            .collect::<Vec<_>>();
        let arr = |x: &[i32]| -> Result<Array> { Ok((x, &[x.len() as i32][..]).try_into()?) };
        Ok(Self {
            base,
            rows,
            ca,
            ncb,
            tiles,
            meta: arr(&meta)?,
            nodes: arr(&nodes)?,
            paths: arr(&paths)?,
            tile_stream: arr(&vec![0; tiles as usize])?,
        })
    }
}

fn kernel(kind: usize) -> Result<MetalKernel> {
    static PARTIAL: OnceLock<MetalKernel> = OnceLock::new();
    static TAIL: OnceLock<MetalKernel> = OnceLock::new();
    static MERGE: OnceLock<MetalKernel> = OnceLock::new();
    let (slot, inputs, outputs, source): (_, &[&str], &[&str], &str) = match kind {
        0 => (
            &PARTIAL,
            &["Qp", "K", "V", "scale", "meta", "tile_stream"],
            &["PO", "PM", "PL"],
            include_str!("m5_tree_attention_partial.metal"),
        ),
        1 => (
            &TAIL,
            &[
                "QB", "K", "V", "scale", "meta", "paths", "nodes", "POA", "PMA", "PLA",
            ],
            &["PO", "PM", "PL"],
            include_str!("m5_tree_attention_tail.metal"),
        ),
        _ => (
            &MERGE,
            &["POA", "PMA", "PLA", "POB", "PMB", "PLB", "meta", "nodes"],
            &["OUT"],
            include_str!("m5_tree_attention_merge.metal"),
        ),
    };
    if slot.get().is_none() {
        let k=MetalKernel::builder(format!("ironmlx_m5_tree_attention_v1_{kind}"))
            .inputs(inputs).outputs(outputs).source(source)
            .header("#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace mpp::tensor_ops;")
            .ensure_row_contiguous(false).build()?;
        let _ = slot.set(k);
    }
    Ok(slot.get().unwrap().clone())
}
pub(crate) fn attend(
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f32,
    plan: &Layout,
    target: StreamOrDevice,
) -> Result<Array> {
    use mlx::ops::shape::*;
    let Layout {
        base,
        rows,
        ca,
        ncb,
        tiles,
        ..
    } = *plan;
    anyhow::ensure!(
        q.shape().as_slice() == [1, 24, rows, 256] && k.shape().as_slice()[2] == base + rows,
        "tree attention layout mismatch"
    );
    let rp = tiles * 16;
    let sc: Array = (&[scale][..], &[1][..]).try_into()?;
    let per = reshape_on(q, &[4, 6, rows, 256][..], target)?;
    let per = transpose_axes_on(&per, &[0, 2, 1, 3][..], target)?;
    let (poa, pma, pla) = if ca > 0 {
        let mut qa = reshape_on(&per, &[4, rows * 6, 256][..], target)?;
        if rp > rows * 6 {
            let z = Array::zeros_on(&[4, rp - rows * 6, 256][..], Dtype::Bfloat16, target)?;
            qa = concatenate_on(&[&qa, &z], 1, target)?;
        }
        let qa = contiguous_on(&qa, false, target)?;
        let sg = super::m5_attention::group_tiles(tiles);
        let mut out = kernel(0)?
            .dispatch_builder()
            .inputs(&[&qa, k, v, &sc, &plan.meta, &plan.tile_stream])
            .template_int("G", 6)
            .template_int("D", 256)
            .template_int("SG", sg)
            .template_int("CK", 512)
            .template_int("TK", 64)
            .template_int("MG", 8)
            .template_int("MS", 12)
            .output_shapes(&[
                Shape::from(&[4, ca, rp, 256][..]),
                Shape::from(&[4, ca, rp][..]),
                Shape::from(&[4, ca, rp][..]),
            ])
            .output_dtypes(&[Dtype::Float32; 3])
            .grid(4 * 32 * sg, ca, (tiles + sg - 1) / sg)
            .threadgroup(32 * sg, 1, 1)
            .stream(target)
            .dispatch()?;
        (out.take_at(0)?, out.take_at(0)?, out.take_at(0)?)
    } else {
        let z = Array::zeros_on(&[16][..], Dtype::Float32, target)?;
        (z.clone(), z.clone(), z)
    };
    let z = Array::zeros_on(&[4, rows, 10, 256][..], Dtype::Bfloat16, target)?;
    let qb = contiguous_on(&concatenate_on(&[&per, &z], 2, target)?, false, target)?;
    let mut out = kernel(1)?
        .dispatch_builder()
        .inputs(&[
            &qb,
            k,
            v,
            &sc,
            &plan.meta,
            &plan.paths,
            &plan.nodes,
            &poa,
            &pma,
            &pla,
        ])
        .template_int("G", 6)
        .template_int("D", 256)
        .template_int("CK", 512)
        .template_int("TK", 64)
        .template_int("MAXD", 16)
        .template_int("MG", 8)
        .template_int("MS", 12)
        .output_shapes(&[
            Shape::from(&[4, ncb, rows, 16, 256][..]),
            Shape::from(&[4, ncb, rows, 16][..]),
            Shape::from(&[4, ncb, rows, 16][..]),
        ])
        .output_dtypes(&[Dtype::Float32; 3])
        .grid(4 * 32, ncb, rows)
        .threadgroup(32, 1, 1)
        .stream(target)
        .dispatch()?;
    let pob = out.take_at(0)?;
    let pmb = out.take_at(0)?;
    let plb = out.take_at(0)?;
    Ok(kernel(2)?
        .dispatch_builder()
        .inputs(&[&poa, &pma, &pla, &pob, &pmb, &plb, &plan.meta, &plan.nodes])
        .template_int("G", 6)
        .template_int("D", 256)
        .template_int("CK", 512)
        .template_int("MG", 8)
        .template_int("MS", 12)
        .output_shapes(&[Shape::from(&[1, 24, rows, 256][..])])
        .output_dtypes(&[Dtype::Bfloat16])
        .grid(4 * 32, rows * 6, 1)
        .threadgroup(32, 1, 1)
        .stream(target)
        .dispatch()?
        .take_at(0)?)
}
