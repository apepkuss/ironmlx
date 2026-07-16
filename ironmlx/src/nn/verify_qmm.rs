//! Quantized matrix multiplication kernels for Qwen MTP verify projections.
//!
//! MTP verify produces a skinny `[1, M, K]` hidden batch where
//! `M = 1 + draft_depth`. MLX's native small-batch path remains the default;
//! these kernels are candidates for the subset of large-output affine
//! projections where a dedicated morphology wins:
//!
//! - split-K for `16K <= N < 100K`;
//! - multi-simdgroup (MSG) for huge outputs such as Qwen's `lm_head`.
//!
//! The Metal morphology is adapted from MTPLX and oMLX's
//! `qwen35_verify_qmm.py` under Apache-2.0.

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use anyhow::{anyhow, Context};
use mlx::{Array, Dtype, MetalKernel, Shape, StreamOrDevice};

use crate::core::QuantMode;
use crate::Result;

use super::linear::QuantizedLinearParts;

const MIN_ROUTE_N: i32 = 16_384;
const MSG_MIN_N: i32 = 100_000;
const MSG_NSG: i32 = 8;

thread_local! {
    static ROUTE_DEPTH: Cell<u32> = const { Cell::new(0) };
}

pub(crate) struct VerifyQmmScope;

impl Drop for VerifyQmmScope {
    fn drop(&mut self) {
        ROUTE_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

pub(crate) fn armed_scope() -> VerifyQmmScope {
    ROUTE_DEPTH.with(|depth| depth.set(depth.get().saturating_add(1)));
    VerifyQmmScope
}

pub(crate) fn is_armed() -> bool {
    ROUTE_DEPTH.with(|depth| depth.get() > 0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum VerifyQmmKind {
    SplitK,
    Msg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct KernelKey {
    kind: VerifyQmmKind,
    rows: i32,
    bits: i32,
    group_size: i32,
    dtype: Dtype,
    k_parts: i32,
}

fn kernel_cache() -> &'static Mutex<HashMap<KernelKey, MetalKernel>> {
    static CACHE: OnceLock<Mutex<HashMap<KernelKey, MetalKernel>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn eligible_kind(x: &Array, parts: &QuantizedLinearParts<'_>) -> Option<VerifyQmmKind> {
    let shape = x.shape();
    let dims = shape.as_slice();
    if dims.len() != 3 || dims[0] != 1 {
        return None;
    }
    let rows = dims[1];
    let k = dims[2];
    let weight_shape = parts.weight.shape();
    let weight_dims = weight_shape.as_slice();
    if weight_dims.len() != 2 {
        return None;
    }
    let n = weight_dims[0];
    if parts.mode != QuantMode::Affine
        || parts.biases.is_none()
        || !matches!(parts.bits, 4 | 8)
        || !matches!(parts.group_size, 32 | 64 | 128)
        || !matches!(x.dtype(), Dtype::Bfloat16 | Dtype::Float16)
        || !(3..=6).contains(&rows)
        || k % 64 != 0
        || n % 4 != 0
        || n < MIN_ROUTE_N
    {
        return None;
    }
    if n >= MSG_MIN_N && n % (4 * MSG_NSG) == 0 {
        Some(VerifyQmmKind::Msg)
    } else {
        Some(VerifyQmmKind::SplitK)
    }
}

pub(crate) fn forward_candidate_on(
    x: &Array,
    parts: QuantizedLinearParts<'_>,
    target: impl Into<StreamOrDevice>,
) -> Result<Option<Array>> {
    let Some(kind) = eligible_kind(x, &parts) else {
        return Ok(None);
    };
    if !route_profile_allows(kind, x, &parts) {
        return Ok(None);
    }
    dispatch_candidate_on(x, parts, kind, target).map(Some)
}

fn dispatch_candidate_on(
    x: &Array,
    parts: QuantizedLinearParts<'_>,
    kind: VerifyQmmKind,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let target = target.into();
    let dims = x.shape();
    let dims = dims.as_slice();
    let rows = dims[1];
    let k = dims[2];
    let n = parts.weight.shape().as_slice()[0];
    let biases = parts
        .biases
        .ok_or_else(|| anyhow!("verify qmm affine route requires biases"))?;
    let k_parts = if kind == VerifyQmmKind::SplitK { 2 } else { 0 };
    let key = KernelKey {
        kind,
        rows,
        bits: parts.bits,
        group_size: parts.group_size,
        dtype: x.dtype(),
        k_parts,
    };
    let kernel = cached_kernel(key)?;
    let (grid_x, grid_y, threads) = match kind {
        VerifyQmmKind::Msg => {
            let cols = 4 * MSG_NSG;
            (32 * MSG_NSG, (n + cols - 1) / cols, 32 * MSG_NSG)
        }
        VerifyQmmKind::SplitK => (32 * k_parts, n / 4, 32 * k_parts),
    };
    let mut outputs = kernel
        .dispatch_builder()
        .inputs(&[x, parts.weight, parts.scales, biases])
        .output_shapes(&[Shape::from((1_i32, rows, n))])
        .output_dtypes(&[x.dtype()])
        .grid(grid_x, grid_y, 1)
        .threadgroup(threads, 1, 1)
        .template_dtype("T", x.dtype())
        .template_int("K_SIZE", k)
        .template_int("N_SIZE", n)
        .stream(target)
        .dispatch()
        .context("dispatch Qwen MTP verify qmm")?;
    let mut y = outputs.take_at(0)?;
    if let Some(bias) = parts.bias {
        y = &y + bias;
    }
    Ok(y)
}

fn route_profile_allows(kind: VerifyQmmKind, x: &Array, parts: &QuantizedLinearParts<'_>) -> bool {
    static ARCHITECTURE: OnceLock<Option<String>> = OnceLock::new();
    let architecture = ARCHITECTURE.get_or_init(|| mlx::metal::architecture().ok());
    let Some(architecture) = architecture.as_deref() else {
        return false;
    };
    let rows = x.shape().as_slice()[1];
    let n = parts.weight.shape().as_slice()[0];

    // The checked-in route profile is intentionally evidence-based. On the
    // M5 Max (applegpu_g17s), M=3..4 beats MLX qmv_wide; M=5..6 regresses.
    // Unknown GPU generations retain the native MLX path until profiled.
    architecture == "applegpu_g17s"
        && (3..=4).contains(&rows)
        && match kind {
            VerifyQmmKind::Msg => true,
            VerifyQmmKind::SplitK => n >= 65_536,
        }
}

fn cached_kernel(key: KernelKey) -> Result<MetalKernel> {
    if let Some(kernel) = kernel_cache()
        .lock()
        .map_err(|_| anyhow!("verify qmm kernel cache poisoned"))?
        .get(&key)
        .cloned()
    {
        return Ok(kernel);
    }
    let kernel = build_kernel(key)?;
    let mut cache = kernel_cache()
        .lock()
        .map_err(|_| anyhow!("verify qmm kernel cache poisoned"))?;
    Ok(cache.entry(key).or_insert_with(|| kernel.clone()).clone())
}

fn build_kernel(key: KernelKey) -> Result<MetalKernel> {
    let (name, source) = match key.kind {
        VerifyQmmKind::Msg => (
            format!(
                "ironmlx_verify_qmm_msg_m{}_q{}_gs{}",
                key.rows, key.bits, key.group_size
            ),
            build_msg_source(key.rows, key.bits, key.group_size),
        ),
        VerifyQmmKind::SplitK => (
            format!(
                "ironmlx_verify_qmm_splitk_m{}_q{}_gs{}_kp{}",
                key.rows, key.bits, key.group_size, key.k_parts
            ),
            build_splitk_source(key.rows, key.bits, key.group_size, key.k_parts),
        ),
    };
    MetalKernel::builder(name)
        .inputs(&["x", "w_q", "scales", "biases"])
        .outputs(&["y"])
        .source(source)
        .ensure_row_contiguous(true)
        .atomic_outputs(false)
        .build()
        .context("build Qwen MTP verify qmm kernel")
}

fn x_loads(rows: i32, suffix: &str) -> String {
    (0..rows)
        .map(|row| format!("Vec8 v{suffix}_{row} = xv[({row} * K + k_base{suffix}) / 8];"))
        .collect::<Vec<_>>()
        .join("\n            ")
}

fn msg_x_loads(rows: i32) -> String {
    (0..rows)
        .map(|row| format!("Vec8 v{row} = xv[({row} * K + k_base) / 8];"))
        .collect::<Vec<_>>()
        .join("\n            ")
}

fn msg_fma_block(rows: i32, bits: i32) -> String {
    let per = if bits == 4 { 8 } else { 4 };
    let mask = if bits == 4 { "0xFu" } else { "0xFFu" };
    let shift = if bits == 4 { 4 } else { 8 };
    let mut lines = vec![format!("for (int ki = 0; ki < {per}; ++ki) {{")];
    for col in 0..4 {
        lines.push(format!(
            "    float w{col} = float((p{col} >> (ki * {shift})) & {mask}) * s{col} + b{col};"
        ));
    }
    for col in 0..4 {
        for row in 0..rows {
            let x_index = if bits == 4 {
                "ki".to_owned()
            } else {
                "ki + koff".to_owned()
            };
            lines.push(format!(
                "    acc[{}] += float(v{row}[{x_index}]) * w{col};",
                col * rows + row
            ));
        }
    }
    lines.push("}".to_owned());
    lines.join("\n            ")
}

fn build_msg_source(rows: i32, bits: i32, group_size: i32) -> String {
    let n_acc = 4 * rows;
    let xloads = msg_x_loads(rows);
    let fma = msg_fma_block(rows, bits);
    let body = if bits == 4 {
        format!(
            r#"
        for (int pack = int(lane); pack < K_by_p; pack += 32) {{
            int k_base = pack * 8;
            int gi = k_base / GS;
            uint32_t p0 = w_q[(n0 + 0) * K_by_p + pack];
            uint32_t p1 = w_q[(n0 + 1) * K_by_p + pack];
            uint32_t p2 = w_q[(n0 + 2) * K_by_p + pack];
            uint32_t p3 = w_q[(n0 + 3) * K_by_p + pack];
            {xloads}
            float s0 = float(scales[(n0 + 0) * K_by_gs + gi]);
            float s1 = float(scales[(n0 + 1) * K_by_gs + gi]);
            float s2 = float(scales[(n0 + 2) * K_by_gs + gi]);
            float s3 = float(scales[(n0 + 3) * K_by_gs + gi]);
            float b0 = float(biases[(n0 + 0) * K_by_gs + gi]);
            float b1 = float(biases[(n0 + 1) * K_by_gs + gi]);
            float b2 = float(biases[(n0 + 2) * K_by_gs + gi]);
            float b3 = float(biases[(n0 + 3) * K_by_gs + gi]);
            {fma}
        }}"#
        )
    } else {
        format!(
            r#"
        for (int pair = int(lane); pair < K_by_p; pair += 32) {{
            int k_base = pair * 8;
            int gi = k_base / GS;
            {xloads}
            for (int wsel = 0; wsel < 2; ++wsel) {{
                int koff = wsel * 4;
                uint32_t p0 = w_q[(n0 + 0) * (K / 4) + pair * 2 + wsel];
                uint32_t p1 = w_q[(n0 + 1) * (K / 4) + pair * 2 + wsel];
                uint32_t p2 = w_q[(n0 + 2) * (K / 4) + pair * 2 + wsel];
                uint32_t p3 = w_q[(n0 + 3) * (K / 4) + pair * 2 + wsel];
                float s0 = float(scales[(n0 + 0) * K_by_gs + gi]);
                float s1 = float(scales[(n0 + 1) * K_by_gs + gi]);
                float s2 = float(scales[(n0 + 2) * K_by_gs + gi]);
                float s3 = float(scales[(n0 + 3) * K_by_gs + gi]);
                float b0 = float(biases[(n0 + 0) * K_by_gs + gi]);
                float b1 = float(biases[(n0 + 1) * K_by_gs + gi]);
                float b2 = float(biases[(n0 + 2) * K_by_gs + gi]);
                float b3 = float(biases[(n0 + 3) * K_by_gs + gi]);
                {fma}
            }}
        }}"#
        )
    };
    format!(
        r#"
        constexpr int GS = {group_size};
        constexpr int NSG = {MSG_NSG};
        constexpr int MROWS = {rows};
        uint sg = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint tg_n = threadgroup_position_in_grid.y;
        int K = K_SIZE;
        int N = N_SIZE;
        int K_by_p = K / 8;
        int K_by_gs = K / GS;
        int n0 = (int(tg_n) * NSG + int(sg)) * 4;
        if (n0 + 3 >= N) {{ return; }}
        float acc[{n_acc}];
        for (int i = 0; i < {n_acc}; ++i) {{ acc[i] = 0.0f; }}
        using Vec8 = vec<T, 8>;
        const device Vec8 *xv = (const device Vec8*)x;
        {body}
        for (int i = 0; i < {n_acc}; ++i) {{ acc[i] = simd_sum(acc[i]); }}
        if (lane < {n_acc}) {{
            int j = int(lane) / MROWS;
            int row = int(lane) - j * MROWS;
            y[row * N + n0 + j] = T(acc[int(lane)]);
        }}
    "#
    )
}

fn split_pack_block(rows: i32, bits: i32, suffix: &str) -> String {
    let pack = format!("pack{suffix}");
    let mut lines = vec![
        format!("int k_base{suffix} = {pack} * 8;"),
        format!("int gi{suffix} = k_base{suffix} / GS;"),
        x_loads(rows, suffix),
    ];
    if bits == 4 {
        for col in 0..4 {
            lines.push(format!(
                "uint32_t p{suffix}_{col} = w_q[(n0 + {col}) * K_by_p + {pack}];"
            ));
        }
        for col in 0..4 {
            lines.push(format!(
                "float s{suffix}_{col} = float(scales[(n0 + {col}) * K_by_gs + gi{suffix}]); float b{suffix}_{col} = float(biases[(n0 + {col}) * K_by_gs + gi{suffix}]);"
            ));
        }
        for col in 0..4 {
            lines.push("{".to_owned());
            lines.push(format!("uint32_t packed = p{suffix}_{col};"));
            lines.push(format!("float s = s{suffix}_{col};"));
            lines.push(format!("float b = b{suffix}_{col};"));
            lines.push("for (int ki = 0; ki < 8; ++ki) {".to_owned());
            lines.push("float wv = float((packed >> (ki * 4)) & 0xFu) * s + b;".to_owned());
            for row in 0..rows {
                lines.push(format!(
                    "acc[{}] += float(v{suffix}_{row}[ki]) * wv;",
                    col * rows + row
                ));
            }
            lines.push("}".to_owned());
            lines.push("}".to_owned());
        }
    } else {
        for col in 0..4 {
            lines.push(format!(
                "uint32_t pa{suffix}_{col} = w_q[(n0 + {col}) * K_by_w + {pack} * 2]; uint32_t pb{suffix}_{col} = w_q[(n0 + {col}) * K_by_w + {pack} * 2 + 1];"
            ));
        }
        for col in 0..4 {
            lines.push(format!(
                "float s{suffix}_{col} = float(scales[(n0 + {col}) * K_by_gs + gi{suffix}]); float b{suffix}_{col} = float(biases[(n0 + {col}) * K_by_gs + gi{suffix}]);"
            ));
        }
        for col in 0..4 {
            lines.push("{".to_owned());
            lines.push(format!("uint32_t pa = pa{suffix}_{col};"));
            lines.push(format!("uint32_t pb = pb{suffix}_{col};"));
            lines.push(format!("float s = s{suffix}_{col};"));
            lines.push(format!("float b = b{suffix}_{col};"));
            lines.push("for (int ki = 0; ki < 4; ++ki) {".to_owned());
            lines.push("float wa = float((pa >> (ki * 8)) & 0xFFu) * s + b;".to_owned());
            lines.push("float wb = float((pb >> (ki * 8)) & 0xFFu) * s + b;".to_owned());
            for row in 0..rows {
                lines.push(format!(
                    "acc[{}] += float(v{suffix}_{row}[ki]) * wa;",
                    col * rows + row
                ));
                lines.push(format!(
                    "acc[{}] += float(v{suffix}_{row}[ki + 4]) * wb;",
                    col * rows + row
                ));
            }
            lines.push("}".to_owned());
            lines.push("}".to_owned());
        }
    }
    lines.join("\n            ")
}

fn build_splitk_source(rows: i32, bits: i32, group_size: i32, k_parts: i32) -> String {
    let n_acc = 4 * rows;
    let pack_block = split_pack_block(rows, bits, "A");
    format!(
        r#"
        constexpr int GS = {group_size};
        constexpr int K_PARTS = {k_parts};
        uint part = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint tg_n = threadgroup_position_in_grid.y;
        int K = K_SIZE;
        int N = N_SIZE;
        int K_by_p = K / 8;
        int K_by_w = K / 4;
        int K_by_gs = K / GS;
        int per_part = K_by_p / K_PARTS;
        int n0 = int(tg_n) * 4;
        int p_start = int(part) * per_part;
        int p_end = (int(part) == K_PARTS - 1) ? K_by_p : p_start + per_part;
        float acc[{n_acc}];
        for (int i = 0; i < {n_acc}; ++i) {{ acc[i] = 0.0f; }}
        using Vec8 = vec<T, 8>;
        const device Vec8 *xv = (const device Vec8*)x;
        for (int packA = p_start + int(lane); packA < p_end; packA += 32) {{
            {pack_block}
        }}
        for (int i = 0; i < {n_acc}; ++i) {{ acc[i] = simd_sum(acc[i]); }}
        threadgroup float partials[K_PARTS * {n_acc}];
        if (lane == 0) {{
            for (int i = 0; i < {n_acc}; ++i) {{
                partials[int(part) * {n_acc} + i] = acc[i];
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (part == 0 && lane < {n_acc}) {{
            float total = 0.0f;
            for (int p = 0; p < K_PARTS; ++p) {{
                total += partials[p * {n_acc} + int(lane)];
            }}
            int j = int(lane) / {rows};
            int row = int(lane) - j * {rows};
            y[row * N + n0 + j] = T(total);
        }}
    "#
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    fn quantized_parts(n: i32, k: i32, bits: i32, group_size: i32) -> (Array, Array, Array) {
        let packed_k = k * bits / 32;
        let weight_data: Vec<u32> = (0..n * packed_k)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761).rotate_left(7))
            .collect();
        let groups = k / group_size;
        let scale_unit = if bits == 4 { 0.004 } else { 0.00025 };
        let scales_data: Vec<f32> = (0..n * groups)
            .map(|i| scale_unit * (1 + i % 5) as f32)
            .collect();
        let biases_data: Vec<f32> = (0..n * groups)
            .map(|i| -0.03 + (i % 7) as f32 * 0.002)
            .collect();
        let weight: Array = (weight_data.as_slice(), &[n, packed_k][..])
            .try_into()
            .unwrap();
        let scales_f32: Array = (scales_data.as_slice(), &[n, groups][..])
            .try_into()
            .unwrap();
        let biases_f32: Array = (biases_data.as_slice(), &[n, groups][..])
            .try_into()
            .unwrap();
        (
            weight,
            mlx::ops::cast::astype(&scales_f32, Dtype::Bfloat16).unwrap(),
            mlx::ops::cast::astype(&biases_f32, Dtype::Bfloat16).unwrap(),
        )
    }

    fn input(rows: i32, k: i32) -> Array {
        let data: Vec<f32> = (0..rows * k)
            .map(|i| ((i * 19 + 7) % 71) as f32 * 0.003 - 0.105)
            .collect();
        let f32_array: Array = (data.as_slice(), &[1_i32, rows, k][..]).try_into().unwrap();
        mlx::ops::cast::astype(&f32_array, Dtype::Bfloat16).unwrap()
    }

    fn assert_candidate_matches_native(
        rows: i32,
        n: i32,
        k: i32,
        bits: i32,
        expected_kind: VerifyQmmKind,
    ) {
        let group_size = 64;
        let (weight, scales, biases) = quantized_parts(n, k, bits, group_size);
        let x = input(rows, k);
        let parts = QuantizedLinearParts {
            weight: &weight,
            scales: &scales,
            biases: Some(&biases),
            bias: None,
            group_size,
            bits,
            mode: QuantMode::Affine,
        };
        assert_eq!(eligible_kind(&x, &parts), Some(expected_kind));

        let kind = eligible_kind(&x, &parts).expect("shape must be kernel-capable");
        let candidate = dispatch_candidate_on(&x, parts, kind, ()).unwrap();
        let native = mlx::quantization::quantized_matmul(
            &x,
            &weight,
            &scales,
            Some(&biases),
            true,
            Some(group_size),
            Some(bits),
            "affine",
        )
        .unwrap();

        assert_eq!(candidate.shape(), native.shape());
        let candidate = mlx::ops::cast::astype(&candidate, Dtype::Float32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let native = mlx::ops::cast::astype(&native, Dtype::Float32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let max_abs = candidate
            .iter()
            .zip(native.iter())
            .map(|(candidate, native)| (candidate - native).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs <= 0.05,
            "verify qmm differs from native: rows={rows} n={n} k={k} bits={bits} max_abs={max_abs}"
        );
    }

    #[test]
    #[serial(mlx_metal)]
    fn splitk_matches_native_for_affine_4bit_and_8bit() {
        for bits in [4, 8] {
            assert_candidate_matches_native(3, MIN_ROUTE_N, 512, bits, VerifyQmmKind::SplitK);
        }
    }

    #[test]
    #[serial(mlx_metal)]
    fn msg_matches_native_for_affine_4bit_and_8bit() {
        for bits in [4, 8] {
            assert_candidate_matches_native(4, MSG_MIN_N, 512, bits, VerifyQmmKind::Msg);
        }
    }

    #[test]
    fn eligibility_is_strictly_limited_to_mtp_verify_shapes() {
        let weight = Array::zeros((MIN_ROUTE_N, 8), Dtype::Uint32).unwrap();
        let scales = Array::zeros((MIN_ROUTE_N, 1), Dtype::Bfloat16).unwrap();
        let biases = Array::zeros((MIN_ROUTE_N, 1), Dtype::Bfloat16).unwrap();
        let x = Array::zeros((1, 3, 64), Dtype::Bfloat16).unwrap();
        let mut parts = QuantizedLinearParts {
            weight: &weight,
            scales: &scales,
            biases: Some(&biases),
            bias: None,
            group_size: 64,
            bits: 4,
            mode: QuantMode::Affine,
        };
        assert_eq!(eligible_kind(&x, &parts), Some(VerifyQmmKind::SplitK));

        let two_rows = Array::zeros((1, 2, 64), Dtype::Bfloat16).unwrap();
        assert_eq!(eligible_kind(&two_rows, &parts), None);
        let unbatched = Array::zeros((3, 64), Dtype::Bfloat16).unwrap();
        assert_eq!(eligible_kind(&unbatched, &parts), None);
        parts.mode = QuantMode::Mxfp4;
        assert_eq!(eligible_kind(&x, &parts), None);
        parts.mode = QuantMode::Affine;
        parts.biases = None;
        assert_eq!(eligible_kind(&x, &parts), None);
    }

    #[test]
    fn armed_scope_is_nested_and_restores_thread_local_state() {
        assert!(!is_armed());
        {
            let _outer = armed_scope();
            assert!(is_armed());
            {
                let _inner = armed_scope();
                assert!(is_armed());
            }
            assert!(is_armed());
        }
        assert!(!is_armed());
    }

    #[test]
    #[serial(mlx_metal)]
    fn current_route_profile_excludes_unproven_or_slower_shapes() {
        if mlx::metal::architecture().unwrap() != "applegpu_g17s" {
            return;
        }
        let weight = Array::zeros((MSG_MIN_N, 8), Dtype::Uint32).unwrap();
        let scales = Array::zeros((MSG_MIN_N, 1), Dtype::Bfloat16).unwrap();
        let biases = Array::zeros((MSG_MIN_N, 1), Dtype::Bfloat16).unwrap();
        let parts = QuantizedLinearParts {
            weight: &weight,
            scales: &scales,
            biases: Some(&biases),
            bias: None,
            group_size: 64,
            bits: 4,
            mode: QuantMode::Affine,
        };
        for rows in [3, 4] {
            let x = Array::zeros((1, rows, 64), Dtype::Bfloat16).unwrap();
            assert!(route_profile_allows(VerifyQmmKind::Msg, &x, &parts));
        }
        for rows in [2, 5, 6] {
            let x = Array::zeros((1, rows, 64), Dtype::Bfloat16).unwrap();
            assert!(!route_profile_allows(VerifyQmmKind::Msg, &x, &parts));
        }
    }
}
