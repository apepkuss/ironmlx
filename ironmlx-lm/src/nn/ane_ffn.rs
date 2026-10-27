//! The Neural Engine's share of the Dense long-prompt prefill FFN.
//!
//! A full 2048-row prefill chunk's MLP splits its intermediate channels: the
//! GPU computes the leading channels from its affine4 weights as before, the
//! ANE the trailing `units x 512` in W8A8 at the same time, and the GPU adds
//! the ANE's down projection to its own. The ANE takes Hadamard-rotated int8
//! operands: per token for the activations, per row for the weights, which
//! the GPU converts from the affine4 weights each chunk (no resident copy).
//! Rotations are orthogonal (block Walsh-Hadamard after fixed sign flips), so
//! the products are those of the unrotated operands up to the int8 rounding.
//!
//! Hand-offs use one Metal shared event: the GPU signals that a layer's ANE
//! operands are written, a worker thread waits for it, evaluates the ANE
//! program and signals back, and the GPU waits on that only to add the ANE's
//! output, after its own channels. The driving thread never blocks.
use crate::Result;
use ironmlx_ane::{Elem, Program, Surface};
use mlx::fast::HandoffEvent;
use mlx::{Array, Dtype, MetalKernel, Shape, StreamOrDevice};
use std::sync::{mpsc, Arc, Mutex, OnceLock};

/// int8 operands enter the ANE program divided by this (against fp16
/// overflow of its products); the per-row weight scales carry it back.
const INT8_UNIT: f32 = 128.0;
/// Walsh-Hadamard block of the hidden (input) rotation.
pub(crate) const INPUT_BLOCK: i32 = 1024;
/// Walsh-Hadamard block of the intermediate rotation (a power of 4, so the
/// ANE's fp16 rotation entries, +-1/sqrt(block), are exact).
pub(crate) const MID_BLOCK: i32 = 256;
const INPUT_SEED: u32 = 0x5eed_0001;
const MID_SEED: u32 = 0x5eed_0002;
/// Smallest per-token peak of the rotated intermediate the ANE quantizes
/// against: 127 / peak must stay below fp16's largest finite value.
const MID_PEAK_FLOOR: f32 = 1.0 / 512.0;
/// Channels a share moves in.
pub(crate) const UNIT_CHANNELS: i32 = 512;
/// The most channels the ANE takes (20 units, the largest program measured
/// to build and run).
const MAX_ANE_CHANNELS: usize = 10240;
/// How long the worker waits for the GPU's operands before it gives up.
const READY_TIMEOUT_MS: u64 = 10_000;

/// The whole surface (`rows x stride`, rows padded to 64 bytes) as an MLX
/// array over the same memory. The surface must outlive the array.
pub(crate) fn surface_array(surface: &Surface) -> Result<Array> {
    let dtype = match surface.elem {
        Elem::I8 => Dtype::Int8,
        Elem::F16 => Dtype::Float16,
    };
    // SAFETY: the IOSurface allocation is page aligned and page sized; the
    // callers keep the surface alive while the array is used.
    Ok(unsafe {
        mlx::fast::array_from_external_buffer(
            surface.base() as usize,
            surface.alloc_bytes,
            &[surface.rows as i32, surface.stride() as i32],
            dtype,
        )?
    })
}

/// The sign flip of channel `index` of a rotation seeded `seed` (murmur3's
/// finalizer; the Metal kernels compute the same).
fn flips(seed: u32, index: u32) -> bool {
    let mut h = index ^ seed;
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    h & 1 == 1
}

/// One simdgroup per 256 elements, each thread `E = 8` consecutive ones in
/// registers: butterflies of strides below 8 in registers, up to 128 across
/// lanes (`simd_shuffle_xor`); for 1024-element blocks (H_1024 = H_4 (x)
/// H_256) a 4-point stage across simdgroups through threadgroup memory. One
/// threadgroup per row; rows longer than a threadgroup's 32 simdgroups take
/// `P` 256-element chunks per simdgroup (256-element blocks only). `LOAD`
/// fills `v[0..E]` from column `c0`.
const ROTATE_SOURCE: &str = "\
    constexpr int E = 8, P = CHUNKS;\n\
    const int L = dims[1], row0 = dims[2], col0 = dims[3], ldw = dims[4], ldg = dims[5], block = dims[6];\n\
    const uint seed = uint(dims[7]);\n\
    const uint r = threadgroup_position_in_grid.y;\n\
    const uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;\n\
    const uint groups = threads_per_threadgroup.x / 32;\n\
    threadgroup float red[32];\n\
    const int64_t row = int64_t(row0) + r;\n\
    float vv[P][E];\n\
    for (int p = 0; p < P; p++) {\n\
    const int base = (p * int(groups) + int(sg)) * (E * 32) + int(lane) * E;\n\
    const int c0 = col0 + base;\n\
    float v[E];\n\
    LOAD\n\
    for (int j = 0; j < E; j++) {\n\
      uint h = uint(base + j) ^ seed; h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16;\n\
      if (h & 1u) v[j] = -v[j];\n\
    }\n\
    for (int h = 1; h < E; h <<= 1)\n\
      for (int j = 0; j < E; j++)\n\
        if ((j & h) == 0) { const float a = v[j], b = v[j + h]; v[j] = a + b; v[j + h] = a - b; }\n\
    for (ushort m = 1; m < 32; m <<= 1)\n\
      for (int j = 0; j < E; j++) {\n\
        const float o = simd_shuffle_xor(v[j], m);\n\
        v[j] = (lane & m) ? (o - v[j]) : (v[j] + o);\n\
      }\n\
    WIDE_BEGIN\n\
      threadgroup float ex[5120];\n\
      for (int j = 0; j < E; j++) ex[base + j] = v[j];\n\
      threadgroup_barrier(mem_flags::mem_threadgroup);\n\
      const int hi = int(sg) & 3, first = (int(sg) - hi) * 256;\n\
      for (int j = 0; j < E; j++) {\n\
        float acc = 0.0f;\n\
        for (int k = 0; k < 4; k++) {\n\
          const float a = ex[first + k * 256 + int(lane) * E + j];\n\
          acc += (popcount(uint(hi & k)) & 1) ? -a : a;\n\
        }\n\
        v[j] = acc;\n\
      }\n\
    WIDE_END\n\
    for (int j = 0; j < E; j++) vv[p][j] = v[j];\n\
    }\n\
    float mx = 0.0f;\n\
    for (int p = 0; p < P; p++)\n\
      for (int j = 0; j < E; j++) mx = max(mx, abs(vv[p][j]));\n\
    mx = simd_max(mx);\n\
    if (lane == 0) red[sg] = mx;\n\
    threadgroup_barrier(mem_flags::mem_threadgroup);\n\
    if (sg == 0) {\n\
      float g = lane < groups ? red[lane] : 0.0f;\n\
      g = simd_max(g);\n\
      if (lane == 0) red[0] = g;\n\
    }\n\
    threadgroup_barrier(mem_flags::mem_threadgroup);\n\
    mx = red[0];\n\
    const float inv = mx > 0.0f ? 127.0f / mx : 0.0f;\n\
    for (int p = 0; p < P; p++) {\n\
      const int base = (p * int(groups) + int(sg)) * (E * 32) + int(lane) * E;\n\
      device char4* q = (device char4*)((device int8_t*)Q + int64_t(r) * L + base);\n\
      for (int j = 0; j < E; j += 4)\n\
        q[j / 4] = char4(clamp(rint(float4(vv[p][j], vv[p][j + 1], vv[p][j + 2], vv[p][j + 3]) * inv), -127.0f, 127.0f));\n\
    }\n\
    if (sg == 0 && lane == 0) S[r] = mx / (127.0f * sqrt(float(block)));";

/// Kernels of one rotation source: 256-element blocks in one chunk per
/// simdgroup, 1024-element blocks, 256-element blocks in two chunks.
type RotateKernels = [OnceLock<MetalKernel>; 3];

/// Rows of `[R, L]` sign-flipped (`seed`), Walsh-Hadamard rotated in blocks of
/// `block` (orthonormal), and quantized to int8 per row: `(q [R, L], scale
/// [R] f32)` with rotated value `q * scale`.
fn rotate_quantize(
    kernels: &'static RotateKernels,
    name: &str,
    load: &str,
    inputs: &[&Array],
    dims: [i32; 8],
    target: StreamOrDevice,
) -> Result<(Array, Array)> {
    let (rows, len, block) = (dims[0], dims[1], dims[6]);
    let wide = block == 1024;
    // One 256-element chunk per simdgroup up to a threadgroup's 32; two beyond.
    let chunks = if len <= 8192 { 1 } else { 2 };
    anyhow::ensure!(
        (block == 256 || block == 1024)
            && len % (256 * chunks) == 0
            && len % block == 0
            && len <= 16384
            && (!wide || len <= 5120),
        "rotation rows of {len} with block {block}"
    );
    let kernel = &kernels[if wide {
        1
    } else if chunks == 2 {
        2
    } else {
        0
    }];
    if kernel.get().is_none() {
        let mut names: Vec<String> = (0..inputs.len()).map(|i| format!("I{i}")).collect();
        names.push("dims".into());
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        // The 4-point stage (and its threadgroup memory) only in the
        // 1024-block kernel.
        let (begin, end) = if wide {
            ("{", "}")
        } else {
            ("#if 0", "#endif")
        };
        let source = ROTATE_SOURCE
            .replace("CHUNKS", &chunks.to_string())
            .replace("LOAD", load)
            .replace("WIDE_BEGIN", begin)
            .replace("WIDE_END", end);
        let built = MetalKernel::builder(format!("{name}_b{block}_p{chunks}"))
            .inputs(&names)
            .outputs(&["Q", "S"])
            .source(source)
            .build()?;
        let _ = kernel.set(built);
    }
    let dims: Array = (&dims[..], &[8][..]).try_into()?;
    let mut all: Vec<&Array> = inputs.to_vec();
    all.push(&dims);
    let threads = 32 * (len / (256 * chunks));
    let mut outs = kernel
        .get()
        .unwrap()
        .dispatch_builder()
        .inputs(&all)
        .output_shapes(&[Shape::from(&[rows, len][..]), Shape::from(&[rows][..])])
        .output_dtypes(&[Dtype::Int8, Dtype::Float32])
        .grid(threads, rows, 1)
        .threadgroup(threads, 1, 1)
        .stream(target)
        .dispatch()?;
    Ok((outs.take_at(0)?, outs.take_at(0)?))
}

/// Rows of a row-major BF16 `[R, L]` (the activations).
fn rotate_quantize_dense(
    x: &Array,
    block: i32,
    seed: u32,
    target: StreamOrDevice,
) -> Result<(Array, Array)> {
    static KERNELS: RotateKernels = [OnceLock::new(), OnceLock::new(), OnceLock::new()];
    let dims = x.shape();
    let (rows, len) = (dims.as_slice()[0], dims.as_slice()[1]);
    rotate_quantize(
        &KERNELS,
        "ironmlx_ane_rotate_dense_v3",
        "const device bfloat* src = (const device bfloat*)I0 + row * ldw + c0;\n\
         for (int j = 0; j < E; j++) v[j] = float(src[j]);",
        &[x],
        [rows, len, 0, 0, len, 0, block, seed as i32],
        target,
    )
}

/// Rows `row0..row0 + rows`, columns `col0..col0 + len` of a native affine4
/// weight (group 64), dequantized in f32.
#[allow(clippy::too_many_arguments)]
fn rotate_quantize_affine4(
    w: &Array,
    scales: &Array,
    biases: &Array,
    row0: i32,
    rows: i32,
    col0: i32,
    len: i32,
    block: i32,
    seed: u32,
    target: StreamOrDevice,
) -> Result<(Array, Array)> {
    static KERNELS: RotateKernels = [OnceLock::new(), OnceLock::new(), OnceLock::new()];
    anyhow::ensure!(
        col0 % 64 == 0,
        "affine4 column offset {col0} splits a group"
    );
    let words = w.shape().as_slice()[1];
    let groups = scales.shape().as_slice()[1];
    let scales = scales.astype_on(Dtype::Float32, target)?;
    let biases = biases.astype_on(Dtype::Float32, target)?;
    // A thread's 8 consecutive columns lie in one group of 64.
    rotate_quantize(
        &KERNELS,
        "ironmlx_ane_rotate_affine4_v3",
        "const float sc = I1[row * ldg + c0 / 64], bi = I2[row * ldg + c0 / 64];\n\
         const device uint* src = (const device uint*)I0 + row * ldw + c0 / 8;\n\
         for (int w = 0; w < E / 8; w++) {\n\
           const uint word = src[w];\n\
           for (int k = 0; k < 8; k++) v[w * 8 + k] = sc * float((word >> (4 * k)) & 0xFu) + bi;\n\
         }",
        &[w, &scales, &biases],
        [rows, len, row0, col0, words, groups, block, seed as i32],
        target,
    )
}

/// [`rotate_quantize_affine4`] reading the M5 tiled layout
/// (`m5_affine4::prepare`: tiles of `nt` columns, each 64-input group of a
/// tile's columns contiguous; `(scale, bias)` pairs `[K/64, N]` bf16): the same
/// dequantized values, so the same rotated codes.
#[allow(clippy::too_many_arguments)]
fn rotate_quantize_tiled(
    prepared: &super::m5_affine4::Prepared,
    row0: i32,
    rows: i32,
    col0: i32,
    len: i32,
    block: i32,
    seed: u32,
    target: StreamOrDevice,
) -> Result<(Array, Array)> {
    static KERNELS32: RotateKernels = [OnceLock::new(), OnceLock::new(), OnceLock::new()];
    static KERNELS64: RotateKernels = [OnceLock::new(), OnceLock::new(), OnceLock::new()];
    anyhow::ensure!(
        col0 % 64 == 0,
        "affine4 column offset {col0} splits a group"
    );
    let (n, words) = (
        prepared.weight.shape().as_slice()[0],
        prepared.weight.shape().as_slice()[1],
    );
    let nt = super::m5_affine4::tile_width(n);
    let (kernels, name, load): (&'static RotateKernels, &str, String) = if nt == 64 {
        (&KERNELS64, "ironmlx_ane_rotate_tiled64_v1", tiled_load(64))
    } else {
        (&KERNELS32, "ironmlx_ane_rotate_tiled32_v1", tiled_load(32))
    };
    rotate_quantize(
        kernels,
        name,
        &load,
        &[&prepared.weight, &prepared.sb],
        [rows, len, row0, col0, words, n, block, seed as i32],
        target,
    )
}

/// `LOAD` of [`rotate_quantize_tiled`]: a thread's 8 consecutive columns are
/// one word of one group (`ldw` words per row, `ldg` = N columns).
fn tiled_load(nt: i32) -> String {
    format!(
        "const int grp = c0 / 64, wi = c0 / 8;\n\
         const device bfloat* sbp = (const device bfloat*)I1 + (int64_t(grp) * ldg + row) * 2;\n\
         const float sc = float(sbp[0]), bi = float(sbp[1]);\n\
         const int64_t at = ((row / {nt}) * (ldw / 8) + wi / 8) * ({nt} * 8) + (row % {nt}) * 8 + (wi % 8);\n\
         const uint word = ((const device uint*)I0)[at];\n\
         for (int k = 0; k < 8; k++) v[k] = sc * float((word >> (4 * k)) & 0xFu) + bi;"
    )
}

/// `out [T, H] bf16 = partial + Y[0..H]^T * Y[H] * xs`: the GPU's channels
/// plus the ANE's down projection at full scale (in f32).
fn join(partial: &Array, y: &Array, xs: &Array, target: StreamOrDevice) -> Result<Array> {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    if KERNEL.get().is_none() {
        let built = MetalKernel::builder("ironmlx_ane_join_v1")
            .inputs(&["P", "Y", "XS", "dims"])
            .outputs(&["O"])
            .source(
                "const int T = dims[0], H = dims[1], ld = dims[2];\n\
                 threadgroup float tile[32][33];\n\
                 const uint2 tg = threadgroup_position_in_grid.xy;\n\
                 const uint2 lt = thread_position_in_threadgroup.xy;\n\
                 const int t0 = tg.x * 32, c0 = tg.y * 32;\n\
                 for (int k = lt.y; k < 32; k += 8) {\n\
                   const int c = c0 + k, t = t0 + lt.x;\n\
                   tile[k][lt.x] = (c < H && t < T) ? float(Y[int64_t(c) * ld + t]) * float(Y[int64_t(H) * ld + t]) * XS[t] : 0.0f;\n\
                 }\n\
                 threadgroup_barrier(mem_flags::mem_threadgroup);\n\
                 for (int k = lt.y; k < 32; k += 8) {\n\
                   const int t = t0 + k, c = c0 + lt.x;\n\
                   if (t < T && c < H) O[int64_t(t) * H + c] = bfloat(float(P[int64_t(t) * H + c]) + tile[lt.x][k]);\n\
                 }",
            )
            .build()?;
        let _ = KERNEL.set(built);
    }
    let pd = partial.shape();
    let (t, h) = (pd.as_slice()[0], pd.as_slice()[1]);
    let ld = y.shape().as_slice()[1];
    let dims: Array = (&[t, h, ld][..], &[3][..]).try_into()?;
    let mut outs = KERNEL
        .get()
        .unwrap()
        .dispatch_builder()
        .inputs(&[partial, y, xs, &dims])
        .output_shapes(&[Shape::from(&[t, h][..])])
        .output_dtypes(&[Dtype::Bfloat16])
        .grid((t + 31) / 32 * 32, (h + 31) / 32 * 8, 1)
        .threadgroup(32, 8, 1)
        .stream(target)
        .dispatch()?;
    Ok(outs.take_at(0)?)
}

fn scalar(v: f32) -> Array {
    Array::try_from((&[v][..], ())).expect("scalar array")
}

/// MIL fp16 literal (exact for any normal f32 value).
fn hex(v: f32) -> String {
    if v == 0.0 {
        return "fp16(0x0p+0)".into();
    }
    let bits = v.abs().to_bits();
    let (exp, mant) = ((bits >> 23) as i32 - 127, bits & 0x7f_ffff);
    let sign = if v < 0.0 { "-" } else { "" };
    format!("fp16({sign}0x1.{:06x}p{exp:+})", mant << 1)
}

/// How the ANE program splits its work (timing variants; production uses
/// `Variant::default()`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Variant {
    /// Matmuls of gate/up over this many segments of the hidden inputs.
    pub(crate) input_segments: usize,
    /// The down matmul over this many segments of the intermediate inputs.
    pub(crate) mid_segments: usize,
    /// Rotate the intermediate before its int8 quantization.
    pub(crate) rotate_mid: bool,
    /// The hidden inputs enter as fp16 (rotated, unquantized): W8A16 gate/up.
    pub(crate) fp16_input: bool,
    /// The intermediate stays fp16 (rotated, unquantized): W8A16 down.
    pub(crate) fp16_mid: bool,
}

impl Default for Variant {
    fn default() -> Self {
        // Two segments each measured fastest (8 units, 2048 rows, M5 Max:
        // 12.5 ms against 15.2 unsegmented; 4 or 5 segments slower).
        Self {
            input_segments: 2,
            mid_segments: 2,
            rotate_mid: true,
            fp16_input: false,
            fp16_mid: false,
        }
    }
}

/// `sum_k matmul(w[:, k-th segment], v[k-th segment rows])` over `segments`
/// equal segments of the contraction dimension `k` (statements appended to
/// `line`, the result named `name`).
#[allow(clippy::too_many_arguments)]
fn segmented_matmul(
    line: &mut dyn FnMut(String),
    name: &str,
    w: &str,
    v: &str,
    out_rows: usize,
    k: usize,
    cols: usize,
    segments: usize,
) {
    let t = |r: usize, c: usize| format!("tensor<fp16, [1, 1, {r}, {c}]>");
    if segments <= 1 {
        line(format!(
            "{} {name} = matmul(transpose_x = bool(false), transpose_y = bool(false), x = {w}, y = {v})",
            t(out_rows, cols)
        ));
        return;
    }
    let seg = k / segments;
    let mut total = String::new();
    for i in 0..segments {
        let b = i * seg;
        line(format!(
            "{} {name}_w{i} = slice_by_size(x = {w}, begin = tensor<int32, [4]>([0, 0, 0, {b}]), size = tensor<int32, [4]>([1, 1, {out_rows}, {seg}]))",
            t(out_rows, seg)
        ));
        line(format!(
            "{} {name}_v{i} = slice_by_size(x = {v}, begin = tensor<int32, [4]>([0, 0, {b}, 0]), size = tensor<int32, [4]>([1, 1, {seg}, {cols}]))",
            t(seg, cols)
        ));
        line(format!(
            "{} {name}_m{i} = matmul(transpose_x = bool(false), transpose_y = bool(false), x = {name}_w{i}, y = {name}_v{i})",
            t(out_rows, cols)
        ));
        total = if i == 0 {
            format!("{name}_m0")
        } else {
            let next = format!("{name}_s{i}");
            line(format!(
                "{} {next} = add(x = {total}, y = {name}_m{i})",
                t(out_rows, cols)
            ));
            next
        };
    }
    line(format!(
        "{} {name} = identity(x = {total})",
        t(out_rows, cols)
    ));
}

/// The ANE program body over `rows` tokens: hidden `h`, `a` ANE channels.
fn program_text(rows: usize, h: usize, a: usize, mid_block: usize, variant: Variant) -> String {
    let t = |dt: &str, r: usize, c: usize| format!("tensor<{dt}, [1, 1, {r}, {c}]>");
    let mut body = String::new();
    let mut line = |s: String| {
        body.push_str("        ");
        body.push_str(&s);
        body.push_str(";\n");
    };
    let unit = hex(1.0 / INT8_UNIT);
    let xt = if variant.fp16_input { "fp16" } else { "int8" };
    for (name, dt, r, c) in [
        ("x", xt, h, rows),
        ("tx", "fp16", 1, rows),
        ("wg", "int8", a, h),
        ("wu", "int8", a, h),
        ("sg", "fp16", a, 1),
        ("su", "fp16", a, 1),
        ("wd", "int8", h, a),
        ("sd", "fp16", h, 1),
    ] {
        line(format!(
            "{} {name}_t = tensor_buffer_to_tensor<ios17>(input = {name})",
            t(dt, r, c)
        ));
    }
    for (name, r, c) in [("x", h, rows), ("wg", a, h), ("wu", a, h), ("wd", h, a)] {
        if name == "x" && variant.fp16_input {
            line(format!("{} x_d = identity(x = x_t)", t("fp16", r, c)));
        } else {
            line(format!(
                "{} {name}_d = dequantize(input = {name}_t, scale = {unit})",
                t("fp16", r, c)
            ));
        }
    }
    let f = |r: usize| t("fp16", r, rows);
    segmented_matmul(
        &mut line,
        "gm",
        "wg_d",
        "x_d",
        a,
        h,
        rows,
        variant.input_segments,
    );
    segmented_matmul(
        &mut line,
        "um",
        "wu_d",
        "x_d",
        a,
        h,
        rows,
        variant.input_segments,
    );
    line(format!("{} gs = mul(x = gm, y = sg_t)", f(a)));
    line(format!("{} g = mul(x = gs, y = tx_t)", f(a)));
    line(format!("{} us = mul(x = um, y = su_t)", f(a)));
    // silu(g) = g/2 (1 + tanh(g/2)).
    line(format!("{} gh = mul(x = g, y = {})", f(a), hex(0.5)));
    line(format!("{} th = tanh(x = gh)", f(a)));
    line(format!("{} tp = add(x = th, y = {})", f(a), hex(1.0)));
    line(format!("{} sl = mul(x = gh, y = tp)", f(a)));
    line(format!("{} hm = mul(x = sl, y = us)", f(a)));
    let blocks = a / mid_block;
    if !variant.rotate_mid {
        line(format!("{} hr = identity(x = hm)", f(a)));
    } else {
        line(format!(
        "tensor<fp16, [1, {a}, 1, {rows}]> h4 = reshape(x = hm, shape = tensor<int32, [4]>([1, {a}, 1, {rows}]))"
    ));
        line(format!(
        "tensor<fp16, [{a}, {b}, 1, 1]> rot = const()[name = string(\"rot\"), val = tensor<fp16, [{a}, {b}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights.bin\"), offset = uint64({off})))]",
        b = mid_block,
        off = ironmlx_ane::CONSTANT_OFFSET
    ));
        line(format!(
        "tensor<fp16, [1, {a}, 1, {rows}]> hr4 = conv(dilations = tensor<int32, [2]>([1, 1]), groups = int32({blocks}), pad = tensor<int32, [4]>([0, 0, 0, 0]), pad_type = string(\"valid\"), strides = tensor<int32, [2]>([1, 1]), weight = rot, x = h4)"
    ));
        line(format!(
            "{} hr = reshape(x = hr4, shape = tensor<int32, [4]>([1, 1, {a}, {rows}]))",
            f(a)
        ));
    }
    if variant.fp16_mid {
        line(format!("{} hd = identity(x = hr)", f(a)));
        line(format!(
            "{} hsc = reduce_max(x = tx_t, axes = tensor<int32, [1]>([2]), keep_dims = bool(true))",
            f(1)
        ));
    } else {
        line(format!("{} ha = abs(x = hr)", f(a)));
        line(format!(
            "{} pk = reduce_max(x = ha, axes = tensor<int32, [1]>([2]), keep_dims = bool(true))",
            f(1)
        ));
        line(format!(
            "{} fl = maximum(x = pk, y = {})",
            f(1),
            hex(MID_PEAK_FLOOR)
        ));
        line(format!(
            "{} iv = real_div(x = {}, y = fl)",
            f(1),
            hex(127.0)
        ));
        line(format!("{} hs = mul(x = hr, y = iv)", f(a)));
        line(format!(
            "{} hq = quantize(input = hs, scale = {}, output_dtype = string(\"int8\"))",
            t("int8", a, rows),
            hex(1.0)
        ));
        line(format!(
            "{} hd = dequantize(input = hq, scale = {unit})",
            f(a)
        ));
        line(format!(
            "{} hsc = mul(x = fl, y = {})",
            f(1),
            hex(1.0 / 127.0)
        ));
    }
    segmented_matmul(
        &mut line,
        "dm",
        "wd_d",
        "hd",
        h,
        a,
        rows,
        variant.mid_segments,
    );
    line(format!("{} ds = mul(x = dm, y = sd_t)", f(h)));
    line(format!(
        "{} yt = concat(axis = int32(2), interleave = bool(false), values = (ds, hsc))",
        f(h + 1)
    ));
    body
}

/// The ANE's intermediate rotation as a grouped 1x1 convolution's weights
/// (`[a, MID_BLOCK, 1, 1]` fp16 bit patterns): output channel `j * B + o` is
/// `sum_i H[o, i] d[j * B + i] h[j * B + i] / sqrt(B)`.
fn mid_rotation(a: usize, b: usize) -> Vec<u16> {
    // +-1/sqrt(b) = +-2^-k, exact in fp16.
    let k = b.trailing_zeros() / 2;
    let magnitude = ((15 - k) as u16) << 10;
    let mut values = vec![0u16; a * b];
    for j in 0..a / b {
        for o in 0..b {
            for i in 0..b {
                let negative =
                    ((o & i).count_ones() & 1 == 1) ^ flips(MID_SEED, (j * b + i) as u32);
                values[(j * b + o) * b + i] = magnitude | if negative { 0x8000 } else { 0 };
            }
        }
    }
    values
}

struct Surfaces {
    x: Surface,
    tx: Surface,
    wg: Surface,
    wu: Surface,
    sg: Surface,
    su: Surface,
    wd: Surface,
    sd: Surface,
    y: Surface,
}

impl Surfaces {
    fn new(rows: usize, hidden: usize, ane: usize, variant: Variant) -> Result<Self> {
        Ok(Self {
            x: Surface::new(
                hidden,
                rows,
                if variant.fp16_input {
                    Elem::F16
                } else {
                    Elem::I8
                },
            )?,
            tx: Surface::new(1, rows, Elem::F16)?,
            wg: Surface::new(ane, hidden, Elem::I8)?,
            wu: Surface::new(ane, hidden, Elem::I8)?,
            sg: Surface::new(ane, 1, Elem::F16)?,
            su: Surface::new(ane, 1, Elem::F16)?,
            wd: Surface::new(hidden, ane, Elem::I8)?,
            sd: Surface::new(hidden, 1, Elem::F16)?,
            y: Surface::new(hidden + 1, rows, Elem::F16)?,
        })
    }

    fn inputs(&self) -> [(&'static str, &Surface); 8] {
        self.inputs_with(self)
    }

    /// This set's input (`x`, `tx`) with the weights of `weights`.
    fn inputs_with<'a>(&'a self, weights: &'a Surfaces) -> [(&'static str, &'a Surface); 8] {
        [
            ("x", &self.x),
            ("tx", &self.tx),
            ("wg", &weights.wg),
            ("wu", &weights.wu),
            ("sg", &weights.sg),
            ("su", &weights.su),
            ("wd", &weights.wd),
            ("sd", &weights.sd),
        ]
    }
}

/// The ANE program over `rows` tokens of a `hidden`-wide MLP's `ane`
/// trailing channels, and its surfaces.
fn build(
    rows: usize,
    hidden: usize,
    intermediate: usize,
    ane: usize,
    mid_block: usize,
    variant: Variant,
) -> Result<(Program, Arc<Surfaces>)> {
    ironmlx_ane::availability().map_err(|reason| anyhow::anyhow!("ANE unavailable: {reason}"))?;
    anyhow::ensure!(
        ane > 0
            && ane < intermediate
            && hidden <= 5120
            && ane <= MAX_ANE_CHANNELS
            && hidden.is_multiple_of(512),
        "ANE FFN share of {ane} of {intermediate} channels, hidden {hidden}"
    );
    let surfaces = Arc::new(Surfaces::new(rows, hidden, ane, variant)?);
    let s = &surfaces;
    anyhow::ensure!(
        s.sg.stride() == 32 && s.sd.stride() == 32,
        "scale surfaces of another stride"
    );
    let signature = s
        .inputs()
        .iter()
        .map(|(name, surface)| format!("{} {name}", surface.buffer_type(surface.rows)))
        .collect::<Vec<_>>()
        .join(", ");
    let ys = s.y.stride();
    let plane = s.y.rows * ys;
    let mil = format!(
        "program(1.3)\n{{\n    func f<ios18>({signature}) {{\n{}        {} y = tensor_to_tensor_buffer<ios17>(input = yt, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>([{plane}, {plane}, {ys}, 1]));\n    }} -> (y);\n}}\n",
        program_text(rows, hidden, ane, mid_block, variant),
        s.y.buffer_type(s.y.rows)
    );
    let program = Program::load(
        &mil,
        &ironmlx_ane::fp16_constant_weights(&mid_rotation(ane, mid_block)),
    )?;
    Ok((program, surfaces))
}

struct Job {
    ready: u64,
    done: u64,
    /// The surface set of the input and output.
    input: usize,
    /// The surface set of the weights.
    weights: usize,
}

/// One split's program, surfaces, worker and hand-off event.
pub(crate) struct AneFfn {
    rows: usize,
    hidden: usize,
    intermediate: usize,
    ane: usize,
    input_block: i32,
    mid_block: i32,
    /// Surface sets: one, or two for the half-chunk pipeline (inputs and
    /// outputs by half, weights by layer parity); keeps the memory under
    /// `arrays` alive.
    sets: Vec<Arc<Surfaces>>,
    /// MLX arrays over each set's surfaces (the eight inputs, then y).
    arrays: Vec<Vec<Array>>,
    /// Signaled by the GPU once a job's operands are written, and by the
    /// worker once the ANE wrote its output: separate events, as the
    /// pipeline signals the next half's operands before the ANE finished the
    /// previous half.
    ready: Arc<HandoffEvent>,
    done: Arc<HandoffEvent>,
    /// The last job's value on both events.
    next: u64,
    /// Pipeline: the weight set of the current layer and the half expected
    /// next.
    parity: usize,
    next_half: usize,
    jobs: mpsc::Sender<Job>,
    failure: Arc<Mutex<Option<String>>>,
}

// SAFETY: the MLX arrays over the surfaces are only used by the thread that
// holds the split's lock; the surfaces and the event are thread safe.
unsafe impl Send for AneFfn {}

/// The pending ANE half of one layer's split.
pub(crate) struct Pending {
    /// The layer input once the ANE's operands are written: the GPU's
    /// channels read this, so they run while the ANE does.
    pub(crate) ready: Array,
    xs: Array,
    done: u64,
    input: usize,
}

impl AneFfn {
    pub(crate) fn new(
        rows: usize,
        hidden: usize,
        intermediate: usize,
        units: usize,
    ) -> Result<Self> {
        Self::with_blocks(rows, hidden, intermediate, units, INPUT_BLOCK, MID_BLOCK)
    }

    /// The half-chunk pipeline's split: `rows` is a half chunk's; two
    /// surface sets ([`Self::begin_tiled_half`]).
    pub(crate) fn pipelined(
        rows: usize,
        hidden: usize,
        intermediate: usize,
        units: usize,
    ) -> Result<Self> {
        Self::with_sets(rows, hidden, intermediate, units, INPUT_BLOCK, MID_BLOCK, 2)
    }

    /// `new` with other rotation blocks (256 or 1024).
    pub(crate) fn with_blocks(
        rows: usize,
        hidden: usize,
        intermediate: usize,
        units: usize,
        input_block: i32,
        mid_block: i32,
    ) -> Result<Self> {
        Self::with_sets(rows, hidden, intermediate, units, input_block, mid_block, 1)
    }

    fn with_sets(
        rows: usize,
        hidden: usize,
        intermediate: usize,
        units: usize,
        input_block: i32,
        mid_block: i32,
        count: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            [input_block, mid_block]
                .iter()
                .all(|b| [256, 1024].contains(b)),
            "rotation blocks {input_block} / {mid_block}"
        );
        let ane = units * UNIT_CHANNELS as usize;
        let (program, surfaces) = build(
            rows,
            hidden,
            intermediate,
            ane,
            mid_block as usize,
            Variant::default(),
        )?;
        let mut sets = vec![surfaces];
        while sets.len() < count {
            sets.push(Arc::new(Surfaces::new(
                rows,
                hidden,
                ane,
                Variant::default(),
            )?));
        }
        let arrays = sets
            .iter()
            .map(|s| {
                [&s.x, &s.tx, &s.wg, &s.wu, &s.sg, &s.su, &s.wd, &s.sd, &s.y]
                    .into_iter()
                    .map(surface_array)
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        Self::start(
            rows,
            hidden,
            intermediate,
            ane,
            input_block,
            mid_block,
            program,
            sets,
            arrays,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn start(
        rows: usize,
        hidden: usize,
        intermediate: usize,
        ane: usize,
        input_block: i32,
        mid_block: i32,
        program: Program,
        sets: Vec<Arc<Surfaces>>,
        arrays: Vec<Vec<Array>>,
    ) -> Result<Self> {
        let ready = Arc::new(HandoffEvent::new()?);
        let done = Arc::new(HandoffEvent::new()?);
        let failure = Arc::new(Mutex::new(None));
        let (jobs, queue) = mpsc::channel::<Job>();
        {
            let (sets, ready, done, failure) =
                (sets.clone(), ready.clone(), done.clone(), failure.clone());
            std::thread::Builder::new()
                .name("ironmlx-ane-ffn".into())
                .spawn(move || {
                    // Diagnostic (`IRONMLX_DIAGNOSTIC_ANE_FFN` = time: per-64-layer
                    // means of the GPU's time between hand-offs and the ANE's
                    // evaluation; skip: also skip the evaluation; jobs / skipjobs:
                    // also each job's).
                    let diagnostic = std::env::var("IRONMLX_DIAGNOSTIC_ANE_FFN").ok();
                    let skip = matches!(diagnostic.as_deref(), Some("skip" | "skipjobs"));
                    let per_job = matches!(diagnostic.as_deref(), Some("jobs" | "skipjobs"));
                    let (mut last_done, mut sums, mut count) =
                        (std::time::Instant::now(), (0.0f64, 0.0f64), 0u32);
                    for job in queue {
                        let mut waited = 0.0;
                        let mut evaluated = 0.0;
                        let outcome = (|| -> Result<()> {
                            anyhow::ensure!(
                            ready.cpu_wait(job.ready, READY_TIMEOUT_MS)?,
                            "the GPU did not hand the ANE its operands within {READY_TIMEOUT_MS} ms"
                        );
                            let ready = std::time::Instant::now();
                            waited = ready.duration_since(last_done).as_secs_f64() * 1e3;
                            if !skip {
                                let (input, weights) = (&sets[job.input], &sets[job.weights]);
                                program.evaluate(0, &input.inputs_with(weights), &input.y)?;
                            }
                            evaluated = ready.elapsed().as_secs_f64() * 1e3;
                            Ok(())
                        })();
                        last_done = std::time::Instant::now();
                        if per_job {
                            eprintln!("ANE_JOB waited_ms={waited:.3} eval_ms={evaluated:.3}");
                        }
                        if diagnostic.is_some() {
                            sums = (sums.0 + waited, sums.1 + evaluated);
                            count += 1;
                            if count == 64 {
                                eprintln!(
                                    "ANE_FFN gpu_between_ms={:.3} ane_eval_ms={:.3}",
                                    sums.0 / 64.0,
                                    sums.1 / 64.0
                                );
                                (sums, count) = ((0.0, 0.0), 0);
                            }
                        }
                        if let Err(error) = outcome {
                            // Poison the output (NaN scales) so no wrong value passes.
                            let y = &sets[job.input].y;
                            // SAFETY: the surface is alive; the GPU does not read
                            // it before `done` is signaled below.
                            unsafe {
                                let row = y.base().add(y.row_bytes * (y.rows - 1)) as *mut u16;
                                for i in 0..y.width {
                                    *row.add(i) = 0x7e00;
                                }
                            }
                            failure
                                .lock()
                                .unwrap()
                                .get_or_insert_with(|| format!("{error:#}"));
                        }
                        let _ = done.cpu_signal(job.done);
                    }
                })?;
        }
        Ok(Self {
            rows,
            hidden,
            intermediate,
            ane,
            input_block,
            mid_block,
            sets,
            arrays,
            ready,
            done,
            next: 0,
            parity: 0,
            next_half: 0,
            jobs,
            failure,
        })
    }

    /// Channels the GPU keeps (the leading ones).
    pub(crate) fn gpu_channels(&self) -> i32 {
        (self.intermediate - self.ane) as i32
    }

    /// An error from the ANE side, once; the outputs it touched are NaN.
    pub(crate) fn take_failure(&self) -> Option<String> {
        self.failure.lock().unwrap().take()
    }

    /// Converts and hands the ANE its operands for input `x` (`[rows,
    /// hidden]` bf16) and the native affine4 gate/up (`[2I, hidden/8]`, gate
    /// rows first) and down (`[hidden, I/8]`) weights; queues its evaluation.
    pub(crate) fn begin(
        &mut self,
        x: &Array,
        gate_up: (&Array, &Array, &Array),
        down: (&Array, &Array, &Array),
        target: StreamOrDevice,
    ) -> Result<Pending> {
        let (h, i, a) = (
            self.hidden as i32,
            self.intermediate as i32,
            self.ane as i32,
        );
        let g = i - a;
        let (gw, gs, gb) = gate_up;
        let (wg, sg) =
            rotate_quantize_affine4(gw, gs, gb, g, a, 0, h, self.input_block, INPUT_SEED, target)?;
        let (wu, su) = rotate_quantize_affine4(
            gw,
            gs,
            gb,
            i + g,
            a,
            0,
            h,
            self.input_block,
            INPUT_SEED,
            target,
        )?;
        let (dw, dsc, db) = down;
        let (wd, sd) =
            rotate_quantize_affine4(dw, dsc, db, 0, h, g, a, self.mid_block, MID_SEED, target)?;
        self.hand_off(x, Some([(wg, sg), (wu, su), (wd, sd)]), 0, 0, target)
    }

    /// [`Self::begin`] with the gate / up and down weights in the M5 tiled
    /// layout (the same codes and scales, so the same operands).
    pub(crate) fn begin_tiled(
        &mut self,
        x: &Array,
        gate_up: &super::m5_affine4::Prepared,
        down: &super::m5_affine4::Prepared,
        target: StreamOrDevice,
    ) -> Result<Pending> {
        let weights = self.tiled_operands(gate_up, down, target)?;
        self.hand_off(x, Some(weights), 0, 0, target)
    }

    /// The half-chunk pipeline's [`Self::begin_tiled`] (a [`Self::pipelined`]
    /// split): half 0 of a layer writes its weights to the next weight set,
    /// half 1 (which must follow half 0 of the same layer) reuses them. Each
    /// half has its own input and output surfaces, so the ANE runs one half
    /// while the GPU works on the other; a layer's weights are not rewritten
    /// before the ANE finished the layer before.
    pub(crate) fn begin_tiled_half(
        &mut self,
        x: &Array,
        gate_up: &super::m5_affine4::Prepared,
        down: &super::m5_affine4::Prepared,
        half: usize,
        target: StreamOrDevice,
    ) -> Result<Pending> {
        anyhow::ensure!(self.sets.len() == 2, "not a pipelined ANE FFN split");
        match half {
            0 => {
                let weights = self.tiled_operands(gate_up, down, target)?;
                let parity = 1 - self.parity;
                let pending = self.hand_off(x, Some(weights), 0, parity, target)?;
                self.parity = parity;
                self.next_half = 1;
                Ok(pending)
            }
            1 => {
                anyhow::ensure!(self.next_half == 1, "pipelined half 1 without half 0");
                self.next_half = 0;
                self.hand_off(x, None, 1, self.parity, target)
            }
            _ => anyhow::bail!("pipelined half {half}"),
        }
    }

    /// The gate, up and down codes and scales of the ANE's channels, from the
    /// M5 tiled layout.
    fn tiled_operands(
        &self,
        gate_up: &super::m5_affine4::Prepared,
        down: &super::m5_affine4::Prepared,
        target: StreamOrDevice,
    ) -> Result<[(Array, Array); 3]> {
        let (h, i, a) = (
            self.hidden as i32,
            self.intermediate as i32,
            self.ane as i32,
        );
        let g = i - a;
        let (wg, sg) =
            rotate_quantize_tiled(gate_up, g, a, 0, h, self.input_block, INPUT_SEED, target)?;
        let (wu, su) = rotate_quantize_tiled(
            gate_up,
            i + g,
            a,
            0,
            h,
            self.input_block,
            INPUT_SEED,
            target,
        )?;
        let (wd, sd) = rotate_quantize_tiled(down, 0, h, g, a, self.mid_block, MID_SEED, target)?;
        Ok([(wg, sg), (wu, su), (wd, sd)])
    }

    /// Writes the ANE's operands (the rotated input to set `input`, and with
    /// `weights` the gate, up and down codes with their scales to set
    /// `weight_set`) and queues its evaluation over those sets.
    fn hand_off(
        &mut self,
        x: &Array,
        weights: Option<[(Array, Array); 3]>,
        input: usize,
        weight_set: usize,
        target: StreamOrDevice,
    ) -> Result<Pending> {
        let (rows, h, a) = (self.rows as i32, self.hidden as i32, self.ane as i32);
        anyhow::ensure!(
            x.shape().as_slice() == [rows, h],
            "ANE FFN input {:?}",
            x.shape().as_slice()
        );
        let (xq, xs) = rotate_quantize_dense(x, self.input_block, INPUT_SEED, target)?;
        let unit2 = scalar(INT8_UNIT * INT8_UNIT);
        let column = |scale: &Array, n: i32| -> Result<Array> {
            let scaled = mlx::ops::multiply_on(scale, &unit2, target)?;
            let half = scaled
                .astype_on(Dtype::Float16, target)?
                .reshape_on(&[n, 1][..], target)?;
            Ok(half.broadcast_to_on(&[n, 32][..], target)?)
        };
        let mut sources = vec![
            (xq.transpose_on(target)?, &self.arrays[input][0]),
            (
                xs.astype_on(Dtype::Float16, target)?
                    .reshape_on(&[1, rows][..], target)?,
                &self.arrays[input][1],
            ),
        ];
        if let Some([(wg, sg), (wu, su), (wd, sd)]) = weights {
            let w = &self.arrays[weight_set];
            sources.extend([
                (wg, &w[2]),
                (wu, &w[3]),
                (column(&sg, a)?, &w[4]),
                (column(&su, a)?, &w[5]),
                (wd, &w[6]),
                (column(&sd, h)?, &w[7]),
            ]);
        }
        let mut written = Vec::with_capacity(sources.len());
        for (source, surface) in &sources {
            written.push(mlx::fast::copy_into_external_on(
                source,
                surface,
                &[x],
                target,
            )?);
        }
        self.next += 1;
        let (ready_value, done) = (self.next, self.next);
        let after: Vec<&Array> = written.iter().collect();
        let ready = self.ready.signal_after_on(x, &after, ready_value, target)?;
        mlx::transforms::async_eval(&[&ready])?;
        self.jobs
            .send(Job {
                ready: ready_value,
                done,
                input,
                weights: weight_set,
            })
            .map_err(|_| anyhow::anyhow!("the ANE FFN worker stopped"))?;
        Ok(Pending {
            ready,
            xs,
            done,
            input,
        })
    }

    /// The layer's output: `partial` (`[rows, hidden]` bf16, the GPU's
    /// channels) plus the ANE's.
    pub(crate) fn finish(
        &self,
        pending: Pending,
        partial: &Array,
        target: StreamOrDevice,
    ) -> Result<Array> {
        // Diagnostic (`IRONMLX_DIAGNOSTIC_ANE_FFN_COMMIT=1`): commit the GPU's
        // channels first, so the read of the ANE's output starts a new
        // command buffer.
        if std::env::var("IRONMLX_DIAGNOSTIC_ANE_FFN_COMMIT").as_deref() == Ok("1") {
            mlx::transforms::async_eval(&[partial])?;
        }
        let y = self.done.wait_before_on(
            &self.arrays[pending.input][8],
            pending.done,
            &[partial],
            target,
        )?;
        join(partial, &y, &pending.xs, target)
    }
}

static INSTANCE: OnceLock<Option<Mutex<AneFfn>>> = OnceLock::new();
static PIPELINED: OnceLock<Option<Mutex<AneFfn>>> = OnceLock::new();
/// Set once the ANE side failed: later chunks keep the MLP on the GPU.
static RETIRED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// `Err` when the ANE side of an evaluated split failed since the last call
/// (its outputs are NaN); the split then retires for the process.
pub(crate) fn check() -> Result<()> {
    for slot in [&INSTANCE, &PIPELINED] {
        let Some(Some(split)) = slot.get() else {
            continue;
        };
        let failure = split
            .lock()
            .map_err(|_| anyhow::anyhow!("ANE FFN split poisoned"))?
            .take_failure();
        if let Some(error) = failure {
            RETIRED.store(true, std::sync::atomic::Ordering::Relaxed);
            tracing::error!(
                "Neural Engine FFN share failed, the GPU takes it from now on: {error}"
            );
            anyhow::bail!("Neural Engine FFN share failed: {error}")
        }
    }
    Ok(())
}

/// The process's split for `[rows, hidden]` chunks of an `intermediate`-wide
/// MLP with `units` ANE units, created on first use; `None` (logged once)
/// where the ANE is unavailable, and once it failed.
pub(crate) fn instance(
    rows: usize,
    hidden: usize,
    intermediate: usize,
    units: usize,
) -> Option<&'static Mutex<AneFfn>> {
    shared(&INSTANCE, rows, hidden, intermediate, units, AneFfn::new)
}

/// [`instance`] for the half-chunk pipeline ([`AneFfn::pipelined`], `rows`
/// a half chunk's).
pub(crate) fn pipelined_instance(
    rows: usize,
    hidden: usize,
    intermediate: usize,
    units: usize,
) -> Option<&'static Mutex<AneFfn>> {
    shared(
        &PIPELINED,
        rows,
        hidden,
        intermediate,
        units,
        AneFfn::pipelined,
    )
}

fn shared(
    slot: &'static OnceLock<Option<Mutex<AneFfn>>>,
    rows: usize,
    hidden: usize,
    intermediate: usize,
    units: usize,
    create: fn(usize, usize, usize, usize) -> Result<AneFfn>,
) -> Option<&'static Mutex<AneFfn>> {
    if RETIRED.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    let slot = slot.get_or_init(|| match create(rows, hidden, intermediate, units) {
        Ok(split) => Some(Mutex::new(split)),
        Err(error) => {
            tracing::warn!("Dense prefill FFN stays on the GPU: {error:#}");
            None
        }
    });
    let split = slot.as_ref()?;
    {
        let s = split.lock().unwrap();
        if (s.rows, s.hidden, s.intermediate, s.ane)
            != (rows, hidden, intermediate, units * UNIT_CHANNELS as usize)
        {
            return None;
        }
    }
    Some(split)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::ops::cast::astype;
    use serial_test::serial;

    fn metallib() -> Result<()> {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        }
        Ok(())
    }

    /// Needs an ANE and a GPU: the GPU writes two fp16 inputs into IOSurfaces
    /// in place, the ANE adds them, and the GPU reads the ANE's output in place.
    #[test]
    #[ignore]
    #[serial(mlx_metal)]
    fn gpu_ane_gpu_zero_copy_chain() -> Result<()> {
        metallib()?;
        if let Err(reason) = ironmlx_ane::availability() {
            eprintln!("ANE unavailable: {reason}");
            return Ok(());
        }
        let (rows, width) = (128usize, 96usize);
        let a = Surface::new(rows, width, Elem::F16)?;
        let b = Surface::new(rows, width, Elem::F16)?;
        let y = Surface::new(rows, width, Elem::F16)?;
        let stride = a.stride() as i32;
        let target = StreamOrDevice::default();
        // Inputs computed on the GPU, padded to the surface stride, written in place.
        let make = |scale: f32| -> Result<Array> {
            let v = mlx::ops::arange(0.0, (rows * width) as f64, 1.0, Dtype::Float32)?;
            let v = mlx::ops::shape::reshape(&(&v * scale), &[rows as i32, width as i32][..])?;
            let pad = Array::zeros(&[rows as i32, stride - width as i32][..], Dtype::Float32)?;
            Ok(astype(
                &mlx::ops::shape::concatenate(&[&v, &pad], 1)?,
                Dtype::Float16,
            )?)
        };
        let wa = mlx::fast::copy_into_external_on(&make(0.001)?, &surface_array(&a)?, &[], target)?;
        let wb = mlx::fast::copy_into_external_on(&make(0.002)?, &surface_array(&b)?, &[], target)?;
        mlx::transforms::eval(&[&wa, &wb])?;
        let p = y.rows * y.stride();
        let mil = format!(
            "program(1.3)\n{{\n    func f<ios18>({} a, {} b) {{\n        tensor<fp16, [1, 1, {rows}, {width}]> at = tensor_buffer_to_tensor<ios17>(input = a);\n        tensor<fp16, [1, 1, {rows}, {width}]> bt = tensor_buffer_to_tensor<ios17>(input = b);\n        tensor<fp16, [1, 1, {rows}, {width}]> s = add(x = at, y = bt);\n        {} y = tensor_to_tensor_buffer<ios17>(input = s, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>([{p}, {p}, {st}, 1]));\n    }} -> (y);\n}}\n",
            a.buffer_type(rows),
            b.buffer_type(rows),
            y.buffer_type(rows),
            st = y.stride()
        );
        let program = Program::load(&mil, &ironmlx_ane::empty_weights())?;
        program.evaluate(0, &[("a", &a), ("b", &b)], &y)?;
        // The GPU reads the ANE's output in place.
        let out = surface_array(&y)?;
        let out = mlx::ops::indexing::slice(&out, &[0, 0][..], &[rows as i32, width as i32][..])?;
        let want = &astype(&make(0.003)?, Dtype::Float32)?;
        let want = mlx::ops::indexing::slice(want, &[0, 0][..], &[rows as i32, width as i32][..])?;
        let diff = mlx::ops::reduction::max(
            &mlx::ops::unary::abs(&(&astype(&out, Dtype::Float32)? - &want))?,
            mlx::ops::reduction::All,
            false,
        )?
        .item::<f32>()?;
        let peak =
            mlx::ops::reduction::max(&want, mlx::ops::reduction::All, false)?.item::<f32>()?;
        eprintln!("ANE chain max |diff| {diff:.4} (values up to {peak:.1})");
        assert!(diff <= peak * 2e-3, "zero-copy chain mismatch {diff}");
        Ok(())
    }

    /// Needs an ANE: evaluation time of the ANE program's variants over
    /// `ANE_FFN_ROWS` rows (default 2048; `ANE_FFN_UNITS`, default 8),
    /// surfaces of pseudo-random int8 and small fp16 scales. ANE only, no GPU
    /// work.
    #[test]
    #[ignore]
    fn program_variant_times() -> Result<()> {
        if ironmlx_ane::availability().is_err() {
            eprintln!("ANE unavailable");
            return Ok(());
        }
        let env = |name: &str, default: usize| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        };
        let (units, rows) = (env("ANE_FFN_UNITS", 8), env("ANE_FFN_ROWS", 2048));
        let (h, i) = (5120usize, 17408usize);
        let a = units * UNIT_CHANNELS as usize;
        // `ANE_VARIANTS="input,mid;input,mid;..."` (default: a fixed set).
        let spec = std::env::var("ANE_VARIANTS").unwrap_or_else(|_| "1,1;2,2;2,4;5,2;5,4".into());
        let variants: Vec<Variant> = spec
            .split(';')
            .map(|v| {
                let mut it = v.split(',').map(|n| n.trim().parse::<usize>().unwrap());
                Variant {
                    input_segments: it.next().unwrap(),
                    mid_segments: it.next().unwrap(),
                    rotate_mid: true,
                    fp16_input: it.next().unwrap_or(0) == 1,
                    fp16_mid: it.next().unwrap_or(0) == 1,
                }
            })
            .collect();
        for variant in variants {
            let (program, s) = build(rows, h, i, a, MID_BLOCK as usize, variant)?;
            let mut state = 0x1234_5678u32;
            for (_, surface) in s.inputs() {
                // SAFETY: writes within the surface's allocation.
                unsafe {
                    let base = surface.base();
                    for r in 0..surface.rows {
                        let row = base.add(r * surface.row_bytes);
                        for c in 0..surface.width {
                            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                            match surface.elem {
                                Elem::I8 => *row.add(c) = ((state >> 24) as i8 / 2) as u8,
                                Elem::F16 => *(row as *mut u16).add(c) = 0x1400, // 2^-10
                            }
                        }
                    }
                }
            }
            let mut times = Vec::new();
            for iter in 0..12 {
                let t0 = std::time::Instant::now();
                program.evaluate(0, &s.inputs(), &s.y)?;
                if iter >= 2 {
                    times.push(t0.elapsed().as_secs_f64() * 1e3);
                }
            }
            times.sort_by(|x, y| x.partial_cmp(y).unwrap());
            eprintln!(
                "VARIANT units={units} {variant:?}: median {:.3} ms min {:.3} ms",
                times[times.len() / 2],
                times[0]
            );
        }
        Ok(())
    }

    /// Needs a GPU: time of the split's GPU-side operand preparation over
    /// one 2048-row layer (Dense shape, 8 units), by part.
    #[test]
    #[ignore]
    #[serial(mlx_metal)]
    fn split_overhead_parts() -> Result<()> {
        metallib()?;
        let (rows, h, i, a) = (2048i32, 5120i32, 17408i32, 4096i32);
        let g = i - a;
        let target = StreamOrDevice::default();
        let gu = mlx::random::normal()
            .shape(&[2 * i, h][..])
            .scale(0.02)
            .key(&mlx::random::key(1)?)
            .sample()?;
        let dn = mlx::random::normal()
            .shape(&[h, i][..])
            .scale(0.02)
            .key(&mlx::random::key(2)?)
            .sample()?;
        let gu = mlx::quantization::quantize(
            &astype(&gu, Dtype::Bfloat16)?,
            Some(64),
            Some(4),
            "affine",
            None,
        )?;
        let dn = mlx::quantization::quantize(
            &astype(&dn, Dtype::Bfloat16)?,
            Some(64),
            Some(4),
            "affine",
            None,
        )?;
        let x = astype(
            &mlx::random::normal()
                .shape(&[rows, h][..])
                .key(&mlx::random::key(3)?)
                .sample()?,
            Dtype::Bfloat16,
        )?;
        mlx::transforms::eval(&[&gu[0], &gu[1], &gu[2], &dn[0], &dn[1], &dn[2], &x])?;
        let time = |name: &str, f: &dyn Fn() -> Result<Vec<Array>>| -> Result<()> {
            let mut v = Vec::new();
            for it in 0..8 {
                let mut outs = Vec::new();
                for _ in 0..8 {
                    outs.extend(f()?);
                }
                let t0 = std::time::Instant::now();
                let refs: Vec<&Array> = outs.iter().collect();
                mlx::transforms::eval(&refs)?;
                if it >= 2 {
                    v.push(t0.elapsed().as_secs_f64() * 1e3 / 8.0);
                }
            }
            v.sort_by(|p, q| p.partial_cmp(q).unwrap());
            eprintln!("PART {name}: {:.3} ms", v[v.len() / 2]);
            Ok(())
        };
        time("rotate x", &|| {
            let (q, s) = rotate_quantize_dense(&x, INPUT_BLOCK, INPUT_SEED, target)?;
            Ok(vec![q, s])
        })?;
        time("convert gate+up", &|| {
            let (a1, b1) = rotate_quantize_affine4(
                &gu[0],
                &gu[1],
                &gu[2],
                g,
                a,
                0,
                h,
                INPUT_BLOCK,
                INPUT_SEED,
                target,
            )?;
            let (a2, b2) = rotate_quantize_affine4(
                &gu[0],
                &gu[1],
                &gu[2],
                i + g,
                a,
                0,
                h,
                INPUT_BLOCK,
                INPUT_SEED,
                target,
            )?;
            Ok(vec![a1, b1, a2, b2])
        })?;
        time("convert down", &|| {
            let (a1, b1) = rotate_quantize_affine4(
                &dn[0], &dn[1], &dn[2], 0, h, g, a, MID_BLOCK, MID_SEED, target,
            )?;
            Ok(vec![a1, b1])
        })?;
        let int8 = astype(&x, Dtype::Int8)?;
        let big = astype(&Array::zeros(&[a, h][..], Dtype::Float32)?, Dtype::Int8)?;
        mlx::transforms::eval(&[&int8, &big])?;
        time("copy x transposed (10 MB)", &|| {
            Ok(vec![mlx::ops::shape::contiguous_on(
                &int8.transpose_on(target)?,
                false,
                target,
            )?])
        })?;
        time("copy 3 weight sets (63 MB)", &|| {
            let one = astype(&scalar(1.0), Dtype::Int8)?;
            let mut v = Vec::new();
            for _ in 0..3 {
                v.push(mlx::ops::multiply_on(&big, &one, target)?);
            }
            Ok(v)
        })?;
        time("down column slices for qmm", &|| {
            let cols = |q: &Array, n: i32| -> Result<Array> {
                Ok(mlx::ops::shape::contiguous_on(
                    &q.slice_on(&[0, 0][..], &[h, n][..], target)?,
                    false,
                    target,
                )?)
            };
            Ok(vec![
                cols(&dn[0], g / 8)?,
                cols(&dn[1], g / 64)?,
                cols(&dn[2], g / 64)?,
            ])
        })?;
        let partial = astype(&x, Dtype::Bfloat16)?;
        let y = astype(
            &Array::zeros(&[h + 1, rows][..], Dtype::Float32)?,
            Dtype::Float16,
        )?;
        let xs = Array::zeros(&[rows][..], Dtype::Float32)?;
        mlx::transforms::eval(&[&partial, &y, &xs])?;
        time("join", &|| Ok(vec![join(&partial, &y, &xs, target)?]))?;
        Ok(())
    }

    /// The ANE program in the token-major layout (`x [T, H]`, products
    /// `[T, channels]` through `transpose_y` matmuls, the intermediate
    /// rotation as a `[T * A / B, B] x [B, B]` matmul after sign flips, output
    /// `[T, H + 1]`), over `segments` contraction segments each.
    fn transposed_text(rows: usize, h: usize, a: usize, b: usize, segments: usize) -> String {
        let t = |dt: &str, r: usize, c: usize| format!("tensor<{dt}, [1, 1, {r}, {c}]>");
        let mut body = String::new();
        let mut line = |s: String| {
            body.push_str("        ");
            body.push_str(&s);
            body.push_str(";\n");
        };
        let unit = hex(1.0 / INT8_UNIT);
        for (name, dt, r, c) in [
            ("x", "int8", rows, h),
            ("tx", "fp16", rows, 1),
            ("wg", "int8", a, h),
            ("wu", "int8", a, h),
            ("sg", "fp16", 1, a),
            ("su", "fp16", 1, a),
            ("wd", "int8", h, a),
            ("sd", "fp16", 1, h),
            ("dm", "fp16", 1, a),
        ] {
            line(format!(
                "{} {name}_t = tensor_buffer_to_tensor<ios17>(input = {name})",
                t(dt, r, c)
            ));
        }
        for (name, r, c) in [("x", rows, h), ("wg", a, h), ("wu", a, h), ("wd", h, a)] {
            line(format!(
                "{} {name}_d = dequantize(input = {name}_t, scale = {unit})",
                t("fp16", r, c)
            ));
        }
        let mm = |line: &mut dyn FnMut(String),
                  name: &str,
                  v: &str,
                  w: &str,
                  out: usize,
                  k: usize| {
            let seg = k / segments;
            let mut total = String::new();
            for i in 0..segments {
                let o = i * seg;
                line(format!("{} {name}_v{i} = slice_by_size(x = {v}, begin = tensor<int32, [4]>([0, 0, 0, {o}]), size = tensor<int32, [4]>([1, 1, {rows}, {seg}]))", t("fp16", rows, seg)));
                line(format!("{} {name}_w{i} = slice_by_size(x = {w}, begin = tensor<int32, [4]>([0, 0, 0, {o}]), size = tensor<int32, [4]>([1, 1, {out}, {seg}]))", t("fp16", out, seg)));
                line(format!("{} {name}_m{i} = matmul(transpose_x = bool(false), transpose_y = bool(true), x = {name}_v{i}, y = {name}_w{i})", t("fp16", rows, out)));
                total = if i == 0 {
                    format!("{name}_m0")
                } else {
                    let next = format!("{name}_s{i}");
                    line(format!(
                        "{} {next} = add(x = {total}, y = {name}_m{i})",
                        t("fp16", rows, out)
                    ));
                    next
                };
            }
            line(format!(
                "{} {name} = identity(x = {total})",
                t("fp16", rows, out)
            ));
        };
        let f = |c: usize| t("fp16", rows, c);
        mm(&mut line, "gm", "x_d", "wg_d", a, h);
        mm(&mut line, "um", "x_d", "wu_d", a, h);
        line(format!("{} gs = mul(x = gm, y = sg_t)", f(a)));
        line(format!("{} g = mul(x = gs, y = tx_t)", f(a)));
        line(format!("{} us = mul(x = um, y = su_t)", f(a)));
        line(format!("{} gh = mul(x = g, y = {})", f(a), hex(0.5)));
        line(format!("{} th = tanh(x = gh)", f(a)));
        line(format!("{} tp = add(x = th, y = {})", f(a), hex(1.0)));
        line(format!("{} sl = mul(x = gh, y = tp)", f(a)));
        line(format!("{} hm = mul(x = sl, y = us)", f(a)));
        line(format!("{} hf = mul(x = hm, y = dm_t)", f(a)));
        let n = rows * a / b;
        line(format!(
            "{} h2 = reshape(x = hf, shape = tensor<int32, [4]>([1, 1, {n}, {b}]))",
            t("fp16", n, b)
        ));
        line(format!("tensor<fp16, [1, 1, {b}, {b}]> rot = const()[name = string(\"rot\"), val = tensor<fp16, [1, 1, {b}, {b}]>(BLOBFILE(path = string(\"@model_path/weights.bin\"), offset = uint64({})))]", ironmlx_ane::CONSTANT_OFFSET));
        line(format!(
            "{} r2 = matmul(transpose_x = bool(false), transpose_y = bool(false), x = h2, y = rot)",
            t("fp16", n, b)
        ));
        line(format!(
            "{} hr = reshape(x = r2, shape = tensor<int32, [4]>([1, 1, {rows}, {a}]))",
            f(a)
        ));
        line(format!("{} ha = abs(x = hr)", f(a)));
        line(format!(
            "{} pk = reduce_max(x = ha, axes = tensor<int32, [1]>([3]), keep_dims = bool(true))",
            f(1)
        ));
        line(format!(
            "{} fl = maximum(x = pk, y = {})",
            f(1),
            hex(MID_PEAK_FLOOR)
        ));
        line(format!(
            "{} iv = real_div(x = {}, y = fl)",
            f(1),
            hex(127.0)
        ));
        line(format!("{} hs = mul(x = hr, y = iv)", f(a)));
        line(format!(
            "{} hq = quantize(input = hs, scale = {}, output_dtype = string(\"int8\"))",
            t("int8", rows, a),
            hex(1.0)
        ));
        line(format!(
            "{} hd = dequantize(input = hq, scale = {unit})",
            f(a)
        ));
        line(format!(
            "{} hsc = mul(x = fl, y = {})",
            f(1),
            hex(1.0 / 127.0)
        ));
        mm(&mut line, "dn", "hd", "wd_d", h, a);
        line(format!("{} ds = mul(x = dn, y = sd_t)", f(h)));
        line(format!(
            "{} yt = concat(axis = int32(3), interleave = bool(false), values = (ds, hsc))",
            f(h + 1)
        ));
        body
    }

    /// Needs an ANE: evaluation time of the token-major program
    /// (`ANE_FFN_UNITS`, default 8; `ANE_T_SEGMENTS`, default 2). ANE only.
    #[test]
    #[ignore]
    fn transposed_program_time() -> Result<()> {
        if ironmlx_ane::availability().is_err() {
            eprintln!("ANE unavailable");
            return Ok(());
        }
        let units: usize = std::env::var("ANE_FFN_UNITS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8);
        let segments: usize = std::env::var("ANE_T_SEGMENTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2);
        let (rows, h) = (2048usize, 5120usize);
        let a = units * UNIT_CHANNELS as usize;
        let b = MID_BLOCK as usize;
        let surfaces = [
            ("x", Surface::new(rows, h, Elem::I8)?),
            ("tx", Surface::new(rows, 1, Elem::F16)?),
            ("wg", Surface::new(a, h, Elem::I8)?),
            ("wu", Surface::new(a, h, Elem::I8)?),
            ("sg", Surface::new(1, a, Elem::F16)?),
            ("su", Surface::new(1, a, Elem::F16)?),
            ("wd", Surface::new(h, a, Elem::I8)?),
            ("sd", Surface::new(1, h, Elem::F16)?),
            ("dm", Surface::new(1, a, Elem::F16)?),
        ];
        let y = Surface::new(rows, h + 1, Elem::F16)?;
        let signature = surfaces
            .iter()
            .map(|(name, surface)| format!("{} {name}", surface.buffer_type(surface.rows)))
            .collect::<Vec<_>>()
            .join(", ");
        let (ys, plane) = (y.stride(), y.rows * y.stride());
        let mil = format!(
            "program(1.3)\n{{\n    func f<ios18>({signature}) {{\n{}        {} y = tensor_to_tensor_buffer<ios17>(input = yt, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>([{plane}, {plane}, {ys}, 1]));\n    }} -> (y);\n}}\n",
            transposed_text(rows, h, a, b, segments),
            y.buffer_type(y.rows)
        );
        let k = b.trailing_zeros() / 2;
        let magnitude = ((15 - k) as u16) << 10;
        let rot: Vec<u16> = (0..b * b)
            .map(|e| {
                magnitude
                    | if ((e / b) & (e % b)).count_ones() & 1 == 1 {
                        0x8000
                    } else {
                        0
                    }
            })
            .collect();
        let program = Program::load(&mil, &ironmlx_ane::fp16_constant_weights(&rot))?;
        let mut state = 0x1234_5678u32;
        for (_, surface) in &surfaces {
            // SAFETY: writes within the surface's allocation.
            unsafe {
                let base = surface.base();
                for r in 0..surface.rows {
                    let row = base.add(r * surface.row_bytes);
                    for c in 0..surface.width {
                        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        match surface.elem {
                            Elem::I8 => *row.add(c) = ((state >> 24) as i8 / 2) as u8,
                            Elem::F16 => *(row as *mut u16).add(c) = 0x1400,
                        }
                    }
                }
            }
        }
        let inputs: Vec<(&str, &Surface)> = surfaces.iter().map(|(n, s)| (*n, s)).collect();
        let mut times = Vec::new();
        for iter in 0..12 {
            let t0 = std::time::Instant::now();
            program.evaluate(0, &inputs, &y)?;
            if iter >= 2 {
                times.push(t0.elapsed().as_secs_f64() * 1e3);
            }
        }
        times.sort_by(|p, q| p.partial_cmp(q).unwrap());
        eprintln!(
            "TRANSPOSED units={units} segments={segments}: median {:.3} ms min {:.3} ms",
            times[times.len() / 2],
            times[0]
        );
        Ok(())
    }

    fn rel_rms(got: &Array, want: &Array) -> Result<f32> {
        let got = astype(got, Dtype::Float32)?;
        let want = astype(want, Dtype::Float32)?;
        let d = &got - &want;
        let num =
            mlx::ops::reduction::sum(&(&d * &d), mlx::ops::reduction::All, false)?.item::<f32>()?;
        let den = mlx::ops::reduction::sum(&(&want * &want), mlx::ops::reduction::All, false)?
            .item::<f32>()?;
        Ok((num / den).sqrt())
    }

    /// Needs an ANE and a GPU: the split FFN of a synthetic affine4 layer
    /// (Dense shape, 2048 rows) against the GPU's (dequantized weights, f32
    /// products), and the split's time against the GPU alone over a chain of
    /// layers. `ANE_FFN_UNITS` (default 8) sets the ANE share.
    #[test]
    #[ignore]
    #[serial(mlx_metal)]
    fn split_ffn_accuracy_and_time() -> Result<()> {
        metallib()?;
        let units: usize = std::env::var("ANE_FFN_UNITS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8);
        let (rows, h, i) = (2048i32, 5120i32, 17408i32);
        let target = StreamOrDevice::default();
        let key = mlx::random::key(21)?;
        let gu = mlx::random::normal()
            .shape(&[2 * i, h][..])
            .scale(0.02)
            .key(&key)
            .sample()?;
        let dn = mlx::random::normal()
            .shape(&[h, i][..])
            .scale(0.02)
            .key(&mlx::random::key(22)?)
            .sample()?;
        let gu = mlx::quantization::quantize(
            &astype(&gu, Dtype::Bfloat16)?,
            Some(64),
            Some(4),
            "affine",
            None,
        )?;
        let dn = mlx::quantization::quantize(
            &astype(&dn, Dtype::Bfloat16)?,
            Some(64),
            Some(4),
            "affine",
            None,
        )?;
        // Activations with a few outlier channels, as after an RMS norm.
        let x = mlx::random::normal()
            .shape(&[rows, h][..])
            .scale(1.0)
            .key(&mlx::random::key(23)?)
            .sample()?;
        let lanes = mlx::ops::arange(0.0, h as f64, 1.0, Dtype::Float32)?;
        let outliers = mlx::ops::less(&lanes, &scalar(8.0))?;
        let boost = mlx::ops::where_(&outliers, &scalar(30.0), &scalar(1.0))?;
        let x = astype(&(&x * &boost), Dtype::Bfloat16)?;
        mlx::transforms::eval(&[&gu[0], &gu[1], &gu[2], &dn[0], &dn[1], &dn[2], &x])?;
        let deq = |q: &[Array], r0: i32, r1: i32| -> Result<Array> {
            let w = q[0].slice_on(&[r0, 0][..], &[r1, q[0].shape().as_slice()[1]][..], target)?;
            let s = q[1].slice_on(&[r0, 0][..], &[r1, q[1].shape().as_slice()[1]][..], target)?;
            let b = q[2].slice_on(&[r0, 0][..], &[r1, q[2].shape().as_slice()[1]][..], target)?;
            Ok(mlx::quantization::dequantize_on(
                &w,
                &s,
                Some(&b),
                Some(64),
                Some(4),
                "affine",
                None,
                Some(Dtype::Bfloat16),
                target,
            )?)
        };
        let silu_mul = |g: &Array, u: &Array| -> Result<Array> {
            let s = mlx::ops::unary::sigmoid(g)?;
            Ok(&(g * &s) * u)
        };
        // The FFN over the first `g` channels on the GPU, in `dtype`.
        let gpu_ffn = |x: &Array, g: i32, dtype: Dtype| -> Result<Array> {
            let xd = astype(x, dtype)?;
            let gate = xd.matmul_on(
                &astype(&deq(&gu, 0, g)?, dtype)?.transpose_on(target)?,
                target,
            )?;
            let up = xd.matmul_on(
                &astype(&deq(&gu, i, i + g)?, dtype)?.transpose_on(target)?,
                target,
            )?;
            let act = silu_mul(&gate, &up)?;
            let down = astype(&deq(&dn, 0, h)?, dtype)?.transpose_on(target)?;
            let down = down.slice_on(&[0, 0][..], &[g, h][..], target)?;
            Ok(act.matmul_on(&down, target)?)
        };
        let reference = gpu_ffn(&x, i, Dtype::Float32)?;
        let bf16_full = gpu_ffn(&x, i, Dtype::Bfloat16)?;
        mlx::transforms::eval(&[&reference, &bf16_full])?;
        let env = |name: &str, default: i32| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        };
        let (ib, mb) = (
            env("ANE_FFN_INPUT_BLOCK", INPUT_BLOCK),
            env("ANE_FFN_MID_BLOCK", MID_BLOCK),
        );
        if ironmlx_ane::availability().is_err() {
            eprintln!("ANE unavailable");
            return Ok(());
        }
        let mut split = AneFfn::with_blocks(rows as usize, h as usize, i as usize, units, ib, mb)?;
        let g = split.gpu_channels();
        let run_split = |split: &mut AneFfn, x: &Array| -> Result<Array> {
            let pending = split.begin(
                x,
                (&gu[0], &gu[1], &gu[2]),
                (&dn[0], &dn[1], &dn[2]),
                target,
            )?;
            let partial = astype(
                &gpu_ffn(&pending.ready, g, Dtype::Bfloat16)?,
                Dtype::Bfloat16,
            )?;
            split.finish(pending, &partial, target)
        };
        let out = run_split(&mut split, &x)?;
        mlx::transforms::eval(&[&out])?;
        if let Some(error) = split.take_failure() {
            anyhow::bail!("ANE side failed: {error}");
        }
        let e_split = rel_rms(&out, &reference)?;
        let e_bf16 = rel_rms(&bf16_full, &reference)?;
        // The ANE's channels alone, against the reference's.
        let gpu_part = gpu_ffn(&x, g, Dtype::Float32)?;
        let ane_ref = &reference - &gpu_part;
        let e_ane = rel_rms(&(&astype(&out, Dtype::Float32)? - &gpu_part), &ane_ref)?;
        // The same int8 operands and per-row/per-token scales in f32 on the
        // GPU: the error of the quantization alone.
        let a = i - g;
        let (xq, xs) = rotate_quantize_dense(&x, ib, INPUT_SEED, target)?;
        let (wg, sg) =
            rotate_quantize_affine4(&gu[0], &gu[1], &gu[2], g, a, 0, h, ib, INPUT_SEED, target)?;
        let (wu, su) = rotate_quantize_affine4(
            &gu[0],
            &gu[1],
            &gu[2],
            i + g,
            a,
            0,
            h,
            ib,
            INPUT_SEED,
            target,
        )?;
        let (wd, sd) =
            rotate_quantize_affine4(&dn[0], &dn[1], &dn[2], 0, h, g, a, mb, MID_SEED, target)?;
        let f = |q: &Array| astype(q, Dtype::Float32);
        let xs2 = xs.reshape_on(&[rows, 1][..], target)?;
        let gate = &(&f(&xq)?.matmul_on(&f(&wg)?.transpose_on(target)?, target)? * &sg) * &xs2;
        let up = &(&f(&xq)?.matmul_on(&f(&wu)?.transpose_on(target)?, target)? * &su) * &xs2;
        let hmid = silu_mul(&gate, &up)?;
        let emulated_unquantized_mid = {
            let down = astype(&deq(&dn, 0, h)?, Dtype::Float32)?.transpose_on(target)?;
            hmid.matmul_on(&down.slice_on(&[g, 0][..], &[i, h][..], target)?, target)?
        };
        let (hq, hs) =
            rotate_quantize_dense(&astype(&hmid, Dtype::Bfloat16)?, mb, MID_SEED, target)?;
        let emulated = &(&f(&hq)?.matmul_on(&f(&wd)?.transpose_on(target)?, target)? * &sd)
            * &hs.reshape_on(&[rows, 1][..], target)?;
        let e_emul = rel_rms(&emulated, &ane_ref)?;
        let e_emul_in = rel_rms(&emulated_unquantized_mid, &ane_ref)?;
        let e_vs_emul = rel_rms(&(&astype(&out, Dtype::Float32)? - &gpu_part), &emulated)?;
        eprintln!(
            "SPLIT units={units} blocks {ib}/{mb} gpu_channels={g}: rel RMS vs f32 split {e_split:.5} | ANE channels {e_ane:.5}; f32 emulation of the int8 operands {e_emul:.5} (inputs only {e_emul_in:.5}); ANE vs emulation {e_vs_emul:.5} | GPU BF16 {e_bf16:.5}"
        );
        // Time: a chain of `layers` FFNs (each layer's input is the previous
        // output, RMS normalized), split vs GPU alone, alternating.
        let layers = 16;
        let norm = |y: &Array| -> Result<Array> {
            let y32 = astype(y, Dtype::Float32)?;
            let ms = mlx::ops::reduction::mean(&(&y32 * &y32), [-1], true)?;
            Ok(astype(
                &(&y32 * &mlx::ops::unary::rsqrt(&(&ms + &scalar(1e-6)))?),
                Dtype::Bfloat16,
            )?)
        };
        let mut times = Vec::new();
        for round in 0..6 {
            for mode in 0..2 {
                let t0 = std::time::Instant::now();
                let mut y = x.clone();
                for _ in 0..layers {
                    let out = if mode == 0 {
                        astype(&gpu_ffn(&y, i, Dtype::Bfloat16)?, Dtype::Bfloat16)?
                    } else {
                        run_split(&mut split, &y)?
                    };
                    y = norm(&out)?;
                }
                mlx::transforms::eval(&[&y])?;
                if round > 0 {
                    times.push((mode, t0.elapsed().as_secs_f64() * 1e3 / layers as f64));
                }
            }
        }
        if let Some(error) = split.take_failure() {
            anyhow::bail!("ANE side failed: {error}");
        }
        let median = |m: usize| {
            let mut v: Vec<f64> = times.iter().filter(|t| t.0 == m).map(|t| t.1).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        eprintln!(
            "SPLIT time per layer: GPU alone {:.3} ms | split {:.3} ms",
            median(0),
            median(1)
        );
        Ok(())
    }

    /// Needs a GPU: the rotation of the tiled layout's codes equals the
    /// native layout's (codes and scales), for gate rows and down columns.
    #[test]
    #[ignore]
    #[serial_test::serial(mlx_metal)]
    fn tiled_rotation_matches_native() -> Result<()> {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        }
        let target = StreamOrDevice::default();
        for (n, k, row0, rows, col0, len, block) in [
            (34816, 5120, 12800, 4608, 0, 5120, INPUT_BLOCK),
            (5120, 17408, 0, 5120, 12800, 4608, MID_BLOCK),
            (5120, 17408, 0, 5120, 7168, 10240, MID_BLOCK),
        ] {
            let w = mlx::random::normal()
                .shape(&[n, k][..])
                .scale(0.02)
                .key(&mlx::random::key(51)?)
                .sample()?;
            let q = mlx::quantization::quantize(
                &w.astype(Dtype::Bfloat16)?,
                Some(64),
                Some(4),
                "affine",
                None,
            )?;
            let parts = ironmlx_core::nn::linear::QuantizedLinearParts {
                weight: &q[0],
                scales: &q[1],
                biases: Some(&q[2]),
                bias: None,
                group_size: 64,
                bits: 4,
                mode: crate::core::QuantMode::Affine,
            };
            let prepared = crate::nn::m5_affine4::prepare(&parts, target)?;
            let (qn, sn) = rotate_quantize_affine4(
                &q[0], &q[1], &q[2], row0, rows, col0, len, block, INPUT_SEED, target,
            )?;
            let (qt, st) =
                rotate_quantize_tiled(&prepared, row0, rows, col0, len, block, INPUT_SEED, target)?;
            let diff = |a: &Array, b: &Array| -> Result<i32> {
                Ok(mlx::ops::reduction::sum(
                    &a.not_equal(b)?.astype(Dtype::Int32)?,
                    mlx::ops::reduction::All,
                    false,
                )?
                .item::<i32>()?)
            };
            let (dq, ds) = (diff(&qn, &qt)?, diff(&sn, &st)?);
            println!("ROTATE n={n} k={k}: code mismatches {dq}, scale mismatches {ds}");
            anyhow::ensure!(dq == 0 && ds == 0, "tiled rotation differs");
        }
        Ok(())
    }

    /// Needs a GPU: rows past 8192 (two 256-element chunks per simdgroup)
    /// rotate as one chunk per simdgroup does. Columns 8192.. are zero, so the
    /// 10240-wide rows' first 8192 codes and their scales equal the
    /// 8192-wide rotation's, and the rest are zero.
    #[test]
    #[ignore]
    #[serial(mlx_metal)]
    fn two_chunk_rotation_matches_one_chunk() -> Result<()> {
        metallib()?;
        let target = StreamOrDevice::default();
        let (rows, short, long) = (64i32, 8192i32, 10240i32);
        let x = mlx::random::normal()
            .shape(&[rows, short][..])
            .key(&mlx::random::key(81)?)
            .sample()?
            .astype(Dtype::Bfloat16)?;
        let zeros = Array::zeros(&[rows, long - short][..], Dtype::Bfloat16)?;
        let wide = mlx::ops::concatenate(&[&x, &zeros][..], 1)?;
        let (q1, s1) = rotate_quantize_dense(&x, MID_BLOCK, MID_SEED, target)?;
        let (q2, s2) = rotate_quantize_dense(&wide, MID_BLOCK, MID_SEED, target)?;
        let head = q2.slice_on(&[0, 0][..], &[rows, short][..], target)?;
        let tail = q2.slice_on(&[0, short][..], &[rows, long][..], target)?;
        let count =
            |a: &Array| -> Result<i32> {
                Ok(mlx::ops::reduction::sum(
                    &a.astype(Dtype::Int32)?,
                    mlx::ops::reduction::All,
                    false,
                )?
                .item::<i32>()?)
            };
        let (dq, ds, nz) = (
            count(&head.not_equal(&q1)?)?,
            count(&s2.not_equal(&s1)?)?,
            count(&tail.not_equal(&scalar(0.0))?)?,
        );
        println!("TWO_CHUNK: code mismatches {dq}, scale mismatches {ds}, nonzero tail codes {nz}");
        anyhow::ensure!(dq == 0 && ds == 0 && nz == 0, "two-chunk rotation differs");
        Ok(())
    }

    /// Needs an ANE and a GPU: the half-chunk pipeline's split
    /// ([`AneFfn::pipelined`], two 1024-row halves per layer, both halves'
    /// GPU channels submitted before either awaits the ANE) over two chained
    /// layers of different weights equals the full-chunk split of the same
    /// inputs. `ANE_FFN_UNITS` (default 20) sets the share; a wide share keeps
    /// the ANE behind the GPU, so a missing wait or a surface the next half
    /// overwrites shows.
    #[test]
    #[ignore]
    #[serial(mlx_metal)]
    fn pipelined_halves_match_full_chunk() -> Result<()> {
        metallib()?;
        if ironmlx_ane::availability().is_err() {
            eprintln!("ANE unavailable");
            return Ok(());
        }
        let units: usize = std::env::var("ANE_FFN_UNITS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(20);
        let (rows, h, i) = (2048i32, 5120i32, 17408i32);
        let target = StreamOrDevice::default();
        let prepared = |n: i32, k: i32, seed: u64| -> Result<super::super::m5_affine4::Prepared> {
            let w = mlx::random::normal()
                .shape(&[n, k][..])
                .scale(0.02)
                .key(&mlx::random::key(seed)?)
                .sample()?;
            let q = mlx::quantization::quantize(
                &astype(&w, Dtype::Bfloat16)?,
                Some(64),
                Some(4),
                "affine",
                None,
            )?;
            let parts = ironmlx_core::nn::linear::QuantizedLinearParts {
                weight: &q[0],
                scales: &q[1],
                biases: Some(&q[2]),
                bias: None,
                group_size: 64,
                bits: 4,
                mode: crate::core::QuantMode::Affine,
            };
            let prepared = super::super::m5_affine4::prepare(&parts, target)?;
            mlx::transforms::eval(&[&prepared.weight, &prepared.sb])?;
            Ok(prepared)
        };
        let layers = [
            (prepared(2 * i, h, 91)?, prepared(h, i, 92)?),
            (prepared(2 * i, h, 93)?, prepared(h, i, 94)?),
        ];
        let lanes = mlx::ops::arange(0.0, h as f64, 1.0, Dtype::Float32)?;
        let boost = mlx::ops::where_(
            &mlx::ops::less(&lanes, &scalar(8.0))?,
            &scalar(30.0),
            &scalar(1.0),
        )?;
        let x = mlx::random::normal()
            .shape(&[rows, h][..])
            .key(&mlx::random::key(95)?)
            .sample()?;
        let x = astype(&(&x * &boost), Dtype::Bfloat16)?;
        let norm = |y: &Array| -> Result<Array> {
            let y32 = astype(y, Dtype::Float32)?;
            let ms = mlx::ops::reduction::mean(&(&y32 * &y32), [-1], true)?;
            Ok(astype(
                &(&y32 * &mlx::ops::unary::rsqrt(&(&ms + &scalar(1e-6)))?),
                Dtype::Bfloat16,
            )?)
        };
        let gpu_part = |ready: &Array,
                        gate_up: &super::super::m5_affine4::Prepared,
                        down: &super::super::m5_affine4::Prepared,
                        g: i32|
         -> Result<Array> {
            let activated = super::super::dense_prefill_kernels::prefill_tiled_swiglu_cols(
                ready, gate_up, g, target,
            )?;
            super::super::dense_prefill_kernels::prefill_tiled_gemm_leading(
                &activated, down, target,
            )
        };
        let mut halves = AneFfn::pipelined(rows as usize / 2, h as usize, i as usize, units)?;
        let mut full = AneFfn::new(rows as usize, h as usize, i as usize, units)?;
        let g = halves.gpu_channels();
        // The pipeline over both layers, unsynchronized.
        let mut input = x.clone();
        let mut inputs = Vec::new();
        let mut piped = Vec::new();
        for (gate_up, down) in &layers {
            inputs.push(input.clone());
            let parts = [
                input.slice_on(&[0, 0][..], &[rows / 2, h][..], target)?,
                input.slice_on(&[rows / 2, 0][..], &[rows, h][..], target)?,
            ];
            let mut started = Vec::new();
            for (half, part) in parts.iter().enumerate() {
                let pending = halves.begin_tiled_half(part, gate_up, down, half, target)?;
                let partial = gpu_part(&pending.ready, gate_up, down, g)?;
                started.push((pending, partial));
            }
            mlx::transforms::async_eval(&started.iter().map(|s| &s.1).collect::<Vec<_>>())?;
            let outs = started
                .into_iter()
                .map(|(pending, partial)| halves.finish(pending, &partial, target))
                .collect::<Result<Vec<_>>>()?;
            let out = mlx::ops::shape::concatenate_on(&[&outs[0], &outs[1]], 0, target)?;
            input = norm(&out)?;
            piped.push(out);
        }
        mlx::transforms::eval(&piped.iter().collect::<Vec<_>>())?;
        if let Some(error) = halves.take_failure() {
            anyhow::bail!("ANE side failed: {error}");
        }
        // The full-chunk split of the same inputs, one layer at a time.
        for (layer, ((gate_up, down), (inp, out))) in
            layers.iter().zip(inputs.iter().zip(&piped)).enumerate()
        {
            let pending = full.begin_tiled(inp, gate_up, down, target)?;
            let partial = gpu_part(&pending.ready, gate_up, down, g)?;
            let want = full.finish(pending, &partial, target)?;
            mlx::transforms::eval(&[&want])?;
            let e = rel_rms(out, &want)?;
            println!("PIPELINED units={units} layer {layer}: rel RMS vs full chunk {e:.6}");
            anyhow::ensure!(e < 1e-3, "pipelined layer {layer} differs: {e}");
        }
        if let Some(error) = full.take_failure() {
            anyhow::bail!("ANE side failed: {error}");
        }
        Ok(())
    }
}
