//! L1 feasibility micro-benchmark (diagnostic, not serving code).
//!
//! Real Qwen3.8-27B-4bit prefill projections at M=2048 rows, BF16
//! activations, affine4 group 64 weights, fused exactly as the DFlash2 lane
//! fuses them (concatenated output rows). One process runs one `--mode`:
//!   production - the embedded BM128 library is configured as the M5 profile
//!                does, so `quantized_matmul` takes BM128 where it matches;
//!   default    - no library, so `quantized_matmul` is MLX's own QMM.
//! Per projection: `qmm` (quantized_matmul, the production FFI), `bf16`
//! (matmul against the same weights dequantized to BF16; preparation time and
//! bytes reported separately). Each variant's first call (compile/prepare)
//! is timed apart from steady state. Steady iterations are separated by idle
//! gaps so an external Metal System Trace can attribute GPU intervals.
//! Numerics: every output is compared with an F32 reference (dequantized
//! weights, F32 activations) and output bits are hashed for cross-process
//! comparison.
use std::collections::HashMap;
use std::fmt::Write as _;
use std::time::{Duration, Instant};

use mlx::{io, ops, quantization, random, Array, Device, Dtype};

struct Projection {
    name: &'static str,
    layer: u32,
    parts: &'static [&'static str],
}

const PROJECTIONS: &[Projection] = &[
    Projection { name: "mlp_gate_up", layer: 0, parts: &["mlp.gate_proj", "mlp.up_proj"] },
    Projection { name: "mlp_down", layer: 0, parts: &["mlp.down_proj"] },
    Projection {
        name: "gdn_in_proj",
        layer: 0,
        parts: &["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b", "linear_attn.in_proj_a"],
    },
    Projection { name: "gdn_out_proj", layer: 0, parts: &["linear_attn.out_proj"] },
    Projection { name: "attn_qkv", layer: 3, parts: &["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"] },
    Projection { name: "attn_o_proj", layer: 3, parts: &["self_attn.o_proj"] },
];
const M: i32 = 2048;

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn bytes(a: &Array) -> usize {
    let width = match a.dtype() {
        Dtype::Uint32 | Dtype::Float32 | Dtype::Int32 => 4,
        Dtype::Bfloat16 | Dtype::Float16 => 2,
        other => panic!("unexpected dtype {other:?}"),
    };
    a.size() * width
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn run(f: &dyn Fn() -> mlx::Result<Array>) -> mlx::Result<(Array, f64)> {
    let start = Instant::now();
    let y = f()?;
    y.eval()?;
    mlx::synchronize()?;
    Ok((y, ms(start.elapsed())))
}

fn fnv(values: &[half::bf16]) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for v in values {
        for b in v.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    format!("{h:016x}")
}

fn numerics(y: &Array, reference: &[f32]) -> mlx::Result<String> {
    let bits = y.to_vec::<half::bf16>()?;
    let (mut max_abs, mut sum_abs, mut ref_max, mut nonfinite) = (0f64, 0f64, 0f64, 0usize);
    for (a, r) in bits.iter().zip(reference) {
        let a = a.to_f32() as f64;
        if !a.is_finite() {
            nonfinite += 1;
            continue;
        }
        let d = (a - *r as f64).abs();
        max_abs = max_abs.max(d);
        sum_abs += d;
        ref_max = ref_max.max((*r as f64).abs());
    }
    Ok(format!(
        "{{\"hash\":\"{}\",\"max_abs_err\":{max_abs},\"mean_abs_err\":{},\"ref_max_abs\":{ref_max},\"nonfinite\":{nonfinite}}}",
        fnv(&bits),
        sum_abs / bits.len() as f64
    ))
}

fn stats(samples: &[f64]) -> String {
    let mut s = samples.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| s[((s.len() - 1) as f64 * p).round() as usize];
    let list = samples.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>().join(",");
    format!(
        "{{\"n\":{},\"median_ms\":{:.4},\"p10_ms\":{:.4},\"p90_ms\":{:.4},\"min_ms\":{:.4},\"max_ms\":{:.4},\"samples_ms\":[{list}]}}",
        s.len(),
        q(0.5),
        q(0.1),
        q(0.9),
        s[0],
        s[s.len() - 1]
    )
}

fn main() -> mlx::Result<()> {
    let mode = arg("--mode").expect("--mode production|default");
    let model = arg("--model").expect("--model <dir>");
    let out = arg("--out").expect("--out <json>");
    let runs: usize = arg("--runs").map_or(30, |v| v.parse().unwrap());
    let warmup: usize = arg("--warmup").map_or(5, |v| v.parse().unwrap());
    let gap = Duration::from_millis(arg("--gap-ms").map_or(30, |v| v.parse().unwrap()));
    let only = arg("--only");
    // Kernel-identity mode: one steady call per variant inside a GPU capture
    // (needs MTL_CAPTURE_ENABLED=1); no timing is reported from this mode.
    let capture = arg("--capture-dir");
    // Back-to-back mode: N independent calls of a variant in one eval; wall/N
    // is the sustained per-call time (CPU encoding overlaps GPU execution).
    let chain: usize = arg("--chain").map_or(0, |v| v.parse().unwrap());
    mlx::metal::set_metallib_path(&arg("--mlx-metallib").expect("--mlx-metallib")).unwrap();
    let library = match mode.as_str() {
        "production" => {
            let path = format!("{out}.prefill_qmm_mtile.metallib");
            std::fs::write(&path, mlx::metal::prefill_shaders::PREFILL_QMM_MTILE_METALLIB).unwrap();
            assert!(mlx::metal::set_prefill_qmm_mtile_library(&path));
            path
        }
        "default" => {
            assert!(mlx::metal::set_prefill_qmm_mtile_library(""));
            String::new()
        }
        other => panic!("unknown mode {other}"),
    };
    mlx::set_default_device(Device::gpu(0));
    let index = std::fs::read_to_string(format!("{model}/model.safetensors.index.json")).unwrap();
    let shard_of = |key: &str| -> String {
        let at = index.find(&format!("\"{key}\"")).unwrap_or_else(|| panic!("{key} not in index"));
        let rest = &index[at + key.len() + 2..];
        let start = rest.find('"').unwrap() + 1;
        let end = start + rest[start..].find('"').unwrap();
        rest[start..end].to_string()
    };
    let mut shards: HashMap<String, HashMap<String, Array>> = HashMap::new();
    let mut json = String::new();
    write!(json, "{{\"mode\":\"{mode}\",\"library\":\"{library}\",\"m\":{M},\"runs\":{runs},\"warmup\":{warmup},\"gap_ms\":{},\"projections\":[", gap.as_millis()).unwrap();
    let process = Instant::now();
    let mut first = true;
    for p in PROJECTIONS {
        if only.as_deref().is_some_and(|o| !o.split(',').any(|n| n == p.name)) {
            continue;
        }
        let mut packed = Vec::new();
        for suffix in ["weight", "scales", "biases"] {
            let mut values = Vec::new();
            for part in p.parts {
                let key = format!("language_model.model.layers.{}.{part}.{suffix}", p.layer);
                let shard = shard_of(&key);
                let tensors = shards
                    .entry(shard.clone())
                    .or_insert_with(|| io::load_safetensors(&format!("{model}/{shard}")).unwrap().0);
                values.push(tensors.get(&key).unwrap().clone());
            }
            let v = if values.len() == 1 {
                values.pop().unwrap()
            } else {
                ops::concatenate(&values.iter().collect::<Vec<_>>(), 0)?
            };
            v.eval()?;
            packed.push(v);
        }
        let n = packed[0].shape().as_slice()[0];
        let k = packed[0].shape().as_slice()[1] * 8;
        random::seed(20261007);
        let x = random::normal().shape((M, k)).dtype(Dtype::Bfloat16).sample()?;
        x.eval()?;
        if let Some(dir) = &capture {
            let qmm = || quantization::quantized_matmul(&x, &packed[0], &packed[1], Some(&packed[2]), true, Some(64), Some(4), "affine");
            let w = quantization::dequantize(&packed[0], &packed[1], Some(&packed[2]), Some(64), Some(4), "affine", None, Some(Dtype::Bfloat16))?;
            let wt = ops::transpose(&w)?;
            let bf16 = || ops::matmul(&x, &wt);
            let variant = arg("--variant").expect("--variant qmm|bf16 with --capture-dir");
            let f: &dyn Fn() -> mlx::Result<Array> = if variant == "qmm" { &qmm } else { &bf16 };
            {
                run(f)?;
                mlx::metal::start(&format!("{dir}/{mode}-{}-{variant}.gputrace", p.name))?;
                run(f)?;
                mlx::metal::stop()?;
            }
            eprintln!("{} captured", p.name);
            continue;
        }
        if chain > 0 {
            let qmm = || quantization::quantized_matmul(&x, &packed[0], &packed[1], Some(&packed[2]), true, Some(64), Some(4), "affine");
            let w = quantization::dequantize(&packed[0], &packed[1], Some(&packed[2]), Some(64), Some(4), "affine", None, Some(Dtype::Bfloat16))?;
            w.eval()?;
            let wt = ops::transpose(&w)?;
            let bf16 = || ops::matmul(&x, &wt);
            let mut entry = format!("{{\"name\":\"{}\",\"k\":{k},\"n\":{n},\"chain\":{chain}", p.name);
            for (variant, f) in [("qmm", &qmm as &dyn Fn() -> mlx::Result<Array>), ("bf16", &bf16)] {
                run(f)?;
                let mut samples = Vec::new();
                for i in 0..warmup + runs {
                    let start = Instant::now();
                    let ys = (0..chain).map(|_| f()).collect::<mlx::Result<Vec<_>>>()?;
                    mlx::eval(&ys.iter().collect::<Vec<_>>())?;
                    mlx::synchronize()?;
                    if i >= warmup {
                        samples.push(ms(start.elapsed()) / chain as f64);
                    }
                    drop(ys);
                }
                write!(entry, ",\"{variant}\":{}", stats(&samples)).unwrap();
            }
            entry.push('}');
            if !first {
                json.push(',');
            }
            first = false;
            json.push_str(&entry);
            eprintln!("{} chained", p.name);
            continue;
        }
        // F32 reference: dequantized weights, F32 activations.
        let reference = {
            let wf = quantization::dequantize(&packed[0], &packed[1], Some(&packed[2]), Some(64), Some(4), "affine", None, Some(Dtype::Float32))?;
            let y = ops::matmul(&x.astype(Dtype::Float32)?, &ops::transpose(&wf)?)?;
            y.eval()?;
            y.to_vec::<f32>()?
        };
        mlx::clear_cache();
        let qmm = || quantization::quantized_matmul(&x, &packed[0], &packed[1], Some(&packed[2]), true, Some(64), Some(4), "affine");
        let mut entry = format!("{{\"name\":\"{}\",\"layer\":{},\"parts\":{:?},\"m\":{M},\"k\":{k},\"n\":{n},\"weight_bytes\":{},", p.name, p.layer, p.parts, packed.iter().map(bytes).sum::<usize>());
        // qmm
        std::thread::sleep(gap * 10);
        let t_qmm = process.elapsed().as_secs_f64();
        let (y, first_ms) = run(&qmm)?;
        let qmm_numerics = numerics(&y, &reference)?;
        drop(y);
        let mut samples = Vec::new();
        for i in 0..warmup + runs {
            std::thread::sleep(gap);
            let (_, t) = run(&qmm)?;
            if i >= warmup {
                samples.push(t);
            }
        }
        write!(entry, "\"qmm\":{{\"t_start_s\":{t_qmm:.3},\"t_end_s\":{:.3},\"first_call_ms\":{first_ms:.4},\"steady\":{},\"numerics\":{qmm_numerics}}},", process.elapsed().as_secs_f64(), stats(&samples)).unwrap();
        // bf16 reference matmul on the same dequantized weights
        std::thread::sleep(gap * 10);
        let t_bf16 = process.elapsed().as_secs_f64();
        let before = mlx::memory::snapshot().active_bytes;
        let start = Instant::now();
        let w = quantization::dequantize(&packed[0], &packed[1], Some(&packed[2]), Some(64), Some(4), "affine", None, Some(Dtype::Bfloat16))?;
        w.eval()?;
        mlx::synchronize()?;
        let prep_ms = ms(start.elapsed());
        let prep_bytes = mlx::memory::snapshot().active_bytes.saturating_sub(before);
        let wt = ops::transpose(&w)?;
        let bf16 = || ops::matmul(&x, &wt);
        let (y, first_ms) = run(&bf16)?;
        let bf16_numerics = numerics(&y, &reference)?;
        drop(y);
        let mut samples = Vec::new();
        for i in 0..warmup + runs {
            std::thread::sleep(gap);
            let (_, t) = run(&bf16)?;
            if i >= warmup {
                samples.push(t);
            }
        }
        write!(entry, "\"bf16\":{{\"t_start_s\":{t_bf16:.3},\"t_end_s\":{:.3},\"prepare_ms\":{prep_ms:.4},\"prepared_bytes\":{},\"prepared_active_delta_bytes\":{prep_bytes},\"first_call_ms\":{first_ms:.4},\"steady\":{},\"numerics\":{bf16_numerics}}}}}", process.elapsed().as_secs_f64(), bytes(&w), stats(&samples)).unwrap();
        drop((w, wt));
        mlx::clear_cache();
        if !first {
            json.push(',');
        }
        first = false;
        json.push_str(&entry);
        eprintln!("{} done", p.name);
    }
    let mem = mlx::memory::snapshot();
    write!(json, "],\"mlx_peak_bytes\":{}}}\n", mem.peak_bytes).unwrap();
    std::fs::write(&out, json).unwrap();
    Ok(())
}
