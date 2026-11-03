//! Flash-Next's independent affine4/g32 shared-expert ANE program.
//! GPU routing/experts overlap the ANE. A failed job falls back at the
//! stateless shared-MLP boundary, before it can modify request cache state.
use super::{ane_ffn::surface_array_owned, Linear};
use anyhow::{ensure, Context as _, Result};
use ironmlx_ane::{Elem, Program, Surface};
use mlx::{Array, Dtype, StreamOrDevice};
use std::{
    cell::RefCell,
    sync::{mpsc, Arc, Mutex},
    thread::JoinHandle,
    time::Duration,
};
fn decode_weight(linear: &Linear, target: StreamOrDevice) -> Result<Array> {
    let p = linear
        .quantized_parts()
        .context("Flash affine4/g32 weights")?;
    ensure!(
        p.group_size == 32
            && p.bits == 4
            && p.mode == crate::core::weights::QuantMode::Affine
            && p.bias.is_none(),
        "Flash affine4/g32 contract"
    );
    // Affine dequantization computes in the scales' dtype before the final
    // output cast. Widen scales/bias first for a true FP16 weight candidate.
    let bf16 = std::env::var("IRONMLX_DIAGNOSTIC_FLASH_ANE_WEIGHT_ROUND").as_deref() == Ok("bf16");
    let dtype = if bf16 {
        Dtype::Bfloat16
    } else {
        Dtype::Float32
    };
    let scales = p.scales.astype_on(dtype, target)?;
    let biases = p.biases.map(|b| b.astype_on(dtype, target)).transpose()?;
    Ok(mlx::quantization::dequantize_on(
        p.weight,
        &scales,
        biases.as_ref(),
        Some(32),
        Some(4),
        "affine",
        None,
        Some(Dtype::Float16),
        target,
    )?)
}
fn constant_weights(tensors: &[Vec<u16>]) -> (Vec<u8>, Vec<u64>) {
    let mut blob = vec![0u8; 64];
    blob[..4].copy_from_slice(&(tensors.len() as u32).to_le_bytes());
    blob[4..8].copy_from_slice(&2u32.to_le_bytes());
    let mut offsets = Vec::new();
    for tensor in tensors {
        let record = (blob.len() + 63) & !63;
        let data = record + 64;
        let bytes = tensor.len() * 2;
        blob.resize(data + bytes, 0);
        offsets.push(record as u64);
        blob[record..record + 4].copy_from_slice(&0xdead_beefu32.to_le_bytes());
        blob[record + 4..record + 8].copy_from_slice(&1u32.to_le_bytes());
        blob[record + 8..record + 16].copy_from_slice(&(bytes as u64).to_le_bytes());
        blob[record + 16..record + 24].copy_from_slice(&(data as u64).to_le_bytes());
        for (i, value) in tensor.iter().enumerate() {
            blob[data + 2 * i..data + 2 * i + 2].copy_from_slice(&value.to_le_bytes());
        }
    }
    (blob, offsets)
}
struct Buffers {
    x: Surface,
    gate: Surface,
    up: Surface,
    down: Surface,
    out: Surface,
}
impl Buffers {
    fn bytes(&self) -> usize {
        [&self.x, &self.gate, &self.up, &self.down, &self.out]
            .into_iter()
            .map(|s| s.alloc_bytes)
            .sum()
    }
}
impl Drop for Buffers {
    fn drop(&mut self) {
        ironmlx_core::ane::release_surfaces(ironmlx_core::ane::Model::FlashNext, self.bytes());
    }
}
struct Job {
    value: u64,
    key: usize,
    constant: Option<Program>,
    reply: mpsc::Sender<std::result::Result<(), String>>,
}
pub(crate) struct SharedExpert {
    buffers: Arc<Buffers>,
    rows: usize,
    projection_only: bool,
    static_weights: bool,
    registered: std::collections::HashSet<usize>,
    ready: Arc<mlx::fast::HandoffEvent>,
    next: u64,
    jobs: Option<mpsc::Sender<Job>>,
    worker: Option<JoinHandle<()>>,
    in_flight: bool,
    retired: bool,
    last_read: Option<Array>,
}
// SAFETY: MLX graph construction and buffer reuse are serialized by Context's mutex.
unsafe impl Send for SharedExpert {}
pub(crate) type Context = Arc<Mutex<SharedExpert>>;
thread_local! { static CONTEXT:RefCell<Vec<Context>> = const { RefCell::new(Vec::new()) }; }
pub(crate) struct Scope;
pub(crate) fn scope(context: Context) -> Scope {
    CONTEXT.with(|s| s.borrow_mut().push(context));
    Scope
}
impl Drop for Scope {
    fn drop(&mut self) {
        CONTEXT.with(|s| {
            s.borrow_mut().pop();
        });
    }
}
pub(crate) fn requested() -> bool {
    ironmlx_core::ane::mode() == ironmlx_core::ane::AneMode::Auto
        && ironmlx_core::ane::hardware().available
        && !ironmlx_core::ane::retired(ironmlx_core::ane::Model::FlashNext)
        && (ironmlx_core::ane::model_adapted(ironmlx_core::ane::Model::FlashNext)
            || std::env::var("IRONMLX_DIAGNOSTIC_FLASH_ANE").as_deref() == Ok("1"))
}
pub(crate) struct Pending {
    context: Context,
    reply: Option<mpsc::Receiver<std::result::Result<(), String>>>,
}
impl Drop for Pending {
    fn drop(&mut self) {
        if let Some(reply) = self.reply.take() {
            let mut context = self.context.lock().unwrap_or_else(|e| e.into_inner());
            if !matches!(reply.recv_timeout(Duration::from_secs(15)), Ok(Ok(()))) {
                context.retired = true;
                ironmlx_core::ane::retire(ironmlx_core::ane::Model::FlashNext);
            }
            context.in_flight = false;
        }
    }
}
pub(crate) fn begin(
    x: &Array,
    gate: &Linear,
    up: &Linear,
    down: &Linear,
    target: StreamOrDevice,
) -> Result<Option<Pending>> {
    // A preparation/handoff error can retire the family while this forward's
    // outer scope is still alive. Do not repeat native preparation in later layers.
    if ironmlx_core::ane::retired(ironmlx_core::ane::Model::FlashNext) {
        return Ok(None);
    }
    let Some(context) = CONTEXT.with(|s| s.borrow().last().cloned()) else {
        return Ok(None);
    };
    let mut guard = context
        .lock()
        .map_err(|_| anyhow::anyhow!("Flash ANE context poisoned"))?;
    if guard.retired || guard.in_flight || x.size() != guard.rows * 2560 {
        return Ok(None);
    }
    let reply = guard.begin_job(x, gate, up, down, target)?;
    drop(guard);
    Ok(Some(Pending {
        context,
        reply: Some(reply),
    }))
}
pub(crate) fn finish(
    mut pending: Pending,
    routed: &Array,
    target: StreamOrDevice,
) -> Result<Option<Array>> {
    // Submit the independent routed graph before waiting for ANE, preserving
    // actual overlap. The wait is bounded and does not block the GPU worker.
    mlx::transforms::async_eval(&[routed])?;
    let reply = pending.reply.take().unwrap();
    let mut context = pending
        .context
        .lock()
        .map_err(|_| anyhow::anyhow!("Flash ANE context poisoned"))?;
    match context.finish_job(reply, target) {
        Ok(out) => Ok(Some(out)),
        Err(error) => {
            context.retired = true;
            ironmlx_core::ane::retire(ironmlx_core::ane::Model::FlashNext);
            tracing::warn!(%error,"Flash ANE job retired; recomputing shared expert on GPU");
            Ok(None)
        }
    }
}
impl Drop for SharedExpert {
    fn drop(&mut self) {
        self.jobs.take();
        if let Some(worker) = self.worker.take() {
            super::ane_ffn::close_worker(worker, ironmlx_core::ane::Model::FlashNext);
        }
    }
}
impl SharedExpert {
    pub(crate) fn new(rows: usize) -> Result<Self> {
        ensure!(
            ironmlx_core::ane::hardware().available,
            "qualified ANE hardware required"
        );
        ensure!(
            [128, 512, 2048].contains(&rows),
            "unqualified Flash row geometry"
        );
        let static_weights =
            std::env::var("IRONMLX_DIAGNOSTIC_FLASH_ANE_WEIGHTS").as_deref() == Ok("static");
        let projection_only =
            std::env::var("IRONMLX_DIAGNOSTIC_FLASH_ANE_MODE").as_deref() != Ok("full");
        let (h, i) = (2560, 640);
        let x = Surface::new(h, rows, Elem::F16)?;
        let gate = Surface::new(i, h, Elem::F16)?;
        let up = Surface::new(i, h, Elem::F16)?;
        let down = Surface::new(h, i, Elem::F16)?;
        let out_rows = if projection_only { i * 2 } else { h };
        let out = Surface::new(out_rows, rows, Elem::F16)?;
        let mut body = String::new();
        let mut line = |v: String| {
            body.push_str("        ");
            body.push_str(&v);
            body.push_str(";\n");
        };
        let t = |r: usize, c: usize| format!("tensor<fp16, [1, 1, {r}, {c}]>");
        for (name, r, c) in [("x", h, rows), ("gate", i, h), ("up", i, h), ("down", h, i)] {
            if name != "down" || !projection_only {
                line(format!(
                    "{} {name}_t = tensor_buffer_to_tensor<ios17>(input = {name})",
                    t(r, c)
                ));
            }
        }
        let segments = std::env::var("IRONMLX_DIAGNOSTIC_FLASH_ANE_SEGMENTS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(2);
        ensure!(
            [1, 2, 4].contains(&segments),
            "invalid Flash ANE contraction segmentation"
        );
        super::ane_ffn::segmented_matmul(&mut line, "g", "gate_t", "x_t", i, h, rows, segments);
        super::ane_ffn::segmented_matmul(&mut line, "u", "up_t", "x_t", i, h, rows, segments);
        if projection_only {
            line(format!(
                "{} result = concat(axis = int32(2), interleave = bool(false), values = (g, u))",
                t(i * 2, rows)
            ));
        } else {
            line(format!(
                "{} half = mul(x = g, y = fp16(0x1p-1))",
                t(i, rows)
            ));
            line(format!("{} th = tanh(x = half)", t(i, rows)));
            line(format!("{} tp = add(x = th, y = fp16(0x1p+0))", t(i, rows)));
            line(format!("{} si = mul(x = half, y = tp)", t(i, rows)));
            line(format!("{} mid = mul(x = si, y = u)", t(i, rows)));
            line(format!("{} result = matmul(transpose_x = bool(false), transpose_y = bool(false), x = down_t, y = mid)",t(h,rows)));
        }
        let stride = out.stride();
        let plane = out_rows * stride;
        line(format!("{} y = tensor_to_tensor_buffer<ios17>(input = result, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>([{plane}, {plane}, {stride}, 1]))",out.buffer_type(out_rows)));
        let down_parameter = if projection_only {
            String::new()
        } else {
            format!(", {} down", down.buffer_type(h))
        };
        let mil=format!("program(1.3)\n{{\n    func f<ios18>({} x, {} gate, {} up{down_parameter}) {{\n{body}    }} -> (y);\n}}\n",x.buffer_type(h),gate.buffer_type(i),up.buffer_type(i));
        let program = Program::load_bounded(
            &mil,
            &ironmlx_ane::empty_weights(),
            std::time::Duration::from_secs(30),
        )
        .context("compiling independent Flash shared-expert program")?;
        let buffers = Arc::new(Buffers {
            x,
            gate,
            up,
            down,
            out,
        });
        ironmlx_core::ane::retain_surfaces(ironmlx_core::ane::Model::FlashNext, buffers.bytes());
        let ready = Arc::new(mlx::fast::HandoffEvent::new()?);
        let (jobs, queue) = mpsc::channel::<Job>();
        let (b, r) = (Arc::clone(&buffers), Arc::clone(&ready));
        let worker = std::thread::Builder::new()
            .name("ironmlx-flash-ane".into())
            .spawn(move || {
                let mut constants = std::collections::HashMap::new();
                for job in queue {
                    if let Some(program) = job.constant {
                        constants.insert(job.key, program);
                    }
                    let result = (|| -> Result<()> {
                        ensure!(r.cpu_wait(job.value, 10000)?, "Flash GPU handoff timeout");
                        ensure!(
                            std::env::var("IRONMLX_DIAGNOSTIC_FLASH_ANE_FAIL").as_deref()
                                != Ok("1"),
                            "injected Flash ANE failure"
                        );
                        if static_weights {
                            constants
                                .get(&job.key)
                                .context("Flash constant program missing")?
                                .evaluate(0, &[("x", &b.x)], &b.out)?;
                        } else {
                            let mut inputs = vec![("x", &b.x), ("gate", &b.gate), ("up", &b.up)];
                            if !projection_only {
                                inputs.push(("down", &b.down));
                            }
                            program.evaluate(0, &inputs, &b.out)?;
                        }
                        // Validate fp16 output on the worker, before the GPU may
                        // consume it, without forcing a GPU synchronization.
                        let values = unsafe {
                            std::slice::from_raw_parts(
                                b.out.base() as *const u16,
                                b.out.rows * b.out.stride(),
                            )
                        };
                        ensure!(
                            values.iter().all(|v| v & 0x7c00 != 0x7c00),
                            "nonfinite Flash ANE result"
                        );
                        ironmlx_core::ane::record_completed(ironmlx_core::ane::Model::FlashNext);
                        Ok(())
                    })()
                    .map_err(|e| format!("{e:#}"));
                    let _ = job.reply.send(result);
                }
            })?;
        let context = Self {
            buffers,
            rows,
            projection_only,
            static_weights,
            registered: std::collections::HashSet::new(),
            ready,
            next: 0,
            jobs: Some(jobs),
            worker: Some(worker),
            in_flight: false,
            retired: false,
            last_read: None,
        };
        Ok(context)
    }
    pub(crate) fn surface_bytes(&self) -> usize {
        self.buffers.bytes()
    }
    /// Static per-layer gate/up weights let ANE use its convolution weight
    /// path. Programs stay model-owned; checkpoint files remain read-only.
    fn constant_program(
        &self,
        gate: &Linear,
        up: &Linear,
        down: &Linear,
        target: StreamOrDevice,
    ) -> Result<Program> {
        let mut gate_up = Vec::new();
        for linear in [gate, up] {
            let w = decode_weight(linear, target)?;
            ensure!(
                w.shape().as_slice() == [640, 2560],
                "Flash constant gate/up shape"
            );
            gate_up.extend(w.to_vec::<half::f16>()?.into_iter().map(|v| v.to_bits()));
        }
        let mut tensors = vec![gate_up];
        if !self.projection_only {
            let w = decode_weight(down, target)?;
            ensure!(
                w.shape().as_slice() == [2560, 640],
                "Flash constant down shape"
            );
            tensors.push(
                w.to_vec::<half::f16>()?
                    .into_iter()
                    .map(|v| v.to_bits())
                    .collect(),
            );
        }
        let (blob, offsets) = constant_weights(&tensors);
        let rows = self.rows;
        let b = &self.buffers;
        let stride = b.out.stride();
        let out_channels = if self.projection_only { 1280 } else { 2560 };
        let plane = out_channels * stride;
        let mut body=format!("        tensor<fp16, [1, 1, 2560, {rows}]> xt = tensor_buffer_to_tensor<ios17>(input = x);\n        tensor<fp16, [1, 2560, 1, {rows}]> xc = reshape(x = xt, shape = tensor<int32, [4]>([1, 2560, 1, {rows}]));\n        tensor<fp16, [1280, 2560, 1, 1]> w = const()[name = string(\"gate_up\"), val = tensor<fp16, [1280, 2560, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights.bin\"), offset = uint64({offset})))];\n        tensor<fp16, [1, 1280, 1, {rows}]> yc = conv(dilations = tensor<int32, [2]>([1, 1]), groups = int32(1), pad = tensor<int32, [4]>([0, 0, 0, 0]), pad_type = string(\"valid\"), strides = tensor<int32, [2]>([1, 1]), weight = w, x = xc);\n",offset=offsets[0]);
        let output = if self.projection_only {
            "yc"
        } else {
            body.push_str(&format!("        tensor<fp16, [1, 640, 1, {rows}]> g = slice_by_size(x = yc, begin = tensor<int32, [4]>([0, 0, 0, 0]), size = tensor<int32, [4]>([1, 640, 1, {rows}]));\n        tensor<fp16, [1, 640, 1, {rows}]> u = slice_by_size(x = yc, begin = tensor<int32, [4]>([0, 640, 0, 0]), size = tensor<int32, [4]>([1, 640, 1, {rows}]));\n        tensor<fp16, [1, 640, 1, {rows}]> sg = sigmoid(x = g);\n        tensor<fp16, [1, 640, 1, {rows}]> si = mul(x = sg, y = g);\n        tensor<fp16, [1, 640, 1, {rows}]> mid = mul(x = si, y = u);\n        tensor<fp16, [2560, 640, 1, 1]> wd = const()[name = string(\"down\"), val = tensor<fp16, [2560, 640, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights.bin\"), offset = uint64({offset})))];\n        tensor<fp16, [1, 2560, 1, {rows}]> yd = conv(dilations = tensor<int32, [2]>([1, 1]), groups = int32(1), pad = tensor<int32, [4]>([0, 0, 0, 0]), pad_type = string(\"valid\"), strides = tensor<int32, [2]>([1, 1]), weight = wd, x = mid);\n",offset=offsets[1]));
            "yd"
        };
        body.push_str(&format!("        tensor<fp16, [1, 1, {out_channels}, {rows}]> yt = reshape(x = {output}, shape = tensor<int32, [4]>([1, 1, {out_channels}, {rows}]));\n        {buffer} y = tensor_to_tensor_buffer<ios17>(input = yt, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>([{plane}, {plane}, {stride}, 1]));\n",buffer=b.out.buffer_type(out_channels)));
        let mil = format!(
            "program(1.3)\n{{\n    func f<ios18>({input} x) {{\n{body}    }} -> (y);\n}}\n",
            input = b.x.buffer_type(2560)
        );
        Program::load_bounded(&mil, &blob, std::time::Duration::from_secs(30))
    }
    fn begin_job(
        &mut self,
        x: &Array,
        gate: &Linear,
        up: &Linear,
        down: &Linear,
        target: StreamOrDevice,
    ) -> Result<mpsc::Receiver<std::result::Result<(), String>>> {
        ensure!(
            !self.in_flight && !self.retired,
            "Flash ANE context is busy or retired"
        );
        ensure!(
            x.size() == self.rows * 2560 && x.dtype() == Dtype::Bfloat16,
            "Flash ANE input contract"
        );
        let key = gate as *const Linear as usize;
        let constant = if self.static_weights && !self.registered.contains(&key) {
            Some(self.constant_program(gate, up, down, target)?)
        } else {
            None
        };
        let b = &self.buffers;
        let mut writes = Vec::new();
        let after = self.last_read.iter().collect::<Vec<_>>();
        for (index, (linear, surface)) in [(gate, &b.gate), (up, &b.up), (down, &b.down)]
            .into_iter()
            .enumerate()
        {
            if self.static_weights || (self.projection_only && index == 2) {
                continue;
            }
            let p = linear
                .quantized_parts()
                .context("Flash ANE requires native g32 weights")?;
            ensure!(
                p.group_size == 32
                    && p.bits == 4
                    && p.mode == crate::core::weights::QuantMode::Affine
                    && p.bias.is_none(),
                "Flash ANE affine4/g32 contract"
            );
            let w = decode_weight(linear, target)?;
            writes.push(mlx::fast::copy_into_external_on(
                &w,
                &surface_array_owned(surface, Arc::clone(b))?,
                &after,
                target,
            )?);
        }
        let input = x
            .reshape_on([self.rows as i32, 2560], target)?
            .transpose_axes_on(&[1, 0][..], target)?
            .astype_on(Dtype::Float16, target)?;
        writes.push(mlx::fast::copy_into_external_on(
            &input,
            &surface_array_owned(&b.x, Arc::clone(b))?,
            &after,
            target,
        )?);
        self.next += 1;
        let ready =
            self.ready
                .signal_after_on(x, &writes.iter().collect::<Vec<_>>(), self.next, target)?;
        mlx::transforms::async_eval(&[&ready])?;
        let (reply, receiver) = mpsc::channel();
        self.jobs
            .as_ref()
            .context("Flash ANE worker closed")?
            .send(Job {
                value: self.next,
                key,
                constant,
                reply,
            })?;
        if self.static_weights {
            self.registered.insert(key);
        }
        self.in_flight = true;
        Ok(receiver)
    }
    fn finish_job(
        &mut self,
        reply: mpsc::Receiver<std::result::Result<(), String>>,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let result = reply.recv_timeout(Duration::from_secs(15));
        self.in_flight = false;
        result
            .context("Flash ANE result timeout")?
            .map_err(anyhow::Error::msg)?;
        // Cast allocates and evaluates an independent MLX-owned result before
        // the reusable IOSurface may be written by the next layer/request.
        let out = surface_array_owned(&self.buffers.out, Arc::clone(&self.buffers))?
            .transpose_axes_on(&[1, 0][..], target)?
            .astype_on(Dtype::Bfloat16, target)?;
        self.last_read = Some(out.clone());
        Ok(out)
    }
    #[cfg(test)]
    fn run(
        &mut self,
        x: &Array,
        gate: &Linear,
        up: &Linear,
        down: &Linear,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let reply = self.begin_job(x, gate, up, down, target)?;
        let out = self.finish_job(reply, target)?;
        if self.projection_only {
            let g = out.slice(&[0, 0][..], &[self.rows as i32, 640][..])?;
            let u = out.slice(&[0, 640][..], &[self.rows as i32, 1280][..])?;
            let mid = &(&g.sigmoid_on(target)? * &g) * &u;
            down.forward_on(&mid, target)?
                .reshape_on(x.shape().as_slice(), target)
                .map_err(Into::into)
        } else {
            out.reshape_on(x.shape().as_slice(), target)
                .map_err(Into::into)
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{core::Loader, nn::Mlp};
    #[test]
    #[ignore = "requires IRONMLX_FLASH_SNAPSHOT and actual qualified ANE"]
    fn flash_shared_expert_hardware_candidate() -> Result<()> {
        mlx::metal::set_metallib_path(&format!("{}/lib/mlx.metallib", std::env::var("MLX_DIR")?))?;
        let loader = Loader::open_flash_text_paged(
            std::path::Path::new(&std::env::var("IRONMLX_FLASH_SNAPSHOT")?),
            8u64 << 30,
        )?;
        let prefix = "model.layers.0.mlp.shared_expert";
        let gate = Linear::from_loader(&loader, &format!("{prefix}.gate_proj"))?;
        let up = Linear::from_loader(&loader, &format!("{prefix}.up_proj"))?;
        let down = Linear::from_loader(&loader, &format!("{prefix}.down_proj"))?;
        let gpu = Mlp::from_loader(&loader, prefix)?;
        drop(loader);
        let mut results = Vec::new();
        for rows in [128, 512, 2048] {
            let values = (0..rows * 2560)
                .map(|n| ((n as f32) * 0.017).sin() * 0.8)
                .collect::<Vec<_>>();
            let x = Array::try_from((values.as_slice(), [rows as i32, 2560]))?
                .astype(Dtype::Bfloat16)?;
            let mut candidate = SharedExpert::new(rows)?;
            let mut times = Vec::new();
            let mut relative = 0.;
            for _ in 0..4 {
                let start = std::time::Instant::now();
                let want = gpu.forward_on(&x, ())?;
                let w = want.astype(Dtype::Float32)?.to_vec::<f32>()?;
                let gpu_s = start.elapsed().as_secs_f64();
                let start = std::time::Instant::now();
                let got = candidate.run(&x, &gate, &up, &down, ().into())?;
                let a = got.astype(Dtype::Float32)?.to_vec::<f32>()?;
                let ane_s = start.elapsed().as_secs_f64();
                ensure!(
                    a.iter().all(|v| v.is_finite()),
                    "nonfinite Flash ANE result"
                );
                let error = a
                    .iter()
                    .zip(&w)
                    .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                    .sum::<f64>();
                let mag = w.iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
                relative = (error / mag).sqrt();
                times.push((gpu_s, ane_s));
            }
            eprintln!(
                "FLASH_ANE_SHARED rows={rows} relative={relative} surface_bytes={} times={times:?}",
                candidate.surface_bytes()
            );
            results.push(serde_json::json!({"rows":rows,"relative_rms":relative,"surface_bytes":candidate.surface_bytes(),"times":times}));
        }
        std::fs::write(
            std::env::var("ANE_FLASH_CANDIDATE_OUTPUT")?,
            serde_json::to_vec_pretty(&results)?,
        )?;
        Ok(())
    }
}
