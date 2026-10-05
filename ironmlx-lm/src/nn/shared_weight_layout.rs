//! Experimental single-resident affine4 weight storage for the Qwen3.8
//! DFlash2 M5 lane.
//!
//! Baseline keeps the checkpoint layout for bulk prefill (M > 128) and lazily
//! adds a persistent tiled copy for the M5 decode/verify kernel (M <= 128), so
//! both layouts stay resident. Here a store holds exactly one layout and
//! converts on demand with an exact byte permutation; the kernels, dispatch
//! decisions and accumulation order are those of the baseline. Default off:
//! `IRONMLX_EXPERIMENTAL_M5_SHARED_WEIGHT_LAYOUT=1`.
use super::linear::{Linear, QuantizedLinearParts};
use super::m5_affine4::{self, Prepared};
use crate::core::QuantMode;
use crate::Result;
use anyhow::anyhow;
use mlx::{Array, StreamOrDevice};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock};

static TO_TILED: AtomicUsize = AtomicUsize::new(0);
static TO_NATIVE: AtomicUsize = AtomicUsize::new(0);
static STORES: AtomicUsize = AtomicUsize::new(0);

/// Diagnostic counters: (stores, conversions to tiled, conversions to native).
pub fn conversion_totals() -> (usize, usize, usize) {
    (
        STORES.load(Relaxed),
        TO_TILED.load(Relaxed),
        TO_NATIVE.load(Relaxed),
    )
}

pub(crate) fn requested() -> bool {
    static REQUESTED: OnceLock<bool> = OnceLock::new();
    *REQUESTED.get_or_init(|| {
        ironmlx_core::m5_profile::flag(ironmlx_core::m5_profile::settings::M5_SHARED_WEIGHT_LAYOUT)
    })
}

/// Fail closed on experiment combinations whose route order precedes M5 or
/// that change the tiled layout.
pub(crate) fn validate_environment() -> Result<()> {
    for name in [
        "IRONMLX_EXPERIMENTAL_PREFILL_PREPARED_GEMM",
        "IRONMLX_EXPERIMENTAL_PREFILL_DEQUANT_GEMM",
        "IRONMLX_EXPERIMENTAL_M5_QMM_COOP",
    ] {
        if std::env::var(name).is_ok_and(|value| !value.is_empty() && value != "0") {
            return Err(anyhow!(
                "experimental shared weight layout cannot be combined with {name}"
            ));
        }
    }
    Ok(())
}

enum State {
    Native {
        weight: Array,
        scales: Array,
        biases: Array,
    },
    Tiled(Prepared),
}

pub(crate) struct Store {
    state: Mutex<State>,
    // Behind a Mutex only for `Sync`; read-only after construction.
    bias: Mutex<Option<Array>>,
    group_size: i32,
    bits: i32,
    mode: QuantMode,
    kw: i32,
}

impl Store {
    fn bias(&self) -> Result<Option<Array>> {
        Ok(self
            .bias
            .lock()
            .map_err(|_| anyhow!("shared layout poisoned"))?
            .clone())
    }

    fn native(&self, target: StreamOrDevice) -> Result<(Array, Array, Array)> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("shared layout poisoned"))?;
        if let State::Tiled(prepared) = &*state {
            let (weight, scales, biases) = m5_affine4::native_from_tiled(prepared, target)?;
            *state = State::Native {
                weight,
                scales,
                biases,
            };
            TO_NATIVE.fetch_add(1, Relaxed);
        }
        match &*state {
            State::Native {
                weight,
                scales,
                biases,
            } => Ok((weight.clone(), scales.clone(), biases.clone())),
            State::Tiled(_) => unreachable!("converted above"),
        }
    }

    fn tiled_forward(&self, x: &Array, target: StreamOrDevice) -> Result<Array> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("shared layout poisoned"))?;
        if let State::Native {
            weight,
            scales,
            biases,
        } = &*state
        {
            let prepared = m5_affine4::tiled_from_native(weight, scales, biases, target)?;
            *state = State::Tiled(prepared);
            TO_TILED.fetch_add(1, Relaxed);
        }
        match &*state {
            State::Tiled(prepared) => {
                let bias = self.bias()?;
                m5_affine4::forward_tiled(x, prepared, bias.as_ref(), target)
            }
            State::Native { .. } => unreachable!("converted above"),
        }
    }
}

/// A Linear's view of a store: the whole projection, or a row range of a
/// fused projection (the former split views).
pub(crate) struct Shared {
    store: Arc<Store>,
    rows: Option<(i32, i32)>,
    in_features: usize,
    out_features: usize,
}

impl Shared {
    pub(crate) fn in_features(&self) -> usize {
        self.in_features
    }

    pub(crate) fn out_features(&self) -> usize {
        self.out_features
    }

    /// Ordinary Linear holding the current checkpoint-layout arrays (row
    /// sliced for views). Its own route selection is the baseline's.
    fn native_linear(&self, target: StreamOrDevice) -> Result<Linear> {
        let (mut weight, mut scales, mut biases) = self.store.native(target)?;
        let mut bias = self.store.bias()?;
        if let Some((start, len)) = self.rows {
            let rows = |array: &Array| -> Result<Array> {
                let width = array.shape().as_slice()[1];
                Ok(mlx::ops::indexing::slice_on(
                    array,
                    &[start, 0][..],
                    &[start + len, width][..],
                    target,
                )?)
            };
            weight = rows(&weight)?;
            scales = rows(&scales)?;
            biases = rows(&biases)?;
            bias = bias
                .map(|array| {
                    mlx::ops::indexing::slice_on(&array, &[start][..], &[start + len][..], target)
                })
                .transpose()?;
        }
        Ok(Linear::new_quant_with_mode(
            weight,
            scales,
            Some(biases),
            bias,
            self.store.group_size,
            self.store.bits,
            self.store.mode,
        ))
    }

    /// Whole-store views take the M5 route; row views use the native path.
    pub(crate) fn m5_route_capable(&self) -> bool {
        self.rows.is_none()
    }

    pub(crate) fn forward_on(&self, x: &Array, target: StreamOrDevice) -> Result<Array> {
        if self.rows.is_none() && m5_affine4::routes_activation(x, self.store.kw) {
            return self.store.tiled_forward(x, target);
        }
        self.native_linear(target)?.forward_on(x, target)
    }

    pub(crate) fn forward_positions_isolated_on(
        &self,
        x: &Array,
        target: StreamOrDevice,
    ) -> Result<Array> {
        self.native_linear(target)?
            .forward_positions_isolated_on(x, target)
    }

    pub(crate) fn forward_mtp_verify_on(&self, x: &Array, target: StreamOrDevice) -> Result<Array> {
        self.native_linear(target)?.forward_mtp_verify_on(x, target)
    }
}

/// Move a quantized projection's arrays into a new store. Returns `None`
/// (leaving the Linear untouched) when the projection is not an M5-eligible
/// affine4/g64 BF16 weight.
pub(crate) fn take(linear: &Linear) -> Option<Arc<Store>> {
    let parts: QuantizedLinearParts<'_> = linear.quantized_parts()?;
    if !m5_affine4::weight_compatible(&parts) {
        return None;
    }
    let store = Store {
        kw: parts.weight.shape().as_slice()[1],
        state: Mutex::new(State::Native {
            weight: parts.weight.clone(),
            scales: parts.scales.clone(),
            biases: parts.biases?.clone(),
        }),
        bias: Mutex::new(parts.bias.cloned()),
        group_size: parts.group_size,
        bits: parts.bits,
        mode: parts.mode,
    };
    STORES.fetch_add(1, Relaxed);
    Some(Arc::new(store))
}

pub(crate) fn whole(store: &Arc<Store>, in_features: usize, out_features: usize) -> Shared {
    Shared {
        store: Arc::clone(store),
        rows: None,
        in_features,
        out_features,
    }
}

pub(crate) fn rows(store: &Arc<Store>, start: i32, len: i32, in_features: usize) -> Shared {
    Shared {
        store: Arc::clone(store),
        rows: Some((start, len)),
        in_features,
        out_features: usize::try_from(len).expect("positive row count"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::Dtype;
    use serial_test::serial;

    fn quantized(n: i32, k: i32, seed: u64) -> Result<Vec<Array>> {
        let key = mlx::random::key(seed)?;
        let w = mlx::random::normal()
            .shape(&[n, k][..])
            .dtype(Dtype::Float32)
            .scale(0.02)
            .key(&key)
            .sample()?;
        let w = mlx::ops::cast::astype(&w, Dtype::Bfloat16)?;
        let q = mlx::quantization::quantize(&w, Some(64), Some(4), "affine", None)?;
        mlx::transforms::eval(&[&q[0], &q[1], &q[2]])?;
        Ok(q)
    }

    fn bits(a: &Array) -> Result<Vec<u32>> {
        Ok(mlx::ops::cast::astype(a, Dtype::Float32)?
            .to_vec::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    }

    /// Shared layout must reproduce the baseline Linear bit-for-bit across
    /// prefill/decode alternation (both conversion directions), for the whole
    /// projection and for row views, with and without the M5 scope.
    #[test]
    #[serial(mlx_metal)]
    fn shared_layout_matches_baseline_bitwise_across_conversions() -> Result<()> {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        }
        if !mlx::metal::architecture().is_ok_and(|arch| {
            arch.strip_prefix("applegpu_g")
                .and_then(|x| x.get(..2))
                .and_then(|x| x.parse::<u32>().ok())
                .is_some_and(|generation| generation >= 17)
        }) {
            return Ok(());
        }
        let (n, k, split) = (1024, 1024, 384);
        let q = quantized(n, k, 20261002)?;
        let make =
            || Linear::new_quant(q[0].clone(), q[1].clone(), Some(q[2].clone()), None, 64, 4);
        let baseline = make();
        let baseline_fused = make();
        let mut views = baseline_fused
            .split_quantized_outputs(&[split as usize, (n - split) as usize], "test split")?;
        let mut shared = make();
        let store = shared.share_whole()?.expect("eligible affine4 projection");
        let mut first = views.remove(0);
        let mut second = views.remove(0);
        let baseline_first_view = baseline_fused
            .split_quantized_outputs(&[split as usize, (n - split) as usize], "test split")?;
        first.share_rows(&store, 0)?;
        second.share_rows(&store, split as usize)?;
        assert!(shared.quantized_parts().is_none());
        assert_eq!(
            (shared.in_features(), shared.out_features()),
            (k as usize, n as usize)
        );
        assert_eq!(second.out_features(), (n - split) as usize);
        let before = conversion_totals();
        let key = mlx::random::key(7)?;
        for (step, m) in [2048, 1, 16, 12, 2048, 129, 1, 2048]
            .into_iter()
            .enumerate()
        {
            let x = mlx::random::normal()
                .shape(&[1, m, k][..])
                .dtype(Dtype::Float32)
                .key(&mlx::random::split(&key)?.0)
                .sample()?;
            let x = mlx::ops::cast::astype(&x, Dtype::Bfloat16)?;
            for armed in [true, false] {
                let _scope = armed.then(m5_affine4::scope);
                let expected = bits(&baseline.forward(&x)?)?;
                let actual = bits(&shared.forward(&x)?)?;
                assert_eq!(expected, actual, "whole step={step} M={m} armed={armed}");
                for (index, view) in [&first, &second].into_iter().enumerate() {
                    let expected = bits(&baseline_first_view[index].forward(&x)?)?;
                    let actual = bits(&view.forward(&x)?)?;
                    assert_eq!(
                        expected, actual,
                        "view{index} step={step} M={m} armed={armed}"
                    );
                }
            }
        }
        let after = conversion_totals();
        // 2048 -> 1 -> ... -> 2048 -> 129 -> 1 -> 2048: tiled at steps 1, 6;
        // native at steps 4, 7 (views and unarmed calls also need native).
        assert!(
            after.1 > before.1 && after.2 > before.2,
            "{before:?} -> {after:?}"
        );
        Ok(())
    }

    /// Diagnostic for the C2 Decode question: production-scale M=16 sweeps
    /// over all 257 projections with tiled weights that stay resident (R)
    /// versus freshly round-tripped before every sweep block (F), under a
    /// bounded MLX cache so conversions allocate new buffers. Run with
    /// `--ignored --nocapture`; prints per-sweep walls.
    #[test]
    #[ignore]
    #[serial(mlx_metal)]
    fn decode_sweep_resident_versus_reconverted() -> Result<()> {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        }
        let shapes = [
            (34816, 5120, 64),
            (5120, 17408, 64),
            (16480, 5120, 48),
            (5120, 6144, 64),
            (14336, 5120, 16),
            (248320, 5120, 1),
        ];
        let target = StreamOrDevice::default();
        let mut stores = Vec::new();
        let mut inputs = std::collections::HashMap::new();
        for (seed, &(n, k, count)) in shapes.iter().enumerate() {
            let q = quantized(n, k, 700 + seed as u64)?;
            for _ in 0..count {
                // Distinct buffers per layer, as in the model.
                let copy = |a: &Array| -> Result<Array> {
                    let c = mlx::ops::shape::contiguous_on(
                        &mlx::ops::multiply(a, &mlx::ops::ones_like(a)?)?,
                        false,
                        target,
                    )?;
                    mlx::transforms::eval(&[&c])?;
                    Ok(c)
                };
                let state = State::Native {
                    weight: copy(&q[0])?,
                    scales: copy(&q[1])?,
                    biases: copy(&q[2])?,
                };
                stores.push((
                    k,
                    Store {
                        state: Mutex::new(state),
                        bias: Mutex::new(None),
                        group_size: 64,
                        bits: 4,
                        mode: QuantMode::Affine,
                        kw: k / 8,
                    },
                ));
            }
            let x = mlx::random::normal()
                .shape(&[1, 16, k][..])
                .dtype(Dtype::Float32)
                .key(&mlx::random::key(900 + seed as u64)?)
                .sample()?;
            let x = mlx::ops::cast::astype(&x, Dtype::Bfloat16)?;
            mlx::transforms::eval(&[&x])?;
            inputs.insert(k, x);
            mlx::clear_cache();
        }
        let previous = mlx::memory::set_cache_limit(2_000_000_000);
        let _route = m5_affine4::scope();
        let sweep = |stores: &Vec<(i32, Store)>| -> Result<f64> {
            let started = std::time::Instant::now();
            let mut outs = Vec::with_capacity(stores.len());
            for (k, store) in stores {
                outs.push(store.tiled_forward(&inputs[k], target)?);
            }
            mlx::transforms::eval(&outs.iter().collect::<Vec<_>>())?;
            Ok(started.elapsed().as_secs_f64() * 1e3)
        };
        let reconvert = |stores: &Vec<(i32, Store)>| -> Result<()> {
            for (_, store) in stores {
                let (w, s, b) = store.native(target)?;
                mlx::transforms::eval(&[&w, &s, &b])?;
            }
            Ok(())
        };
        // Initial conversion to tiled (resident from here on for R).
        sweep(&stores)?;
        for _ in 0..3 {
            sweep(&stores)?;
        }
        for block in 0..6 {
            for &fresh in &[false, true, true, false] {
                if fresh {
                    reconvert(&stores)?;
                }
                let walls = (0..12)
                    .map(|_| sweep(&stores))
                    .collect::<Result<Vec<_>>>()?;
                eprintln!(
                    "sweep block={block} fresh={fresh} first_ms={:.3} rest_ms={}",
                    walls[0],
                    walls[1..]
                        .iter()
                        .map(|w| format!("{w:.3}"))
                        .collect::<Vec<_>>()
                        .join(",")
                );
            }
        }
        mlx::memory::set_cache_limit(previous);
        Ok(())
    }
}
