use std::time::Instant;

use mlx::Array;

use crate::Result;

use super::config::Gemma4LayerKind;

pub(super) fn vl_enabled() -> bool {
    std::env::var_os("IRONMLX_GEMMA4_VL_PROFILE").is_some()
}

pub(super) fn vl_layer_enabled() -> bool {
    vl_enabled() && std::env::var_os("IRONMLX_GEMMA4_VL_LAYER_PROFILE").is_some()
}

pub(super) fn eval(label: &str, arrays: &[&Array], start: Instant, enabled: bool) -> Result<()> {
    if enabled {
        mlx::transforms::eval(arrays)?;
        tracing::info!(
            "[gemma4-vl-profile] {label}_ms={:.3}",
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
    Ok(())
}

pub(super) fn log(label: &str, start: Instant, enabled: bool) {
    if enabled {
        tracing::info!(
            "[gemma4-vl-profile] {label}_ms={:.3}",
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
}

pub(super) fn eval_layer(
    label: &str,
    layer_idx: usize,
    layer_kind: Gemma4LayerKind,
    arrays: &[&Array],
    start: Instant,
    enabled: bool,
) -> Result<()> {
    if enabled {
        mlx::transforms::eval(arrays)?;
        tracing::info!(
            "[gemma4-vl-profile] {label}_ms={:.3} layer_idx={} layer_kind={}",
            start.elapsed().as_secs_f64() * 1000.0,
            layer_idx,
            layer_kind.as_key()
        );
    }
    Ok(())
}

pub(super) fn log_layer(
    label: &str,
    layer_idx: usize,
    layer_kind: Gemma4LayerKind,
    start: Instant,
    enabled: bool,
) {
    if enabled {
        tracing::info!(
            "[gemma4-vl-profile] {label}_ms={:.3} layer_idx={} layer_kind={}",
            start.elapsed().as_secs_f64() * 1000.0,
            layer_idx,
            layer_kind.as_key()
        );
    }
}
