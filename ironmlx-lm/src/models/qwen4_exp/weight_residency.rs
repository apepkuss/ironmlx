//! Explicit residency of full Flash text weights, independent of process KV.
use crate::core::Loader;
use anyhow::{bail, Context, Result};

const GIB: usize = 1 << 30;

fn qualified(full: bool, device: Option<&str>, physical: Option<usize>) -> bool {
    full && device == Some("Apple M5 Max") && physical.is_some_and(|n| n >= 128 * GIB)
}

fn text_weight(key: &str) -> bool {
    [
        "model.layers.",
        "model.embed_tokens.",
        "model.hyper_connection_mixer.",
        "lm_head.",
    ]
    .iter()
    .any(|prefix| key.starts_with(prefix))
        && !key.contains(".ngram_embedding.")
        && !key.contains(".mtp.")
}

pub(super) fn prepare(loader: &Loader, full: bool) -> Result<Option<mlx::memory::WeightResidency>> {
    let mode = std::env::var("IRONMLX_FLASH_GPU_RESIDENCY").unwrap_or_else(|_| "auto".into());
    match mode.as_str() {
        "off" => return Ok(None),
        "auto" => {}
        _ => bail!("IRONMLX_FLASH_GPU_RESIDENCY must be auto or off"),
    }
    let memory = mlx::memory::snapshot();
    if !qualified(full, memory.device_name.as_deref(), memory.total_bytes) {
        return Ok(None);
    }
    // Allow allocator/page alignment, while bounding
    // the request to the loaded text inventory, not process active memory.
    let weights: Vec<_> = loader
        .keys()
        .filter(|key| text_weight(key))
        .filter_map(|key| loader.tensor_opt(key))
        .collect();
    let budget = weights
        .iter()
        .try_fold(64usize << 20, |bytes, a| {
            a.size()
                .checked_mul(a.dtype().byte_size())
                .and_then(|n| bytes.checked_add(n))
        })
        .context("Flash weight residency size overflow")?;
    match mlx::memory::WeightResidency::new(&weights, budget) {
        Ok(lease) => {
            tracing::info!(
                bytes = lease.bytes(),
                "Flash text weight GPU residency prepared"
            );
            Ok(Some(lease))
        }
        Err(error) => {
            tracing::warn!(%error, "Flash GPU residency unavailable; using automatic Metal residency");
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_qualified_full_weight_hardware_enables_residency() {
        assert!(qualified(true, Some("Apple M5 Max"), Some(128 * GIB)));
        for memory in [32, 64] {
            assert!(!qualified(true, Some("Apple M5 Max"), Some(memory * GIB)));
        }
        assert!(!qualified(false, Some("Apple M5 Max"), Some(128 * GIB)));
        for device in [
            None,
            Some("Apple M1 Pro"),
            Some("Apple M4 Max"),
            Some("Apple M5 Pro"),
            Some("Apple M6 Max"),
        ] {
            assert!(!qualified(true, device, Some(128 * GIB)));
        }
        assert!(!qualified(true, Some("Apple M5 Max"), None));
    }
    #[test]
    fn residency_excludes_paged_tables_vision_and_auxiliary_models() {
        assert!(text_weight(
            "model.layers.0.mlp.switch_mlp.gate_proj.weight"
        ));
        assert!(text_weight("lm_head.scales"));
        assert!(text_weight("model.embed_tokens.weight"));
        assert!(!text_weight(
            "model.layers.0.ple.ple_embedding.ngram_embedding.weight"
        ));
        assert!(!text_weight("vision_tower.blocks.0.weight"));
        assert!(!text_weight("model.mtp.layers.0.weight"));
        assert!(!text_weight("draft.layers.0.weight"));
    }
}
