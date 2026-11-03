//! Flash-Next text contract. Vision metadata is retained in config.json but
//! is not admission to the text execution path or a multimodal capability.
use anyhow::{ensure, Context, Result};
use serde::Deserialize;

use super::ple_index::{PleIndex, PleIndexSpec};

#[derive(Clone, Debug, Deserialize)]
pub struct FlashRope {
    pub rope_theta: f32,
    pub rope_type: String,
    pub partial_rotary_factor: f32,
    pub mrope_interleaved: bool,
    pub mrope_section: Vec<i32>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct FlashTextConfig {
    pub model_type: String,
    pub hidden_size: i32,
    pub vocab_size: u32,
    pub num_hidden_layers: usize,
    pub max_position_embeddings: u32,
    pub layer_types: Vec<String>,
    pub hidden_act: String,
    pub output_gate_type: String,
    pub attention_bias: bool,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
    pub hc_count: i32,
    pub hc_lowrank: i32,
    pub num_experts: i32,
    pub num_experts_per_tok: i32,
    pub moe_intermediate_size: i32,
    pub shared_expert_intermediate_size: i32,
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,
    pub linear_num_key_heads: i32,
    pub linear_num_value_heads: i32,
    pub linear_key_head_dim: i32,
    pub linear_value_head_dim: i32,
    pub linear_conv_kernel_dim: i32,
    pub mamba_ssm_dtype: String,
    pub indexer_budget: i32,
    pub indexer_compress_ratio: i32,
    pub indexer_head_dim: i32,
    pub indexer_kv_heads: i32,
    pub indexer_n_heads: i32,
    pub ple_layer_ids: Vec<usize>,
    pub ple_embed_dim: i32,
    pub ple_conv_kernel_size: i32,
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    pub ngram_vocab_size_base: u64,
    pub make_ngram_vocab_size_divisible_by: u64,
    pub split_ngram_parts: u64,
    #[serde(default = "default_seed")]
    pub seed: u64,
    pub eos_token_id: u32,
    pub rope_parameters: FlashRope,
}

fn default_true() -> bool {
    true
}
fn default_seed() -> u64 {
    1234
}

impl FlashTextConfig {
    pub fn from_root(root: &serde_json::Value) -> Result<Self> {
        ensure!(
            root["model_type"] == "qwen4_exp",
            "expected qwen4_exp checkpoint"
        );
        let cfg: Self = serde_json::from_value(root["text_config"].clone())
            .context("deserializing Flash-Next text_config")?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.model_type == "qwen4_exp_text",
            "expected qwen4_exp_text"
        );
        for value in [
            self.hidden_size,
            self.head_dim,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.hc_lowrank,
            self.num_experts,
            self.num_experts_per_tok,
            self.moe_intermediate_size,
            self.shared_expert_intermediate_size,
            self.linear_num_key_heads,
            self.linear_num_value_heads,
            self.linear_key_head_dim,
            self.linear_value_head_dim,
            self.linear_conv_kernel_dim,
            self.indexer_budget,
            self.indexer_compress_ratio,
            self.indexer_head_dim,
            self.indexer_kv_heads,
            self.indexer_n_heads,
            self.ple_embed_dim,
            self.ple_conv_kernel_size,
        ] {
            ensure!(value > 0, "Flash dimensions must be positive");
        }
        ensure!(
            self.hc_count > 1 && self.hidden_size.checked_mul(self.hc_count).is_some(),
            "invalid hyper-connection geometry"
        );
        ensure!(
            self.num_attention_heads
                .checked_mul(self.head_dim)
                .is_some()
                && self.num_attention_heads % self.num_key_value_heads == 0,
            "invalid Flash attention geometry"
        );
        ensure!(
            self.linear_num_value_heads % self.linear_num_key_heads == 0
                && self
                    .linear_num_key_heads
                    .checked_mul(self.linear_key_head_dim)
                    .and_then(|n| n.checked_mul(2))
                    .and_then(|n| self
                        .linear_num_value_heads
                        .checked_mul(self.linear_value_head_dim)
                        .and_then(|v| n.checked_add(v)))
                    .is_some(),
            "invalid Flash GDN geometry"
        );
        ensure!(
            self.num_experts_per_tok <= self.num_experts,
            "top-k exceeds expert count"
        );
        ensure!(
            self.vocab_size > 0
                && self.vocab_size <= i32::MAX as u32
                && self.max_position_embeddings > 0
                && self.max_position_embeddings <= i32::MAX as u32
                && self.eos_token_id < self.vocab_size,
            "invalid Flash vocabulary or context"
        );
        ensure!(
            self.hidden_act == "silu"
                && self.output_gate_type == "sigmoid"
                && !self.attention_bias
                && self.mamba_ssm_dtype == "float32",
            "unsupported Flash activation, attention bias or recurrent dtype"
        );
        ensure!(
            self.rms_norm_eps.is_finite() && self.rms_norm_eps > 0.,
            "invalid RMS epsilon"
        );
        ensure!(
            self.num_hidden_layers > 0
                && self.layer_types.len() == self.num_hidden_layers
                && self.layer_types.iter().all(|s| matches!(
                    s.as_str(),
                    "linear_attention" | "full_attention" | "qwen_sparse_attention"
                )),
            "invalid Flash layer types"
        );
        ensure!(
            self.indexer_kv_heads == 1
                && self.indexer_n_heads % self.indexer_kv_heads == 0
                && self.indexer_budget % self.indexer_compress_ratio == 0,
            "unsupported QSA indexer heads"
        );
        let rope = &self.rope_parameters;
        let rotary = self.head_dim as f32 * rope.partial_rotary_factor;
        ensure!(
            rope.rope_type == "default"
                && rope.rope_theta.is_finite()
                && rope.rope_theta > 0.
                && rope.partial_rotary_factor.is_finite()
                && rope.partial_rotary_factor > 0.
                && rope.partial_rotary_factor <= 1.
                && rotary.fract() == 0.
                && (rotary as i32) % 2 == 0
                && rotary <= self.indexer_head_dim as f32
                && rope.mrope_interleaved
                && rope.mrope_section.len() == 3
                && rope.mrope_section.iter().all(|&n| n > 0)
                && rope.mrope_section.iter().map(|&n| n as i64).sum::<i64>() * 2 == rotary as i64,
            "unsupported Flash RoPE geometry"
        );
        let mut prior = 0;
        for (index, &id) in self.ple_layer_ids.iter().enumerate() {
            ensure!(
                id > prior
                    && id <= self.num_hidden_layers
                    && self.layer_types[id - 1] == "linear_attention",
                "PLE layer IDs must be increasing, one-based linear-attention layers"
            );
            prior = id;
            let ple = self.ple_index(index)?;
            ensure!(
                (self.ple_embed_dim as usize).is_multiple_of(ple.heads()),
                "PLE width must divide across n-gram heads"
            );
        }
        self.ple_conv_kernel_size
            .checked_sub(1)
            .and_then(|n| n.checked_mul(i32::try_from(self.ngram_size).ok()?))
            .context("PLE convolution history overflows")?;
        Ok(())
    }

    pub fn ple_index(&self, index: usize) -> Result<PleIndex> {
        ensure!(
            index < self.ple_layer_ids.len(),
            "PLE layer index out of bounds"
        );
        PleIndex::new(PleIndexSpec {
            vocab_size: self.vocab_size,
            eos_token_id: self.eos_token_id,
            ngram_size: self.ngram_size,
            heads_per_ngram: self.heads_per_ngram,
            vocab_size_base: self.ngram_vocab_size_base,
            ple_layer_index: index,
            seed: self.seed,
            alignment: self.make_ngram_vocab_size_divisible_by,
            shard_count: self.split_ngram_parts,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn root() -> serde_json::Value {
        serde_json::from_str(include_str!(
            "../../../tests/fixtures/qwen38_flash_next/config.json"
        ))
        .unwrap()
    }
    #[test]
    fn snapshot_contract_uses_text_eos_and_default_seed() {
        let cfg = FlashTextConfig::from_root(&root()).unwrap();
        assert_eq!(cfg.eos_token_id, 248044);
        assert_eq!(cfg.seed, 1234);
        assert!(cfg.norm_topk_prob);
        assert_eq!(cfg.ple_layer_ids, [2]);
        assert_eq!(cfg.ple_index(0).unwrap().row_count(), 320001536);
    }

    #[test]
    fn rejects_incompatible_geometry_before_loading_weights() {
        for (key, value) in [
            ("hc_count", serde_json::json!(i32::MAX)),
            ("num_key_value_heads", serde_json::json!(0)),
            ("linear_num_value_heads", serde_json::json!(47)),
            ("ple_layer_ids", serde_json::json!([4])),
            ("ple_layer_ids", serde_json::json!([2, 2])),
            ("output_gate_type", serde_json::json!("silu")),
            ("eos_token_id", serde_json::json!(248320)),
            ("ple_embed_dim", serde_json::json!(2559)),
            ("ngram_size", serde_json::json!(0)),
        ] {
            let mut cfg = root();
            cfg["text_config"][key] = value;
            assert!(FlashTextConfig::from_root(&cfg).is_err(), "{key}");
        }
        let mut cfg = root();
        cfg["text_config"]["rope_parameters"]["partial_rotary_factor"] = serde_json::json!(0.3);
        assert!(FlashTextConfig::from_root(&cfg).is_err());
    }
}
