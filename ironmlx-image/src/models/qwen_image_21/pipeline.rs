use std::path::Path;

use anyhow::{anyhow, Context};
use mlx::{Array, Dtype};

use crate::loader::{preflight_model_metadata, ComponentLoader};
use crate::Result;
use ironmlx_lm::core::{Tokenizer, TokenizerConfig};

use super::{
    FlowMatchSchedule, QwenImage21Condition, QwenImage21TextEncoder, QwenImage21Transformer,
    QwenImage21Vae,
};

const SYSTEM_PROMPT: &str = "Comprehend and analyze the provided prompt.";

#[derive(Debug, Clone, Copy)]
pub struct QwenImage21GenerationConfig {
    pub width: u32,
    pub height: u32,
    pub inference_steps: usize,
    pub seed: u64,
}

impl Default for QwenImage21GenerationConfig {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 1024,
            inference_steps: 40,
            seed: 0,
        }
    }
}

impl QwenImage21GenerationConfig {
    pub fn validate(self) -> Result<Self> {
        if self.width < 256
            || self.height < 256
            || self.width > 2048
            || self.height > 2048
            || !self.width.is_multiple_of(32)
            || !self.height.is_multiple_of(32)
        {
            return Err(anyhow!(
                "Qwen Image 2.1 width/height must be multiples of 32 in 256..=2048"
            ));
        }
        let pixels = u64::from(self.width) * u64::from(self.height);
        if pixels > 1_572_864 {
            return Err(anyhow!(
                "Qwen Image 2.1 output exceeds the 1,572,864-pixel service limit"
            ));
        }
        if !(2..=200).contains(&self.inference_steps) {
            return Err(anyhow!("Qwen Image 2.1 inference_steps must be in 2..=200"));
        }
        Ok(self)
    }
}

pub struct QwenImage21Pipeline {
    tokenizer: Tokenizer,
    system_prefix_tokens: usize,
    image_token_id: u32,
    text_encoder: QwenImage21TextEncoder,
    transformer: QwenImage21Transformer,
    vae: QwenImage21Vae,
}

impl QwenImage21Pipeline {
    pub fn load(model_dir: &Path) -> Result<Self> {
        preflight_model_metadata(model_dir)
            .context("preflighting Qwen Image 2.1 model metadata")?;

        let processor_dir = model_dir.join("processor");
        let tokenizer_config = TokenizerConfig::from_model_dir(&processor_dir)?;
        let tokenizer =
            Tokenizer::from_files(&processor_dir.join("tokenizer.json"), &tokenizer_config)?;
        let system_prefix_tokens = tokenizer
            .encode(
                &format!("<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"),
                false,
            )?
            .len();
        let image_token_id = tokenizer
            .token_to_id("<|image_pad|>")
            .ok_or_else(|| anyhow!("Qwen Image 2.1 tokenizer is missing <|image_pad|>"))?;

        let text_loader = ComponentLoader::open_filtered(model_dir, "text_encoder", |key, _| {
            key.starts_with("language_model.model.") || key.starts_with("vision_tower.")
        })?;
        let text_encoder = QwenImage21TextEncoder::from_loader(&text_loader)?;
        drop(text_loader);

        let transformer_loader = ComponentLoader::open(model_dir, "transformer")?;
        let transformer = QwenImage21Transformer::from_loader(&transformer_loader)?;
        drop(transformer_loader);

        let vae_loader = ComponentLoader::open_filtered(model_dir, "vae", |key, _| {
            key.starts_with("encoder.")
                || key.starts_with("quant_conv.")
                || key.starts_with("decoder.")
                || key.starts_with("post_quant_conv.")
        })?;
        let vae = QwenImage21Vae::from_loader(&vae_loader)?;
        drop(vae_loader);

        Ok(Self {
            tokenizer,
            system_prefix_tokens,
            image_token_id,
            text_encoder,
            transformer,
            vae,
        })
    }

    fn prompt_tokens(&self, prompt: &str) -> Result<Vec<u32>> {
        let prompt = if prompt.trim().is_empty() {
            " "
        } else {
            prompt
        };
        let formatted = format!(
            "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        );
        let tokens = self.tokenizer.encode(&formatted, false)?;
        if tokens.len() > QwenImage21TextEncoder::MAX_SEQUENCE_LENGTH {
            return Err(anyhow!(
                "Qwen Image 2.1 prompt expands to {} tokens; maximum is {}",
                tokens.len(),
                QwenImage21TextEncoder::MAX_SEQUENCE_LENGTH
            ));
        }
        Ok(tokens)
    }

    fn conditioned_prompt_tokens(
        &self,
        prompt: &str,
        grid: (i32, i32, i32),
    ) -> Result<(Vec<u32>, Vec<bool>)> {
        let prompt = if prompt.trim().is_empty() {
            " "
        } else {
            prompt
        };
        let image_slots = usize::try_from(grid.0 * grid.1 * grid.2 / 4)?;
        let image_placeholders = "<|image_pad|>".repeat(image_slots);
        let formatted = format!(
            "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n<image1><|vision_start|>{image_placeholders}<|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n"
        );
        let tokens = self.tokenizer.encode(&formatted, false)?;
        if tokens.len() > QwenImage21TextEncoder::MAX_SEQUENCE_LENGTH {
            return Err(anyhow!(
                "Qwen Image 2.1 conditioned prompt expands to {} tokens; maximum is {}",
                tokens.len(),
                QwenImage21TextEncoder::MAX_SEQUENCE_LENGTH
            ));
        }
        let image_mask = tokens
            .iter()
            .map(|token| *token == self.image_token_id)
            .collect::<Vec<_>>();
        let actual_slots = image_mask.iter().filter(|is_image| **is_image).count();
        if actual_slots != image_slots {
            return Err(anyhow!(
                "Qwen Image 2.1 tokenizer produced {actual_slots} image slots; expected {image_slots}"
            ));
        }
        Ok((tokens, image_mask))
    }

    pub fn generate(
        &self,
        prompt: &str,
        config: QwenImage21GenerationConfig,
        should_cancel: impl Fn() -> bool,
    ) -> Result<Array> {
        let config = config.validate()?;
        if should_cancel() {
            return Err(anyhow!("Qwen Image 2.1 generation cancelled"));
        }
        let tokens = self.prompt_tokens(prompt)?;
        let hidden = self.text_encoder.forward(&tokens)?;
        let hidden_shape = hidden.shape();
        let hidden_dims = hidden_shape.as_slice();
        if self.system_prefix_tokens >= hidden_dims[1] as usize {
            return Err(anyhow!(
                "Qwen Image 2.1 system prefix consumes the entire prompt"
            ));
        }
        let encoder_hidden = mlx::ops::indexing::slice(
            &hidden,
            &[0_i32, self.system_prefix_tokens as i32, 0][..],
            &[hidden_dims[0], hidden_dims[1], hidden_dims[2]][..],
        )?;
        mlx::transforms::eval(&[&encoder_hidden])?;

        let latent_height = i32::try_from(config.height / 16)?;
        let latent_width = i32::try_from(config.width / 16)?;
        let image_tokens = latent_height * latent_width;
        let key = mlx::random::key(config.seed)?;
        let mut latents = mlx::random::normal()
            .shape((1_i32, image_tokens, 64_i32))
            .dtype(Dtype::Bfloat16)
            .key(&key)
            .sample()?;
        let schedule =
            FlowMatchSchedule::qwen_image_21(image_tokens as usize, config.inference_steps)?;

        for step in 0..config.inference_steps {
            if should_cancel() {
                return Err(anyhow!(
                    "Qwen Image 2.1 generation cancelled at step {step}"
                ));
            }
            let noise = self.transformer.forward(
                &latents,
                &encoder_hidden,
                schedule.sigmas[step],
                latent_height,
                latent_width,
            )?;
            let delta: Array = (&[schedule.delta(step)?][..], ()).try_into()?;
            latents = (&latents.astype(Dtype::Float32)?
                + &(&noise.astype(Dtype::Float32)? * &delta))
                .astype(Dtype::Bfloat16)?;
            mlx::transforms::eval(&[&latents])?;
        }

        let latents = latents
            .reshape((1_i32, latent_height, latent_width, 64_i32))?
            .transpose_axes(&[0_i32, 3, 1, 2][..])?;
        let image = self.vae.decode(&latents)?;
        mlx::transforms::eval(&[&image])?;
        Ok(image)
    }

    pub fn generate_conditioned(
        &self,
        prompt: &str,
        condition_bytes: &[u8],
        config: QwenImage21GenerationConfig,
        should_cancel: impl Fn() -> bool,
    ) -> Result<Array> {
        let config = config.validate()?;
        if should_cancel() {
            return Err(anyhow!("Qwen Image 2.1 generation cancelled"));
        }
        let condition = QwenImage21Condition::from_bytes(condition_bytes)?;
        let (tokens, image_mask) = self.conditioned_prompt_tokens(prompt, condition.grid)?;
        let hidden = self.text_encoder.forward_conditioned(
            &tokens,
            &condition.vision_pixels,
            condition.grid,
            i32::try_from(self.image_token_id)?,
        )?;
        let hidden_shape = hidden.shape();
        let hidden_dims = hidden_shape.as_slice();
        if self.system_prefix_tokens >= hidden_dims[1] as usize {
            return Err(anyhow!(
                "Qwen Image 2.1 system prefix consumes the entire prompt"
            ));
        }
        let encoder_hidden = mlx::ops::indexing::slice(
            &hidden,
            &[0_i32, self.system_prefix_tokens as i32, 0][..],
            &[hidden_dims[0], hidden_dims[1], hidden_dims[2]][..],
        )?;
        let encoder_image_mask = &image_mask[self.system_prefix_tokens..];
        mlx::transforms::eval(&[&encoder_hidden])?;

        let condition_latents = self
            .vae
            .encode_condition(&condition.vae_pixels.astype(Dtype::Bfloat16)?)?;
        let condition_dims = condition_latents.shape();
        let condition_dims = condition_dims.as_slice();
        if condition_dims != [1, 64, condition.latent_height, condition.latent_width] {
            return Err(anyhow!(
                "Qwen Image 2.1 VAE condition latent geometry mismatch: {:?}",
                condition_dims
            ));
        }
        let condition_latents = condition_latents
            .reshape((
                1_i32,
                64_i32,
                condition.latent_height * condition.latent_width,
            ))?
            .transpose_axes(&[0_i32, 2, 1][..])?;
        mlx::transforms::eval(&[&condition_latents])?;

        let latent_height = i32::try_from(config.height / 16)?;
        let latent_width = i32::try_from(config.width / 16)?;
        let image_tokens = latent_height * latent_width;
        let key = mlx::random::key(config.seed)?;
        let mut latents = mlx::random::normal()
            .shape((1_i32, image_tokens, 64_i32))
            .dtype(Dtype::Bfloat16)
            .key(&key)
            .sample()?;
        let schedule =
            FlowMatchSchedule::qwen_image_21(image_tokens as usize, config.inference_steps)?;

        for step in 0..config.inference_steps {
            if should_cancel() {
                return Err(anyhow!(
                    "Qwen Image 2.1 generation cancelled at step {step}"
                ));
            }
            let noise = self.transformer.forward_conditioned(
                &latents,
                &condition_latents,
                &encoder_hidden,
                encoder_image_mask,
                schedule.sigmas[step],
                condition.latent_height,
                condition.latent_width,
                latent_height,
                latent_width,
            )?;
            let delta: Array = (&[schedule.delta(step)?][..], ()).try_into()?;
            latents = (&latents.astype(Dtype::Float32)?
                + &(&noise.astype(Dtype::Float32)? * &delta))
                .astype(Dtype::Bfloat16)?;
            mlx::transforms::eval(&[&latents])?;
        }

        let latents = latents
            .reshape((1_i32, latent_height, latent_width, 64_i32))?
            .transpose_axes(&[0_i32, 3, 1, 2][..])?;
        let image = self.vae.decode(&latents)?;
        mlx::transforms::eval(&[&image])?;
        Ok(image)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_config_enforces_service_geometry_contract() {
        assert!(QwenImage21GenerationConfig::default().validate().is_ok());
        assert!(QwenImage21GenerationConfig {
            width: 1025,
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(QwenImage21GenerationConfig {
            width: 2048,
            height: 2048,
            ..Default::default()
        }
        .validate()
        .is_err());
    }
}
