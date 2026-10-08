//! Encode local text, image and audio inputs for numerical qualification against the upstream model.
use ironmlx_lm::models::embedding_gemma2::{
    EmbeddingContent, EmbeddingGemma2Model, EmbeddingSample,
};
fn main() -> anyhow::Result<()> {
    if let Ok(dir) = std::env::var("MLX_DIR") {
        mlx::metal::set_metallib_path(&format!("{dir}/lib/mlx.metallib"))?;
    }
    let device = mlx::Device::gpu(0);
    mlx::set_default_device(device);
    mlx::set_default_stream(mlx::new_stream(device)?);
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(
        args.len() == 3,
        "usage: embedding_gemma2 MODEL_DIR INPUT_JSON"
    );
    let values: Vec<serde_json::Value> = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let samples: Vec<EmbeddingSample> = values
        .into_iter()
        .map(|value| -> anyhow::Result<_> {
            if let Some(text) = value.as_str() {
                return Ok(EmbeddingSample {
                    content: vec![EmbeddingContent::Text(text.to_owned())],
                });
            }
            let mut content = Vec::new();
            for part in value["content"]
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("missing content"))?
            {
                if let Some(text) = part["text"].as_str() {
                    content.push(EmbeddingContent::Text(text.to_owned()));
                } else if let Some(path) = part["audio_path"].as_str() {
                    content.push(EmbeddingContent::Audio(
                        ironmlx_lm::core::audio_input::EmbeddingAudio::decode(&std::fs::read(
                            path,
                        )?)?,
                    ));
                } else {
                    content.push(EmbeddingContent::Image(std::fs::read(
                        part["image_path"]
                            .as_str()
                            .ok_or_else(|| anyhow::anyhow!("missing image path"))?,
                    )?));
                }
            }
            Ok(EmbeddingSample { content })
        })
        .collect::<anyhow::Result<_>>()?;
    let started = std::time::Instant::now();
    let model = EmbeddingGemma2Model::load(std::path::Path::new(&args[1]))?;
    let loaded_ms = started.elapsed().as_millis();
    let started = std::time::Instant::now();
    let output = model.encode_inputs(&samples, 768)?;
    println!(
        "{}",
        serde_json::json!({"embeddings": output.embeddings, "input_tokens": output.input_tokens, "weight_bytes": model.weight_bytes(), "load_ms": loaded_ms, "encode_ms": started.elapsed().as_millis()})
    );
    Ok(())
}
