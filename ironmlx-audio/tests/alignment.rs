use ironmlx_audio::{qwen3_asr::Qwen3Asr, AudioIo, SessionControl};
struct Control;
impl SessionControl for Control {
    fn check(&self) -> ironmlx_audio::Result<()> {
        Ok(())
    }
}
#[test]
#[ignore = "requires local ForcedAligner checkpoint and reference audio"]
fn qwen3_forced_aligner_real_timestamps() -> anyhow::Result<()> {
    let root = std::env::var("MLX_DIR")?;
    mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
    let model = Qwen3Asr::load(std::path::Path::new(&std::env::var("IRONMLX_ALIGNER_DIR")?))?;
    let pcm = ironmlx_audio::io::NativeAudioIo.decode(
        &std::fs::read(std::env::var("IRONMLX_ASR_TEST_AUDIO")?)?,
        &Default::default(),
    )?;
    let result = model.align(&pcm, "甚至出现交易几乎停滞的情况。", "zh", &Control)?;
    assert_eq!(result.words.len(), 13);
    assert!(result.words.iter().any(|v| v.end > v.start));
    assert!(result.words.windows(2).all(|v| v[0].end <= v[1].start));
    assert!(result.words.last().unwrap().end <= result.duration + 0.001);
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}
