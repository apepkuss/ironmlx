use ironmlx_audio::{
    io::NativeAudioIo, signal::resample_mono, AudioIo, DecodeLimits, Result, SessionControl,
};
struct Active;
impl SessionControl for Active {
    fn check(&self) -> Result<()> {
        Ok(())
    }
}
#[test]
fn resampling_matches_torchaudio() {
    let fixtures: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/resample.json")).unwrap();
    for case in fixtures["cases"].as_array().unwrap() {
        let source = case["source_rate"].as_u64().unwrap() as u32;
        let target = case["target_rate"].as_u64().unwrap() as u32;
        let input: Vec<f32> = serde_json::from_value(case["input"].clone()).unwrap();
        let expected: Vec<f32> = serde_json::from_value(case["output"].clone()).unwrap();
        let actual = resample_mono(&input, source, target, &Active).unwrap();
        assert_eq!(actual.len(), expected.len(), "{source}->{target}");
        let error = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f32::max);
        assert!(error < 2e-5, "{source}->{target}: max error {error}");
    }
}
#[test]
fn actual_wav_flac_mp3_decoders() {
    let files: [&[u8]; 6] = [
        include_bytes!("fixtures/reference-PCM_16.wav"),
        include_bytes!("fixtures/reference-PCM_24.wav"),
        include_bytes!("fixtures/reference-PCM_32.wav"),
        include_bytes!("fixtures/reference-FLOAT.wav"),
        include_bytes!("fixtures/reference.flac"),
        include_bytes!("fixtures/reference.mp3"),
    ];
    for (i, bytes) in files.iter().enumerate() {
        let audio = NativeAudioIo
            .decode(bytes, &DecodeLimits::default())
            .unwrap();
        assert_eq!(audio.format.sample_rate, 8000);
        assert_eq!(audio.format.channels, 1);
        assert_eq!(audio.samples.len(), 8000, "format {i}");
        let error = audio
            .samples
            .iter()
            .enumerate()
            .map(|(n, x)| {
                let reference = (12000. * (n as f64 * 0.1).sin()) as i16 as f32 / 32768.;
                (x - reference).abs()
            })
            .fold(0., f32::max);
        assert!(
            error < if i == 5 { 0.08 } else { 1e-7 },
            "format {i}: {error}"
        );
    }
    let mp3 = files[5];
    assert!(NativeAudioIo
        .decode(&mp3[..mp3.len() - 10], &DecodeLimits::default())
        .is_err());
    let flac = files[4];
    assert!(NativeAudioIo
        .decode(&flac[..flac.len() - 100], &DecodeLimits::default())
        .is_err());
}
