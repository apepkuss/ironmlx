use clap::Args;
use ironmlx_audio::{
    diarization::{Latency, NemotronDiarization, Options},
    AudioIo, DecodeLimits, SessionControl,
};
use std::path::PathBuf;
#[derive(Args, Debug)]
pub struct DiarizeArgs {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    audio: PathBuf,
    #[arg(long, default_value_t = 0.5)]
    threshold: f32,
    #[arg(long, default_value_t = 0.)]
    min_duration: f64,
    #[arg(long, default_value_t = 0.)]
    merge_gap: f64,
    #[arg(long,default_value="offline",value_parser=["offline","low","very_low","ultra_low"])]
    latency: String,
    #[arg(long,default_value="json",value_parser=["json","rttm"])]
    response_format: String,
}
struct Control;
impl SessionControl for Control {
    fn check(&self) -> ironmlx_audio::Result<()> {
        Ok(())
    }
}
pub fn run(args: DiarizeArgs) -> crate::Result<()> {
    let bytes = std::fs::read(&args.audio)?;
    let pcm = ironmlx_audio::io::NativeAudioIo.decode(
        &bytes,
        &DecodeLimits {
            max_duration_seconds: 7200,
            max_encoded_bytes: 128 * 1024 * 1024,
            max_pcm_bytes: 512 * 1024 * 1024,
            ..DecodeLimits::default()
        },
    )?;
    let options = Options {
        threshold: args.threshold,
        min_duration: args.min_duration,
        merge_gap: args.merge_gap,
        latency: serde_json::from_value::<Latency>(args.latency.into())?,
    };
    let model = NemotronDiarization::load(&args.model)?;
    let output = model.diarize(&pcm, options, &Control, &mut |_, _| Ok(()))?;
    if args.response_format == "rttm" {
        print!("{}", output.rttm());
    } else {
        println!("{}", serde_json::to_string_pretty(&output)?);
    }
    Ok(())
}
