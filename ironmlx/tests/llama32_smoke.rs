//! Llama 3.2 1B Instruct numeric correctness smoke test.
//!
//! Reference: `mlx_lm 0.31.3` / `mlx 0.31.2`, checkpoint
//! `mlx-community/Llama-3.2-1B-Instruct-4bit` at revision
//! `08231374eeacb049a0eade7922910865b8fce912`. The raw prompt
//! `The capital of France is` encodes without BOS as
//! `[791, 6864, 315, 9822, 374]`; the next-token top-5 is
//! `[12366, 3146, 1131, 25, 539]`, with token 12366 decoding to ` Paris`.
//! This pins the Llama 3 frequency scaling and tied embedding projection.

use ironmlx::core::{Loader, Model};
use ironmlx::models::LlamaModel;
use ironmlx::Tokenizer;
use mlx::{Array, Dtype, StreamOrDevice};

const PROMPT: &str = "The capital of France is";
const PROMPT_IDS: &[i32] = &[791, 6864, 315, 9822, 374];
const REF_TOP5: &[usize] = &[12366, 3146, 1131, 25, 539];
const REF_TOP1_LOGIT: f32 = 19.296875;

#[test]
fn llama32_first_token_matches_mlx_lm_reference() {
    let Ok(model_dir) = std::env::var("LLAMA32_MODEL") else {
        eprintln!("skip: set LLAMA32_MODEL to a Llama 3.1/3.2 Instruct checkpoint");
        return;
    };
    let loader = Loader::open(std::path::Path::new(&model_dir)).expect("Loader::open");
    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");
    let model = LlamaModel::from_loader(&loader).expect("LlamaModel::from_loader");
    let scaling = model
        .config()
        .rope_scaling
        .as_ref()
        .expect("Llama 3 rope scaling");
    assert_eq!(scaling.rope_type, "llama3");
    assert!(model.config().tie_word_embeddings);

    let prompt_ids = tokenizer.encode(PROMPT, false).expect("encode prompt");
    assert_eq!(
        prompt_ids
            .iter()
            .map(|token| *token as i32)
            .collect::<Vec<_>>(),
        PROMPT_IDS
    );

    let input: Array = (PROMPT_IDS, &[1, PROMPT_IDS.len() as i32][..])
        .try_into()
        .unwrap();
    let position: Array = (&[0_i32][..], &[1][..]).try_into().unwrap();
    let mut cache = Model::make_cache(&model, 1, 64, Dtype::Bfloat16).expect("make cache");
    let logits = Model::forward_on(
        &model,
        &input,
        &position,
        None,
        None,
        Some(&mut cache),
        StreamOrDevice::default(),
    )
    .expect("Llama 3.2 forward");
    assert_eq!(
        logits.shape().as_slice(),
        &[1, 1, model.config().vocab_size]
    );
    let values: Vec<f32> = mlx::ops::cast::astype(&logits, Dtype::Float32)
        .unwrap()
        .to_vec()
        .unwrap();
    let mut indexes = (0..values.len()).collect::<Vec<_>>();
    indexes.sort_unstable_by(|left, right| {
        values[*right]
            .partial_cmp(&values[*left])
            .expect("finite logits")
    });
    assert_eq!(&indexes[..5], REF_TOP5);
    assert!(
        (values[indexes[0]] - REF_TOP1_LOGIT).abs() <= 0.5,
        "top-1 logit {} diverges from reference {REF_TOP1_LOGIT}",
        values[indexes[0]]
    );
}
