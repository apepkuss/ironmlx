use ironmlx_audio::resources::{
    inspect_component, inspect_components, load_component, ComponentSpec, IndexTts25Component,
    TensorSpec,
};
use ironmlx_core::weights::WeightSource;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
fn fixture(path: &std::path::Path) -> ComponentSpec {
    let mut values = HashMap::new();
    values.insert(
        "proj.weight".into(),
        mlx::Array::try_from((&[1f32, 2., 3., 4.][..], &[2, 2][..])).unwrap(),
    );
    mlx::io::save_safetensors(path.to_str().unwrap(), &values, &HashMap::new()).unwrap();
    let bytes = std::fs::read(path).unwrap();
    ComponentSpec {
        name: "test".into(),
        bytes: bytes.len() as u64,
        sha256: format!("{:x}", Sha256::digest(&bytes)),
        tensors: BTreeMap::from([(
            "proj.weight".into(),
            TensorSpec {
                dtype: "F32".into(),
                shape: vec![2, 2],
            },
        )]),
    }
}
#[test]
fn strict_loading_rejects_hash_shape_missing_and_nonfinite() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("weights.safetensors");
    let mut spec = fixture(&path);
    assert_eq!(inspect_component(&path, &spec).unwrap(), 16);
    let weights = load_component(&path, &spec).unwrap();
    assert_eq!(
        weights
            .tensor("proj.weight")
            .unwrap()
            .to_vec::<f32>()
            .unwrap(),
        [1., 2., 3., 4.]
    );
    assert!(weights.tensor("absent").is_err());
    spec.tensors.get_mut("proj.weight").unwrap().shape = vec![4];
    assert!(inspect_component(&path, &spec).is_err());
    spec.tensors.get_mut("proj.weight").unwrap().shape = vec![2, 2];
    let mut bytes = std::fs::read(&path).unwrap();
    let len = bytes.len();
    bytes[len - 4..].copy_from_slice(&f32::NAN.to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();
    assert!(inspect_component(&path, &spec).is_err());
    spec.sha256 = format!("{:x}", Sha256::digest(&bytes));
    assert!(load_component(&path, &spec)
        .err()
        .unwrap()
        .to_string()
        .contains("non-finite"));
    let report = inspect_components(&[
        (dir.path().join("missing1"), spec.clone()),
        (dir.path().join("missing2"), spec),
    ]);
    assert!(!report.complete);
    assert_eq!(report.issues.len(), 2);
}
#[test]
fn all_pinned_component_schemas_and_capacities() {
    for (component, count) in [
        (IndexTts25Component::Gpt, 457),
        (IndexTts25Component::Codec, 241),
        (IndexTts25Component::S2Mel, 264),
        (IndexTts25Component::BigVgan, 449),
        (IndexTts25Component::W2vBert, 773),
    ] {
        let spec = component.spec().unwrap();
        assert_eq!(spec.tensors.len(), count);
    }
    let gpt = IndexTts25Component::Gpt.spec().unwrap();
    assert_eq!(gpt.tensors["text_embedding.weight"].shape, [60510, 1280]);
    assert_eq!(
        gpt.tensors["text_pos_embedding.emb.weight"].shape,
        [602, 1280]
    );
    assert_eq!(
        gpt.tensors["mel_pos_embedding.emb.weight"].shape,
        [1818, 1280]
    );
}
/// Exercises all published weights through the native loader, one component at a time.
#[test]
#[ignore = "requires IRONMLX_INDEXTTS25_SNAPSHOT and Apple GPU memory for the largest component"]
fn real_component_loading() {
    let root =
        std::path::PathBuf::from(std::env::var("IRONMLX_INDEXTTS25_SNAPSHOT").expect("snapshot"));
    for component in [
        IndexTts25Component::Codec,
        IndexTts25Component::S2Mel,
        IndexTts25Component::BigVgan,
        IndexTts25Component::Gpt,
        IndexTts25Component::W2vBert,
    ] {
        let spec = component.spec().unwrap();
        let loaded = load_component(&root.join(component.file_name()), &spec).unwrap();
        assert_eq!(loaded.tensors().len(), spec.tensors.len());
        for (name, expected) in &spec.tensors {
            assert_eq!(
                loaded.tensor(name).unwrap().shape().as_slice(),
                expected.shape.iter().map(|n| *n as i32).collect::<Vec<_>>()
            );
        }
        println!(
            "verified {} tensors in {}",
            spec.tensors.len(),
            component.file_name()
        );
        drop(loaded);
        mlx::clear_cache();
    }
}
