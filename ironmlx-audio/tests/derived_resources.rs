use ironmlx_audio::{
    indextts25::IndexTts25ReferenceEncoder,
    resources::derived::{self, AuxiliaryComponent},
};
use std::{fs, path::Path};

#[test]
fn missing_reference_resources_report_all_components() {
    let root = tempfile::tempdir().unwrap();
    let report = IndexTts25ReferenceEncoder::inspect(root.path(), root.path());
    assert!(!report.complete);
    for name in [
        "manifest",
        "auxiliary.safetensors",
        "campplus.safetensors",
        "model.safetensors",
        "gpt.safetensors",
        "s2mel.safetensors",
        "config.json",
    ] {
        assert!(
            report.issues.iter().any(|issue| issue.component == name),
            "{name}"
        );
    }
    assert!(derived::load_auxiliary(root.path(), AuxiliaryComponent::Matrices).is_err());
}

fn copy_directory(source: &Path, destination: &Path) {
    fs::create_dir_all(destination).unwrap();
    for entry in fs::read_dir(source).unwrap() {
        let entry = entry.unwrap();
        let target = destination.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_directory(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), target).unwrap();
        }
    }
}

#[test]
#[ignore = "requires IRONMLX_INDEXTTS25_DERIVED; mutates only a temporary copy"]
fn real_derived_resources_reject_tampering() {
    let original = std::env::var("IRONMLX_INDEXTTS25_DERIVED").unwrap();
    assert!(derived::inspect_derived(Path::new(&original)).complete);
    let root = tempfile::tempdir().unwrap();
    copy_directory(Path::new(&original), root.path());
    let manifest_path = root.path().join("manifest.json");
    let original_manifest = fs::read(&manifest_path).unwrap();
    let mut manifest: serde_json::Value = serde_json::from_slice(&original_manifest).unwrap();
    manifest["allowed_unused"] = serde_json::json!({});
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(!derived::inspect_derived(root.path()).complete);
    fs::write(&manifest_path, &original_manifest).unwrap();
    let auxiliary = root.path().join("auxiliary.safetensors");
    let mut bytes = fs::read(&auxiliary).unwrap();
    *bytes.last_mut().unwrap() ^= 1;
    fs::write(&auxiliary, &bytes).unwrap();
    // Updating a self-declared manifest hash must not authorize altered weights.
    use sha2::{Digest, Sha256};
    let mut manifest: serde_json::Value = serde_json::from_slice(&original_manifest).unwrap();
    manifest["components"]["auxiliary.safetensors"]["sha256"] =
        format!("{:x}", Sha256::digest(&bytes)).into();
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(derived::load_auxiliary(root.path(), AuxiliaryComponent::Matrices).is_err());
    fs::remove_file(root.path().join("campplus.safetensors")).unwrap();
    let report = derived::inspect_derived(root.path());
    assert!(report
        .issues
        .iter()
        .any(|issue| issue.component == "campplus.safetensors"));
    assert!(!report.complete);
    assert!(derived::inspect_derived(Path::new(&original)).complete);
}
