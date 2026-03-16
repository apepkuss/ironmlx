use std::collections::HashMap;
use std::path::Path;

use crate::array::Array;
use crate::device::Device;
use crate::error::Result;
use crate::stream::Stream;

/// Load all safetensors files from a model directory into a single HashMap.
///
/// Always loads on the default CPU stream (safetensors data originates on CPU).
/// MLX will lazily transfer to GPU when weights are used in GPU operations.
pub fn load_model_weights(dir: &Path, _stream: &Stream) -> Result<HashMap<String, Array>> {
    let cpu_stream = Stream::default_stream(&Device::cpu());
    let mut all_weights = HashMap::new();

    // Find all .safetensors files
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| crate::error::Error::Mlx(format!("failed to read dir: {}", e)))?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .path()
                .extension()
                .is_some_and(|ext| ext == "safetensors")
        })
        .collect();
    files.sort_by_key(|e| e.file_name());

    if files.is_empty() {
        return Err(crate::error::Error::Mlx(
            "no .safetensors files found".into(),
        ));
    }

    for entry in &files {
        let path_str = entry.path().to_string_lossy().to_string();
        let (arrays_map, _metadata) = crate::io::load_safetensors(&path_str, &cpu_stream)?;
        for (key, array) in arrays_map.iter() {
            all_weights.insert(key, array);
        }
    }

    Ok(all_weights)
}
