//! Metadata and bounded-I/O qualification, not Flash inference acceptance.
//! Only selected rows are materialized by MLX; the snapshot remains on SSD.
use ironmlx_lm::core::tensor_file::ModelTensorFiles;
use ironmlx_lm::models::qwen4_exp::{config::FlashTextConfig, ple_rows::PleRows};
use std::sync::Arc;

#[test]
#[ignore = "requires IRONMLX_FLASH_SNAPSHOT pointing to the imported checkpoint"]
fn local_flash_snapshot_metadata_and_bounded_reads() {
    let directory = std::env::var("IRONMLX_FLASH_SNAPSHOT").expect("explicit snapshot required");
    let model = Arc::new(ModelTensorFiles::open(std::path::Path::new(&directory)).unwrap());
    let mut total = 0u64;
    let mut ple = 0u64;
    let mut experts = 0u64;
    for (name, info) in model.tensors() {
        total += info.bytes();
        if name.contains(".ngram_embedding.shards.") {
            ple += info.bytes();
        }
        if name.contains(".mlp.switch_mlp.") {
            experts += info.bytes();
        }
    }
    eprintln!("snapshot={directory} tensors={} total_bytes={total} ple_table_bytes={ple} routed_expert_bytes={experts}", model.tensors().count());
    assert!(total > 100 * (1u64 << 30));
    assert!(ple > 29 * (1u64 << 30));
    assert!(experts > 70 * (1u64 << 30));
    let mut bytes_read = 0usize;
    for suffix in ["weight", "scales", "biases"] {
        let key = format!(
            "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.17.{suffix}"
        );
        let info = model.info(&key).unwrap();
        assert_eq!(info.shape.len(), 2);
        for row in [0, 42, info.shape[0] - 1] {
            let bytes = model.read_rows(&key, row, 1, 4096).unwrap();
            assert_eq!(bytes.len() as u64, info.bytes() / info.shape[0]);
            bytes_read += bytes.len();
        }
        let key = format!("language_model.model.layers.0.mlp.switch_mlp.gate_proj.{suffix}");
        let info = model.info(&key).unwrap();
        assert_eq!(info.shape[0], 512);
        assert_eq!(info.shape.len(), 3);
        for expert in [0, 511] {
            let bytes = model.read_rows(&key, expert, 1, 2 * 1024 * 1024).unwrap();
            assert_eq!(bytes.len() as u64, info.bytes() / 512);
            bytes_read += bytes.len();
        }
        assert!(model.read_rows(&key, 512, 1, 2 * 1024 * 1024).is_err());
    }
    eprintln!("bounded payload reads: {bytes_read} bytes");
    assert!(bytes_read < 4 * 1024 * 1024);

    // Compare the complete SSD -> typed MLX rows -> affine dequantization
    // path with independent scalar decoding of the original checkpoint bytes.
    let prefix = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.17";
    for row in [0, 42] {
        let key = |suffix| format!("{prefix}.{suffix}");
        let weight = model.read_rows_array(&key("weight"), row, 1, 4096).unwrap();
        let scale = model.read_rows_array(&key("scales"), row, 1, 4096).unwrap();
        let bias = model.read_rows_array(&key("biases"), row, 1, 4096).unwrap();
        let actual = mlx::quantization::dequantize(
            &weight,
            &scale,
            Some(&bias),
            Some(32),
            Some(4),
            "affine",
            None,
            None,
        )
        .unwrap()
        .astype(mlx::Dtype::Float32)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
        let raw = model.read_rows(&key("weight"), row, 1, 4096).unwrap();
        let packed: Vec<u32> = raw
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let bf16 = |suffix| {
            model
                .read_rows(&key(suffix), row, 1, 4096)
                .unwrap()
                .chunks_exact(2)
                .map(|b| f32::from_bits((u16::from_le_bytes(b.try_into().unwrap()) as u32) << 16))
                .collect::<Vec<_>>()
        };
        let scales = bf16("scales");
        let biases = bf16("biases");
        for (column, &actual) in actual.iter().enumerate() {
            let quant = (packed[column / 8] >> (4 * (column % 8))) & 15;
            let expected = (quant as f32).mul_add(scales[column / 32], biases[column / 32]);
            let bits = expected.to_bits();
            let expected =
                f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff0000);
            assert_eq!(actual, expected, "PLE row={row} column={column}");
        }
    }

    let config: serde_json::Value = serde_json::from_reader(
        std::fs::File::open(std::path::Path::new(&directory).join("config.json")).unwrap(),
    )
    .unwrap();
    let cfg = FlashTextConfig::from_root(&config).unwrap();
    let index = cfg.ple_index(0).unwrap();
    let rows = index
        .indices(&[42], &mut index.history())
        .unwrap()
        .remove(0);
    let width = cfg.ple_embed_dim as usize / index.heads();
    let mut provider = PleRows::new(
        Arc::clone(&model),
        "language_model.model.layers.1.ple.ple_embedding.ngram_embedding",
        index,
        width,
        2 * 1124,
        64 * 1024,
    )
    .unwrap();
    let selected = [rows[0], rows[15], rows[0]];
    let cached = provider.gather_on(&selected, ().into()).unwrap();
    assert_eq!(provider.resident_rows(), 2);
    assert_eq!(provider.charged_bytes(), 2248);
    assert_eq!((provider.hits, provider.misses), (1, 2));
    let cached = cached
        .astype(mlx::Dtype::Float32)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    for (position, row) in selected.into_iter().enumerate() {
        let (shard, local) = provider.index().shard_row(row).unwrap();
        let prefix = format!(
            "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.{shard}"
        );
        let read = |suffix| {
            model
                .read_rows_array(&format!("{prefix}.{suffix}"), local, 1, 4096)
                .unwrap()
        };
        let expected = mlx::quantization::dequantize(
            &read("weight"),
            &read("scales"),
            Some(&read("biases")),
            Some(32),
            Some(4),
            "affine",
            None,
            None,
        )
        .unwrap()
        .astype(mlx::Dtype::Float32)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
        assert_eq!(&cached[position * width..(position + 1) * width], expected);
    }
    provider.gather_on(&[rows[1]], ().into()).unwrap();
    assert_eq!(provider.resident_rows(), 2);
    assert!(provider.gather_on(&[u64::MAX], ().into()).is_err());
    assert_eq!((provider.hits, provider.misses), (1, 3));
}
