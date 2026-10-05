//! Composed gate/up + down production FFI and realized-layout fallback.
use half::bf16;
use mlx::{io, ops, quantization, random, Array, Device, Dtype};

fn compare(x: &Array, packed: &[Array], label: &str) {
    let candidate = quantization::quantized_matmul(
        x,
        &packed[0],
        &packed[1],
        Some(&packed[2]),
        true,
        Some(64),
        Some(4),
        "affine",
    )
    .unwrap();
    let native = quantization::quantized_matmul_batch_isolated(
        x,
        &packed[0],
        &packed[1],
        Some(&packed[2]),
        true,
        Some(64),
        Some(4),
        "affine",
    )
    .unwrap();
    assert_eq!(candidate.shape(), native.shape());
    let a = candidate.to_vec::<bf16>().unwrap();
    let b = native.to_vec::<bf16>().unwrap();
    for (index, (a, b)) in a.iter().zip(&b).enumerate() {
        assert!(
            a.is_finite() && b.is_finite(),
            "{label}: nonfinite at {index}"
        );
        assert_eq!(a.to_bits(), b.to_bits(), "{label}: mismatch at {index}");
    }
    eprintln!(
        "QMM down-extension FFI {label}: byte parity {} values",
        a.len()
    );
}

#[test]
#[ignore = "requires pinned private QMM metallib, real checkpoint shard and M5 GPU"]
fn real_gate_up_and_down_ffi_and_fallback_are_byte_equal() {
    assert!(std::env::var("IRONMLX_EXPERIMENTAL_PREFILL_QMM_MTILE_METALLIB").is_ok());
    mlx::metal::set_metallib_path(&std::env::var("IRONMLX_TEST_METALLIB").unwrap()).unwrap();
    mlx::set_default_device(Device::gpu(0));
    let (mut weights, _) =
        io::load_safetensors(&std::env::var("IRONMLX_TEST_QMM_SHARD").unwrap()).unwrap();
    let mut projections = Vec::new();
    for (label, names, k) in [
        ("gate-up", vec!["gate_proj", "up_proj"], 5120),
        ("down", vec!["down_proj"], 17408),
    ] {
        let mut packed = Vec::new();
        for suffix in ["weight", "scales", "biases"] {
            let values: Vec<_> = names
                .iter()
                .map(|name| {
                    weights
                        .remove(&format!(
                            "language_model.model.layers.0.mlp.{name}.{suffix}"
                        ))
                        .unwrap()
                })
                .collect();
            let value = if values.len() == 1 {
                values.into_iter().next().unwrap()
            } else {
                ops::concatenate(&values.iter().collect::<Vec<_>>(), 0).unwrap()
            };
            packed.push(value);
        }
        projections.push((label, k, packed));
    }
    drop(weights);
    for (label, k, packed) in projections {
        random::seed(20261001);
        let x = random::normal()
            .shape((2048, k))
            .dtype(Dtype::Bfloat16)
            .sample()
            .unwrap();
        compare(&x, &packed, &format!("{label}-eligible-rank2-lazy"));
        let x3 = x.reshape((1, 2048, k)).unwrap();
        compare(&x3, &packed, &format!("{label}-eligible-rank3"));
        compare(
            &x3.slice((0, 0, 0), (1, 17, k)).unwrap(),
            &packed,
            &format!("{label}-verify17-fallback"),
        );
        compare(
            &x3.slice((0, 0, 0), (1, 2047, k)).unwrap(),
            &packed,
            &format!("{label}-tail2047-fallback"),
        );
        let pitched = random::normal()
            .shape((1, k, 2048))
            .dtype(Dtype::Bfloat16)
            .sample()
            .unwrap()
            .transpose_axes(&[0, 2, 1][..])
            .unwrap();
        compare(
            &pitched,
            &packed,
            &format!("{label}-noncontiguous-fallback"),
        );
    }
}
