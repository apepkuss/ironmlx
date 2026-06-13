//! Fast Walsh-Hadamard Transform (WHT) for TurboQuant rotation.
//!
//! The rotation is: y = WHT(signs * x), where signs are in {+1, -1}^d.
//! WHT is self-inverse: WHT(WHT(x)) = x, so the inverse rotation is the same
//! operation: x = WHT(signs * y).
//!
//! WHT uses the butterfly decomposition: O(d log d) additions/subtractions.
//! The transform is normalized by 1/sqrt(d).

/// Generate deterministic sign pattern for WHT rotation.
///
/// Uses a simple hash (golden ratio based) for reproducibility.
/// The same seed always produces the same signs, ensuring quantize/dequantize consistency.
pub fn generate_signs(d: usize, seed: u64) -> Vec<f32> {
    let mut signs = Vec::with_capacity(d);
    let golden = 0x9E3779B97F4A7C15u64; // golden ratio hash constant
    for i in 0..d {
        let hash = seed.wrapping_mul(golden).wrapping_add(i as u64);
        let hash = hash.wrapping_mul(golden);
        if hash & 1 == 0 {
            signs.push(1.0);
        } else {
            signs.push(-1.0);
        }
    }
    signs
}

/// In-place Fast Walsh-Hadamard Transform, normalized by 1/sqrt(n).
///
/// `x` must have length that is a power of 2.
/// Performs the butterfly decomposition in O(n log n).
pub fn wht_inplace(x: &mut [f32]) {
    let n = x.len();
    debug_assert!(
        n.is_power_of_two(),
        "WHT requires power-of-2 length, got {n}"
    );

    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let a = x[j];
                let b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
            i += h * 2;
        }
        h *= 2;
    }

    // Normalize
    let scale = 1.0 / (n as f32).sqrt();
    for v in x.iter_mut() {
        *v *= scale;
    }
}

/// Apply forward rotation: y = WHT(signs * x).
///
/// `x` is the input vector of length `d` (must be power of 2).
/// `signs` is the sign pattern of length `d`.
/// Returns the rotated vector.
pub fn rotate_forward(x: &[f32], signs: &[f32]) -> Vec<f32> {
    debug_assert_eq!(x.len(), signs.len());
    let mut y: Vec<f32> = x.iter().zip(signs).map(|(&xi, &si)| xi * si).collect();
    wht_inplace(&mut y);
    y
}

/// Apply inverse rotation: x = signs * WHT(y).
///
/// Forward: y = WHT(signs * x) = WHT @ diag(signs) @ x
/// Inverse: x = diag(signs) @ WHT @ y = signs * WHT(y)
/// (since WHT^-1 = WHT and diag(signs)^-1 = diag(signs))
pub fn rotate_inverse(y: &[f32], signs: &[f32]) -> Vec<f32> {
    debug_assert_eq!(y.len(), signs.len());
    let mut result = y.to_vec();
    wht_inplace(&mut result);
    for (v, &s) in result.iter_mut().zip(signs) {
        *v *= s;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wht_self_inverse() {
        let n = 128;
        let original: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut x = original.clone();
        wht_inplace(&mut x);
        wht_inplace(&mut x);
        for (a, b) in original.iter().zip(x.iter()) {
            assert!((a - b).abs() < 1e-4, "WHT not self-inverse: {a} vs {b}");
        }
    }

    #[test]
    fn rotation_roundtrip() {
        let d = 128;
        let signs = generate_signs(d, 42);
        let x: Vec<f32> = (0..d).map(|i| (i as f32 * 0.07).cos()).collect();

        let y = rotate_forward(&x, &signs);
        let x_recovered = rotate_inverse(&y, &signs);

        for (a, b) in x.iter().zip(x_recovered.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "rotation roundtrip failed: {a} vs {b}"
            );
        }
    }

    #[test]
    fn rotation_preserves_norm() {
        let d = 128;
        let signs = generate_signs(d, 42);
        let x: Vec<f32> = (0..d).map(|i| (i as f32 * 0.03).sin()).collect();

        let norm_before: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let y = rotate_forward(&x, &signs);
        let norm_after: f32 = y.iter().map(|v| v * v).sum::<f32>().sqrt();

        assert!(
            (norm_before - norm_after).abs() / norm_before < 1e-4,
            "norm not preserved: {norm_before} vs {norm_after}"
        );
    }

    #[test]
    fn signs_deterministic() {
        let s1 = generate_signs(128, 42);
        let s2 = generate_signs(128, 42);
        assert_eq!(s1, s2);

        let s3 = generate_signs(128, 99);
        assert_ne!(s1, s3);
    }

    #[test]
    fn signs_balanced() {
        let signs = generate_signs(1024, 42);
        let pos: usize = signs.iter().filter(|&&s| s > 0.0).count();
        let neg = signs.len() - pos;
        // Should be roughly balanced (within 10%)
        let ratio = pos as f32 / neg as f32;
        assert!(
            (0.8..1.2).contains(&ratio),
            "signs not balanced: {pos} pos, {neg} neg"
        );
    }

    #[test]
    fn wht_different_sizes() {
        for n in [64, 128, 256] {
            let original: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
            let mut x = original.clone();
            wht_inplace(&mut x);
            wht_inplace(&mut x);
            for (a, b) in original.iter().zip(x.iter()) {
                assert!(
                    (a - b).abs() < 1e-4,
                    "WHT roundtrip failed for n={n}: {a} vs {b}"
                );
            }
        }
    }
}
