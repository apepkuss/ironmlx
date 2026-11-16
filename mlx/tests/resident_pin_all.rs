//! `pin_all_resident` pins every buffer in use, not the freed ones in the
//! allocator cache, and each pin still ends with its buffer. One test: the
//! counters are process-wide.

use mlx::{Array, Dtype};

#[test]
fn pin_all_covers_the_buffers_in_use_only() {
    const BYTES: usize = 8 << 20;
    let weights = Array::zeros(&[2 * 1024, 1024], Dtype::Float32).expect("zeros");
    weights.eval().expect("eval");
    // A buffer freed into the allocator cache.
    let freed = Array::zeros(&[4 * 1024, 1024], Dtype::Float32).expect("zeros");
    freed.eval().expect("eval");
    drop(freed);
    mlx::transforms::synchronize().expect("synchronize");
    assert!(mlx::memory::snapshot().cache_bytes >= 2 * BYTES);
    let before = mlx::memory::pinned_bytes();

    let pinned = mlx::memory::pin_all_resident().expect("pin all");
    if pinned == 0 {
        // No Metal residency sets on this machine (older macOS or GPU).
        return;
    }
    assert!(pinned >= BYTES, "{pinned}");
    assert_eq!(mlx::memory::pinned_bytes(), before + pinned);
    // The cache was released, not pinned.
    assert_eq!(mlx::memory::snapshot().cache_bytes, 0);
    // The weights are among the pinned buffers.
    assert_eq!(mlx::memory::pin_resident(&[&weights]).expect("pin"), 0);
    assert_eq!(mlx::memory::pin_all_resident().expect("pin all"), 0);

    // A later allocation is not pinned.
    let later = Array::zeros(&[2 * 1024, 1024], Dtype::Float32).expect("zeros");
    later.eval().expect("eval");
    assert_eq!(mlx::memory::pinned_bytes(), before + pinned);

    // Freed: the pin goes with it.
    drop(weights);
    mlx::transforms::synchronize().expect("synchronize");
    assert_eq!(mlx::memory::pinned_bytes(), before + pinned - BYTES);
    drop(later);
}
