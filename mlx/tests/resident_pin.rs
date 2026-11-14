//! `pin_resident` keeps exactly the buffers it is given resident, and the pin
//! ends when the buffer is freed. One test: the counters are process-wide.

use mlx::{Array, Dtype};

#[test]
fn pin_follows_the_buffer_it_was_given() {
    const BYTES: usize = 8 << 20;
    let weights = Array::zeros(&[2 * 1024, 1024], Dtype::Float32).expect("zeros");
    weights.eval().expect("eval");
    let lazy = &weights + &weights;
    mlx::transforms::synchronize().expect("synchronize");
    let before = mlx::memory::pinned_bytes();

    // Not evaluated: no buffer to pin.
    assert_eq!(mlx::memory::pin_resident(&[&lazy]).expect("pin"), 0);
    let pinned = mlx::memory::pin_resident(&[&weights, &weights]).expect("pin");
    if pinned == 0 {
        // No Metal residency sets on this machine (older macOS or GPU).
        return;
    }
    assert!(pinned >= BYTES, "{pinned}");
    assert_eq!(mlx::memory::pinned_bytes(), before + pinned);
    // A clone shares the buffer, which is pinned already.
    assert_eq!(
        mlx::memory::pin_resident(&[&weights.clone()]).expect("pin"),
        0
    );

    assert_eq!(
        mlx::memory::unpin_resident(&[&weights]).expect("unpin"),
        pinned
    );
    assert_eq!(mlx::memory::pinned_bytes(), before);
    assert_eq!(mlx::memory::pin_resident(&[&weights]).expect("pin"), pinned);

    // Freed: the pin goes with it, whether the buffer is released or cached.
    // (`lazy` holds the buffer too, as an input of its graph.)
    drop(lazy);
    drop(weights);
    mlx::transforms::synchronize().expect("synchronize");
    assert_eq!(mlx::memory::pinned_bytes(), before);
}
