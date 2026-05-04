use mlx::Stream;

#[test]
fn default_stream_for_default_device() {
    let d = mlx::default_device();
    let s = mlx::default_stream(d);
    assert_eq!(s.device, d);
}

#[test]
fn new_stream_has_unique_index() {
    let d = mlx::default_device();
    let default_s = mlx::default_stream(d);
    let new_s = mlx::new_stream(d).expect("new_stream");
    assert_ne!(
        default_s.index, new_s.index,
        "new stream should have a fresh index"
    );
    assert_eq!(new_s.device, d);
}

#[test]
fn get_streams_includes_default() {
    let d = mlx::default_device();
    let default_s = mlx::default_stream(d);
    let all = mlx::get_streams();
    assert!(
        all.iter()
            .any(|s| s.index == default_s.index && s.device == d),
        "default stream should appear in get_streams()"
    );
}

#[test]
fn set_default_stream_round_trip() {
    let d = mlx::default_device();
    let original = mlx::default_stream(d);
    let new_s = mlx::new_stream(d).expect("new_stream");
    mlx::set_default_stream(new_s);
    assert_eq!(mlx::default_stream(d), new_s);
    // Restore.
    mlx::set_default_stream(original);
    assert_eq!(mlx::default_stream(d), original);
}

#[test]
fn stream_equality_and_copy() {
    let d = mlx::default_device();
    let a = mlx::default_stream(d);
    let b = a; // Copy
    assert_eq!(a, b);
    let other = mlx::new_stream(d).expect("new_stream");
    assert_ne!(a, other);
}

#[test]
fn clear_streams_then_default_still_works() {
    // clear_streams destroys all streams created on current thread.
    // The default stream is recreated lazily by MLX.
    mlx::clear_streams();
    let d = mlx::default_device();
    let _s: Stream = mlx::default_stream(d); // Must not panic.
}
