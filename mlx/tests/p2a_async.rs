use mlx::{transforms, Array, Dtype};

#[test]
fn async_eval_under_futures_lite() {
    // Verify the Future runs under a minimal executor (no tokio dep).
    let arr = Array::zeros(&[1024], Dtype::Float32).expect("zeros");
    futures_lite::future::block_on(arr.async_eval()).expect("async_eval should complete");
    // After eval, to_vec should not need to re-eval (data is materialized).
    let v: Vec<f32> = arr.to_vec().expect("to_vec");
    assert_eq!(v.len(), 1024);
    assert!(v.iter().all(|x| *x == 0.0));
}

#[test]
fn async_eval_under_tokio_current_thread() {
    // Verify the Future runs under tokio (proves runtime-agnostic).
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("tokio rt");
    rt.block_on(async {
        let arr = Array::zeros(&[256], Dtype::Float32).expect("zeros");
        arr.async_eval().await.expect("async_eval under tokio");
        let v: Vec<f32> = arr.to_vec().expect("to_vec");
        assert_eq!(v.len(), 256);
    });
}

#[test]
fn async_eval_under_tokio_multi_thread() {
    // Multi-threaded runtime: future may be polled on a worker thread
    // different from the submitter. Verifies our captured-stream fix.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("tokio multi-thread rt");
    rt.block_on(async {
        let arr = Array::zeros(&[256], Dtype::Float32).expect("zeros");
        arr.async_eval().await.expect("async_eval under multi-thread tokio");
        let v: Vec<f32> = arr.to_vec().expect("to_vec");
        assert_eq!(v.len(), 256);
    });
}

#[test]
fn async_eval_multiple_arrays() {
    // Submit multiple arrays in one async_eval call.
    let a = Array::zeros(&[64], Dtype::Float32).expect("zeros a");
    let b = Array::zeros(&[64], Dtype::Float32).expect("zeros b");
    futures_lite::future::block_on(transforms::async_eval(&[&a, &b]))
        .expect("async_eval multiple");
    assert_eq!(a.to_vec::<f32>().expect("to_vec a").len(), 64);
    assert_eq!(b.to_vec::<f32>().expect("to_vec b").len(), 64);
}

#[test]
fn synchronize_blocks_until_default_stream_drains() {
    // Submit work, then synchronously block on default stream.
    let arr = Array::zeros(&[128], Dtype::Float32).expect("zeros");
    futures_lite::future::block_on(arr.async_eval()).expect("async_eval");
    transforms::synchronize().expect("synchronize");
    // After explicit sync, the array must be evaluated.
    let v: Vec<f32> = arr.to_vec().expect("to_vec");
    assert_eq!(v.len(), 128);
}

#[test]
fn synchronize_stream_for_explicit_stream() {
    // Get the default stream explicitly and synchronize on it.
    let s = mlx::default_stream(mlx::default_device());
    let arr = Array::zeros(&[32], Dtype::Float32).expect("zeros");
    futures_lite::future::block_on(arr.async_eval()).expect("async_eval");
    transforms::synchronize_stream(s).expect("synchronize_stream");
    let v: Vec<f32> = arr.to_vec().expect("to_vec");
    assert_eq!(v.len(), 32);
}
