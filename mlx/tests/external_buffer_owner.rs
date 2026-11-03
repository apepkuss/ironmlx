//! An external Metal buffer must keep its allocation alive through lazy
//! graphs and asynchronous GPU references, even after its origin is dropped.
use mlx::Dtype;
use std::{
    alloc::{alloc_zeroed, dealloc, Layout},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};
struct Allocation {
    ptr: *mut u8,
    layout: Layout,
    drops: Arc<AtomicUsize>,
}
unsafe impl Send for Allocation {}
unsafe impl Sync for Allocation {}
impl Drop for Allocation {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr, self.layout) };
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}
fn allocation() -> (Arc<Allocation>, Arc<AtomicUsize>) {
    let layout = Layout::from_size_align(16384, 16384).unwrap();
    let ptr = unsafe { alloc_zeroed(layout) };
    assert!(!ptr.is_null());
    let drops = Arc::new(AtomicUsize::new(0));
    (
        Arc::new(Allocation {
            ptr,
            layout,
            drops: Arc::clone(&drops),
        }),
        drops,
    )
}
#[test]
fn external_owner_survives_origin_drop_and_lazy_graph() {
    if let Ok(root) = std::env::var("MLX_DIR") {
        mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib")).unwrap();
    }
    let (owner, drops) = allocation();
    let weak = Arc::downgrade(&owner);
    let input = unsafe {
        mlx::fast::array_from_external_buffer_owned(
            owner.ptr as usize,
            16384,
            &[4096],
            Dtype::Float32,
            owner,
        )
    }
    .unwrap();
    let alias = input.clone();
    let ones = mlx::ops::ones(&[4096][..], Dtype::Float32).unwrap();
    let out = &input + &ones;
    drop(input);
    drop(alias);
    assert!(weak.upgrade().is_some());
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert!(out.to_vec::<f32>().unwrap().iter().all(|v| *v == 1.));
    drop(out);
    drop(ones);
    // Command-buffer completion may release the Metal buffer asynchronously.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
    while drops.load(Ordering::SeqCst) == 0 && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(weak.upgrade().is_none());
}
#[test]
fn external_owner_is_released_on_validation_failure() {
    let (owner, drops) = allocation();
    let error = unsafe {
        mlx::fast::array_from_external_buffer_owned(
            owner.ptr as usize,
            16384,
            &[8192],
            Dtype::Float32,
            owner,
        )
    };
    assert!(error.is_err());
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}
