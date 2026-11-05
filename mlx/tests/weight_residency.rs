use mlx::{memory, Array};

fn configure_test_metallib() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib")).unwrap();
        }
    });
}

// One test owns the process-wide policy while checking all lifecycle edges.
#[test]
fn scoped_residency_owns_only_weights_and_releases_on_every_exit() {
    configure_test_metallib();
    let hardware = memory::snapshot();
    if hardware.total_bytes.is_none_or(|n| n <= 8usize << 30)
        || hardware
            .max_recommended_bytes
            .is_none_or(|n| n <= 2usize << 30)
    {
        eprintln!("scoped residency lifecycle requires host/Metal budget headroom");
        return;
    }
    let baseline = memory::weight_residency_stats();
    memory::set_weight_residency_enabled(true).unwrap();
    let weight = Array::try_from((vec![1f32; 1024 * 1024].as_slice(), &[1024, 1024][..])).unwrap();
    weight.eval().unwrap();
    let view = weight.slice(&[0, 0][..], &[1, 1024][..]).unwrap();
    view.eval().unwrap();
    assert!(memory::WeightResidency::new(&[], 16 << 20).is_err());
    assert!(memory::WeightResidency::new(&[&weight], 1).is_err());
    let pending = Array::zeros([1024], mlx::Dtype::Float32).unwrap();
    assert!(memory::WeightResidency::new(&[&pending], 16 << 20).is_err());
    assert!(
        !mlx_sys::array::ffi::array_is_available(pending.as_inner()),
        "residency must not evaluate a pending graph"
    );
    let empty = Array::zeros([0], mlx::Dtype::Float32).unwrap();
    empty.eval().unwrap();
    assert!(memory::WeightResidency::new(&[&empty], 16 << 20).is_err());
    assert_eq!(memory::weight_residency_stats(), baseline);

    let lease = match memory::WeightResidency::new(&[&weight], 16 << 20) {
        Ok(lease) => lease,
        Err(error)
            if error.to_string().contains("requires macOS 15")
                || error.to_string().contains("requires Metal 3") =>
        {
            eprintln!("scoped residency lifecycle test unavailable: {error}");
            return;
        }
        Err(error) => panic!("scoped residency setup failed: {error}"),
    };
    assert!(lease.activate_on_current_queue().unwrap());
    let activated = memory::weight_residency_stats();
    assert_eq!(activated.queue_attachments, baseline.queue_attachments + 1);
    assert!(
        lease.activate_on_current_queue().unwrap(),
        "activation of an already attached queue stays idempotent"
    );
    assert_eq!(memory::weight_residency_stats(), activated);
    let bytes = lease.bytes();
    let duplicate = memory::WeightResidency::new(&[&weight, &weight, &view], 16 << 20).unwrap();
    assert_eq!(
        duplicate.bytes(),
        bytes,
        "views must not charge storage twice"
    );
    let active = memory::weight_residency_stats();
    assert_eq!(active.leases, baseline.leases + 2);
    assert_eq!(active.reserved_bytes, baseline.reserved_bytes + 2 * bytes);
    assert_eq!(active.requested_bytes, active.reserved_bytes);
    assert_eq!(active.queue_attachments, baseline.queue_attachments + 1);

    // Foreign global policy takes priority; no scoped set may crowd its queues.
    let previous = memory::set_wired_limit(8 << 20).unwrap();
    assert_eq!(memory::weight_residency_stats().queue_attachments, 0);
    assert_eq!(memory::weight_residency_stats().requested_bytes, 0);
    assert!(!lease.bind_on(()).unwrap());
    assert_eq!(memory::set_wired_limit(0).unwrap(), 8 << 20);
    assert!(lease.bind_on(()).unwrap());
    assert!(duplicate.bind_on(()).unwrap());
    assert!(memory::set_wired_limit(usize::MAX).is_err());
    assert_eq!(
        memory::weight_residency_stats().requested_bytes,
        active.requested_bytes
    );
    assert!(
        lease.bind_on(()).unwrap(),
        "failed foreign policy must restore scoped eligibility"
    );

    memory::set_weight_residency_enabled(false).unwrap();
    assert_eq!(memory::weight_residency_stats().requested_bytes, 0);
    assert_eq!(memory::weight_residency_stats().queue_attachments, 0);
    assert_eq!(
        weight
            .sum(mlx::ops::All, false)
            .unwrap()
            .item::<f32>()
            .unwrap(),
        1048576.
    );
    let paused = memory::WeightResidency::new(&[&weight], 16 << 20).unwrap();
    assert_eq!(memory::weight_residency_stats().requested_bytes, 0);
    assert!(!paused.activate_on_current_queue().unwrap());
    drop(paused);
    memory::set_weight_residency_enabled(true).unwrap();
    assert_eq!(
        memory::weight_residency_stats().requested_bytes,
        active.requested_bytes
    );
    assert!(lease.bind_on(()).unwrap());
    drop(duplicate);
    assert_eq!(
        memory::weight_residency_stats().reserved_bytes,
        baseline.reserved_bytes + bytes
    );

    // A failed caller must unwind its lease without retaining model storage.
    fn fail_start(weight: &Array) -> Result<(), &'static str> {
        let _temporary = memory::WeightResidency::new(&[weight], 16 << 20).unwrap();
        Err("model startup failed")
    }
    let failure = fail_start(&weight);
    assert!(failure.is_err());
    assert_eq!(memory::weight_residency_stats().leases, baseline.leases + 1);
    let surviving_graph = &weight + &weight;
    drop(view);
    drop(weight);
    // Moving and dropping a lease on a model-worker thread is supported.
    std::thread::spawn(move || drop(lease)).join().unwrap();
    assert_eq!(memory::weight_residency_stats(), baseline);
    assert_eq!(
        surviving_graph
            .sum(mlx::ops::All, false)
            .unwrap()
            .item::<f32>()
            .unwrap(),
        2097152.
    );

    let reloaded = Array::try_from((vec![1f32; 1024].as_slice(), [1024])).unwrap();
    reloaded.eval().unwrap();
    let lease = memory::WeightResidency::new(&[&reloaded], 16 << 20).unwrap();
    assert_eq!(memory::weight_residency_stats().leases, baseline.leases + 1);
    drop(lease);
    assert_eq!(memory::weight_residency_stats(), baseline);
    // Share one small weight across enough owners to hit the queue limit.
    let mut owners = Vec::new();
    let mut bound = 0;
    for _ in 0..40 {
        let lease = memory::WeightResidency::new(&[&reloaded], 16 << 20).unwrap();
        bound += usize::from(lease.bind_on(()).unwrap());
        owners.push(lease);
    }
    assert!(
        bound > 0 && bound <= 31,
        "MLX and scoped sets must fit Metal's queue cap"
    );
    assert_eq!(
        memory::weight_residency_stats().queue_attachments,
        baseline.queue_attachments + bound
    );
    drop(owners);
    assert_eq!(memory::weight_residency_stats(), baseline);
    memory::set_wired_limit(previous).unwrap();
}

#[test]
fn weight_lease_can_move_but_cannot_be_shared_without_synchronization() {
    fn send<T: Send>() {}
    send::<memory::WeightResidency>();
    static_assertions::assert_not_impl_any!(memory::WeightResidency: Sync);
}

#[test]
fn concurrent_pressure_cancellation_and_cross_thread_retirement() {
    const CHILD: &str = "IRONMLX_RESIDENCY_STRESS_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "concurrent_pressure_cancellation_and_cross_thread_retirement",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CHILD, "1")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
        while child.try_wait().unwrap().is_none() {
            if std::time::Instant::now() >= deadline {
                child.kill().unwrap();
                let output = child.wait_with_output().unwrap();
                panic!(
                    "concurrent residency timed out\n{}\n{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "concurrent residency failed\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        print!("{stdout}");
        assert!(stdout.contains("completed=256") || stdout.contains("skipped: "));
        return;
    }

    use std::sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc, Arc, Barrier,
    };
    configure_test_metallib();
    let hardware = memory::snapshot();
    if hardware.total_bytes.is_none_or(|n| n <= 8usize << 30)
        || hardware
            .max_recommended_bytes
            .is_none_or(|n| n <= 2usize << 30)
    {
        println!("skipped: host/Metal budget headroom");
        return;
    }
    let probe = Array::try_from(([1f32].as_slice(), [1])).unwrap();
    probe.eval().unwrap();
    match memory::WeightResidency::new(&[&probe], 16 << 20) {
        Ok(lease) => drop(lease),
        Err(error)
            if error.to_string().contains("requires macOS 15")
                || error.to_string().contains("requires Metal 3") =>
        {
            println!("skipped: {error}");
            return;
        }
        Err(error) => panic!("scoped residency setup failed: {error}"),
    }
    const WORKERS: usize = 4;
    const ROUNDS: usize = 64;
    const ELEMENTS: usize = 1 << 18;
    let baseline = memory::weight_residency_stats();
    let previous = memory::set_wired_limit(0).unwrap();
    memory::set_weight_residency_enabled(true).unwrap();
    let barrier = Arc::new(Barrier::new(WORKERS + 3));
    let done = Arc::new(AtomicBool::new(false));
    let pressure_cycles = Arc::new(AtomicUsize::new(0));
    let binds = Arc::new(AtomicUsize::new(0));
    let (tx, rx) = mpsc::channel::<memory::WeightResidency>();
    let reaper = std::thread::spawn(move || {
        for lease in rx {
            drop(lease);
        }
    });
    let mut workers = Vec::new();
    for worker in 0..WORKERS {
        let (barrier, tx, binds) = (Arc::clone(&barrier), tx.clone(), Arc::clone(&binds));
        workers.push(std::thread::spawn(move || {
            let stream = mlx::new_stream(mlx::Device::gpu(0)).unwrap();
            mlx::set_default_stream(stream);
            let warm =
                Array::try_from((vec![1f32; ELEMENTS].as_slice(), [ELEMENTS as i32])).unwrap();
            warm.eval().unwrap();
            let initial = memory::WeightResidency::new(&[&warm], 16 << 20).unwrap();
            binds.fetch_add(
                usize::from(initial.bind_on(stream).unwrap()),
                Ordering::Relaxed,
            );
            warm.sum(mlx::ops::All, false)
                .unwrap()
                .item::<f32>()
                .unwrap();
            barrier.wait();
            tx.send(initial).unwrap();
            for round in 0..ROUNDS {
                let value = ((worker + round) % 7 + 1) as f32;
                let weight =
                    Array::try_from((vec![value; ELEMENTS].as_slice(), [ELEMENTS as i32])).unwrap();
                weight.eval().unwrap();
                let lease = memory::WeightResidency::new(&[&weight], 16 << 20).unwrap();
                binds.fetch_add(
                    usize::from(lease.bind_on(stream).unwrap()),
                    Ordering::Relaxed,
                );
                let output = &weight + &weight;
                mlx::transforms::async_eval(&[&output]).unwrap();
                tx.send(lease).unwrap();
                let result = if round % 4 == 0 {
                    // Abandon submitted work, then reuse the same execution stream.
                    drop(output);
                    &weight + &weight
                } else {
                    output
                };
                assert_eq!(
                    result
                        .sum(mlx::ops::All, false)
                        .unwrap()
                        .item::<f32>()
                        .unwrap(),
                    2. * value * ELEMENTS as f32
                );
            }
            mlx::synchronize_stream(stream).unwrap();
        }));
    }
    drop(tx);
    let (pressure_barrier, pressure_done, cycles) = (
        Arc::clone(&barrier),
        Arc::clone(&done),
        Arc::clone(&pressure_cycles),
    );
    let pressure = std::thread::spawn(move || {
        pressure_barrier.wait();
        while !pressure_done.load(Ordering::Acquire) {
            let observed = memory::weight_residency_stats().leases > 0;
            memory::set_weight_residency_enabled(false).unwrap();
            memory::set_weight_residency_enabled(true).unwrap();
            if observed {
                cycles.fetch_add(1, Ordering::Relaxed);
            }
            std::thread::sleep(std::time::Duration::from_micros(100));
        }
    });
    let foreign_barrier = Arc::clone(&barrier);
    let foreign = std::thread::spawn(move || {
        foreign_barrier.wait();
        for _ in 0..64 {
            memory::set_wired_limit(16 << 20).unwrap();
            assert!(memory::set_wired_limit(usize::MAX).is_err());
            memory::set_wired_limit(0).unwrap();
            std::thread::yield_now();
        }
    });
    barrier.wait();
    for worker in workers {
        worker.join().unwrap();
    }
    done.store(true, Ordering::Release);
    pressure.join().unwrap();
    foreign.join().unwrap();
    reaper.join().unwrap();
    memory::set_weight_residency_enabled(true).unwrap();
    memory::set_wired_limit(previous).unwrap();
    assert_eq!(memory::weight_residency_stats(), baseline);
    assert!(pressure_cycles.load(Ordering::Relaxed) > 0);
    assert!(binds.load(Ordering::Relaxed) > 0);
    println!(
        "completed=256 pressure_cycles={} bound={}",
        pressure_cycles.load(Ordering::Relaxed),
        binds.load(Ordering::Relaxed)
    );
}
