use ironmlx::core::cache::{
    ActiveKvEntryChunkReader, ActiveKvLayerChunkKind, ActiveKvOffloadConfig,
    ActiveKvOffloadSharedStats, ActiveKvOffloadStatus, ActiveKvOffloadStore, ActiveKvPageResidency,
    ActiveKvResidencyState, ActiveKvResidencySummary, ActiveKvResidencyTracker, PagedPrefixEntry,
    PagedPrefixEntryStats, PrefixLayerPayload,
};
use mlx::{Array, Dtype};

fn temp_dir(prefix: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("{prefix}-{}", uuid::Uuid::new_v4().simple()))
}

fn sample_entry() -> PagedPrefixEntry {
    let k_pages = Array::zeros(&[1_i32, 1_i32, 4_i32, 2_i32][..], Dtype::Float32).expect("k pages");
    let v_pages = Array::zeros(&[1_i32, 1_i32, 4_i32, 2_i32][..], Dtype::Float32).expect("v pages");
    PagedPrefixEntry {
        main_layers: vec![PrefixLayerPayload::FullPaged { k_pages, v_pages }],
        mtp_layers: Vec::new(),
        mtp_last_hidden: None,
        gemma4_drafter_last_hidden: None,
    }
}

fn sample_dense_entry() -> PagedPrefixEntry {
    let k = Array::zeros(&[1_i32, 1_i32, 4_i32, 2_i32][..], Dtype::Float32).expect("dense k");
    let v = Array::zeros(&[1_i32, 1_i32, 4_i32, 2_i32][..], Dtype::Float32).expect("dense v");
    PagedPrefixEntry {
        main_layers: vec![PrefixLayerPayload::FullDense { k, v }],
        mtp_layers: Vec::new(),
        mtp_last_hidden: None,
        gemma4_drafter_last_hidden: None,
    }
}

#[test]
fn residency_tracker_counts_active_kv_page_states() {
    let mut tracker = ActiveKvResidencyTracker::new();
    tracker.insert(ActiveKvPageResidency::resident(1, 0));
    let mut loading = ActiveKvPageResidency::resident(1, 1);
    loading.mark_loading();
    tracker.insert(loading);
    let mut dirty = ActiveKvPageResidency::resident(2, 0);
    dirty.mark_dirty();
    tracker.insert(dirty);
    let mut offloaded = ActiveKvPageResidency::resident(2, 1);
    offloaded.mark_offloaded("/tmp/ironmlx-active-kv-test".into(), 4096);
    assert_eq!(offloaded.state, ActiveKvResidencyState::Offloaded);
    tracker.insert(offloaded);

    let summary = tracker.summary();
    assert_eq!(summary.resident_pages, 1);
    assert_eq!(summary.loading_pages, 1);
    assert_eq!(summary.dirty_pages, 1);
    assert_eq!(summary.offloaded_pages, 1);
    assert_eq!(summary.offloaded_bytes, 4096);
}

#[test]
fn shared_stats_combines_parked_and_live_residency_offload_counts() {
    let root = temp_dir("ironmlx-active-kv-stats");
    let config = ActiveKvOffloadConfig::enabled(root);
    let stats = ActiveKvOffloadSharedStats::new(&config);
    let parked = PagedPrefixEntryStats {
        full_paged_pages: 2,
        payload_bytes: 4096,
        ..PagedPrefixEntryStats::default()
    };
    stats.record_swap_out(parked, 123);
    stats.set_residency_summary(ActiveKvResidencySummary {
        resident_pages: 3,
        offloaded_pages: 5,
        dirty_pages: 1,
        offloaded_bytes: 8192,
        swap_out_count: 3,
        swap_in_count: 2,
        stream_read_count: 7,
        ..ActiveKvResidencySummary::default()
    });

    let snapshot = stats.snapshot();
    assert_eq!(snapshot.resident_pages, 3);
    assert_eq!(snapshot.offloaded_pages, 7);
    assert_eq!(snapshot.offloaded_bytes, 12_288);
    assert_eq!(snapshot.dirty_pages, 1);
    assert_eq!(snapshot.swap_out_count, 4);
    assert_eq!(snapshot.swap_in_count, 2);
    assert_eq!(snapshot.stream_read_count, 7);

    stats.record_swap_in(parked, 456);
    let snapshot = stats.snapshot();
    assert_eq!(snapshot.offloaded_pages, 5);
    assert_eq!(snapshot.offloaded_bytes, 8192);
    assert_eq!(snapshot.swap_in_count, 3);
}

#[test]
fn shared_stats_reports_production_status_flags() {
    let disabled = ActiveKvOffloadSharedStats::new(&ActiveKvOffloadConfig::disabled()).snapshot();
    assert_eq!(disabled.status, ActiveKvOffloadStatus::Disabled);
    assert!(!disabled.active);
    assert!(!disabled.degraded);

    let root = temp_dir("ironmlx-active-kv-status");
    let stats = ActiveKvOffloadSharedStats::new(&ActiveKvOffloadConfig::enabled(root));
    let idle = stats.snapshot();
    assert_eq!(idle.status, ActiveKvOffloadStatus::Idle);
    assert!(!idle.active);
    assert!(!idle.degraded);

    stats.set_residency_summary(ActiveKvResidencySummary {
        offloaded_pages: 2,
        ..ActiveKvResidencySummary::default()
    });
    let active = stats.snapshot();
    assert_eq!(active.status, ActiveKvOffloadStatus::Active);
    assert!(active.active);
    assert!(!active.degraded);

    stats.record_error();
    let degraded = stats.snapshot();
    assert_eq!(degraded.status, ActiveKvOffloadStatus::Degraded);
    assert!(degraded.active);
    assert!(degraded.degraded);
}

#[test]
fn active_kv_offload_store_round_trips_dense_full_kv_entry() {
    let root = temp_dir("ironmlx-active-kv-dense-store");
    let store = ActiveKvOffloadStore::new(ActiveKvOffloadConfig::enabled(root.clone()))
        .expect("active kv store");
    let entry = sample_dense_entry();
    let payload = store
        .save(7, &[1, 2, 3, 4], 4, &entry)
        .expect("save dense active kv payload");

    let loaded = store.load(&payload).expect("load dense active kv payload");
    assert!(matches!(
        loaded.main_layers.as_slice(),
        [PrefixLayerPayload::FullDense { .. }]
    ));

    store.cleanup_all().expect("cleanup active kv store");
    assert!(!root.exists());
}

#[test]
fn active_kv_chunk_reader_iterates_persisted_payload_layers() {
    let root = temp_dir("ironmlx-active-kv-reader");
    let store = ActiveKvOffloadStore::new(ActiveKvOffloadConfig::enabled(root.clone()))
        .expect("active kv store");
    let entry = sample_dense_entry();
    let payload = store
        .save(9, &[1, 2, 3, 4], 4, &entry)
        .expect("save dense active kv payload");
    let loaded = store.load(&payload).expect("load dense active kv payload");

    let reader = ActiveKvEntryChunkReader::new(&loaded);
    let chunks: Vec<_> = reader.chunks().collect();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].layer_index, 0);
    assert_eq!(chunks[0].kind, ActiveKvLayerChunkKind::FullDense);
    assert!(chunks[0].is_main_layer);

    store.cleanup_all().expect("cleanup active kv store");
}

#[test]
fn active_kv_offload_store_round_trips_entry_and_cleans_up() {
    let root = temp_dir("ironmlx-active-kv-store");
    let store = ActiveKvOffloadStore::new(ActiveKvOffloadConfig::enabled(root.clone()))
        .expect("active kv store");
    let entry = sample_entry();
    let payload = store
        .save(42, &[10, 11, 12, 13], 4, &entry)
        .expect("save active kv payload");

    assert!(payload.path.exists());
    assert_eq!(payload.request_id, 42);
    assert_eq!(payload.cached_len, 4);
    assert!(payload.stats.payload_bytes > 0);

    let loaded = store.load(&payload).expect("load active kv payload");
    assert_eq!(loaded.main_layers.len(), 1);

    store.remove(&payload).expect("remove active kv payload");
    assert!(!payload.path.exists());
    store.cleanup_all().expect("cleanup active kv store");
    assert!(!root.exists());
}
