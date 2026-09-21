use ironmlx_lm::core::cache::PagedKVCache;
use ironmlx_lm::core::model_input::build_batch_attention_mask;
use ironmlx_lm::core::model_input::build_per_row_decode_mask;
use mlx::{Array, Dtype};
use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};
fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "idx={idx} actual={a} expected={e} diff={}",
            (a - e).abs()
        );
    }
}

fn unique_test_dir(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock before epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("ironmlx-{label}-{}-{nanos}", std::process::id()))
}
#[test]
#[serial_test::serial(mlx_metal)]
fn paged_kv_hot_cold_allows_direct_prefix_restore_when_empty_and_budget_fits() {
    let root = unique_test_dir("paged-kv-hot-cold-direct-restore");
    let mut empty = PagedKVCache::new(1, 1, 2, 2, Dtype::Float32, 8, 2, 8).expect("empty cache");
    empty
        .enable_hot_cold_tiering(
            ironmlx_runtime::core::page_storage::file_paged_kv_config(root.join("empty"), 4, 1)
                .expect("hot/cold config"),
        )
        .expect("enable hot/cold tiering");
    assert!(ironmlx_lm::test_support::paged_can_direct_install_prefix_pages(&empty, 2));
    assert!(ironmlx_lm::test_support::paged_can_direct_install_prefix_pages(&empty, 5));
    assert!(!ironmlx_lm::test_support::paged_can_direct_install_prefix_pages(&empty, 6));

    let mut live = PagedKVCache::new(1, 1, 2, 2, Dtype::Float32, 8, 2, 8).expect("live cache");
    live.enable_hot_cold_tiering(
        ironmlx_runtime::core::page_storage::file_paged_kv_config(root.join("live"), 4, 1)
            .expect("hot/cold config"),
    )
    .expect("enable hot/cold tiering");
    let k: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (1_i32, 1_i32, 2_i32, 2_i32))
        .try_into()
        .unwrap();
    let v: Array = (
        &[10.0_f32, 20.0, 30.0, 40.0][..],
        (1_i32, 1_i32, 2_i32, 2_i32),
    )
        .try_into()
        .unwrap();
    let mut offsets = vec![0_i32];
    live.update_and_fetch_on(&k, &v, &mut offsets, &[2], ())
        .expect("append live page");
    assert!(!ironmlx_lm::test_support::paged_can_direct_install_prefix_pages(&live, 1));

    fs::remove_dir_all(&root).expect("remove test hot/cold root");
}

#[test]
#[serial_test::serial(mlx_metal)]
fn paged_kv_hot_cold_direct_prefix_restore_installs_cold_pages_without_swap_in() {
    let root = unique_test_dir("paged-kv-hot-cold-direct-cold-restore");
    let mut paged = PagedKVCache::new(1, 1, 2, 2, Dtype::Float32, 12, 2, 16).expect("paged cache");
    paged
        .enable_hot_cold_tiering(
            ironmlx_runtime::core::page_storage::file_paged_kv_config(&root, 1, 1)
                .expect("hot/cold config"),
        )
        .expect("enable hot/cold tiering");

    let k_data: Vec<f32> = (0..(5 * 2 * 2))
        .map(|i| ((i % 17) as f32 - 8.0) * 0.013)
        .collect();
    let v_data: Vec<f32> = (0..(5 * 2 * 2))
        .map(|i| ((i % 19) as f32 - 9.0) * 0.017)
        .collect();
    let k_pages: Array = (k_data.as_slice(), (5_i32, 1_i32, 2_i32, 2_i32))
        .try_into()
        .unwrap();
    let v_pages: Array = (v_data.as_slice(), (5_i32, 1_i32, 2_i32, 2_i32))
        .try_into()
        .unwrap();
    let mut offsets = vec![0_i32];

    paged
        .restore_prefix_pages_for_row_on(&k_pages, &v_pages, &mut offsets, 0, 10, ())
        .expect("restore prefix into cold hot/cold cache");

    let summary = paged.hot_cold_summary().expect("hot/cold summary");
    assert_eq!(offsets, vec![10]);
    assert_eq!(summary.resident_pages, 2);
    assert_eq!(summary.offloaded_pages, 3);
    assert_eq!(
        summary.swap_in_count, 0,
        "direct cold prefix restore should not load offloaded pages back into resident slots"
    );
    assert!(
            summary.swap_out_count <= summary.offloaded_pages as u64,
            "direct cold prefix restore should not evict the same logical prefix repeatedly: {summary:?}"
        );

    let (k_ref, v_ref) = paged.materialize_prefix_on(&offsets, 10, ()).expect("ref");
    assert_close(
        &k_ref.to_vec::<f32>().unwrap(),
        &k_pages.to_vec::<f32>().unwrap(),
        1.0e-6,
    );
    assert_close(
        &v_ref.to_vec::<f32>().unwrap(),
        &v_pages.to_vec::<f32>().unwrap(),
        1.0e-6,
    );

    drop(paged);
    fs::remove_dir_all(&root).ok();
}

#[test]
#[serial_test::serial(mlx_metal)]
fn paged_kv_hot_cold_direct_prefix_restore_preserves_shared_and_private_tail_pages() {
    let root = unique_test_dir("paged-kv-hot-cold-direct-cold-restore-tail");
    let mut paged = PagedKVCache::new(2, 1, 2, 2, Dtype::Float32, 12, 2, 16).expect("paged cache");
    paged
        .enable_hot_cold_tiering(
            ironmlx_runtime::core::page_storage::file_paged_kv_config(&root, 1, 1)
                .expect("hot/cold config"),
        )
        .expect("enable hot/cold tiering");

    let k_data: Vec<f32> = (0..(5 * 2 * 2)).map(|i| i as f32 * 0.031).collect();
    let v_data: Vec<f32> = (0..(5 * 2 * 2)).map(|i| i as f32 * 0.047).collect();
    let k_pages: Array = (k_data.as_slice(), (5_i32, 1_i32, 2_i32, 2_i32))
        .try_into()
        .unwrap();
    let v_pages: Array = (v_data.as_slice(), (5_i32, 1_i32, 2_i32, 2_i32))
        .try_into()
        .unwrap();
    let mut offsets = vec![0_i32, 0_i32];

    paged
        .restore_prefix_pages_for_rows_on(&k_pages, &v_pages, &mut offsets, &[0, 1], 9, ())
        .expect("restore shared prefix with private tails");

    let summary = paged.hot_cold_summary().expect("hot/cold summary");
    assert_eq!(offsets, vec![9, 9]);
    assert_eq!(paged.block_table_row(0)[..5], [0, 1, 2, 3, 4]);
    assert_eq!(paged.block_table_row(1)[..5], [0, 1, 2, 3, 5]);
    assert!(
        summary.offloaded_pages > 0,
        "cold direct restore should keep older shared prefix pages offloaded: {summary:?}"
    );
    assert_eq!(
            summary.swap_in_count, 0,
            "direct cold restore should not stage cold shared prefix pages through resident slots: {summary:?}"
        );

    let (k_ref, v_ref) = paged
        .materialize_prefix_on(&offsets, 9, ())
        .expect("materialize restored rows");
    let expected_k = &k_data[..18];
    let expected_v = &v_data[..18];
    let actual_k = k_ref.to_vec::<f32>().unwrap();
    let actual_v = v_ref.to_vec::<f32>().unwrap();
    assert_close(&actual_k[..18], expected_k, 1.0e-6);
    assert_close(&actual_k[18..36], expected_k, 1.0e-6);
    assert_close(&actual_v[..18], expected_v, 1.0e-6);
    assert_close(&actual_v[18..36], expected_v, 1.0e-6);

    drop(paged);
    fs::remove_dir_all(&root).ok();
}

#[test]
fn paged_kv_hot_cold_drop_removes_segment_cache_dir() {
    let root = unique_test_dir("paged-kv-hot-cold-drop-cleanup");
    let storage_dir = {
        let mut paged =
            PagedKVCache::new(1, 1, 4, 4, Dtype::Float32, 8, 2, 8).expect("paged cache");
        paged
            .enable_hot_cold_tiering(
                ironmlx_runtime::core::page_storage::file_paged_kv_config(&root, 1, 1)
                    .expect("hot/cold config"),
            )
            .expect("enable hot/cold tiering");
        let storage_dir = paged
            .hot_cold_summary()
            .expect("hot/cold summary")
            .storage_dir;
        assert!(storage_dir.exists());
        storage_dir
    };

    assert!(
        !storage_dir.exists(),
        "dropping a hot/cold cache should remove its temporary segment directory: {}",
        storage_dir.display()
    );
    fs::remove_dir_all(&root).ok();
}

#[test]
#[serial_test::serial(mlx_metal)]
fn paged_kv_pressure_shrink_demotes_resident_pages_idempotently() {
    let root = unique_test_dir("paged-kv-pressure-shrink");
    let mut paged = PagedKVCache::new(1, 1, 2, 2, Dtype::Float32, 8, 2, 8).expect("paged cache");
    paged
        .enable_hot_cold_tiering(
            ironmlx_runtime::core::page_storage::file_paged_kv_config(&root, 4, 1)
                .expect("hot/cold config"),
        )
        .expect("enable hot/cold tiering");
    let data = vec![1.0_f32; 16];
    let k: Array = (data.as_slice(), (1_i32, 1_i32, 8_i32, 2_i32))
        .try_into()
        .unwrap();
    let v = k.clone();
    let mut offsets = vec![0_i32];
    paged
        .update_and_fetch_on(&k, &v, &mut offsets, &[8], ())
        .expect("populate hot cache");
    assert_eq!(paged.hot_cold_summary().unwrap().resident_pages, 4);

    let reclaimed = paged
        .shrink_hot_window_on(&offsets, 1, ())
        .expect("pressure shrink");
    assert_eq!(reclaimed, 3);
    let summary = paged.hot_cold_summary().unwrap();
    assert_eq!(summary.resident_pages, 1);
    assert_eq!(summary.offloaded_pages, 3);
    assert_eq!(summary.hot_window_pages, 1);
    assert_eq!(summary.configured_hot_window_pages, 4);
    assert_eq!(
        paged
            .shrink_hot_window_on(&offsets, 1, ())
            .expect("repeated pressure shrink"),
        0
    );
    assert!(paged.restore_configured_hot_window());
    assert_eq!(paged.hot_cold_summary().unwrap().hot_window_pages, 4);
    assert!(!paged.restore_configured_hot_window());

    drop(paged);
    fs::remove_dir_all(root).ok();
}

#[test]
#[serial_test::serial(mlx_metal)]
fn paged_kv_decode_stages_cold_pages_after_hot_window_recovery() {
    let root = unique_test_dir("paged-kv-hot-window-recovery");
    let mut paged = PagedKVCache::new(1, 1, 2, 2, Dtype::Float32, 8, 2, 8).expect("paged cache");
    paged
        .enable_hot_cold_tiering(
            ironmlx_runtime::core::page_storage::file_paged_kv_config(&root, 4, 1)
                .expect("hot/cold config"),
        )
        .expect("enable hot/cold tiering");

    let prefix_data: Vec<f32> = (0..12).map(|i| i as f32 * 0.03125).collect();
    let prefix: Array = (prefix_data.as_slice(), (1_i32, 1_i32, 6_i32, 2_i32))
        .try_into()
        .unwrap();
    let mut offsets = vec![0_i32];
    paged
        .update_and_fetch_on(&prefix, &prefix, &mut offsets, &[6], ())
        .expect("populate cache");
    assert_eq!(paged.hot_cold_summary().unwrap().resident_pages, 3);

    assert_eq!(
        paged
            .shrink_hot_window_on(&offsets, 1, ())
            .expect("pressure shrink"),
        2
    );
    assert!(paged.restore_configured_hot_window());
    assert!(ironmlx_lm::test_support::paged_has_nonresident_referenced_pages(&paged, &offsets));

    let q: Array = (&[0.25_f32, -0.5][..], (1_i32, 1_i32, 1_i32, 2_i32))
        .try_into()
        .unwrap();
    let step: Array = (&[0.75_f32, 0.125][..], (1_i32, 1_i32, 1_i32, 2_i32))
        .try_into()
        .unwrap();
    let actual = paged
        .update_and_attend_decode_on(&q, &step, &step, &mut offsets, &[1], 0.5, ())
        .expect("decode after restoring configured hot window");

    assert_eq!(offsets, vec![7]);
    assert_eq!(actual.shape().as_slice(), &[1, 1, 1, 2]);
    let summary = paged.hot_cold_summary().expect("hot/cold summary");
    assert_eq!(summary.offloaded_pages, 0);
    assert!(summary.swap_in_count >= 2);

    drop(paged);
    fs::remove_dir_all(root).ok();
}

#[test]
#[serial_test::serial(mlx_metal)]
fn paged_kv_hot_cold_clear_resets_stream_cache_and_counters() {
    let root = unique_test_dir("paged-kv-hot-cold-clear-reset");
    let mut paged = PagedKVCache::new(1, 1, 4, 4, Dtype::Float32, 8, 2, 8).expect("paged cache");
    paged
        .enable_hot_cold_tiering(
            ironmlx_runtime::core::page_storage::file_paged_kv_config(&root, 1, 1)
                .expect("hot/cold config"),
        )
        .expect("enable hot/cold tiering");

    let q_data: Vec<f32> = (0..(2 * 5 * 4))
        .map(|i| ((i % 19) as f32 - 9.0) * 0.017)
        .collect();
    let k_data: Vec<f32> = (0..(5 * 4))
        .map(|i| ((i % 23) as f32 - 11.0) * 0.021)
        .collect();
    let v_data: Vec<f32> = (0..(5 * 4))
        .map(|i| ((i % 29) as f32 - 14.0) * 0.019)
        .collect();
    let q: Array = (q_data.as_slice(), (1_i32, 2_i32, 5_i32, 4_i32))
        .try_into()
        .unwrap();
    let k: Array = (k_data.as_slice(), (1_i32, 1_i32, 5_i32, 4_i32))
        .try_into()
        .unwrap();
    let v: Array = (v_data.as_slice(), (1_i32, 1_i32, 5_i32, 4_i32))
        .try_into()
        .unwrap();
    let mut offsets = vec![0_i32];
    paged
        .update_and_attend_prefill_on(&q, &k, &v, &mut offsets, &[5], 0.5, None, ())
        .expect("hot/cold streaming prefill");
    let before_clear = paged.hot_cold_summary().expect("hot/cold summary");
    assert!(before_clear.swap_out_count > 0);
    assert!(before_clear.stream_read_count > 0);
    assert!(ironmlx_lm::test_support::paged_stream_cache_lengths(&paged).0 > 0);

    paged.clear();

    let after_clear = paged.hot_cold_summary().expect("hot/cold summary");
    assert_eq!(after_clear.resident_pages, 0);
    assert_eq!(after_clear.offloaded_pages, 0);
    assert_eq!(after_clear.loading_pages, 0);
    assert_eq!(after_clear.dirty_pages, 0);
    assert_eq!(after_clear.offloaded_bytes, 0);
    assert_eq!(after_clear.swap_out_count, 0);
    assert_eq!(after_clear.swap_in_count, 0);
    assert_eq!(after_clear.stream_read_count, 0);
    let (cache_len, lru_len) = ironmlx_lm::test_support::paged_stream_cache_lengths(&paged);
    assert_eq!(cache_len, 0);
    assert_eq!(lru_len, 0);

    drop(paged);
    fs::remove_dir_all(&root).ok();
}

#[test]
#[serial_test::serial(mlx_metal)]
fn paged_kv_hot_cold_streaming_decode_streams_cold_pages_without_resident_swap_in() {
    let root = unique_test_dir("paged-kv-hot-cold");
    let mut paged = PagedKVCache::new(2, 1, 4, 4, Dtype::Float32, 8, 2, 8).expect("paged cache");
    paged
        .enable_hot_cold_tiering(
            ironmlx_runtime::core::page_storage::file_paged_kv_config(&root, 1, 1)
                .expect("hot/cold config"),
        )
        .expect("enable hot/cold tiering");

    let prefix_k_data: Vec<f32> = (0..(2 * 4 * 4))
        .map(|i| ((i % 29) as f32 - 14.0) * 0.027)
        .collect();
    let prefix_v_data: Vec<f32> = (0..(2 * 4 * 4))
        .map(|i| ((i % 31) as f32 - 15.0) * 0.021)
        .collect();
    let prefix_k: Array = (prefix_k_data.as_slice(), (2_i32, 1_i32, 4_i32, 4_i32))
        .try_into()
        .unwrap();
    let prefix_v: Array = (prefix_v_data.as_slice(), (2_i32, 1_i32, 4_i32, 4_i32))
        .try_into()
        .unwrap();
    let mut offsets = vec![0_i32, 0_i32];
    paged
        .update_and_fetch_on(&prefix_k, &prefix_v, &mut offsets, &[4, 2], ())
        .expect("prefix append");
    let prefix_summary = paged.hot_cold_summary().expect("hot/cold summary");
    assert!(
        prefix_summary.offloaded_pages > 0,
        "expected prefix append to offload at least one page: {prefix_summary:?}"
    );
    assert!(
        prefix_summary.resident_pages <= 2,
        "hot window should bound resident pages: {prefix_summary:?}"
    );

    let q_data: Vec<f32> = (0..(2 * 2 * 4))
        .map(|i| ((i % 17) as f32 - 8.0) * 0.019)
        .collect();
    let step_k_data: Vec<f32> = (0..(2 * 4))
        .map(|i| ((i % 13) as f32 - 6.0) * 0.037)
        .collect();
    let step_v_data: Vec<f32> = (0..(2 * 4))
        .map(|i| ((i % 11) as f32 - 5.0) * 0.043)
        .collect();
    let q: Array = (q_data.as_slice(), (2_i32, 2_i32, 1_i32, 4_i32))
        .try_into()
        .unwrap();
    let step_k: Array = (step_k_data.as_slice(), (2_i32, 1_i32, 1_i32, 4_i32))
        .try_into()
        .unwrap();
    let step_v: Array = (step_v_data.as_slice(), (2_i32, 1_i32, 1_i32, 4_i32))
        .try_into()
        .unwrap();
    let scale = 0.5_f32;

    let actual = paged
        .update_and_attend_decode_on(&q, &step_k, &step_v, &mut offsets, &[1, 1], scale, ())
        .expect("hot/cold streaming decode");
    assert_eq!(offsets, vec![5, 3]);
    let decode_summary = paged.hot_cold_summary().expect("hot/cold summary");
    assert!(
        decode_summary.offloaded_pages >= prefix_summary.offloaded_pages,
        "decode should retain cold pages offloaded: {decode_summary:?}"
    );
    assert!(
        decode_summary.resident_pages <= 4,
        "decode should respect hot window plus staging budget: {decode_summary:?}"
    );
    assert_eq!(
            decode_summary.swap_in_count, prefix_summary.swap_in_count,
            "streaming decode should read immutable cold pages without promoting them to resident slots"
        );
    assert!(
            decode_summary.stream_read_count > prefix_summary.stream_read_count,
            "decode should account read-only cold page streaming: before={prefix_summary:?} after={decode_summary:?}"
        );

    let (k_ref, v_ref) = paged.materialize_prefix_on(&offsets, 5, ()).expect("ref");
    let mask = build_per_row_decode_mask(&offsets, 5, Dtype::Float32).expect("mask");
    let expected =
        mlx::fast::scaled_dot_product_attention(&q, &k_ref, &v_ref, scale, "", Some(&mask), None)
            .expect("dense sdpa");

    assert_eq!(actual.shape().as_slice(), &[2, 2, 1, 4]);
    assert_close(
        &actual.to_vec::<f32>().unwrap(),
        &expected.to_vec::<f32>().unwrap(),
        1.0e-3,
    );

    drop(paged);
    fs::remove_dir_all(&root).expect("remove test hot/cold root");
}

#[test]
#[serial_test::serial(mlx_metal)]
fn paged_kv_hot_cold_streaming_decode_reuses_cached_cold_pages() {
    let root = unique_test_dir("paged-kv-hot-cold-stream-cache");
    let mut paged = PagedKVCache::new(1, 1, 4, 4, Dtype::Float32, 8, 2, 16).expect("paged cache");
    paged
        .enable_hot_cold_tiering(
            ironmlx_runtime::core::page_storage::file_paged_kv_config(&root, 1, 1)
                .expect("hot/cold config"),
        )
        .expect("enable hot/cold tiering");

    let prefix_k_data: Vec<f32> = (0..(6 * 4))
        .map(|i| ((i % 29) as f32 - 14.0) * 0.027)
        .collect();
    let prefix_v_data: Vec<f32> = (0..(6 * 4))
        .map(|i| ((i % 31) as f32 - 15.0) * 0.021)
        .collect();
    let prefix_k: Array = (prefix_k_data.as_slice(), (1_i32, 1_i32, 6_i32, 4_i32))
        .try_into()
        .unwrap();
    let prefix_v: Array = (prefix_v_data.as_slice(), (1_i32, 1_i32, 6_i32, 4_i32))
        .try_into()
        .unwrap();
    let mut offsets = vec![0_i32];
    paged
        .update_and_fetch_on(&prefix_k, &prefix_v, &mut offsets, &[6], ())
        .expect("prefix append");

    let mut stream_counts = Vec::new();
    for step in 0..2 {
        let q_data: Vec<f32> = (0..(2 * 4))
            .map(|i| (((i + step * 7) % 17) as f32 - 8.0) * 0.019)
            .collect();
        let step_k_data: Vec<f32> = (0..4)
            .map(|i| (((i + step * 5) % 13) as f32 - 6.0) * 0.037)
            .collect();
        let step_v_data: Vec<f32> = (0..4)
            .map(|i| (((i + step * 3) % 11) as f32 - 5.0) * 0.043)
            .collect();
        let q: Array = (q_data.as_slice(), (1_i32, 2_i32, 1_i32, 4_i32))
            .try_into()
            .unwrap();
        let step_k: Array = (step_k_data.as_slice(), (1_i32, 1_i32, 1_i32, 4_i32))
            .try_into()
            .unwrap();
        let step_v: Array = (step_v_data.as_slice(), (1_i32, 1_i32, 1_i32, 4_i32))
            .try_into()
            .unwrap();
        let actual = paged
            .update_and_attend_decode_on(&q, &step_k, &step_v, &mut offsets, &[1], 0.5, ())
            .expect("hot/cold streaming decode");
        let (k_ref, v_ref) = paged
            .materialize_prefix_on(&offsets, offsets[0], ())
            .expect("ref");
        let mask = build_per_row_decode_mask(&offsets, offsets[0], Dtype::Float32).expect("mask");
        let expected =
            mlx::fast::scaled_dot_product_attention(&q, &k_ref, &v_ref, 0.5, "", Some(&mask), None)
                .expect("dense sdpa");
        assert_close(
            &actual.to_vec::<f32>().unwrap(),
            &expected.to_vec::<f32>().unwrap(),
            1.0e-4,
        );
        stream_counts.push(
            paged
                .hot_cold_summary()
                .expect("hot/cold summary")
                .stream_read_count,
        );
    }

    let [after_first_streams, after_second_streams]: [u64; 2] =
        stream_counts.try_into().expect("two stream counts");
    assert!(
        after_first_streams > 0,
        "first decode should stream cold pages"
    );
    assert!(
            after_second_streams <= after_first_streams + 1,
            "second decode may read the page newly offloaded after the first step, but should reuse older cached immutable cold pages: first={after_first_streams} second={after_second_streams}"
        );

    drop(paged);
    fs::remove_dir_all(&root).expect("remove test hot/cold root");
}

#[test]
#[serial_test::serial(mlx_metal)]
fn paged_kv_hot_cold_streaming_decode_loads_offloaded_segments_once() {
    let root = unique_test_dir("paged-kv-hot-cold-segment-cache");
    let mut paged = PagedKVCache::new(1, 1, 4, 4, Dtype::Float32, 12, 2, 16).expect("paged cache");
    paged
        .enable_hot_cold_tiering(
            ironmlx_runtime::core::page_storage::file_paged_kv_config(&root, 1, 4)
                .expect("hot/cold config"),
        )
        .expect("enable hot/cold tiering");

    let prefix_len = 10_i32;
    let prefix_k_data: Vec<f32> = (0..(prefix_len * 4))
        .map(|i| ((i % 29) as f32 - 14.0) * 0.027)
        .collect();
    let prefix_v_data: Vec<f32> = (0..(prefix_len * 4))
        .map(|i| ((i % 31) as f32 - 15.0) * 0.021)
        .collect();
    let prefix_k: Array = (prefix_k_data.as_slice(), (1_i32, 1_i32, prefix_len, 4_i32))
        .try_into()
        .unwrap();
    let prefix_v: Array = (prefix_v_data.as_slice(), (1_i32, 1_i32, prefix_len, 4_i32))
        .try_into()
        .unwrap();
    let mut offsets = vec![0_i32];
    paged
        .update_and_fetch_on(&prefix_k, &prefix_v, &mut offsets, &[prefix_len], ())
        .expect("prefix append");
    let prefix_summary = paged.hot_cold_summary().expect("hot/cold summary");
    assert_eq!(
        prefix_summary.offloaded_pages, 4,
        "prefix should offload the four cold pages as one chunk-sized run: {prefix_summary:?}"
    );

    let q_data: Vec<f32> = (0..(2 * 4))
        .map(|i| ((i % 17) as f32 - 8.0) * 0.019)
        .collect();
    let step_k_data: Vec<f32> = (0..4).map(|i| ((i % 13) as f32 - 6.0) * 0.037).collect();
    let step_v_data: Vec<f32> = (0..4).map(|i| ((i % 11) as f32 - 5.0) * 0.043).collect();
    let q: Array = (q_data.as_slice(), (1_i32, 2_i32, 1_i32, 4_i32))
        .try_into()
        .unwrap();
    let step_k: Array = (step_k_data.as_slice(), (1_i32, 1_i32, 1_i32, 4_i32))
        .try_into()
        .unwrap();
    let step_v: Array = (step_v_data.as_slice(), (1_i32, 1_i32, 1_i32, 4_i32))
        .try_into()
        .unwrap();
    let actual = paged
        .update_and_attend_decode_on(&q, &step_k, &step_v, &mut offsets, &[1], 0.5, ())
        .expect("hot/cold streaming decode");
    let summary = paged.hot_cold_summary().expect("hot/cold summary");
    assert_eq!(
        summary.stream_read_count, 1,
        "chunk-sized offloaded runs should be loaded once and sliced from memory: {summary:?}"
    );

    let (k_ref, v_ref) = paged
        .materialize_prefix_on(&offsets, offsets[0], ())
        .expect("ref");
    let mask = build_per_row_decode_mask(&offsets, offsets[0], Dtype::Float32).expect("mask");
    let expected =
        mlx::fast::scaled_dot_product_attention(&q, &k_ref, &v_ref, 0.5, "", Some(&mask), None)
            .expect("dense sdpa");
    assert_close(
        &actual.to_vec::<f32>().unwrap(),
        &expected.to_vec::<f32>().unwrap(),
        1.0e-4,
    );

    drop(paged);
    fs::remove_dir_all(&root).expect("remove test hot/cold root");
}

#[test]
#[serial_test::serial(mlx_metal)]
fn paged_kv_hot_cold_decode_stages_context_to_paged_kernel_when_budget_allows() {
    let root = unique_test_dir("paged-kv-hot-cold-staged-paged");
    let mut paged = PagedKVCache::new(1, 1, 4, 4, Dtype::Float32, 12, 2, 16).expect("paged cache");
    paged
        .enable_hot_cold_tiering(
            ironmlx_runtime::core::page_storage::file_paged_kv_config(&root, 1, 8)
                .expect("hot/cold config"),
        )
        .expect("enable hot/cold tiering");

    let prefix_len = 10_i32;
    let prefix_k_data: Vec<f32> = (0..(prefix_len * 4))
        .map(|i| ((i % 29) as f32 - 14.0) * 0.027)
        .collect();
    let prefix_v_data: Vec<f32> = (0..(prefix_len * 4))
        .map(|i| ((i % 31) as f32 - 15.0) * 0.021)
        .collect();
    let prefix_k: Array = (prefix_k_data.as_slice(), (1_i32, 1_i32, prefix_len, 4_i32))
        .try_into()
        .unwrap();
    let prefix_v: Array = (prefix_v_data.as_slice(), (1_i32, 1_i32, prefix_len, 4_i32))
        .try_into()
        .unwrap();
    let mut offsets = vec![0_i32];
    paged
        .update_and_fetch_on(&prefix_k, &prefix_v, &mut offsets, &[prefix_len], ())
        .expect("prefix append");
    let prefix_summary = paged.hot_cold_summary().expect("hot/cold summary");
    assert!(
        prefix_summary.offloaded_pages > 0,
        "prefix should create cold pages: {prefix_summary:?}"
    );

    let q_data: Vec<f32> = (0..(2 * 4))
        .map(|i| ((i % 17) as f32 - 8.0) * 0.019)
        .collect();
    let step_k_data: Vec<f32> = (0..4).map(|i| ((i % 13) as f32 - 6.0) * 0.037).collect();
    let step_v_data: Vec<f32> = (0..4).map(|i| ((i % 11) as f32 - 5.0) * 0.043).collect();
    let q: Array = (q_data.as_slice(), (1_i32, 2_i32, 1_i32, 4_i32))
        .try_into()
        .unwrap();
    let step_k: Array = (step_k_data.as_slice(), (1_i32, 1_i32, 1_i32, 4_i32))
        .try_into()
        .unwrap();
    let step_v: Array = (step_v_data.as_slice(), (1_i32, 1_i32, 1_i32, 4_i32))
        .try_into()
        .unwrap();

    let actual = paged
        .update_and_attend_decode_on(&q, &step_k, &step_v, &mut offsets, &[1], 0.5, ())
        .expect("staged paged decode");
    let decode_summary = paged.hot_cold_summary().expect("hot/cold summary");
    assert_eq!(
            decode_summary.stream_read_count, prefix_summary.stream_read_count,
            "when staging budget covers the decode context, hot/cold should use paged decode instead of streaming row chunks: before={prefix_summary:?} after={decode_summary:?}"
        );

    let (k_ref, v_ref) = paged
        .materialize_prefix_on(&offsets, offsets[0], ())
        .expect("ref");
    let mask = build_per_row_decode_mask(&offsets, offsets[0], Dtype::Float32).expect("mask");
    let expected =
        mlx::fast::scaled_dot_product_attention(&q, &k_ref, &v_ref, 0.5, "", Some(&mask), None)
            .expect("dense sdpa");
    assert_close(
        &actual.to_vec::<f32>().unwrap(),
        &expected.to_vec::<f32>().unwrap(),
        1.0e-4,
    );

    drop(paged);
    fs::remove_dir_all(&root).expect("remove test hot/cold root");
}

#[test]
#[serial_test::serial(mlx_metal)]
fn paged_kv_hot_cold_decode_keeps_staged_clean_pages_when_budget_allows() {
    let root = unique_test_dir("paged-kv-hot-cold-staged-retention");
    let mut paged = PagedKVCache::new(1, 1, 4, 4, Dtype::Float32, 16, 2, 16).expect("paged cache");
    paged
        .enable_hot_cold_tiering(
            ironmlx_runtime::core::page_storage::file_paged_kv_config(&root, 1, 8)
                .expect("hot/cold config"),
        )
        .expect("enable hot/cold tiering");

    let prefix_len = 10_i32;
    let prefix_k_data: Vec<f32> = (0..(prefix_len * 4))
        .map(|i| ((i % 29) as f32 - 14.0) * 0.027)
        .collect();
    let prefix_v_data: Vec<f32> = (0..(prefix_len * 4))
        .map(|i| ((i % 31) as f32 - 15.0) * 0.021)
        .collect();
    let prefix_k: Array = (prefix_k_data.as_slice(), (1_i32, 1_i32, prefix_len, 4_i32))
        .try_into()
        .unwrap();
    let prefix_v: Array = (prefix_v_data.as_slice(), (1_i32, 1_i32, prefix_len, 4_i32))
        .try_into()
        .unwrap();
    let mut offsets = vec![0_i32];
    paged
        .update_and_fetch_on(&prefix_k, &prefix_v, &mut offsets, &[prefix_len], ())
        .expect("prefix append");

    let q_data: Vec<f32> = (0..(2 * 4))
        .map(|i| ((i % 17) as f32 - 8.0) * 0.019)
        .collect();
    let q: Array = (q_data.as_slice(), (1_i32, 2_i32, 1_i32, 4_i32))
        .try_into()
        .unwrap();

    for step in 0..2 {
        let step_k_data: Vec<f32> = (0..4)
            .map(|i| ((i + step * 4) as f32 % 13.0 - 6.0) * 0.037)
            .collect();
        let step_v_data: Vec<f32> = (0..4)
            .map(|i| ((i + step * 4) as f32 % 11.0 - 5.0) * 0.043)
            .collect();
        let step_k: Array = (step_k_data.as_slice(), (1_i32, 1_i32, 1_i32, 4_i32))
            .try_into()
            .unwrap();
        let step_v: Array = (step_v_data.as_slice(), (1_i32, 1_i32, 1_i32, 4_i32))
            .try_into()
            .unwrap();
        paged
            .update_and_attend_decode_on(&q, &step_k, &step_v, &mut offsets, &[1], 0.5, ())
            .expect("staged paged decode");
    }

    let after_first_two_decodes = paged.hot_cold_summary().expect("hot/cold summary");
    let swap_in_after_two_decodes = after_first_two_decodes.swap_in_count;

    let step_k: Array = (
        [0.031_f32, -0.017, 0.011, 0.023].as_slice(),
        (1_i32, 1_i32, 1_i32, 4_i32),
    )
        .try_into()
        .unwrap();
    let step_v: Array = (
        [0.019_f32, 0.007, -0.029, 0.013].as_slice(),
        (1_i32, 1_i32, 1_i32, 4_i32),
    )
        .try_into()
        .unwrap();
    paged
        .update_and_attend_decode_on(&q, &step_k, &step_v, &mut offsets, &[1], 0.5, ())
        .expect("staged paged decode");

    let after_third_decode = paged.hot_cold_summary().expect("hot/cold summary");
    assert_eq!(
            after_third_decode.swap_in_count, swap_in_after_two_decodes,
            "budget-sized staged clean pages should remain resident across decode steps instead of being reloaded every token: before={after_first_two_decodes:?} after={after_third_decode:?}"
        );
    assert_eq!(
        after_third_decode.stream_read_count, 0,
        "budget-sized staged decode should keep using paged kernel rather than streaming chunks"
    );

    drop(paged);
    fs::remove_dir_all(&root).expect("remove test hot/cold root");
}

#[test]
#[serial_test::serial(mlx_metal)]
fn paged_kv_hot_cold_decode_uses_resident_paged_kernel_when_window_covers_context() {
    let root = unique_test_dir("paged-kv-hot-cold-resident");
    let mut paged = PagedKVCache::new(2, 1, 4, 4, Dtype::Float32, 8, 2, 8).expect("paged cache");
    paged
        .enable_hot_cold_tiering(
            ironmlx_runtime::core::page_storage::file_paged_kv_config(&root, 4, 1)
                .expect("hot/cold config"),
        )
        .expect("enable hot/cold tiering");

    let prefix_k_data: Vec<f32> = (0..(2 * 4 * 4))
        .map(|i| ((i % 29) as f32 - 14.0) * 0.027)
        .collect();
    let prefix_v_data: Vec<f32> = (0..(2 * 4 * 4))
        .map(|i| ((i % 31) as f32 - 15.0) * 0.021)
        .collect();
    let prefix_k: Array = (prefix_k_data.as_slice(), (2_i32, 1_i32, 4_i32, 4_i32))
        .try_into()
        .unwrap();
    let prefix_v: Array = (prefix_v_data.as_slice(), (2_i32, 1_i32, 4_i32, 4_i32))
        .try_into()
        .unwrap();
    let mut offsets = vec![0_i32, 0_i32];
    paged
        .update_and_fetch_on(&prefix_k, &prefix_v, &mut offsets, &[4, 2], ())
        .expect("prefix append");
    let prefix_summary = paged.hot_cold_summary().expect("hot/cold summary");
    assert_eq!(prefix_summary.offloaded_pages, 0);
    assert_eq!(prefix_summary.stream_read_count, 0);

    let q_data: Vec<f32> = (0..(2 * 2 * 4))
        .map(|i| ((i % 17) as f32 - 8.0) * 0.019)
        .collect();
    let step_k_data: Vec<f32> = (0..(2 * 4))
        .map(|i| ((i % 13) as f32 - 6.0) * 0.037)
        .collect();
    let step_v_data: Vec<f32> = (0..(2 * 4))
        .map(|i| ((i % 11) as f32 - 5.0) * 0.043)
        .collect();
    let q: Array = (q_data.as_slice(), (2_i32, 2_i32, 1_i32, 4_i32))
        .try_into()
        .unwrap();
    let step_k: Array = (step_k_data.as_slice(), (2_i32, 1_i32, 1_i32, 4_i32))
        .try_into()
        .unwrap();
    let step_v: Array = (step_v_data.as_slice(), (2_i32, 1_i32, 1_i32, 4_i32))
        .try_into()
        .unwrap();

    let actual = paged
        .update_and_attend_decode_on(&q, &step_k, &step_v, &mut offsets, &[1, 1], 0.5, ())
        .expect("resident paged decode");
    assert_eq!(offsets, vec![5, 3]);
    let decode_summary = paged.hot_cold_summary().expect("hot/cold summary");
    assert_eq!(decode_summary.offloaded_pages, 0);
    assert_eq!(
        decode_summary.stream_read_count, 0,
        "resident context should use the normal paged decode kernel instead of streaming"
    );

    let (k_ref, v_ref) = paged.materialize_prefix_on(&offsets, 5, ()).expect("ref");
    let mask = build_per_row_decode_mask(&offsets, 5, Dtype::Float32).expect("mask");
    let expected =
        mlx::fast::scaled_dot_product_attention(&q, &k_ref, &v_ref, 0.5, "", Some(&mask), None)
            .expect("dense sdpa");

    assert_eq!(actual.shape().as_slice(), &[2, 2, 1, 4]);
    assert_close(
        &actual.to_vec::<f32>().unwrap(),
        &expected.to_vec::<f32>().unwrap(),
        1.0e-4,
    );

    drop(paged);
    fs::remove_dir_all(&root).expect("remove test hot/cold root");
}

#[test]
#[serial_test::serial(mlx_metal)]
fn paged_kv_hot_cold_prefill_streaming_matches_dense_causal_sdpa() {
    let root = unique_test_dir("paged-kv-hot-cold-prefill");
    let mut paged = PagedKVCache::new(1, 1, 4, 4, Dtype::Float32, 8, 2, 8).expect("paged cache");
    paged
        .enable_hot_cold_tiering(
            ironmlx_runtime::core::page_storage::file_paged_kv_config(&root, 1, 1)
                .expect("hot/cold config"),
        )
        .expect("enable hot/cold tiering");

    let q_data: Vec<f32> = (0..(2 * 5 * 4))
        .map(|i| ((i % 19) as f32 - 9.0) * 0.017)
        .collect();
    let k_data: Vec<f32> = (0..(5 * 4))
        .map(|i| ((i % 23) as f32 - 11.0) * 0.021)
        .collect();
    let v_data: Vec<f32> = (0..(5 * 4))
        .map(|i| ((i % 29) as f32 - 14.0) * 0.019)
        .collect();
    let q: Array = (q_data.as_slice(), (1_i32, 2_i32, 5_i32, 4_i32))
        .try_into()
        .unwrap();
    let k: Array = (k_data.as_slice(), (1_i32, 1_i32, 5_i32, 4_i32))
        .try_into()
        .unwrap();
    let v: Array = (v_data.as_slice(), (1_i32, 1_i32, 5_i32, 4_i32))
        .try_into()
        .unwrap();
    let mut offsets = vec![0_i32];
    let scale = 0.5_f32;

    let actual = paged
        .update_and_attend_prefill_on(&q, &k, &v, &mut offsets, &[5], scale, None, ())
        .expect("hot/cold streaming prefill");
    assert_eq!(offsets, vec![5]);
    let summary = paged.hot_cold_summary().expect("hot/cold summary");
    assert!(
        summary.offloaded_pages > 0,
        "prefill should offload cold pages: {summary:?}"
    );
    assert!(
        summary.stream_read_count > 0,
        "prefill should stream cold pages: {summary:?}"
    );

    let (k_ref, v_ref) = paged.materialize_prefix_on(&offsets, 5, ()).expect("ref");
    let expected =
        mlx::fast::scaled_dot_product_attention(&q, &k_ref, &v_ref, scale, "causal", None, None)
            .expect("dense sdpa");
    assert_eq!(actual.shape().as_slice(), &[1, 2, 5, 4]);
    assert_close(
        &actual.to_vec::<f32>().unwrap(),
        &expected.to_vec::<f32>().unwrap(),
        1.0e-4,
    );

    drop(paged);
    fs::remove_dir_all(&root).expect("remove test hot/cold root");
}

#[test]
#[serial_test::serial(mlx_metal)]
fn paged_kv_hot_cold_prefill_keeps_resident_when_budget_allows() {
    let root = unique_test_dir("paged-kv-hot-cold-prefill-resident-budget");
    let mut paged = PagedKVCache::new(1, 1, 4, 4, Dtype::Float32, 8, 2, 8).expect("paged cache");
    paged
        .enable_hot_cold_tiering(
            ironmlx_runtime::core::page_storage::file_paged_kv_config(&root, 1, 7)
                .expect("hot/cold config"),
        )
        .expect("enable hot/cold tiering");

    let q_data: Vec<f32> = (0..(2 * 5 * 4))
        .map(|i| ((i % 19) as f32 - 9.0) * 0.017)
        .collect();
    let k_data: Vec<f32> = (0..(5 * 4))
        .map(|i| ((i % 23) as f32 - 11.0) * 0.021)
        .collect();
    let v_data: Vec<f32> = (0..(5 * 4))
        .map(|i| ((i % 29) as f32 - 14.0) * 0.019)
        .collect();
    let q: Array = (q_data.as_slice(), (1_i32, 2_i32, 5_i32, 4_i32))
        .try_into()
        .unwrap();
    let k: Array = (k_data.as_slice(), (1_i32, 1_i32, 5_i32, 4_i32))
        .try_into()
        .unwrap();
    let v: Array = (v_data.as_slice(), (1_i32, 1_i32, 5_i32, 4_i32))
        .try_into()
        .unwrap();
    let mut offsets = vec![0_i32];
    let scale = 0.5_f32;

    let actual = paged
        .update_and_attend_prefill_on(&q, &k, &v, &mut offsets, &[5], scale, None, ())
        .expect("hot/cold prefill");
    assert_eq!(offsets, vec![5]);

    let summary = paged.hot_cold_summary().expect("hot/cold summary");
    assert_eq!(
            summary.offloaded_pages, 0,
            "prefill should keep all resident pages when the staging budget covers the context: {summary:?}"
        );
    assert_eq!(
            summary.swap_out_count, 0,
            "prefill should not write cold pages to SSD when resident budget is sufficient: {summary:?}"
        );
    assert_eq!(
        summary.stream_read_count, 0,
        "resident prefill should not read cold pages from SSD: {summary:?}"
    );

    let (k_ref, v_ref) = paged.materialize_prefix_on(&offsets, 5, ()).expect("ref");
    let expected =
        mlx::fast::scaled_dot_product_attention(&q, &k_ref, &v_ref, scale, "causal", None, None)
            .expect("dense sdpa");
    assert_eq!(actual.shape().as_slice(), &[1, 2, 5, 4]);
    assert_close(
        &actual.to_vec::<f32>().unwrap(),
        &expected.to_vec::<f32>().unwrap(),
        1.0e-4,
    );

    drop(paged);
    fs::remove_dir_all(&root).expect("remove test hot/cold root");
}

#[test]
#[serial_test::serial(mlx_metal)]
fn paged_kv_hot_cold_prefill_streaming_reads_cold_pages_once_per_row() {
    let root = unique_test_dir("paged-kv-hot-cold-prefill-read-amp");
    let mut paged = PagedKVCache::new(1, 1, 4, 4, Dtype::Float32, 132, 4, 64).expect("paged cache");
    paged
        .enable_hot_cold_tiering(
            ironmlx_runtime::core::page_storage::file_paged_kv_config(&root, 2, 4)
                .expect("hot/cold config"),
        )
        .expect("enable hot/cold tiering");

    let q_len = 130_i32;
    let q_data: Vec<f32> = (0..(2 * q_len * 4))
        .map(|i| ((i % 31) as f32 - 15.0) * 0.009)
        .collect();
    let k_data: Vec<f32> = (0..(q_len * 4))
        .map(|i| ((i % 37) as f32 - 18.0) * 0.011)
        .collect();
    let v_data: Vec<f32> = (0..(q_len * 4))
        .map(|i| ((i % 41) as f32 - 20.0) * 0.013)
        .collect();
    let q: Array = (q_data.as_slice(), (1_i32, 2_i32, q_len, 4_i32))
        .try_into()
        .unwrap();
    let k: Array = (k_data.as_slice(), (1_i32, 1_i32, q_len, 4_i32))
        .try_into()
        .unwrap();
    let v: Array = (v_data.as_slice(), (1_i32, 1_i32, q_len, 4_i32))
        .try_into()
        .unwrap();
    let mut offsets = vec![0_i32];
    let scale = 0.5_f32;

    let actual = paged
        .update_and_attend_prefill_on(&q, &k, &v, &mut offsets, &[q_len], scale, None, ())
        .expect("hot/cold streaming prefill");
    assert_eq!(offsets, vec![q_len]);

    let summary = paged.hot_cold_summary().expect("hot/cold summary");
    assert!(
        summary.offloaded_pages > 0,
        "prefill should offload cold pages: {summary:?}"
    );
    assert!(
        summary.stream_read_count <= (summary.offloaded_pages as u64) + 1,
        "streaming prefill should not reread cold pages for each query chunk: {summary:?}"
    );

    let (k_ref, v_ref) = paged
        .materialize_prefix_on(&offsets, q_len, ())
        .expect("ref");
    let expected =
        mlx::fast::scaled_dot_product_attention(&q, &k_ref, &v_ref, scale, "causal", None, None)
            .expect("dense sdpa");
    assert_eq!(actual.shape().as_slice(), &[1, 2, q_len, 4]);
    assert_close(
        &actual.to_vec::<f32>().unwrap(),
        &expected.to_vec::<f32>().unwrap(),
        1.0e-3,
    );

    drop(paged);
    fs::remove_dir_all(&root).expect("remove test hot/cold root");
}

#[test]
#[serial_test::serial(mlx_metal)]
fn paged_kv_hot_cold_prefill_streaming_matches_dense_batched_mask_sdpa() {
    let root = unique_test_dir("paged-kv-hot-cold-prefill-batch");
    let mut paged = PagedKVCache::new(2, 1, 4, 4, Dtype::Float32, 8, 2, 8).expect("paged cache");
    paged
        .enable_hot_cold_tiering(
            ironmlx_runtime::core::page_storage::file_paged_kv_config(&root, 1, 1)
                .expect("hot/cold config"),
        )
        .expect("enable hot/cold tiering");

    let q_data: Vec<f32> = (0..(2 * 2 * 4 * 4))
        .map(|i| ((i % 31) as f32 - 15.0) * 0.013)
        .collect();
    let k_data: Vec<f32> = (0..(2 * 4 * 4))
        .map(|i| ((i % 37) as f32 - 18.0) * 0.011)
        .collect();
    let v_data: Vec<f32> = (0..(2 * 4 * 4))
        .map(|i| ((i % 41) as f32 - 20.0) * 0.015)
        .collect();
    let q: Array = (q_data.as_slice(), (2_i32, 2_i32, 4_i32, 4_i32))
        .try_into()
        .unwrap();
    let k: Array = (k_data.as_slice(), (2_i32, 1_i32, 4_i32, 4_i32))
        .try_into()
        .unwrap();
    let v: Array = (v_data.as_slice(), (2_i32, 1_i32, 4_i32, 4_i32))
        .try_into()
        .unwrap();
    let mask = build_batch_attention_mask(&[2, 4], 4, Dtype::Float32).expect("mask");
    let mut offsets = vec![0_i32, 0_i32];
    let scale = 0.5_f32;

    let actual = paged
        .update_and_attend_prefill_on(&q, &k, &v, &mut offsets, &[2, 4], scale, Some(&mask), ())
        .expect("hot/cold batched streaming prefill");
    assert_eq!(offsets, vec![2, 4]);

    let (k_ref, v_ref) = paged.materialize_prefix_on(&offsets, 4, ()).expect("ref");
    let expected =
        mlx::fast::scaled_dot_product_attention(&q, &k_ref, &v_ref, scale, "", Some(&mask), None)
            .expect("dense sdpa");
    assert_eq!(actual.shape().as_slice(), &[2, 2, 4, 4]);
    assert_close(
        &actual.to_vec::<f32>().unwrap(),
        &expected.to_vec::<f32>().unwrap(),
        1.0e-4,
    );

    drop(paged);
    fs::remove_dir_all(&root).expect("remove test hot/cold root");
}

#[test]
#[serial_test::serial(mlx_metal)]
fn paged_kv_hot_cold_repeated_decode_matches_dense_sdpa() {
    let root = unique_test_dir("paged-kv-hot-cold-repeated");
    let mut paged = PagedKVCache::new(3, 2, 4, 4, Dtype::Float32, 24, 2, 48).expect("paged cache");
    paged
        .enable_hot_cold_tiering(
            ironmlx_runtime::core::page_storage::file_paged_kv_config(&root, 2, 2)
                .expect("hot/cold config"),
        )
        .expect("enable hot/cold tiering");

    let prefix_k_data: Vec<f32> = (0..(3 * 2 * 5 * 4))
        .map(|i| ((i % 37) as f32 - 18.0) * 0.017)
        .collect();
    let prefix_v_data: Vec<f32> = (0..(3 * 2 * 5 * 4))
        .map(|i| ((i % 41) as f32 - 20.0) * 0.013)
        .collect();
    let prefix_k: Array = (prefix_k_data.as_slice(), (3_i32, 2_i32, 5_i32, 4_i32))
        .try_into()
        .unwrap();
    let prefix_v: Array = (prefix_v_data.as_slice(), (3_i32, 2_i32, 5_i32, 4_i32))
        .try_into()
        .unwrap();
    let mut offsets = vec![0_i32, 0_i32, 0_i32];
    paged
        .update_and_fetch_on(&prefix_k, &prefix_v, &mut offsets, &[5, 3, 4], ())
        .expect("prefix append");

    for step in 0..6 {
        let q_data: Vec<f32> = (0..(3 * 4 * 4))
            .map(|i| (((i + step * 11) % 43) as f32 - 21.0) * 0.011)
            .collect();
        let step_k_data: Vec<f32> = (0..(3 * 2 * 4))
            .map(|i| (((i + step * 7) % 31) as f32 - 15.0) * 0.019)
            .collect();
        let step_v_data: Vec<f32> = (0..(3 * 2 * 4))
            .map(|i| (((i + step * 5) % 29) as f32 - 14.0) * 0.023)
            .collect();
        let q: Array = (q_data.as_slice(), (3_i32, 4_i32, 1_i32, 4_i32))
            .try_into()
            .unwrap();
        let step_k: Array = (step_k_data.as_slice(), (3_i32, 2_i32, 1_i32, 4_i32))
            .try_into()
            .unwrap();
        let step_v: Array = (step_v_data.as_slice(), (3_i32, 2_i32, 1_i32, 4_i32))
            .try_into()
            .unwrap();
        let scale = 0.5_f32;

        let actual = paged
            .update_and_attend_decode_on(&q, &step_k, &step_v, &mut offsets, &[1, 1, 1], scale, ())
            .expect("hot/cold streaming decode");
        let max_len = offsets.iter().copied().max().expect("offsets");
        let (k_ref, v_ref) = paged
            .materialize_prefix_on(&offsets, max_len, ())
            .expect("ref");
        let mask = build_per_row_decode_mask(&offsets, max_len, Dtype::Float32).expect("mask");
        let expected = mlx::fast::scaled_dot_product_attention(
            &q,
            &k_ref,
            &v_ref,
            scale,
            "",
            Some(&mask),
            None,
        )
        .expect("dense sdpa");
        assert_eq!(actual.shape().as_slice(), &[3, 4, 1, 4]);
        assert_close(
            &actual.to_vec::<f32>().unwrap(),
            &expected.to_vec::<f32>().unwrap(),
            1.0e-4,
        );

        let summary = paged.hot_cold_summary().expect("hot/cold summary");
        assert!(
            summary.resident_pages <= 12,
            "decode step {step} should respect the hot window plus staging budget: {summary:?}"
        );
    }

    drop(paged);
    fs::remove_dir_all(&root).expect("remove test hot/cold root");
}
