//! Tests of the `wp8` SSD format: whole-prefix entries stored as shared
//! segment files plus a private part.

use super::*;
use ironmlx_lm::core::cache::{export_prefix_segment_for_row, KVCache, TurboQuantKVBits};
use mlx::{Array, Dtype};
use std::os::unix::fs::MetadataExt;

const BS: i32 = 128;
const HEADS: i32 = 2;
const DIM: i32 = 8;

fn temp_root(name: &str) -> PathBuf {
    let root =
        std::env::temp_dir().join(format!("ironmlx-{name}-{}", uuid::Uuid::new_v4().simple()));
    fs::create_dir_all(&root).unwrap();
    root
}

fn token_data(n: i32, salt: f32) -> Array {
    let data: Vec<f32> = (0..(HEADS * n * DIM))
        .map(|i| ((i as f32) * 0.013 + salt).sin())
        .collect();
    (data.as_slice(), (1_i32, HEADS, n, DIM))
        .try_into()
        .unwrap()
}

fn live(kind: &str, n: i32, salt: f32) -> Vec<LayerCache> {
    (0..2)
        .map(|layer| {
            let base = KVCache::new(1, HEADS, DIM, DIM, Dtype::Float32, 1024).with_step(BS);
            let mut kv = match kind {
                "paged" => base.with_paged(BS, 8).expect("paged"),
                _ => base
                    .with_turboquant(TurboQuantKVBits::K3V4)
                    .expect("turboquant"),
            };
            kv.update_and_fetch(
                &token_data(n, salt + layer as f32),
                &token_data(n, salt + 3.0 + layer as f32),
                &[n],
            )
            .expect("fill");
            LayerCache::Full(kv)
        })
        .collect()
}

fn extend(caches: &mut [LayerCache], n: i32, salt: f32) {
    for (layer, cache) in caches.iter_mut().enumerate() {
        let LayerCache::Full(kv) = cache else {
            unreachable!()
        };
        kv.update_and_fetch(
            &token_data(n, salt + layer as f32),
            &token_data(n, salt + 3.0 + layer as f32),
            &[n],
        )
        .expect("extend");
    }
}

fn whole_spec(kind: &str, token_ids: Vec<i32>, entry: &PagedPrefixEntry) -> PagedPrefixKeySpec {
    PagedPrefixKeySpec {
        entry_kind: PrefixEntryKind::WholePrefix,
        model_id: "wp8-test".to_owned(),
        cached_len: token_ids.len() as i32,
        token_ids,
        fingerprint: None,
        block_size: BS,
        kv_cache_profile: (kind != "paged").then(|| "turboquant-k3v4".to_owned()),
        main_layers: entry.main_layer_specs(),
        mtp_layers: vec![],
        mtp_last_hidden: None,
        gemma4_drafter_last_hidden: None,
    }
}

/// Export row 0 of `caches` after the segments `refs` cover: the new
/// aligned range becomes one segment. `token_ids` key the entry.
fn export(
    kind: &str,
    caches: &[LayerCache],
    refs: Vec<PrefixSegmentRef>,
    token_ids: Vec<i32>,
) -> (PagedPrefixKeySpec, SegmentedPrefixEntry) {
    let from = refs.last().map_or(0, |r| r.stop);
    let (segment, private, cached_len) = export_prefix_segment_for_row(
        caches,
        0,
        from,
        BS,
        PrefixSegmentLayout("placeholder".into()),
    )
    .unwrap()
    .unwrap();
    assert_eq!(cached_len as usize, token_ids.len());
    let mut segments = refs;
    let fresh = segment.map(|segment| {
        let stop = segment.stop;
        (segment, stop)
    });
    let aligned_len = fresh.as_ref().map_or(from, |(_, stop)| *stop);
    // The layout ignores lengths: key a spec with any one first.
    let probe = SegmentedPrefixEntry {
        window_refs: 0,
        window_start: 0,
        segments: segments.clone(),
        aligned_len: from,
        private: private.clone(),
    };
    let spec = match &fresh {
        None => {
            let materialized = materialize_prefix_entry(&probe, cached_len, BS).unwrap();
            whole_spec(kind, token_ids, &materialized)
        }
        Some((segment, stop)) => {
            let probe_segment = Arc::new(PrefixKvSegment {
                id: segment.id,
                layout: segment.layout.clone(),
                start: segment.start,
                stop: segment.stop,
                layers: segment.layers.clone(),
            });
            let mut probe = probe;
            probe.segments.push(PrefixSegmentRef {
                segment: probe_segment,
                stop: *stop,
            });
            probe.aligned_len = *stop;
            let materialized = materialize_prefix_entry(&probe, cached_len, BS).unwrap();
            whole_spec(kind, token_ids, &materialized)
        }
    };
    let layout = prefix_segment_layout(&spec);
    for r in &segments {
        assert_eq!(r.segment.layout, layout, "refs share the entry's layout");
    }
    if let Some((segment, stop)) = fresh {
        segments.push(PrefixSegmentRef {
            segment: Arc::new(PrefixKvSegment { layout, ..segment }),
            stop,
        });
    }
    (
        spec,
        SegmentedPrefixEntry {
            window_refs: 0,
            window_start: 0,
            segments,
            aligned_len,
            private,
        },
    )
}

fn flat(a: &Array) -> (Vec<i32>, Vec<f32>, Vec<u32>) {
    a.eval().unwrap();
    let shape = a.shape().as_slice().to_vec();
    match a.dtype() {
        Dtype::Uint32 => (shape, vec![], a.to_vec::<u32>().unwrap()),
        _ => (shape, a.to_vec::<f32>().unwrap(), vec![]),
    }
}

fn entry_arrays(entry: &PagedPrefixEntry) -> Vec<Array> {
    let mut tensors = HashMap::new();
    insert_entry_tensors(entry, &mut tensors);
    let mut names = tensors.keys().cloned().collect::<Vec<_>>();
    names.sort();
    names.into_iter().map(|n| tensors[&n].clone()).collect()
}

fn assert_same_entry(a: &PagedPrefixEntry, b: &PagedPrefixEntry, what: &str) {
    let (x, y) = (entry_arrays(a), entry_arrays(b));
    assert_eq!(x.len(), y.len(), "{what}");
    for (x, y) in x.iter().zip(&y) {
        assert_eq!(flat(x), flat(y), "{what}");
    }
}

fn whole(spec: &PagedPrefixKeySpec, entry: &SegmentedPrefixEntry) -> PagedPrefixEntry {
    materialize_prefix_entry(entry, spec.cached_len, spec.block_size).unwrap()
}

fn segment_path(root: &Path, spec: &PagedPrefixKeySpec, segment: &PrefixKvSegment) -> PathBuf {
    wp8_layout_dir(root, &prefix_segment_layout(spec))
        .join(WP8_SEGMENTS_DIR)
        .join(wp8_segment_file_name(
            segment.id,
            segment.start,
            segment.stop,
        ))
}

fn entry_dir(root: &Path, spec: &PagedPrefixKeySpec) -> PathBuf {
    PagedPrefixStore::new(root).wp8_entry_dir(spec)
}

fn directory_size(path: &Path) -> u64 {
    directory_disk_bytes(path).unwrap()
}

fn copy_tree(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for item in fs::read_dir(from).unwrap() {
        let item = item.unwrap();
        let target = to.join(item.file_name());
        if item.file_type().unwrap().is_dir() {
            copy_tree(&item.path(), &target);
        } else {
            fs::copy(item.path(), target).unwrap();
        }
    }
}

/// A (`[0, BS)` segment S + 5-token tail) and B (S + `[BS, 2*BS)` + `tail`).
struct Pair {
    spec_a: PagedPrefixKeySpec,
    a: SegmentedPrefixEntry,
    spec_b: PagedPrefixKeySpec,
    b: SegmentedPrefixEntry,
}

fn pair(kind: &str, tail: i32) -> Pair {
    let mut caches = live(kind, BS + 5, 0.0);
    let (spec_a, a) = export(kind, &caches, vec![], (0..BS + 5).collect());
    assert_eq!(a.segments.len(), 1);
    extend(&mut caches, BS - 5 + tail, 7.0);
    let (spec_b, b) = export(
        kind,
        &caches,
        vec![a.segments[0].clone()],
        (0..2 * BS + tail).collect(),
    );
    assert_eq!(b.segments.len(), 2);
    Pair {
        spec_a,
        a,
        spec_b,
        b,
    }
}

#[test]
#[serial_test::serial(mlx_metal)]
fn wp8_round_trip_equals_whole_copy() {
    for kind in ["paged", "turboquant"] {
        for tail in [0, 1, 127] {
            let root = temp_root("wp8-round-trip");
            let store = PagedPrefixStore::new(&root);
            let Pair {
                spec_a,
                a,
                spec_b,
                b,
            } = pair(kind, tail);
            store.save_segmented(&spec_a, &a).unwrap();
            store.save_segmented(&spec_b, &b).unwrap();

            let loaded = store.load_segmented_observed(&spec_b, None).unwrap();
            assert_eq!(loaded.status, PagedPrefixLoadStatus::Hit, "{kind} {tail}");
            let loaded_b = loaded.entry.expect("entry");
            assert_eq!(loaded_b.aligned_len, b.aligned_len);
            assert_eq!(
                loaded_b
                    .segments
                    .iter()
                    .map(|r| (r.segment.id, r.segment.start, r.stop))
                    .collect::<Vec<_>>(),
                b.segments
                    .iter()
                    .map(|r| (r.segment.id, r.segment.start, r.stop))
                    .collect::<Vec<_>>()
            );
            let what = format!("{kind} tail={tail}");
            assert_same_entry(&whole(&spec_b, &loaded_b), &whole(&spec_b, &b), &what);
            // The plain load path rebuilds the same entry.
            let plain = store.load(&spec_b).unwrap().expect("plain hit");
            assert_same_entry(&plain, &whole(&spec_b, &b), &what);

            // A segment the hot cache holds is reused, not read again.
            let s = a.segments[0].segment.clone();
            let lookup = |id: PrefixSegmentId| (id == s.id).then(|| s.clone());
            let reused = store
                .load_segmented_observed(&spec_b, Some(&lookup))
                .unwrap()
                .entry
                .expect("entry");
            assert!(Arc::ptr_eq(&reused.segments[0].segment, &s));
            assert!(!Arc::ptr_eq(
                &reused.segments[1].segment,
                &b.segments[1].segment
            ));
            assert_same_entry(&whole(&spec_b, &reused), &whole(&spec_b, &b), &what);
            fs::remove_dir_all(root).unwrap();
        }
    }
}

/// An SSD hit comes back evaluated, so inserting it into the hot cache
/// reads nothing from disk under the cache lock.
#[test]
#[serial_test::serial(mlx_metal)]
fn ssd_hit_is_evaluated_before_it_reaches_the_hot_cache() {
    let root = temp_root("wp8-hit-evaluated");
    let store = PagedPrefixStore::new(&root);
    let Pair {
        spec_a,
        a,
        spec_b,
        b,
    } = pair("paged", 1);
    store.save_segmented(&spec_a, &a).unwrap();
    store.save_segmented(&spec_b, &b).unwrap();

    let loaded = load_store_prefix_entry(&store, None, &spec_b).unwrap();
    let hit = loaded.hit.expect("hit");
    let segmented = format!("{:?}", hit.segmented);
    assert!(segmented.contains("evaluated: true"), "{segmented}");
    assert!(!segmented.contains("evaluated: false"), "{segmented}");
    fs::remove_dir_all(root).unwrap();
}

#[test]
#[serial_test::serial(mlx_metal)]
fn wp8_writes_only_missing_segments() {
    let root = temp_root("wp8-only-missing");
    let store = PagedPrefixStore::new(&root);
    let Pair {
        spec_a,
        a,
        spec_b,
        b,
    } = pair("paged", 9);
    let s = a.segments[0].segment.clone();
    let (_, written_a) = store.save_segmented_counted(&spec_a, &a).unwrap();
    assert_eq!(written_a, 1);
    let s_path = segment_path(&root, &spec_a, &s);
    let before = fs::metadata(&s_path).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(20));

    let (_, written_b) = store.save_segmented_counted(&spec_b, &b).unwrap();
    assert_eq!(written_b, 1, "only B's own segment is new");
    let after = fs::metadata(&s_path).unwrap();
    assert_eq!(before.ino(), after.ino());
    assert_eq!(before.modified().unwrap(), after.modified().unwrap());
    assert!(segment_path(&root, &spec_b, &b.segments[1].segment).is_file());

    // Saving B again writes nothing.
    let (_, again) = store.save_segmented_counted(&spec_b, &b).unwrap();
    assert_eq!(again, 0);
    // B's entry names S's file, which exists.
    let meta: serde_json::Value =
        serde_json::from_slice(&fs::read(entry_dir(&root, &spec_b).join(META_FILE)).unwrap())
            .unwrap();
    assert_eq!(meta["format"], "wp8-v1");
    assert_eq!(meta["segments"][0]["id"], s.id.to_string());
    fs::remove_dir_all(root).unwrap();
}

#[test]
#[serial_test::serial(mlx_metal)]
fn wp8_rewrites_a_segment_that_only_lived_in_memory() {
    let root = temp_root("wp8-memory-only-segment");
    let store = PagedPrefixStore::new(&root);
    // A was never written (backpressure); B references its segment S.
    let Pair { a, spec_b, b, .. } = pair("paged", 3);
    let s = a.segments[0].segment.clone();
    let (_, written) = store.save_segmented_counted(&spec_b, &b).unwrap();
    assert_eq!(written, 2, "S is written with B");
    assert!(segment_path(&root, &spec_b, &s).is_file());
    let loaded = store.load_segmented_observed(&spec_b, None).unwrap();
    assert_eq!(loaded.status, PagedPrefixLoadStatus::Hit);
    assert_same_entry(
        &whole(&spec_b, &loaded.entry.unwrap()),
        &whole(&spec_b, &b),
        "B",
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
#[serial_test::serial(mlx_metal)]
fn wp8_entry_with_missing_segment_is_a_miss_and_removed() {
    let root = temp_root("wp8-missing-segment");
    let store = PagedPrefixStore::new(&root);
    let Pair {
        spec_a,
        a,
        spec_b,
        b,
    } = pair("paged", 3);
    store.save_segmented(&spec_a, &a).unwrap();
    store.save_segmented(&spec_b, &b).unwrap();

    // S is gone: both A and B miss and are removed.
    fs::remove_file(segment_path(&root, &spec_a, &a.segments[0].segment)).unwrap();
    let missed = store.load_segmented_observed(&spec_b, None).unwrap();
    assert_eq!(missed.status, PagedPrefixLoadStatus::MissingEntry);
    assert!(!entry_dir(&root, &spec_b).exists());
    assert!(!store.contains(&spec_b).unwrap());
    assert_eq!(
        store.load_observed(&spec_a).unwrap().status,
        PagedPrefixLoadStatus::MissingEntry
    );
    assert!(!entry_dir(&root, &spec_a).exists());

    // A corrupt segment file is a miss too.
    let (spec_c, c) = export(
        "paged",
        &live("paged", BS + 2, 4.0),
        vec![],
        (500..500 + BS + 2).collect(),
    );
    store.save_segmented(&spec_c, &c).unwrap();
    fs::write(
        segment_path(&root, &spec_c, &c.segments[0].segment),
        b"not a safetensors file",
    )
    .unwrap();
    let missed = store.load_segmented_observed(&spec_c, None).unwrap();
    assert_eq!(missed.status, PagedPrefixLoadStatus::MissingEntry);
    assert!(!entry_dir(&root, &spec_c).exists());
    fs::remove_dir_all(root).unwrap();
}

#[test]
#[serial_test::serial(mlx_metal)]
fn wp8_disk_eviction_takes_covered_entries_first_and_collects_segments() {
    let root = temp_root("wp8-eviction");
    let store = PagedPrefixStore::new(&root);
    // C is unrelated and oldest; B continues A.
    let (spec_c, c) = export(
        "paged",
        &live("paged", BS + 2, 4.0),
        vec![],
        (500..500 + BS + 2).collect(),
    );
    store.save_segmented(&spec_c, &c).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(20));
    let Pair {
        spec_a,
        a,
        spec_b,
        b,
    } = pair("paged", 3);
    store.save_segmented(&spec_a, &a).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(20));
    let key_b = store.save_segmented(&spec_b, &b).unwrap();
    let s_path = segment_path(&root, &spec_a, &a.segments[0].segment);
    let t_path = segment_path(&root, &spec_c, &c.segments[0].segment);

    // One byte over: the covered entry A goes, not the older C; S stays
    // because B uses it.
    store
        .evict_to_disk_capacity(&key_b, directory_size(&root) as usize - 1)
        .unwrap();
    assert!(!entry_dir(&root, &spec_a).exists());
    assert!(entry_dir(&root, &spec_c).exists());
    assert!(s_path.is_file());

    // Next the oldest entry, C, goes with its segment T.
    store
        .evict_to_disk_capacity(&key_b, directory_size(&root) as usize - 1)
        .unwrap();
    assert!(!entry_dir(&root, &spec_c).exists());
    assert!(!t_path.exists(), "T has no entry left");
    assert!(s_path.is_file());
    assert_eq!(
        store.load_segmented_observed(&spec_b, None).unwrap().status,
        PagedPrefixLoadStatus::Hit
    );

    // A save through a capped store evicts the same way.
    let max_bytes = directory_size(&root) as usize + 1;
    let capped = PagedPrefixStore::new(&root).with_max_bytes(max_bytes);
    let (spec_d, d) = export(
        "paged",
        &live("paged", BS + 1, 9.0),
        vec![],
        (900..900 + BS + 1).collect(),
    );
    let key_d = capped.save_segmented(&spec_d, &d).unwrap();
    assert!(capped.wp8_entry_dir(&spec_d).ends_with(&key_d));
    assert!(!entry_dir(&root, &spec_b).exists(), "B was the oldest");
    assert!(!s_path.exists() && !segment_path(&root, &spec_b, &b.segments[1].segment).exists());
    assert!(directory_size(&root) <= max_bytes as u64);
    fs::remove_dir_all(root).unwrap();
}

#[test]
#[serial_test::serial(mlx_metal)]
fn wp8_startup_removes_orphan_segments_and_tmp_dirs() {
    let first = temp_root("wp8-startup-source");
    let Pair {
        spec_a,
        a,
        spec_b,
        b,
    } = pair("paged", 3);
    PagedPrefixStore::new(&first)
        .save_segmented(&spec_a, &a)
        .unwrap();
    // The same files as a fresh process finds them under another root.
    let root = temp_root("wp8-startup");
    copy_tree(&first, &root);
    let layout_dir = wp8_layout_dir(&root, &prefix_segment_layout(&spec_a));
    let s_path = segment_path(&root, &spec_a, &a.segments[0].segment);
    let orphan = layout_dir
        .join(WP8_SEGMENTS_DIR)
        .join(wp8_segment_file_name(PrefixSegmentId::new(), 0, BS));
    fs::copy(&s_path, &orphan).unwrap();
    let tmp_segment = layout_dir.join(WP8_SEGMENTS_DIR).join(".tmp-x.safetensors");
    fs::write(&tmp_segment, b"partial").unwrap();
    let tmp_entry = layout_dir.join(WP8_ENTRIES_DIR).join(".tmp-key-1");
    fs::create_dir_all(&tmp_entry).unwrap();
    let tmp_root_dir = root.join(".tmp-block-2");
    fs::create_dir_all(&tmp_root_dir).unwrap();
    let dflash2 = root.join(crate::core::dflash2_prefix_store::DFLASH2_STORE_DIR);
    fs::create_dir_all(dflash2.join(".tmp-own")).unwrap();

    // The first write to this root cleans up before writing.
    PagedPrefixStore::new(&root)
        .save_segmented(&spec_b, &b)
        .unwrap();
    assert!(!orphan.exists());
    assert!(!tmp_segment.exists());
    assert!(!tmp_entry.exists());
    assert!(!tmp_root_dir.exists());
    assert!(
        dflash2.join(".tmp-own").is_dir(),
        "dflash2 is never touched"
    );
    assert!(s_path.is_file(), "A still uses S");
    assert_eq!(
        PagedPrefixStore::new(&root)
            .load_segmented_observed(&spec_a, None)
            .unwrap()
            .status,
        PagedPrefixLoadStatus::Hit
    );
    fs::remove_dir_all(first).unwrap();
    fs::remove_dir_all(root).unwrap();
}

fn block_entry() -> (PagedPrefixKeySpec, PagedPrefixEntry) {
    let k_pages: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (1_i32, 1_i32, 2_i32, 2_i32))
        .try_into()
        .unwrap();
    let v_pages: Array = (&[5.0_f32, 6.0, 7.0, 8.0][..], (1_i32, 1_i32, 2_i32, 2_i32))
        .try_into()
        .unwrap();
    let entry = PagedPrefixEntry {
        main_layers: vec![PrefixLayerPayload::FullPaged { k_pages, v_pages }],
        mtp_layers: vec![],
        mtp_last_hidden: None,
        gemma4_drafter_last_hidden: None,
        last_logits: None,
    };
    let spec = PagedPrefixKeySpec {
        entry_kind: PrefixEntryKind::ImmutableBlock,
        model_id: "wp8-test".to_owned(),
        token_ids: vec![1, 2],
        cached_len: 2,
        fingerprint: Some("immutable-root".to_owned()),
        block_size: 2,
        kv_cache_profile: None,
        main_layers: entry.main_layer_specs(),
        mtp_layers: vec![],
        mtp_last_hidden: None,
        gemma4_drafter_last_hidden: None,
    };
    (spec, entry)
}

#[test]
#[serial_test::serial(mlx_metal)]
fn block_pool_entries_survive_the_new_format() {
    let root = temp_root("wp8-block-pool");
    let store = PagedPrefixStore::new(&root);
    let (block_spec, block) = block_entry();
    let block_key = store.save(&block_spec, &block).unwrap();
    assert!(root.join(&block_key).join(PAYLOAD_FILE).is_file());

    let Pair { spec_a, a, .. } = pair("paged", 3);
    store.save_segmented(&spec_a, &a).unwrap();
    assert!(store.contains_persisted(&block_spec).unwrap());
    let loaded = store.load_observed(&block_spec).unwrap();
    assert_eq!(loaded.status, PagedPrefixLoadStatus::Hit);
    assert_same_entry(&loaded.entry.unwrap(), &block, "block");
    assert_eq!(
        store.cached_lengths_descending(1 << 20).unwrap(),
        vec![spec_a.cached_len]
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
#[serial_test::serial(mlx_metal)]
fn legacy_whole_prefix_dirs_are_not_read_but_are_evicted() {
    let root = temp_root("wp8-legacy");
    let store = PagedPrefixStore::new(&root);
    let Pair { spec_a, a, .. } = pair("paged", 3);
    // Today's format of the same entry, as an older build wrote it.
    let legacy_key = store.save_root_entry(&spec_a, &whole(&spec_a, &a)).unwrap();
    let legacy_dir = root.join(&legacy_key);
    assert!(legacy_dir.join(META_FILE).is_file());
    assert_eq!(
        store.load_observed(&spec_a).unwrap().status,
        PagedPrefixLoadStatus::MissingEntry
    );
    assert!(!store.contains(&spec_a).unwrap());
    assert!(store.cached_lengths_descending(1 << 20).unwrap().is_empty());

    std::thread::sleep(std::time::Duration::from_millis(20));
    let (spec_c, c) = export(
        "paged",
        &live("paged", BS + 2, 4.0),
        vec![],
        (500..500 + BS + 2).collect(),
    );
    let key_c = store.save_segmented(&spec_c, &c).unwrap();
    store
        .evict_to_disk_capacity(&key_c, directory_size(&root) as usize - 1)
        .unwrap();
    assert!(!legacy_dir.exists(), "the legacy dir counts and is evicted");
    assert!(entry_dir(&root, &spec_c).exists());
    fs::remove_dir_all(root).unwrap();
}

#[test]
#[serial_test::serial(mlx_metal)]
fn wp8_async_queue_writes_segmented_entries() {
    let root = temp_root("wp8-async");
    let store = PagedPrefixStore::new(&root);
    let Pair {
        spec_a,
        a,
        spec_b,
        b,
    } = pair("paged", 3);
    let queue = AsyncPrefixStoreQueue::new(2, 1 << 30).expect("queue");
    for (spec, entry) in [(&spec_a, &a), (&spec_b, &b)] {
        let permit = match queue.try_admit(
            store.clone(),
            spec.clone(),
            AsyncPrefixStoreCancellation::default(),
        ) {
            AsyncPrefixStoreAdmission::Admitted(permit) => *permit,
            _ => panic!("admitted"),
        };
        // Bytes are reserved at submit, by what the writer puts on disk.
        assert_eq!(permit.payload_bytes, 0);
        assert_eq!(
            permit.submit_segmented(entry.clone()),
            AsyncPrefixStoreSubmit::Queued
        );
    }
    queue.wait_idle();
    assert_eq!(queue.stats().completed_total, 2);
    assert_eq!(queue.stats().pending_bytes, 0);
    let loaded = store.load_segmented_observed(&spec_b, None).unwrap();
    assert_eq!(loaded.status, PagedPrefixLoadStatus::Hit);
    assert_same_entry(
        &whole(&spec_b, &loaded.entry.unwrap()),
        &whole(&spec_b, &b),
        "B",
    );
    queue.shutdown();
    fs::remove_dir_all(root).unwrap();
}

/// An entry a reader deleted between the eviction scan and its removal
/// counts as removed, not as an eviction failure.
#[test]
fn eviction_treats_an_entry_removed_meanwhile_as_gone() {
    let root = temp_root("wp8-removed-meanwhile");
    let missing = root.join("gone");
    assert_eq!(directory_disk_bytes(&missing).unwrap(), 0);
    remove_entry_dir(&missing).unwrap();
    fs::remove_dir_all(root).unwrap();
}

/// A row that saved A (segment S = [0, 256)) and grew to 500 tokens, with
/// S's file on disk and its lineage recorded.
fn row_after_saving_a(
    kind: &str,
    root: &Path,
) -> (Vec<LayerCache>, Arc<PrefixKvSegment>, PrefixLineage) {
    let store = PagedPrefixStore::new(root);
    let mut caches = live(kind, 300, 0.1);
    let (spec_a, _) = export(kind, &caches, vec![], (0..300).collect());
    let (a, _) = export_segmented_prefix_for_row(None, &caches, 0, &spec_a)
        .unwrap()
        .unwrap();
    assert_eq!(a.segments.len(), 1);
    store.save_segmented(&spec_a, &a).unwrap();
    let lineage = segmented_entry_lineage(&a, prefix_segment_layout(&spec_a));
    ironmlx_lm::core::cache::layer::set_row_prefix_lineage(&mut caches, 0, Some(lineage.clone()));
    extend(&mut caches, 200, 0.7);
    (caches, a.segments[0].segment.clone(), lineage)
}

#[test]
#[serial_test::serial(mlx_metal)]
fn a_segment_the_hot_cache_dropped_is_rebuilt_from_the_row_under_its_id() {
    for kind in ["paged", "turboquant"] {
        for with_hot_cache in [false, true] {
            let root = temp_root("wp8-rebuild-dropped-segment");
            let store = PagedPrefixStore::new(&root);
            let (caches, s, _) = row_after_saving_a(kind, &root);
            // An empty hot cache: S was evicted (or never kept).
            let hot = with_hot_cache.then(|| {
                Arc::new(Mutex::new(
                    PrefixLruCache::new(PrefixLruCacheConfig::new(1 << 30).unwrap()).unwrap(),
                ))
            });
            let (spec_b, _) = export(kind, &caches, vec![], (0..500).collect());
            let (b, cached_len) =
                export_segmented_prefix_for_row(hot.as_ref(), &caches, 0, &spec_b)
                    .unwrap()
                    .unwrap();
            assert_eq!(cached_len, 500);
            assert_eq!(b.segments.len(), 2, "{kind}: S plus the new [256, 384)");
            let rebuilt = &b.segments[0].segment;
            assert_eq!(rebuilt.id, s.id, "{kind}: S keeps its id");
            assert_eq!((rebuilt.start, rebuilt.stop), (s.start, s.stop));
            assert_eq!(b.segments[1].segment.start, 256);

            let (_, written) = store.save_segmented_counted(&spec_b, &b).unwrap();
            assert_eq!(written, 1, "{kind}: S's file is reused");

            // Same content as a fresh whole export of the row.
            let (_, plain) = export(kind, &caches, vec![], (0..500).collect());
            let loaded = store.load_segmented_observed(&spec_b, None).unwrap();
            assert_eq!(loaded.status, PagedPrefixLoadStatus::Hit);
            assert_same_entry(
                &whole(&spec_b, &loaded.entry.unwrap()),
                &whole(&spec_b, &plain),
                kind,
            );
            // The rebuilt copy is S: the hot cache takes it as S.
            if let Some(hot) = &hot {
                hot.lock().unwrap().insert_segmented(spec_b, b).unwrap();
                assert_eq!(hot.lock().unwrap().segment(s.id).unwrap().stop, s.stop);
            }
            fs::remove_dir_all(root).unwrap();
        }
    }
}

#[test]
#[serial_test::serial(mlx_metal)]
fn a_dropped_segment_the_row_only_partly_holds_is_not_rebuilt() {
    let root = temp_root("wp8-partial-dropped-segment");
    let (mut caches, s, lineage) = row_after_saving_a("paged", &root);
    // The row took only S's head (as a restore of a shorter prompt does).
    let mut partial = lineage;
    partial.refs[0].stop = 128;
    ironmlx_lm::core::cache::layer::set_row_prefix_lineage(&mut caches, 0, Some(partial));
    let (spec_b, _) = export("paged", &caches, vec![], (0..500).collect());
    let (b, _) = export_segmented_prefix_for_row(None, &caches, 0, &spec_b)
        .unwrap()
        .unwrap();
    assert_eq!(b.segments.len(), 1, "one new segment from 0");
    assert_ne!(b.segments[0].segment.id, s.id);
    assert_eq!(b.segments[0].segment.start, 0);
    fs::remove_dir_all(root).unwrap();
}

/// One layer that keeps its whole history and one sliding-window layer
/// whose queries reach `WINDOW - 1` positions back, both holding `n` tokens.
const WINDOW: i32 = 200;

fn windowed_pair(n: i32) -> Vec<LayerCache> {
    let mut caches: Vec<LayerCache> = [0, WINDOW]
        .into_iter()
        .map(|window| {
            LayerCache::Full(
                KVCache::new(1, HEADS, DIM, DIM, Dtype::Float32, 8192)
                    .with_step(BS)
                    .with_window(window, 0),
            )
        })
        .collect();
    if n > 0 {
        extend(&mut caches, n, 0.1);
    }
    caches
}

fn windowed_spec(caches: &[LayerCache], len: i32) -> PagedPrefixKeySpec {
    let tokens: Vec<u32> = (0..len as u32).collect();
    ironmlx_lm::core::cache::layer::prefix_key_spec_for_caches(
        "wp8-window-test",
        &tokens,
        len,
        None,
        BS,
        caches,
    )
    .unwrap()
    .expect("a spec for dense layers")
}

/// What the next query of each layer attends to: the K the row returns
/// after appending one more token.
fn next_keys(caches: &mut [LayerCache]) -> Vec<(Vec<i32>, Vec<f32>, Vec<u32>)> {
    caches
        .iter_mut()
        .map(|cache| {
            let LayerCache::Full(kv) = cache else {
                unreachable!()
            };
            let (k, _) = kv
                .update_and_fetch(&token_data(1, 9.0), &token_data(1, 9.5), &[1])
                .unwrap();
            // A sliding-window layer may hold more than its window reaches.
            let held = k.shape().as_slice()[2];
            let reach = if kv.window() > 0 {
                held.min(kv.window())
            } else {
                held
            };
            flat(
                &mlx::ops::indexing::slice_strided_on(
                    &k,
                    [0, 0, held - reach, 0],
                    [1, HEADS, held, DIM],
                    [1, 1, 1, 1],
                    (),
                )
                .unwrap(),
            )
        })
        .collect()
}

#[test]
#[serial_test::serial(mlx_metal)]
fn sliding_window_layers_keep_only_their_window_and_share_it_between_saves() {
    let root = temp_root("wp8-window");
    let store = PagedPrefixStore::new(&root);
    let hot = Arc::new(Mutex::new(
        PrefixLruCache::new(PrefixLruCacheConfig::new(1 << 30).unwrap()).unwrap(),
    ));
    let window_segment = |entry: &SegmentedPrefixEntry, index: usize| {
        let r = &entry.window()[index];
        (r.segment.start, r.stop, r.segment.clone())
    };

    // A at 700 tokens: the window reaches back to 501, stored from 384.
    let mut caches = windowed_pair(700);
    let spec_a = windowed_spec(&caches, 700);
    assert_eq!(spec_a.main_layers[0].tensors[0].shape[2], 700);
    assert_eq!(spec_a.main_layers[1].tensors[0].shape[2], 700 - 384);
    let (a, len) = export_segmented_prefix_for_row(Some(&hot), &caches, 0, &spec_a)
        .unwrap()
        .unwrap();
    assert_eq!(
        (len, a.aligned_len, a.window_start, a.window_refs),
        (700, 640, 384, 1)
    );
    assert_eq!(a.head().len(), 1);
    let (start, stop, a_window) = window_segment(&a, 0);
    assert_eq!((start, stop), (384, 640));
    assert_eq!(a_window.layers.len(), 1, "only the sliding-window layer");
    assert_eq!(a.head()[0].segment.layers.len(), 1, "only the other layer");
    hot.lock()
        .unwrap()
        .insert_segmented(spec_a.clone(), a.clone())
        .unwrap();
    store.save_segmented(&spec_a, &a).unwrap();
    ironmlx_lm::core::cache::layer::set_row_prefix_lineage(
        &mut caches,
        0,
        Some(segmented_entry_lineage(&a, prefix_segment_layout(&spec_a))),
    );

    // B at 830: stored from 512, which A's window segment still covers up
    // to 640; only [640, 768) is new for either part.
    extend(&mut caches, 130, 0.7);
    let spec_b = windowed_spec(&caches, 830);
    let (b, _) = export_segmented_prefix_for_row(Some(&hot), &caches, 0, &spec_b)
        .unwrap()
        .unwrap();
    assert_eq!(
        (b.aligned_len, b.window_start, b.window_refs),
        (768, 512, 2)
    );
    assert!(
        Arc::ptr_eq(&window_segment(&b, 0).2, &a_window),
        "A's window is shared"
    );
    assert_eq!(
        (window_segment(&b, 1).0, window_segment(&b, 1).1),
        (640, 768)
    );
    assert!(Arc::ptr_eq(&b.head()[0].segment, &a.head()[0].segment));
    let (_, written) = store.save_segmented_counted(&spec_b, &b).unwrap();
    assert_eq!(
        written, 2,
        "one new head segment and one new window segment"
    );
    hot.lock()
        .unwrap()
        .insert_segmented(spec_b.clone(), b.clone())
        .unwrap();

    // Read back from disk and restored, B gives the next query of either
    // layer exactly what the live row gives it.
    let loaded = PagedPrefixStore::new(&root)
        .load_segmented_observed(&spec_b, None)
        .unwrap();
    assert_eq!(loaded.status, PagedPrefixLoadStatus::Hit);
    let loaded = loaded.entry.unwrap();
    assert_eq!((loaded.window_start, loaded.window_refs), (512, 2));
    for entry in [&b, &loaded] {
        let mut restored = windowed_pair(0);
        ironmlx_lm::core::cache::prefix_segments::restore_segmented_prefix_entry_for_rows(
            &mut restored,
            entry,
            &[0],
            830,
            BS,
        )
        .unwrap();
        let LayerCache::Full(kv) = &restored[1] else {
            unreachable!()
        };
        assert_eq!(
            (kv.offsets(), kv.bases()),
            (&[830][..], &[830 - (WINDOW - 1)][..])
        );
        let mut live_copy = windowed_pair(700);
        extend(&mut live_copy, 130, 0.7);
        assert_eq!(next_keys(&mut restored), next_keys(&mut live_copy));
    }

    // C far beyond: nothing of the earlier windows is referenced.
    ironmlx_lm::core::cache::layer::set_row_prefix_lineage(
        &mut caches,
        0,
        Some(segmented_entry_lineage(&b, prefix_segment_layout(&spec_b))),
    );
    extend(&mut caches, 1170, 1.3);
    let spec_c = windowed_spec(&caches, 2000);
    let (c, _) = export_segmented_prefix_for_row(Some(&hot), &caches, 0, &spec_c)
        .unwrap()
        .unwrap();
    assert_eq!(
        (c.aligned_len, c.window_start, c.window_refs),
        (1920, 1792, 1)
    );
    assert_eq!(
        c.head().len(),
        3,
        "the head still shares A's and B's segments"
    );
    assert_eq!(
        (window_segment(&c, 0).0, window_segment(&c, 0).1),
        (1792, 1920)
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
#[serial_test::serial(mlx_metal)]
fn checkpoints_of_sliding_window_entries_are_thinned_out() {
    let hot = Arc::new(Mutex::new(
        PrefixLruCache::new(PrefixLruCacheConfig::new(1 << 30).unwrap()).unwrap(),
    ));
    // A prefill that saves a checkpoint every 512 tokens up to 6144.
    let mut caches = windowed_pair(0);
    let mut specs = Vec::new();
    for step in 1..=12 {
        extend(&mut caches, 512, step as f32);
        let len = step * 512;
        let spec = windowed_spec(&caches, len);
        let (entry, _) = export_segmented_prefix_for_row(Some(&hot), &caches, 0, &spec)
            .unwrap()
            .unwrap();
        assert!(entry.window_refs > 0);
        ironmlx_lm::core::cache::layer::set_row_prefix_lineage(
            &mut caches,
            0,
            Some(segmented_entry_lineage(
                &entry,
                prefix_segment_layout(&spec),
            )),
        );
        hot.lock()
            .unwrap()
            .insert_segmented(spec.clone(), entry)
            .unwrap();
        specs.push(spec);
    }
    let held: Vec<i32> = specs
        .iter()
        .filter(|spec| hot.lock().unwrap().load_segmented(spec).unwrap().is_some())
        .map(|spec| spec.cached_len)
        .collect();
    // The three newest, and the shortest of each power-of-two length range
    // below them: [512, 1024), [1024, 2048), [2048, 4096), [4096, 8192).
    assert_eq!(held, vec![512, 1024, 2048, 4096, 5120, 5632, 6144]);
    // The head still reaches back to position 0 for every entry kept.
    let longest = hot
        .lock()
        .unwrap()
        .load_segmented(specs.last().unwrap())
        .unwrap()
        .unwrap()
        .0;
    assert_eq!(longest.head().first().unwrap().segment.start, 0);
    assert_eq!(longest.aligned_len, 6144);

    // Entries without sliding-window layers keep every checkpoint.
    let hot = Arc::new(Mutex::new(
        PrefixLruCache::new(PrefixLruCacheConfig::new(1 << 30).unwrap()).unwrap(),
    ));
    let mut caches = live("paged", 128, 0.1);
    let mut kept = 0;
    for step in 1..=6 {
        if step > 1 {
            extend(&mut caches, 128, step as f32);
        }
        let (spec, entry) = export("paged", &caches, vec![], (0..step * 128).collect());
        hot.lock()
            .unwrap()
            .insert_segmented(spec.clone(), entry)
            .unwrap();
        kept += usize::from(hot.lock().unwrap().load_segmented(&spec).unwrap().is_some());
    }
    assert_eq!(kept, 6);
    assert_eq!(hot.lock().unwrap().health().entry_count, 6);
}

#[test]
#[serial_test::serial(mlx_metal)]
fn stored_checkpoints_of_sliding_window_entries_are_thinned_out() {
    let root = temp_root("wp8-window-thin");
    let store = PagedPrefixStore::new(&root);
    // A prefill that stores a checkpoint every 512 tokens up to 6144.
    let mut caches = windowed_pair(0);
    let mut specs = Vec::new();
    for step in 1..=12 {
        extend(&mut caches, 512, step as f32);
        let spec = windowed_spec(&caches, step * 512);
        let (entry, _) = export_segmented_prefix_for_row(None, &caches, 0, &spec)
            .unwrap()
            .unwrap();
        ironmlx_lm::core::cache::layer::set_row_prefix_lineage(
            &mut caches,
            0,
            Some(segmented_entry_lineage(
                &entry,
                prefix_segment_layout(&spec),
            )),
        );
        store.save_segmented(&spec, &entry).unwrap();
        specs.push(spec);
    }
    let stored = |store: &PagedPrefixStore, spec: &PagedPrefixKeySpec| {
        store.load_segmented_observed(spec, None).unwrap().status == PagedPrefixLoadStatus::Hit
    };
    let held: Vec<i32> = specs
        .iter()
        .filter(|spec| stored(&store, spec))
        .map(|spec| spec.cached_len)
        .collect();
    assert_eq!(held, vec![512, 1024, 2048, 4096, 5120, 5632, 6144]);

    // Every head segment is still referenced (the longest entry reaches back
    // to 0); only the windows of the entries kept remain: 12 head files and
    // 7 window files.
    let layout_dir = wp8_layout_dir(&root, &prefix_segment_layout(specs.last().unwrap()));
    let segment_files = fs::read_dir(layout_dir.join(WP8_SEGMENTS_DIR))
        .unwrap()
        .count();
    assert_eq!(segment_files, 12 + 7);

    // The longest entry restores from a store opened afresh.
    let reopened = PagedPrefixStore::new(&root);
    let loaded = reopened
        .load_segmented_observed(specs.last().unwrap(), None)
        .unwrap()
        .entry
        .expect("the longest entry");
    let mut restored = windowed_pair(0);
    ironmlx_lm::core::cache::prefix_segments::restore_segmented_prefix_entry_for_rows(
        &mut restored,
        &loaded,
        &[0],
        6144,
        BS,
    )
    .unwrap();
    assert_eq!(next_keys(&mut restored), next_keys(&mut caches));
    fs::remove_dir_all(root).unwrap();
}

#[test]
#[serial_test::serial(mlx_metal)]
fn a_queued_save_reserves_only_what_it_writes() {
    let root = temp_root("wp8-write-bytes");
    let store = PagedPrefixStore::new(&root);
    let Pair {
        spec_a,
        a,
        spec_b,
        b,
    } = pair("paged", 3);
    store.save_segmented(&spec_a, &a).unwrap();
    // B shares A's first segment, which is on disk now: its write is the
    // new segment and its private tail, well under its logical size.
    let write = store.segmented_write_bytes(&spec_b, &b, &std::collections::HashSet::new());
    let shared = b.segments[0].segment.payload_bytes();
    let logical = spec_b.payload_bytes();
    assert!(
        write > 0 && write + shared <= logical,
        "{write} {shared} {logical}"
    );
    // A segment a save queued ahead writes first is not counted either.
    let queued: std::collections::HashSet<_> = b.segments.iter().map(|r| r.segment.id).collect();
    let private = b
        .private
        .observability_stats(spec_b.cached_len)
        .payload_bytes;
    assert_eq!(store.segmented_write_bytes(&spec_b, &b, &queued), private);
    // A limit B's logical size would break but its write fits, with
    // something else already reserved.
    let queue = AsyncPrefixStoreQueue::new(2, write + 1).expect("queue");
    assert!(queue.reserve_pending(1));
    let permit = match queue.try_admit(
        store.clone(),
        spec_b.clone(),
        AsyncPrefixStoreCancellation::default(),
    ) {
        AsyncPrefixStoreAdmission::Admitted(permit) => *permit,
        _ => panic!("admitted"),
    };
    assert_eq!(
        permit.submit_segmented(b.clone()),
        AsyncPrefixStoreSubmit::Queued
    );
    queue.release_pending(0, 1);
    queue.wait_idle();
    assert_eq!(queue.stats().completed_total, 1);
    assert_eq!(queue.stats().pending_bytes, 0);
    assert_eq!(
        store.load_segmented_observed(&spec_b, None).unwrap().status,
        PagedPrefixLoadStatus::Hit
    );
    queue.shutdown();
    fs::remove_dir_all(root).unwrap();
}

#[test]
#[serial_test::serial(mlx_metal)]
fn a_small_save_is_admitted_while_a_large_one_waits() {
    let queue = AsyncPrefixStoreQueue::new(4, 100).expect("queue");
    let store = PagedPrefixStore::new(temp_root("wp8-oversize-unused"));
    let Pair { spec_a, spec_b, .. } = pair("paged", 3);
    let admit = |spec: &PagedPrefixKeySpec| match queue.try_admit(
        store.clone(),
        spec.clone(),
        AsyncPrefixStoreCancellation::default(),
    ) {
        AsyncPrefixStoreAdmission::Admitted(permit) => *permit,
        _ => panic!("admitted"),
    };
    let mut large = admit(&spec_a);
    // Its write is past the limit: it is the queue's one oversize job.
    assert!(queue.reserve_pending_bytes(large.id, 1_000));
    large.payload_bytes = 1_000;
    // A second save is still admitted and its small write fits; a second
    // large one does not.
    let mut small = admit(&spec_b);
    assert!(!queue.reserve_pending_bytes(small.id, 1_000));
    assert!(queue.reserve_pending_bytes(small.id, 50));
    small.payload_bytes = 50;
    assert_eq!(queue.stats().pending_bytes, 1_050);
    drop(large);
    // Once the large one is gone, the next large write is taken again.
    let mut next = admit(&spec_a);
    assert!(queue.reserve_pending_bytes(next.id, 1_000));
    next.payload_bytes = 1_000;
    drop(next);
    drop(small);
    assert_eq!(
        (queue.stats().pending_jobs, queue.stats().pending_bytes),
        (0, 0)
    );
    queue.shutdown();
}

/// An entry's last logits are written with its private part and read back
/// under the same key; an entry without them loads without them.
#[test]
#[serial_test::serial(mlx_metal)]
fn wp8_keeps_the_last_logits_outside_the_key() {
    let root = temp_root("wp8-last-logits");
    let store = PagedPrefixStore::new(&root);
    let Pair {
        spec_a,
        a,
        spec_b,
        mut b,
    } = pair("paged", 1);
    let logits: Array = (&[0.25_f32, -1.5, 3.0][..], &[1_i32, 1, 3][..])
        .try_into()
        .unwrap();
    let key_before = PagedPrefixStore::key_for(&spec_b);
    b.private.last_logits = Some(logits.clone());
    store.save_segmented(&spec_a, &a).unwrap();
    store.save_segmented(&spec_b, &b).unwrap();
    assert_eq!(PagedPrefixStore::key_for(&spec_b), key_before);

    let loaded = store
        .load_segmented_observed(&spec_b, None)
        .unwrap()
        .entry
        .expect("entry");
    let read = loaded.private.last_logits.clone().expect("last logits");
    assert_eq!(flat(&read), flat(&logits));
    assert_same_entry(&whole(&spec_b, &loaded), &whole(&spec_b, &b), "with logits");

    let without = store
        .load_segmented_observed(&spec_a, None)
        .unwrap()
        .entry
        .expect("entry");
    assert!(without.private.last_logits.is_none());
    fs::remove_dir_all(root).unwrap();
}
