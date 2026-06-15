//! B1-p2.3a scheduler skeleton — end-to-end admit/evict integration test.
//!
//! No model load, no GPU work. Exercises the scheduler API across a
//! realistic admit / evict / re-admit sequence to verify the skeleton
//! is internally consistent before B1-p2.3b layers the forward pass on
//! top of it.
//!
//! Run with:
//!   cargo test -p ironmlx --release --test b1_p2_3a_scheduler_skeleton

use ironmlx::core::generate::GenerateRequest;
use ironmlx::core::sampler::Sampler;
use ironmlx::core::scheduler::{RequestId, Scheduler};

/// Build a minimal `GenerateRequest` for a synthetic prompt of length `n`.
fn mk_req(seed: u32, n: usize) -> GenerateRequest {
    let prompt: Vec<u32> = (0..n as u32).map(|i| seed.wrapping_add(i)).collect();
    GenerateRequest {
        prompt_ids: prompt,
        max_new_tokens: 32,
        sampler: Sampler::greedy(),
        stop_token_ids: vec![2],
        prefill_chunk_size: 0,
        decode_cadence_mid_chunk_cap: 256,
        kv_cache_turboquant_bits: None,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: 248056,
        #[cfg(feature = "p5h-profile")]
        p5h_trace: None,
        #[cfg(feature = "p5h-profile")]
        p5h_root_span: None,
    }
}

#[test]
fn b1_p2_3a_admit_evict_sequence() {
    let mut s = Scheduler::<ironmlx::models::Qwen35Model>::new(
        4,
        32768,
        ironmlx::core::memory_budget::test_meta_qwen35(),
    )
    .expect("scheduler startup");

    // 1. Admit 4 mock requests; verify monotonic ids 0..3 and row_idx 0..3.
    let mut ids: Vec<RequestId> = Vec::new();
    for i in 0..4 {
        let id = s.admit(mk_req(100 * (i + 1) as u32, 8 + i)).expect("admit");
        let state = s.get(id).expect("get");
        assert_eq!(state.row_idx, i as usize);
        assert_eq!(state.real_len, (8 + i) as i32);
        assert_eq!(state.prompt_ids.len(), 8 + i);
        ids.push(id);
    }
    assert_eq!(s.active_count(), 4);
    assert_eq!(s.occupied_rows(), vec![0, 1, 2, 3]);

    // 2. Evict id=1 and id=3.
    s.evict(ids[1]).expect("evict 1");
    s.evict(ids[3]).expect("evict 3");
    assert_eq!(s.active_count(), 2);
    assert_eq!(s.occupied_rows(), vec![0, 2]);
    let actives = s.active();
    assert_eq!(actives.len(), 2);
    let rows: Vec<usize> = actives.iter().map(|r| r.row_idx).collect();
    assert_eq!(rows, vec![0, 2]);

    // 3. Admit a fifth request; verify it reuses row 1.
    let id_5 = s.admit(mk_req(500, 12)).expect("admit 5");
    assert_eq!(s.get(id_5).unwrap().row_idx, 1);
    assert!(id_5.0 > ids[3].0); // monotonic across evict

    // 4. Admit a sixth; verify it reuses row 3.
    let id_6 = s.admit(mk_req(600, 14)).expect("admit 6");
    assert_eq!(s.get(id_6).unwrap().row_idx, 3);
    assert_eq!(s.active_count(), 4);

    // 5. Admit a seventh; verify Err (b_max full).
    let err = s.admit(mk_req(700, 16)).expect_err("admit 7 must fail");
    let msg = format!("{err}");
    assert!(msg.contains("scheduler full"), "unexpected err: {msg}");
    assert!(msg.contains("b_max=4"), "missing b_max in err: {msg}");

    // 6. Evicting an already-evicted id returns Err.
    let err = s.evict(ids[1]).expect_err("evict id=1 again must fail");
    assert!(format!("{err}").contains("not found"));

    // 7. Final state: 4 active rows, distinct ids.
    assert_eq!(s.active_count(), 4);
    let final_ids: Vec<u64> = s.active().iter().map(|r| r.id.0).collect();
    // active() returns rows in slot order (0..b_max), filtering occupied:
    //   slot 0 → id 0 (original, never evicted)
    //   slot 1 → id 4 (re-admitted after evict of ids[1]=1)
    //   slot 2 → id 2 (original, never evicted)
    //   slot 3 → id 5 (re-admitted after evict of ids[3]=3)
    // next_id advances on admit only, so counter went 0,1,2,3,4,5 across the 6 admits.
    assert_eq!(final_ids, vec![0, 4, 2, 5]);
}
