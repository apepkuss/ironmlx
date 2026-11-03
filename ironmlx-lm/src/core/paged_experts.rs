//! Read-only expert paging with bounded resident storage and explicit leases.
//! A cache entry cannot be evicted while a packed execution batch pins it.
//! Callers evaluate the resulting output before dropping the batch lease;
//! CPU readers and one bounded known-route prefetch never construct MLX arrays.
use super::{
    paging_stats::{elapsed_us, PagingTelemetry},
    tensor_file::{ModelTensorFiles, RawRows},
    weights::{QuantMeta, QuantMode},
};
use anyhow::{ensure, Context, Result};
use mlx::{Array, StreamOrDevice};
use std::{
    collections::{BTreeMap, HashMap},
    sync::{atomic::Ordering::Relaxed, mpsc, Arc, Mutex},
    time::Instant,
};

const ENTRY_OVERHEAD: usize = 2048;

#[derive(Clone)]
pub struct ExpertTableSpec {
    pub prefix: String,
    pub experts: u32,
    pub hidden: i32,
    pub intermediate: i32,
    pub quant: QuantMeta,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    struct Fixture(PathBuf);
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn fixture() -> (Fixture, Arc<ModelTensorFiles>, ExpertTableSpec) {
        let directory =
            std::env::temp_dir().join(format!("ironmlx-expert-cache-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&directory).unwrap();
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        let spec = ExpertTableSpec {
            prefix: "experts".into(),
            experts: 3,
            hidden: 64,
            intermediate: 32,
            quant: QuantMeta {
                group_size: 32,
                bits: 4,
                mode: QuantMode::Affine,
            },
        };
        for (projection, output, input) in [
            ("gate_proj", 32, 64),
            ("up_proj", 32, 64),
            ("down_proj", 64, 32),
        ] {
            for (suffix, dtype, columns, item) in [
                ("weight", "U32", input / 8, 4),
                ("scales", "BF16", input / 32, 2),
                ("biases", "BF16", input / 32, 2),
            ] {
                let start = payload.len();
                for expert in 0..3 {
                    for _ in 0..output * columns {
                        if item == 4 {
                            payload.extend_from_slice(&(0x11111111u32 * expert).to_le_bytes());
                        } else {
                            payload.extend_from_slice(
                                &(if suffix == "scales" { 0x3f80u16 } else { 0 }).to_le_bytes(),
                            );
                        }
                    }
                }
                header.insert(
                    format!("experts.{projection}.{suffix}"),
                    serde_json::json!({
                    "dtype":dtype,"shape":[3,output,columns],"data_offsets":[start,payload.len()]}),
                );
            }
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(payload);
        std::fs::write(directory.join("model.safetensors"), bytes).unwrap();
        let files = Arc::new(ModelTensorFiles::open(&directory).unwrap());
        (Fixture(directory), files, spec)
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn pinned_experts_block_eviction_and_release_allows_bounded_progress() {
        let (_fixture, files, spec) = fixture();
        let mut cache = PagedExperts::new(files, vec![spec], 5888, 16 * 1024, 3840).unwrap();
        assert_eq!(cache.max_batch_experts(0).unwrap(), 1);
        let pinned = cache.lease(0, 0, ().into()).unwrap();
        assert!(cache.lease(0, 1, ().into()).is_err());
        assert_eq!(cache.stats().charged_bytes, 5888);
        let hit = cache.lease(0, 0, ().into()).unwrap();
        drop(hit);
        drop(pinned);
        let next = cache.lease(0, 1, ().into()).unwrap();
        assert_eq!(next.arrays.gate_up[0].shape().as_slice(), [1, 64, 8]);
        assert!(next.arrays.gate_up[0]
            .to_vec::<u32>()
            .unwrap()
            .iter()
            .all(|&n| n == 0x11111111));
        assert_eq!(cache.stats().evictions, 1);
        assert_eq!(cache.stats().peak_charged_bytes, 5888);
        drop(next);
        let batch = cache.pack(0, &[2], ().into()).unwrap();
        assert!(cache.lease(0, 0, ().into()).is_err());
        drop(batch);
        assert!(cache.lease(0, 0, ().into()).is_ok());
        assert_eq!(cache.stats().charged_bytes, 5888);
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn rejects_bad_budgets_and_invalid_working_sets_before_reading_payload() {
        let (_fixture, files, spec) = fixture();
        assert!(PagedExperts::new(
            Arc::clone(&files),
            vec![spec.clone()],
            5887,
            16 * 1024,
            3840
        )
        .is_err());
        assert!(
            PagedExperts::new(Arc::clone(&files), vec![spec.clone()], 5888, 1024, 3840).is_err()
        );
        let mut cache =
            PagedExperts::new(files, vec![spec], 2 * 5888, 16 * 1024, 2 * 3840).unwrap();
        for ids in [&[][..], &[0, 0][..], &[0, 3][..], &[0, 1, 2][..]] {
            assert!(cache.pack(0, ids, ().into()).is_err());
        }
        assert_eq!(cache.stats().misses, 0);
        let batch = cache.pack(0, &[2, 0], ().into()).unwrap();
        assert_eq!(batch.arrays.down[0].shape().as_slice(), [2, 64, 4]);
        assert_eq!(cache.stats().resident_experts, 2);
        drop(batch);
        let source = cache.files.tensors().next().unwrap().0.to_owned();
        assert!(cache.files.read_rows(&source, 3, 1, 4096).is_err());
    }
    #[test]
    #[serial_test::serial(mlx_metal)]
    fn prefill_scans_preserve_decode_hot_experts_and_prefetch_is_bounded() {
        use crate::core::paging_stats::{PagingPhase, PhaseGuard};
        let (_fixture, files, spec) = fixture();
        let store = Arc::new(Mutex::new(
            PagedExperts::new(files, vec![spec], 2 * 5888, 16 * 1024, 3840).unwrap(),
        ));
        {
            let _phase = PhaseGuard::enter(PagingPhase::Decode);
            for _ in 0..2 {
                drop(PagedExperts::pack_shared(&store, 0, &[0], ().into()).unwrap());
            }
        }
        {
            let _phase = PhaseGuard::enter(PagingPhase::Prefill);
            for id in [1, 2, 1, 2] {
                drop(PagedExperts::pack_shared(&store, 0, &[id], ().into()).unwrap());
            }
        }
        let mut cache = store.lock().unwrap();
        assert!(
            cache.entries.contains_key(&(0, 0)),
            "a prefill scan flushed the decode hot entry"
        );
        cache.prefetch(0, 1).unwrap();
        cache.prefetch(0, 2).unwrap();
        assert_eq!(cache.pending.as_ref().unwrap().key, (0, 1));
        drop(cache);
        drop(PagedExperts::pack_shared(&store, 0, &[1], ().into()).unwrap());
        let cache = store.lock().unwrap();
        assert!(cache.pending.is_none());
        assert_eq!(
            cache
                .telemetry
                .experts
                .current()
                .prefetch_hits
                .load(Relaxed),
            1
        );
        assert!(cache.stats().charged_bytes <= 2 * 5888);
    }
}

#[derive(Clone)]
pub struct ExpertArrays {
    pub gate_up: [Array; 3],
    pub down: [Array; 3],
}
pub struct ExpertLease {
    pub arrays: ExpertArrays,
    _pin: Arc<()>,
}
pub struct ExpertBatch {
    pub arrays: ExpertArrays,
    _leases: Vec<ExpertLease>,
}
struct Entry {
    arrays: ExpertArrays,
    pin: Arc<()>,
    age: u64,
    charge: usize,
    protected: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ExpertCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub charged_bytes: usize,
    pub peak_charged_bytes: usize,
    pub resident_experts: usize,
}

pub struct PagedExperts {
    files: Arc<ModelTensorFiles>,
    tables: Vec<ExpertTableSpec>,
    charges: Vec<usize>,
    entries: HashMap<(usize, u32), Entry>,
    ages: BTreeMap<u64, (usize, u32)>,
    clock: u64,
    cache_bytes: usize,
    staging_bytes: usize,
    pack_bytes: usize,
    protected_bytes: usize,
    stats: ExpertCacheStats,
    telemetry: Arc<PagingTelemetry>,
    pending: Option<PendingExpert>,
    prefetch_enabled: bool,
}

impl PagedExperts {
    pub fn new(
        files: Arc<ModelTensorFiles>,
        tables: Vec<ExpertTableSpec>,
        cache_bytes: usize,
        staging_bytes: usize,
        pack_bytes: usize,
    ) -> Result<Self> {
        ensure!(
            !tables.is_empty(),
            "expert paging requires at least one adapted table"
        );
        let mut charges = Vec::new();
        for table in &tables {
            ensure!(
                table.experts > 0
                    && table.hidden > 0
                    && table.intermediate > 0
                    && table.quant.mode == QuantMode::Affine
                    && table.quant.bits == 4
                    && matches!(table.quant.group_size, 32 | 64 | 128)
                    && table.hidden % table.quant.group_size == 0
                    && table.intermediate % table.quant.group_size == 0,
                "unsupported paged expert quantization/geometry"
            );
            let mut bytes = 0usize;
            for (projection, output, input) in [
                ("gate_proj", table.intermediate, table.hidden),
                ("up_proj", table.intermediate, table.hidden),
                ("down_proj", table.hidden, table.intermediate),
            ] {
                for (suffix, dtype, columns) in [
                    ("weight", "U32", input / 8),
                    ("scales", "BF16", input / table.quant.group_size),
                    ("biases", "BF16", input / table.quant.group_size),
                ] {
                    let name = format!("{}.{}.{}", table.prefix, projection, suffix);
                    let info = files
                        .info(&name)
                        .with_context(|| format!("missing expert tensor {name}"))?;
                    ensure!(
                        info.dtype == dtype
                            && info.shape == [table.experts as u64, output as u64, columns as u64],
                        "paged expert checkpoint shape/dtype mismatch at {name}"
                    );
                    bytes = bytes
                        .checked_add(usize::try_from(info.bytes() / u64::from(table.experts))?)
                        .context("expert residency size overflow")?;
                }
            }
            let charge = bytes
                .checked_add(ENTRY_OVERHEAD)
                .context("expert entry charge overflow")?;
            ensure!(
                charge <= cache_bytes && bytes <= pack_bytes,
                "expert cache/pack reservation cannot fit one expert"
            );
            // Includes all raw projection rows, their fused replacement and
            // the safetensors decoder envelope while one projection loads.
            ensure!(
                bytes
                    .checked_mul(4)
                    .is_some_and(|peak| peak <= staging_bytes),
                "expert I/O staging reservation is below a complete bounded load"
            );
            charges.push(charge);
        }
        Ok(Self {
            files,
            tables,
            charges,
            entries: HashMap::new(),
            ages: BTreeMap::new(),
            clock: 0,
            cache_bytes,
            staging_bytes,
            pack_bytes,
            protected_bytes: 0,
            stats: ExpertCacheStats::default(),
            telemetry: Arc::new(PagingTelemetry::new(true)),
            pending: None,
            prefetch_enabled: match std::env::var("IRONMLX_FLASH_PREFETCH")
                .unwrap_or_else(|_| "auto".into())
                .as_str()
            {
                "auto" => true,
                "off" => false,
                _ => anyhow::bail!("IRONMLX_FLASH_PREFETCH must be auto or off"),
            },
        })
    }

    pub fn with_telemetry(mut self, telemetry: Arc<PagingTelemetry>) -> Self {
        self.telemetry = telemetry;
        self.publish();
        self
    }
    pub fn telemetry(&self) -> &Arc<PagingTelemetry> {
        &self.telemetry
    }
    fn publish(&self) {
        let t = &self.telemetry.experts;
        t.budget_bytes.store(self.cache_bytes as u64, Relaxed);
        t.charged_bytes
            .store(self.stats.charged_bytes as u64, Relaxed);
        t.payload_bytes.store(
            self.entries
                .values()
                .map(|e| e.charge - ENTRY_OVERHEAD)
                .sum::<usize>() as u64,
            Relaxed,
        );
        t.peak_charged_bytes
            .fetch_max(self.stats.peak_charged_bytes as u64, Relaxed);
        t.entries.store(self.entries.len() as u64, Relaxed);
        t.pinned_bytes.store(
            self.entries
                .values()
                .filter(|e| Arc::strong_count(&e.pin) > 1)
                .map(|e| e.charge)
                .sum::<usize>() as u64,
            Relaxed,
        );
    }
    pub fn finished_batch(&self) {
        self.publish();
    }
    fn requests(&self, table: usize, expert: u32) -> Vec<(String, u64, u64)> {
        ["gate_proj", "up_proj", "down_proj"]
            .into_iter()
            .flat_map(|projection| {
                ["weight", "scales", "biases"]
                    .into_iter()
                    .map(move |suffix| {
                        (
                            format!("{}.{projection}.{suffix}", self.tables[table].prefix),
                            expert as u64,
                            1,
                        )
                    })
            })
            .collect()
    }
    /// One known next expert, never a speculative model graph. A single raw
    /// expert fits the already reserved staging area while GPU consumes a pack.
    pub fn prefetch(&mut self, table: usize, expert: u32) -> Result<()> {
        ensure!(
            table < self.tables.len() && expert < self.tables[table].experts,
            "invalid prefetch expert"
        );
        if !self.prefetch_enabled
            || self.pending.is_some()
            || self.entries.contains_key(&(table, expert))
        {
            return Ok(());
        }
        let files = Arc::clone(&self.files);
        let requests = self.requests(table, expert);
        let budget = self.staging_bytes;
        let telemetry = Arc::clone(&self.telemetry);
        let phase = crate::core::paging_stats::current_phase();
        telemetry
            .experts
            .phase(phase)
            .prefetch_scheduled
            .fetch_add(1, Relaxed);
        telemetry
            .experts
            .pending_bytes
            .store((self.charges[table] - ENTRY_OVERHEAD) as u64, Relaxed);
        let receiver = crate::core::tensor_file::io_task(move || {
            let start = Instant::now();
            let mut rows = Vec::new();
            let mut bytes = 0usize;
            for (name, start, count) in requests {
                // Sequential inside a reader: never recursively submit to the pool.
                let row = files.read_raw_rows(&name, start, count, budget)?;
                bytes += row.data.len();
                ensure!(bytes <= budget, "prefetch staging exceeded");
                rows.push(row);
            }
            let read_us = elapsed_us(start);
            telemetry
                .experts
                .phase(phase)
                .read_bytes
                .fetch_add(bytes as u64, Relaxed);
            telemetry
                .experts
                .phase(phase)
                .read_us
                .fetch_add(read_us, Relaxed);
            Ok((rows, read_us))
        });
        self.pending = Some(PendingExpert {
            key: (table, expert),
            receiver: Some(receiver),
            telemetry: Arc::clone(&self.telemetry),
            phase,
        });
        Ok(())
    }

    pub fn stats(&self) -> ExpertCacheStats {
        ExpertCacheStats {
            resident_experts: self.entries.len(),
            ..self.stats
        }
    }

    pub fn max_batch_experts(&self, table: usize) -> Result<usize> {
        let charge = *self
            .charges
            .get(table)
            .context("expert table index out of bounds")?;
        Ok((self.cache_bytes / charge).min(self.pack_bytes / (charge - ENTRY_OVERHEAD)))
    }

    pub fn lease(
        &mut self,
        table: usize,
        expert: u32,
        target: StreamOrDevice,
    ) -> Result<ExpertLease> {
        let spec = self
            .tables
            .get(table)
            .context("expert table index out of bounds")?;
        ensure!(expert < spec.experts, "expert index out of bounds");
        self.files.validate_sources()?;
        let lease = self.lease_validated(table, expert, target)?;
        self.files.validate_sources()?;
        Ok(lease)
    }
    fn lease_validated(
        &mut self,
        table: usize,
        expert: u32,
        target: StreamOrDevice,
    ) -> Result<ExpertLease> {
        self.lease_loaded(table, expert, target, None)
    }
    fn lease_loaded(
        &mut self,
        table: usize,
        expert: u32,
        target: StreamOrDevice,
        loaded: Option<ExpertArrays>,
    ) -> Result<ExpertLease> {
        self.tick();
        let key = (table, expert);
        if let Some(entry) = self.entries.get_mut(&key) {
            // A scan hit must not change the decode recency/frequency history.
            if !crate::core::paging_stats::is_prefill() {
                self.ages.remove(&entry.age);
                entry.age = self.clock;
                if !entry.protected {
                    self.protected_bytes += entry.charge;
                }
                entry.protected = true;
                self.ages.insert(self.clock, key);
            }
            self.stats.hits = self.stats.hits.saturating_add(1);
            self.telemetry.experts.current().hits.fetch_add(1, Relaxed);
            return Ok(ExpertLease {
                arrays: entry.arrays.clone(),
                _pin: Arc::clone(&entry.pin),
            });
        }
        let charge = self.charges[table];
        if self.protected_bytes > self.cache_bytes / 4 * 3 {
            for key in self.ages.values() {
                let entry = self.entries.get_mut(key).expect("indexed entry");
                if entry.protected {
                    entry.protected = false;
                    self.protected_bytes -= entry.charge;
                }
                if self.protected_bytes <= self.cache_bytes / 4 * 3 {
                    break;
                }
            }
        }
        while self.stats.charged_bytes > self.cache_bytes - charge {
            let victim = [false, true].into_iter().find_map(|protected| {
                self.ages.iter().find_map(|(&age, &key)| {
                    let entry = &self.entries[&key];
                    (entry.protected == protected && Arc::strong_count(&entry.pin) == 1)
                        .then_some((age, key))
                })
            });
            let (age, key) = victim
                .context("expert cache is pinned by in-flight execution; finish/chunk the batch")?;
            self.ages.remove(&age);
            let removed = self.entries.remove(&key).expect("indexed cache entry");
            self.stats.charged_bytes -= removed.charge;
            if removed.protected {
                self.protected_bytes -= removed.charge;
            }
            self.stats.evictions = self.stats.evictions.saturating_add(1);
            self.telemetry
                .experts
                .current()
                .evictions
                .fetch_add(1, Relaxed);
            drop(removed);
        }
        let arrays = match loaded {
            Some(arrays) => arrays,
            None => self.load(table, expert, target)?,
        };
        let pin = Arc::new(());
        let lease = ExpertLease {
            arrays: arrays.clone(),
            _pin: Arc::clone(&pin),
        };
        self.entries.insert(
            key,
            Entry {
                arrays,
                pin,
                age: self.clock,
                charge,
                protected: false,
            },
        );
        self.ages.insert(self.clock, key);
        self.stats.charged_bytes += charge;
        self.stats.peak_charged_bytes = self.stats.peak_charged_bytes.max(self.stats.charged_bytes);
        self.stats.misses = self.stats.misses.saturating_add(1);
        self.telemetry
            .experts
            .current()
            .misses
            .fetch_add(1, Relaxed);
        if crate::core::paging_stats::is_prefill() {
            self.telemetry
                .experts
                .current()
                .scan_admissions
                .fetch_add(1, Relaxed);
        }
        Ok(lease)
    }

    /// `experts` must be unique; order defines the compact gather indices.
    pub fn pack(
        &mut self,
        table: usize,
        experts: &[u32],
        target: StreamOrDevice,
    ) -> Result<ExpertBatch> {
        let spec = self
            .tables
            .get(table)
            .context("expert table index out of bounds")?;
        let mut seen = std::collections::HashSet::new();
        ensure!(
            !experts.is_empty()
                && experts.len() <= self.max_batch_experts(table)?
                && experts
                    .iter()
                    .all(|&id| id < spec.experts && seen.insert(id)),
            "invalid or oversized compact expert working set"
        );
        self.files.validate_sources()?;
        let pack_start = Instant::now();
        let mut leases = Vec::with_capacity(experts.len());
        for &expert in experts {
            leases.push(self.lease_validated(table, expert, target)?);
        }
        let combine = |gate_up: bool, index: usize| -> Result<Array> {
            let rows: Vec<&Array> = leases
                .iter()
                .map(|lease| {
                    if gate_up {
                        &lease.arrays.gate_up[index]
                    } else {
                        &lease.arrays.down[index]
                    }
                })
                .collect();
            Ok(if rows.len() == 1 {
                rows[0].clone()
            } else {
                mlx::ops::shape::concatenate_on(&rows, 0, target)?
            })
        };
        let arrays = ExpertArrays {
            gate_up: [combine(true, 0)?, combine(true, 1)?, combine(true, 2)?],
            down: [combine(false, 0)?, combine(false, 1)?, combine(false, 2)?],
        };
        let refs: Vec<&Array> = arrays.gate_up.iter().chain(&arrays.down).collect();
        mlx::transforms::eval(&refs)?;
        self.files.validate_sources()?;
        self.telemetry
            .experts
            .current()
            .pack_us
            .fetch_add(elapsed_us(pack_start), Relaxed);
        self.publish();
        Ok(ExpertBatch {
            arrays,
            _leases: leases,
        })
    }

    fn prepare_load(&mut self, table: usize, expert: u32) -> PreparedExpertLoad {
        let pending = if self
            .pending
            .as_ref()
            .is_some_and(|p| p.key == (table, expert))
        {
            self.pending.take()
        } else {
            None
        };
        PreparedExpertLoad {
            files: Arc::clone(&self.files),
            requests: self.requests(table, expert),
            staging_bytes: self.staging_bytes,
            charge: self.charges[table],
            telemetry: Arc::clone(&self.telemetry),
            pending,
        }
    }
    fn load(&mut self, table: usize, expert: u32, target: StreamOrDevice) -> Result<ExpertArrays> {
        self.prepare_load(table, expert).run(target)
    }
    /// Loading and MLX materialization happen outside the metadata lock. Raw
    /// and decoded replacement overlap is bounded by the staging reservation.
    pub fn pack_shared(
        store: &Arc<Mutex<Self>>,
        table: usize,
        experts: &[u32],
        target: StreamOrDevice,
    ) -> Result<ExpertBatch> {
        let started = Instant::now();
        let (files, telemetry) = {
            let cache = store
                .lock()
                .map_err(|_| anyhow::anyhow!("expert cache lock poisoned"))?;
            let spec = cache
                .tables
                .get(table)
                .context("expert table out of bounds")?;
            let mut unique = std::collections::HashSet::new();
            ensure!(
                !experts.is_empty()
                    && experts.len() <= cache.max_batch_experts(table)?
                    && experts
                        .iter()
                        .all(|&id| id < spec.experts && unique.insert(id)),
                "invalid or oversized compact expert working set"
            );
            (Arc::clone(&cache.files), Arc::clone(&cache.telemetry))
        };
        files.validate_sources()?;
        let mut leases = Vec::new();
        for &expert in experts {
            let waiting = Instant::now();
            let prepared = {
                let mut cache = store
                    .lock()
                    .map_err(|_| anyhow::anyhow!("expert cache lock poisoned"))?;
                telemetry
                    .experts
                    .current()
                    .lock_wait_us
                    .fetch_add(elapsed_us(waiting), Relaxed);
                if cache.entries.contains_key(&(table, expert)) {
                    leases.push(cache.lease_validated(table, expert, target)?);
                    None
                } else {
                    Some(cache.prepare_load(table, expert))
                }
            };
            if let Some(prepared) = prepared {
                let arrays = prepared.run(target)?;
                let mut cache = store
                    .lock()
                    .map_err(|_| anyhow::anyhow!("expert cache lock poisoned"))?;
                leases.push(cache.lease_loaded(table, expert, target, Some(arrays))?);
            }
        }
        let combine = |gate_up: bool, index: usize| -> Result<Array> {
            let rows: Vec<_> = leases
                .iter()
                .map(|l| {
                    if gate_up {
                        &l.arrays.gate_up[index]
                    } else {
                        &l.arrays.down[index]
                    }
                })
                .collect();
            Ok(if rows.len() == 1 {
                rows[0].clone()
            } else {
                mlx::ops::shape::concatenate_on(&rows, 0, target)?
            })
        };
        let arrays = ExpertArrays {
            gate_up: [combine(true, 0)?, combine(true, 1)?, combine(true, 2)?],
            down: [combine(false, 0)?, combine(false, 1)?, combine(false, 2)?],
        };
        mlx::transforms::eval(
            &arrays
                .gate_up
                .iter()
                .chain(&arrays.down)
                .collect::<Vec<_>>(),
        )?;
        files.validate_sources()?;
        telemetry
            .experts
            .current()
            .pack_us
            .fetch_add(elapsed_us(started), Relaxed);
        store
            .lock()
            .map_err(|_| anyhow::anyhow!("expert cache lock poisoned"))?
            .publish();
        Ok(ExpertBatch {
            arrays,
            _leases: leases,
        })
    }

    fn tick(&mut self) {
        if self.clock == u64::MAX {
            let order: Vec<_> = self.ages.values().copied().collect();
            self.ages.clear();
            for (i, key) in order.into_iter().enumerate() {
                let age = i as u64 + 1;
                self.entries.get_mut(&key).unwrap().age = age;
                self.ages.insert(age, key);
            }
            self.clock = self.entries.len() as u64;
        }
        self.clock += 1;
    }
}

type PrefetchResult = Result<(Vec<RawRows>, u64)>;
struct PendingExpert {
    key: (usize, u32),
    receiver: Option<mpsc::Receiver<PrefetchResult>>,
    telemetry: Arc<PagingTelemetry>,
    phase: crate::core::paging_stats::PagingPhase,
}
impl Drop for PendingExpert {
    fn drop(&mut self) {
        if let Some(receiver) = self.receiver.take() {
            let _ = receiver.recv();
            self.telemetry
                .experts
                .phase(self.phase)
                .prefetch_unused
                .fetch_add(1, Relaxed);
        }
        self.telemetry.experts.pending_bytes.store(0, Relaxed);
    }
}

struct PreparedExpertLoad {
    files: Arc<ModelTensorFiles>,
    requests: Vec<(String, u64, u64)>,
    staging_bytes: usize,
    charge: usize,
    telemetry: Arc<PagingTelemetry>,
    pending: Option<PendingExpert>,
}
impl PreparedExpertLoad {
    fn run(mut self, target: StreamOrDevice) -> Result<ExpertArrays> {
        let started = Instant::now();
        let pending = self.pending.take();
        let prefetched = pending.is_some();
        let (raw, read_us) = if let Some(mut pending) = pending {
            self.telemetry
                .experts
                .current()
                .prefetch_hits
                .fetch_add(1, Relaxed);
            pending
                .receiver
                .take()
                .expect("pending receiver")
                .recv()
                .context("prefetch reader stopped")??
        } else {
            let raw = self
                .files
                .read_rows_batch(&self.requests, self.staging_bytes)?;
            (raw, elapsed_us(started))
        };
        if !prefetched {
            self.telemetry
                .experts
                .current()
                .read_bytes
                .fetch_add(raw.iter().map(|r| r.data.len() as u64).sum(), Relaxed);
            self.telemetry
                .experts
                .current()
                .read_us
                .fetch_add(read_us, Relaxed);
        }
        let mut arrays = raw
            .into_iter()
            .map(|r| r.into_array(self.staging_bytes))
            .collect::<Result<Vec<_>>>()?
            .into_iter();
        let mut gate_up = Vec::new();
        let gate: Vec<_> = arrays.by_ref().take(3).collect();
        let up: Vec<_> = arrays.by_ref().take(3).collect();
        for (gate, up) in gate.iter().zip(&up) {
            let fused = mlx::ops::shape::concatenate_on(&[gate, up], 1, target)?;
            fused.eval()?;
            gate_up.push(fused);
        }
        let arrays = ExpertArrays {
            gate_up: gate_up
                .try_into()
                .map_err(|_| anyhow::anyhow!("missing gate rows"))?,
            down: arrays
                .collect::<Vec<_>>()
                .try_into()
                .map_err(|_| anyhow::anyhow!("missing down rows"))?,
        };
        let bytes: usize = arrays
            .gate_up
            .iter()
            .chain(&arrays.down)
            .map(|a| a.size() * a.dtype().byte_size())
            .sum();
        ensure!(
            bytes == self.charge - ENTRY_OVERHEAD,
            "expert load disagrees with residency accounting"
        );
        Ok(arrays)
    }
}
