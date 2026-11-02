//! Read-only expert paging with bounded resident storage and explicit leases.
//! A cache entry cannot be evicted while a packed execution batch pins it.
//! Callers evaluate the resulting output before dropping the batch lease;
//! asynchronous prefetch is intentionally not part of this initial path.
use super::{
    tensor_file::ModelTensorFiles,
    weights::{QuantMeta, QuantMode},
};
use anyhow::{ensure, Context, Result};
use mlx::{Array, StreamOrDevice};
use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
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
    stats: ExpertCacheStats,
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
            stats: ExpertCacheStats::default(),
        })
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
        self.tick();
        let key = (table, expert);
        if let Some(entry) = self.entries.get_mut(&key) {
            self.ages.remove(&entry.age);
            entry.age = self.clock;
            self.ages.insert(self.clock, key);
            self.stats.hits = self.stats.hits.saturating_add(1);
            return Ok(ExpertLease {
                arrays: entry.arrays.clone(),
                _pin: Arc::clone(&entry.pin),
            });
        }
        let charge = self.charges[table];
        while self.stats.charged_bytes > self.cache_bytes - charge {
            let victim = self.ages.iter().find_map(|(&age, &key)| {
                (Arc::strong_count(&self.entries[&key].pin) == 1).then_some((age, key))
            });
            let (age, key) = victim
                .context("expert cache is pinned by in-flight execution; finish/chunk the batch")?;
            self.ages.remove(&age);
            let removed = self.entries.remove(&key).expect("indexed cache entry");
            self.stats.charged_bytes -= removed.charge;
            self.stats.evictions = self.stats.evictions.saturating_add(1);
            drop(removed);
        }
        let arrays = self.load(table, expert, target)?;
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
            },
        );
        self.ages.insert(self.clock, key);
        self.stats.charged_bytes += charge;
        self.stats.peak_charged_bytes = self.stats.peak_charged_bytes.max(self.stats.charged_bytes);
        self.stats.misses = self.stats.misses.saturating_add(1);
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
        let mut leases = Vec::with_capacity(experts.len());
        for &expert in experts {
            leases.push(self.lease(table, expert, target)?);
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
        Ok(ExpertBatch {
            arrays,
            _leases: leases,
        })
    }

    fn load(&self, table: usize, expert: u32, target: StreamOrDevice) -> Result<ExpertArrays> {
        let spec = &self.tables[table];
        let read = |projection, suffix| {
            self.files.read_rows_array(
                &format!("{}.{}.{}", spec.prefix, projection, suffix),
                expert as u64,
                1,
                self.staging_bytes,
            )
        };
        let fuse = |suffix| -> Result<Array> {
            let gate = read("gate_proj", suffix)?;
            let up = read("up_proj", suffix)?;
            let result = mlx::ops::shape::concatenate_on(&[&gate, &up], 1, target)?;
            result.eval()?;
            Ok(result)
        };
        let arrays = ExpertArrays {
            gate_up: [fuse("weight")?, fuse("scales")?, fuse("biases")?],
            down: [
                read("down_proj", "weight")?,
                read("down_proj", "scales")?,
                read("down_proj", "biases")?,
            ],
        };
        let bytes: usize = arrays
            .gate_up
            .iter()
            .chain(&arrays.down)
            .map(|a| a.size() * a.dtype().byte_size())
            .sum();
        ensure!(
            bytes == self.charges[table] - ENTRY_OVERHEAD,
            "expert load disagrees with residency accounting"
        );
        self.files.validate_sources()?;
        Ok(arrays)
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
