//! Bounded, read-only cache of affine-4 PLE rows in their original checkpoint
//! layout. No giant tensor is mapped/evaluated by MLX. This cache is shared
//! model data; n-gram and convolution history belong to individual requests.
use super::ple_index::PleIndex;
use crate::core::paging_stats::{elapsed_us, PagingTelemetry};
use crate::core::tensor_file::ModelTensorFiles;
use anyhow::{ensure, Context, Result};
use mlx::{Array, Dtype, StreamOrDevice};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    sync::Arc,
};
use std::{sync::atomic::Ordering::Relaxed, time::Instant};

// Packed row payload lives in one slab (no allocation per row). The charge
// bounds the hash entry, ordered-map node and allocator overhead together.
const ENTRY_OVERHEAD: usize = 256;

pub struct PleRows {
    files: Arc<ModelTensorFiles>,
    shards: Vec<String>,
    index: PleIndex,
    width: usize,
    row_bytes: usize,
    capacity: usize,
    staging_bytes: usize,
    rows: HashMap<u64, (usize, u64)>,
    payload: Vec<u8>,
    telemetry: Arc<PagingTelemetry>,
    ages: BTreeMap<u64, u64>,
    clock: u64,
    pub hits: u64,
    pub misses: u64,
}

impl PleRows {
    /// Cache and staging are separate reservations. At least one complete
    /// packed row must fit. Callers must chunk long prefill lookups to the
    /// staging limit rather than silently expanding their memory budget.
    pub fn new(
        files: Arc<ModelTensorFiles>,
        prefix: &str,
        index: PleIndex,
        width: usize,
        cache_bytes: usize,
        staging_bytes: usize,
    ) -> Result<Self> {
        ensure!(
            width > 0 && width <= i32::MAX as usize && width.is_multiple_of(32),
            "PLE affine-4 width must divide by 32 and fit MLX dimensions"
        );
        let row_bytes = width
            .checked_div(2)
            .and_then(|n| n.checked_add(width / 32 * 4))
            .context("PLE row size overflow")?;
        let capacity = cache_bytes / (row_bytes + ENTRY_OVERHEAD);
        ensure!(capacity > 0, "PLE cache cannot fit one complete row");
        ensure!(
            width
                .checked_mul(16)
                .and_then(|n| n.checked_add(1024))
                .and_then(|n| n.checked_mul(index.heads()))
                .is_some_and(|bytes| bytes <= staging_bytes),
            "PLE staging cannot fit one complete token lookup"
        );
        let mut shards = Vec::new();
        let mut first = 0u64;
        while first < index.row_count() {
            let (shard, local) = index.shard_row(first)?;
            ensure!(local == 0, "invalid PLE shard boundary");
            let name = format!("{prefix}.shards.{shard}");
            let weight = files
                .info(&format!("{name}.weight"))
                .context("missing PLE shard")?;
            let count = weight.shape.first().copied().unwrap_or(0);
            ensure!(
                count > 0 && weight.dtype == "U32" && weight.shape == [count, (width / 8) as u64],
                "invalid PLE packed weight"
            );
            for suffix in ["scales", "biases"] {
                let info = files
                    .info(&format!("{name}.{suffix}"))
                    .context("missing PLE quantization row")?;
                ensure!(
                    info.dtype == "BF16" && info.shape == [count, (width / 32) as u64],
                    "invalid PLE {suffix}"
                );
            }
            let last = first
                .checked_add(count - 1)
                .context("PLE shard range overflow")?;
            ensure!(
                last < index.row_count() && index.shard_row(last)? == (shard, count - 1),
                "PLE checkpoint shard geometry disagrees with hash table"
            );
            first = last + 1;
            shards.push(name);
        }
        Ok(Self {
            files,
            shards,
            index,
            width,
            row_bytes,
            capacity,
            staging_bytes,
            rows: HashMap::new(),
            payload: vec![0; capacity * row_bytes],
            telemetry: Arc::new(PagingTelemetry::new(false)),
            ages: BTreeMap::new(),
            clock: 0,
            hits: 0,
            misses: 0,
        })
    }

    pub fn with_telemetry(mut self, telemetry: Arc<PagingTelemetry>) -> Self {
        self.telemetry = telemetry;
        self.telemetry.ple.budget_bytes.fetch_add(
            (self.capacity * (self.row_bytes + ENTRY_OVERHEAD)) as u64,
            Relaxed,
        );
        let charged = self
            .telemetry
            .ple
            .charged_bytes
            .fetch_add(self.payload.len() as u64, Relaxed)
            + self.payload.len() as u64;
        self.telemetry
            .ple
            .peak_charged_bytes
            .fetch_max(charged, Relaxed);
        self
    }
    pub fn index(&self) -> &PleIndex {
        &self.index
    }
    pub fn max_gather_rows(&self) -> usize {
        (self.staging_bytes / (self.width * 16 + 1024)).min(i32::MAX as usize)
    }
    pub fn resident_rows(&self) -> usize {
        self.rows.len()
    }
    pub fn charged_bytes(&self) -> usize {
        self.payload.len() + self.rows.len() * ENTRY_OVERHEAD
    }

    fn read_lookup_rows(&self, rows: &[u64]) -> Result<Vec<(u64, Vec<u8>)>> {
        let mut runs: Vec<(usize, u64, u64, u64)> = Vec::new();
        for &row in rows {
            let (shard, local) = self.index.shard_row(row)?;
            if let Some(last) = runs.last_mut() {
                if last.0 == shard as usize && last.1 + last.3 == local && last.2 + last.3 == row {
                    last.3 += 1;
                    continue;
                }
            }
            runs.push((shard as usize, local, row, 1));
        }
        let requests: Vec<_> = runs
            .iter()
            .flat_map(|(shard, local, _, count)| {
                ["weight", "scales", "biases"]
                    .map(|suffix| (format!("{}.{suffix}", self.shards[*shard]), *local, *count))
            })
            .collect();
        let mut parts = self
            .files
            .read_rows_batch(&requests, rows.len() * self.row_bytes)?
            .into_iter();
        let sizes = [self.width / 2, self.width / 32 * 2, self.width / 32 * 2];
        let mut output = Vec::with_capacity(rows.len());
        for (_, _, row, count) in runs {
            let columns: Vec<_> = parts.by_ref().take(3).collect();
            ensure!(columns.len() == 3, "incomplete PLE row read");
            for offset in 0..count as usize {
                let mut bytes = Vec::with_capacity(self.row_bytes);
                for (column, size) in columns.iter().zip(sizes) {
                    bytes.extend_from_slice(&column.data[offset * size..(offset + 1) * size]);
                }
                output.push((row + offset as u64, bytes));
            }
        }
        Ok(output)
    }

    /// Returns [row_count, row_width] BF16 entries in caller order, including
    /// duplicates. Affine arithmetic uses one fp32 FMA and one BF16 rounding,
    /// matching MLX's gather/dequantize contract for these checkpoint rows.
    pub fn gather_on(&mut self, indices: &[u64], target: StreamOrDevice) -> Result<Array> {
        ensure!(
            !indices.is_empty() && indices.len() <= i32::MAX as usize,
            "PLE gather needs a nonempty supported row count"
        );
        let peak = indices
            .len()
            .checked_mul(self.width * 16 + 1024)
            .context("PLE staging size overflow")?;
        ensure!(
            peak <= self.staging_bytes,
            "PLE gather exceeds staging reservation; chunk prefill"
        );
        // Validate all addresses and sources before touching the cache.
        for &row in indices {
            self.index.shard_row(row)?;
        }
        self.files.validate_sources()?;
        let missing: Vec<_> = indices
            .iter()
            .copied()
            .filter(|r| !self.rows.contains_key(r))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let prefetched = if missing.is_empty() {
            Vec::new()
        } else {
            let read_start = Instant::now();
            let rows = self.read_lookup_rows(&missing)?;
            self.telemetry
                .ple
                .current()
                .read_bytes
                .fetch_add((missing.len() * self.row_bytes) as u64, Relaxed);
            self.telemetry
                .ple
                .current()
                .read_us
                .fetch_add(elapsed_us(read_start), Relaxed);
            rows
        };
        let mut output = Vec::with_capacity(indices.len() * self.width);
        for &row in indices {
            if self.clock == u64::MAX {
                self.rebase_clock();
            }
            self.clock += 1;
            if let Some((_, age)) = self.rows.get_mut(&row) {
                self.ages.remove(age);
                *age = self.clock;
                self.hits = self.hits.saturating_add(1);
                self.telemetry.ple.current().hits.fetch_add(1, Relaxed);
            } else {
                let bytes =
                    if let Ok(index) = prefetched.binary_search_by_key(&row, |(key, _)| *key) {
                        prefetched[index].1.clone()
                    } else {
                        // An initially resident row can be evicted earlier in this
                        // same gather when the configured cache is exceptionally small.
                        let started = Instant::now();
                        let mut rows = self.read_lookup_rows(&[row])?;
                        self.telemetry
                            .ple
                            .current()
                            .read_bytes
                            .fetch_add(self.row_bytes as u64, Relaxed);
                        self.telemetry
                            .ple
                            .current()
                            .read_us
                            .fetch_add(elapsed_us(started), Relaxed);
                        rows.remove(0).1
                    };
                ensure!(bytes.len() == self.row_bytes, "PLE row storage changed");
                let slot = if self.rows.len() == self.capacity {
                    let (old_age, old_row) =
                        self.ages.pop_first().expect("nonempty bounded row cache");
                    let (slot, age) = self.rows.remove(&old_row).expect("LRU row exists");
                    debug_assert_eq!(old_age, age);
                    self.telemetry.ple.current().evictions.fetch_add(1, Relaxed);
                    slot
                } else {
                    self.telemetry.ple.entries.fetch_add(1, Relaxed);
                    self.telemetry
                        .ple
                        .payload_bytes
                        .fetch_add(self.row_bytes as u64, Relaxed);
                    let charge = self
                        .telemetry
                        .ple
                        .charged_bytes
                        .fetch_add(ENTRY_OVERHEAD as u64, Relaxed)
                        + ENTRY_OVERHEAD as u64;
                    self.telemetry
                        .ple
                        .peak_charged_bytes
                        .fetch_max(charge, Relaxed);
                    self.rows.len()
                };
                self.payload[slot * self.row_bytes..(slot + 1) * self.row_bytes]
                    .copy_from_slice(&bytes);
                self.rows.insert(row, (slot, self.clock));
                self.misses = self.misses.saturating_add(1);
                self.telemetry.ple.current().misses.fetch_add(1, Relaxed);
            }
            self.ages.insert(self.clock, row);
            let slot = self.rows[&row].0;
            let bytes = &self.payload[slot * self.row_bytes..(slot + 1) * self.row_bytes];
            let scales = self.width / 2;
            let biases = scales + self.width / 32 * 2;
            for column in 0..self.width {
                let byte = bytes[column / 2];
                let quant = (byte >> (4 * (column % 2))) & 15;
                let offset = column / 32 * 2;
                let bf16 = |start| {
                    f32::from_bits(
                        (u16::from_le_bytes(bytes[start..start + 2].try_into().unwrap()) as u32)
                            << 16,
                    )
                };
                output.push((quant as f32).mul_add(bf16(scales + offset), bf16(biases + offset)));
            }
        }
        // Source validation also catches edits racing a hot-cache lookup.
        self.files.validate_sources()?;
        let result =
            Array::try_from((output.as_slice(), [indices.len() as i32, self.width as i32]))?
                .astype_on(Dtype::Bfloat16, target)?;
        result.eval()?;
        Ok(result)
    }

    fn rebase_clock(&mut self) {
        let order: Vec<u64> = self.ages.values().copied().collect();
        self.ages.clear();
        for (i, row) in order.into_iter().enumerate() {
            let age = i as u64 + 1;
            self.rows.get_mut(&row).unwrap().1 = age;
            self.ages.insert(age, row);
        }
        self.clock = self.rows.len() as u64;
    }
}
