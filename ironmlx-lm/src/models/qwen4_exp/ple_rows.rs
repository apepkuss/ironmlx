//! Bounded, read-only cache of affine-4 PLE rows in their original checkpoint
//! layout. No giant tensor is mapped/evaluated by MLX. This cache is shared
//! model data; n-gram and convolution history belong to individual requests.
use super::ple_index::PleIndex;
use crate::core::tensor_file::ModelTensorFiles;
use anyhow::{ensure, Context, Result};
use mlx::{Array, Dtype, StreamOrDevice};
use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

// Conservative charge for both map nodes, Vec allocation and allocator
// bookkeeping. Payload and entry metadata are bounded together.
const ENTRY_OVERHEAD: usize = 1024;

pub struct PleRows {
    files: Arc<ModelTensorFiles>,
    shards: Vec<String>,
    index: PleIndex,
    width: usize,
    row_bytes: usize,
    capacity: usize,
    staging_bytes: usize,
    rows: HashMap<u64, (Vec<u8>, u64)>,
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
            width > 0 && width <= i32::MAX as usize && width % 32 == 0,
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
                .checked_mul(12)
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
            ages: BTreeMap::new(),
            clock: 0,
            hits: 0,
            misses: 0,
        })
    }

    pub fn index(&self) -> &PleIndex {
        &self.index
    }
    pub fn max_gather_rows(&self) -> usize {
        (self.staging_bytes / (self.width * 12)).min(i32::MAX as usize)
    }
    pub fn resident_rows(&self) -> usize {
        self.rows.len()
    }
    pub fn charged_bytes(&self) -> usize {
        self.rows.len() * (self.row_bytes + ENTRY_OVERHEAD)
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
            .checked_mul(self.width)
            .and_then(|n| n.checked_mul(12))
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
            } else {
                let (shard, local) = self.index.shard_row(row)?;
                let prefix = &self.shards[usize::try_from(shard)?];
                let mut bytes = Vec::with_capacity(self.row_bytes);
                for suffix in ["weight", "scales", "biases"] {
                    bytes.extend(self.files.read_rows(
                        &format!("{prefix}.{suffix}"),
                        local,
                        1,
                        self.row_bytes,
                    )?);
                }
                ensure!(bytes.len() == self.row_bytes, "PLE row storage changed");
                if self.rows.len() == self.capacity {
                    let (old_age, old_row) =
                        self.ages.pop_first().expect("nonempty bounded row cache");
                    let (_, age) = self.rows.remove(&old_row).expect("LRU row exists");
                    debug_assert_eq!(old_age, age);
                }
                self.rows.insert(row, (bytes, self.clock));
                self.misses = self.misses.saturating_add(1);
            }
            self.ages.insert(self.clock, row);
            let bytes = &self.rows[&row].0;
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
