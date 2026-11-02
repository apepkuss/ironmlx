//! CPU-side PLE row addresses. This computes a few integer hashes per token;
//! the 320M-row embedding table remains in its checkpoint shards on SSD.
use anyhow::{anyhow, ensure, Result};

const GAMMA: u64 = 0x9e3779b97f4a7c15;

fn splitmix64(x: u64) -> u64 {
    let x = x.wrapping_add(GAMMA);
    let x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    let x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

fn prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n % 2 == 0 {
        return n == 2;
    }
    let mut divisor = 3;
    while divisor <= n / divisor {
        if n % divisor == 0 {
            return false;
        }
        divisor += 2;
    }
    true
}

#[derive(Clone, Debug)]
pub struct PleIndexSpec {
    pub vocab_size: u32,
    /// Uses text_config.eos_token_id; it is independent of generation's stop set.
    pub eos_token_id: u32,
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    pub vocab_size_base: u64,
    pub ple_layer_index: usize,
    pub seed: u64,
    pub alignment: u64,
    pub shard_count: u64,
}

pub struct PleIndex {
    spec: PleIndexSpec,
    multipliers: Vec<u64>,
    head_sizes: Vec<u64>,
    offsets: Vec<u64>,
    rows: u64,
}

/// Request-owned state: cloning this small value is enough to checkpoint
/// n-gram history. Convolution and recurrent state are checkpointed separately.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PleHistory {
    context: Vec<u32>,
    pub position: u64,
}
impl PleHistory {
    pub fn tokens(&self) -> &[u32] {
        &self.context
    }
}

impl PleIndex {
    pub fn new(spec: PleIndexSpec) -> Result<Self> {
        ensure!(
            spec.vocab_size > 0 && spec.eos_token_id < spec.vocab_size,
            "invalid PLE vocabulary/EOS"
        );
        ensure!(
            (2..=8).contains(&spec.ngram_size) && (1..=64).contains(&spec.heads_per_ngram),
            "unsupported PLE n-gram geometry"
        );
        ensure!(
            (2..=u32::MAX as u64).contains(&spec.vocab_size_base)
                && spec.alignment > 0
                && spec.shard_count > 0,
            "invalid PLE table geometry"
        );
        let heads = (spec.ngram_size - 1)
            .checked_mul(spec.heads_per_ngram)
            .ok_or_else(|| anyhow!("PLE head count overflow"))?;
        let first = spec
            .ple_layer_index
            .checked_mul(heads)
            .ok_or_else(|| anyhow!("PLE layer index overflow"))?;
        let last = first
            .checked_add(heads)
            .ok_or_else(|| anyhow!("PLE layer head count overflow"))?;
        ensure!(
            last <= 4096,
            "PLE prime search exceeds supported layer/head geometry"
        );
        let mut head_sizes = Vec::with_capacity(heads);
        let mut candidate = spec.vocab_size_base - 1;
        for index in 0..last {
            candidate = candidate
                .checked_add(1)
                .ok_or_else(|| anyhow!("PLE vocabulary overflow"))?;
            while !prime(candidate) {
                candidate += 1;
            }
            if index >= first {
                head_sizes.push(candidate);
            }
        }
        let mut total = 0u64;
        let mut offsets = Vec::with_capacity(heads);
        for &size in &head_sizes {
            offsets.push(total);
            total = total
                .checked_add(size)
                .ok_or_else(|| anyhow!("PLE table size overflow"))?;
        }
        let rows = total
            .checked_add(spec.alignment - 1)
            .ok_or_else(|| anyhow!("PLE padding overflow"))?
            / spec.alignment
            * spec.alignment;
        ensure!(
            spec.shard_count <= rows,
            "PLE shard count exceeds row count"
        );
        let bound = ((i64::MAX as u64 / spec.vocab_size as u64) / 2).max(1);
        let layer_seed = spec
            .seed
            .wrapping_add(10007u64.wrapping_mul(spec.ple_layer_index as u64));
        let multipliers = (0..spec.ngram_size)
            .map(|position| {
                2 * (splitmix64(layer_seed.wrapping_add(GAMMA.wrapping_mul(position as u64 + 1)))
                    % bound)
                    + 1
            })
            .collect();
        Ok(Self {
            spec,
            multipliers,
            head_sizes,
            offsets,
            rows,
        })
    }

    pub fn row_count(&self) -> u64 {
        self.rows
    }
    pub fn heads(&self) -> usize {
        self.head_sizes.len()
    }

    pub fn history(&self) -> PleHistory {
        PleHistory {
            context: vec![self.spec.eos_token_id; self.spec.ngram_size - 1],
            position: 0,
        }
    }

    pub fn indices(&self, input: &[u32], history: &mut PleHistory) -> Result<Vec<Vec<u64>>> {
        ensure!(
            history.context.len() == self.spec.ngram_size - 1,
            "PLE history belongs to another n-gram geometry"
        );
        ensure!(
            input
                .iter()
                .chain(&history.context)
                .all(|&token| token < self.spec.vocab_size),
            "PLE input contains an out-of-vocabulary token"
        );
        let end = history
            .position
            .checked_add(input.len() as u64)
            .ok_or_else(|| anyhow!("PLE position overflow"))?;
        let mut rows = Vec::with_capacity(input.len());
        for &token in input {
            let mut shifted = vec![token];
            let mut reset = false;
            for &previous in history.context.iter().rev() {
                shifted.push(if reset {
                    self.spec.eos_token_id
                } else {
                    previous
                });
                reset |= previous == self.spec.eos_token_id;
            }
            let mut mixed = shifted[0] as u64 * self.multipliers[0];
            let mut row = Vec::with_capacity(self.heads());
            for ngram in 2..=self.spec.ngram_size {
                mixed ^= shifted[ngram - 1] as u64 * self.multipliers[ngram - 1];
                let first = (ngram - 2) * self.spec.heads_per_ngram;
                for head in first..first + self.spec.heads_per_ngram {
                    row.push(mixed % self.head_sizes[head] + self.offsets[head]);
                }
            }
            rows.push(row);
            history.context.rotate_left(1);
            *history.context.last_mut().unwrap() = token;
        }
        history.position = end;
        Ok(rows)
    }

    /// Match checkpoint ShardedEmbedding's quotient/remainder distribution.
    pub fn shard_row(&self, row: u64) -> Result<(u64, u64)> {
        ensure!(row < self.rows, "PLE row is outside the padded vocabulary");
        let base = self.rows / self.spec.shard_count;
        let extra = self.rows % self.spec.shard_count;
        let split = (base + 1) * extra;
        Ok(if row < split {
            (row / (base + 1), row % (base + 1))
        } else {
            (extra + (row - split) / base, (row - split) % base)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn spec() -> PleIndexSpec {
        PleIndexSpec {
            vocab_size: 248320,
            eos_token_id: 248044,
            ngram_size: 3,
            heads_per_ngram: 8,
            vocab_size_base: 20000000,
            ple_layer_index: 0,
            seed: 1234,
            alignment: 128,
            shard_count: 128,
        }
    }
    fn fixture() -> serde_json::Value {
        serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/qwen38_flash_next/components.json"
        )))
        .unwrap()
    }
    #[test]
    fn matches_reference_hashes_including_eos_reset() {
        let reference = fixture();
        let p = &reference["ple"];
        let index = PleIndex::new(spec()).unwrap();
        let tokens: Vec<u32> = serde_json::from_value(p["tokens"].clone()).unwrap();
        let expected: Vec<Vec<u64>> = serde_json::from_value(p["rows"].clone()).unwrap();
        assert_eq!(
            index.indices(&tokens, &mut index.history()).unwrap(),
            expected
        );
        assert_eq!(index.row_count(), p["padded_rows"].as_u64().unwrap());
        assert_eq!(
            index.multipliers,
            serde_json::from_value::<Vec<u64>>(p["multipliers"].clone()).unwrap()
        );
        assert_eq!(
            index.head_sizes,
            serde_json::from_value::<Vec<u64>>(p["head_vocab_sizes"].clone()).unwrap()
        );
    }
    #[test]
    fn chunks_rollback_and_independent_requests_preserve_history() {
        let index = PleIndex::new(spec()).unwrap();
        let tokens = [100, 200, 248044, 300, 400];
        let mut history = index.history();
        let all = index.indices(&tokens, &mut history).unwrap();
        let final_state = history.clone();
        let mut history = index.history();
        let mut chunks = index.indices(&tokens[..2], &mut history).unwrap();
        let checkpoint = history.clone();
        let _ = index.indices(&[777, 888], &mut history).unwrap();
        history = checkpoint;
        chunks.extend(index.indices(&tokens[2..], &mut history).unwrap());
        assert_eq!(all, chunks);
        assert_eq!(final_state, history);
        let mut other = index.history();
        assert_eq!(index.indices(&tokens, &mut other).unwrap(), all);
        let before = other.clone();
        assert!(index.indices(&[100, 248320], &mut other).is_err());
        assert_eq!(other, before);
    }
    #[test]
    fn shard_boundaries_cover_all_checkpoint_rows() {
        let index = PleIndex::new(spec()).unwrap();
        let size = index.row_count() / 128;
        for shard in 0..128 {
            assert_eq!(index.shard_row(shard * size).unwrap(), (shard, 0));
            assert_eq!(
                index.shard_row((shard + 1) * size - 1).unwrap(),
                (shard, size - 1)
            );
        }
        assert!(index.shard_row(index.row_count()).is_err());
        let mut s = spec();
        s.shard_count = 7;
        let index = PleIndex::new(s).unwrap();
        let mut previous = (0, 0);
        for row in [0, 1, index.row_count() / 7, index.row_count() - 1] {
            let current = index.shard_row(row).unwrap();
            assert!(current >= previous);
            previous = current;
        }
    }
}
