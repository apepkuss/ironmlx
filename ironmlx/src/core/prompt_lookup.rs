use std::collections::{HashMap, VecDeque};

use anyhow::{anyhow, bail};
use serde::{Deserialize, Serialize};

use crate::Result;

const POSITIONS_PER_NGRAM: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptLookupConfig {
    pub min_ngram: usize,
    pub max_ngram: usize,
    pub max_draft_tokens: usize,
    pub history_window_tokens: usize,
    pub max_index_entries: usize,
}

impl PromptLookupConfig {
    pub const DEFAULT_MIN_NGRAM: usize = 2;
    pub const DEFAULT_MAX_NGRAM: usize = 4;
    pub const DEFAULT_MAX_DRAFT_TOKENS: usize = 4;
    pub const DEFAULT_HISTORY_WINDOW_TOKENS: usize = 32 * 1024;
    pub const DEFAULT_MAX_INDEX_ENTRIES: usize = 64 * 1024;

    pub fn validate(self) -> Result<Self> {
        if self.min_ngram == 0 {
            bail!("prompt lookup min_ngram must be >= 1");
        }
        if self.max_ngram < self.min_ngram {
            bail!(
                "prompt lookup max_ngram {} must be >= min_ngram {}",
                self.max_ngram,
                self.min_ngram
            );
        }
        if self.max_draft_tokens == 0 {
            bail!("prompt lookup max_draft_tokens must be >= 1");
        }
        if self.history_window_tokens <= self.max_ngram {
            bail!(
                "prompt lookup history_window_tokens {} must exceed max_ngram {}",
                self.history_window_tokens,
                self.max_ngram
            );
        }
        if self.max_index_entries == 0 {
            bail!("prompt lookup max_index_entries must be >= 1");
        }
        Ok(self)
    }
}

impl Default for PromptLookupConfig {
    fn default() -> Self {
        Self {
            min_ngram: Self::DEFAULT_MIN_NGRAM,
            max_ngram: Self::DEFAULT_MAX_NGRAM,
            max_draft_tokens: Self::DEFAULT_MAX_DRAFT_TOKENS,
            history_window_tokens: Self::DEFAULT_HISTORY_WINDOW_TOKENS,
            max_index_entries: Self::DEFAULT_MAX_INDEX_ENTRIES,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PromptLookupStats {
    pub queries: u64,
    pub hits: u64,
    pub misses: u64,
    pub drafted_tokens: u64,
    pub accepted_tokens: u64,
    pub rejected_tokens: u64,
    pub zero_accept_windows: u64,
    pub propose_us: u64,
    pub index_build_us: u64,
    pub index_update_us: u64,
    pub index_entries_current: u64,
    pub index_entries_peak: u64,
    pub index_evictions: u64,
    pub verify_forward_us: u64,
    pub projection_us: u64,
    pub verify_accept_host_sync_count: u64,
    pub verify_accept_host_sync_us: u64,
    pub rollback_count: u64,
    pub rollback_us: u64,
}

impl PromptLookupStats {
    pub fn saturating_delta_since(self, before: Self) -> Self {
        Self {
            queries: self.queries.saturating_sub(before.queries),
            hits: self.hits.saturating_sub(before.hits),
            misses: self.misses.saturating_sub(before.misses),
            drafted_tokens: self.drafted_tokens.saturating_sub(before.drafted_tokens),
            accepted_tokens: self.accepted_tokens.saturating_sub(before.accepted_tokens),
            rejected_tokens: self.rejected_tokens.saturating_sub(before.rejected_tokens),
            zero_accept_windows: self
                .zero_accept_windows
                .saturating_sub(before.zero_accept_windows),
            propose_us: self.propose_us.saturating_sub(before.propose_us),
            index_build_us: self.index_build_us.saturating_sub(before.index_build_us),
            index_update_us: self.index_update_us.saturating_sub(before.index_update_us),
            index_entries_current: self.index_entries_current,
            index_entries_peak: self.index_entries_peak.max(before.index_entries_peak),
            index_evictions: self.index_evictions.saturating_sub(before.index_evictions),
            verify_forward_us: self
                .verify_forward_us
                .saturating_sub(before.verify_forward_us),
            projection_us: self.projection_us.saturating_sub(before.projection_us),
            verify_accept_host_sync_count: self
                .verify_accept_host_sync_count
                .saturating_sub(before.verify_accept_host_sync_count),
            verify_accept_host_sync_us: self
                .verify_accept_host_sync_us
                .saturating_sub(before.verify_accept_host_sync_us),
            rollback_count: self.rollback_count.saturating_sub(before.rollback_count),
            rollback_us: self.rollback_us.saturating_sub(before.rollback_us),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct NgramKey(Box<[u32]>);

#[derive(Debug, Clone)]
struct IndexLedgerEntry {
    key: NgramKey,
    continuation: usize,
}

#[derive(Debug, Clone)]
pub struct PromptLookupRowState {
    config: PromptLookupConfig,
    history: Vec<u32>,
    index: HashMap<NgramKey, VecDeque<usize>>,
    ledger: VecDeque<IndexLedgerEntry>,
    index_entries_peak: usize,
    index_evictions: u64,
}

impl PromptLookupRowState {
    pub fn new(history: &[u32], config: PromptLookupConfig) -> Result<Self> {
        let config = config.validate()?;
        let mut state = Self {
            config,
            history: Vec::with_capacity(history.len()),
            index: HashMap::new(),
            ledger: VecDeque::new(),
            index_entries_peak: 0,
            index_evictions: 0,
        };
        for &token in history {
            state.commit(token);
        }
        Ok(state)
    }

    pub fn config(&self) -> PromptLookupConfig {
        self.config
    }

    pub fn history(&self) -> &[u32] {
        &self.history
    }

    pub fn index_entries(&self) -> usize {
        self.index.len()
    }

    pub fn index_entries_peak(&self) -> usize {
        self.index_entries_peak
    }

    pub fn index_evictions(&self) -> u64 {
        self.index_evictions
    }

    pub fn propose(&self, limit: usize) -> Option<(usize, Vec<u32>)> {
        let max_draft = limit.min(self.config.max_draft_tokens);
        if max_draft == 0 {
            return None;
        }
        let history_len = self.history.len();
        let window_start = history_len.saturating_sub(self.config.history_window_tokens);
        let max_ngram = self.config.max_ngram.min(history_len);
        for n in (self.config.min_ngram..=max_ngram).rev() {
            let suffix_start = history_len - n;
            let key = NgramKey(self.history[suffix_start..].into());
            let Some(positions) = self.index.get(&key) else {
                continue;
            };
            let mut best: Option<(usize, usize)> = None;
            for &continuation in positions.iter().rev() {
                if continuation < window_start || continuation >= suffix_start {
                    continue;
                }
                let available = history_len.saturating_sub(continuation).min(max_draft);
                if available == 0 {
                    continue;
                }
                match best {
                    Some((best_len, best_pos))
                        if best_len > available
                            || (best_len == available && best_pos > continuation) => {}
                    _ => best = Some((available, continuation)),
                }
            }
            if let Some((draft_len, continuation)) = best {
                return Some((
                    n,
                    self.history[continuation..continuation + draft_len].to_vec(),
                ));
            }
        }
        None
    }

    pub fn commit(&mut self, token: u32) {
        self.history.push(token);
        let continuation = self.history.len() - 1;
        for n in self.config.min_ngram..=self.config.max_ngram {
            if continuation < n {
                continue;
            }
            let key = NgramKey(self.history[continuation - n..continuation].into());
            let positions = self.index.entry(key.clone()).or_default();
            positions.push_back(continuation);
            while positions.len() > POSITIONS_PER_NGRAM {
                positions.pop_front();
            }
            self.ledger
                .push_back(IndexLedgerEntry { key, continuation });
        }
        self.evict_stale();
        self.evict_to_entry_cap();
        self.index_entries_peak = self.index_entries_peak.max(self.index.len());
    }

    fn evict_stale(&mut self) {
        let min_continuation = self
            .history
            .len()
            .saturating_sub(self.config.history_window_tokens);
        while self
            .ledger
            .front()
            .is_some_and(|entry| entry.continuation < min_continuation)
        {
            self.evict_oldest_ledger_entry();
        }
    }

    fn evict_to_entry_cap(&mut self) {
        while self.index.len() > self.config.max_index_entries {
            if self.ledger.is_empty() {
                break;
            }
            self.evict_oldest_ledger_entry();
        }
    }

    fn evict_oldest_ledger_entry(&mut self) {
        let Some(entry) = self.ledger.pop_front() else {
            return;
        };
        let mut remove_key = false;
        if let Some(positions) = self.index.get_mut(&entry.key) {
            if positions.front() == Some(&entry.continuation) {
                positions.pop_front();
            }
            remove_key = positions.is_empty();
        }
        if remove_key {
            self.index.remove(&entry.key);
            self.index_evictions = self.index_evictions.saturating_add(1);
        }
    }

    pub fn validate_history(&self, expected: &[u32]) -> Result<()> {
        if self.history != expected {
            return Err(anyhow!(
                "prompt lookup history diverged: indexed {} tokens, request has {}",
                self.history.len(),
                expected.len()
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> PromptLookupConfig {
        PromptLookupConfig {
            min_ngram: 2,
            max_ngram: 3,
            max_draft_tokens: 4,
            history_window_tokens: 64,
            max_index_entries: 64,
        }
    }

    #[test]
    fn proposes_continuation_for_longest_suffix_match() {
        let state = PromptLookupRowState::new(&[1, 2, 3, 4, 1, 2, 3], cfg()).unwrap();
        assert_eq!(state.propose(4), Some((3, vec![4, 1, 2, 3])));
    }

    #[test]
    fn does_not_match_current_suffix_to_itself() {
        let state = PromptLookupRowState::new(&[1, 2, 3], cfg()).unwrap();
        assert_eq!(state.propose(4), None);
    }

    #[test]
    fn rejected_draft_is_not_committed() {
        let mut state = PromptLookupRowState::new(&[1, 2, 3, 4, 1, 2, 3], cfg()).unwrap();
        let before = state.history().to_vec();
        let _draft = state.propose(4).unwrap();
        assert_eq!(state.history(), before);
        state.commit(9);
        assert_eq!(state.history().last(), Some(&9));
    }

    #[test]
    fn index_entry_cap_is_enforced() {
        let config = PromptLookupConfig {
            max_index_entries: 3,
            ..cfg()
        };
        let state = PromptLookupRowState::new(&(0..32).collect::<Vec<_>>(), config).unwrap();
        assert!(state.index_entries() <= 3);
        assert!(state.index_evictions() > 0);
    }

    #[test]
    fn repetitive_history_keeps_ledger_bounded_by_window() {
        let config = PromptLookupConfig {
            history_window_tokens: 16,
            ..cfg()
        };
        let state = PromptLookupRowState::new(&vec![7; 256], config).unwrap();
        let variants = config.max_ngram - config.min_ngram + 1;
        assert!(state.ledger.len() <= config.history_window_tokens * variants);
        assert!(state.index_entries() <= variants);
    }

    #[test]
    fn invalid_config_is_rejected() {
        let config = PromptLookupConfig {
            min_ngram: 4,
            max_ngram: 3,
            ..cfg()
        };
        assert!(config.validate().is_err());
    }
}
