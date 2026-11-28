//! Where a chat prompt's prefix can be reused by the next turn.
//!
//! A chat template's generation prompt can end in tokens the next turn's
//! rendering of the same history does not repeat: Qwen3.6 opens the assistant
//! turn with `<think>\n` (or an empty think block), while history renders the
//! previous reply without it. The prefix the next turn shares then ends just
//! before those tokens, so a prefix cache that stores state only at the end
//! of a prompt never matches a follow-up turn. [`ChatTurnBoundary`] learns the
//! template's generation suffixes once from the tokenizer; a prompt ending in
//! one is reusable up to the suffix.

use ironmlx_lm::core::{Message, Tokenizer};

/// Generation suffixes of a chat template (the tokens a prompt ends in that
/// the next turn's rendering does not repeat), longest first.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ChatTurnBoundary {
    suffixes: Vec<Vec<u32>>,
}

/// Template keyword variants the probe renders with (servers pass
/// `enable_thinking` through `chat_template_kwargs`).
fn probe_kwargs() -> [Option<serde_json::Value>; 3] {
    [
        None,
        Some(serde_json::json!({ "enable_thinking": false })),
        Some(serde_json::json!({ "enable_thinking": true })),
    ]
}

impl ChatTurnBoundary {
    /// Learns the generation suffixes of `tokenizer`'s chat template the way
    /// servers render and encode chat prompts; `None` without a template, or
    /// when no variant ends in a suffix the next turn drops.
    pub(crate) fn probe(tokenizer: &Tokenizer) -> Option<Self> {
        let mut suffixes: Vec<Vec<u32>> = Vec::new();
        for suffix in probe_suffixes(tokenizer) {
            if !suffix.is_empty() && !suffixes.contains(&suffix) {
                suffixes.push(suffix);
            }
        }
        suffixes.sort_by_key(|suffix| std::cmp::Reverse(suffix.len()));
        (!suffixes.is_empty()).then_some(Self { suffixes })
    }

    /// Whether `tokenizer`'s chat template adds no generation suffix: every
    /// variant renders a prompt that the next turn's prompt starts with, so
    /// the next turn shares the whole of it (Inkling). [`Self::probe`] finds
    /// no boundary for such a template.
    pub(crate) fn whole_prompt_continues(tokenizer: &Tokenizer) -> bool {
        let suffixes = probe_suffixes(tokenizer);
        !suffixes.is_empty() && suffixes.iter().all(Vec::is_empty)
    }

    #[cfg(test)]
    pub(crate) fn from_suffixes(mut suffixes: Vec<Vec<u32>>) -> Self {
        suffixes.sort_by_key(|suffix| std::cmp::Reverse(suffix.len()));
        Self { suffixes }
    }

    /// Token count of `prompt_ids` the next turn of the same chat shares:
    /// the prompt without its generation suffix, when it ends in one.
    pub(crate) fn reusable_len(&self, prompt_ids: &[u32]) -> Option<usize> {
        self.suffixes
            .iter()
            .find(|suffix| prompt_ids.len() > suffix.len() && prompt_ids.ends_with(suffix))
            .map(|suffix| prompt_ids.len() - suffix.len())
    }

    pub(crate) fn suffixes(&self) -> &[Vec<u32>] {
        &self.suffixes
    }
}

/// The tokens each template variant's first-turn prompt ends in that the
/// next turn's prompt does not repeat (empty when it repeats all of it); one
/// entry per variant that renders.
fn probe_suffixes(tokenizer: &Tokenizer) -> Vec<Vec<u32>> {
    if !tokenizer.has_chat_template() {
        return Vec::new();
    }
    let message = |role: &str, content: &str| Message {
        role: role.to_owned(),
        content: content.to_owned(),
    };
    let first = [message("user", "Hello there.")];
    let next = [
        message("user", "Hello there."),
        message("assistant", "Hi."),
        message("user", "How are you?"),
    ];
    let mut suffixes = Vec::new();
    for kwargs in probe_kwargs() {
        let render = |messages: &[Message]| {
            tokenizer
                .apply_chat_template(messages, true, kwargs.as_ref())
                .ok()
                .and_then(|text| tokenizer.encode(&text, false).ok())
        };
        let (Some(first_ids), Some(next_ids)) = (render(&first), render(&next)) else {
            continue;
        };
        let shared = first_ids
            .iter()
            .zip(&next_ids)
            .take_while(|(a, b)| a == b)
            .count();
        suffixes.push(first_ids[shared..].to_vec());
    }
    suffixes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reusable_len_drops_the_longest_matching_generation_suffix() {
        let boundary = ChatTurnBoundary::from_suffixes(vec![vec![9], vec![7, 8, 9]]);
        assert_eq!(boundary.reusable_len(&[1, 2, 7, 8, 9]), Some(2));
        assert_eq!(boundary.reusable_len(&[1, 2, 3, 9]), Some(3));
        assert_eq!(boundary.reusable_len(&[1, 2, 3]), None);
        // The whole prompt is never the suffix.
        assert_eq!(boundary.reusable_len(&[9]), None);
    }

    /// The real Qwen3.6 template: the probe finds the thinking and
    /// non-thinking generation suffixes, and an encoded follow-up turn
    /// starts with turn 1's prompt up to its reusable length.
    #[test]
    #[ignore = "needs a local Qwen3.6 checkpoint in QWEN36_MOE_DFLASH2_TARGET"]
    fn qwen36_follow_up_turns_share_the_prompt_up_to_the_generation_suffix() {
        let dir = std::env::var("QWEN36_MOE_DFLASH2_TARGET").expect("QWEN36_MOE_DFLASH2_TARGET");
        let loader = ironmlx_lm::core::Loader::open(std::path::Path::new(&dir)).unwrap();
        let tokenizer = Tokenizer::from_loader(&loader).unwrap();
        let boundary = ChatTurnBoundary::probe(&tokenizer).expect("generation suffixes");
        assert!(boundary.suffixes().len() >= 2, "{boundary:?}");
        let message = |role: &str, content: &str| Message {
            role: role.to_owned(),
            content: content.to_owned(),
        };
        for kwargs in probe_kwargs() {
            let encode = |messages: &[Message]| {
                let text = tokenizer
                    .apply_chat_template(messages, true, kwargs.as_ref())
                    .unwrap();
                tokenizer.encode(&text, false).unwrap()
            };
            let turn1 = encode(&[message("user", "Describe a lighthouse at night.")]);
            let turn2 = encode(&[
                message("user", "Describe a lighthouse at night."),
                message(
                    "assistant",
                    "A tall white tower sweeps its beam over dark water.",
                ),
                message("user", "Now describe it at dawn."),
            ]);
            let reusable = boundary
                .reusable_len(&turn1)
                .expect("turn 1 ends in a suffix");
            assert!(reusable < turn1.len());
            assert_eq!(turn2[..reusable], turn1[..reusable], "{kwargs:?}");
        }
    }

    /// The real Inkling template adds no generation suffix: the probe finds
    /// no boundary, and the next turn starts with the whole first prompt.
    #[test]
    #[ignore = "needs a local Inkling checkpoint in INKLING_DS_PARK_TEST_DIR"]
    fn inkling_follow_up_turns_share_the_whole_prompt() {
        let dir = std::env::var("INKLING_DS_PARK_TEST_DIR").expect("INKLING_DS_PARK_TEST_DIR");
        let loader = ironmlx_lm::core::Loader::open(std::path::Path::new(&dir)).unwrap();
        let tokenizer = Tokenizer::from_loader(&loader).unwrap();
        assert_eq!(ChatTurnBoundary::probe(&tokenizer), None);
        assert!(ChatTurnBoundary::whole_prompt_continues(&tokenizer));
    }
}
