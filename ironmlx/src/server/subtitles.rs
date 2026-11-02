//! Subtitle cue construction from aligned units, independent of ASR windows.
use ironmlx_audio::qwen3_asr::AlignedWord;
use std::fmt::Write;

const LINE_WIDTH: usize = 42;
const CUE_WIDTH: usize = LINE_WIDTH * 2;
const TARGET_DURATION_MS: u64 = 6000;
const PAUSE_MS: u64 = 600;

#[derive(Clone, Copy, Debug)]
pub(super) enum SubtitleFormat {
    Srt,
    Vtt,
}

impl SubtitleFormat {
    pub(super) fn content_type(self) -> &'static str {
        match self {
            Self::Srt => "application/x-subrip; charset=utf-8",
            Self::Vtt => "text/vtt; charset=utf-8",
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub(super) enum SubtitleError {
    #[error("Aligned units do not match the transcript")]
    TextMismatch,
    #[error("Alignment contains invalid or overlapping timestamps")]
    InvalidTiming,
    #[error("Alignment has no positive subtitle interval at millisecond precision")]
    UnavailableTiming,
}

struct Unit<'a> {
    text: &'a str,
    start: u64,
    end: u64,
    sentence_end: bool,
    clause_end: bool,
}

struct Cue {
    first: usize,
    end: usize,
}

pub(super) fn render(
    text: &str,
    words: &[AlignedWord],
    duration: f64,
    format: SubtitleFormat,
) -> Result<String, SubtitleError> {
    let text = clean_text(text);
    let units = aligned_units(&text, words, duration)?;
    let mut output = match format {
        SubtitleFormat::Srt => String::new(),
        SubtitleFormat::Vtt => "WEBVTT\n\n".to_owned(),
    };
    if units.is_empty() {
        return Ok(output);
    }
    let mut cues: Vec<Cue> = Vec::new();
    let mut first = 0;
    let mut width = display_width(units[0].text);
    for i in 1..units.len() {
        let previous = &units[i - 1];
        let elapsed = previous.end - units[first].start;
        let next_width = display_width(units[i].text);
        let should_split = units[i].start - previous.end >= PAUSE_MS
            || (previous.sentence_end && elapsed >= 1000)
            || (previous.clause_end && (elapsed >= 3000 || width >= LINE_WIDTH))
            || units[i].end - units[first].start > TARGET_DURATION_MS
            || width + next_width > CUE_WIDTH;
        // Keep zero-duration units with timed neighbours; do not invent times.
        if should_split && elapsed > 0 {
            cues.push(Cue { first, end: i });
            first = i;
            width = 0;
        }
        width += next_width;
    }
    if units.last().unwrap().end == units[first].start {
        if let Some(previous) = cues.last_mut() {
            previous.end = units.len();
        } else {
            return Err(SubtitleError::UnavailableTiming);
        }
    } else {
        cues.push(Cue {
            first,
            end: units.len(),
        });
    }
    let separator = match format {
        SubtitleFormat::Srt => ',',
        SubtitleFormat::Vtt => '.',
    };
    for (index, cue) in cues.iter().enumerate() {
        let start = units[cue.first].start;
        let end = units[cue.end - 1].end;
        let content = units[cue.first..cue.end]
            .iter()
            .map(|unit| unit.text)
            .collect::<String>();
        // Escaping also prevents transcript text from becoming VTT tags/timings.
        let content = wrap_text(content.trim())
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;");
        writeln!(output, "{}", index + 1).unwrap();
        writeln!(
            output,
            "{} --> {}\n{content}\n",
            timestamp(start, separator),
            timestamp(end, separator)
        )
        .unwrap();
    }
    Ok(output)
}

fn clean_text(text: &str) -> String {
    let mut cleaned = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch.is_whitespace() {
            if !cleaned.is_empty() && !cleaned.ends_with(' ') {
                cleaned.push(' ');
            }
        } else if !ch.is_control() {
            cleaned.push(ch);
        }
    }
    cleaned.truncate(cleaned.trim_end().len());
    cleaned
}

pub(super) fn aligned_text(
    text: &str,
    words: &[AlignedWord],
    duration: f64,
) -> Result<Vec<(String, f64, f64)>, SubtitleError> {
    Ok(aligned_units(text, words, duration)?
        .into_iter()
        .map(|u| {
            (
                u.text.to_owned(),
                u.start as f64 / 1000.,
                u.end as f64 / 1000.,
            )
        })
        .collect())
}

fn aligned_units<'a>(
    text: &'a str,
    words: &[AlignedWord],
    duration: f64,
) -> Result<Vec<Unit<'a>>, SubtitleError> {
    if !duration.is_finite() || !(0. ..=7_200.).contains(&duration) {
        return Err(SubtitleError::InvalidTiming);
    }
    // Native alignment strips punctuation from its units. Recover the matching
    // source spans, so subtitles retain punctuation, case and original spacing.
    let mut significant = text
        .char_indices()
        .filter(|(_, ch)| ch.is_alphanumeric() || *ch == '\'');
    let mut spans = Vec::with_capacity(words.len());
    let mut previous_end = 0;
    let duration_ms = (duration * 1000.).round() as u64;
    for word in words {
        if !word.start.is_finite()
            || !word.end.is_finite()
            || word.start < 0.
            || word.end < word.start
            || word.end > duration + 0.001
        {
            return Err(SubtitleError::InvalidTiming);
        }
        let start = ((word.start * 1000.).round() as u64).min(duration_ms);
        let end = ((word.end * 1000.).round() as u64).min(duration_ms);
        if start < previous_end {
            return Err(SubtitleError::InvalidTiming);
        }
        previous_end = end;
        let mut source_start = None;
        let mut source_end = 0;
        for expected in word.word.chars() {
            let (index, ch) = significant.next().ok_or(SubtitleError::TextMismatch)?;
            if ch != expected {
                return Err(SubtitleError::TextMismatch);
            }
            source_start.get_or_insert(index);
            source_end = index + ch.len_utf8();
        }
        let source_start = source_start.ok_or(SubtitleError::TextMismatch)?;
        spans.push((source_start, source_end, start, end));
    }
    if significant.next().is_some() {
        return Err(SubtitleError::TextMismatch);
    }
    let mut units = Vec::with_capacity(words.len());
    let mut source_first = 0;
    for (i, &(_, source_end, start, end)) in spans.iter().enumerate() {
        let next_start = spans.get(i + 1).map_or(text.len(), |span| span.0);
        let mut boundary = next_start;
        if i + 1 < spans.len() {
            for (offset, ch) in text[source_end..next_start].char_indices().rev() {
                if ch.is_whitespace() || is_opening(ch) {
                    boundary = source_end + offset;
                } else {
                    break;
                }
            }
        }
        let suffix = &text[source_end..boundary];
        units.push(Unit {
            text: &text[source_first..boundary],
            start,
            end,
            sentence_end: suffix.chars().any(|c| ".!?。！？;；".contains(c)),
            clause_end: suffix.chars().any(|c| ",，:：、".contains(c)),
        });
        source_first = boundary;
    }
    Ok(units)
}

fn is_opening(ch: char) -> bool {
    "([{（【《〈「『“‘".contains(ch)
}

fn char_width(ch: char) -> usize {
    match ch as u32 {
        0x0300..=0x036f
        | 0x1ab0..=0x1aff
        | 0x1dc0..=0x1dff
        | 0x200d
        | 0x20d0..=0x20ff
        | 0xfe00..=0xfe0f
        | 0xfe20..=0xfe2f => 0,
        0x1100..=0x115f
        | 0x2e80..=0xa4cf
        | 0xac00..=0xd7a3
        | 0xf900..=0xfaff
        | 0xfe10..=0xfe6f
        | 0xff00..=0xff60
        | 0xffe0..=0xffe6
        | 0x1f000..=0x1faff
        | 0x20000..=0x3fffd => 2,
        _ => 1,
    }
}

fn display_width(text: &str) -> usize {
    text.chars().map(char_width).sum()
}

fn wrap_text(text: &str) -> String {
    let total = display_width(text);
    if total <= LINE_WIDTH {
        return text.into();
    }
    let mut width = 0;
    let mut fallback = 0;
    let mut best = None;
    for (index, ch) in text.char_indices() {
        width += char_width(ch);
        if width > LINE_WIDTH {
            break;
        }
        let boundary = index + ch.len_utf8();
        fallback = boundary;
        let right = text[boundary..].trim_start();
        let right_width = display_width(right);
        if (ch.is_whitespace() || char_width(ch) == 2)
            && !is_opening(ch)
            && !right.starts_with(|c: char| ",.!?，。！？、；：)]}）】》〉」』”’".contains(c))
            && right_width <= LINE_WIDTH
        {
            let imbalance = width.abs_diff(right_width);
            if best.is_none_or(|(_, previous)| imbalance < previous) {
                best = Some((boundary, imbalance));
            }
        }
    }
    let split = best.map_or(fallback, |(index, _)| index);
    format!(
        "{}\n{}",
        text[..split].trim_end(),
        text[split..].trim_start()
    )
}

fn timestamp(ms: u64, separator: char) -> String {
    format!(
        "{:02}:{:02}:{:02}{separator}{:03}",
        ms / 3_600_000,
        ms / 60_000 % 60,
        ms / 1000 % 60,
        ms % 1000
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(items: &[(&str, f64, f64)]) -> Vec<AlignedWord> {
        items
            .iter()
            .map(|&(word, start, end)| AlignedWord {
                word: word.into(),
                start,
                end,
            })
            .collect()
    }

    fn payloads(output: &str) -> Vec<&str> {
        output
            .split("\n\n")
            .filter_map(|block| {
                let mut lines = block.splitn(3, '\n');
                lines.next()?;
                let timing = lines.next()?;
                timing.contains(" --> ").then(|| lines.next().unwrap())
            })
            .collect()
    }

    #[test]
    fn formats_have_exact_headers_milliseconds_and_separators() {
        let aligned = words(&[("Hello", 0.4, 1.2), ("world", 1.2, 2.4)]);
        assert_eq!(
            render("Hello, world!", &aligned, 3., SubtitleFormat::Srt).unwrap(),
            "1\n00:00:00,400 --> 00:00:02,400\nHello, world!\n\n"
        );
        assert_eq!(
            render("Hello, world!", &aligned, 3., SubtitleFormat::Vtt).unwrap(),
            "WEBVTT\n\n1\n00:00:00.400 --> 00:00:02.400\nHello, world!\n\n"
        );
        assert_eq!(timestamp(3_599_999, ','), "00:59:59,999");
        assert_eq!(timestamp(3_600_000, '.'), "01:00:00.000");
    }

    #[test]
    fn punctuation_and_opening_quotes_stay_with_their_sentences() {
        let aligned = words(&[("Hello", 0., 1.2), ("Next", 1.25, 2.5)]);
        let result = render("“Hello.” 「Next!」", &aligned, 3., SubtitleFormat::Srt).unwrap();
        assert_eq!(payloads(&result), ["“Hello.”", "「Next!」"]);
    }

    #[test]
    fn gaps_and_duration_split_cues_without_asr_window_boundaries() {
        let aligned = words(&[
            ("one", 0., 3.),
            ("two", 3., 6.),
            ("three", 6., 9.),
            ("four", 10., 11.),
        ]);
        let result = render("one two three four", &aligned, 12., SubtitleFormat::Vtt).unwrap();
        assert_eq!(payloads(&result), ["one two", "three", "four"]);
        assert!(result.contains("00:00:06.000 --> 00:00:09.000"));
    }

    #[test]
    fn clause_breaks_are_used_after_three_seconds() {
        let aligned = words(&[("First", 0., 1.5), ("clause", 1.5, 3.1), ("next", 3.1, 4.)]);
        let result = render("First clause, next.", &aligned, 4., SubtitleFormat::Srt).unwrap();
        assert_eq!(payloads(&result), ["First clause,", "next."]);
    }

    #[test]
    fn cjk_cues_are_bounded_and_wrapped_without_losing_characters() {
        let text = format!("{}。", "字幕".repeat(25));
        let aligned = text
            .chars()
            .filter(|c| c.is_alphanumeric())
            .enumerate()
            .map(|(i, c)| AlignedWord {
                word: c.to_string(),
                start: i as f64 * 0.1,
                end: (i + 1) as f64 * 0.1,
            })
            .collect::<Vec<_>>();
        let result = render(&text, &aligned, 5., SubtitleFormat::Srt).unwrap();
        let captions = payloads(&result);
        assert_eq!(captions.len(), 2);
        assert!(captions.iter().all(|v| v.lines().count() <= 2));
        assert!(captions
            .iter()
            .flat_map(|v| v.lines())
            .all(|v| display_width(v) <= LINE_WIDTH));
        assert_eq!(captions.join("").replace('\n', ""), text);
    }

    #[test]
    fn english_wrapping_keeps_words_and_combining_accents() {
        let text = "Cafe\u{301}, let’s preserve every word and punctuation in this caption.";
        let source_words = [
            "Cafe",
            "lets",
            "preserve",
            "every",
            "word",
            "and",
            "punctuation",
            "in",
            "this",
            "caption",
        ];
        let aligned = source_words
            .iter()
            .enumerate()
            .map(|(i, w)| AlignedWord {
                word: (*w).into(),
                start: i as f64 * 0.2,
                end: (i + 1) as f64 * 0.2,
            })
            .collect::<Vec<_>>();
        let result = render(text, &aligned, 3., SubtitleFormat::Vtt).unwrap();
        let captions = payloads(&result);
        assert_eq!(captions.len(), 1);
        assert_eq!(captions[0].lines().collect::<Vec<_>>().join(" "), text);
        assert!(captions[0]
            .lines()
            .all(|line| display_width(line) <= LINE_WIDTH));
    }

    #[test]
    fn literal_markup_and_blank_lines_cannot_inject_vtt_blocks() {
        let aligned = words(&[
            ("b", 0., 0.1),
            ("Ready", 0.1, 0.3),
            ("b", 0.3, 0.5),
            ("go", 0.5, 0.7),
            ("now", 0.7, 0.9),
            ("WEBVTT", 0.9, 1.1),
        ]);
        let result = render(
            "<b>Ready</b> & go --> now.\n\nWEBVTT\0",
            &aligned,
            2.,
            SubtitleFormat::Vtt,
        )
        .unwrap();
        assert_eq!(
            payloads(&result),
            ["&lt;b&gt;Ready&lt;/b&gt; &amp; go --&gt; now. WEBVTT"]
        );
        assert_eq!(result.matches(" --> ").count(), 1);
    }

    #[test]
    fn zero_duration_units_are_retained_in_neighbouring_cues() {
        let aligned = words(&[("A", 0., 0.), ("B", 0., 1.), ("C", 3., 3.)]);
        let result = render("ABC", &aligned, 4., SubtitleFormat::Srt).unwrap();
        assert_eq!(payloads(&result), ["ABC"]);
        assert!(result.contains("00:00:00,000 --> 00:00:03,000"));
    }

    #[test]
    fn all_zero_or_submillisecond_intervals_are_not_fabricated() {
        for aligned in [words(&[("A", 1., 1.)]), words(&[("A", 0., 0.0004)])] {
            assert!(matches!(
                render("A", &aligned, 2., SubtitleFormat::Srt),
                Err(SubtitleError::UnavailableTiming)
            ));
        }
    }

    #[test]
    fn invalid_or_mismatched_alignment_is_rejected() {
        for aligned in [
            words(&[("A", f64::NAN, 1.)]),
            words(&[("A", 0., f64::INFINITY)]),
            words(&[("A", -1., 1.)]),
            words(&[("A", 2., 1.)]),
            words(&[("A", 0., 3.)]),
        ] {
            assert!(matches!(
                render("A", &aligned, 2., SubtitleFormat::Vtt),
                Err(SubtitleError::InvalidTiming)
            ));
        }
        assert!(matches!(
            render(
                "AB",
                &words(&[("A", 0., 1.), ("B", 0.9, 2.)]),
                2.,
                SubtitleFormat::Vtt
            ),
            Err(SubtitleError::InvalidTiming)
        ));
        assert!(matches!(
            render(
                "Different",
                &words(&[("A", 0., 1.)]),
                2.,
                SubtitleFormat::Vtt
            ),
            Err(SubtitleError::TextMismatch)
        ));
        assert!(matches!(
            render("A", &[], 2., SubtitleFormat::Srt),
            Err(SubtitleError::TextMismatch)
        ));
    }

    #[test]
    fn empty_speech_has_no_placeholder_cues() {
        assert_eq!(render("", &[], 0., SubtitleFormat::Srt).unwrap(), "");
        assert_eq!(
            render(" \n", &[], 0., SubtitleFormat::Vtt).unwrap(),
            "WEBVTT\n\n"
        );
    }

    #[test]
    fn a_long_single_unit_is_not_truncated_or_given_invented_times() {
        let text = "a".repeat(100);
        let aligned = words(&[(&text, 0., 10.)]);
        let result = render(&text, &aligned, 10., SubtitleFormat::Srt).unwrap();
        assert_eq!(payloads(&result)[0].replace('\n', ""), text);
        assert!(result.contains("00:00:00,000 --> 00:00:10,000"));
        assert_eq!(payloads(&result)[0].lines().count(), 2);
    }

    #[test]
    fn global_offsets_and_hour_boundaries_are_preserved() {
        let aligned = words(&[("First", 59.9996, 60.1), ("Last", 3600.4, 3600.8)]);
        let result = render("First. Last.", &aligned, 3601., SubtitleFormat::Vtt).unwrap();
        assert!(result.contains("00:01:00.000 --> 00:01:00.100"));
        assert!(result.contains("01:00:00.400 --> 01:00:00.800"));
    }
}
