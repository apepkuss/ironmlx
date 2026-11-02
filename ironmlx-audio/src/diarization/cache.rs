//! Arrival-order speaker cache selection, following pinned mlx-audio AOSC.
use anyhow::Result;
use mlx::{Array, Dtype};

pub(super) fn compress(
    embeddings: &Array,
    probabilities: &Array,
    silence: &Array,
) -> Result<(Array, Array)> {
    let n = embeddings.shape().as_slice()[1] as usize;
    let probabilities = probabilities.to_vec::<f32>()?;
    let mut scores = vec![[f32::NEG_INFINITY; 8]; n];
    let mut positive = [0usize; 8];
    for (i, p) in probabilities.chunks_exact(8).enumerate() {
        let sum = p.iter().map(|v| (1. - v).max(0.25).ln()).sum::<f32>();
        for s in 0..8 {
            if p[s] > 0.5 {
                let score = p[s].max(0.25).ln() - (1. - p[s]).max(0.25).ln() + sum - 0.5f32.ln();
                scores[i][s] = score;
                positive[s] += usize::from(score > 0.);
            }
        }
    }
    for (i, row) in scores.iter_mut().enumerate() {
        for s in 0..8 {
            if positive[s] >= 16 && row[s] <= 0. {
                row[s] = f32::NEG_INFINITY;
            }
            if i >= 264 {
                row[s] += 0.05;
            }
        }
    }
    for (k, boost) in [(24, -2. * 0.5f32.ln()), (48, -0.5f32.ln())] {
        for (s, _) in positive.iter().enumerate() {
            let mut indices = (0..n).collect::<Vec<_>>();
            indices.sort_by(|&a, &b| scores[b][s].total_cmp(&scores[a][s]).then(a.cmp(&b)));
            for i in indices.into_iter().take(k) {
                if scores[i][s].is_finite() {
                    scores[i][s] += boost;
                }
            }
        }
    }
    // Keep duplicates across channels, and sort speaker-major indices, exactly
    // as the reference's flattened (speaker, frame) top-k cache representation.
    let mut flat = (0..8 * (n + 1)).collect::<Vec<_>>();
    let score = |i: usize| {
        if i % (n + 1) == n {
            f32::INFINITY
        } else {
            scores[i % (n + 1)][i / (n + 1)]
        }
    };
    flat.sort_by(|&a, &b| score(b).total_cmp(&score(a)).then(a.cmp(&b)));
    flat.truncate(264);
    flat.sort_by_key(|&i| {
        if score(i) > f32::NEG_INFINITY {
            i
        } else {
            99999
        }
    });
    let source = embeddings.astype(Dtype::Float32)?.to_vec::<f32>()?;
    let silence = silence.astype(Dtype::Float32)?.to_vec::<f32>()?;
    let mut selected = Vec::with_capacity(264 * 512);
    let mut preds = Vec::with_capacity(264 * 8);
    for i in flat {
        let frame = i % (n + 1);
        if frame == n || score(i) == f32::NEG_INFINITY {
            selected.extend_from_slice(&silence);
            preds.extend_from_slice(&[0.; 8]);
        } else {
            selected.extend_from_slice(&source[frame * 512..(frame + 1) * 512]);
            preds.extend_from_slice(&probabilities[frame * 8..(frame + 1) * 8]);
        }
    }
    Ok((
        Array::try_from((selected.as_slice(), [1, 264, 512]))?.astype(embeddings.dtype())?,
        Array::try_from((preds.as_slice(), [1, 264, 8]))?,
    ))
}
