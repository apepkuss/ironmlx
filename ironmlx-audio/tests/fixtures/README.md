# Reference fixtures

These are synthetic test assets, independent of the Rust implementation.
`text.json` stores the authored input corpus alongside upstream normalization,
segmentation, language decisions and complete token sequences. Add input cases
there before regenerating. Source hashes in `reference-sources.json` identify
the exact upstream Python files used. Model and dictionary identities are in
`resources/indextts25/sources.json` relative to the crate root.

Use a separate Python 3.11 environment with `requirements.txt`. Python is only
needed to regenerate these assets; Rust tests read the checked-in fixtures.
Provide the pinned upstream `mlx_indextts` source directory and model snapshot:

```sh
python -m pip install -r ironmlx-audio/tests/fixtures/requirements.txt
PYTHONHASHSEED=0 python ironmlx-audio/tests/fixtures/generate.py text \
  --reference /path/to/mlx_indextts --snapshot /path/to/snapshot \
  --output /path/to/review-fixtures
python ironmlx-audio/tests/fixtures/generate.py signal \
  --output /path/to/review-fixtures
```

Review outputs before replacing fixtures. General WeText contractions,
glossary and NeMo normalization are disabled by this profile; the upstream
wrapper still applies its limited English `'s` replacements.

The signal corpus contains seven torchaudio default Hann-sinc rate conversions.
Audio files encode one second of an 8 kHz mono synthetic signal, with integer
samples `trunc(12000 * sin(n * 0.1))`. WAV and FLAC should reproduce those samples;
MP3 is lossy and uses a separate numeric tolerance. Encoded bytes may differ
across libsndfile/codec builds; decoding and sample comparisons are the test
contract. Real text/resource tests additionally require the external pinned
FST, dictionary and model assets described in the crate README.

`reference-features.safetensors` contains one second of synthetic chirp/noise
PCM and the fixed Python reference outputs for mel, Kaldi fbank and SeamlessM4T
features/mask. The associated JSON records reference source identity. Generation
uses `../../tools/reference_features.py --synthetic`, the tool requirements and
fixed model/config snapshots described in `../../tools/README.md`. No real
speech or model weights are included in this fixture.
