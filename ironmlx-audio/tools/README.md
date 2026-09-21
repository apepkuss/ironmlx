# Offline IndexTTS resource and parity tools

These tools are developer utilities. Neither the Rust library nor the server
starts Python. Use Python 3.11 in an isolated environment with `requirements.txt`.
Source snapshot identities are pinned in `../resources/indextts25/sources.json`.
Supply existing local resources; the tools do not download them.

```sh
python convert_indextts25.py --snapshot "$SNAPSHOT" \
  --campplus "$CAMPPLUS" --w2v "$W2V" --output "$DERIVED"
python -m unittest discover -s . -p test_conversion.py -v
```

`DERIVED` must not exist. Conversion verifies all input hashes, serializes all
941 tensors without casts, and compares every restored tensor byte. Both
floating-point signed zero and integer batch counters are preserved. Original
PyTorch tensor layouts remain unchanged. The statistics retain variance, not
standard deviation. All CAMPPlus batch counters are retained and explicitly
listed as unused during inference. Published w2v-BERT weights are reused from
the snapshot, never copied or downloaded.

Output contains two safetensors files, copied notices/configuration and a
manifest with tool/source/output hashes, schemas and key mappings. A sibling
lock serializes cooperative writers; files are staged and verified before an
atomic directory rename. Failed conversions clean their own stage and lock.
Never edit an immutable snapshot or replace an existing derived directory.
The Rust loader checks compiled output identities, including notices/configs;
a changed manifest cannot authorize different weights.

## App preparation profile

The App prepares these resources without Python. Its bundled, data-only recipe
maps each output tensor to a storage member in a hash-pinned PyTorch ZIP archive;
it never interprets pickle. Source archives and reconstructed safetensors must
match the same identities checked by the Rust loader. No model weights are
embedded in the App recipe.

After regenerating and validating the offline artifacts, update or check the
App recipe using this standard-library-only developer tool:

```sh
python generate_app_resources.py --snapshot "$SNAPSHOT" --campplus "$CAMPPLUS" --derived "$DERIVED"
python generate_app_resources.py --snapshot "$SNAPSHOT" --campplus "$CAMPPLUS" --derived "$DERIVED" --check
```

The generated file lives in the App's `Resources/indextts25-preparation.json`.
App tests compare its resource identities with the native library's profiles;
the opt-in real-resource test verifies every reconstructed output byte by hash.

## Numerical reference generation

Use a clean checkout of `vanch007/mlx-indextts2` at
`a7666367b8551656a2029ad75f259cb5e4936b3b` as `REFERENCE`. Keep real speech,
large baselines, environments and process reports outside tracked source.

```sh
python reference_features.py --reference "$REFERENCE" --snapshot "$SNAPSHOT" \
  --w2v "$W2V" --audio "$SPEECH_FILE" --output "$FEATURES"
MLX_ENABLE_TF32=0 python reference_encoders.py --reference "$REFERENCE" \
  --snapshot "$SNAPSHOT" --w2v "$W2V" --campplus "$CAMPPLUS" \
  --features "$FEATURES" --case speech_1s --output "$OUTPUT_DIRECTORY/encoders-fp32-1s.safetensors"
```

Repeat the encoder command with `speech_odd` / `encoders-fp32-odd.safetensors`
and `speech_5s` / `encoders-fp32-5s.safetensors`. PyTorch encoders run on CPU;
MLX conditioning runs with TF32 disabled. The reference writes hidden layer 17,
normalized semantic features, style, emotion, GPT conditioning and length
regulator output. It does not run speech synthesis.

For the checked-in synthetic regression fixture, replace `--audio` with
`--synthetic` and review the resulting safetensors and JSON before replacement.
The synthetic input is a chirp plus seeded noise, with no real speech data.
`generate_feature_windows.py --output PATH` regenerates the fixed float32 Hann
and Povey coefficients using PyTorch 2.10.0. Keeping these coefficients avoids
host math-library differences amplified by logarithmic mel features.

`generate_resample_kernel.py --output PATH` regenerates the compact float32
16 kHz to 22.05 kHz sinc kernel used by the common reference conversion.
Other rate pairs use the same float32 algorithm computed at runtime. Sparse
kernels omit only the clamped support tail (coefficients below 1e-15).
