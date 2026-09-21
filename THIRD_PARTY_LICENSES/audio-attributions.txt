# Source and resource notices

- The IndexTTS text frontend, character replacement policy and reference-conditioning networks are adapted from
  `vanch007/mlx-indextts2` at `a7666367b8551656a2029ad75f259cb5e4936b3b`.
  Copyright (c) 2026 Didi. MIT license: `licenses/mlx-indextts2-MIT.txt`.
  The Rust implementation adds explicit resource validation, bounded graph
  traversal, cancellation checkpoints and request capacity validation.
- WeText TN field ordering and wrapper behavior follow WeText 0.1.2.
  Copyright (c) 2022–2025 Zhendong Peng (pzd17@tsinghua.org.cn).
  Apache-2.0, the same license text as the repository's `LICENSE`.
  FST data is supplied externally and retains the distribution's license.
- `mecab-sys` 0.1.0 builds its bundled MeCab source. The Rust bindings are
  MIT OR Apache-2.0; MeCab is Copyright (c) 2001–2008 Taku Kudo and Copyright
  (c) 2004–2008 Nippon Telegraph and Telephone Corporation. This integration
  selects the BSD alternative, reproduced in `licenses/MeCab-BSD.txt`.
  The dependency source and checksum are locked by `Cargo.lock`.
- The resampling implementation follows the default Hann-window sinc algorithm
  used by torchaudio 2.10.0. Regression data records that reference version.
- Symphonia 0.5.5 is MPL-2.0; tiktoken-rs 0.12.0 is MIT; rustfst 1.3.1 is
  MIT OR Apache-2.0. Their sources are used without modification. Dependency
  license texts are available in their Cargo distributions. Product packaging
  must include them when this crate is linked into the shipped executable.

The files in `resources/indextts25/` contain resource identities and tensor
schemas and deterministic analysis-window and sinc-kernel coefficients, not model weights, FST graphs or dictionaries. Checkpoint, CAMPPlus,
w2v-BERT, WeText and UniDic distributions retain their own licenses and notices;
this crate's source license does not replace them.

Audio test fixtures are synthetic sine/chirp/noise signals generated for this project.
Text fixtures are authored test sentences and reference tokenization outputs.

- The native CAMPPlus architecture follows the 3D-Speaker implementation included
  in the pinned IndexTTS reference. Copyright 3D-Speaker
  (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
  Licensed under Apache-2.0; license text is the repository's `LICENSE`.
- Native w2v-BERT and SeamlessM4T feature computation follow Transformers 5.3.0
  (Hugging Face, Apache-2.0). The SeamlessM4T feature extractor carries
  Copyright 2023 The HuggingFace Inc. team. The pinned source retains its notices.
- Reference mel and Kaldi fbank use the algorithms of the fixed upstream,
  torchaudio 2.10.0 (Copyright (c) 2017 Facebook Inc. (Soumith Chintala),
  BSD-2-Clause; `licenses/torchaudio-BSD.txt`) and librosa 0.11.0
  (Copyright (c) 2013–2023, librosa development team, ISC;
  `licenses/librosa-ISC.txt`). PyTorch-generated fixed Hann/Povey
  window coefficients and synthetic spectral regression data contain no model
  weights or recorded speech. Conversion preserves the external model notices
  and configurations byte-for-byte in the derived artifact.
