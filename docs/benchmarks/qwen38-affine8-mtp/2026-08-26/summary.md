# Qwen3.8-27B affine8 MTP B2/B4 Performance Archive

Date: 2026-08-26

Status: passed for the retained non-DFlash2 B2 and B4 MTP paths. This archive
records paired ordinary-decode and forced-MTP measurements for
`Qwen3.8-27B-8bit` with its matching 8-bit MTP checkpoint. B8 MTP experiments
are intentionally outside this archive and are not a supported production
path.

## Revisions and Environment

| Component | Revision or configuration |
|---|---|
| IronMLX 8-bit support | `fc7dacfffc0032c1ad65951caf68f67b553467b2` |
| IronMLX B2 candidate | `bce715f9f108355de8ce947d594c7ec8b6236618` |
| IronMLX B4 candidate | `eb62c64af6ebeee57b11819b22a1f18e430397f8` |
| MLX | `fix/nax-qmm-product-shape-main@73ad5df20cb30be4192e5c4d0ae8130674773427` |
| macOS | `26.4` (`25E246`) |
| Hardware | MacBook Pro `Mac17,6`, Apple M5 Max, 18 CPU cores, 128 GB unified memory |
| Base model | `mlx-community/Qwen3.8-27B-8bit@815b83c0df8ffd1d1b5244cf75fd6ef14fca9ef9` |
| MTP model | `mlx-community/Qwen3.8-27B-MTP-8bit@e88e48d055732ad75d9435f3059139d5279f2064` |

## Measurement Contract

- In-process `ironmlx-core-bench` `scheduler-text` mode; HTTP and client
  overhead are excluded.
- One fixed 78-token chat prompt repeated once per batch row. Prompt text and
  generated response bodies are not retained in this archive.
- Greedy decoding, EOS ignored, fixed one-token MTP draft depth, and ordinary
  decode fallback disabled.
- `prefill-chunk-size=2048`; Dense rows use unquantized KV unless a profile is
  named explicitly.
- One warmup per path. Short-output rows generate 64 tokens and use five
  measured pairs with 15-second cooldowns. Long-output rows generate 256
  tokens and use three measured pairs with 45-second cooldowns.
- The benchmark alternates execution order by pair: even pairs run ordinary
  then MTP, while odd pairs run MTP then ordinary. Aggregate TPS is the batch's
  total generated tokens divided by batched decode time.
- Correctness requires every batch row's generated token IDs to match exactly
  between the paired ordinary and forced-MTP records.

The sanitized Dense command shape was:

```text
target/release/ironmlx-core-bench \
  --model <Qwen3.8-27B-8bit> \
  --mtp-model-dir <Qwen3.8-27B-MTP-8bit> \
  --prompt-file <fixed-prompt> [repeat B times] \
  --chat --ignore-eos --mode scheduler-text \
  --b-max <2|4> --max-tokens <64|256> \
  --mtp-draft-tokens 1 --qwen-fixed-mtp-draft-depth \
  --warmup-runs 1 --runs <5|3> \
  --run-cooldown-ms <15000|45000> \
  --scheduler-baseline-out <ordinary-output> --out <mtp-output>
```

## Dense Summary

| Batch / output | Pairs | Ordinary aggregate TPS P50 | Forced-MTP aggregate TPS P50 | Median gain | Minimum paired gain | TTFT P50 change | TTFT P95 change | E2E P95 change | Peak memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B2 / 64 | 5 | 35.745 | 48.414 | +35.44% | +35.18% | +3.77% | +3.08% | -22.91% | 27.744 GiB |
| B2 / 256 | 3 | 35.722 | 47.502 | +32.98% | +32.92% | +2.60% | +3.21% | -23.82% | 27.768 GiB |
| B4 / 64 | 5 | 66.875 | 86.646 | +29.56% | +27.77% | +1.13% | +1.43% | -17.01% | 27.966 GiB |
| B4 / 256 | 3 | 67.698 | 81.578 | +20.50% | +20.24% | +4.30% | +2.66% | -15.61% | 27.997 GiB |

The frozen +5% absolute aggregate-TPS gates were 37.532 for B2/64, 37.508
for B2/256, 70.219 for B4/64, and 71.082 for B4/256. Every MTP median and
every individual pair exceeded its corresponding ordinary decode result.
Across the MTP records, draft-token acceptance was 61.54% for 64-token output
and 59.38% for 256-token output.

The paired records reported the same peak for ordinary and MTP. A separate B2
memory control measured 27.534 GiB for ordinary decode and 27.744 GiB for MTP,
a 0.76% increase and below the 3% memory gate.

### Dense Paired Rows

| Scenario | Pair | Ordinary TPS | MTP TPS | Gain | Ordinary TTFT ms | MTP TTFT ms | Ordinary E2E ms | MTP E2E ms | Ordinary peak GiB | MTP peak GiB | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| B2 / 64 | 1 | 35.778 | 48.453 | +35.43% | 502.456 | 515.990 | 4024.164 | 3116.428 | 27.744 | 27.744 | Exact |
| B2 / 64 | 2 | 35.739 | 48.414 | +35.46% | 513.208 | 523.730 | 4038.720 | 3126.303 | 27.744 | 27.744 | Exact |
| B2 / 64 | 3 | 35.748 | 48.337 | +35.22% | 504.714 | 527.616 | 4029.435 | 3134.341 | 27.744 | 27.744 | Exact |
| B2 / 64 | 4 | 35.406 | 48.425 | +36.77% | 512.969 | 529.327 | 4071.710 | 3131.267 | 27.744 | 27.744 | Exact |
| B2 / 64 | 5 | 35.745 | 48.321 | +35.18% | 502.993 | 519.927 | 4027.940 | 3127.504 | 27.744 | 27.744 | Exact |
| B2 / 256 | 1 | 35.719 | 47.502 | +32.99% | 501.425 | 515.417 | 14779.555 | 11251.804 | 27.768 | 27.768 | Exact |
| B2 / 256 | 2 | 35.722 | 47.505 | +32.99% | 502.369 | 509.910 | 14779.316 | 11245.628 | 27.768 | 27.768 | Exact |
| B2 / 256 | 3 | 35.727 | 47.487 | +32.92% | 503.863 | 520.388 | 14778.777 | 11260.269 | 27.768 | 27.768 | Exact |
| B4 / 64 | 1 | 66.466 | 85.954 | +29.32% | 1123.175 | 1146.863 | 4914.571 | 4078.679 | 27.966 | 27.966 | Exact |
| B4 / 64 | 2 | 66.848 | 85.994 | +28.64% | 975.986 | 1133.264 | 4745.707 | 4063.707 | 27.966 | 27.966 | Exact |
| B4 / 64 | 3 | 66.875 | 86.646 | +29.56% | 1129.215 | 1108.921 | 4897.419 | 4017.299 | 27.966 | 27.966 | Exact |
| B4 / 64 | 4 | 67.829 | 86.663 | +27.77% | 971.971 | 1085.073 | 4687.191 | 3992.888 | 27.966 | 27.966 | Exact |
| B4 / 64 | 5 | 67.808 | 86.780 | +27.98% | 1096.526 | 984.765 | 4812.904 | 3888.657 | 27.966 | 27.966 | Exact |
| B4 / 256 | 1 | 67.613 | 81.295 | +20.24% | 1028.572 | 990.018 | 16114.385 | 13536.885 | 27.997 | 27.997 | Exact |
| B4 / 256 | 2 | 67.838 | 81.586 | +20.27% | 1070.397 | 1072.851 | 16106.160 | 13575.009 | 27.997 | 27.997 | Exact |
| B4 / 256 | 3 | 67.698 | 81.578 | +20.50% | 962.401 | 1096.962 | 16029.421 | 13600.291 | 27.997 | 27.997 | Exact |

## B4 KV Profile Summary

Each B4 KV profile used 64 output tokens, one warmup, three measured pairs and
15-second cooldowns. Paged adds the paged-prefix-cache configuration; Turbo3,
Turbo4 and K3V4 select the named quantized KV representation.

| Profile | Ordinary aggregate TPS P50 | Forced-MTP aggregate TPS P50 | Median gain | Minimum paired gain | TTFT P50 change | TTFT P95 change | E2E P95 change | Peak memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Paged | 66.871 | 90.654 | +35.57% | +35.56% | +5.64% | +2.11% | -22.77% | 27.966 GiB |
| Turbo3 | 63.873 | 83.011 | +29.96% | +29.90% | -0.01% | +0.68% | -17.78% | 27.918 GiB |
| Turbo4 | 63.581 | 80.127 | +26.03% | +25.97% | -9.49% | -1.96% | -16.84% | 27.920 GiB |
| K3V4 | 63.482 | 80.168 | +26.29% | +25.56% | +15.64% | +3.45% | -15.94% | 27.919 GiB |

All profile pairs produced exact token IDs and positive aggregate-throughput
gains. The Paged and K3V4 TTFT P50 increases must remain visible: this matrix
does not support a claim that every profile improves every latency percentile.
Their TTFT P95 changes remained below 5%, and all four profiles improved E2E
P95.

### B4 KV Profile Paired Rows

| Profile | Pair | Ordinary TPS | MTP TPS | Gain | Ordinary TTFT ms | MTP TTFT ms | Ordinary E2E ms | MTP E2E ms | Ordinary peak GiB | MTP peak GiB | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Paged | 1 | 66.871 | 90.654 | +35.57% | 508.120 | 536.768 | 4276.562 | 3316.571 | 27.966 | 27.966 | Exact |
| Paged | 2 | 66.904 | 90.694 | +35.56% | 504.024 | 514.176 | 4270.617 | 3292.744 | 27.966 | 27.966 | Exact |
| Paged | 3 | 66.825 | 90.613 | +35.60% | 528.796 | 537.962 | 4299.829 | 3319.029 | 27.966 | 27.966 | Exact |
| Turbo3 | 1 | 63.873 | 83.011 | +29.96% | 1114.885 | 1114.026 | 5060.188 | 4149.779 | 27.918 | 27.918 | Exact |
| Turbo3 | 2 | 63.857 | 82.949 | +29.90% | 1024.707 | 1123.356 | 4971.016 | 4161.364 | 27.918 | 27.918 | Exact |
| Turbo3 | 3 | 63.875 | 83.042 | +30.01% | 1114.096 | 992.982 | 5059.300 | 4027.595 | 27.918 | 27.918 | Exact |
| Turbo4 | 1 | 63.252 | 80.063 | +26.58% | 1118.163 | 1007.502 | 5102.256 | 4155.017 | 27.920 | 27.920 | Exact |
| Turbo4 | 2 | 63.581 | 80.127 | +26.03% | 1113.153 | 1105.516 | 5076.630 | 4250.508 | 27.920 | 27.920 | Exact |
| Turbo4 | 3 | 63.608 | 80.128 | +25.97% | 1014.266 | 1001.669 | 4976.057 | 4146.637 | 27.920 | 27.920 | Exact |
| K3V4 | 1 | 63.482 | 80.168 | +26.29% | 980.373 | 1144.270 | 4950.010 | 4287.660 | 27.919 | 27.919 | Exact |
| K3V4 | 2 | 63.034 | 80.175 | +27.19% | 1118.918 | 1001.021 | 5116.783 | 4144.163 | 27.919 | 27.919 | Exact |
| K3V4 | 3 | 63.735 | 80.027 | +25.56% | 980.596 | 1133.945 | 4934.483 | 4282.894 | 27.919 | 27.919 | Exact |

## B1/B2 Regression Against the B2 Commit

The B4 candidate was compared with the retained B2 commit using forced MTP,
64 output tokens, one warmup, five measured runs and 15-second cooldowns. The
candidate's aggregate-TPS P50 changed by +0.39% at B1 and +0.31% at B2. Peak
memory and generated token IDs were unchanged. The minimum row-aligned changes,
-0.18% at B1 and -0.01% at B2, remained within the 3% regression gate.

| Batch | Run | B2 commit TPS | B4 candidate TPS | Change | B2 commit TTFT ms | B4 candidate TTFT ms | B2 commit E2E ms | B4 candidate E2E ms | B2 commit peak GiB | B4 candidate peak GiB | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| B1 | 1 | 22.496 | 22.469 | -0.12% | 343.506 | 447.048 | 3143.955 | 3250.886 | 27.431 | 27.431 | Exact |
| B1 | 2 | 22.481 | 22.442 | -0.18% | 343.017 | 487.497 | 3145.354 | 3294.767 | 27.431 | 27.431 | Exact |
| B1 | 3 | 22.322 | 22.461 | +0.62% | 419.520 | 404.088 | 3241.822 | 3208.968 | 27.431 | 27.431 | Exact |
| B1 | 4 | 22.197 | 22.404 | +0.94% | 474.778 | 347.500 | 3313.049 | 3159.471 | 27.431 | 27.431 | Exact |
| B1 | 5 | 22.360 | 22.448 | +0.39% | 489.525 | 422.439 | 3307.004 | 3228.863 | 27.431 | 27.431 | Exact |
| B2 | 1 | 48.177 | 48.373 | +0.41% | 653.354 | 625.589 | 3268.704 | 3230.349 | 27.744 | 27.744 | Exact |
| B2 | 2 | 48.225 | 48.293 | +0.14% | 655.933 | 637.926 | 3268.693 | 3247.019 | 27.744 | 27.744 | Exact |
| B2 | 3 | 48.311 | 48.304 | -0.01% | 649.273 | 655.843 | 3257.377 | 3264.332 | 27.744 | 27.744 | Exact |
| B2 | 4 | 48.217 | 48.381 | +0.34% | 639.840 | 645.865 | 3253.031 | 3250.168 | 27.744 | 27.744 | Exact |
| B2 | 5 | 48.319 | 48.455 | +0.28% | 638.657 | 655.423 | 3246.329 | 3255.774 | 27.744 | 27.744 | Exact |

The summary latency changes were +0.70% TTFT P50, -1.47% TTFT P95 and -0.78%
E2E P95 at B1; and -0.52% TTFT P50, +0.05% TTFT P95 and -0.19% E2E P95 at
B2.

## Evidence Boundaries

- Raw benchmark JSON files are intentionally not committed. This document
  retains the sanitized per-pair rows, command shape, checkpoint identities,
  revision provenance and correctness outcome.
- The results apply only to the pinned machine, revisions, checkpoints,
  prompt shape, batch widths, output lengths, cache profiles and thermal
  controls above. Aggregate TPS is not a universal per-request TPS promise.
- Dense and KV-profile ordinary/MTP records are interleaved within one loaded
  process. The B1/B2 cross-commit regression records use separate binaries and
  are aligned by run index rather than treated as same-process pairs.
- Exact in this archive means every generated token ID matched for every row
  in each pair. Internal QMM, hidden-state, logits and accepted-prefix-cache
  exactness remain source/test gates rather than benchmark-table columns.
- B2 profile probes with only one measured pair, QMM microbenchmarks, B8
  experiments, and DFlash2 are deliberately excluded.
- Paged and K3V4 improve aggregate throughput and E2E P95, but their TTFT P50
  changes prevent a blanket all-percentile latency-improvement claim.
