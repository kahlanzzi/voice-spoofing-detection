# Track V V1c real-build evidence

This directory records the official V1c selection-set materialization and training-half augmentation
resource-cache build. The run used notebook 08 on Colab runtime 2026.10 with a Tesla T4. Only the
small JSON evidence is committed: generated NPZ files, audio, corpora and caches remain on Drive.

## Run identity

| Item | Value |
|---|---|
| Code commit | `d5ffe18da5912b4a4d4174f738dbe1b2d5603fcd` (PR #159 merge commit) |
| Runtime / GPU | Colab `2026.10` / Tesla T4 |
| Python / torch | `3.13.16` / `2.11.0+cu130` |
| numpy / scipy / soundfile | `2.5.3` / `1.18.1` / `0.14.0` |
| FFmpeg | `6.1.1-3ubuntu5` |
| Result | selection set and resource cache built and verified |

The build report records that both outputs were built in this session. The selection builder checked
all 6,650 waveforms against the D64 recipe before writing the five condition files.

## Selection set S

- 1,330 val-A sources for each of the five frozen conditions.
- Recipe SHA-256:
  `70a2105425838c64b9bf674fbdbf3f3e121fc97ae2d6d3bf647c9aceff12d0a1`.
- Manifest SHA-256:
  `7aaa69f916027f52bfa29a8cce935f48d3a9ac7d9481cf8a0fb04fdebfdc092e`.
- The five generated NPZ hashes are recorded under `files` in
  [`selection_set_manifest.json`](selection_set_manifest.json); the NPZ files stay on Drive.

## Training-half resource cache

- MUSAN noise resources: 465; all-zero resources: 0.
- RIRS resources: 30,162; all-zero resources: 0.
- Sample rate: 16,000 Hz; deterministic split seed: 2026.
- `training_resources.npz` SHA-256:
  `becc00609a667a68eda02a5184466e42cdb9a232912607639ee69e67f2e393c8`.
- Metadata SHA-256:
  `0afc6e439d3201fd26dae921db62d082ae94e4dd214cbb404e19526f6f11aa24`.

The metadata binds the cache to the committed MUSAN and RIRS inventory-file hashes and archive pins.
The ZIP64 path introduced by PR #159 was exercised by the real 2.10 GiB packed RIR member.

## Committed files

| File | Bytes | SHA-256 |
|---|---:|---|
| `selection_set_manifest.json` | 2,696 | `7aaa69f916027f52bfa29a8cce935f48d3a9ac7d9481cf8a0fb04fdebfdc092e` |
| `training_resources.json` | 685 | `0afc6e439d3201fd26dae921db62d082ae94e4dd214cbb404e19526f6f11aa24` |
| `v1c_build_report.json` | 2,026 | `dfd53330cc6521c377d83b46b09e6cc6573548cccbe9f496c7eeb14b3578ffff` |

`tests/augmentation/test_v1c_build_evidence.py` checks these exact bytes, the cross-file bindings,
the official runtime, resource counts, zero checks, inventory hashes and generated-file hash shape.
The v2 rehearsal and search are independent later runs and are not evidence in this directory.
