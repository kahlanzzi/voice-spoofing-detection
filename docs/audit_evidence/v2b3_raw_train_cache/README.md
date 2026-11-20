# Track V V2b-3: raw train cache evidence

The metadata file of the raw FoR train cache that model v2 trains from (`src/features/raw_train_cache.py`,
D50 (1)). Only `train_raw.json` is committed; the 1.6 GB `train_raw.npz` stays on Drive at
`processed/v2/raw_train_cache/` (AGENTS.md rule 7). The file was supplied by Donia as written by the
build and is committed unchanged; it was not regenerated.

## Run identity

| Item | Value |
|---|---|
| Runtime | Colab T4, runtime 2026.10 [doc: supplied with the file] |
| Code commit | `37b236b936afffd150bde5d9ad9020fb68952f2d` (recorded in the JSON) |
| F2 manifest SHA-256 | `badfe948fb7b833357783dd18be00f6bb36253cf8699dc4c4beb4df2b07706f8` (recorded in the JSON; the approved D28 value) |
| FoR archive | not recorded in this JSON: the cache format records the F2 manifest hash and its own npz hash only; every clip's file hash was checked against F2 before decoding (`read_verified_audio`) |

## Contents [verified against the committed JSON]

| Field | Value |
|---|---|
| `split` | `train` |
| `n_rows` | 12,404 (`n_spoof` 6,244 FAKE / `n_bonafide` 6,160 REAL) |
| `sample_rate` | 16,000 Hz |
| `total_samples` | 396,928,000 = 12,404 × 32,000 (every clip 32,000 samples) |
| `n_all_zero_clips` | **0** |
| `npz_sha256` | `026a917debfd262ff3e507ea073fa33ef14280fd8dbe235366906f549b4a75e8` |

`train_raw.json` SHA-256: `34065507847689817d9b29af6972108d9c2fc8d42edc85cbad3908c8bc1d25a9`.

## What it decides

- **No all-zero clip:** `n_all_zero_clips = 0`, so `Augmenter`'s `recorded_noise` never meets a silent
  train clip and decision **D62 is not needed** (it is recorded as not used in
  `docs/decisions/README.md`).
- **Pin for the search:** `python -m src.tuning.v2_search search` and `rehearse` refuse a raw train
  cache whose npz SHA-256 differs from `npz_sha256` above or whose `train_raw.json` differs from this
  file byte for byte (`check_raw_cache_pin`). `run_search`'s own checks (F2, paths, labels, groups,
  the recount of all-zero clips) still run.
- `tests/features/test_v2b3_raw_train_cache_evidence.py` (torch-free) checks this file's SHA-256 and
  its counts.
