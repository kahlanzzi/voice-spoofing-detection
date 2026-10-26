# Stage 8 CNN run evidence (`cnn-v1`)

**Validation evidence (FoR train and val-A), not test results.** val-B is first scored in Stage 10
and the FoR test split once in Stage 11 (D9, D31).

The official Stage 8–9 run: notebook 04 on Colab (Tesla T4), from commit
`9be2f2545520432db08a0d083a06a668cd84103a` (merge of PR #50, the Entry 82 fix), 2026-10-26.
Results and interpretation: [results/cnn_selection.md](../../results/cnn_selection.md). Method:
[cnn.md](../../cnn.md), [cnn_training.md](../../cnn_training.md).

## What is here

Byte-identical copies of the files the notebook exported to Drive
`MyDrive/voice-spoofing-detection/processed/metadata/stage8/`:

| File | Bytes | SHA-256 | Content |
|---|---|---|---|
| `search_report.json` | 22,220 | `3d72956bbe5757943d2572e40aff4a7cf8f80778c3e5d27e25e3fac6eea9916f` | the 4 D30 search runs (seed 0), per-epoch history |
| `selection_report.json` | 11,006 | `5c55f2da6c0c12ef9bf344ab5684e547f5e9776110bda5ebc52dbcaf50ba4c4a` | selected configuration, tie rule, seeds 0–2 with history, bundle seed |
| `run_report.json` | 2,082 | `c0e1be359949cceffde74444c026b2f55607134d3ada3bc88fd96f298db89b23` | provenance: commit, environment, config/F2/F3/cache hashes, `checkpoint_sha256`, peak GPU memory, W&B run names |
| `stage8_export_manifest.json` | 2,741 | `97544c54b0a7374c52e13b61940d443c8de49f6df8336025b4dde021a6e6237e` | SHA-256 and size of the 9 exported files, cache hashes, pytest counts |

The manifest cannot list its own hash. The notebook checked it after the Drive remount against the
text it had just written (cell 11: "10 files" = the 9 listed + the manifest) but did not print the
hash, so the manifest SHA-256 above is **observed only**: computed on the Drive download, not
recorded by the run.

## What stayed on Drive (not in Git, AGENTS.md rule 7)

Paths under `MyDrive/voice-spoofing-detection/`. Hashes from `stage8_export_manifest.json`,
recomputed on the download (below).

| Path | Bytes | SHA-256 |
|---|---|---|
| `checkpoints/cnn/cnn-v1/seed_0/checkpoint.json` | 1,350 | `dc0f8b0e9a8549583478db7d1a94e24d01668af9459bfd692895abcc83c98932` |
| `checkpoints/cnn/cnn-v1/seed_0/weights.safetensors` | 396,084 | `6722f8587b54aaea649456a04ec3074fb30883d549bf48ea6a6ffa9111dcf835` |
| `checkpoints/cnn/cnn-v1/seed_1/checkpoint.json` | 1,350 | `15d0ed5d0fd92acbd724a09332120d01e6ef72b6ecf18170e96d27561299e32e` |
| `checkpoints/cnn/cnn-v1/seed_1/weights.safetensors` | 396,084 | `b687b780c537aec10ab3e239d8ebe76dd4f2750ce855df51e97fd351520a3d4d` |
| `checkpoints/cnn/cnn-v1/seed_2/checkpoint.json` | 1,349 | `2611eeb7fe756fcd5bf16a9e377ca17653b8af4d5911cedf95b6a176ba1c6250` |
| `checkpoints/cnn/cnn-v1/seed_2/weights.safetensors` | 396,084 | `0dd41a6e7fb5561c47cf158fe6b7bddccc536b46b85748f14443e036df568709` |
| `runs/stage8/04_stage8_cnn.ipynb` (executed notebook) | 70,401 | `42c4a21996cb2c3af33af7d029de38f7eddb36f7cf1391d3243a28d205433d5a` |

The executed notebook is not committed: it is the run copy of
[`notebooks/04_stage8_cnn.ipynb`](../../../notebooks/04_stage8_cnn.ipynb), and its outputs include
a personal W&B account name. Its hash is observed on the Drive download (the run does not record
it). The log-Mel caches (`train`, `val-A`) were built in the Colab runtime only; their hashes are
in the manifest `cache_sha256` and `run_report.json`, as recorded evidence (the cache files are
not on Drive).

### `checkpoint_sha256`

One hash per checkpoint folder, defined in `src/models/cnn_checkpoint.py:187-188`:

```python
lines = "".join(f"{name}:{hashlib.sha256(data[name]).hexdigest()}\n" for name in FILES)
return hashlib.sha256(lines.encode("ascii")).hexdigest()
```

with `FILES = ("checkpoint.json", "weights.safetensors")`. It hashes both files, so it differs
from the raw `weights.safetensors` hash, and a change to either file changes it. It is recorded
only in `run_report.json` (`checkpoint_sha256`); `checkpoint.json` does not contain it (not one of
its `METADATA_KEYS`, and a file cannot hold its own hash).

| Seed | `checkpoint_sha256` |
|---|---|
| 0 (bundle seed) | `c33eb82086d3bc40f3dedf1356addc6e5c1924ca2e517b0d4527b447e1958932` |
| 1 | `4bee66b6ee54323d31208a2ee1f47970622691e77fa371fd424830225fca4dac` |
| 2 | `476c4bee2d735de71236c202324274e2b68d0d9cd9ccdc68bac8691be36ba110` |

## W&B runs (D40)

Entity `voice-spoofing-detection-must`, project `voice-spoofing-detection`, mode `online`
(`run_report.json`). Run IDs from the notebook output (cell 9):

| Run name | Run ID | Role |
|---|---|---|
| `cnn-v1-b3-lr0.001-s0` | `cp3erkf2` | search |
| `cnn-v1-b3-lr0.0003-s0` | `4zclphxr` | search |
| `cnn-v1-b4-lr0.001-s0` | `zfkmpvj2` | search; reused as final seed 0 |
| `cnn-v1-b4-lr0.0003-s0` | `k6p50ij6` | search |
| `cnn-v1-b4-lr0.001-s1` | `cw0zr0av` | final seed 1 |
| `cnn-v1-b4-lr0.001-s2` | `hy5s1yxq` | final seed 2 |

Logged keys, as printed in the W&B run summaries: per epoch `epoch`, `train_loss`, `val_a_eer`,
`val_a_auc`, `epoch_seconds`; summary `best_epoch`, `best_val_a_eer`, `epochs_run`,
`peak_gpu_memory_bytes`. These are exactly `EPOCH_KEYS` and `SUMMARY_KEYS` of
`src/training/wandb_logger.py`, which refuses any other key and any key naming val-B, test or
external data.

## Verification (PR D section 0, 2026-10-26)

Done locally on the Drive download (`C:\Users\donia\Desktop\stage8_drive\`, outside the repo),
against the code at `9be2f25`. Read-only; nothing in the run was changed.

1. **[verified] Export hashes.** The SHA-256 and size of all 9 files listed in the manifest match;
   no file missing, none extra.
2. **[verified] `checkpoint_sha256`.** Recomputed for each seed with the repository function and
   with an independent implementation of lines 187–188; both equal `run_report.json` for seeds 0,
   1 and 2.
3. **[verified] Provenance.**
   - `cnn_config_sha256` `d6772100…` = SHA-256 of `configs/cnn.yaml` at `9be2f25`.
   - `f2_manifest_sha256` `badfe948…706f8` = D28 = SHA-256 of
     `data/manifests/for2sec_frozen_manifest.csv` at `9be2f25`.
   - `profile_config_sha256` `7bd35bfd…c8630` (= `configs/preprocessing.yaml`) and
     `profile_fingerprint` `5b4f5849…d68f2` = `stage6_preprocessing/profile_report.json` (F3, D39).
   - `code_commit` = `9be2f25…` in `run_report.json`, all three `checkpoint.json` and the manifest.
   - The 3 `checkpoint.json` files differ only in `seed` (twice) and `best_epoch`.
4. **[verified] Class sizes.** val-A in the F2 manifest: 670 FAKE (spoof) and 660 REAL (bonafide),
   1,330 rows = `run_report.json` `caches.val-A.n_rows`; train 12,404 rows. The cache encodes
   1 = spoof / FAKE (`src/features/logmel_cache.py`). One clip is 1/660 ≈ 0.152% of bonafide or
   1/670 ≈ 0.149% of spoof; every val-A EER in the reports is on that grid.
5. **[verified] Selection, recomputed from the per-epoch histories** (not the summary fields):
   each run's best epoch, best EER and epochs run follow the D42 rule (strictly lower improves,
   ties keep the earliest, stop after 5 non-improving epochs or 30 epochs); the lowest EER is
   unique (b4, lr 1e-3), so the D30 tie rule was not needed; seed 0's history is byte-for-byte the
   search run's; the median rule (ties to the lower seed) gives bundle seed 0. All equal the
   reports.
6. **[verified] Notebook.** The executed copy has the same 23 cells as `notebooks/04_stage8_cnn.ipynb`
   at `9be2f25`; markdown identical; the only code difference is the `COMMIT` line
   (`'REPLACE_WITH_APPROVED_COMMIT'` → `'9be2f2545520432db08a0d083a06a668cd84103a'`). Execution
   counts 1–11 in order, no error output. pytest `717 passed` (0 failures, errors, skipped); the two
   log-Mel cache builds were byte-identical (cell 7 assertion); "Drive verified after remount: 10
   files match the manifest." No secret-like strings in the outputs (W&B key loaded from Colab
   secrets, not printed; the only 40-hex strings are the commit and the FoR revision).
