# CNN model selection: validation results (`cnn-v1`)

**Validation results (FoR val-A only), not test results.** val-B is not scored until Stage 10
(D9) and the FoR test split is evaluated once at Stage 11 (D31). Nothing here touches the external
set.

Stages 8–9. The official D30 search of the log-Mel CNN (D42), the three final seeds and the bundle
seed. Every number below comes from the committed evidence in
[`docs/audit_evidence/stage8_cnn/`](../audit_evidence/stage8_cnn/), cited by file. Method:
[cnn.md](../cnn.md), [cnn_training.md](../cnn_training.md). Baseline counterpart:
[baseline.md](baseline.md).

## 1. Provenance

| Item | Value | Source |
|---|---|---|
| Code commit | `9be2f2545520432db08a0d083a06a668cd84103a` (merge of PR #50, the Entry 82 fix) | `run_report.json`, `stage8_export_manifest.json` |
| Run | Donia, on Colab (Tesla T4), 2026-10-26; manifest written 11:07:57 UTC | `stage8_export_manifest.json` |
| Notebook | run copy of `notebooks/04_stage8_cnn.ipynb`, on Drive `runs/stage8/04_stage8_cnn.ipynb`, SHA-256 `42c4a219…5d433d5a` (observed) | evidence README |
| Environment | Python 3.13.16, torch 2.11.0+cu130, CUDA 13.0, cuDNN 92700, NumPy 2.5.3, safetensors 0.8.0, wandb 0.31.0 | `run_report.json` `environment` |
| Version | `cnn-v1` | `run_report.json` |
| Tests in the run | 717 passed, 0 failures / errors / skipped | `stage8_export_manifest.json` `pytest_counts` |
| Config | `configs/cnn.yaml`, SHA-256 `d6772100…a8d6ee` | `run_report.json` `cnn_config_sha256` |
| Data | F2 manifest `badfe948…706f8` (D28); F3 profile `7bd35bfd…c8630`, fingerprint `5b4f5849…d68f2` (D39) | `run_report.json` |
| Checkpoints | 3 folders on Drive `checkpoints/cnn/cnn-v1/seed_{0,1,2}/`, not in Git; `checkpoint_sha256` per seed | `run_report.json` |
| W&B | 6 runs, `online`, allow-listed keys only (D40) | `run_report.json` `wandb_runs`, evidence README |

**First attempt.** The first official attempt, from the PR C merge commit `0d88844`, stopped at
the notebook's pytest cell (8 collection errors from a site-packages `tests` package). Nothing ran
after it, and no result was seen (project log Entry 82). This run is the re-run from the fix's
merge commit.

## 2. Consistency checks [verified]

Checked on the Drive download before the evidence was committed (evidence
[README](../audit_evidence/stage8_cnn/README.md), "Verification").

| Check | Result |
|---|---|
| SHA-256 and size of the 9 exported files = `stage8_export_manifest.json` `files` | equal (9/9), none missing or extra |
| `checkpoint_sha256` recomputed (repository function and an independent implementation) = `run_report.json` | equal (3/3) |
| `cnn_config_sha256` = SHA-256 of `configs/cnn.yaml` at the run commit | equal |
| `f2_manifest_sha256` = D28 = SHA-256 of the committed F2 manifest | equal |
| F3 config hash and fingerprint = `stage6_preprocessing/profile_report.json` | equal |
| `code_commit` in `run_report.json`, the 3 `checkpoint.json` and the manifest | all `9be2f25…` |
| Cache rows train / val-A = Stage 4 split sizes 12,404 / 1,330 | equal |
| Selection, early stopping and bundle seed recomputed from the per-epoch histories | equal to the reports |
| Executed notebook vs `notebooks/04_stage8_cnn.ipynb` at the run commit | only the `COMMIT` line differs; execution counts 1–11 |

## 3. Model selection (D30)

Four configurations, seed 0, trained on train (12,404 clips) and selected by lowest val-A EER
(1,330 clips: 660 bonafide, 670 spoof; class counts from the F2 manifest). Source:
`search_report.json`, recomputed from the per-epoch histories.

| Blocks | Learning rate | Best val-A EER | In clips | Best epoch | Epochs run | Stopped by |
|---|---|---|---|---|---|---|
| 3 | 1e-3 | 0.152% (0.0015152) | 1 of 660 | 27 | 30 | 30-epoch cap |
| 3 | 3e-4 | 0.597% (0.0059701) | 4 of 670 | 26 | 30 | 30-epoch cap |
| **4** | **1e-3** | **0.000% ← selected** | **0** | **10** | **15** | patience 5 |
| 4 | 3e-4 | 0.152% (0.0015152) | 1 of 660 | 17 | 22 | patience 5 |

**val-A resolution [verified].** One clip is 1/660 ≈ 0.152% of the bonafide class or
1/670 ≈ 0.149% of the spoof class, and every EER above sits on that grid. The four configurations
differ by **0 to 4 clips**. The lowest EER is unique, so the D30 tie rule (3 blocks before 4, then
1e-3 before 3e-4) was not needed (`selection_report.json`: `tie_rule_decided` false, one
`tied_rows` entry). At this resolution the selection mostly says that 4 blocks at 1e-3 reached
zero errors earlier than the others; it is not a precise ranking of the configurations
[inference].

## 4. Final seeds and bundle seed (D11, D30)

The selected configuration trained with seeds 0, 1, 2; seed 0 reuses the search run (its history
is byte-for-byte the search row). Source: `selection_report.json`.

| Seed | Best val-A EER | Best epoch | Epochs run | val-A ROC-AUC at best epoch | Train loss at best epoch |
|---|---|---|---|---|---|
| **0 (bundle)** | 0.0 | 10 | 15 | 1.000000 | 0.00689 |
| 1 | 0.0 | 13 | 18 | 1.000000 | 0.00243 |
| 2 | 0.0 | 9 | 14 | 1.000000 | 0.00671 |

**Bundle seed.** The rule is the seed with the median val-A EER, ties to the lower seed
(`select_bundle_seed`, D11, Protocol §5). All three EERs are 0.0, so all three tie at the median
and the rule gives **seed 0** (`bundle_seed` 0, recomputed). With three equal EERs the choice is
made by the tie-break, not by performance. Seed 0's checkpoint is the one the ModelBundle will
use (Stage 12).

## 5. Early stopping (D30, D42)

An epoch improves only if its val-A EER is strictly lower than the best so far; the best epoch is
restored (ties: the earliest) and training stops after 5 non-improving epochs or 30 epochs.
Recomputed from the histories, every run follows the rule (§2).

- The 4-block runs stopped by patience: seeds 0, 1, 2 at best + 5 (15, 18, 14).
- A tie counts as non-improving. Seed 1 reached EER 0 at epoch 13 and again at epochs 14, 16, 17
  and 18; the earliest (13) is kept and the run stopped at 18 (`selection_report.json` history).
- After their best epoch, the seeds' val-A EER moved between 0 and about 1 clip (seed 0: 1 clip at
  epochs 11–15; seed 2: 1–6 clips at epochs 10–14). Restoring the best epoch picks a zero-error
  epoch out of this noise [inference].
- Both 3-block runs were stopped by the 30-epoch cap, not by patience, with best epochs 27 and 26
  (§3). See §9 (Limitations).

## 6. Compute

Source: `run_report.json` `peak_gpu_memory_bytes`, history `epoch_seconds`.

| Run | Peak GPU memory | Median epoch time | Training time |
|---|---|---|---|
| b3, lr 1e-3, seed 0 | 1,893,388,288 B (1.76 GiB) | 4.4 s | 135 s (30 epochs) |
| b3, lr 3e-4, seed 0 | 1,893,388,288 B (1.76 GiB) | 4.6 s | 139 s (30 epochs) |
| b4, lr 1e-3, seed 0 | 1,894,575,616 B (1.76 GiB) | 5.0 s | 75 s (15 epochs) |
| b4, lr 3e-4, seed 0 | 1,894,575,616 B (1.76 GiB) | 5.0 s | 110 s (22 epochs) |
| b4, lr 1e-3, seed 1 | 1,894,575,616 B (1.76 GiB) | 5.0 s | 90 s (18 epochs) |
| b4, lr 1e-3, seed 2 | 1,894,575,616 B (1.76 GiB) | 5.0 s | 70 s (14 epochs) |

Batch size 256 (D30, from the Stage 6 benchmark). About 10 minutes of training in total on the T4,
excluding cache building and tests.

## 7. Reproducibility as observed

D42 does not promise byte-identical GPU runs; reproducibility is reported as observed.

- **[verified]** The train and val-A log-Mel caches were built twice in the run and compared
  byte for byte; they were identical (notebook cell 7 assertion). Their hashes are recorded in
  `run_report.json` and the manifest (`cache_sha256`).
- **[verified]** Every run used the D42 settings: seeded `random`, NumPy, torch and torch.cuda,
  `cudnn.deterministic` true, `cudnn.benchmark` false, deterministic algorithms with
  `warn_only` true, `CUBLAS_WORKSPACE_CONFIG` `:4096:8` (`checkpoint.json` `determinism`).
- **Not measured.** Training ran once per configuration and seed. Unlike the Stage 7 baseline
  (fit twice, byte-identical), no repeat run was made, so whether a re-run gives identical
  weights is unknown. The `checkpoint_sha256` values identify these checkpoints; they are not a
  prediction of a re-run.

## 8. Interpretation [hypothesis]

Everything in this section is [hypothesis] unless a source is marked [verified] or [doc].

**Comparison with the baseline [inference].** Baseline-v1's val-A EER is 0.0014925 ≈ 1/670
([baseline.md](baseline.md) §3), about one spoof clip. The CNN's is 0. On val-A the difference is
**one clip**, which this resolution cannot separate from chance, and val-A was used to select both
models, so both values are optimistic. This is **not** a claimed improvement. The pre-registered
in-domain comparison is H1 on the FoR test split (Stage 11), and the main RQ2 comparison is the
single external run under D36.

**Too good?** A val-A EER of 0 for all three seeds, with ROC-AUC 1.000000, is the case AGENTS.md
says to suspect first. It is consistent with in-domain shortcuts the Stage 7 analysis already
described for FoR ([baseline.md](baseline.md) §6, [verified] there):

1. **Bandwidth:** FAKE clips are band-limited (median 99%-energy bandwidth about 3.3 kHz against
   about 5.1 kHz for REAL, `dataset_audit.md` §3.2). A log-Mel input shows a missing high band
   directly. Tested by **H6** (Stage 14).
2. **Loudness:** train-only FAKE-vs-REAL RMS AUC 0.698602 (D39). D42 keeps level information on
   purpose (no per-clip normalization) and says shortcuts are tested by H5–H7 and D41 [doc]; the
   Stage 14 conditions are noise, MP3 and bandwidth (Protocol §9), so no pre-registered test
   targets loudness directly.
3. **Silence:** train-only FAKE_other-vs-REAL silence-fraction AUC 0.749901 (D39). Tested by **H7**
   (Stage 16).
4. **MP3 traces:** tested by **H5** (subgroup recall, Stage 11/16).
5. **Same-source split:** train and val-A come from one FoR-2sec package (same generators,
   probably the same speakers; `dataset_audit.md` §8), so val-A is an in-domain upper bound.

These are possible explanations, not findings: nothing here shows which cue the CNN uses. They are
to be tested by H5–H7 and the Stage 14 runs (D41), and the external run (D36) is the real test
of generalization; loudness has no pre-registered test (item 2). They are not explained away.

**What was not done:** no grid, epoch cap, feature or seed change after seeing these results
(Protocol §12 change control). No val-B, test or external score was computed.

## 9. Implications

- **Selection stands.** b4 / lr 1e-3 and bundle seed 0 were chosen by the pre-declared D30 and D11
  rules; nothing changes.
- **Stage 10** fits the risk policy and calibration on val-B with the seed-0 checkpoint. With a
  CNN this separable in-domain, the val-B thresholds may be weakly determined, as the baseline's
  D29 threshold was ([baseline.md](baseline.md) §8.1) [hypothesis].
- **H1** may hit a ceiling: with both models near 0% EER in-domain, the FoR test may not show a
  difference whose CI excludes 0 [hypothesis]. H1 is kept unchanged and reported as pre-registered.
- **RQ2:** D36 (both frozen models on the same external windows, paired difference) remains the
  main test.

## 10. Limitations

1. **Resolution.** val-A cannot resolve error rates below about 0.15%: the four configurations
   differ by 0–4 clips and the three seeds by none.
2. **Selection on val-A.** The reported EERs are on the split used for selection and early
   stopping, so they are optimistic; val-B (Stage 10) is the first split not used for these
   choices.
3. **Epoch cap [hypothesis].** The 3-block runs had their best epochs at 27 and 26 and were still
   moving at the 30-epoch cap (val-A EER between 1 and 11 clips in epochs 21–30), so the cap may
   have limited them. The selection is valid under the pre-declared D30 protocol, which fixes the
   cap at 30; nothing changes.
4. **Bundle seed by tie-break.** All three seeds have EER 0, so seed 0 was chosen by the
   lower-seed tie-break, not because it was the median performer in any measurable sense.
5. **Single runs.** One run per configuration and seed (D30); no repeat-run reproducibility (§7).
