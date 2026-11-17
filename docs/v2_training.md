# Model v2 training (Track V, PR V2b-1)

The code that trains the three D49 candidates with the D50 augmentation, selects among them on the
selection condition set S, and measures their batch sizes. **Code and synthetic tests only:** no
dataset, cache build, Drive, W&B online run, training run, FoR test split or DEEP-VOICE access in
this PR. No new decision: everything follows D49, D50/D56, D61 and Protocol §9.

Decisions: D9 (val-A only), D40 (W&B), D42 (training structure), D49 (candidates, selection, seeds),
D50/D56 (augmentation, S), D58 (training runtime), D61 (implementation details). Candidates and the
v2 checkpoint: [v2_candidates.md](v2_candidates.md). Augmentation: [augmentation.md](augmentation.md).

## Order of one training item

```
raw train waveform (as decoded, 16 kHz mono)
  -> Augmenter.augment(split="train", label, seed)      D50 (1), (2); D56
  -> preprocess_waveform (unchanged F3 profile)          D39
  -> aasist:          the F3 waveform (32000,)
     cnn-aug / crnn:  log_mel_spectrogram -> apply_bin_stats(clean-train mean, std)   D61 (5)
```

REAL and FAKE clips go through the same augmentation probabilities (the label is passed but does
not change the draw). Validation data is never augmented: S holds fixed degraded copies of val-A.

## Inputs and why each exists

| Input | Module | Why |
|---|---|---|
| Raw train cache (`train_raw.npz` + `.json`) | `src/features/raw_train_cache.py` | Augmentation acts **before** F3, so the post-F3 Stage 8 caches cannot be used, and decoding 12,404 files from Drive each epoch would dominate run time. Train split only; every clip as decoded (hash-checked, 1-D, 16,000 Hz enforced). |
| Clean-train statistics | Stage 8 train log-Mel cache + `fit_bin_stats` | D61 (5): per-bin mean/std on the **clean** F3 log-Mel of FoR train (augmented features change every epoch, so stats fitted on them would not be reproducible). Its `npz_sha256` is recorded as `train_stats_npz_sha256` (`null` for aasist). Its paths and labels must equal the raw cache's. |
| Selection set S | `src/training/selection_set.py` | D50 (4): five fixed val-A conditions; M = mean val-A EER over them (D49). |
| Corpus halves (noise, RIR) | `AugmentationResources`, passed in | D50 (5): the audited training halves; the loader belongs to V1b (Malak), so `run_search` takes `resources` as an argument. |

Memory (float32): raw cache ≈ 12,404 × ~2 s × 16,000 × 4 B ≈ **1.6 GB** (held once; the loader
hashes the npz by streaming); Stage 8 train log-Mel ≈ 1.25 GB, read once for the statistics and then
released; S ≈ **0.85 GB** in waveform form (aasist) or **0.34 GB** in log-Mel form (cnn-aug, crnn),
one form in memory at a time. With fork workers (Colab) the raw cache is shared copy-on-write; with
spawn workers (Windows) it is pickled to each worker.

## S layout (contract with the V1b builder)

One folder, outside the repository (AGENTS.md rule 7):

- `selection_set_manifest.json` — `schema_version` 1; `source_split` `"val-A"`; `seed` 2026;
  `conditions`: `clean`, `noise_10db`, `reverberation`, `opus_16`, `narrowband` in this order, each
  `{name, parameters}` (`parameters` a JSON object, e.g. SNR, codec option, corpus half);
  `f2_manifest_sha256`; `profile_fingerprint`; `profile_config_sha256`; `augmentation_config_sha256`;
  `code_commit`; `runtime` (object of strings: Colab runtime version, python, numpy, librosa, CPU
  flags, free text allowed); `n_rows`; `files` = `{relative path: sha256}` for every other file.
- `<condition>.npz` (deterministic writer) — `waveforms` float32 (n, 32000) = the F3 waveform;
  `log_mel` float32 (n, 1, 64, 197) = the F3 log-Mel of that waveform, **not** standardized;
  `labels` int8 (1 = FAKE); `paths` unicode, sorted, unique; `groups` unicode (F2 group IDs).
- Optional sidecars (e.g. a per-clip corpus file index), allowed only if listed in `files`.

`selection_set_sha256` = SHA-256 of the manifest file (it holds every file hash).

`write_selection_set(out_dir, arrays_by_condition, manifest_fields, *, profile)` validates what the
loader checks and writes the manifest last; the builder supplies `conditions`, the four hashes,
`code_commit` and `runtime`; the writer adds `schema_version`, `source_split`, `seed`, `n_rows` and
`files`. `profile` is a required keyword so the writer can check the fingerprint and recompute the
log-Mel. `load_selection_set` is the contract either way. It checks: every file hash, no unlisted
file, strict keys, `source_split` val-A, seed 2026, condition names and order, identical
paths/labels/groups across conditions, every path a val-A F2 row with the same label and group,
shapes/dtypes/finite values, the F3 fingerprint, a byte-exact recompute of the log-Mel of the first
8 rows of each condition from the stored waveform (**S must be built on the training runtime**, Colab
T4 2026.10, D58), the val-A size (1,330 = 660 REAL / 670 FAKE), and optionally that the clean
condition equals the Stage 8 val-A log-Mel cache byte for byte (free known-answer check:
`4502eea2…` on T4).

## All-zero clips (gate before training)

The raw cache counts the train clips whose decoded samples are all exactly 0.0 and records the count
as `n_all_zero_clips` in `train_raw.json` (strict key, recounted from the samples on load); `build`
and `verify` print it and the first 10 such paths (or `none`). The cache still builds with such
clips: nothing is dropped.

**Why it matters [verified]:** `Augmenter._recorded_noise` raises when `signal_rms <= 0`
("recorded-noise mixing requires non-silent signal and noise"). Training draws `recorded_noise` for about 16 % of items each
epoch, so a run that meets an all-zero clip on that family stops, deterministically, on the same
item every time. Malak's S builder has the same issue on val-A for the `noise_10db` condition.

**Gate:** if `n_all_zero_clips > 0` when the cache is built on the training runtime (V2b-3), a
decision (**D62**) is needed **before any training**, covering both the train cache and S. The
options (e.g. how the noise family treats a silent clip) are for that decision, not for this code.
`run_search` refuses such a cache, naming the count.

## Item seed

`item_seed(aug_seed, run_seed, epoch, clip_index) =
SeedSequence([aug_seed, run_seed, epoch, clip_index]).generate_state(1)[0]`, with `aug_seed` =
`AugmentationConfig.seed` (2026, enforced by its loader). The sampler yields `(epoch, clip_index)`
pairs (one `torch.randperm` per epoch from a generator seeded with the run seed, as `train_one`) and
the dataset derives the seed itself. Nothing reads global NumPy/Python/torch RNG state, worker ids
or timing, so an item is the same for any `num_workers` (tested with 0 and 2 spawn workers and two
batch sizes) and when regenerated alone. Two details make this hold:

- The DataLoader gets a private generator: by default each new iterator draws its worker base seed
  from the global torch RNG, and persistent workers draw only once, which would shift the dropout
  masks between `num_workers=0` and `> 0`.
- `PreprocessingProfile` and `AugmentationConfig` hold `MappingProxyType` fields, which do not
  pickle; the dataset's `__getstate__` stores them as plain dicts and `__setstate__` rebuilds equal
  frozen objects (`src/preprocessing` and `src/augmentation` unchanged).

## Family counts

Each epoch counts the items per family (`clean`, `recorded_noise`, `reverberation`, `codec`, `gain`,
`rawboost`) and per codec option (the 7 D56 ids). They are logged to W&B and stored in the history,
so the D50 mix (0.2 clean, 0.16 per family, 1/7 per codec option) can be checked on every run.

## Training loop and early stopping (`src/training/train_v2.py`)

`train_v2(candidate, *, config, run_seed, raw_cache, selection, augmenter, profile, mean, std,
max_epochs, patience, batch_size, num_workers, logger, device, resume_dir=None, provenance=None)` has
no val-B, test or external input (`resume_dir` and `provenance`: see Resumability).

1. `set_seed(run_seed)` first; `build_model(candidate)`; optimizer from the config: cnn-aug / crnn
   Adam (lr 1e-3, weight decay 0), constant; aasist Adam (lr 1e-4, betas 0.9/0.999, weight decay
   1e-4, no AMSGrad) with `cosine_lr(step, 30 × steps_per_epoch, 1e-4, 5e-6)` set before every
   optimizer step (global step from 0; the total uses 30 epochs, not the epochs run; D61 (3)).
   `BCEWithLogitsLoss`, no weights, float32. mean/std required for log-Mel models, refused for aasist.
2. Each epoch: one shuffled pass over all of train (last batch may be smaller); then eval mode and
   no_grad scoring of the five S conditions (log-Mel S standardized once with the clean stats); EER
   and ROC-AUC per condition on the raw logits; **M = mean of the 5 EERs**.
3. Early stopping (D49 under D42): best starts at +inf; an epoch improves only if M is strictly
   lower; patience 5; max 30 epochs; a tie keeps the earliest epoch; the best epoch's state_dict is
   returned (CPU copy), not the last.
4. History per epoch: every logged field (below) including the family and codec counts.

W&B (`src/training/v2_wandb_logger.py`, D40): the v1 logger's checks reused (v1 unchanged). Config =
the `v2_candidates.yaml` fields + `candidate`, `seed`, `batch_size`, `num_workers`, `code_commit`,
`profile_fingerprint`, `f2_manifest_sha256`, `raw_train_cache_npz_sha256`, `selection_set_sha256`,
`augmentation_config_sha256`, `train_stats_npz_sha256` (null only for aasist). Per epoch: `epoch`,
`train_loss`, `selection_m`, `val_a_eer_<condition>` ×5, `val_a_auc_<condition>` ×5,
`learning_rate`, `epoch_seconds`, `train_seconds`, `selection_seconds`, `family_<family>` ×6,
`codec_<option>` ×7. Summary: `best_epoch`, `best_selection_m`, `epochs_run`,
`peak_gpu_memory_bytes`. Keys containing `val_b`, `val-b`, `test` or `external` are refused.

## Selection and bundle seed (`src/tuning/v2_search.py`)

Each candidate once with seed 0 → lowest best M wins; ties: fewer trainable parameters (cnn-aug
97,761 < aasist 297,705 < crnn 319,585), then candidate order (cnn-aug, crnn, aasist) → the winner is
trained with seeds 1 and 2 → bundle seed = `select_bundle_seed` over the three M values (median,
ties to the lower seed). Before reading anything: a clean worktree (`code_version`), argument
checks and `require_batch_sizes()`. The corpus and ffmpeg pins must be complete and
`verify_pin()` is called once (the codec family is never skipped). The raw cache's paths, labels
and groups must match F2, it must hold no all-zero clip (`n_all_zero_clips > 0` is refused:
decision D62 needed before training), its paths and labels must equal the Stage 8 train cache's, and
S's augmentation-config, F2 and profile-config hashes must equal the current files'. The Stage 8
val-A log-Mel cache (`stage8_val_a_logmel_dir`, required; same F2 and profile checks as the train
cache) is the known answer: S's clean condition must equal it byte for byte for both forms
(`clean_reference_logmel`); its `npz_sha256` is recorded as `val_a_cache_npz_sha256` in the
provenance and `run_report.json`.

## Resumability

**Between runs.** `out_dir/runs/<candidate>_seed<k>/` is either complete (`checkpoint/` from
`save_v2_checkpoint` with full metadata — `selection_m`, `augmentation_config_sha256`,
`selection_set_sha256`, `train_stats_npz_sha256` or null, `determinism` with `num_workers` — plus
`run.json` with the history, the checkpoint hash and the provenance hashes) or absent. A run is
built in `<name>.partial/` and renamed when complete. Every complete run's checkpoint hash is
verified, and a complete run is reused only if its provenance (code commit, v2 and augmentation
config hashes, S, raw cache, stats, F2, F3) equals the current one; otherwise the search refuses.
The final `search_report.json`, `selection_report.json` and `run_report.json` are written once (run
report last); if they exist they are verified against the runs and returned, never overwritten.

**Within a run, per epoch (PR V2b-R, Entry 184; Execution Plan v3.0 §9 V3, §13).** A worst-case run
(30 epochs × ~10–17 min on the T4, Entry 173) may not fit one Colab session. After every completed
epoch (S scored, best and patience updated, epoch logged) `train_v2` writes the full training state
to `<name>.partial/resume/` (`src/training/v2_resume.py`); no pickle, no `torch.save`:

- `state.safetensors`: `model.*` (current weights), `best.*` (best epoch's weights), the Adam state
  flattened as `opt.<param_index>.exp_avg` / `.exp_avg_sq` / `.step`, `rng.torch_cpu`,
  `rng.torch_cuda` (CUDA only), `rng.sampler` (the EpochSampler generator);
- `state.json` (canonical: sorted keys, indent 2, NaN refused): schema version, candidate, run seed,
  last completed epoch, global optimizer step, best M and epoch, patience counter, full history,
  NumPy and Python `random` states, determinism dict, W&B run id, provenance, run settings (max
  epochs, patience, batch size, steps per epoch, train size, device type), peak GPU memory so far,
  and the SHA-256 of `state.safetensors`.

The write is atomic: the new state goes to `resume.tmp/`, then `resume/` → `resume.old/`,
`resume.tmp/` → `resume/`, and `resume.old/` is removed. At any interruption one complete state
remains (`resume/`, or `resume.old/`, which is renamed back on read).

On restart, `run_search` scans `runs/*.partial`: one **with** a resume state of the same candidate,
seed and provenance is resumed; one **without** (stopped before its first epoch ended) is removed
(the count is printed); one whose state is invalid (hash, keys, NaN, canonical form, counters) or has
another provenance is **refused and kept**, since trained epochs are never deleted silently (inspect
it, then remove it by hand to retrain). `train_v2` verifies the state again (also run settings and
the W&B run id), calls `set_seed(run_seed)` and builds as usual, then loads weights, Adam state and
counters and finally overwrites every RNG state; it continues at the next epoch, and the D61 cosine
schedule continues from the saved global step. A state whose run had already stopped (patience or
max epochs) returns the result without training. Epochs already logged are not logged again; in
online/offline mode the same W&B run is re-opened (`wandb.init(id=…, resume="must")`), so the logger
mode must not change between sessions. `resume/` is deleted only after the renamed run folder
verifies (a leftover is deleted on the next start, after verification).

**Guarantee.** On CPU a resumed run gives exactly the uninterrupted run's best `state_dict`, best
epoch, best M and history (all fields except `*_seconds`), for every candidate and for `num_workers`
0 or 2 (tested). On GPU, reproducibility is reported as observed (D42), as for uninterrupted runs.
The `determinism` dict and the timing fields describe the session that finished the run.

**Operating rule:** the provenance includes the code commit, so a search must run from one fixed
commit from start to finish (the notebook checks out that commit); pulling a newer main mid-search
makes every saved state refuse to resume.

**What a Colab disconnect costs:** at most the epoch in progress (~10–17 min on the T4). A narrow
case: a stop between logging an epoch to W&B and writing its state retrains that epoch and logs it
once more. There is no CLI yet: V2b-4 adds it once the V1b corpus loader exists.

## Batch-size benchmark (`src/tuning/v2_benchmark.py`)

`python -m src.tuning.v2_benchmark --out <dir> --device cuda|cpu [--candidates ...]
[--batch-sizes ...] [--throughput-items N] [--codec-unpinned-timing]` writes `benchmark.json` to a
new folder outside the repository; it does not edit the config (V2b-2 does).

**Result (V2b-2, Entry 173):** on the Colab T4 (runtime 2026.10, commit `9c46baf`), two GPU runs
chose **`cnn-aug` 256, `crnn` 256, `aasist` 32**, with byte-identical peaks (`aasist` 64 peaks at
13,895,315,968 B > 80 % of 15,637,086,208 B; 128 and 256 OOM). The sizes are in
`configs/v2_candidates.yaml`; evidence, rows table and the unpinned CPU throughput (planning only)
in [audit_evidence/v2b2_benchmark/](audit_evidence/v2b2_benchmark/README.md);
`tests/training/test_v2_batch_size_evidence.py` keeps the config equal to it.

- **GPU:** the Stage 6 rule (`choose_batch_size`, 80 % of VRAM, candidates 16–256, 3 warm-up + 10
  measured steps, from `configs/preprocessing.yaml`): per candidate and batch, `set_seed(0)`,
  `build_model`, synthetic input of its shape, random 0/1 targets, `BCEWithLogitsLoss`, forward +
  backward **+ the candidate's real optimizer step**. **Difference from Stage 6** (D39: forward and
  backward only, no optimizer): Adam keeps two state tensors per parameter, which is real training
  memory. The selection rule is unchanged and the measured peak can only be ≥ Stage 6's, so the
  chosen batch is at least as safe.
- **CPU:** per-item transform time on synthetic 2 s audio with synthetic noise/RIR resources, per
  family (augment) and per stage (augment / F3 / log-Mel + stats), mean and p95, then an estimated
  epoch time = 12,404 × (0.2 × clean + 0.16 × each family + F3 [+ log-Mel]) ÷ workers (1, 2, 4).
  The estimate assumes linear scaling with workers, which Colab's 2 vCPUs (often one physical core)
  may not give.
  The S scoring time is not estimated (GPU). Real RIR lengths change the reverberation cost.
- **Codec:** through `FFmpegCodecBackend` if `ffmpeg` is pinned; otherwise only with
  `--codec-unpinned-timing`, via `UnpinnedTimingCodecBackend` (defined in the benchmark module only),
  which records the ffmpeg version instead of verifying it; the result is marked
  `codec_timing: "unpinned"`. `train_v2`, `v2_data` and `v2_search` never import the benchmark
  (tested). `round_trip` runs `ffmpeg -version` on every call; the benchmark measures that as part of
  the codec cost and does not change it.

## Determinism of crnn on GPU

`set_seed` asks for deterministic algorithms (warn only) and deterministic cuDNN. Whether the cuDNN
LSTM of crnn is bit-reproducible on the T4 is **not claimed here**: it is measured in the V2b-4
rehearsal (two identical runs) and reported only as observed (D42).

## v2 risk policy (V4, `src/tuning/v2_risk_run.py`)

Protocol §6 unchanged, on clean val-B, for the bundle seed's checkpoint only (D45, D49 addendum).
`risk_run.py` (Stage 10) stays v1-only; the v2 run reuses its val-B loader, diagnostics and file
formats, and `fit_risk_policy` as is. Code only so far: the real run follows the search.

- **Inputs:** the committed search evidence `docs/audit_evidence/v2_search/` (path fixed in code;
  written after the search), the bundle checkpoint folder, S, the F3 profile,
  `configs/v2_candidates.yaml` (CPU batch size of the winner), the F2 manifest and, for `cnn-aug` /
  `crnn`, the Stage 8 val-B log-Mel cache. `selected` and `bundle_seed` must agree between
  `run_report.json` and `selection_report.json`; the checkpoint must hash to
  `checkpoint_sha256[<winner>_seed<k>]` and its metadata must name that architecture and seed.
- **Provenance:** `profile_fingerprint`, `f2_manifest_sha256` and `selection_set_sha256` of the
  inputs must equal the checkpoint's (recorded as `{expected, actual}`).
- **Exact M reproduction:** the five S EERs and M are recomputed on CPU with `train_v2`'s own
  `selection_tensors`, `evaluate_selection` and `selection_m`; M must equal the checkpoint's
  `selection_m` **exactly** (as Stage 10 requires the val-A EER). On mismatch the run stops and prints
  both values and the five EERs; any tolerance would be a team decision. M was recorded on the T4
  GPU, so this check is also the first test of GPU-to-CPU agreement for v2 [hypothesis: it holds, as
  it did for v1 at Stage 10].
- **val-B scoring:** log-Mel standardized with the checkpoint's `mean`/`std`, then the same raw-logit
  path as `train_v2`; two CPU passes must be equal. **aasist** needs an F3 waveform val-B source, a
  function returning `(waveforms float32 (n, 32000), labels int8, paths sorted)` that must cover
  exactly the F2 val-B rows. None exists: it is built only if aasist wins (Execution Plan v3.0 §9 V4).
  Without one, aasist is refused, so the CLI (`--cache-dir` only) cannot fit an aasist policy.
- **Outputs:** the Stage 10 set (`risk_policy.json` as bytes, `val_b_scores.npz`,
  `diagnostics_report.json`, `run_report.json`); `run_report.json` adds `architecture`,
  `bundle_seed`, `checkpoint_sha256`, `selection_m_recomputed`, `selection_eers_recomputed`,
  `val_b_input` and the hashes of the two search reports. The output folder must be new and is
  removed on any failure. No train, test or external path, no split argument, no W&B (D9, D31, D40).

```
python -m src.tuning.v2_risk_run --checkpoint <bundle ckpt> --selection-set <S> \
  --profile configs/preprocessing.yaml --config configs/v2_candidates.yaml \
  --manifest <F2 manifest> --cache-dir <Stage 8 log-Mel caches> --out <new folder> --version risk-v2
```

## v2 Stage 11 (V4b, `src/evaluation/stage11_v2_run.py`)

The separate `stage11_test` claim for the v2 model set (D48 (7)): the winner's three seeds,
`<winner>-v2-seed<k>`, `p` and prediction for the bundle seed only (D45), scored on CPU through
`v2_risk_run.model_inputs` + `score_clips`. It reads the committed search evidence and the risk-v2
outputs in `docs/audit_evidence/v2_risk/`; its rehearsal recomputes M per seed and the val-B bytes
of risk-v2. Details: [stage11_run.md](stage11_run.md#v2-model-set-d48-7). Code only so far; the
real run (notebook 09) follows risk-v2.

## F5 (V5b, `src/bundle/f5.py`)

ModelBundle v2 from the committed search, risk-v2 and v2 Stage 11 evidence and the run registry
(D48 (7): refused until the v2 Stage 11 run is completed), the bundle-seed checkpoint folder
`runs/<winner>_seed<k>/checkpoint/`, F3 profile and `risk-v2`; `build` / `verify` and
`f5_record.json`. Details: [model_bundle.md](model_bundle.md#f5-modelbundle-v2). Code only; the build,
the freeze decision and the hand-over follow the v2 Stage 11 run (cut-off Fri 4 Dec 23:59, D48 (4)).

## Tests (synthetic, CPU)

`tests/features/test_raw_train_cache.py`, `tests/training/test_selection_set.py`,
`tests/training/test_v2_data.py`, `tests/training/test_v2_wandb_logger.py`,
`tests/training/test_train_v2.py`, `tests/training/test_v2_resume.py` (per-epoch resume on a
4-clip subset, real CPU training), `tests/tuning/test_v2_search.py`,
`tests/tuning/test_v2_benchmark.py`, `tests/tuning/test_v2_risk_run.py` (V4a: round trips for all
three candidates and the refusals), `tests/training/test_v2_batch_size_evidence.py` (config vs
the V2b-2 evidence, torch-free); fakes in `tests/_v2_fakes.py` (24-clip raw cache, 10-clip S,
deterministic fake codec, tiny noise/RIR, config copies with batch size 8 (or null, for the
refusals) and fake pins).

## Next

- **V2b-2 (done, Entry 173):** benchmark run on the Colab T4 (2026.10); batch sizes 256 / 256 / 32
  written into `configs/v2_candidates.yaml` (D49).
- **V1b (Malak):** confirm the S layout and build S with `write_selection_set` on the training
  runtime; verify it with `load_selection_set` (including the Stage 8 val-A known answer).
- **V2b-4:** CLI and notebook wired to the V1b corpus loader; rehearsal on T4 (`check_clean_transform`
  against the Stage 8 train cache, a short run twice for the observed determinism), then the search.
