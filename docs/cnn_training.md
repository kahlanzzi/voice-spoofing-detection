# CNN training and model selection (Stage 8, PR B2)

This PR (**B2**) adds the parts that train: the CNN config with its drift guard, the training loop
with D42 early stopping, the allow-list W&B logger (D40) and the D30 search orchestrator. It builds
on B1 ([cnn.md](cnn.md): model, train-only normalization, seeding, checkpoint). Everything is tested
on synthetic tensors and tiny synthetic caches, on CPU, with W&B never contacted. No dataset was
read and no real training run was made.

## Config and drift guard (`configs/cnn.yaml`, `src/training/cnn_config.py`)

| Field | Value | Source |
|---|---|---|
| `grid.n_blocks` × `grid.learning_rate` | `[3, 4]` × `[0.001, 0.0003]`, n_blocks outer | D30 |
| `optimizer`, `weight_decay` | `adam`, `0.0` (constant rate, float32) | D30, D42 |
| `loss`, `class_weights` | `bce_with_logits`, `none` | D42 |
| `dropout` | `0.3` | D42 |
| `max_epochs`, `patience` | `30`, `5` | D30 |
| `batch_size` | `256` (Stage 6 benchmark) | D30, D39 |
| `early_stopping` | `val_a_eer`, `strictly_lower`, `earliest_epoch` | D42 |
| `search_seed`, `final_seeds` | `0`, `[0, 1, 2]` | D30, D11 |
| `tie_rule` | `[n_blocks_ascending, learning_rate_descending]` | D30 |
| `wandb` | entity `voice-spoofing-detection-must`, project `voice-spoofing-detection` | D40 |

`load_cnn_config` reads the file with `yaml.safe_load` and refuses a missing or unknown key (top
level and nested) and **any value other than the one above**, compared with types at every level, so
`true` for `1`, `256.0` for `256` or a reordered grid is refused. Only these values are implemented;
changing one needs a new decision and a code change (as `configs/baseline.yaml` for D37). It returns
a frozen `CNNConfig` with `grid()` (the 4 pairs in grid order) and `to_dict()` (the YAML layout).

## Training loop (`src/training/train_cnn.py`)

`train_one(train_x, train_y, val_a_x, val_a_y, *, n_blocks, learning_rate, seed, max_epochs,
patience, batch_size, logger, device="cpu") → TrainResult`. There is no parameter for val-B, test or
external data (D9, D42).

1. `set_seed(seed)` first (D42 "Determinism"), then the hyperparameter checks.
2. `mean, std = fit_bin_stats(train_x)` on train only; train and val-A standardized with
   `apply_bin_stats` (D42 "Input").
3. `build_cnn(n_blocks)`, `Adam(lr, weight_decay=0)`, `BCEWithLogitsLoss()` (D30, D42).
4. Each epoch: one pass over all of train in an order drawn from a `torch.Generator` seeded with
   `seed` (the last batch may be smaller); then `model.eval()` and `no_grad` scoring of val-A in
   batches; `eer` and `roc_auc` of the logits from `src.evaluation.metrics` (D42 "Epoch").
5. Early stopping (D30, D42), below. After each epoch `logger.log_epoch(...)`; at the end
   `logger.log_summary(...)`.

`TrainResult` (frozen): `state_dict` (best epoch, CPU copies), `mean`, `std`, `best_epoch`,
`best_val_a_eer`, `epochs_run`, `history` (per epoch: `epoch`, `train_loss`, `val_a_eer`,
`val_a_auc`, `epoch_seconds`), `determinism` (from `set_seed`), `peak_gpu_memory_bytes`
(`torch.cuda.max_memory_allocated` after a reset before training; `None` on CPU).

### Early stopping rules (D30, D42)

- The best starts at +inf. An epoch improves only if its val-A EER is **strictly lower** than the
  best; it then becomes the best epoch, a copy of its `state_dict` is kept and the counter resets.
- Otherwise (higher **or equal**) the counter of non-improving epochs increases. Training stops when
  the counter reaches `patience`, or after `max_epochs`.
- A tie never replaces the best, so the **earliest** epoch wins. The returned `state_dict` is the
  best epoch's, not the last epoch's.

Example with `patience = 3` (the test script):

| Epoch | val-A EER | Best so far | Non-improving count | Action |
|---|---|---|---|---|
| 1 | 0.30 | 0.30 (epoch 1) | 0 | keep epoch 1 |
| 2 | 0.20 | 0.20 (epoch 2) | 0 | keep epoch 2 |
| 3 | 0.20 | 0.20 (epoch 2) | 1 | tie: earlier epoch stays |
| 4 | 0.25 | 0.20 (epoch 2) | 2 | |
| 5 | 0.10 | 0.10 (epoch 5) | 0 | keep epoch 5 |
| 6 | 0.10 | 0.10 (epoch 5) | 1 | tie |
| 7 | 0.10 | 0.10 (epoch 5) | 2 | tie |
| 8 | 0.10 | 0.10 (epoch 5) | 3 | stop; return epoch 5's weights |

## W&B allow-list (`src/training/wandb_logger.py`)

`RunLogger(mode, config_dict, run_name)` with `mode` in `online`, `offline`, `disabled`; `wandb` is
imported only for `online` / `offline`. D40 allows FoR training and validation metrics and configs
only; D42 lists the fields.

| Call | Exact keys |
|---|---|
| config (at init) | the 15 `cnn.yaml` fields + `seed`, `n_blocks`, `learning_rate`, `code_commit`, `profile_fingerprint`, `f2_manifest_sha256`, `train_cache_npz_sha256`, `val_a_cache_npz_sha256` |
| `log_epoch` | `epoch`, `train_loss`, `val_a_eer`, `val_a_auc`, `epoch_seconds` |
| `log_summary` | `best_epoch`, `best_val_a_eer`, `epochs_run`, `peak_gpu_memory_bytes` (`None` allowed on CPU) |

Refused with `ValueError` **before anything is sent**: any other or missing key; a metric value that
is not a finite `int`, `float` or `bool` (arrays, tensors, lists, strings, images, tables, media,
NaN, ±inf); a key at any level containing `val_b`, `val-b`, `test` or `external` (checked before the
allow-list, so a future allow-list edit cannot let one through). The run starts with
`save_code=False`, `disable_code=True`, `console="off"`, `x_save_requirements=False`,
`disable_job_creation=True`; the module never calls `wandb.save`, `wandb.Artifact` or
`log_artifact`. `finish()` closes the run (a second call does nothing; later logging is refused).
The API key comes from the environment (`WANDB_API_KEY`, from `.env`), never from code or config.

## Search and selection (`src/tuning/cnn_search.py`)

`run_search(cache_dir, profile_path, config_path, manifest_path, out_dir, *, version, logger_mode,
device)` and a CLI (`python -m src.tuning.cnn_search --cache-dir … --profile … --config … --manifest
… --out … --version cnn-v1 --wandb-mode online --device cuda`; mode and device are required).

1. `code_version()` first: a dirty worktree is refused before any input is read. An existing
   `out_dir` is refused.
2. Load the config, the F3 profile and the F2 manifest; load **only** the train and val-A log-Mel
   caches with `load_logmel_cache` (approved F2 hash, profile fingerprint and profile file hash
   checked; every cached path must be an F2 row of that split with the same label). val-B is not
   loaded or scored in Stages 8–9 (D42); the module has no val-B, test or external path.
3. Search (D30): `train_one` for the 4 configurations with `search_seed = 0`, each with its own
   `RunLogger` named `<version>-b<n_blocks>-lr<lr>-s<seed>` (e.g. `cnn-v1-b3-lr0.001-s0`).
4. Selection (D30, D9): lowest `best_val_a_eer`; ties by fewer blocks, then the higher learning rate
   (sort key `(eer, n_blocks, −lr)`).
5. Final seeds (D30, D11): the selected configuration is trained with seeds 1 and 2; the seed-0
   search run is reused as seed 0. `bundle_seed = select_bundle_seed({0: …, 1: …, 2: …})` (median
   val-A EER, tie → lower seed; Protocol §5).
6. Outputs, all or nothing: training happens before `out_dir` is created, and any failure while
   writing removes it.

| Output | Content |
|---|---|
| `checkpoints/seed_<s>/` | the selected configuration's checkpoint for seeds 0, 1, 2 (`save_checkpoint`, full metadata incl. environment, determinism, F2, profile and cache hashes) |
| `search_report.json` | `search_seed`; 4 rows: `n_blocks`, `learning_rate`, `best_epoch`, `best_val_a_eer`, `epochs_run`, `history` |
| `selection_report.json` | `selected`, `selected_val_a_eer`, `tie_rule`, `tie_rule_decided`, `tied_rows`, `seeds` (per seed: the row fields + `reused_search_run`), `bundle_seed` |
| `run_report.json` | `schema_version`, `version`, `code_commit`, `environment` (python, numpy, torch, safetensors, wandb, `cuda_available`, `gpu_name`, `device`), `profile_fingerprint`, `profile_config_sha256`, `f2_manifest_sha256`, `cnn_config_sha256`, `caches` (npz hash, rows), the two cache hashes, `checkpoint_sha256` per seed, `peak_gpu_memory_bytes` per run (6 runs), `wandb_mode`, `wandb_runs`, `selected`, `bundle_seed` |

All JSON is written with sorted keys and `allow_nan=False`.

## Why (team reasoning)

- **(a) One implemented config.** D30 and D42 fix every choice before any CNN result exists; a
  loader that accepts only those values makes a silent change impossible (AGENTS.md hard rule 6).
- **(b) Strict improvement, earliest tie, best weights.** D42's rule; with a near-zero val-A EER
  (baseline, docs/results/baseline.md) ties are plausible, and keeping the earliest epoch is the
  pre-declared, less-trained choice.
- **(c) val-A only, by construction.** No parameter, path or load call for val-B, test or external
  data, so Stage 10 (val-B), Stage 11 (test, D31) and Stage 13 (DEEP-VOICE) stay untouched (D9, D42).
- **(d) Allow-list, not deny-list, for W&B.** D40 limits W&B to FoR training and validation metrics;
  an exact key list plus a scalar-only rule keeps per-clip data, images and audio out, and the
  forbidden substrings are a second line of defence.
- **(e) Pre-declared selection.** Grid order, tie rule and bundle-seed rule are code, written before
  results (D30, D11, Protocol §12).

## Limitations

- **[hypothesis]** Batch 256 with BatchNorm stays within the T4 memory limit; the real peak is
  recorded per run in `run_report.json` (cnn.md (a)).
- Same seed gives identical results on CPU (tested); on GPU reproducibility is reported as observed,
  not promised (D42).
- W&B still records its own run metadata (system information) at init; only code, console output
  and requirements files are turned off. The allow-list governs what this code sends.
- Config values sent to W&B may be lists and mappings of scalars (the `cnn.yaml` layout); metric
  values must be scalars.
- The end-to-end test uses tiny epochs through a test-only config patch; the real 30-epoch,
  12,404-clip run happens on Colab.

## Tests

| Property | Test |
|---|---|
| Config equals D30 + D42 (a) | `tests/training/test_cnn_config.py`: `test_loaded_config_equals_d30_d42`, `test_config_matches_model_and_checkpoint_constants` |
| Every changed field refused (a) | `test_every_changed_field_is_refused` (every field, plus same value with another type), `test_missing_or_unknown_keys_are_refused`, `test_unreadable_or_non_mapping_config_is_refused` |
| Early stopping (b) | `tests/training/test_train_cnn.py`: `test_early_stopping_is_strict_keeps_earliest_tie_and_stops_after_patience` (scripted EER; returned weights equal the best epoch's snapshot, not the last), `test_tie_with_best_never_replaces_it`, `test_stops_after_exactly_patience_non_improving_epochs`, `test_max_epochs_is_respected` |
| Train-only statistics | `test_fit_bin_stats_is_called_once_on_train_only` (spy) |
| No val-B / test input (c) | `test_signature_has_no_val_b_test_or_external_input`; `tests/tuning/test_cnn_search.py`: `test_module_has_no_val_b_test_or_external_path`, `test_end_to_end_outputs_checkpoints_and_reports` (spy: only train and val-A loaded, though a val-B cache exists) |
| CPU determinism | `test_same_seed_gives_identical_results_on_cpu` |
| Result and refusals | `test_result_fields_and_logging_with_a_real_disabled_logger`, `test_invalid_hyperparameters_raise`, `test_invalid_labels_raise`, `test_cuda_requested_without_cuda_raises` |
| W&B allow-list (d) | `tests/training/test_wandb_logger.py`: `test_full_run_sends_only_allowed_calls` (fake wandb records every call; no save / Artifact / log_artifact), `test_disabled_mode_never_imports_wandb`, `test_bad_epoch_values_are_refused_before_sending`, `test_tensor_is_refused`, `test_media_objects_are_refused`, `test_forbidden_or_unknown_keys_are_refused`, `test_forbidden_key_parts_win_even_over_an_allow_list_entry`, `test_forbidden_nested_config_key_is_refused`, `test_logger_module_never_references_uploads` |
| Selection (e) | `test_lowest_eer_wins_and_only_selected_config_gets_seeds_1_and_2`, `test_full_tie_selects_3_blocks_and_lr_0_001`, `test_tie_between_learning_rates_within_4_blocks_selects_0_001`, `test_select_configuration_tie_rule_order`, `test_bundle_seed_follows_select_bundle_seed` |
| Outputs | `test_end_to_end_outputs_checkpoints_and_reports` (exact files and keys; checkpoints load with `expected_sha256`) |
| All or nothing | `test_failure_leaves_nothing`, `test_training_failure_writes_nothing`, `test_existing_out_dir_is_refused`, `test_dirty_worktree_is_refused_before_reading`, `test_invalid_arguments_are_refused` |
| CLI | `test_cli_parses_required_arguments` |
