# CNN foundations (Stage 8, PR B1)

The deployed model is one CNN on the F3 log-Mel (AGENTS.md hard rule 3, D7). D42 fixes every model
and training choice before any CNN result exists; D30 fixes the search space (blocks {3, 4} ×
learning rate {1e-3, 3e-4}). This PR (**B1**) adds only the parts that do not train: the model, the
train-only normalization, the seeding helper and the non-pickle checkpoint, plus the torch
dependencies and CI. The training loop, W&B logging (D40) and `configs/cnn.yaml` are **PR B2**.
Everything here is tested on synthetic tensors only; no dataset is read.

## Dependencies and CI

| What | Where | Version |
|---|---|---|
| PyTorch | not in any requirements file (comment in `requirements-cnn.txt`) | `2.11.0`: Colab's preinstalled CUDA build (the notebook asserts it); CPU build from `https://download.pytorch.org/whl/cpu` for CI and local development |
| safetensors | `requirements-cnn.txt` | `0.8.0` |
| wandb | `requirements-cnn.txt` | `0.31.0` (used from B2) |

`2.11.0` is the version the Stage 6 benchmark ran with (`benchmark.json`, `2.11.0+cu130`). CI
installs `requirements.txt`, then CPU torch, then `requirements-cnn.txt`, and runs `pytest -q` with
`REQUIRE_TORCH_TESTS=1` (see [setup.md](setup.md#cnn-dependencies-stage-8)). D42 left dependency
pinning and CI installation to this PR, with Iheb's review.

## Model (`src/models/cnn.py`)

- `build_cnn(n_blocks)` with `n_blocks` in `N_BLOCKS = (3, 4)` (D30); any other value (including
  `True` or `3.0`) raises `ValueError`.
- Block (D42): `Conv2d(3×3, padding 1) → BatchNorm2d → ReLU → MaxPool2d(2)`.
  `CHANNELS = (16, 32, 64, 128)`: blocks 1–3 use 16, 32, 64; the 4th block uses 128.
- Head (D42): global average pooling → `Dropout(DROPOUT = 0.3)` → `Linear(C, 1)`, C = 64 or 128.
- `forward` takes `(n, 1, 64, 197)` (D39) and returns `(n,)` logits; any other input shape raises.
  The logit is the raw `score(audio)` (D26 C→D; higher = more likely spoof).

## Normalization (`src/training/normalization.py`, NumPy only)

- `fit_bin_stats(train_features)`: float32 `(n, 1, 64, 197)`, n ≥ 2, finite → per-Mel-bin mean and
  std (ddof 0) over all clips and frames, accumulated in float64 one bin at a time, returned as two
  float32 arrays of shape `(64,)`. A constant bin (std = 0) raises.
- `apply_bin_stats(features, mean, std)`: `(x − mean[b]) / std[b]` per bin, for any split; it reads
  the statistics and never refits or modifies them.
- The statistics come **from the FoR train log-Mel cache only** (D42), are stored in the checkpoint,
  and are applied unchanged to val-A, val-B, test and the external windows. No per-clip
  normalization (D42).

## Determinism (`src/training/determinism.py`)

`set_seed(seed)` (torch imported inside the function) seeds `random`, NumPy, torch and every CUDA
device, sets `cudnn.deterministic = True`, `cudnn.benchmark = False`,
`torch.use_deterministic_algorithms(True, warn_only=True)`, and sets
`CUBLAS_WORKSPACE_CONFIG=":4096:8"` if it is unset (an existing value is kept). It returns a dict
of what was set, which B2 stores in the checkpoint's `determinism` field (D42).

## Checkpoint format (`src/models/cnn_checkpoint.py`)

One folder holding exactly two files:

| File | Content |
|---|---|
| `weights.safetensors` | the model `state_dict` plus `norm.mean` and `norm.std` (float32, 64), via safetensors |
| `checkpoint.json` | the metadata below; sorted keys, indent 2, `allow_nan=False` |

Metadata keys (exactly; B2 fills them, B1 validates them): `schema_version` (1), `version`
(non-empty), `n_blocks` (3 or 4), `channels` (`CHANNELS[:n_blocks]`), `dropout` (0.3),
`input_shape` (`[1, 64, 197]`), `learning_rate` (finite > 0), `seed` (int in [0, 2³²)),
`best_epoch` (int in [1, 30], D30 maximum), `val_a_eer` (in [0, 1]), `code_commit` (40 hex),
`environment` (non-empty object of strings), `profile_fingerprint`, `f2_manifest_sha256`,
`train_cache_npz_sha256`, `val_a_cache_npz_sha256` (64 hex each), `determinism` (object whose
`seed` equals `seed`).

- `save_checkpoint(model, mean, std, metadata, out_dir)` validates everything first (metadata,
  `n_blocks` equal to the model's, statistics), refuses an existing non-empty folder, writes both
  files, and on failure removes what it wrote (and the folder if it created it).
- `checkpoint_sha256(out_dir)` = SHA-256 of
  `checkpoint.json:<sha256>\nweights.safetensors:<sha256>\n`, as for the baseline artifact (D36).
- `load_checkpoint(out_dir, *, expected_sha256=None)` checks the folder holds exactly the two
  regular files, reads each once, optionally checks `expected_sha256` against those bytes,
  validates the JSON, loads the tensors with safetensors (never `torch.load`, never pickle), checks
  every tensor's name, dtype, shape and finiteness against `build_cnn(n_blocks)`, loads with
  `strict=True`, and returns `(model in eval mode, mean, std, metadata)`.
- Saving the same model twice gives byte-identical files and the same `checkpoint_sha256`.

## Why (team reasoning)

- **(a) Exactly the D42 network.** 16/32/64(/128) is the network the Stage 6 benchmark measured at
  batch 256, so D30's batch size stays valid (D42). D42 adds BatchNorm, which the benchmark network
  did not have; **[hypothesis]** BatchNorm's extra activations do not push batch 256 past the 80%
  VRAM limit (the benchmark peak was 0.90 GB of 15.6 GB); B2 records the real peak.
- **(b) Train-only statistics, stored with the weights.** Fitting on train only keeps val-A, val-B,
  test and the external set out of the fit (D9, D42); storing them in the checkpoint puts the same
  scaling under the same hash for every split and at Stage 13.
- **(c) No pickle.** A pickled checkpoint can execute code when loaded (model_bundle.md,
  "Unsafe deserialization"). safetensors holds only a JSON header and raw tensor bytes.
- **(d) Verify before use, single read.** Exact file set, strict schema, strict state_dict, and one
  read per file, so the hashed bytes are the loaded bytes (model_bundle.md (a), (f)). A changed
  weight value leaves a valid file; it is detected against the recorded `checkpoint_sha256`
  (`expected_sha256`), the external anchor F4 stores (model_bundle.md (e)).
- **(e) Seeds recorded, not promised.** D42 seeds everything and asks for deterministic
  algorithms, but does not promise byte-identical GPU runs; the dict from `set_seed` records what
  was set, so reproducibility can be reported as observed.
- **(f) Torch tests fail in CI, never skip.** A skipped CNN test would look like a pass; the guard
  `tests/_torch.py` fails when `REQUIRE_TORCH_TESTS=1` and torch is missing.

## Limitations

- `learning_rate` is checked as a positive number, not against the D30 grid; the grid check
  belongs to `configs/cnn.yaml` and its loader (B2).
- `environment` and `determinism` are checked for type only; their exact content is set in B2.
- Byte-identical checkpoints are shown for the same weights on CPU; whether two GPU training runs
  give the same weights is reported as observed in B2 (D42).
- The checkpoint is not yet in a ModelBundle; that is Stage 12 (model_bundle.md).

## Tests

| Property | Test |
|---|---|
| Guard fails, not skips (f) | `tests/test_torch_guard.py`: `test_guard_fails_not_skips_when_required_and_torch_missing` (torch import made to fail), `test_guard_skips_when_not_required_and_torch_missing`, `test_guard_returns_torch_when_importable` |
| D42 constants and layers (a) | `tests/models/test_cnn.py`: `test_constants_match_d42`, `test_architecture_matches_d42` (block order, kernel, padding, channels, pooling, dropout, head) |
| Output shape | `test_output_is_one_logit_per_clip` (3 and 4 blocks → `(n,)`), `test_wrong_input_shape_raises` |
| Only 3 or 4 blocks | `test_only_3_or_4_blocks` (2, 5, 0, `True`, `"3"`, `3.0`, `None`) |
| Seeded init (e) | `test_same_seed_same_initial_weights` (same seed identical; another seed changes every conv and linear weight) |
| set_seed (e) | `tests/training/test_determinism.py`: flags and returned dict, existing `CUBLAS_WORKSPACE_CONFIG` kept, same/different seed draws, invalid seeds |
| Train-only statistics (b) | `tests/training/test_normalization.py`: `test_fit_bin_stats_are_per_bin_over_clips_and_frames`, `test_apply_never_refits_and_leaves_stats_unchanged` (spy on `fit_bin_stats`; stats unchanged; val keeps its offset) |
| Normalization refusals | wrong shape or dtype, n < 2, NaN / ±inf, zero-std bin, bad stats |
| Round trip | `tests/models/test_cnn_checkpoint.py`: `test_round_trip_gives_identical_logits_and_stats` (3 and 4 blocks, BatchNorm buffers included) |
| Determinism of bytes | `test_double_save_is_byte_identical_with_same_hash`, `test_json_is_sorted_strict_and_has_exact_keys` |
| Tampering (d) | `test_tampered_weight_value_changes_hash_and_is_refused_against_record`, `test_tampered_json_value_changes_hash_and_is_refused_against_record`, `test_corrupt_weights_file_is_refused`, `test_invalid_tensors_are_refused`, `test_invalid_json_fields_are_refused`, `test_missing_or_unknown_json_key_is_refused`, `test_unparseable_json_is_refused`, `test_nan_inside_metadata_is_refused` |
| Architecture mismatch | `test_four_block_weights_cannot_load_as_three_blocks` |
| Folder contents | `test_extra_file_is_refused`, `test_missing_file_is_refused`, `test_missing_folder_is_refused` |
| Saving | `test_save_refuses_non_empty_folder_and_leaves_it_unchanged`, `test_save_into_empty_folder_is_allowed`, `test_save_validates_before_writing`, `test_save_refuses_bad_stats_or_model`, `test_failed_save_removes_partial_output` |
| No pickle (c) | `test_load_never_uses_torch_load_or_pickle` (`torch.load` and `pickle` patched to raise), `test_checkpoint_module_imports_no_pickle_or_research_code` |
