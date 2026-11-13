# Model v2 candidates (Track V, PR V2a)

The three model v2 candidates of D49, their checkpoint format and their training configuration,
with the implementation details D61 fixes before any v2 training. Code and synthetic tests only:
no dataset, cache, W&B, training run, FoR test or DEEP-VOICE access.

Decisions: D49 (candidates, selection, F5), D50 / D56 (augmentation), D59 (bundle layout),
**D61 (approved, PR #127)** (implementation details). ADR: [0013](decisions/0013-model-v2-candidates-and-augmentation.md).

## The three candidates

Every candidate: FoR train only with the D50 augmentation, the unchanged F3 profile (D39), one
output logit, `BCEWithLogitsLoss` without class weights, float32, at most 30 epochs (D49).

| Candidate | Module | Input | Trainable parameters |
|---|---|---|---|
| C1 `cnn-aug` | `src/models/cnn.py`, `build_cnn(4)` (unchanged v1 code) | log-Mel (n, 1, 64, 197) | 97,761 |
| C2 `crnn` | `src/models/crnn.py`, `build_crnn()` | log-Mel (n, 1, 64, 197) | 319,585 |
| C3 `aasist` | `src/models/aasist.py`, `build_aasist()` | waveform (n, 32000) | 297,705 |

The parameter counts are pinned in tests (`tests/models/test_v2_checkpoint.py`). They decide
D49's tie rule (equal M → fewer trainable parameters first, then C1, C2, C3).

### C1 `cnn-aug`
Exactly v1 (D42; D30's selected configuration): 4 blocks Conv2d 3×3 (padding 1) → BatchNorm2d →
ReLU → MaxPool2d 2, channels 16, 32, 64, 128 → global average pooling → Dropout(0.3) →
`Linear(128, 1)`. Adam, learning rate 1e-3, no weight decay, constant rate. The only difference from
v1 is the training data (D50 augmentation): C1 isolates the effect of augmentation.

### C2 `crnn`
- Three D42 blocks, channels 16, 32, 64: (n, 1, 64, 197) → (n, 64, 8, 24) (floor pooling).
- Per time step, channel × frequency flattened, channel-major: (n, 24, 512).
- One bidirectional `nn.LSTM(512, 64, batch_first=True)`: (n, 24, 128).
- Mean over time (n, 128) → Dropout(0.3) → `Linear(128, 1)` → (n,) logits.
- Adam, learning rate 1e-3, no weight decay, constant rate.

The D42 block is restated in `crnn.py` (4 lines) so `cnn.py`, part of F4, stays unchanged.
`CRNN.sequence(x)` returns the LSTM input; it refuses any other input shape.

### C3 `aasist`
AASIST (Jung et al., 2022) ported from the reference implementation:

- **Source:** <https://github.com/clovaai/aasist>, commit `a04c9863f63d44471dde8a6abcb3b082b07cd1d1`,
  file `models/AASIST.py`; MIT licence, Copyright (c) 2021-present NAVER Corp. The module docstring
  credits it; the reference `LICENSE` and `NOTICE` are copied verbatim into
  [`licenses/AASIST-LICENSE`](../licenses/AASIST-LICENSE). The reference repository is not vendored.
- **Layers:** sinc-filter front-end (70 filters, kernel 129) → |·|, max-pool 3×3, BatchNorm, SELU →
  six residual blocks (filters 32, 32, 64, 64, 64, 64; each max-pools time by 3) → spectral GAT
  (23 nodes) and temporal GAT (14 nodes at 32,000 samples) with graph pooling → two branches of
  heterogeneous stacking GAT with a master node → max of branches → [T max, T mean, S max, S mean,
  master] = 160 values → Dropout(0.5) → `Linear(160, 1)`.
- **Config:** the reference `model_config` (`first_conv` 128, `filts`
  [70, [1, 32], [32, 32], [32, 64], [64, 64]], `gat_dims` [64, 32], `pool_ratios`
  [0.5, 0.7, 0.5, 0.5], `temperatures` [2.0, 2.0, 100.0, 100.0]) with `nb_samp` 32,000.
- **Optimizer and schedule:** the reference Adam (learning rate 1e-4, betas 0.9/0.999, weight decay
  1e-4, no AMSGrad) and per-step cosine annealing to 5e-6 over 30 epochs × steps per epoch
  (`src/training/v2_schedule.py`, `cosine_lr`).

**Diff against the reference** (each marked `# D61` in the code; names, structure and order of
operations otherwise unchanged, so a reference state_dict without `out_layer` loads into the port):

| Reference | Port | Why |
|---|---|---|
| `out_layer = Linear(160, 2)`, `CrossEntropyLoss(weight=[0.1, 0.9])` | `Linear(160, 1)`, `BCEWithLogitsLoss`, no weights | D61 (2): one loss for every candidate, so M compares architectures, not losses |
| `forward(x, Freq_aug=False)` returns `(last_hidden, output)` | `forward(x)` returns (n,) logits | D61 (2), (4) |
| `CONV(..., mask=False)`, random band zeroing in `CONV.forward(x, mask)`, `import random` | removed | D61 (4): D50 is the only augmentation |
| any input length (64,600 in the reference config) | (n, 32000) float32 only; other shapes or dtypes raise | D49: 2 s at 16 kHz |
| 100 epochs, SWA in the training script | 30 epochs, no SWA (not ported) | D61 (3), (4) |

Kept from the reference on purpose (not D61 changes): `Residual_block.forward` computes `bn1`/SELU
for non-first blocks but then applies `conv1` to `x`, so `bn1` does not affect the output (its
parameters still count as trainable); the sinc band-pass filters are a fixed tensor rebuilt from the
config (not in the state_dict).

**Length:** the spectral axis (23 nodes, matching `pos_S`) does not depend on the input length, and
the GAT layers take any node count, so 32,000 samples need no architectural change.

## D61 points and reasons

1. **AASIST source** pinned by commit, ported with credit and licence file; reference config
   unchanged except `nb_samp`. Reason: D49 asks for a reference implementation pinned by commit.
2. **One logit, BCE, no class weights** for AASIST too. Reason: D49 applies to every candidate; a
   different loss would confound the comparison.
3. **Reference Adam and cosine per step over 30 epochs.** Reason: over 100 epochs the rate would
   barely decay within D49's 30; the schedule does not change if early stopping ends the run sooner.
4. **No SWA, no `Freq_aug`.** Reason: SWA's averaged weights were never evaluated on M (D49 restores
   the best epoch); D50 is the only augmentation.
5. **Log-Mel standardization for C1 and C2** fitted on the clean (un-augmented) F3 log-Mel of FoR
   train, stored in the checkpoint. Reason: augmented features change every epoch, so statistics
   fitted on them would not be reproducible.
6. **v2 checkpoint** in a new module with the D59 layout (below); `cnn_checkpoint` (F4's loader)
   unchanged.
7. **Batch size** per candidate from the D49 compute benchmark, written before training.

## v2 checkpoint (`src/models/v2_checkpoint.py`)

One folder with exactly `checkpoint.json` + `weights.safetensors`, hashed with the D59 combined rule
(SHA-256 of `"checkpoint.json:<sha256>\nweights.safetensors:<sha256>\n"`). The file reading and
hashing are `cnn_checkpoint`'s own functions (imported, not copied), so `v2_checkpoint_sha256`
equals `cnn_checkpoint.checkpoint_sha256` and `src.bundle.bundle.checkpoint_dir_sha256` on the same
folder (tested): F5 fits the D59 bundle code unchanged. F4's loader refuses a v2 checkpoint (different
keys).

- `ARCHITECTURES = ("cnn-aug", "crnn", "aasist")`, `build_model(architecture)`.
- **Metadata** (exact keys, sorted, no NaN): `schema_version` 1, `architecture`, `version`,
  `input` (`logmel` for cnn-aug and crnn, `waveform` for aasist), `input_shape` ([1, 64, 197] or
  [32000]), `seed`, `best_epoch` (1..30), `selection_m` (finite, [0, 1]), `code_commit` (40 hex),
  `environment`, `profile_fingerprint`, `f2_manifest_sha256`, `augmentation_config_sha256`,
  `selection_set_sha256` (64 hex), `train_stats_npz_sha256` (64 hex for log-Mel models, `null` for
  aasist), `determinism` (`seed` equal to `seed`), `trainable_parameters` (equal to the built model's).
- **Tensors:** the state_dict, plus `norm.mean` / `norm.std` (float32 (64,), std > 0) if and only if
  the input is the log-Mel.
- `save_v2_checkpoint(model, metadata, out_dir, *, mean=None, std=None)`: new or empty folder only,
  every check before writing, cleanup on failure.
- `load_v2_checkpoint(out_dir, *, expected_sha256=None)` → `(model in eval mode, mean | None,
  std | None, metadata)`: safetensors only (never `torch.load`), dtype / shape / finite checks,
  `load_state_dict(strict=True)`.
- `trainable_parameters(model)`: the D49 tie-rule count.

The F5 model loader will dispatch on `architecture` when F5 is built (D61 (6)); not part of this PR.

## Config (`configs/v2_candidates.yaml`, `src/training/v2_config.py`)

Strict loader in the `cnn_config.py` style: exact keys at every level, exact values and types
(drift guard). Common settings at the top level (decision D49/D50/D61, 30 epochs, patience 5, BCE,
no class weights, float32, early stopping on `selection_m` strictly lower / earliest epoch,
`selection_metric: mean_val_a_eer_over_S`, search seed 0, final seeds 0/1/2, tie rule, candidate
order, `logmel_stats: clean_train`, W&B target); one block per candidate under `candidates:`.

`batch_size` is `null` for every candidate until the D49 compute benchmark (PR V2b) writes a
positive int. The loader accepts `null`; `require_batch_sizes()` raises while any is `null`, so
training code cannot start without them. Floats are written with a decimal point (`0.0001`), as
PyYAML reads `1e-4` as a string.

## Tests

Torch tests use `tests._torch.require_torch` (fail in CI without torch); the config tests are
torch-free except one model-constants check.

- `tests/models/test_crnn.py` (11): shapes (intermediate (n, 64, 8, 24), LSTM input 512, output
  (n,)) for batches of 1 and 3 in eval mode; per-frame flattening; layers; parameter count; wrong
  shapes refused; determinism.
- `tests/models/test_aasist.py` (14): config, (n,) logits, `Linear(160, 1)`, wrong shape/dtype
  refused, no `Freq_aug` / mask path, parameter count, determinism, and the **faithfulness test**:
  with `AASIST_REF_DIR` pointing to the reference clone at the pinned commit (skipped with the reason
  otherwise), reference weights (after two train-mode passes, so BatchNorm statistics are not the
  defaults) are copied into the port and the 160-d embedding before `out_layer` must match the
  reference's `last_hidden` (atol 1e-6, rtol 1e-5). Result: max |port − reference| = 0.000e+00
  (Entry 170).
- `tests/models/test_v2_checkpoint.py` (62): round trip per architecture; hash equals the D59
  bundle rule; byte-identical double save; metadata refusals (keys, architecture/input/shape
  mismatch, stats hash, parameter count, NaN); norm tensors on aasist / missing on log-Mel; tensor
  refusals; tampered weights against the recorded hash; extra/missing file; non-empty folder; save
  checks before writing; cleanup on failure; never `torch.load`; the three new model files import
  no pickle and no `src.evaluation` / `src.tuning`.
- `tests/training/test_v2_config.py` (67): committed file equals D49 + D61; matches the model
  constants; `require_batch_sizes()`; every changed value, type, missing or unknown key refused;
  `cosine_lr` endpoints, midpoint, quarter point, monotonic decrease, invalid arguments.

## Not yet

Training loop, selection metric M over the condition set S, and the batch-size benchmark: PR V2b,
after V1b (augmentation pipeline) and set S. No v2 model has been trained.
