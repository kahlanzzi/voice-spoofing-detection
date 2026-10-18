# MFCC/SVM baseline

The baseline is an RBF-SVM on the 40-value MFCC vector (mean + std of 20 coefficients, Protocol
v1.0 §3, §11). It is **evaluation-only** (D7): it is compared with the CNN on every split and at the
external run, but it is never deployed and has no risk policy. It gets its own forced-binary
threshold at the val-B EER point of its score (D29); its tuning, train-only scaling and score are
fixed by D37; its artifact is recorded in the F4 freeze (D36).

The work has two steps:
- **Step B1:** train-only scaler, NumPy scorer and a non-pickle artifact,
  `src/models/baseline.py`. Tested on synthetic data only; no dataset is read.
- **Step B2:** fitting with the D37 grid and the D29 threshold, `src/tuning/baseline_fit.py` and
  `configs/baseline.yaml` (see [Step B2: fitting](#step-b2-fitting)). Synthetic data only.

## Format

`src/models/baseline.py` holds:
- `N_FEATURES = 40`, `SCHEMA_VERSION = 1`.
- `fit_scaler(train_features) -> (mean, std)`: per-feature mean and std (ddof=0) of the train
  features only.
- `BaselineModel(version, mean, std, support_vectors, dual_coef, intercept, gamma, C, threshold)`,
  a frozen dataclass; arrays are stored as read-only float64 copies. `C` is not used to score and is
  kept for provenance.
- `from_sklearn(svc, *, mean, std, threshold, version)`, `save(model, directory)`,
  `load(directory)`, `artifact_sha256(directory)`.

The artifact is one folder holding exactly two files:

| File | Content |
|---|---|
| `baseline.npz` | Arrays `mean` (40), `std` (40), `support_vectors` (k × 40), `dual_coef` (k); float only |
| `baseline.json` | `schema_version`, `version`, `gamma`, `C`, `intercept`, `threshold`, `n_features` (strict keys) |

`artifact_sha256` is the SHA-256 of the ASCII text
`baseline.json:<sha256 of baseline.json>\nbaseline.npz:<sha256 of baseline.npz>\n`, the single value
the F4 record stores (D36). The artifact is byte-deterministic: `baseline.npz` is an uncompressed
zip with one `<name>.npy` entry per array (sorted by name, written by `np.lib.format.write_array`
with `allow_pickle=False`), each with a fixed date (1980-01-01) and fixed system and permission
fields, and the JSON has sorted keys. Saving the same model again gives identical bytes and the same
hash.

## Method

1. `fit_scaler` takes the train features (n × 40, finite) and returns their mean and std; a
   constant feature (std == 0) raises.
2. `score(features)` standardizes with the stored mean and std, then computes
   `f(x) = Σ_i dual_coef_i · exp(−gamma · ‖sv_i − x‖²) + intercept` in NumPy (vectorized; the squared
   distance is expanded and clipped at 0). Higher score = more likely spoof. Rows are scored in
   batches of `SCORE_BATCH = 2048`, so the kernel matrix is at most 2048 × n_support_vectors; the
   result equals scoring all rows at once.
3. `predict(features)` = `three_way_decisions(score, threshold, half_width=0)` from
   `src.risk.decisions`: `synthetic` if score ≥ threshold, else `authentic`; never `uncertain`.
4. `from_sklearn` accepts only a fitted `SVC(kernel="rbf")` with `classes_ == [0, 1]` and copies
   `support_vectors_`, `dual_coef_[0]`, `intercept_[0]`, the public `gamma` and `C`. D37 fixes gamma
   as explicit numbers, so `gamma` must be numeric: `"scale"` and `"auto"` are refused.
5. `save` refuses a non-empty folder, writes the npz (fixed metadata, see Format) then the JSON (no
   NaN allowed), and removes what
   it wrote if a write fails. `load` reads the JSON (strict keys, schema 1, 40 features) and the npz
   with `allow_pickle=False` (strict keys, float arrays), and `BaselineModel` checks shapes and
   finiteness; any problem raises `ValueError`.

## Why (team reasoning)

- **(a) Non-pickle artifact.** A pickled or joblib scikit-learn model can execute code when it is
  loaded, the same risk [model_bundle.md](model_bundle.md) names for PyTorch checkpoints. Numbers in
  an npz read with `allow_pickle=False` and a strict JSON file cannot run code, and they are readable
  without the scikit-learn version that wrote them.
- **(b) Own NumPy scorer, verified against scikit-learn.** Stage 13 then scores without
  scikit-learn, from the stored numbers alone. The formula is the RBF decision function itself, and
  the tests check it equals `SVC.decision_function` within 1e-9 (observed max difference 4.7e-15).
- **(c) Train-only scaler stored in the artifact.** RBF-SVMs depend strongly on feature scale (D37).
  Fitting the scaler on train only keeps val-A, val-B, test and the external windows out of the fit;
  storing it in the artifact means the same scaling is applied everywhere and is covered by the hash.
- **(d) Shared band rule for the forced-binary label.** The label uses the one band rule in
  `src.risk.decisions` with `half_width = 0` (D29: no uncertain band, no calibration), so the
  baseline and the CNN are thresholded by the same code.
- **(e) Batched scoring for Stage 13 memory.** Stage 13 scores ~15,000 windows; one kernel matrix
  against thousands of support vectors could need hundreds of MB in a single allocation. Batches
  of 2048 rows bound it, and a test checks the batched score equals the one-shot formula (≤ 1e-12)
  for 1, 2048 and 5000 rows.
- **(f) Layering.** Stage 13 imports this module to score, so it does not import `src.evaluation` or
  `src.tuning` (research code), nor scikit-learn, pickle or joblib. Fitting lives in `src/tuning`
  (step B2).

## Tests (`tests/models/test_baseline.py`)

Synthetic data: two 40-dimensional Gaussian classes with different feature scales, seed 2026.

| Property | Test |
|---|---|
| Train-only scaler | `test_fit_scaler_uses_train_only` (scaler = train statistics; val with different statistics does not change it) |
| Scaler input checks | `test_fit_scaler_rejects_bad_input` (constant feature, NaN, inf, wrong shape) |
| Scorer = scikit-learn | `test_score_equals_sklearn_decision_function` (C=1, gamma=0.025; train and shifted val; ≤ 1e-9) |
| Batched scoring | `test_batched_score_equals_one_shot` (1, 2048, 5000 rows; ≤ 1e-12 vs one batch) |
| Sign and label | `test_spoof_scores_higher_and_predict_is_forced_binary` (never `uncertain`; synthetic ⇔ score ≥ threshold) |
| Round trip | `test_save_load_round_trip` (identical scores, no `.pkl`, npz loads with `allow_pickle=False`), `test_save_is_byte_deterministic` (two saves: identical bytes, same `artifact_sha256`), `test_save_refuses_non_empty_folder` |
| Tampering | `test_tampering_changes_artifact_hash` (one value in npz or JSON), `test_load_rejects_corrupted_artifact` (renamed/extra key, schema, n_features, missing array, shape, NaN, object array, corrupt npz, missing file) |
| Refusals | `test_from_sklearn_refuses_other_classes_and_kernels` (labels 1/2, three classes, linear/poly/sigmoid, unfitted), `test_from_sklearn_refuses_non_numeric_gamma` (`"scale"`, `"auto"`), `test_model_rejects_invalid_fields` |
| Layering | `test_baseline_does_not_import_research_code_or_sklearn` |

## Step B2: fitting

`src/tuning/baseline_fit.py` (research tier: it uses `eer()` from `src.evaluation` and
scikit-learn) fits the baseline from the settings in `configs/baseline.yaml`.

### Conventions

- Features: n × 40 finite floats; labels 1 = spoof, 0 = bonafide, both classes present; feature
  rows and labels of equal length. Anything else raises `ValueError`.
- `configs/baseline.yaml` (D37), read with `yaml.safe_load`: `decision: D37`,
  `features: mfcc_mean_std_40`, `scaling: standardize_train_only`, `grid: {C: [0.1, 1, 10, 100],
  gamma: [0.001, 0.01, 0.025, 0.1]}`, `selection: lowest_val_a_eer`, `tie_break: [C, gamma]`,
  `score: decision_function`, `threshold: d29_val_b_eer`.
- `load_baseline_config(path) -> BaselineConfig` (frozen): strict keys; the fixed fields must have
  exactly the D37 values (the only ones the code implements); grid values must be finite numbers
  > 0, strictly increasing.

### Method

1. `select_baseline(train_X, train_y, val_a_X, val_a_y, *, config)`: `fit_scaler` on train only;
   for each (C, gamma) in grid order (C outer, gamma inner), fit `SVC(kernel="rbf", C, gamma)` on
   standardized train and compute the val-A EER of `decision_function` with `eer()`. The selected
   configuration is the minimum of the sort key (EER, C, gamma): lowest EER, then smaller C, then
   smaller gamma. Returns the SVC, the scaler (mean, std) and a `SelectionReport` (frozen): all 16
   rows (C, gamma, val-A EER, number of support vectors), the selected (C, gamma), n_train,
   n_val_a, the scikit-learn version (`sklearn.__version__`) and the SVC settings of every fit
   (`kernel`, `tol`, `shrinking`, `cache_size`, `max_iter`, `class_weight`, read from
   `get_params()`; they must be the same for all 16 fits); `to_json()` writes sorted keys.
2. `fit_baseline_threshold(model_scores_val_b, val_b_y) = eer(val_b_y, scores)[1]`: the val-B EER
   point on the score scale (D29), forced binary.
3. `fit_baseline(train…, val_a…, val_b…, *, config, version)` runs step 1, copies the SVC with
   `from_sklearn`, scores val-B with `BaselineModel.score` (the NumPy scorer Stage 13 uses), sets
   the threshold from step 2 and returns `(BaselineModel, SelectionReport, ThresholdReport)`. The
   `ThresholdReport` holds the threshold, the val-B EER there and the val-B class counts.

### Why (team reasoning)

- **(a) Selection and threshold use disjoint data (D9).** val-A is passed only to
  `select_baseline`, val-B only to the threshold step; they are separate functions, and a test spies
  on every `eer()` call to check that selection sees only val-A labels and the threshold only val-B
  labels, and that a different val-B leaves the selection unchanged.
- **(b) Config drift guard (D37).** The grid and scaling were fixed before any val-A result. The
  loader refuses other values for the fixed fields, and a test asserts the whole config equals
  D37; if it fails, the change needs a new decision, not a test edit.
- **(c) All 16 results kept.** The report lists every configuration, not only the winner, so the
  reviewer can see how close the choice was and that the tie rule was applied.
- **(d) `decision_function` as the score (D37).** The D29 threshold is fitted on the same score the
  artifact produces and Stage 13 stores (`BaselineModel.score`), so threshold and scores share one
  scale; no probability outputs are used.
- **(e) Provenance.** The fitted SVM depends on the scikit-learn version and on solver settings
  (tolerance, shrinking, iteration limit, class weights) that D37 does not list. Recording them in
  the report makes the fit reproducible and shows that the pinned version and the defaults were
  used.

### Tests (`tests/tuning/test_baseline_fit.py`)

Synthetic data: two 40-dimensional Gaussian classes (spoof shifted by 0.25 per feature, feature
scales 1..40), train / val-A / val-B with 60 / 40 / 50 clips per class, seeds derived from 2026.

| Property | Test |
|---|---|
| Config = D37 | `test_config_equals_d37` (drift guard) |
| Full grid, lowest EER | `test_all_16_configurations_in_grid_order_and_lowest_eer_selected` |
| Tie rule | `test_tie_goes_to_smallest_c_then_smallest_gamma` (constant EER ⇒ C 0.1, gamma 0.001) |
| Data separation (D9) | `test_val_a_only_for_selection_val_b_only_for_threshold` (eer spy; val-B change; scaler = train statistics) |
| D29 threshold | `test_threshold_is_val_b_eer_point_of_model_score_and_predict_is_binary` (bit-for-bit; never `uncertain`) |
| Provenance | `test_report_records_sklearn_version_and_svc_settings` (pinned version, default SVC settings, in the JSON) |
| Determinism | `test_fitting_twice_is_identical` (reports, scores, `artifact_sha256`), `test_report_json_has_sorted_keys_and_all_rows` |
| Refusals | `test_invalid_inputs_raise` (shape, one class, NaN, length, label 2, one-class val-B), `test_bad_config_raises` (extra/missing key, other score, unordered/negative/NaN/non-numeric grid, other tie rule), `test_select_refuses_non_config` |
