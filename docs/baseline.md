# MFCC/SVM baseline

The baseline is an RBF-SVM on the 40-value MFCC vector (mean + std of 20 coefficients, Protocol
v1.0 §3, §11). It is **evaluation-only** (D7): it is compared with the CNN on every split and at the
external run, but it is never deployed and has no risk policy. It gets its own forced-binary
threshold at the val-B EER point of its score (D29); its tuning, train-only scaling and score are
fixed by D37; its artifact is recorded in the F4 freeze (D36).

The work has two steps:
- **Step B1 (this step):** train-only scaler, NumPy scorer and a non-pickle artifact,
  `src/models/baseline.py`. Tested on synthetic data only; no dataset is read.
- **Step B2 (to follow):** fitting and the D37 grid search, `src/tuning/baseline_fit.py`.

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
   distance is expanded and clipped at 0). Higher score = more likely spoof.
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
- **(e) Layering.** Stage 13 imports this module to score, so it does not import `src.evaluation` or
  `src.tuning` (research code), nor scikit-learn, pickle or joblib. Fitting lives in `src/tuning`
  (step B2).

## Tests (`tests/models/test_baseline.py`)

Synthetic data: two 40-dimensional Gaussian classes with different feature scales, seed 2026.

| Property | Test |
|---|---|
| Train-only scaler | `test_fit_scaler_uses_train_only` (scaler = train statistics; val with different statistics does not change it) |
| Scaler input checks | `test_fit_scaler_rejects_bad_input` (constant feature, NaN, inf, wrong shape) |
| Scorer = scikit-learn | `test_score_equals_sklearn_decision_function` (C=1, gamma=0.025; train and shifted val; ≤ 1e-9) |
| Sign and label | `test_spoof_scores_higher_and_predict_is_forced_binary` (never `uncertain`; synthetic ⇔ score ≥ threshold) |
| Round trip | `test_save_load_round_trip` (identical scores, no `.pkl`, npz loads with `allow_pickle=False`), `test_save_is_byte_deterministic` (two saves: identical bytes, same `artifact_sha256`), `test_save_refuses_non_empty_folder` |
| Tampering | `test_tampering_changes_artifact_hash` (one value in npz or JSON), `test_load_rejects_corrupted_artifact` (renamed/extra key, schema, n_features, missing array, shape, NaN, object array, corrupt npz, missing file) |
| Refusals | `test_from_sklearn_refuses_other_classes_and_kernels` (labels 1/2, three classes, linear/poly/sigmoid, unfitted), `test_from_sklearn_refuses_non_numeric_gamma` (`"scale"`, `"auto"`), `test_model_rejects_invalid_fields` |
| Layering | `test_baseline_does_not_import_research_code_or_sklearn` |
