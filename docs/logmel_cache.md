# Log-Mel feature cache

Stage 8 PR A (D42 item "Features"). The CNN (D42) is trained on the F3 log-Mel of every FoR train
clip and early-stopped on val-A; val-B is used from Stage 10 (D9). As for the baseline, decoding,
resampling and log-Mel extraction are the slow part, so the spectrograms are computed once with the
frozen F3 profile (D39) and stored as a verified cache. The cache holds features only: no model, no
normalization statistics, no threshold.

The design, method and reasons are the same as the MFCC cache: test split refused in code (D31),
non-pickle, F2 and F3 binding, single read, deterministic bytes, all or nothing. They are written
once in [feature_cache.md](feature_cache.md) and not repeated here. This note records only what
differs.

## What differs from the MFCC cache

| | MFCC cache | Log-Mel cache |
|---|---|---|
| Module | `src/features/mfcc_cache.py` | `src/features/logmel_cache.py` |
| Feature | `extract_features(...)[2]`, MFCC mean + std | `extract_features(...)[1]`, 64-bin natural-log Mel (D39) |
| `features` array | n × 40 float32 | n × 1 × 64 × 197 float32 |
| Extra JSON keys | none | `"feature": "log_mel"`, `"feature_shape": [1, 64, 197]` |
| Per clip | 160 B | 50,432 B (about 50 KB) |
| Train split (12,404 clips) | about 2 MB | about 625 MB |

- `build_logmel_cache` checks every computed log-Mel: anything other than float32 `(1, 64, 197)`
  raises and nothing is written.
- `load_logmel_cache` checks the two extra keys exactly (strict key set, as for the MFCC JSON) and
  the 4-D shape of `features`; the other checks are those of `load_mfcc_cache`.
- CLI: `python -m src.features.logmel_cache --root <FoR root> --manifest <F2 csv> --profile
  configs/preprocessing.yaml --split <train|val-A|val-B> --out <new folder>`.
- The split, manifest and metadata helpers are imported from `src/features/mfcc_cache.py`
  (including private ones) and the npz writer from `src/models/baseline.py`, so both caches share
  one implementation of those rules.

## Normalization is not stored here

D42 standardizes each Mel bin with 64 means and 64 standard deviations computed on the FoR train
log-Mel only. These statistics are computed later, from this train cache, by the training code, and
stored in the checkpoint. The cache stays a pure function of audio + profile, so the same val-A and
val-B caches serve every D30 configuration and the statistics cannot be fitted on a split other
than train by accident here.

## Limitations

- Everything in [feature_cache.md](feature_cache.md#limitations) applies.
- Loading holds the npz bytes and the decoded arrays at the same time (single-read rule), so loading
  the train split needs about 1.3 GB of memory at peak. **[hypothesis]** This fits a Colab runtime;
  it has not been measured on the real data yet.
- Tested on synthetic WAV files only; the real build runs on Colab.

## Tests (`tests/features/test_logmel_cache.py`)

Same synthetic corpus and the same tests as `tests/features/test_mfcc_cache.py` (see
[feature_cache.md](feature_cache.md#tests-testsfeaturestest_mfcc_cachepy)), with log-Mel shapes,
plus:

| Property | Test |
|---|---|
| Feature name and shape recorded | `test_metadata_records_feature_and_shape` |
| Wrong extractor output refused, nothing written | `test_unexpected_log_mel_shape_or_dtype_raises_and_writes_nothing` |
| Changed `feature` / `feature_shape` refused | extra cases in `test_load_refuses_changed_json` |
| 3-D, wrong-width, float64 or NaN features refused | extra cases in `test_load_refuses_bad_arrays_even_with_matching_hash` |
