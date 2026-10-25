# MFCC feature cache

Stage 7 step 1. The baseline (D7, D37) is fitted on the 40-value MFCC vector of every FoR train,
val-A and val-B clip. Decoding, resampling and MFCC extraction for every non-test F2 clip is the slow
part of the baseline, so the vectors are computed once with the frozen F3 profile (D39) and
stored as a verified cache that step 2 feeds to `fit_baseline`. The cache holds features only: no
model is fitted and no threshold is chosen here.

## Format

`src/features/mfcc_cache.py` holds:
- `CACHE_SPLITS = {"train", "val-A", "val-B"}`, `SCHEMA_VERSION = 1`.
- `build_mfcc_cache(root, manifest, profile, split, out_dir, *, f2_manifest_sha256,
  profile_config_sha256, code_commit)`: returns the metadata dict.
- `load_mfcc_cache(out_dir, split, profile) -> (features, labels, paths, metadata)`.
- CLI: `python -m src.features.mfcc_cache --root <FoR root> --manifest <F2 csv> --profile
  configs/preprocessing.yaml --split <train|val-A|val-B> --out <new folder>`.

One cached split is one new folder holding exactly two files:

| File | Content |
|---|---|
| `<split>.npz` | `features` (n × 40 float32), `labels` (n int8; 1 = spoof/FAKE, 0 = bonafide/REAL), `paths` (n fixed-width unicode, sorted) |
| `<split>.json` | `schema_version`, `split`, `n_rows`, `n_spoof`, `n_bonafide`, `profile_version`, `profile_fingerprint`, `profile_config_sha256`, `f2_manifest_sha256`, `code_commit`, `npz_sha256` (sorted keys) |

The npz is written by the deterministic writer of `src/models/baseline.py` (uncompressed zip, one
`<name>.npy` entry per array sorted by name, fixed date and system/permission fields,
`allow_pickle=False`), so the same inputs give the same bytes.

## Method

1. The split is checked first: anything other than `train`, `val-A`, `val-B` raises `ValueError`
   before any file is read. The provenance values must be hex digests; an existing `out_dir` is
   refused.
2. The rows of that split are sorted by path; duplicate paths, an empty split or a label other than
   `FAKE`/`REAL` raise.
3. Each row is read with `read_verified_audio` (sample SHA-256 checked against the manifest before
   decoding), then `extract_features(waveform, sample_rate, profile)[2]` gives the 40-value vector.
4. Only after every row succeeds is `out_dir` created and the npz then the JSON written; if a write
   fails, the folder is removed.
5. `load_mfcc_cache` reads the JSON (exact keys, schema 1, split, counts, digests) and checks
   `f2_manifest_sha256` against `expected_f2_manifest_sha256` (keyword argument, default the
   approved D28 value `F2_MANIFEST_SHA256` from `src.preprocessing.validation`; a mismatch raises
   "F2 manifest check failed"). It then reads the npz into memory once, checks the SHA-256 of
   those bytes, checks the profile fingerprint and version, and loads the same bytes with
   `allow_pickle=False` (exact entries); it checks shapes, dtypes, finiteness, label values and
   counts, and path order. Each failure raises `ValueError` naming the check.
6. The CLI refuses a dirty worktree (`code_version`, before reading any input), loads the manifest
   with `load_f2_manifest` (approved F2 hash and schema) and records the SHA-256 of the manifest and
   profile files and the commit.

## Why (team reasoning)

- **(a) Test split excluded.** The FoR test split is evaluated once at Stage 11 (D31). A cache of
  its features would make a second, unrecorded test pass easy, so the split is refused in code
  before any file is read, and the CLI does not offer it. Test-split features are computed at
  Stage 11 under the run registry.
- **(b) Non-pickle.** Same reason as the baseline artifact ([baseline.md](baseline.md)): an npz read
  with `allow_pickle=False` and a strict JSON cannot run code; paths are fixed-width unicode, never
  object arrays.
- **(c) Hash and fingerprint binding.** Each sample hash is checked before decoding, and the JSON
  binds the arrays to the F2 manifest hash, the profile config hash and fingerprint and the code
  commit. Loading with a different profile fails, so features from another profile cannot reach
  the fit silently; a changed npz fails its hash. The npz is read once and the hashed bytes are the
  parsed bytes, so the file cannot change between the check and the load. By default, loading
  also refuses a cache whose recorded manifest hash is not the approved F2 value, so a cache
  built from another manifest cannot reach the official fit; tests pass their synthetic hash
  explicitly.
- **(d) Deterministic bytes.** Sorted rows, the fixed-metadata npz writer and sorted JSON keys make
  two builds byte-identical, so a reviewer can rebuild and compare `npz_sha256`.
- **(e) All or nothing.** A cache missing a failed clip would bias the fit silently; any failure
  raises and nothing is written (fail loudly, AGENTS.md code rules).

## Limitations

- One folder per split: `out_dir` must not exist, so train, val-A and val-B go in three folders.
- `build_mfcc_cache` records the provenance values it is given; only the CLI computes them from the
  files and Git. Library callers are trusted to pass the real values.
- Feature values depend on the pinned librosa/soxr versions (D39); the profile fingerprint does not
  cover library versions.
- Tested on synthetic WAV files only; the real build runs on Colab in step 2.

## Tests (`tests/features/test_mfcc_cache.py`)

Synthetic data: 8 PCM-16 WAV files (tones, noise, one 22.05 kHz stereo clip) over train, val-A,
val-B and test, with a manifest whose hashes match the files; seed 2026.

| Property | Test |
|---|---|
| Shapes, dtypes, labels, order | `test_shapes_dtypes_labels_and_path_order`, `test_validation_splits_are_cached` |
| Test split refused, nothing read | `test_test_and_unknown_splits_raise_before_reading` (spy on `read_verified_audio` and `sha256_file`), `test_cli_refuses_test_split_before_git_or_inputs` |
| Sample hash mismatch | `test_sample_hash_mismatch_raises_and_writes_nothing` |
| Existing folder | `test_existing_out_dir_is_refused` |
| Input checks | `test_invalid_label_and_provenance_raise` |
| Determinism | `test_two_builds_are_byte_identical` (npz and JSON bytes, sorted keys, `npz_sha256`) |
| Non-pickle | `test_npz_has_no_object_arrays`, `test_load_refuses_object_array` |
| Same features as the profile | `test_features_equal_extract_features` (exact equality) |
| Tampering | `test_load_refuses_tampered_npz`, `test_load_refuses_changed_json` (extra, missing, renamed key; split; schema; counts), `test_load_refuses_bad_arrays_even_with_matching_hash`, `test_load_refuses_missing_file_and_corrupt_npz` |
| Profile binding | `test_load_refuses_other_profile_fingerprint` |
| Single read | `test_load_reads_npz_once_and_checks_the_loaded_bytes` (`Path.read_bytes` spy: npz read exactly once) |
| F2 binding | `test_load_default_refuses_non_f2_manifest_hash` (default expected hash = approved F2 value) |
| CLI | `test_cli_records_provenance_from_files`, `test_cli_refuses_dirty_worktree_before_reading` |
