# Model bundle

The ModelBundle is the one frozen artifact the deployed system and the external run use: the CNN
checkpoint, the preprocessing profile and the risk policy, fitted on FoR only and frozen together in
F4 before the external run (D2, AGENTS.md hard rule 2, Protocol v1.0 §6 and §12). It is the
**C→D** hand-over (the bundle Stage 13 scores with) and the **C→E** hand-over (the bundle the API
loads, its hash verified, refusal to start otherwise; Protocol v1.0 §11, D26; SRS v4.0 FR-29, SDD
v2.0 §3.5 and §6.4). It lives in `src/bundle/`, the Model Registry of the shared core (SDD Fig. 1).

The layout is fixed by **D59** (proposed in the PR that adds it): the Stage 8 checkpoint is a folder
of two files (D42), so the bundle keeps it as its own folder and records its combined hash. This
step defines the format, the verifying loader and the model loader. It is tested on synthetic files
only; no dataset, Drive or real checkpoint is read.

## Format

One folder:

```
<bundle>/
  bundle_manifest.json
  policy.json                 bytes of the source policy file, unchanged
  <profile file name>         e.g. preprocessing.yaml, bytes unchanged
  checkpoint/
    checkpoint.json           bytes unchanged
    weights.safetensors       bytes unchanged
```

| Part | Source for F4 | Content |
|---|---|---|
| `checkpoint/` | Stage 8 `seed_0/` (bundle seed) | D42 checkpoint: `checkpoint.json` (strict metadata) + `weights.safetensors` (state_dict and normalization statistics); combined `checkpoint_sha256` `c33eb820…` ([stage8 evidence](audit_evidence/stage8_cnn/README.md)) |
| profile | `configs/preprocessing.yaml` | Preprocessing profile (F3), fingerprint `5b4f5849…` |
| `policy.json` | Stage 10 `risk_policy.json` | `RiskPolicy.to_json()` ([risk_policy.md](risk_policy.md)); SHA-256 `a74f1f51…` ([stage10 evidence](audit_evidence/stage10_risk/README.md)) |
| `bundle_manifest.json` | written by `build_bundle` | The manifest below |

`baseline-v1` is not in the bundle (D36, D7): its `artifact_sha256` (`42cc4d53…`) is recorded with
the F4 `manifest_sha256` in the freeze entry (D59 (6)). F5 uses the same layout and hash rule (D59 (7)).

`src/bundle/bundle.py` (core, torch-free) holds:
- `MANIFEST_NAME = "bundle_manifest.json"`, `SCHEMA_VERSION = 2`,
  `COMPONENTS = ("checkpoint", "profile", "policy")`, `POLICY_FILE = "policy.json"`,
  `CHECKPOINT_DIR = "checkpoint"`, `CHECKPOINT_FILES = ("checkpoint.json", "weights.safetensors")`
  (same names and order as `cnn_checkpoint.FILES`).
- `checkpoint_dir_sha256(files) -> str`: the D42 rule, SHA-256 of the ASCII text
  `"checkpoint.json:<sha256>\nweights.safetensors:<sha256>\n"`, the same value as
  `cnn_checkpoint.checkpoint_sha256` (a test proves it on a real checkpoint).
- `Component(file, sha256, version)` for the profile and the policy: plain file name (not the
  manifest name, not `checkpoint`), 64 lowercase hex, non-empty version.
- `CheckpointComponent(dir, files, sha256, version)`: `dir == "checkpoint"`, `files` has exactly
  `CHECKPOINT_FILES` with 64-hex values, and `sha256 == checkpoint_dir_sha256(files)` (checked when
  parsed).
- `BundleManifest(schema_version, bundle_version, created_at, code_commit, protocol_version,
  f2_ledger_sha256, components)`, a frozen dataclass. `created_at` is ISO 8601 UTC
  (`YYYY-MM-DDTHH:MM:SSZ`, optional fraction), `code_commit` 40 lowercase hex, `protocol_version`
  like `1.0`, `f2_ledger_sha256` the F2 hash ledger (D28). In JSON the checkpoint entry is
  `{"dir", "files": {"checkpoint.json", "weights.safetensors"}, "sha256", "version"}`, the two others
  `{"file", "sha256", "version"}`. `to_json` / `from_json` are strict: sorted keys, 2-space indent,
  trailing newline, and unknown or missing keys at any level raise `ValueError`. Schema 1 (one flat
  checkpoint file) is retired: no schema-1 bundle was ever built.
- `build_bundle(directory, *, checkpoint_dir, expected_checkpoint_sha256, profile, policy,
  bundle_version, model_version, profile_version, code_commit, protocol_version, f2_ledger_sha256,
  created_at) -> Path` (the manifest path). `profile` and `policy` are file paths.
- `LoadedBundle(directory, manifest, checkpoint_dir, checkpoint_sha256, profile_path, profile,
  policy, manifest_sha256)` (`profile` is the parsed `PreprocessingProfile`) and
  `load_bundle(directory) -> LoadedBundle`.

`src/bundle/model_loader.py` (torch; the API and Stage 13) holds `load_bundle_model(directory) ->
BundleModel(bundle, model, mean, std, metadata)`.

`src/preprocessing/profile.py` gains `parse_profile(text, source)`; `load_profile(path)` reads the
file and calls it (same errors, same result, same fingerprint).

## Method

Build (every check before anything is written):
1. Refuse if the folder exists and is not empty.
2. The checkpoint folder holds exactly `CHECKPOINT_FILES`, regular files, no symlinks; each is read
   once and the combined hash must equal `expected_checkpoint_sha256` (the Stage 8 recorded value),
   which refuses a wrong seed or a changed file.
3. The policy file's bytes are read once and must load as a `RiskPolicy`; the policy version is its
   own `version`. The bytes are copied unchanged, so the manifest hash equals the source file's.
4. The profile's bytes are read once and parsed (`parse_profile`); its `profile_version`,
   `protocol_version` and `f2_ledger_sha256` must equal the arguments.
5. `checkpoint.json` is parsed as JSON (no torch): `version == model_version` and
   `profile_fingerprint == profile.fingerprint()`.
6. Write the bytes read above (profile, `policy.json`, `checkpoint/` and its two files), re-hash every
   copy, write the manifest last.
7. If any write fails, remove every file and folder this call wrote (including `checkpoint/`, and the
   bundle folder if this call created it), then re-raise: no half-built bundle is left behind.

Load (read-only; nothing is returned until every check passes; any failure is a `ValueError` naming
the check and the part):
1. The manifest is read once and parses strictly (plain names, checkpoint entry shape, combined hash
   consistent with the per-file hashes); `manifest_sha256` is computed from those bytes.
2. The folder holds exactly the manifest, `policy.json`, the profile file and `checkpoint/`;
   `checkpoint/` is a real directory (not a symlink) holding exactly `CHECKPOINT_FILES`.
3. Every file is a regular file and its SHA-256 matches (per file for the checkpoint, then the
   combined value). `policy.json`, the profile and `checkpoint.json` are read once: the bytes that are
   hashed are the bytes that are parsed. `weights.safetensors` is hashed streamed, not parsed.
4. Policy: loads as a `RiskPolicy`; its version equals the manifest's.
5. Profile: parses; `profile_version`, `protocol_version` and `f2_ledger_sha256` equal the manifest's.
6. `checkpoint.json`: `version` equals the manifest checkpoint version; `profile_fingerprint` equals
   the bundled profile's fingerprint.

Model load (`load_bundle_model`): `load_bundle`, then `cnn_checkpoint.load_checkpoint(checkpoint_dir,
expected_sha256=checkpoint_sha256)`, which hashes the bytes it loads against the manifest anchor;
then the loaded metadata `version` must equal the manifest checkpoint version.
`src/models/cnn_checkpoint.py` is unchanged.

## Why (team reasoning)
- **(a) Verify before use.** NFR-09 and threat model §10 (tampering: model, profile or thresholds
  replaced) require a refusal at load, and §10 recommends checking the checkpoint hash; SDD v2.0
  §3.5 (2) checks every component and hash before activation, and §6.4 rejects missing, extra or
  mismatched files. Hashing every file, and checking the folder contents at both levels, means a
  replaced or altered part stops the API or the external run instead of producing numbers from an
  unknown model.
- **(b) Path safety.** The manifest is data. If it could name `../x`, an absolute path or `sub/x`,
  an edited manifest could make the loader hash or load a file outside the bundle. Plain names are
  checked while the manifest is parsed, before any file is opened; the checkpoint folder and its two
  names are fixed constants, not manifest data.
- **(c) Immutability.** A frozen bundle is never changed (Protocol v1.0 §12): `build_bundle` never
  writes into a non-empty folder and `load_bundle` only reads. A new bundle is a new folder and a new
  version.
- **(d) Provenance.** Every run is dataset version + config + code version → results (AGENTS.md).
  The manifest records the code commit, the protocol version and the F2 ledger hash (D28), so a
  bundle names the data, protocol and code it was built from.
- **(e) External anchor for the manifest hash.** A file cannot contain its own SHA-256. The loader
  returns `manifest_sha256` and the F4 freeze decision records it, so an edited manifest, including
  fields no file can confirm (such as the bundle version), is detected by comparing with the
  decision. Same inputs and `created_at` give byte-identical manifests, so the hash can be reproduced.
- **(f) Single read.** A file read once to hash and again to parse could be swapped in between;
  reading the manifest, `policy.json`, the profile and `checkpoint.json` once closes that gap for
  every part the loader parses.
- **(g) Checkpoint kept as its own folder (D59 (1)–(3), (5); SRS v4.0 FR-29).** The Stage 8
  checkpoint is two files whose identity is the combined `checkpoint_sha256` (`c33eb820…`). Keeping
  the folder byte for byte keeps that recorded value as the model's identity in the manifest, and
  lets `load_bundle_model` reuse the reviewed non-pickle `load_checkpoint` unchanged, with
  `expected_sha256` set to the manifest value, so the bytes loaded are the bytes verified (SDD v2.0
  §3.5 (1): F4 is loaded only through the verifying loader). An archive or a re-saved single file
  would give a new hash unrelated to Stage 8 (D59, options rejected).
- **(h) Cross-checks between parts (D59 (4); SRS v4.0 FR-04, NFR-09; SDD v2.0 §3.5 (3)).** Hashes
  prove each file is the recorded one, not that the parts belong together. The profile's version,
  protocol and F2 ledger must equal the manifest's, and the checkpoint's `profile_fingerprint` must
  equal the bundled profile's (`5b4f5849…`), so a bundle pairing a model with a profile it was not
  trained on is refused: under FR-04 and NFR-09 a profile mismatch blocks activation. The
  checkpoint's own `version` must equal the manifest model version, so that field is checked against
  a file too.

**Limitations (what remains):**
- **Profile by path.** The profile and checkpoint hashes are checked, but `profile_path` is still a
  path a caller could re-read after the check. The API must use `LoadedBundle.profile` (parsed from
  the verified bytes), never re-load the profile from `profile_path`.
- **Weights not parsed by `load_bundle`.** `load_bundle` is torch-free, so it hashes
  `weights.safetensors` but does not parse it; its content (tensor names, shapes, dtypes, finite
  values) is checked by `load_checkpoint` inside `load_bundle_model`. Use `load_bundle_model` for
  scoring, not `load_bundle` plus a separate checkpoint load.
- **Unsafe deserialization.** Handled by the D42 format: safetensors weights and strict JSON
  metadata, never `torch.load` or pickle. v2 checkpoints use the same two files (D59 (7)).

Closed by D59 (were limitations of the format step): the profile version is now checked against the
profile file (load step 5); check-then-use for the checkpoint is closed by `load_bundle_model`.

## Real build (F4)
Not in this step. Stage 12's next PR builds F4 from Stage 8 `seed_0/` (expected
`checkpoint_sha256` `c33eb820…`), Stage 10 `risk_policy.json` (`a74f1f51…`) and
`configs/preprocessing.yaml`, records `manifest_sha256` with `baseline-v1`'s `artifact_sha256` in the
freeze entry, and wires `load_bundle_model` into the API in place of the mock (D48).

## Tests
`tests/bundle/test_bundle.py` (no torch; dummy `checkpoint.json` with `version` and
`profile_fingerprint`, dummy weight bytes, a tmp copy of `configs/preprocessing.yaml`, a tmp policy
file). Each case has its reasoning as a comment.

| Point | Test |
|---|---|
| Round trip | `test_build_then_load_round_trip` (per-file and combined hashes, versions, `checkpoint_sha256`, policy hash = source file SHA-256, bundle bytes = source bytes for all four files, parsed profile), `test_manifest_checkpoint_entry_shape`, `test_checkpoint_dir_sha256_rule` |
| (g) build refusals | `test_build_refuses_wrong_expected_checkpoint_sha256`, `test_build_refuses_checkpoint_folder_contents` (extra, missing), `test_build_refuses_symlinked_checkpoint_file` (skipped where symlinks are unavailable, e.g. Windows without Developer Mode) |
| (h) build refusals | `test_build_refuses_declared_value_mismatch` (profile version, protocol, F2 ledger, model version), `test_build_refuses_checkpoint_profile_fingerprint_mismatch`, `test_build_refuses_invalid_policy_file`; each leaves nothing written |
| (a) tampering | `test_tampered_file_raises` (one bit flipped in `checkpoint.json`, `weights.safetensors`, the profile, the policy) |
| (a), (e), (h) edited manifest | `test_edited_manifest_sha256_raises`, `test_edited_manifest_policy_version_raises`, `test_edited_manifest_profile_version_raises`, `test_edited_manifest_checkpoint_version_raises`, `test_edited_manifest_changes_manifest_sha256` |
| Strict manifest | `test_invalid_manifest_raises` (schema 1 and 3, missing / unknown keys and components, checkpoint `files` keys wrong or not an object, combined `sha256` inconsistent with `files`, `dir` other than `checkpoint`, reserved name, commit, date, protocol format), `test_schema_1_manifest_is_named_as_retired` |
| (h) cross-checks | `test_fingerprint_mismatch_in_rehashed_bundle_raises`, `test_invalid_checkpoint_json_in_rehashed_bundle_raises`, `test_invalid_profile_in_rehashed_bundle_raises` (manifest re-hashed, so only the cross-check or parse can catch it) |
| (b) path safety | `test_unsafe_component_path_raises_and_reads_nothing` (`../x`, absolute POSIX and Windows paths, `sub/x`, `sub\x`, `..`, `.`; nothing hashed, only the manifest read) |
| (a) folder contents | `test_extra_top_file_raises`, `test_missing_top_file_raises`, `test_missing_manifest_raises`, `test_extra_file_in_checkpoint_dir_raises`, `test_missing_file_in_checkpoint_dir_raises`, `test_checkpoint_dir_replaced_by_file_raises` |
| (a) policy | `test_policy_version_mismatch_at_load_raises`, `test_invalid_policy_at_load_raises` |
| (f) single read | `test_policy_is_parsed_from_the_hashed_bytes`, `test_profile_and_checkpoint_json_are_parsed_from_the_hashed_bytes` (each file swapped right after its one read; the verified values are returned) |
| (c) immutability | `test_build_into_non_empty_directory_raises` (folder bytes unchanged), `test_loaded_bundle_is_immutable`, `test_build_into_empty_directory_is_allowed`, `test_failed_build_removes_what_it_wrote` (each of the five writes fails → error re-raised; created folders, including `checkpoint/`, gone; pre-existing folder empty) |
| (e) determinism | `test_build_is_deterministic` (manifest bytes and the whole tree) |
| Layering | `test_bundle_does_not_import_research_code`, `test_bundle_module_is_torch_free` (no `torch`, `safetensors`, `src.models`, `src.evaluation`, `src.tuning` import, and none loaded at import time) |

`tests/bundle/test_model_loader.py` (torch; a tiny real D42 checkpoint whose `profile_fingerprint` is
the real profile's):

| Point | Test |
|---|---|
| Same rule as D42 | `test_checkpoint_constants_and_hash_rule_match_cnn_checkpoint` (`CHECKPOINT_FILES == cnn_checkpoint.FILES`; `checkpoint_dir_sha256` = `checkpoint_sha256(folder)`) |
| Round trip | `test_build_then_load_bundle_model` (eval mode, equal state_dict, equal mean/std) |
| (g) check-then-use | `test_weight_changed_after_load_bundle_is_refused` (a weight byte changes between `load_bundle` and `load_checkpoint` → `checkpoint_sha256 mismatch`) |
| (h) version | `test_checkpoint_version_mismatch_is_refused`, `test_model_loader_checks_loaded_version_itself` |
| Layering | `test_model_loader_does_not_import_research_code` |

`tests/preprocessing/test_profile.py` adds `test_profile_fingerprint_is_pinned_and_parse_profile_agrees`
(fingerprint `5b4f5849…` from the file and from its text) and
`test_parse_profile_names_the_source_on_bad_yaml`.
