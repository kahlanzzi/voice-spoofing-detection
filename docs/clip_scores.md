# Per-clip score files

Stage 11 (D41, D45; Protocol v1.0 §7). The FoR test split is scored once (D31), so every later use
of those scores reads files written by that one run: Stage 14 pairs its degraded scores with the
clean ones by clip ID (no second clean pass, D41), and Stage 16 reads them for H7 and the D43
loudness diagnostic. This note covers the format, the writer/reader in
`src/evaluation/clip_scores.py` and its synthetic tests; the Stage 11 runner that calls it is a
later, separately reviewed PR. The registry it refers to is described in
[evaluation_harness.md](evaluation_harness.md) "Run registry".

## Files

One registered run (`RunRegistry.claim`, purpose `stage11_test`) writes two files per model into
a folder passed by the caller (no path is hard-coded):

| File | Content |
|---|---|
| `<model_id>.csv` | columns `clip_id, label, group_id, raw_score, p, prediction`, in this order; header line; one row per clip, sorted by `clip_id`; UTF-8; LF line endings, also after the last row |
| `<model_id>.json` | sidecar, strict JSON, sorted keys: `evaluation_purpose`, `run_id`, `model_id`, `model_role`, `model_artifact_sha256`, `manifest_sha256`, `n_rows`, `csv_sha256`, `applicable_columns` |

```
write_clip_scores(out_dir, *, evaluation_purpose, run_id, model_id, model_role,
                  model_artifact_sha256, manifest_sha256, clip_id, label, group_id,
                  raw_score, p=None, prediction=None) -> (csv_path, json_path)
read_clip_scores(out_dir, model_id) -> ClipScores
```

### Columns

| Column | Value |
|---|---|
| `clip_id` | the F2 manifest `path`: canonical relative POSIX path (the rule of `src.freezing.manifests.safe_relative_path`, reused), unique |
| `label` | the F2 manifest label, `FAKE` or `REAL` (positive class = spoof) |
| `group_id` | the D14 near-duplicate group, F2 manifest column `group` (Stage 4 built it from `neardup_group_095`) |
| `raw_score` | the model's continuous score, higher = more likely spoof (CNN logit; baseline decision function) |
| `p` | calibrated P(spoof) from the frozen F4 risk policy, in [0, 1] |
| `prediction` | `authentic` / `synthetic` / `uncertain` (`src.risk.decisions`) |

Floats are written as `repr(float(x))` of the float64 value, which reads back bit for bit; a
float32 score becomes its exact float64 value first. NaN and infinity are refused.

### Applicability (fixed in code by model role)

The caller passes a model role; the columns that apply are fixed in code, never read from the file.

| `model_role` | `raw_score` | `p` | `prediction` | Source |
|---|---|---|---|---|
| `baseline` (`baseline-v1`) | yes | not applicable | D29 forced binary: `authentic` / `synthetic` | D41, D29 (no calibration, no uncertain band) |
| `cnn_bundle_seed` | yes | yes (F4 policy) | `authentic` / `synthetic` / `uncertain` | D41, Protocol §7, D45 |
| `cnn_other_seed` (seeds 1–2) | yes | not applicable | not applicable | D45 (no risk policy) |

A not-applicable column is the empty string in every row. The writer refuses a value there (the
argument must be `None`) and refuses `None` for an applicable column; the reader refuses a
non-empty cell in a not-applicable column and an empty cell in an applicable one. The sidecar's
`applicable_columns` must equal the role's fixed list.

### Checks

Writer, before anything is written: purpose in `EVALUATION_PURPOSES` (D41), `run_id` 32 hex,
`model_artifact_sha256` and `manifest_sha256` 64 lowercase hex, `model_id` a plain file name,
known role; at least one clip; equal column lengths; `clip_id` canonical, safe and unique;
`label` in {FAKE, REAL}; `group_id` a non-empty trimmed string; `p` in [0, 1]; `prediction` in
the role's set; no line breaks in any cell. Existing files are never overwritten.

Reader: strict sidecar (exact key set, NaN refused, valid values, exact writer format); CSV
SHA-256 and row count equal to the sidecar's; no CR anywhere; header; the same row rules as the
writer; each float cell the exact `repr` spelling; finally the bytes must equal what the writer
would produce from the parsed rows (so unsorted rows or other quoting are refused).

### What is committed

As for Stage 10 (`val_b_scores.npz`, [risk_run.md](risk_run.md)): the real Stage 11 score files
stay on Drive. What is committed is each file's SHA-256 (`csv_sha256` and the sidecar's own hash)
and row count, in the Stage 11 run report and project-log entry, so a later reader can check that
it reads exactly the stored files. `read_clip_scores` returns the sidecar as `metadata`; the
caller compares it with the committed values and with the registry record of `run_id`.

## Why (team reasoning)

- **(a) clip_id = manifest path.** The F2 manifest path is already unique, validated and frozen;
  Stage 14 and Stage 16 join to the manifest and to each other by exact string. Only the canonical
  spelling is accepted, so `a\b` and `a/b` cannot be two IDs for one clip.
- **(b) CSV, not npz.** The files are small (one FoR test split per model), are read by Stages 14–16
  and by the team, and need no pickle-safe loader. Text with a fixed column order, a fixed float
  spelling and LF endings is reviewable in a diff and byte-deterministic, so its SHA-256 identifies
  the content.
- **(c) Exact floats.** `repr(float64)` is the shortest string that reads back to the same float,
  so stored scores give exactly the metrics computed from them; a fixed decimal format would round.
- **(d) Applicability in code.** D41 and D45 decide which outputs exist per model; storing them in
  code (not in the file) means a file cannot declare its own rules, and an empty cell always means
  "not applicable", never "missing".
- **(e) Same bytes on every OS.** Both files are written with `write_bytes` (no newline
  translation), as in [risk_run.md](risk_run.md) (d2); the reader refuses CR, so a file re-saved
  with CRLF is detected even if someone recomputes its hash.
- **(f) Purpose in the sidecar.** The sidecar names the D41 purpose and `run_id`, so a score file
  is tied to one registered run; files without a valid purpose are refused, like the registry.
- **(g) Limitation.** The files cannot prove that `run_id` is in the registry or that `prediction`
  agrees with `p` and the frozen threshold; the Stage 11 runner takes `run_id` from `claim` and
  computes predictions with the frozen policy, and its review checks that.

## Tests (`tests/evaluation/test_clip_scores.py`)

Synthetic rows and temporary folders only.

| Point | Test |
|---|---|
| (d) applicability table | `test_applicability_table_is_d41_d45` |
| Format and round trip | `test_round_trip_and_format` (header, LF, sorted rows, empty not-applicable cells, sidecar keys and hash; per role) |
| Determinism | `test_byte_identical_and_order_independent` (two writes and reversed input order → same bytes; per role) |
| (c) exact floats | `test_float_round_trip_exact` (float32-derived values, subnormal, extremes, −0.0, p = 0 and 1) |
| Floats refused | `test_bad_raw_score_refused` (NaN, ±inf, bool, string, None), `test_p_outside_unit_interval_refused` |
| (d) writer applicability | `test_writer_applicability` (value in a not-applicable column, missing applicable column, empty prediction) |
| Writer validation | `test_writer_validation` (D29 baseline never `uncertain`, bad labels, empty/non-string group, duplicate / empty / absolute / drive / `..` / non-canonical / multi-line `clip_id`, unequal lengths, no rows, unknown or case-variant purpose, bad hex, unsafe `model_id`, unknown role; nothing written), `test_never_overwrites` |
| Tampered CSV | `test_tampered_csv_byte_detected` (one byte → SHA-256 mismatch) |
| (d) reader applicability and cells | `test_reader_refuses_bad_cell` (with a re-signed sidecar, so the row checks must catch it: non-empty not-applicable cell, empty applicable cell, NaN, inf, non-repr float, p > 1, bad prediction / label / group / clip ID) |
| (e) CRLF | `test_reader_refuses_crlf` (re-signed CRLF file still refused) |
| Order, duplicates, truncation | `test_reader_refuses_unsorted_or_duplicate_rows`, `test_reader_refuses_header_count_and_truncation` |
| Sidecar | `test_reader_refuses_bad_sidecar` (extra key, unknown purpose, role/columns mismatch, bad hex, non-int `n_rows`, other `model_id`), `test_reader_refuses_missing_sidecar_key` (each key), `test_reader_refuses_non_canonical_sidecar` (CRLF, NaN, other formatting) |

## FoR test-split access guard (`tests/governance/test_for_test_access.py`)

Before the Stage 11 runner exists, no code in `src/`, `notebooks/` or `configs/` may ask for the
FoR test split. The guard parses Python files and notebook code cells with `ast` and YAML with
`yaml.safe_load`, and flags a `"test"` split passed as a keyword or call argument, compared with
`==` / `in` / `.isin()`, assigned to a name, joined as a path part (or a `test.npz/.json/.csv` file
name), a `split == 'test'` query string, a `--split test` CLI line, or a YAML value `test`. Refusal
code does not count: `!=` and `not in`, a comparison in the condition of an `if` whose body only
raises, valid-split sets and the audit's name map (literals that are not compared or passed),
docstrings, comments and YAML keys. Its allow-list is empty; the Stage 11 runner adds itself in its
own reviewed PR. It is a text guard, not a sandbox (ADR 0007): a split name built at run time, a
subscript such as `caches["test"]`, or code outside the three folders is not seen. The test file's
docstring lists the rules and limits in full.
