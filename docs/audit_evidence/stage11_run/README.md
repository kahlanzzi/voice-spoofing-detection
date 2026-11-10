# Stage 11 official run evidence (FoR test split, once)

**Test results: the one FoR test-split run (D31, D46).** Run `d5306a55a77d4e9c85d237d959e3dbae`,
model set `baseline-v1`, `cnn-v1-seed0` (bundle seed), `cnn-v1-seed1`, `cnn-v1-seed2`. Notebook 06
(`MODE = 'official'`) on a Colab T4 runtime, version 2026.10 (Python 3.13.16, torch 2.11.0+cu130,
CPU scoring), from commit `8219562b9dd97762a38d61171f48577798853dad` (merge of PR #115), 2026-11-10;
`completed` 16:21:50 UTC, export manifest written 16:27:37 UTC. Results:
[results/stage11_for_test.md](../../results/stage11_for_test.md). Method:
[stage11_run.md](../../stage11_run.md), [stage11_report.md](../../stage11_report.md). Log: Entry 161.

## What is here

Byte-identical copies of the files the notebook wrote to Drive under
`MyDrive/voice-spoofing-detection/processed/metadata/stage11/` (written as "`stage11/`" below):

| File | Drive path | Bytes | SHA-256 | Content |
|---|---|---|---|---|
| [`../run_registry.jsonl`](../run_registry.jsonl) | `stage11/run_registry.jsonl` | 834 | `b0133fc8d76f318a2d45001bcdf04ef41051b15927d1b13627bb1871af93b66d` | the live run registry (D31, D41, D46 (6)): 2 lines, `started` then `completed` |
| `stage11_export_manifest.json` | `stage11/run/` | 6,250 | `f3310f2f82b93ec06e4f68f0cef3a89c4a98f452d55a8920f3f317c2f0620308` | SHA-256 and size of the 18 exported files, registry SHA-256 and line count, caches, F3, MFCC reference, environment, pytest counts, `rehearsal_passed` |
| `metrics_report.json` | `stage11/run/` | 36,269 | `28b458020f35d7e411f30d369e9afe982a59ddb5f530ce854e5fd94cf89c79ae` | `stage11_report` output: 84 integrity checks, D4 metrics, CIs, H1, H5, seed rows |
| `claim.json` | `stage11/run/runner/` | 6,744 | `53a570b67395a3488a05b72c43da771a1b94a9a70866e94ef6e9bb7949777f27` | the claim: run identity, model set, pre-claim checks, `registry_sha256` after the claim |
| `environment.json` | `stage11/run/runner/` | 1,213 | `43c8e95b343ba145664a6e780ad7477151e66feb3ad8c89da7a03af1e5d9850f` | environment record at `claim` (D46 (2)) |
| `environment_score.json` | `stage11/run/runner/` | 1,213 | `43c8e95b343ba145664a6e780ad7477151e66feb3ad8c89da7a03af1e5d9850f` | environment record at `score`; identical bytes to the claim's |
| `pip_freeze.txt` | `stage11/run/runner/` | 14,390 | `2c5297373ef748794200976bcce96cd11530d976ae3bcea830bc1785ede06b9b` | `pip freeze` of the run (same bytes as `stage11/run/rehearsal/pip_freeze.txt`) |
| `scoring_started.json` | `stage11/run/runner/` | 95 | `9dfdb7c0e3ec21b44587cecb4ca6a6f8dc6d00ed93acebbd16a787be0d346d00` | the once-only scoring marker, 16:18:39 UTC (D46 (8)) |
| `run_report.json` | `stage11/run/runner/` | 8,784 | `cb2bfb9ec01e7b7a7868ff1eefa174c612dacf3232e87f419a3ad7e41db6099f` | score provenance: counts, pre-scoring checks, hashes of the claim, marker and 8 per-clip files |
| `rehearsal/rehearsal_report.json` | `stage11/run/rehearsal/` | 253,932 | `a0fde5fefc0f0496dda30d6c35080c71638a322321af52efc24df666a3b2f4fa` | the official run's val-A/val-B rehearsal before the claim, `passed` |
| `dry_run_20261110T114951Z_rehearsal_report.json` | `stage11/dry_run/20261110T114951Z/rehearsal/rehearsal_report.json` | 6,240 | `af12019cc632ca760914fd7320b4d95253b115f9aa43873d6f29b7391ae95460` | first dry run, failed: `failed_check` = `reproduction.mfcc.val-A.npz_sha256` (the D58 trigger, Entry 159) |
| `dry_run_20261110T150257Z_manifest.json` | `stage11/dry_run/20261110T150257Z/stage11_dry_run_manifest.json` | 2,910 | `7967f76edd65889d77b2218d0eec3c003da4cd1f93e525bdf72049db3f74c142` | second dry run (commit `ad69eddc…`), `rehearsal_passed` true |
| [`../../results/stage11_for_test.md`](../../results/stage11_for_test.md) | `stage11/run/metrics_report.md` | 6,856 | `cf0b8689fb12a67a893891706faae831d894f014e7249d198611c2b7cbbcb59b` | the Markdown report (`RESULTS_REF`) |

The export manifest cannot list its own hash: its SHA-256 above is **observed only**, computed on
the download. The two dry-run files are not in the export manifest; their hashes are observed only
too. The second dry run's `rehearsal_report.json` stays on Drive, bound by that manifest
(`746c84ed…a8b5412`, 253,932 bytes, equal to the download).

Line endings: `.gitattributes` here (`* -text`), `docs/results/.gitattributes`
(`stage11_for_test.md -text`) and the root `*.jsonl -text` stop Git from converting them, so the
committed bytes stay the exported bytes (all files are LF only).

## What stayed on Drive (not in Git, AGENTS.md rule 7)

`stage11/run/runner/clip_scores/`: one CSV (per-clip `clip_id`, `label`, `group_id`, raw score,
`p`, prediction) and one JSON sidecar per model, 2,658 rows each. Stages 14 and 16 read them there.

| File | Bytes | SHA-256 |
|---|---|---|
| `baseline-v1.csv` | 287,109 | `b950437b91d8864400f51b8bffd9acde519f6d535a920e0855e73a10b1d3b826` |
| `baseline-v1.json` | 543 | `56ffd0d4e333f1ef76d8bf5b25b6a957b8cfecbfd79720a5011b0c5ca8461e3e` |
| `cnn-v1-seed0.csv` | 338,652 | `2bc638e46100c577bb79166cd3fc99d1bc3220b166be67aaf3b1134d9d97d81c` |
| `cnn-v1-seed0.json` | 560 | `b67c3c83d19363b7cfbc4e841f48d783369b839e064c479c4f94709092805c36` |
| `cnn-v1-seed1.csv` | 262,238 | `165d299c4744b200978b8f8faccb9e8d2e456184350f4289608f473d488c6473` |
| `cnn-v1-seed1.json` | 532 | `4bc24e3c847c402d9777cdc335a54ffe967375cd0e4640722207f730223b1448` |
| `cnn-v1-seed2.csv` | 261,744 | `6b498c88012711d4cf5f41af65ae07f24b97908b320dcc1f4e1e90f47ac0470c` |
| `cnn-v1-seed2.json` | 532 | `7465a9de41c96c25bb5d852375493953416ec447f8c82f5cfd02a9e82fbeda8c` |

Bound by `run_report.json` `clip_scores.files` and the export manifest; downloaded for the checks
below and not committed.

## Binding

- Registry line 1: `started`, `prev_sha256` = `"0" * 64`; its bytes hash to
  `ebe02d028bfea061ce8e61c98b82685f74045b6f2c3de7f590d414d0c509c8b6` = `claim.json`
  `registry_sha256`. Line 2: `completed`, `prev_sha256` = that hash, `results_ref` =
  `docs/results/stage11_for_test.md`. Both name run `d5306a55…` and commit `8219562b…`, purpose
  `stage11_test`.
- Registry SHA-256 = the export manifest's `registry.sha256`, 2 lines = `registry.lines`.
- `run_report.json` `claim_sha256` = SHA-256 of `claim.json`; `scoring_marker_sha256` = SHA-256 of
  `scoring_started.json`; `clip_scores.files` = the 8 per-clip hashes above.
- Every file of the export manifest = the committed or Drive file (hash and size).

## Verification

1. **Run, step 15 (remount check).** After export, Drive was remounted and every file under `run/`
   and the registry were hashed again against the manifest (19 files).
2. **Download.** Donia downloaded `stage11/` (registry, `run/`, the two dry-run folders) to a folder
   outside the repo.
3. **Checks on the download** (Claude Code, local Python 3.14.7, project venv): every manifest file's
   SHA-256 and size (18 of 18 equal); the registry SHA-256, line count and `RunRegistry(path).records()`
   (hash chain and replayed rules); the bindings above; the dry-run fields (`failed_check`,
   `rehearsal_passed`, `commit`).
4. **Reproduction (stored files only, D31).** `python -m src.evaluation.stage11_report --run-dir
   <download>/run/runner --registry <download>/run_registry.jsonl --manifest
   data/manifests/for2sec_frozen_manifest.csv --risk-policy
   docs/audit_evidence/stage10_risk/risk_policy.json --stage7-evidence
   docs/audit_evidence/stage7_baseline --stage10-evidence docs/audit_evidence/stage10_risk --out
   <new folder outside the repo>`:
   - on the run's platform (Colab T4, runtime 2026.10, Python 3.13.16, commit `8219562`, Donia):
     `metrics_report.json` and `.md` byte-identical to the stored ones;
   - on local Windows (Python 3.14.7, numpy 2.5.3): stopped at `rederive.bundle.p_bytes`; 69 of
     2,658 recomputed `p` differ from the stored ones by at most 1 ulp (max 1.4e-17), no decision
     changes [hypothesis: `np.exp` differs across platforms, as in D58]. Entry 161.
5. **Copy into the repo.** Byte copies (`cp`); SHA-256 of each copy equal to the source.
6. **After commit.** The SHA-256 of each stored blob (`git cat-file blob HEAD:<path> | sha256sum`)
   equals the table above.
