# Stage 10 risk-policy run evidence (`risk-v1`)

**Validation evidence (FoR val-A and val-B), not test results.** val-A is only re-scored to check
the checkpoint; the policy is fitted on val-B (Protocol v1.0 §6; D2, D9, D12). The FoR test split
is evaluated once in Stage 11 (D31).

The official Stage 10 run: notebook 05 on Colab, from commit
`11c07fedcd755dc72f16b2695f180cf094aeca04` (merge of PR #59, notebook 05), 2026-10-28; export
manifest written 20:26:08 UTC. Results and interpretation:
[results/risk_policy_fit.md](../../results/risk_policy_fit.md). Method:
[risk_run.md](../../risk_run.md), [risk_policy.md](../../risk_policy.md).

## What is here

Byte-identical copies of the files the notebook exported to Drive under
`MyDrive/voice-spoofing-detection/`:

| File | Drive path | Bytes | SHA-256 | Content |
|---|---|---|---|---|
| `risk_policy.json` | `checkpoints/risk/risk-v1/` | 358 | `a74f1f512b2e7c527df5f34e56acaf5fbe2ceb58071e3d46a441cc2f3f4966c1` | the fitted policy (`RiskPolicy.to_json()`): Platt `a`, `b`, threshold, `half_width`, target and achieved deferral; the file Stage 12 freezes into F4 |
| `run_report.json` | `processed/metadata/stage10/` | 2,062 | `4775fff54a7e38a4392c3d3a770bef5f0d989e66cf355e8c35946cc119c266fc` | provenance: commit, environment, F2/F3/cache hashes, the three provenance checks, `checkpoint_sha256`, recomputed val-A EER, hashes of the policy and the val-B scores, headline |
| `diagnostics_report.json` | `processed/metadata/stage10/` | 1,559 | `4846a9d1d420cdfe397622d5a2e6643b232e01152749d4b76832d68d715f9df0` | the pre-declared val-B diagnostics, descriptive only ([risk_run.md](../../risk_run.md) Why (e)) |
| `stage10_export_manifest.json` | `processed/metadata/stage10/` | 2,222 | `a8f5844be730401e18e7f12bfe3716d8929ae179f490a1e0ae024e8c28475a10` | SHA-256 and size of the 4 exported files, cache hashes of both builds, F3, FoR revision, pytest counts |

The manifest cannot list its own hash. The notebook checked it after the Drive remount against the
text it had just written (section 12) but did not print the hash, so the manifest SHA-256 above is
**observed only**: computed on the download, not recorded by the run.

`.gitattributes` in this folder (`* -text`) stops Git from converting line endings, so the
committed bytes stay the exported bytes (the JSON files are written with LF endings,
[risk_run.md](../../risk_run.md) Why (d2)).

## What stayed on Drive (not in Git, AGENTS.md rule 7)

| Path | Bytes | SHA-256 |
|---|---|---|
| `checkpoints/risk/risk-v1/val_b_scores.npz` | 390,301 | `43e5b198a5e0581ac0ee5c3e94d70a4b91e475bc3057b26751d7e88e795ffcd2` |

One line per val-B clip (paths, labels, raw score, calibrated `p`); it is data, so it stays on
Drive ([risk_run.md](../../risk_run.md) Why (g)). Its hash is from `stage10_export_manifest.json`
and `run_report.json`; it was not downloaded for this commit.

## Binding

- `run_report.json` `risk_policy_sha256` = SHA-256 of `risk_policy.json` (`a74f1f51…4966c1`).
- `run_report.json` `val_b_scores_sha256` = the manifest's hash of `val_b_scores.npz`
  (`43e5b198…5ffcd2`).
- The manifest's hashes and sizes of `risk_policy.json`, `run_report.json` and
  `diagnostics_report.json` = the committed files.
- `run_report.json` `checkpoint_sha256` (`c33eb820…958932`) = the bundle-seed (seed 0) value in
  [`stage8_cnn/run_report.json`](../stage8_cnn/run_report.json) (D45).

## Verification

1. **Run, section 12 (remount check).** After export, Drive was remounted and every file in the
   two export folders was hashed again against the manifest (the manifest against the text just
   written); a missing, extra or changed file would have stopped the notebook.
2. **Download.** Donia downloaded the four files and checked them locally with `certutil`.
3. **Copy into the repo.** The four files were copied byte for byte from the download; size and
   SHA-256 of each copy equal the table above (Python `hashlib`).
4. **After commit.** The SHA-256 of each stored blob (`git cat-file blob HEAD:<path>`, read as
   bytes) equals the table above.
5. **Binding.** The equalities in "Binding" were checked with a short Python script on the
   committed files.
