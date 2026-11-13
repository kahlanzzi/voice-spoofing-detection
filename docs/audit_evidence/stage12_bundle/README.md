# Stage 12 F4 evidence (`bundle-v1`, ModelBundle v1)

**Freeze record, not a result.** F4 is the frozen v1 ModelBundle (Protocol v1.0 §12; D59 layout;
D60 freeze). No dataset, cache or audio was read to build it, and nothing was scored.

The F4 build: `python -m src.bundle.f4 build` (Stage 12 tool, PR #124), locally on Windows
(Python 3.14.7, torch 2.11.0+cpu), from the clean merge commit
`ea5544a768876001ee66c8a7446b688c042bdf72`, 2026-11-12T22:40:45Z. Method:
[model_bundle.md](../../model_bundle.md) ("Real build (F4)").

## What is here

| File | Bytes | SHA-256 | Content |
|---|---|---|---|
| `bundle_manifest.json` | 1,027 | `edf4a205e12fc948a68575ebaa929145481aaa535d34e3353ef9535f92840efc` | the manifest inside the bundle, byte for byte: **this hash is F4's identity** (`manifest_sha256`, D60) |
| `f4_record.json` | 11,701 | `b8f0f5399271083958f828377232e2b36d4a2f70427c50e3c2d2e166dc5355b2` | the F4 record written by the build: every component hash and version, `baseline-v1` hash, environment, 29 checks with their sources (all `ok`), SHA-256 of each evidence file read |

## What stays on Drive (not in Git, AGENTS.md rule 7)

`MyDrive/voice-spoofing-detection/checkpoints/bundle/bundle-v1/`:

| Path in the bundle | SHA-256 | Source |
|---|---|---|
| `bundle_manifest.json` | `edf4a205…40efc` | written by the build |
| `checkpoint/checkpoint.json` | `dc0f8b0e9a8549583478db7d1a94e24d01668af9459bfd692895abcc83c98932` (1,350 B) | Drive `checkpoints/cnn/cnn-v1/seed_0/` (Stage 8, bundle seed 0) |
| `checkpoint/weights.safetensors` | `6722f8587b54aaea649456a04ec3074fb30883d549bf48ea6a6ffa9111dcf835` (396,084 B) | same |
| `preprocessing.yaml` | `7bd35bfdcb9270dd55ea9e73449c7bf6c6a359a80335b2551c0c89dde77c8630` | `configs/preprocessing.yaml` (F3) |
| `policy.json` | `a74f1f512b2e7c527df5f34e56acaf5fbe2ceb58071e3d46a441cc2f3f4966c1` | `docs/audit_evidence/stage10_risk/risk_policy.json` (`risk-v1`) |

Combined checkpoint hash (D42 rule) `c33eb82086d3bc40f3dedf1356addc6e5c1924ca2e517b0d4527b447e1958932`
= the Stage 8 bundle-seed value and the value in the Stage 10 run report. Recorded beside F4, not in
it (D36, D7): `baseline-v1` `artifact_sha256`
`42cc4d53a7cd881f56e4fa54fcf4797de8885b404d0a2b31c293555c868df10a`.

## How it was checked

- [verified] Build: `F4 build OK`; 29/29 checks in `f4_record.json` (sources cross-checked, both
  downloaded checkpoint files by size and SHA-256 against `stage8_export_manifest.json`, every
  manifest value, `load_bundle_model`).
- [verified] Drive copy: uploaded to the path above, downloaded again as a zip and extracted;
  `python -m src.bundle.f4 verify --bundle <copy>\bundle-v1 --record f4_record.json` →
  `manifest_sha256: edf4a205…40efc`, `F4 verify OK`.
- [verified] Review (Linux, independent of the build): the manifest above is **reproduced byte for
  byte** from the committed evidence at `ea5544a` plus `code_commit` and `created_at`; the 7
  `evidence_sha256` values equal the files at `ea5544a`; no local path and no CRLF in either file;
  the record is canonical JSON (sorted keys, 2-space indent, trailing newline).

## How to use F4 (Iheb: API, P1; Khalil: Stage 13)

1. Download `checkpoints/bundle/bundle-v1/` from Drive (outside the repository).
2. `python -m src.bundle.f4 verify --bundle <your copy>\bundle-v1 --record docs/audit_evidence/stage12_bundle/f4_record.json`
   → must print `manifest_sha256: edf4a205…40efc` and `F4 verify OK`. Anything else: do not use it.
3. Load it only with `src.bundle.model_loader.load_bundle_model`; use `LoadedBundle.profile`, never
   re-load the profile from `profile_path` ([model_bundle.md](../../model_bundle.md), Limitations).
