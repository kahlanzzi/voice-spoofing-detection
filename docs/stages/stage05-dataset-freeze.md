# Stage 5 — Dataset freeze (F2)

| | |
|---|---|
| **Stage** | 5 — Dataset freeze (F2) |
| **Owner** | Malak Ben Salem (D26, Part B) |
| **Reviewer(s)** | Donia Mabrouk (consumer; D26) |
| **Status** | Ready for review — retrospective brief; teach-back pending |
| **Date** | 2026-10-19 |
| **Related PRs** | #15 (F2 implementation and evidence; merged as `89b37fc`) · #22 (administrative closeout; merged as `02f53e2`) |

## 1. In one paragraph

Stage 5 turns the approved Stage 4 split into the immutable dataset manifests used by later
experiments. It joins the split assignments to the pinned FoR metadata and records a separate
file-level external-test manifest using only the metadata access allowed by D20. For each manifest,
it validates the expected fields, labels, paths, group/split assignments, and hashes, then writes a
hash ledger and human-readable reports. Two clean runs from the same implementation commit
produced byte-identical outputs. After Donia reviewed and PR #15 merged, D28 became approved and
F2 effective; subsequent changes to frozen data identity or split fields require a new decision.
The full implementation is in [`src/freezing/manifests.py`](../../src/freezing/manifests.py), and
the official evidence is in [`docs/audit_evidence/stage5_freeze/`](../audit_evidence/stage5_freeze/).

## 2. Inputs → outputs

| | What | From / to | Path |
|---|---|---|---|
| In | Pinned FoR inventory and approved split assignments | Stage 1 (Donia) + Stage 4 (Malak) | `--for-inventory`, `--assignments`; pins in `configs/freeze.yaml` |
| In | External file-level metadata inventory (metadata only) | Stage 1 audit, constrained by D20 | `--deepvoice-inventory`; pin in `configs/freeze.yaml` |
| In | Frozen F2 rules and expected counts / hashes | Team decision register | `configs/freeze.yaml`, approved D28 |
| Out | FoR frozen manifest (17,721 rows) | Stage 6 and onward | `data/manifests/for2sec_frozen_manifest.csv` |
| Out | External metadata-only manifest (64 rows) | Later guarded external evaluation | `data/manifests/deepvoice_frozen_manifest.csv` |
| Out | Manifest hash ledger and run reports | All consumers / reviewers | `data/manifests/manifest_hashes.csv`; `docs/audit_evidence/stage5_freeze/` |

The FoR manifest carries path, label, subgroup, group, split, and sample hash fields required by
the D26 B→C contract. The external manifest is metadata-only; this stage does not authorize the
later external evaluation.

## 3. How it works, step by step

1. Read the pinned FoR inventory, the approved Stage 4 assignments, and the allowed external
   metadata inventory as CSV metadata. The freezer does not open or decode audio.
2. Verify the configured FoR revision, Stage 1 inventory hash, Stage 4 assignment hash, external
   inventory and archive pins, row totals, and required columns.
3. Join each FoR inventory row to exactly one assignment. Refuse missing or duplicate paths,
   inconsistent labels, invalid groups/splits, or any mismatch with the frozen counts.
4. Build the FoR manifest with the stable identifiers and split information required downstream;
   build the external manifest from its permitted file-level metadata fields only.
5. Validate each manifest: row count, path uniqueness, labels, headers, positive metadata fields,
   valid speaker metadata, sample hashes, FoR counts matching Stage 4, and no group crossing
   partitions.
6. Write both manifests, a CSV hash ledger, and JSON/Markdown reports. Compare the outputs from two
   clean runs byte-for-byte; no dataset audio or generated model features are output here.
7. Record the proposed F2 hashes in D28. Donia reviewed the evidence; PR #15 merged, approving D28
   and making F2 effective (Entry 36). PR #22 later recorded administrative closeout.

See [the freezer](../../src/freezing/manifests.py), [F2 config](../../configs/freeze.yaml),
[official report](../audit_evidence/stage5_freeze/freeze_report.md), and
[determinism check](../audit_evidence/stage5_freeze/determinism_check.json) for implementation and
field-level details.

## 4. Key decisions and why

- **D20** — the external set is represented here by permitted file-level metadata only; no audio is
  decoded or inspected before the one guarded run.
- **D21** — pin the dataset revisions and archive identity: a manifest is meaningful only for the
  exact audited inputs from which it was built.
- **D26** — Stage 5 is Malak's Part B hand-over; its consumer checks the manifests before use.
- **D28** — freeze exact manifest and ledger hashes; after approval, changing a pin, path, label,
  subgroup, group, split, or sample hash requires a new decision.
- **D32** — the stage owner writes a teach-back brief so the next consumer can explain and re-run
  the freeze.

## 5. Evidence

- [verified] FoR manifest: **17,721 rows**, SHA-256
  `badfe948fb7b833357783dd18be00f6bb36253cf8699dc4c4beb4df2b07706f8` — source:
  `docs/audit_evidence/stage5_freeze/freeze_report.md` and approved D28.
- [verified] External manifest: **64 metadata rows**, SHA-256
  `7c7a3e52ef254841f53bc08018bd3f9fe7ad83d6cd84531832513c616695355f` — source: same report and
  D28. No audio was decoded or inspected.
- [verified] Hash ledger SHA-256:
  `c9fc2e230c0883028459d0b2b8de4f6ee537676cd5a4e259e9d0f545c60943cc` — source: same report.
- [verified] The FoR and external manifest checks all pass; the report records **14 validation checks** —
  source: `freeze_report.json` and `freeze_report.md`.
- [verified] **Two clean runs** from final code commit `df1d6acab9df707b4702e038c45d5d22b38b8e66` produced
  all five outputs byte-identically — source: `determinism_check.json`.
- [verified] The Stage 5 suite passed **12 tests**, and the full suite passed **45 tests** at implementation
  review — source: `docs/project_log.md`, Entries 29–30.
- [verified] Donia approved PR #15; merge `89b37fc` made D28 approved and F2 effective. PR #22 recorded the
  administrative closeout — source: Entry 36.

## 6. Limitations

- [verified limitation] The external manifest records metadata only. It provides no evidence about audio quality,
  preprocessing, model performance, or generalization; only the separately approved guarded run
  may evaluate those questions.
- [verified limitation] The freezer validates supplied inventories and hashes. It does not independently reconstruct the
  Stage 1 audit or re-open audio to verify those metadata claims.
- [doc limitation] Determinism means the recorded outputs match for the same clean implementation, inputs, config,
  and environment. It does not guarantee that changed inputs or future code produce the same bytes.
- [doc limitation] A valid hash proves identity of a file, not that its labels or provenance are scientifically
  correct; those depend on the upstream audit and approved decisions.
- [doc limitation] The manifests freeze dataset membership and split metadata. They do not freeze a model,
  preprocessing profile, threshold, or a right to run the external evaluation.

## 7. Questions the professor may ask

1. **Q:** What does F2 freeze?
   **A:** The exact FoR split manifest, the file-level external metadata manifest, and their hash
   ledger, with counts and hashes recorded in approved D28.
2. **Q:** Why does Stage 5 receive the Stage 4 assignment file?
   **A:** The freezer joins each pinned FoR inventory record to its already-approved partition; it
   must not invent or change the split during freezing.
3. **Q:** Did the external audio get opened during F2?
   **A:** No. The external output is a 64-row metadata manifest, and the official report records no
   audio decoding or inspection. D20 still governs the later single-use evaluation.
4. **Q:** What does byte-identical determinism establish?
   **A:** With the same clean code, inputs and config, all five outputs reproduced exactly across
   two runs. It does not establish the validity of upstream labels or future environments.
5. **Q:** Can someone edit a manifest after approval?
   **A:** Not silently. D28 requires a new decision for changes to a pin, path, label, subgroup,
   group, split, or sample hash.
6. **Q:** What is handed to Stage 6 and then Stage 7?
   **A:** The frozen FoR manifest identifies each sample and its split. Stage 6 consumes that frozen
   identity for preprocessing validation; the B→C contract later supplies the manifests and
   preprocessing I/O to Stage 7.

## 8. How to verify or re-run it

Use the clean Stage 5 implementation commit and the same pinned metadata CSVs and config. The
external input is the approved metadata inventory only. Do not provide an audio directory or add
dataset content to Git.

```bash
python -m pytest -q tests/freezing/test_manifests.py
python -m src.freezing.manifests \
  --for-inventory <for2sec_inventory.csv> \
  --assignments <for2sec_split_assignments.csv> \
  --deepvoice-inventory <approved_external_metadata.csv> \
  --config configs/freeze.yaml \
  --manifest-dir <stage5_manifest_out> \
  --report-dir <stage5_report_out>
```

Expected: **17,721** FoR rows, **64** external metadata rows, all 14 checks passing, and the five
outputs matching the hashes in the evidence files. Use clean temporary output locations when
independently verifying; do not overwrite the committed F2 manifests or evidence.
F2 was accepted in PR #15 (Entry 36); these instructions do not authorize a changed freeze.

