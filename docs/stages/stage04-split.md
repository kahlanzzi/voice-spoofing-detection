# Stage 4 — Leakage-aware FoR-2sec split

| | |
|---|---|
| **Stage** | 4 — Leakage-aware FoR-2sec split |
| **Owner** | Malak Ben Salem (D26, Part B) |
| **Reviewer(s)** | Donia Mabrouk (consumer; D26) |
| **Status** | Ready for review — retrospective brief; teach-back pending |
| **Date** | 2026-10-19 |
| **Related PRs** | #12 (`feat/stage4-split`; merged as `865faf4`, Entry 28) |

## 1. In one paragraph

Stage 4 assigns every audited FoR-2sec recording to a training, validation, or test partition in a
way that limits leakage between near-duplicate recordings. It starts from Stage 1's pinned
metadata inventory, not from audio, and makes the split before any Stage 5 manifest is created.
The assignment preserves the approved REAL / FAKE_mp3 / FAKE_other proportions, keeps each
D14 near-duplicate group intact, and reserves separate validation views for model choices and
threshold decisions. Its output is a candidate assignment plus checks and a report; Stage 5 then
validates that assignment and freezes it as part of F2. The full implementation is in
[`src/splitting/for2sec.py`](../../src/splitting/for2sec.py), with the official evidence in
[`docs/audit_evidence/stage4_split/`](../audit_evidence/stage4_split/).

## 2. Inputs → outputs

| | What | From / to | Path |
|---|---|---|---|
| In | Pinned FoR inventory and audited summary (metadata only) | Stage 1 (Donia) | Supplied as `--inventory` and `--summary`; pin in `configs/split.yaml` |
| In | Frozen split rules: seed, near-duplicate group, ratios, tolerances | Protocol v1.0 / team decisions | `configs/split.yaml`; Protocol §2 |
| Out | 17,721-row split assignment candidate (kept outside Git) | Stage 5 (Malak) | Drive `processed/metadata/stage4/for2sec_split_assignments.csv` |
| Out | Check report and committed summary evidence | Stage 5, reviewers | `docs/audit_evidence/stage4_split/for2sec_split_report.md` and `.json` |

This is the Part B split step. The later B→C hand-over is the frozen manifest contract recorded in
D26; Stage 4 alone does not produce those manifests.

## 3. How it works, step by step

1. Load the Stage 1 FoR inventory and summary and verify their expected pin, counts, labels, and
   subgroup totals against `configs/split.yaml`.
2. Use `neardup_group_095` as the indivisible grouping unit, as required by D14. No group may be
   assigned across two partitions.
3. Allocate groups deterministically with seed `2026`, stratifying by REAL, FAKE_mp3, and
   FAKE_other (D16).
4. Use the frozen ratios: 70% train, 7.5% val-A, 7.5% val-B, and 15% test. Together val-A and
   val-B form the 15% validation portion, split evenly as required by D9.
5. Check row count, unique and complete paths, valid partition names, group isolation, and class
   and subgroup proportions within one percentage point in every partition.
6. Write the candidate CSV and JSON/Markdown reports outside the frozen-manifest directory.
   Keeping assignment separate from F2 prevents Stage 4 from silently freezing Stage 5 inputs.
7. Regenerate from the final clean implementation commit and compare row count and SHA-256. Upload
   the candidate to its recorded Drive destination; commit only the small reports, not the data.
8. Donia reviewed the evidence; PR #12 merged to `main`, closing Stage 4 (Entry 28).

Details and edge cases are in the [split implementation](../../src/splitting/for2sec.py),
[configuration](../../configs/split.yaml), and [official report](../audit_evidence/stage4_split/for2sec_split_report.md).

## 4. Key decisions and why

- **D13** — re-split FoR instead of trusting a packaged split: the project needs one controlled
  protocol-defined partition.
- **D14** — keep 0.95 near-duplicate components together: the dataset has no speaker IDs, so
  group separation is the available leakage control.
- **D16** — stratify by REAL / FAKE_mp3 / FAKE_other: balancing only REAL/FAKE could hide subgroup
  imbalance relevant to the shortcut hypotheses.
- **D9** — reserve val-A and val-B separately: model choices use val-A; threshold and calibration
  decisions use val-B.
- **D26** — Stage 4 belongs to Malak; its consumer/reviewer is Donia, and the later frozen
  manifests are handed to Stage 7 through the defined B→C interface.

## 5. Evidence

- [verified] The pinned source contains **17,721** unique recordings: 8,800 REAL and 8,921 FAKE; the FAKE
  groups are 7,592 FAKE_mp3 and 1,329 FAKE_other — source: `configs/split.yaml` and
  `docs/audit_evidence/stage4_split/for2sec_split_report.md`.
- [verified] Final split counts are **12,404 train, 1,330 val-A, 1,329 val-B, and 2,658 test** — source:
  `docs/audit_evidence/stage4_split/for2sec_split_report.md`.
- [verified] All reported acceptance checks pass, including path completeness, group isolation, and class /
  subgroup balance; the maximum deviation is **0.034537 percentage points** — source: same report.
- [verified] The candidate assignment has **17,721 rows** and SHA-256
  `853d947499e76c4cc9da5b69df384588f0023c21ed93f6a2dec1940f8988aa20` before and after rerun —
  source: same report; clean code commit `10a5e6809bb55484908cbcce547a75c14d33a908`.
- [verified] The report records **18 focused split tests** and **33 tests in the complete suite** passing —
  source: same report. The cross-split-group test deliberately creates a violation and confirms
  that the checker rejects it.
- [verified] Donia reviewed and PR #12 merged as `865faf4`; the candidate CSV remains outside Git at the
  recorded Drive destination — source: `docs/project_log.md`, Entry 28.

## 6. Limitations

- [verified] The split is metadata-driven. Stage 4 does not listen to, decode, or independently re-audit the
  recordings; it depends on the Stage 1 inventory and the hashes represented there.
- [verified limitation] Near-duplicate grouping is a proxy because speaker identity metadata is unavailable. It reduces
  the identified leakage risk but cannot prove that every same-speaker recording is isolated.
- [doc limitation] Stratification and the ±1 percentage-point checks control the listed classes and subgroups only;
  they do not make the partitions representative of every recording source or future traffic.
- [verified limitation] The assignment CSV is an operational data artifact in shared Drive, not a tracked repository
  file. The committed report and its hash support review, but are not a substitute for access to
  the pinned source inventory when reproducing the full run.
- [doc limitation] Stage 4 establishes assignments only. It makes no model-performance, spoof-detection, or
  external-generalization claim.

## 7. Questions the professor may ask

1. **Q:** Why not use the dataset's packaged split?
   **A:** D13 selects a project-controlled re-split so every later result uses the same registered
   partition rules.
2. **Q:** Why group at the 0.95 threshold?
   **A:** Speaker IDs are unavailable; D14 uses near-duplicate components as a conservative
   leakage-control proxy and keeps each component intact.
3. **Q:** Why are there two validation sets?
   **A:** Val-A is for training choices and val-B for threshold/calibration decisions (D9), reducing
   reuse of the same validation examples for both purposes.
4. **Q:** What does stratifying over three groups protect against?
   **A:** It keeps the REAL, FAKE_mp3, and FAKE_other composition close across partitions (D16),
   rather than hiding subgroup imbalance inside the broad FAKE label.
5. **Q:** How do you know the split is reproducible?
   **A:** The seed and rules are pinned, and clean reruns produced the same 17,721-row CSV SHA-256.
6. **Q:** Does this stage create the frozen training and test manifests?
   **A:** No. It emits assignments and reports only; Stage 5 validates and freezes the manifests as
   F2.

## 8. How to verify or re-run it

Use the clean Stage 4 implementation commit and the same pinned Stage 1 metadata inputs. The
candidate output directory should be outside `data/manifests/`; the full inventory is an approved
project input and should not be added to Git.

```bash
python -m pytest -q tests/splitting/test_for2sec.py
python -m src.splitting.for2sec \
  --inventory <for2sec_inventory.csv> \
  --summary <for2sec_summary.json> \
  --config configs/split.yaml \
  --out <stage4_candidate_output>
```

The command prints the accepted split sizes and writes a candidate assignment plus reports. The
official run's expected sizes are `12404 / 1330 / 1329 / 2658`; the clean rerun should reproduce
SHA-256 `853d9474…8988aa20`. Evidence and acceptance history: Entry 25 and Entry 28.

