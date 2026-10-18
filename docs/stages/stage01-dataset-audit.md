# Stage 1 — Dataset audit

| | |
|---|---|
| **Stage** | 1 — Dataset audit (FoR-2sec full, DEEP-VOICE metadata-only) |
| **Owner** | Donia Mabrouk (D26, Part A) |
| **Reviewer(s)** | Malak Ben Salem (her Stage 4 split builds on the audit; D26) |
| **Status** | Accepted (PR #25) |
| **Date** | 2026-10-18 |
| **Related PRs** | #2 (`stage-1-audit-run`, approved and merged by Malak, 2026-10-05; Entry 14) |

## 1. In one paragraph

Before any model is trained, we must know what is really inside the two datasets. Stage 1 checks
FoR-2sec (the training data) file by file: formats, durations, duplicates, near-duplicates, hidden
shortcuts that could let a model "cheat", and whether the package has a usable split. It checks
DEEP-VOICE (the external test set) by metadata only, without listening to it or computing any
feature, so it stays a clean, unseen test. The decision rules were written and approved before the
audit ran, so the results could not bend them. The outputs are the facts and decisions that the
protocol (Stage 3) and the split (Stage 4) build on. The full report is
[`docs/dataset_audit.md`](../dataset_audit.md); this brief explains it.

## 2. Inputs → outputs

| | What | From / to | Path |
|---|---|---|---|
| In | Pinned FoR-2sec (HF revision `ff8c82c7…9055`) and DEEP-VOICE zip (SHA-256 `8cb25853…bdcf`) | Shared Drive `raw/` (D21) | `configs/audit.yaml` |
| In | Pre-registered decision rules | Team, approved 2026-10-04 before any output | `docs/dataset_audit.md` §0 |
| In | Audit tools and runbook | Stage 0 repo | `src/audit/`, `notebooks/01_stage1_dataset_audit.ipynb` |
| Out | Audit report: facts, decisions, limitations | Stage 2 threat model, Stage 3 protocol (Donia) | `docs/dataset_audit.md` |
| Out | Committed evidence summaries | Reviewers, the final report | `docs/audit_evidence/` |
| Out | Full FoR inventory and summary (per-file, pairs, fingerprints) | Stage 4 split (Malak) | Drive `processed/metadata/stage1/for2sec` |
| Out | Decisions D5, D9, D13, D14 settled; D16–D20, H5, H6 added; the pins were set in Stage 1 (Entry 7) and recorded as D21 in Stage 2 (Entry 11) | Every later stage | `docs/decisions/README.md` |

## 3. How it works, step by step

1. **Rules first.** The §0 rules (split, grouping, ratios, evaluation unit, shortcut flags, corrupt
   files, duplicates) were approved before any audit output existed; D14 was refined the same way
   (Entry 3; `docs/dataset_audit.md` §0 and §9).
2. **Download at a pinned version.** FoR-2sec at the exact HF revision, backed up as one `.tar` in
   Drive; the DEEP-VOICE zip is hashed and the notebook stops if the hash differs (Entries 4, 7).
3. **Dry run** on 500 FoR files as a smoke test and time estimate (notebook cell 4).
4. **Full FoR audit** (17,721 clips, 3.3 min): headers, decoding, durations, exact duplicates (bytes
   and PCM), near-duplicates (log-Mel fingerprint similarity at 0.90 / 0.95 / 0.98), shortcut
   screens (Entry 4; `docs/dataset_audit.md` §2–§4).
5. **Codec check:** FAKE split into FAKE_mp3 and FAKE_other, each compared with REAL (§3).
6. **DEEP-VOICE metadata-only audit:** headers, byte hashes, filenames and CSV schema; no audio
   decoded (§5, blind-audit attestation).
7. **Overlap check:** FoR byte hashes vs DEEP-VOICE byte hashes (§4).
8. **Apply the §0 rules** to the results → decisions (§7) and limitations (§8).
9. **Archive** all outputs to Drive; commit the summaries to `docs/audit_evidence/` (Entry 8).

## 4. Key decisions and why

- **D13** — re-split FoR: the package has no train/val/test folders, so §0 condition (a) fails.
- **D14** — group by near-duplicate components at 0.95: no speaker metadata; 0.95 passes both checks.
- **D9** — val-A / val-B applies: val-B ≈ 665 clips per class ≥ 500.
- **D5** — 2 s windows confirmed: DEEP-VOICE files are 79–600 s.
- **D16** — no data changes for the MP3 imbalance; stratify the split by REAL / FAKE_mp3 / FAKE_other; record a limitation.
- **D6, D20** — DEEP-VOICE audit is metadata-only, with a closed list of accesses: keeps the external test unseen.
- **D17, D18, D19** — speech-window filter, mono + 16 kHz everywhere, per-speaker table: fix how the one external run handles DEEP-VOICE's format, before anything is scored.
- **D21** — both datasets pinned: the audit is valid only for the exact recorded data.

## 5. Evidence

- 17,721 FoR clips: 8,800 REAL / 8,921 FAKE; all `.wav`, PCM_16, 16 kHz mono, exactly 2.0 s; 0
  header or decode failures [verified] — source: `docs/audit_evidence/for2sec_summary.md` §1–§3
  (audit commit `ae45e9f`).
- 0 exact duplicates (bytes and PCM); near-duplicate pairs 181 / 98 / 91 at 0.90 / 0.95 / 0.98, 0
  across labels; largest group at 0.95 = 6 clips (0.0339%) [verified] — source:
  `for2sec_summary.md` §7–§8.
- MP3 signature on 7,592 FAKE clips (85.1% of FAKE), 0 REAL; highest numeric shortcut `rms_dbfs`
  AUC 0.698; FAKE_other silence separability 0.750 [verified] — source: `docs/dataset_audit.md` §2–§3,
  `for2sec_codec_check.csv`.
- DEEP-VOICE: 64 WAV (8 REAL / 56 FAKE), all stereo, 40 / 44.1 / 48 kHz, 79.49–600.4 s; 2 s windows
  1,870 REAL / 13,090 FAKE (87.5% FAKE); 8 speakers [verified] — source: `deepvoice_summary.md`,
  `docs/dataset_audit.md` §5.
- 0 identical files between FoR (17,721 hashes) and DEEP-VOICE (64 hashes) [verified] — source:
  `docs/audit_evidence/cross_dataset_overlap.json` (commit `c2136cb`).
- The ported notebook re-ran in Colab at `c2136cb` with results identical to the original run;
  12 tests passed [verified] — source: Entry 8.
- Sign-off: Malak approved via PR #2 (2026-10-05); Iheb not yet reviewed [doc] — source:
  `docs/dataset_audit.md` §10.

## 6. Limitations

From `docs/dataset_audit.md` §8 (full wording there):

- **No speaker metadata in FoR:** the same speaker can be in train and test, so FoR scores are an
  upper bound [verified absence; effect is a hypothesis].
- **Codec-history imbalance** (85% FAKE vs 0% REAL with MP3 history) cannot be removed without
  changing the data; tested by H5 [verified].
- **Fingerprints miss time-shifted overlaps** of the same recording: residual leakage risk [verified
  as a method property].
- **DEEP-VOICE is small and skewed:** 8 source recordings (wide CIs), 87.5% FAKE windows [verified].
- **FoR → DEEP-VOICE differences are confounded** (method, speakers, conditions, format change
  together): a gap cannot be blamed on TTS vs RVC alone (D8).
- The Drive-archive-only read path of cells 6 and 8 was tested on synthetic data only (Entry 8).
  Generators in FoR are not determinable from the package (§2).

## 7. Questions the professor may ask

1. **Q:** Why did you re-split FoR instead of using its original split?
   **A:** The pinned package has only `FAKE/` and `REAL/` folders, no train/val/test, so the
   pre-registered rule's condition (a) fails and the rule says re-split (D13; `docs/dataset_audit.md`
   §1, §7).
2. **Q:** Without speaker metadata, how do you stop the same recording appearing in train and test,
   and what does that limitation mean for your results?
   **A:** Exact (byte and PCM) duplicates and near-duplicate groups at similarity 0.95 are kept inside
   one split (D14, §4). Speakers can still cross splits, so FoR results are an upper bound (§8 item 1;
   threat model [§8](../threat_model.md#8-threats--evidence-what-the-experiments-can-support) T1).
3. **Q:** 85% of fake clips had MP3 history and 0% of real ones did: how will you know the model
   isn't just detecting MP3?
   **A:** The split is stratified by REAL / FAKE_mp3 / FAKE_other (D16), and H5 compares recall on
   FAKE_other vs FAKE_mp3 on the FoR test: a clearly lower FAKE_other recall points to MP3 reliance
   (Stages 11/16; threat model §8 T5).
4. **Q:** Fake clips have less high-frequency energy: why is that a risk for genuine phone callers?
   **A:** Median 99%-energy bandwidth is 5,074.5 Hz for REAL vs 3,264 Hz for FAKE (§3.2). Phone lines
   are narrowband, so a model that learned "narrow band = fake" would flag genuine callers
   [hypothesis]; H6 tests it in Stage 14 (threat model §8 T4).
5. **Q:** Why was DEEP-VOICE only audited by metadata?
   **A:** It is the external test, used once. Listening, spectrograms or feature statistics would let
   its content shape our choices and invalidate the test (D6, D20, hard rule 1, ADR 0005).
6. **Q:** Why were the decision rules written before running the audit?
   **A:** So the results cannot shape the rules that interpret them; changing a rule after seeing data
   needs change control and a §9 note (`docs/dataset_audit.md` §0; Entry 3).
7. **Q:** DEEP-VOICE is 87.5% fake: how do you report results honestly with that imbalance?
   **A:** Never accuracy alone (hard rule 5). D4 reports EER, ROC-AUC, precision/recall/F1, FPR/FNR,
   class counts and 95% CIs; D19 adds a per-speaker table, and CIs are clustered by source speaker
   (D11; §8 item 4).
8. **Q:** The proposal says DEEP-VOICE has 628 / 4,425 samples but your audit does not: who is right?
   **A:** For the pinned package, the audit: those counts match none of files (8 / 56), 2 s windows
   (1,870 / 13,090) or CSV rows (5,889 / 5,889) [verified] (§6). Where 628 / 4,425 came from is not
   recorded; correcting the Proposal and Execution Plan is still an open item.

## 8. How to verify or re-run it

Without data (local, any time): `python -m pytest -q tests/audit tests/governance` → `15 passed`
on this branch (Entry 38). Full re-run in Colab with `notebooks/01_stage1_dataset_audit.ipynb`
(Python 3.13, Drive access, HF access, a Kaggle API token typed in at the prompt, never saved). The
notebook runs, in order (paths are the `configs/audit.yaml` values):

```bash
# cell 1: clone at BRANCH, print Python + commit, install, run tests (stops if any fails)
python -m pip install -q -r requirements.txt && python -m pytest -q
# cell 2: download FoR at the pinned revision (stops if for2sec.revision is empty), back up to Drive
# cell 3: download the DEEP-VOICE zip, stop if its SHA-256 differs from deepvoice.zip_sha256, unzip, list only
# cell 4: dry run (first 500 files, all FAKE: smoke test and timing only)
python -m src.audit.for2sec_audit --root <for2sec.root> --out <for2sec.out>_dryrun \
    --revision ff8c82c79e7bbefef811941bac9775c9328e9055 --limit 500
# cell 5: full FoR audit
python -m src.audit.for2sec_audit --root <for2sec.root> --out <for2sec.out> \
    --revision ff8c82c79e7bbefef811941bac9775c9328e9055
# cell 6: codec check · cell 7: metadata-only audit on <deepvoice.root>/<deepvoice.audit_subdir>
python -m src.audit.deepvoice_metadata_audit --root <deepvoice.root>/<deepvoice.audit_subdir> \
    --out <deepvoice.out> --revision "<zip-sha256 label built by cell 7>"
# cell 8: FoR vs DEEP-VOICE byte-hash overlap · cell 9: archive outputs to Drive
```

Expected: `for2sec_summary.md` equal to the committed one (17,721 files; "D14 selected threshold:
0.95"); the full run took 3.3 min (Entry 4); codec groups REAL 8,800 / FAKE_mp3 7,592 / FAKE_other
1,329; `deepvoice_summary.md` starting with the blind-audit attestation (64 files); overlap
`n_overlap: 0`. Cells 6 and 8 can also run from the Drive archive alone (notebook intro).
