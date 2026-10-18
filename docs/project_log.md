# Project Log — Voice-Spoofing Detection for Fraud Prevention

Team: Donia Mabrouk · Iheb Zemzemi · Malak Ben Salem · Khalil Dridi (ownership by part: D26)
Repo: `kahlanzzi/voice-spoofing-detection` (owner Iheb Zemzemi; transferred from Doniaa00 on 2026-10-14, Entry 20)

---

## How to use this file

- **Purpose:** the chronological story of the project — what we did, what we found, what we decided, and why. Anyone joining (teammate, instructor, new Claude Code session) reads the **Current status** block at the bottom first.
- **This file points; it does not duplicate.** Official decisions live in `docs/decisions/README.md`. Official audit facts live in `docs/dataset_audit.md`. The log references them by ID (D13, H5, …).
- **Log entries are append-only.** Never edit an old entry; if something was wrong, add a new entry that corrects it.
- **At every update:** (1) add a new log entry, (2) tick or add items in **Open items**, (3) rewrite **Current status**.
- **Evidence labels:** **[verified]** = checked against data or sources (evidence for review, not acceptance) · **[doc]** = from documentation · **[hypothesis]** = not yet tested · **[to verify]** = we are not sure it happened.
- **Acceptance** = team review, recorded in the sign-off tables (`docs/dataset_audit.md` §10, `docs/threat_model.md` §14). AI-produced work is not accepted until then (`CLAUDE.md`, human in the loop).

---

## Log entries

### Entry 1 — Project Readiness Audit (before 2026-10-04)

**What we did:** compared the Proposal, SRS v3.0, SDD v1.0 and Execution Plan against each other and against the research goal, before writing any code.

**What we found:**
- Readiness score 60/100: governance and architecture strong; experimental protocol missing.
- 5 pre-training blockers:
  - B1: no Experimental Protocol document.
  - B2: the schedule ran the DEEP-VOICE test before thresholds and calibration existed.
  - B3: the Execution Plan touched DEEP-VOICE twice.
  - B4: metrics undefined for a 3-way output.
  - B5: evaluation unit / clip length undefined.

**Decisions:** D1–D14 approved (D5 provisional, D13 deferred to the audit) → `docs/decisions/README.md`. Research question narrowed to FoR (TTS) → DEEP-VOICE (RVC) generalization (D8).

**Why it matters:** fixing the order and the rules before coding protects the project's central claim — an honest, one-time external test.

### Entry 2 — Stage 0: repo scaffold (2026-10-04)

**What we did:**
- Built the repo with Claude Code from a bootstrap package (`CLAUDE.md`, `docs/decisions/README.md`, audit code, reference documents).
- `CLAUDE.md` holds the standing rules for every Claude Code session.
- Fixed two Windows bugs in the audit code: UTF-8 file writing, and `/` paths in manifests.
- **D15:** official Python = 3.13 (Colab 3.13.15); local development on 3.14 allowed; CI will test on 3.13.
- Created the GitHub repo `Doniaa00/voice-spoofing-detection`, pushed the first commit (31 files, checked with `git status` before committing), and invited Iheb and Malak.
- Claude Code added a "Git workflow" section to `CLAUDE.md`: one branch per task, PRs only, Claude Code never merges.

**Evidence:** 5 tests passed on Windows after the fixes [verified].

**Decisions:** Iheb will become repo owner via ownership transfer (Option A), timing open.

### Entry 3 — Stage 1 preparation: rules before results (2026-10-04)

**What we did:**
- Approved the pre-registered decision rules in `docs/dataset_audit.md` §0 **before** any audit output existed.
- Refined D14 before running anything:
  - Chaining guard: the largest group must hold ≤ 5% of decodable clips.
  - PCM sanity check now part of threshold eligibility.
  - Selection order: 0.95 → 0.98 → PCM hash only.
  - Edge cases: unverifiable pairs fail; partially checkable pairs are decided on the checkable ones.
- Extended the audit code so it applies the rule automatically.

**Commits on `stage-1-audit-run`:** `05dd22e` (group stats per threshold) → `31f6503` (sanity check wired into selection) → `ae45e9f` (§0 text + §9 change log).

**Evidence:** 12 tests passed locally (3.14) and in Colab on Python 3.13.15 at commit `ae45e9f` [verified].

**Why it matters:** rules written before seeing data cannot be bent to fit the data.

### Entry 4 — Stage 1: FoR-2sec audit run (2026-10-04)

**What we did:**
- Downloaded FoR-2sec at the exact revision `ff8c82c79e7bbefef811941bac9775c9328e9055`.
- Backed it up as one archive: Drive `raw/for_2sec_ff8c82c7.tar` (1.22 GB), with the sha in `raw/for_2sec_REVISION.txt`.
- Ran the full audit on 17,721 clips (3.3 min), then a codec follow-up check.

**What we found [verified]:**
- 17,721 WAV: 8,921 FAKE / 8,800 REAL (50.3 / 49.7%).
- 0 corrupt files.
- All clips: 16 kHz mono PCM-16, exactly 2.0 s.
- 0 exact duplicates.
- Near-duplicates at 0.95: 98 pairs, largest group 6 clips, 0 across labels.
- No train/val/test folders. No speaker or source metadata: `fileN` numbers are independent per folder.
- The package contains no README or license file.
- **MP3 history:** 7,592 FAKE clips (85% of FAKE) were MP3 files before conversion; 0 REAL clips were.
- **Codec check:** both fake subgroups differ from REAL in similar ways.
  - Bandwidth (99% energy) median: FAKE_mp3 3,221 Hz, FAKE_other 3,461 Hz, REAL 5,074 Hz.
  - Separability on bandwidth: 0.666 for FAKE_mp3 vs 0.600 for FAKE_other.
  - Loudness separability ≈ 0.70 for both.
  - Silence: FAKE_other median 6% vs REAL 1% (separability 0.750, moderate).
- Interpretation: MP3 history is not the main separator in coarse statistics. The CNN could still exploit finer traces [hypothesis].

**Decisions (applying the §0 rules):**
- **D13 → re-split** (no packaged split).
- **D14 → near-duplicate groups at 0.95** (no speaker metadata; both checks pass).
- **D9 → val-A / val-B applies** (val-B ≈ 665 per class ≥ 500).
- **D16 (accepted by the team):**
  - no data changes;
  - re-split stratified by subgroup (REAL / FAKE_mp3 / FAKE_other);
  - pre-registered subgroup test (H5);
  - silence re-checked after trimming in Stage 6;
  - codec-history imbalance recorded as a limitation.
- **H5:** if the CNN relies on MP3 traces, its recall on FAKE_other will be clearly lower than on FAKE_mp3 (FoR test, Stages 11/16).
- **H6:** if the CNN relies on bandwidth, band-limiting REAL clips to ~3.4 kHz will raise false "FAKE" calls (Stage 14 robustness test).

**Not yet written into `docs/decisions/README.md` / `docs/dataset_audit.md`** — planned as one commit at the end of Stage 1.

### Entry 5 — Stage 1: DEEP-VOICE metadata audit and audit documentation (2026-10-04)

**What we did:**
- Downloaded DEEP-VOICE from Kaggle (`birdy654/deep-voice-deepfake-voice-recognition`, "updated 3 years ago", no version number).
- Archived the download: zip 3.96 GB, sha256 `8cb25853…f732bdcf` (full hash in `docs/dataset_audit.md` §1). Drive location TBD.
- Ran the metadata-only audit (`src/audit/deepvoice_metadata_audit.py`) on `KAGGLE/` only. It read headers, byte hashes, filenames and the CSV schema; it decoded no audio (attestation in `docs/dataset_audit.md` §5). `DEMONSTRATION/` (2 MP3) was excluded and never opened; `DATASET-balanced.csv` is never used.
- Checked FoR ∩ DEEP-VOICE for identical files.
- Wrote up Stage 1: `docs/dataset_audit.md` §1–§9, `docs/decisions/README.md` (D5, D9, D13, D14 settled; D16–D19 and H5–H6 added). Evidence summaries committed in `docs/audit_evidence/`.

**What we found [verified]:**
- 64 WAV files: 8 REAL / 56 FAKE. All stereo. Sample rates 40 / 44.1 / 48 kHz.
- 8 speakers (biden, linus, margot, musk, obama, ryan, taylor, trump). FAKE = every source → target pair (8 × 7).
- Durations 79–600 s. As 2-s windows: 1,870 REAL / 13,090 FAKE (87.5% FAKE).
- 0 exact duplicates inside DEEP-VOICE; 0 identical files shared with FoR.
- The "628 / 4,425" counts in the Proposal and Execution Plan match nothing in the package (files, windows or CSV).

**Decisions:**
- **D5 confirmed:** 2-s windows on DEEP-VOICE.
- **D17:** speech-window filter on DEEP-VOICE using the frozen silence detector. Threshold fixed in Protocol v1.0. Filtered (primary) and unfiltered (sensitivity) views come from the same single run. Drop counts reported per class and per speaker.
- **D18:** mono averaging + 16 kHz resampling, identical everywhere. Stage 6 tests pipeline invariance on FoR clips converted to 40 / 44.1 / 48 kHz stereo.
- **D19:** external results include a per-speaker table (8 rows) and a speaker average.

**Why it matters:** DEEP-VOICE differs from FoR in format (stereo, higher sample rates, long files) and has only 8 speakers. D17–D19 fix how the one external run handles that, before anything is scored.

### Entry 6 — Stage 1: open TBDs resolved (2026-10-04)

**What we did:** filled every TBD left in `docs/dataset_audit.md` (commit `c4a1b56`) and aligned D5 wording in `docs/decisions/README.md`.
- **Provenance:**
  - Audit run by Donia Mabrouk.
  - DEEP-VOICE audit code commit `ae45e9f` (printed by notebook cell 1, same Colab session as the FoR run).
  - Both datasets downloaded 2026-10-04.
- **Drive locations:**
  - Full audit outputs: `MyDrive/voice-spoofing-detection/processed/metadata/stage1/{for2sec,deepvoice}`.
  - DEEP-VOICE zip: `MyDrive/voice-spoofing-detection/raw/deep-voice-deepfake-voice-recognition.zip`, hash in `raw/deep_voice_ZIP_SHA256.txt`. This resolves "Drive location TBD" in Entry 5.
- **HF card checked [verified]:** the card is empty; the dataset viewer shows only `audio` and `label`. No per-file speaker, source or generator metadata. **D14 confirmed** on filenames + card (groups at 0.95).
- **Licenses recorded [doc]:**
  - FoR: not stated by the HF repackager; the original FoR terms apply (Reimao & Tzerpos, 2019).
  - DEEP-VOICE: MIT (Kaggle description), attribution Bird & Lotfi 2023, arXiv:2308.12734.
  - Both are used for academic research only; audio is never redistributed.
- **Ethics / rights:** DEEP-VOICE recordings are of real public figures, and the dataset license does not necessarily cover rights in the original speeches. Added as limitation 6 in `docs/dataset_audit.md` §8.
- **D5 wording:** raw FoR clips need no pad/crop (all 2.0 s); whether silence trimming introduces length variation is decided in Stage 6.

**Why it matters:** the audit document now has no open placeholders, so the team can sign it off as written.

### Entry 7 — Stage 1: audit notebook ported, datasets pinned, D20 (2026-10-04)

**What we did:**
- **Notebook port** (commit `f8af88a`): rewrote `notebooks/01_stage1_dataset_audit.ipynb` from the working Colab copy. It has 9 code cells, each with a markdown explanation, and outputs stripped:
  - setup + tests
  - FoR download + `.tar` backup
  - DEEP-VOICE download
  - dry run
  - full FoR audit
  - codec check, now saved to `for2sec_codec_check.csv/.json`
  - DEEP-VOICE metadata audit
  - FoR ∩ DEEP-VOICE overlap, now saved to `cross_dataset_overlap.json`
  - archive

  Cells 6 and 8 also run from the Drive archive alone. The Colab copy was used as reference only and deleted, not committed.
- **Pins** in `configs/audit.yaml`:
  - `for2sec.revision = ff8c82c7…`; cell 2 stops if it is empty.
  - `deepvoice.zip_sha256 = 8cb25853…`; cell 3 stops before unzipping if the downloaded zip differs.
  - The DEEP-VOICE revision label no longer uses a relative date: `kaggle zip-sha256:<hash>, downloaded 2026-10-04`.
  - Outdated config comments fixed.
- **D20:** before the external run, the only permitted DEEP-VOICE accesses are download, zip hash, unzip, listing paths / extensions / sizes / byte hashes, and `deepvoice_metadata_audit.py`. Nothing reads audio content. `CLAUDE.md` hard rule 1 now says the same.

**Evidence [verified locally]:**
- The notebook validates with nbformat, and all 9 code cells compile.
- A local smoke test with synthetic inventories passed: the empty-revision and zip-mismatch guards raise, and cells 6 and 8 run from an archive-only layout.
- `pytest -q`: 12 passed.

Not yet run in Colab.

**Why it matters:** the audit can now be re-run from the repo alone, against exactly the same data, and every DEEP-VOICE touch before the external run is written down and limited.

### Entry 8 — Stage 1: Colab run of the ported notebook (2026-10-04)

**What we did:** ran the ported notebook in Colab, in the same session as the original audit: commit `c2136cb`, Python 3.13.15.
- **Cell 1 (setup):** clone/checkout, install, tests — 12 passed.
- **Cell 6 (codec check):** saved `for2sec_codec_check.csv` / `.json`.
- **Cell 8 (FoR ∩ DEEP-VOICE overlap):** saved `cross_dataset_overlap.json`.
- **Cell 9:** archived to Drive.
- Committed the three files to `docs/audit_evidence/`. `docs/dataset_audit.md` now cites them instead of "not yet archived".

**What we found [verified]:** the results are identical to the original run.
- Codec groups: REAL 8,800 / FAKE_mp3 7,592 / FAKE_other 1,329. 0 REAL clips have `mp3` in the stem.
- FAKE_other silence separability: 0.750.
- Overlap: 0 identical files (17,721 FoR vs 64 DEEP-VOICE file hashes).

**Caveat:** the Colab session still had its local outputs, so both cells read the local inventories (`source_inventory` = `/content/audit_out/...`). The archive-only read path of cells 6 and 8 has been exercised **on synthetic data only** (local smoke test, Entry 7), not on the real Drive archive.

**Why it matters:** every number in the audit document now points to a committed evidence file, produced by code in the repo at a recorded commit.

### Entry 9 — Stacked branch for Stage 2 (2026-10-04)

**Decision:** Stage 1 is waiting for team sign-off and PR approval. Work continues on a **stacked branch**, so the team review doesn't block progress.
- Created `stage-2-threat-model` **from `stage-1-audit-run`**, not from `main`.
- **Rule:** `stage-1-audit-run` receives no more commits except the §10 sign-off in `docs/dataset_audit.md`. All `docs/project_log.md` and `CLAUDE.md` updates now go on `stage-2-threat-model` (and later on the newest branch).
- `CLAUDE.md`:
  - The stacked-branch rule is added to the Git workflow section.
  - Roadmap: Stage 1 = "done — awaiting sign-off/merge", Stage 2 = CURRENT.
- The Git workflow section was only on the unmerged `stage-0-git-workflow` branch. It was brought in by merging `origin/stage-0-git-workflow` (`428e0b8`) into `stage-2-threat-model`, so both PRs carry the same commit and should not conflict.

**Merge order:** `stage-0-git-workflow` → `stage-1-audit-run` → `stage-2-threat-model`. Until the first two are merged, the Stage 2 PR also shows their changes.

**Why it matters:** the Stage 1 evidence stays frozen while it is reviewed, and the log stays on a single line of history.

### Entry 10 — Stage 2: threat model draft v0.1 (2026-10-04)

**What we did:**
- The team wrote `docs/threat_model.md` v0.1. It covers:
  - scenarios: call-center triage of high-risk requests (primary) and an executive voice message (secondary);
  - the attack path and detection point;
  - assets, attacker profiles and attack classes in and out of scope;
  - threats T1–T7 mapped to the planned evidence;
  - the error policy and threats to the system itself;
  - the oracle-abuse residual risk.
- Committed as-is (commit `8b7b679`).
- Checked the draft (Claude Code):
  - every FR / NFR / PC / WH / UN ID against the SRS v3.0 text;
  - every D* and H* against `docs/decisions/README.md`;
  - the cited audit facts against `docs/dataset_audit.md`;
  - Proposal §2.3;
  - the mermaid syntax.

**What we found [verified]:**
- **All cited IDs exist** and their meaning matches the SRS:
  - FR-03, FR-05–FR-07, FR-11, FR-12–FR-18, FR-20
  - NFR-01–NFR-05, NFR-07, NFR-09
  - PC-03–PC-05, WH-01–WH-04, UN-01, UN-02
  - D2, D8, D11, D12, D14, D16–D20, H5, H6
- SRS §1.2 and §7.4 support the cited content: scope, open limits, access-control mechanism. Proposal §2.3 is "Attacker capability assumptions".
- **Mermaid block parses** (flowchart; `mermaid.parse` 11.17.2, with a broken-diagram negative control).
- **Two mismatches, reported for team review, not fixed:**
  1. §3.2 says "all files `src-to-tgt`". Only the 56 FAKE files are; the 8 REAL files are `<speaker>-original.wav` (`docs/dataset_audit.md` §5).
  2. §10, "Tampering (data)", cites **D20** for the pinned FoR revision / DEEP-VOICE zip SHA-256 and the stop-on-mismatch. D20 defines the *permitted DEEP-VOICE accesses*; the pinning is in `configs/audit.yaml` (Entry 7) and has no decision ID.
- **Observations (not mismatches):**
  - The §10 spoofing control relies on FR-11, which is a *Could* requirement.
  - T6 cites D2 for calibration on val-B; the val-A / val-B split itself is D9.

**Why it matters:** the threat model sets the maximum claim for each threat. Its references have to be exact, because the final report will quote them.

### Entry 11 — Stage 2: threat model v0.2, D21 (2026-10-04)

**What we did:** applied the approved fixes from the v0.1 review (Entry 10). `docs/threat_model.md` → **v0.2**:
- **§3.2:** "8 source speakers [verified]; the 56 FAKE files are named `src-to-tgt`, the 8 REAL files `<speaker>-original`."
- **§10, Tampering (data):** now cites **D21** and `configs/audit.yaml` instead of D20.
- **§10, Spoofing:** status adds "FR-11 is Could priority; if not implemented, this threat is unmitigated."
- **§11:** new residual risk 7, analyst impersonation if FR-11 is not implemented.
- **§8 T6:** cites D2 and D9.
- **§12:** §8 row adds D9; §10 row cites D21 instead of D20.
- **D21** added to `docs/decisions/README.md`: datasets are pinned (FoR-2sec HF revision, DEEP-VOICE zip SHA-256) in `configs/audit.yaml`; the notebook refuses unpinned or mismatched data; changing a pin requires a new decision entry.

**Evidence [verified]:** citation check re-run on every changed line.
- All cited IDs exist: SRS v3.0, plus D2, D9 and D21 in `docs/decisions/README.md`.
- The D21 hashes equal the values in `configs/audit.yaml`.
- `deepvoice_summary` confirms 56 FAKE `src-to-tgt` and 8 REAL `<speaker>-original` files.
- The SRS lists FR-11 as Could (requirements table and MoSCoW table).
- D20 is no longer cited in the threat model.

**Why it matters:** every claim and citation in the threat model now traces to its source. The one gap, analyst authentication, is stated as a residual risk instead of being implied as covered.

### Entry 12 — Plan review and corrections (2026-10-04)

**Correction to Entry 10:** `docs/threat_model.md` v0.1 was drafted with Claude (AI assistant, chat) from the team's documents and decisions, then committed for team review. **The team has not reviewed it yet.** The audit code in `src/audit/` was likewise written with AI assistance and verified by tests.

**Execution Plan comparison:**
- **Phase 1:** SRS requirements covered. Threat model in progress. GitHub Issues board not created.
- **Phase 2:** dataset audit done. Preprocessing, manifests and compute benchmark not started.

**Document discrepancies found [doc, checked against the reference files]** — to fix later, not now:
- DEEP-VOICE zip is 3.96 GB [verified]; the Execution Plan says ~0.55 GB.
- The Proposal's architecture figure still shows "ASVspoof + WaveFake" as training data and "baseline + deep model" inside the deployed detector.
- Proposal §3.2 calls real-time / streaming detection a stretch goal; SRS WH-01 excludes it.
- The Execution Plan lists six named TTS engines for FoR; this is not determinable from the package (`docs/dataset_audit.md` §2).
- Execution Plan steps superseded by D2, D3, D6 and D10.

**Stage 6 flag:** the Execution Plan suggests `silero-vad` (a small pretrained model) for voice-activity detection / silence trimming. Choosing it must be an **explicit Stage 6 decision**, not a default. D17 (DEEP-VOICE speech-window filter) depends on the chosen detector.

**Stage 3 rule:** Experimental Protocol v1.0 may be **drafted now** on a stacked branch. It is **frozen only after the Stage 1 sign-off**.

### Entry 13 — Human in the loop; label meaning; Issues board (2026-10-04)

- **Human-in-the-loop rule** added to `CLAUDE.md` "Working mode" (commit `684f7cd`). Every AI-produced artifact (code, documents, analysis) is reviewed by a team member before it is accepted. AI output is never merged, cited or reported as verified without that review. AI assistance is recorded honestly in this log.
- **Meaning of labels:**
  - **[verified]** = checked against data or sources. It is evidence for review.
  - **Acceptance** = team review, recorded in the sign-off tables: `docs/dataset_audit.md` §10 and `docs/threat_model.md` §14.
  - The evidence-labels line in "How to use this file" is updated to match.
- **Issues board:** follows the Execution Plan (Phase 1, task 5). One issue per SRS requirement (FR / NFR / PC IDs), grouped under one milestone per stage. **This corrects the wording logged in Entry 12** ("one issue per upcoming stage"); there is no plan deviation. The open item is updated to match.

### Entry 14 — Closeout of Stages 0–2: merges and reviews (2026-10-06)

**Merges and reviewers (from GitHub; merge commits confirmed in git):**

| PR | Branch → base | Reviewed / approved by | Merged by | Date |
|---|---|---|---|---|
| #1 | `stage-0-git-workflow` → `main` | Malak Ben Salem | Malak Ben Salem | 2026-10-05 |
| #2 | `stage-1-audit-run` → `main` | Malak Ben Salem | Malak Ben Salem | 2026-10-05 |
| #3 | `stage-2-threat-model` → **`stage-1-audit-run`** (by mistake) | Malak Ben Salem | Malak Ben Salem | 2026-10-05 |
| #4 | `stage-2-into-main` → `main` | Iheb Zemzemi | Donia Mabrouk | 2026-10-06 |

**What went wrong with PR #3:**
- PR #3 had `stage-1-audit-run` as its base. It was merged one minute *after* PR #2 had already merged `stage-1-audit-run` into `main`, so the Stage 2 work landed on a branch nothing merges any more.
- A direct PR `stage-2-threat-model` → `main` then showed a conflict on GitHub. The branch had **two merge bases** with `main` (`4f4202f` and `428e0b8`, the stage-0 commit merged into both lines). Git's default merge strategy merges cleanly; a merge on a single base conflicts in `CLAUDE.md`.
- The work was re-delivered through `stage-2-into-main`, built from `main` (merge `d2f6618`; content identical to the Stage 2 tip `28255c1`), as **PR #4**.
- The **Stacked PRs** rule was added to `CLAUDE.md` (`5e4bd2d`): before merging a stacked PR, change its base to `main` once the PR below it is merged.

**Sign-offs recorded:**
- `docs/dataset_audit.md` §10:
  - Malak: reviewed and approved via PR #2 (2026-10-05).
  - Donia: author.
  - Iheb: not yet reviewed.
  - Status: "Audit complete; signed off by Malak (PR #2)".
- `docs/threat_model.md` §14 → **v1.0 "reviewed"**:
  - Malak: approved via PR #3 (2026-10-05).
  - Iheb: approved via PR #4 (2026-10-06).
  - Donia: author (AI-assisted draft).
  - The §13 open questions stay open.

**Other facts:**
- Iheb and Malak accepted the collaborator invitations.
- GitHub shows a branch-protection rule that applies to 1 branch. Its exact settings (PR + 1 approval, no force-push, no bypass) are not yet confirmed.
- `CLAUDE.md` roadmap: Stages 0, 1 and 2 done; **Stage 3 (Experimental Protocol) CURRENT**.

**Why it matters:** the review evidence for Stages 1 and 2 is now in the documents themselves. The stacked-PR mistake is understood and guarded against by a written rule.

### Entry 15 — AI context and workflow structure (CS 488 template) (2026-10-11)

**Authorship:** requested and specified by Donia Mabrouk. All files were drafted by **Claude Code** (Claude Opus 5.5), an AI tool, on branch `chore/ai-context`. **Not yet reviewed by the team** (human-in-the-loop: acceptance = PR review).

**Context:** PR #5 (`docs/stage2-closeout`) was merged into `main` by Donia Mabrouk on 2026-10-06 (merge commit `1bb25c5`; the approver is not recorded in git).

**What we did:** adopted the CS 488 starter-template AI-context and workflow structure. Docs and config only; no change in `src/`.
- **`AGENTS.md`** (new): all content of the former `CLAUDE.md`, tool-neutral ("AI agents never merge"), with the roadmap (Stages 0–2 done, Stage 3 CURRENT). Branch naming updated to D23; pointers to `CONTRIBUTING.md` and the decision register.
- **`CLAUDE.md`**: now `@AGENTS.md` plus a short "Claude-specific notes" section.
- **`AI_CONTEXT.md`**: pointer to `AGENTS.md` and ADR 0003. **`.gemini/settings.json`**: `{"context": {"fileName": ["AGENTS.md"]}}`.
- **`CONTRIBUTING.md`**: branch names `<type>/stage<N>-<name>`, Conventional Commits, one unit = one branch = one PR, PR + 1 approval, stacked-PR rule, human in the loop, AI agents never merge, responsible AI use. Written from the points in the request; the CS 488 template text itself was not available to the AI tool.
- **`.github/`**: PR template (checkboxes: DEEP-VOICE untouched (D20), decision logged, AI assistance used), issue templates (feature request asks for the SRS ID; bug report), `CODEOWNERS` (`@Doniaa00 @kahlanzzi @malakbensalemm`).
- **Decisions:** the single-file register (`decisions.md` directly under `docs/`) moved to **`docs/decisions/README.md`** (register; content unchanged except ADR links and new rows). Added `TEMPLATE.md` and ADRs 0001 (record decisions, D24), 0002 (tech stack, D15), 0003 (`AGENTS.md` single AI context, D22), 0004 (dataset pinning, D21), 0005 (DEEP-VOICE single use, D2/D3/D6/D20).
- **Register rows added** (approved by the team, 2026-10-06): **D22** `AGENTS.md` single AI context; **D23** branch naming `<type>/stage<N>-<name>`; **D24** register + ADRs for major decisions only; **D25** MIT license for code only, datasets never redistributed.
- **References updated:** every reference to the old register path replaced by `docs/decisions/README.md` in `README.md`, `KICKOFF_PROMPT.md`, `docs/dataset_audit.md`, `docs/threat_model.md` and this log. `README.md` now points to `AGENTS.md` instead of `CLAUDE.md`.

**One-time exception to the append-only rule (approved by Donia):** the old register path was replaced by `docs/decisions/README.md` inside old log entries (10 occurrences in total in this file). Only the path changed; no other wording. Old entries that say "`CLAUDE.md`" now refer to rules that live in `AGENTS.md` (ADR 0003).

**Evidence [verified]:** `pytest -q` passes; `git grep` for the old register path returns nothing; `CLAUDE.md` contains `@AGENTS.md`; `AGENTS.md` size recorded in the PR.

**Why it matters:** every AI tool now reads the same rules, decisions have a durable home with reasoning for the major ones, and PRs carry the checks (DEEP-VOICE untouched, decision logged, AI use disclosed) by default.

### Entry 16 — Review of `chore/ai-context` (2026-10-11)

**Review (Donia Mabrouk):** `chore/ai-context` approved with changes. Changes applied on the same branch by **Claude Code** (Claude Opus 5.5); not yet re-reviewed.
- `kahlanzzi` is confirmed as Iheb Zemzemi's GitHub account (`.github/CODEOWNERS` unchanged).
- The path-only edit inside old log entries (Entry 15) is accepted as a **one-time exception**. Append-only applies again from now on.
- **D20:** annotation appended, decision text unchanged: "(Since D22, this rule lives in AGENTS.md, which CLAUDE.md imports.)"
- The stale bootstrap prompt file at the repo root (Stage 0 instructions that still pointed to `CLAUDE.md`) is **deleted**.
- **Remote branches deleted:** `stage-1-audit-run` (`2d1d872`), `stage-2-threat-model` (`28255c1`), `stage-2-into-main` (`5e4bd2d`). Before deletion, each tip was re-checked: unchanged since the content check, and merging it into `main` changes nothing. `git ls-remote --heads origin` now lists only `main` and `chore/ai-context`.
- **License:** the MIT `LICENSE` file (D25) goes in a separate PR ("PR B").
- **Branch protection:** no code-owner review requirement.

**Not done — CS 488 template comparison:** the template folder named in the review (`cs488-template` on the Desktop) does not exist on this machine. The only CS 488 file found is the course PDF, which does not contain the template files. `CONTRIBUTING.md`, the `.github/` templates, `docs/decisions/TEMPLATE.md` and ADR 0001 are therefore **not yet aligned**; the Open item stays open.

**Evidence [verified]:** `pytest -q` passes (12 tests). `git grep` for the deleted file's name finds one remaining mention, inside Entry 15 (committed before this review). It is left unchanged because of the append-only rule.

### Entry 17 — Repo structure: CS 488 template alignment, hygiene, CI, docs (2026-10-11)

**Authorship:** requested and specified by Donia Mabrouk. All changes drafted by **Claude Code** (Claude Opus 5.5), an AI tool. **Not yet reviewed by the team** (acceptance = PR review).

**Branch:** `chore/ai-context` renamed to **`chore/repo-structure`** (local and remote; old remote branch deleted; no PR existed). One PR, three topic commits on top of Entries 15–16:
- **A — workflow docs:** `CONTRIBUTING.md`, PR template, issue templates, ADR `TEMPLATE.md` and ADR 0001 restructured on the CS 488 template (now found at `Desktop/cs488-template`, read only, not copied). Kept: D20 checkbox, stacked branches/PRs, D23 naming, human in the loop, AI agents never merge. ADRs 0002–0005 keep their table header (accepted ADRs are not rewritten).
- **B — hygiene, CI, safety:** `.gitattributes` (LF; pdf/docx/png/wav/npy/pt binary; `--renormalize` changed no file), `.editorconfig`, `.env.example` (placeholders only), `.gitignore` merged with the template's secrets/editor/logs sections plus local AI-tool overrides (manifest CSVs now trackable), `scripts/setup.sh`, `check-docs.sh`, `update-structure.sh` (adapted: tree built from `git ls-files`, so `.venv/` and caches never appear), `.githooks/pre-push`, `.vscode/extensions.json`, CI (`.github/workflows/ci.yml`: Python 3.13, `pytest -q`, docs reminder, no dataset downloads), `requirements.txt` pinned to the versions in the local venv.
- **Governance test** `tests/governance/test_deepvoice_access.py`: fails if a tracked or untracked (not ignored) file outside the permitted places references the DEEP-VOICE root (`deep_voice`), the Kaggle slug/owner or the `KAGGLE` subfolder; a second test proves it flags planted violations and ignores allowed files. **Allowlist addition to review:** `data/README.md` (documents the Drive layout `raw/deep_voice/`; no code), beyond the list in the request.
- **C — docs:** `docs/README.md` (index), `docs/architecture.md` (SDD summary, one Mermaid diagram, deviations set by D2/D3/D9), `docs/setup.md`, `docs/CHANGELOG.md`, `docs/setup-report.md` (Lab 6 skeleton, TODOs only), MIT `LICENSE` (D25; standard text, the "code only" scope is stated in `README.md`), `README.md` in the template format with a generated structure tree. `data/README.md` pointers updated from `CLAUDE.md` to `AGENTS.md` and to D20.
- **`AGENTS.md`:** Reports rule (max ~20 lines); docs index pointers; governance test under hard rule 1; secrets under hard rule 7; exact pins and CI path in Code rules; "CI must be green" in Git workflow.

**Evidence [verified]:** `pytest -q` → `15 passed`. `bash scripts/setup.sh` → hooks enabled (`core.hooksPath = .githooks`). `git check-ignore`: `.env` ignored; `data/manifests/*.csv` and `docs/audit_evidence/*` not ignored. No `.env` in `git status`. `git grep KICKOFF_PROMPT` → only `docs/project_log.md`; `git grep` for the old register path → no match. `pip download --only-binary=:all:` resolves the full pinned set as Linux wheels for Python 3.13 and 3.14; local install verified on 3.14.7 only (no 3.13 locally; CI installs on 3.13).

**Why it matters:** the repo now matches the course template, CI runs the tests on every PR, secrets and line endings are handled by default, and hard rule 1 has an automated guard.

### Entry 18 — Team of four, D26, repo cleanup before Stage 3 (2026-10-12)

**Authorship:** requested and specified by Donia Mabrouk. All changes drafted by **Claude Code** (Claude Opus 5.5), an AI tool, on branch `docs/stage3-repo-cleanup`. **Not yet reviewed by the team** (acceptance = PR review).

**Context:**
- PR #6 (`chore/repo-structure`, Entries 15–17) merged into `main` (merge commit `549b34b`).
- Branch protection on `main` confirmed: PR + 1 approval + required CI check `checks` + branch up to date; no direct pushes.
- **Execution Plan v2.1** shared (`docs/reference/Execution_Plan_v2.1.pdf`, 2026-10-12). It supersedes the v1 `Execution_Plan.md`, which is kept for history.
- **Khalil Dridi joined the team** (approved by the team and the professor).
- **D26** (work distribution, Execution Plan v2.1) added to `docs/decisions/README.md`.

**What we did (this cleanup PR; docs/config text only, no change in `src/`, dataset pins, test logic or the `ALLOWED` list):**
- Team of four in `AGENTS.md`, `README.md`, `LICENSE`; `.github/CODEOWNERS` adds `@dridi-khalil`.
- `AGENTS.md`: Owner column in the roadmap (D26), branch-protection line, Execution Plan pointer → v2.1.
- Register header no longer limited to the readiness audit; `Execution_Plan.md` marked superseded.
- "track (ml / backend / frontend)" → "part (A–E) and owner (D26)" in `CONTRIBUTING.md` §8 and the feature-request template.
- `data/README.md`: Drive tree redrawn as it really is (raw archives and hash files, `processed/metadata/stage1/`; manifests, features, checkpoints planned for Stage 4+). Only the comment on its governance-test allowlist entry changed.
- Stale `CLAUDE.md` mentions → `AGENTS.md` in `configs/audit.yaml` and the notebook intro (outputs still stripped).
- D26 completed with the four v2.1 §5 hand-overs.
- Per-folder `CODEOWNERS` (owner + consumer reviewer).
- Transfer decision (after Part A).
- Kaggle token regenerated, stored only in local `.env`.

**Evidence [verified]:** `pytest -q` → 15 passed (Python 3.14 local); CI result on the PR.

### Entry 19 — Stage 3 started: protocol skeleton (2026-10-12)

**Authorship:** requested and specified by Donia Mabrouk. Drafted by **Claude Code** (Claude Opus 5.5), an AI tool, on branch `docs/stage3-protocol-skeleton`. **Not yet reviewed by the team** (acceptance = PR review).

**Context:**
- PR #7 (`docs/stage3-repo-cleanup`, Entry 18) merged into `main` (merge commit `b03bfe1`).
- All merged branches deleted; the remote has only `main` [verified: `git fetch --prune` → `origin/main` only].
- **Stage 3 started.** Plan: (1) skeleton PR (this one) → (2) one PR per author for their sections (D26; v2.1 Stage 3 authors) → (3) integration by Donia + sign-off by all four (freeze F1).

**What we did (docs only; no code, no data access):**
- `docs/experimental_protocol.md` v0.1: header (freezes at v1.0 = F1, D1), sections 0–14, each with owner, reviewer (consumer, D26), "Must decide" checkboxes citing a D-ID or source, and Status TODO. Nothing decided; v2.1 values marked "proposed in v2.1; to confirm". H5 and H6 copied unchanged from the register.
- `AGENTS.md`, `docs/README.md`: protocol listed as draft v0.1. `docs/CHANGELOG.md`: one line.

**Evidence [verified]:** `pytest -q` → 15 passed (local venv); CI result on the PR.

### Entry 20 — Repo transferred to Iheb (2026-10-14)

**Authorship:** drafted by **Claude Code** (Claude Opus 5.5), an AI tool, on branch `chore/stage3-repo-transfer`. **Not yet reviewed by the team** (acceptance = PR review).

**Context:**
- PR #8 (`docs/stage3-protocol-skeleton`, Entry 19) merged into `main` (merge commit `fe5bb07`); the merged remote branch `docs/stage3-protocol-skeleton` was deleted.
- The repo moved from `Doniaa00/voice-spoofing-detection` to **`kahlanzzi/voice-spoofing-detection`** on 2026-10-14. This is earlier than planned ("after Part A", Entry 18): no PRs were open, so the team agreed to do it now.

**What we did:**
- Donia stays a collaborator with write access.
- - Donia ran `git remote set-url origin https://github.com/kahlanzzi/voice-spoofing-detection.git`; a push to the new remote worked (deleting `docs/stage3-protocol-skeleton`). Iheb, Malak and Khalil must run the same command.
- Repo URL updated in `AGENTS.md` (Git workflow), `README.md` and `docs/setup.md` (clone command), `notebooks/01_stage1_dataset_audit.ipynb` (`REPO_URL`; outputs still stripped, valid per `nbformat`), and this log's header. Not changed: `.github/CODEOWNERS` (`@Doniaa00` is a username), old log entries, `docs/audit_evidence/`, `docs/reference/`.
- Collaborators, branch protection and Actions settings on the new repo: to be confirmed by Iheb [to verify].

**Evidence [verified]:** `pytest -q` → 15 passed (local venv); `git grep -n "Doniaa00"` → only `.github/CODEOWNERS` and the log (header line and Entries 2, 15, 20).

### Entry 21 — Stage 3: Experimental Protocol v1.0 (2026-10-14)

**Authorship:** requested and specified by Donia Mabrouk. Written by **Claude Code** (Claude Opus 5.5), an AI tool, on branch `docs/stage3-protocol-v1`, from the team-accepted defaults. **Signed off by all four via PR #10** (approvals: Malak 2026-10-14, Khalil 2026-10-14, Iheb 2026-10-14; Donia author, PR opened 2026-10-14); **v1.0 = freeze F1**.

**Context:**
- PR #9 (`chore/stage3-repo-transfer`, Entry 20) merged into `main` (merge commit `7089b0e`).
- Donia drafted "Stage 3 Protocol — Proposed Defaults" with Claude (claude.ai): one proposed value per open "Must decide" item, marked Fixed / Judgment / Team, with its basis. Shared with the team as a Claude doc; a Markdown export was used as input, kept outside the repo.
- All four members accepted every row on 2026-10-14 in the team chat. So the protocol is written in one PR instead of one PR per author (Entry 19 plan).

**What we did (docs only; no code, no data access):**
- `docs/experimental_protocol.md` → **v1.0**: in §0–§12 each "Must decide" checklist replaced by a "Decided" list (one bullet per accepted row, citing its D-ID / FR / basis; Judgment and Team rows marked "team choice, 2026-10-14"); every Status = Decided.
- Key values: seeds 0/1/2 + split/bootstrap seed 2026; bundle seed = median val-A EER; Platt scaling, threshold at the val-B EER point, symmetric uncertain band deferring 10% of val-B; D17 speech-window filter ≥ 50% speech frames; robustness levels per §9; H5/H6 operational thresholds (≥ 5 points, CI excludes 0).
- §12: Change control paragraph (Stage 6 deferred items → F3; other changes before data is seen → new decision D27+; no change after results are seen, extra analyses exploratory). §13 (sign-off commit): Date and PR #10 filled for all four; Donia's role "Lead (author)". §14: row 1.0, PR #10.
- Not placed in the protocol: online handling of clips longer than 2 s (open in the SDD; to decide before Stage 17) and the meeting dates from the defaults file.
- `docs/CHANGELOG.md`: one line (v1.0 frozen, F1). `AGENTS.md`: Stage 3 done, Stages 4–5 current; source-of-truth line updated.

**Evidence [verified]:** `python -m pytest -q` → 15 passed (local venv, Python 3.14.7); `grep` of the protocol for the governance-test patterns → no match; CI result on the PR.

---

### Entry 22 — Malak's Codex context test (2026-10-15)

**What we did (docs only; no data access):** added Malak Ben Salem's AI-tool test to
`docs/setup-report.md` Section 5, preserving the exact question and Codex answer. Marked the test
passed because the answer identified D20, the one-time external evaluation of DEEP-VOICE, and
`src/preprocessing/`.

**AI assistance:** **Codex** read `AGENTS.md`, `CONTRIBUTING.md`, the decision register, and the
repository structure; it documented the test and updated this session log. Human review is required
before acceptance (D22 and `CONTRIBUTING.md` Section 6).

**Evidence [verified]:** repository inspection confirmed `src/preprocessing/`; the recorded answer
names D20 and states that DEEP-VOICE is evaluated once after the ModelBundle freeze. Python 3.13
`pytest -q` completed with **11 passed, 4 failed in 165.23 s**. Two tests failed with explicit Windows
Application Control DLL-import errors. Two tests failed with `ValueError: cannot convert float NaN
to integer` because the test state had zero decodable files. The relationship between these failures
is not confirmed; CI must determine whether the two `ValueError` failures reproduce in a
policy-compatible environment. The focused governance suite passed: **3 passed in 0.07 s**.

---

### Entry 23 — Local pytest failure diagnosis (2026-10-15)

**What we did (read-only; synthetic fixtures only; no real dataset access):** reran the full local
suite with complete tracebacks, ran each failing test separately, inspected the audit call paths and
Windows Code Integrity events, and compared the installed package metadata with Python 3.13.

**Evidence [verified]:** the full local suite remained **11 passed, 4 failed**. Windows Application
Control blocked Numba 0.68.0's `_typeconv.cp313-win_amd64.pyd`. Two tests failed directly on that
import; two reached `ValueError: cannot convert float NaN to integer` after the same blocked import
made every otherwise-decodable synthetic FoR fixture appear undecodable. A controlled synthetic WAV
decoded successfully with fingerprinting disabled and failed when fingerprinting triggered the
blocked Numba import. The four failures therefore share one environment root cause. This is a Windows
enterprise-signing-policy issue, not a Python 3.13 incompatibility or an independently reproduced
repository defect.

**Conclusion:** no repository-code fix is required for the environment problem. GitHub CI or another
policy-compatible environment must provide the final full-suite verification. The audit's secondary
error-reporting weakness is optional future hardening; if the team approves it, use a separate
`fix/stage1-audit-error-propagation` branch.

**AI assistance:** **Codex** performed the diagnosis. Human review is required before acceptance.

---

### Entry 24 — Stage 4 leakage-aware split implementation (2026-10-15)

**Boundary [doc]:** Stage 4 is the FoR leakage-aware split and its automated checks. Stage 5
creates and hashes the frozen manifests (F2); Stage 6 creates the preprocessing profile. Neither
later stage was started.

**What we did (FoR metadata only; no audio access):** added `configs/split.yaml` and reusable split
code under `src/splitting/`. The code validates the pinned Stage 1 inventory/summary, applies the
D16 subgroup rule, keeps D14 0.95 groups atomic, uses seed 2026 and the frozen
70/7.5/7.5/15 ratios, and checks group isolation plus class/subgroup proportions within ±1 point.
It emits a candidate assignment CSV and JSON/Markdown check reports at a caller-supplied path;
it deliberately emits no Stage 5 manifest or hash. Added synthetic metadata-only tests and a
changelog line. Review corrections make official runs refuse dirty Git worktrees, reject Windows,
UNC and POSIX absolute paths plus traversal, and normalize validated split ratios to floats.

**Evidence [verified locally]:** final focused Stage 4 suite: **18 passed in 4.37 s**. Final full suite:
**33 passed in 6.36 s**; the Windows Application Control failure from Entries 22–23 did not
recur. An official-scale synthetic metadata check (17,721 rows with the audited subgroup totals)
passed every acceptance check: sizes 12,404 / 1,330 / 1,329 / 2,658 and maximum class/subgroup
deviation 0.034537 percentage points. `git diff --check` produced no output. The connected Drive
search exposed no accessible full `for2sec_inventory.csv`; the pinned summary already exists at
`docs/audit_evidence/for2sec_summary.json`. The split has therefore not yet been run on the real
inventory, and no real split report has been committed.

**AI assistance:** **Codex** established the Stage 4 boundary, implemented the config/code/tests,
ran the checks, and documented the session. Human review by Donia (consumer, D26) is required.

**Gate:** implementation checks pass; real-data acceptance checks and human review remain open.

---

### Entry 25 — Corrected official Stage 4 FoR split run (2026-10-16)

**Boundary [doc]:** Stage 4 FoR metadata only. No audio or DEEP-VOICE was accessed. No Stage 5
manifest/hash freeze or Stage 6 preprocessing was started.

**What we did:** Donia reviewed the first result and requested removal of four extra EOF blank
lines, a final clean implementation commit, and a determinism rerun. The local branch was renamed
to `feat/stage4-split` before its first push. The EOF-only correction is commit `10a5e68`; the
worktree was clean before the official rerun. The candidate CSV remains outside Git.

**Evidence [verified locally]:** the pre-rerun candidate had 17,721 rows and SHA-256
`853d947499e76c4cc9da5b69df384588f0023c21ed93f6a2dec1940f8988aa20`. The regenerated candidate
has the same row count and byte-identical SHA-256. Its intended destination is
`MyDrive/voice-spoofing-detection/processed/metadata/stage4/for2sec_split_assignments.csv`; upload
is pending Malak's confirmation. The reports cite clean commit
`10a5e6809bb55484908cbcce547a75c14d33a908`, include split × class × subgroup counts, and record all
checks passing. Focused tests: **18 passed in 10.83 s**. Full suite: **33 passed in 17.62 s**.
Windows Application Control blocked the `.venv` launcher, so Python 3.13.14 used the existing
pinned `.venv` site-packages; no dependency was installed.

**AI assistance:** **Codex** preserved and compared the candidate fingerprints, made the approved
EOF-only correction, performed the clean-commit rerun, ran tests and prepared the evidence. Donia
reviewed the prior result; final evidence acceptance remains human-controlled.

**Gate:** corrected Stage 4 evidence is ready for review. The Drive upload, evidence commit, PR and
final Stage 4 acceptance remain pending; Stage 5 has not started.

---

### Entry 26 — Stage 4 candidate Drive upload confirmed (2026-10-16)

**Evidence [verified by Malak]:** the final untracked 17,721-row candidate CSV was uploaded to
`MyDrive/voice-spoofing-detection/processed/metadata/stage4/for2sec_split_assignments.csv`.
Its recorded SHA-256 remains
`853d947499e76c4cc9da5b69df384588f0023c21ed93f6a2dec1940f8988aa20`. The repository retains only
the Markdown/JSON evidence; the candidate is ignored and untracked.

**AI assistance:** **Codex** recorded the human-confirmed upload and prepared the restricted
evidence commit and PR handoff. No dataset or candidate CSV is included in Git.

**Gate:** ready for the evidence commit and Donia's GitHub review. Stage 5 has not started.

### Entry 27 — Shared AI tooling v1: pr-prep skill and Context7 (2026-10-16)

**What:** on `chore/stage5-ai-tooling` (Iheb), one PR sets up shared AI tooling (D27, ADR 0006):
the `pr-prep` skill in `.claude/skills/pr-prep/SKILL.md` and one MCP server, Context7, in
`.mcp.json` with its key referenced as `${CONTEXT7_API_KEY}` (no token committed). GitHub MCP is
excluded from v1: read-only cannot be guaranteed at the token level for collaborators (ADR 0006).
No Filesystem, Git or Fetch MCP; no MCP in Stage 13. Scope agreed with Donia and Malak before the PR.

**Evidence [verified by Iheb]:** `python -m pytest -q` passes on the branch (33 passed); the new
files contain none of the governance-test patterns; `.mcp.json` is valid JSON; `.gitignore`
already covered `.env`, `.venv/` and `.claude/settings.local.json`, so it was not changed.

**AI assistance:** **Claude** (claude.ai chat) researched GitHub token and MCP read-only behaviour,
drafted ADR 0006, the `pr-prep` skill (with Anthropic's `skill-creator`), `.mcp.json` and these doc
edits. Iheb reviewed every file and ran all checks.

**Gate:** waiting for Donia's GitHub review. AI agents do not merge.

### Entry 28 — Stage 4 closeout: leakage-aware split merged (2026-10-17)

**What:** Stage 4 is closed. PR #12 (`feat/stage4-split`) was merged into `main` as merge
commit `865faf4` after human review and green CI. The approved implementation, deterministic
split evidence and project-log updates are now on `main`; the candidate assignment CSV remains
ignored and stored only at its recorded Drive path.

**Evidence [verified]:** local history contains merge commit `865faf4`; the merged reports record
17,721 unique paths, no group crossing splits, all eight checks passing, and byte-identical reruns
with candidate SHA-256 `853d947499e76c4cc9da5b69df384588f0023c21ed93f6a2dec1940f8988aa20`.
The Stage 5 branch starts from updated `main` at `d9907f3`; its baseline is 33 tests passing and a
clean worktree.

**AI assistance:** Codex helped implement and test the Stage 4 splitter, run the metadata-only
checks and prepare the evidence. Malak reviewed the outputs; the PR required human approval and
was merged by a human.

**Gate:** **PASSED. Stage 4 is complete.** Stage 5 may create the F2 manifests and hashes; Stage 6
has not started.

### Entry 29 — Stage 5 F2 implementation prepared (2026-10-17)

**What:** on `feat/stage5-dataset-freeze-malak`, added a metadata-only Stage 5 freeze generator,
`configs/freeze.yaml` and synthetic tests. It validates the exact Stage 1 inventory hashes and the
approved Stage 4 assignment hash, joins FoR hashes to the frozen split, builds the D20-permitted
external metadata manifest, writes deterministic CSVs and records each manifest hash. The official
run has not occurred and no Stage 5 output has been created or committed.

**Evidence [verified]:** the complete suite passes (`45 passed in 14.68s`). The focused Stage 5
suite passes (`12 passed in 0.85s`), and governance + splitting + freezing pass (`33 passed in
7.14s`). A no-write dry validation of the real metadata accepted 17,721 FoR rows and 64 external
rows with all 14 checks true. No audio was opened or decoded; Stage 6 did not start.

**AI assistance:** **Codex** read the frozen protocol and Execution Plan, drafted the Stage 5 code,
config and tests, validated the three metadata CSVs and prepared these documentation updates. Malak
must review the implementation; Donia reviews the eventual F2 PR. AI does not approve or merge.

**Gate:** implementation is ready for a human-approved commit. F2 remains open until a clean-commit
official run produces reviewed manifests and hashes that merge through a green PR.

### Entry 30 — Stage 5 official F2 run and deterministic evidence (2026-10-17)

**What:** after Malak approved the implementation commit, commit
`9e005b1e2b697bece73253b1871f2c98e0e5daa0` was created. Staging the evidence exposed
platform-dependent report line endings and Markdown trailing spaces, so Malak approved correction
commit `df1d6acab9df707b4702e038c45d5d22b38b8e66`. From that final clean commit, the official
metadata-only run produced the FoR and external file manifests, their hash ledger, and JSON/Markdown
reports. A second clean run from the same commit produced all five generated files byte-identically.
D28 records the proposed F2 hashes; it becomes effective only after PR approval and merge.

**Evidence [verified]:** FoR manifest: 17,721 unique paths, SHA-256
`badfe948fb7b833357783dd18be00f6bb36253cf8699dc4c4beb4df2b07706f8`; external manifest:
64 unique paths, SHA-256 `7c7a3e52ef254841f53bc08018bd3f9fe7ad83d6cd84531832513c616695355f`;
hash ledger SHA-256 `c9fc2e230c0883028459d0b2b8de4f6ee537676cd5a4e259e9d0f545c60943cc`.
All 14 manifest checks pass. The implementation suite passed 45 tests; the D20 governance test
passed 3 tests. No audio was opened or decoded, and Stage 6 did not start.

**AI assistance:** **Codex** implemented and tested the freezer, executed the approved metadata-only
run, compared both runs byte-for-byte, and prepared the evidence and proposed decision entry. Malak
approved the implementation commit. Donia must review the F2 artifacts; AI does not approve or merge.

**Gate:** **WAITING FOR DONIA'S HUMAN REVIEW.** F2 is not effective and Stage 5 is not complete until
the evidence commit is reviewed in a green PR and merged by a human. Do not begin Stage 6.

### Entry 31 — Stage 7 evaluation harness, steps 1–5 (2026-10-17)

**Authorship:** specified and reviewed by Donia Mabrouk. Code, tests and the design note were
written by **Claude Code**. Every commit's diff was reviewed by Donia with Claude (claude.ai) before
push.

**What:** on `feat/stage7-eval-harness`, the shared evaluation harness (`src/evaluation/`), built in
five steps:
1. EER (with threshold rule) and ROC-AUC, with input validation (`metrics.py`).
2. Decision-level metrics at a threshold: accuracy, precision, recall, F1, confusion matrix (`metrics.py`).
3. Three-way decisions with the uncertain band, coverage and deferral (`metrics.py`).
4. Cluster bootstrap confidence intervals, with limitations near 0 and `n_invalid` reporting (`bootstrap.py`).
5. Seed summary across CNN seeds and the bundle-seed rule (`seeds.py`).

Design note: `docs/evaluation_harness.md`. All tests use synthetic data; no dataset was accessed and
DEEP-VOICE was not touched.

**Dependency:** `scikit-learn==1.9.1` pinned in `requirements.txt` (already a transitive dependency
of librosa). The wheel check used manylinux_2_28 (manylinux2014 has no 1.9.1 wheels). Install on
Python 3.13 [verified] by CI on the PR; Colab install [to verify] at the first Stage 7 Colab run.

**Evidence [verified]:** `.venv` (Python 3.14.7) `python -m pytest -q`: `161 passed in 20.62s`;
governance test `3 passed in 0.28s`.

**Open protocol clarifications** found while building the harness, to decide before the data they
affect is seen (Protocol §12 change control; each gets the next free D-ID when decided):
1. Baseline threshold: its own val-B EER point, forced binary only (Donia, before Stage 7 results).
2. Exact CNN model-selection search space in `configs/` (Donia, before Stage 8).
3. Enforcement of "Stage 11 runs once" (run registry) (Donia, before Stage 11).
4. File-level decisions for DEEP-VOICE: apply the frozen band to the file mean of p, as a secondary
   view (Khalil, before Stage 13).

**Gate:** harness PR ready for human review; AI does not approve or merge.

---

### Entry 32 — Protocol clarifications D29–D31 (2026-10-17)

**Authorship:** decided by Donia Mabrouk. The decision-register rows, the Protocol §14 row and this
entry were written by **Claude Code** from Donia's wording; reviewed by Donia before push.

**What:** on `docs/stage7-protocol-clarifications`, three decisions in `docs/decisions/README.md`
and one Protocol §14 change-history row:
- **D29:** baseline threshold = its own val-B EER point, forced binary only.
- **D30:** CNN model-selection search space: conv blocks {3, 4} × learning rate {1e-3, 3e-4};
  search runs on seed 0, the chosen configuration then on seeds 0, 1, 2 (D11), reusing the seed-0
  run; tie rule 3 blocks before 4, then 1e-3 before 3e-4.
- **D31:** enforcement of "Stage 11 runs once".

**Why:** these gaps were found while building the evaluation harness (Entry 31). The protocol left
them open; no existing rule changed (Protocol §12 change control). Decided by Donia **before any
Stage 7–11 data is seen**.

**Still open:** file-level decisions for DEEP-VOICE (Khalil, before Stage 13); Colab install of
scikit-learn 1.9.1.

**Gate:** PR ready for human review; AI does not approve or merge.

---

### Entry 33 — Stage 10 risk policy, steps R1–R2 (2026-10-17)

**Authorship:** specified and reviewed by Donia Mabrouk. Code, tests and the design note were
written by **Claude Code**. Every commit's diff was reviewed by Donia with Claude (claude.ai) before
push.

**What:** on `feat/stage10-risk-policy`, the risk policy (design note `docs/risk_policy.md`):
- **R1:** the band rule moved from the harness to the core (`src/risk/decisions.py`; the harness
  re-exports it), and Platt calibration with smoothed targets, fitted by Newton's method
  (`src/risk/calibration.py`).
- **R2:** the fit in `src/tuning/risk_fit.py` (calibration, val-B EER threshold on p, largest band
  with deferral ≤ 10%); apply (confidence, risk level) and the strict JSON policy file in
  `src/risk/policy.py`. A follow-up fix rejects self-inconsistent policies (fit counts ≠
  calibration counts, or deferral on fit > target), in the constructor and through `from_json`.

**Why the layering:** SDD Fig. 1 puts Risk & Confidence in the API container but not Evaluation, so
the band rule and the policy application live in the core `src/risk`, which imports neither
`src.evaluation` nor `src.tuning`. The new `src/tuning` package is the SDD Training & Tuning
component (research tier); the fit lives there because it uses the harness `eer()`.

All tests use synthetic scores; no dataset was accessed and DEEP-VOICE was not touched.

**Evidence [verified]:** `.venv` `python -m pytest -q`: `207 passed in 10.99s`; governance test
`3 passed in 0.15s`.

**Watch item:** Platt's stopping rule is an absolute gradient norm (< 1e-10). Fine at val-B size;
on much larger fit sets it may not be reached, and the fit then fails loudly (`ValueError` after
100 iterations), never silently.

**Gate:** PR ready for human review; AI does not approve or merge.

---

### Entry 34 — Stage 11 run registry for D31 (2026-10-17)

**Authorship:** specified and reviewed by Donia Mabrouk. Code, tests and the design note were
written by **Claude Code**. Every commit's diff was reviewed by Donia with Claude (claude.ai) before
push.

**What:** on `feat/stage11-run-registry`, the run registry that enforces "Stage 11 runs once" (D31),
in `src/evaluation/run_registry.py` (design note: `docs/evaluation_harness.md`, "Run registry (D31)"):
- **Claim before run:** a `"started"` record is appended before the evaluation and a `"completed"`
  record after it; any record for the same (dataset, split, sorted model set) blocks a second
  claim, so a crashed run still blocks a rerun.
- **Hash chain:** each line carries the SHA-256 of the previous line's exact bytes; editing,
  deleting or reordering an earlier line is detected on load.
- **Rule replay on load and strict parsing:** a file the code would never write is rejected;
  unknown or missing keys and bad values raise `ValueError` naming the line.
- `*.jsonl -text` in `.gitattributes`, so Git never converts the registry's line endings (the
  chain hashes exact bytes).

All tests use temporary files; no dataset was accessed and DEEP-VOICE was not touched.

**Limitation and mitigation:** the hash chain cannot detect removal of the last line(s): a
truncated file is still a valid chain (`test_truncation_is_not_detected_by_the_chain` documents
it). Mitigation (procedure): right after Stage 11, the registry's SHA-256 and line count are
recorded in the project-log entry and the results PR, so a shortened file no longer matches.

**Evidence [verified]:** `.venv` `python -m pytest -q`: `230 passed in 12.68s`; governance test
`3 passed in 0.19s`.

**Gate:** PR ready for human review; AI does not approve or merge.

---

### Entry 35 — Stage 12 prep: ModelBundle format and verifying loader (2026-10-17)

**Authorship:** specified and reviewed by Donia Mabrouk. Code, tests and the design note were
written by **Claude Code**. Every commit's diff was reviewed by Donia with Claude (claude.ai) before
push.

**What:** on `feat/stage12-bundle-format`, the ModelBundle format and its verifying loader in
`src/bundle/bundle.py` (design note: `docs/model_bundle.md`):
- **Format:** one folder with the checkpoint, the preprocessing profile, `policy.json` and
  `bundle_manifest.json` (SHA-256 and version of each part, code commit, protocol version, F2
  ledger hash D28); strict manifest parsing, deterministic bytes.
- **Verify before use:** `load_bundle` checks folder contents, every SHA-256 and the policy version
  before returning anything, and returns `manifest_sha256` for the F4 freeze decision.
- **Path safety:** component files must be plain names inside the bundle, checked before any file
  is opened. **Immutability:** `build_bundle` never writes into a non-empty folder; `load_bundle`
  only reads.
- **Follow-up fix:** `policy.json` (like the manifest) is read once, so the parsed bytes are the
  hashed bytes; a failed build removes every file it wrote (and the folder it created).

**Why:** threat model §10 (tampering: model, profile or thresholds replaced) and NFR-09 require a
refusal at load; the bundle is the C→D (Stage 13) and C→E (API) hand-over of Protocol v1.0 §11,
so both consumers must get exactly the frozen parts (D2).

All tests use dummy files; no dataset was accessed, no PyTorch is needed, and DEEP-VOICE was not
touched.

**Limitations:**
- The profile version is recorded but not checked against the profile file; added in Stage 12
  once the F3 profile format is frozen.
- Check-then-use: the checkpoint and profile are verified by path; Stage 12 and the API (Stage 17)
  must load the checkpoint from the verified bytes or re-verify its hash at load.
- A pickled PyTorch checkpoint can execute code when loaded: Stage 8 saves checkpoints in a
  non-executable form (safetensors, or a `state_dict` loaded with `torch.load(...,
  weights_only=True)`), and Stage 12/17 load them that way.

**Evidence [verified]:** `.venv` `python -m pytest -q`: `266 passed in 13.34s`; governance test
`3 passed in 0.25s`. The two new tests fail against the pre-fix loader.

**Gate:** PR ready for human review; AI does not approve or merge.

---

### Entry 36 — Stage 5 closeout: F2 effective (2026-10-17)

**What [verified]:** Stage 5 is complete. Donia reviewed and approved PR #15, which merged into
`main` as `89b37fc` on 2026-10-17. D28 is now approved, the reviewed F2 manifests and hash ledger are
effective, and the roadmap advances to Stage 6 as the current stage. Entry 30 remains unchanged.

**Evidence [verified]:** local Git history identifies `89b37fc` as
`Merge pull request #15 from kahlanzzi/feat/stage5-dataset-freeze-malak`, dated 2026-10-17. The merge
is contained by `main`. The Stage 5 implementation, correction and deterministic evidence are the
artifacts reviewed in PR #15.

**AI assistance:** **Codex** checked the merge evidence and prepared this documentation-only closeout.
Malak reviews the diff; Donia reviews the closeout PR. AI does not approve or merge.

**Gate:** **PASSED. Stage 5 is complete and F2 is effective.** Stage 6 may begin on its own branch;
this closeout branch contains documentation only.

---

## Open items

**GitHub / repo**
- [x] PR `stage-0-git-workflow`: merged as PR #1 (approved and merged by Malak, 2026-10-05).
- [x] Iheb and Malak accept the collaborator invitations (accepted; Entry 14).
- [x] Branch protection on `main` is active: PR + 1 approval + required CI check `checks` + branch up to date; no direct pushes (confirmed, Entry 18).
- [x] `.gitattributes` (LF line endings for Windows + Colab): added on `chore/repo-structure` (Entry 17).
- [x] Transfer repo ownership to Iheb after Part A (Stage 3 merged); then everyone runs `git remote set-url` and one PR updates `REPO_URL` in `README.md`, `AGENTS.md`, `docs/setup.md` and the notebook (team decision, 2026-10-12): done early, Entry 20.
- [ ] Iheb confirms collaborators, branch protection and Actions settings on `kahlanzzi/voice-spoofing-detection` (Entry 20).
- [x] `stage-1-audit-run` → PR + merge at the end of Stage 1 (PR #2, 2026-10-05).
- [x] Merge PRs in order: `stage-0-git-workflow` → `stage-1-audit-run` → `stage-2-threat-model` (stacked; see Entry 9). Done as PR #1 → #2 → #3. #3 landed on `stage-1-audit-run`, so it was re-delivered as PR #4 (Entry 14).
- [x] Delete merged remote branches (`stage-1-audit-run`, `stage-2-threat-model`, `stage-2-into-main`): deleted after the content check and team OK (Entry 16).
- [x] Review and merge PR `chore/repo-structure` (formerly `chore/ai-context`; Entries 15–17): merged as PR #6 (Entry 18).
- [x] Review and merge PR `docs/stage3-repo-cleanup` (Entry 18): merged as PR #7 (Entry 19).
- [x] Delete merged branches: remote has only `main` (Entry 19).
- [x] Review and merge PR `docs/stage3-protocol-skeleton` (Entry 19): merged as PR #8; branch deleted (Entry 20).
- [ ] Review and merge PR `chore/stage5-ai-tooling` (Entry 27); after merge each member sets `CONTEXT7_API_KEY` locally (never committed) and approves the project MCP server in their own tool.
- [ ] Stage 13 governance: decide (new decision, before Stage 13) how the guarded external-run code is added to the DEEP-VOICE allowlist of `tests/governance/test_deepvoice_access.py`.
- [x] `LICENSE` (MIT, code only, D25) and the README statement that datasets are not redistributed: in `chore/repo-structure` instead of a separate PR B (Entry 17).
- [x] Decide whether branch protection should require code-owner review: **no** (review of Entry 16).
- [x] Compare `CONTRIBUTING.md`, the `.github/` templates, `docs/decisions/TEMPLATE.md` and ADR 0001 with the actual CS 488 template and align them (keep D20 checkbox, stacked PRs, D23 naming, human in the loop): done on `chore/repo-structure` (Entry 17).

**Stage 3 gate**
- [x] Each owner fills in their sections of `docs/experimental_protocol.md`: replaced by one integrated PR from the team-accepted defaults (2026-10-14, Entry 21).
- [x] Review of `docs/stage3-protocol-v1` and sign-off by all four (§13 dates) → v1.0 = freeze F1. Then mark Stage 3 done in `AGENTS.md`: PR #10 (Malak 2026-10-14, Khalil 2026-10-14, Iheb 2026-10-14); Stage 3 done (Entry 21).
- [ ] Before Stage 17: decide the online handling of clips longer than 2 s (reject, truncate or window-and-aggregate; open in the SDD).

**Stage 4 gate**
- [x] Implement the FoR-only group-aware split and automated checks from Protocol v1.0 §2 (Entry 24).
- [x] Run it from clean commit `10a5e68` against the full pinned Stage 1 inventory and summary;
  all checks pass and the rerun is byte-identical (Entry 25).
- [x] Malak uploaded the untracked candidate CSV to the exact Stage 4 Drive destination (Entry 26).
- [x] Donia reviewed the corrected evidence; PR #12 merged as `865faf4` and Stage 4 closed (Entry 28).

**Stage 5 gate**
- [x] Record the Stage 4 closeout before beginning Stage 5 (Entry 28).
- [x] Implement and test the metadata-only manifest freezer (Entry 29).
- [x] Malak approved implementation commit `9e005b1` and correction commit `df1d6ac`; run the
  official freeze twice from the final clean commit.
- [x] Record the generated manifest hashes in proposed D28 and project-log Entry 30.
- [x] Donia reviewed and approved the F2 PR; PR #15 merged as `89b37fc`, Stage 5 is done and F2 is
  effective (Entry 36).

**Stage 7 harness: protocol clarifications (Entry 31; Protocol §12, next free D-ID when decided)**
- [x] Baseline threshold: own val-B EER point, forced binary only (Donia, before Stage 7 results): D29 (Entry 32).
- [x] Exact CNN model-selection search space in `configs/` (Donia, before Stage 8): D30 (Entry 32).
- [x] Enforcement of "Stage 11 runs once" (run registry) (Donia, before Stage 11): D31 (Entry 32).
- [ ] File-level decisions for DEEP-VOICE: frozen band on the file mean of p, secondary view (Khalil, before Stage 13).
- [ ] Colab install of scikit-learn 1.9.1 [to verify] at the first Stage 7 Colab run.

**Stage 10 risk policy (Entry 33)**
- [x] Human review of `feat/stage10-risk-policy` (R1–R2); CI green on Python 3.13: merged as PR #17.
- [ ] Watch: Platt's absolute gradient-norm stop; if the real val-B fit fails to converge, propose a
  relative criterion (no silent change).

**Stage 11 run registry (Entry 34)**
- [x] Human review of `feat/stage11-run-registry`; CI green on Python 3.13: merged as PR #18.
- [ ] Stage 11: record the registry SHA-256 and line count right after the run (project-log entry
  and results PR).

**Stage 12 ModelBundle (Entry 35)**
- [ ] Human review of `feat/stage12-bundle-format`; CI green on Python 3.13.
- [ ] Stage 12: add the profile-version check once the F3 profile format is frozen.
- [ ] Stage 12/17: load the checkpoint from verified bytes (or re-verify at load).
- [ ] Stage 8: save checkpoints in a non-executable form (safetensors or state_dict + weights_only=True).
- [ ] F4: record manifest_sha256 in the freeze decision.

**Notebook (ported to the repo, `f8af88a`; pins and guards added in Entry 7)**
- [x] Cell 1: real `REPO_URL`; clone with `-b <branch>`; print Python version and commit hash; capture and print the `pytest` output.
- [x] Cell 2: download FoR to local disk (not Drive); print the folder/label structure; back up as one `.tar` + `REVISION.txt` in Drive.
- [x] Add the dry-run, full-run, and codec-check cells as used in the session.
- [x] Port Colab cells 1 / 2 / 3 and the run cells (FoR audit, codec check, DEEP-VOICE metadata audit, FoR ∩ DEEP-VOICE overlap check) to `notebooks/01_stage1_dataset_audit.ipynb` (9 cells, outputs stripped; `BRANCH` defaults to `main`; `deepvoice.audit_subdir` added to `configs/audit.yaml`).
- [x] Run the ported notebook once in Colab (cells 1, 6, 8, 9 at `c2136cb`; results identical, Entry 8). Archive-only read path exercised on synthetic data only.
- [ ] Optional: exercise the archive-only path of cells 6 and 8 on the real Drive archive in a fresh Colab session (no local outputs).

**Security**
- [x] Regenerate the Kaggle API token (done 2026-10-12; old token revoked; stored only in local `.env`, git-ignored).

**Course deliverables**
- [ ] Setup report (`docs/setup-report.md`, Lab 6 skeleton): Malak's Codex test is recorded and
  passed (Entry 22); the remaining sections and team members' AI-tool answers are still open.

**Local verification**
- [x] Re-run the full local suite: Python 3.13 completed with **33 passed in 6.36 s** during Stage 4;
  the Windows Application Control failure from Entries 22–23 did not recur (Entry 24). The eventual
  PR still requires its own green CI check before merge.
- [ ] Optional: decide whether to harden audit error reporting. If approved, use a separate
  `fix/stage1-audit-error-propagation` branch (Entry 23).

**Documentation**
- [ ] `docs/dataset_audit.md` §9: confirm Iheb and Malak really re-confirmed D14 on 2026-10-04 [to verify]; correct the line if not.
- [x] Licenses recorded in `docs/dataset_audit.md` §1: FoR not stated by the HF repackager (original FoR terms apply); DEEP-VOICE MIT (Kaggle description), with attribution. Ethics / rights limitation added to §8.
- [ ] Check the original FoR source (Reimao & Tzerpos, 2019) for its stated terms of use.
- [x] HF dataset card checked 2026-10-04: empty; viewer shows only `audio` and `label`. No speaker / source / generator metadata; D14 stands.
- [x] Remaining TBDs in `docs/dataset_audit.md` filled: run by Donia Mabrouk, DEEP-VOICE audit commit `ae45e9f`, download dates 2026-10-04, Drive locations of full outputs (`processed/metadata/stage1/`) and of the DEEP-VOICE zip (`raw/deep-voice-deepfake-voice-recognition.zip`). D5 wording amended.
- [x] Codec-check and FoR ∩ DEEP-VOICE overlap outputs committed to `docs/audit_evidence/` (`for2sec_codec_check.csv/.json`, `cross_dataset_overlap.json`); "not yet archived" notes replaced in `docs/dataset_audit.md`.
- [x] Log D13, D14, D9, D16, H5, H6 in `docs/decisions/README.md` and fill in `docs/dataset_audit.md` (done, with D17–D19).
- [ ] Update Proposal and Execution Plan discrepancies after the audit (e.g., the DEEP-VOICE 628 / 4,425 counts).
  Also, from Entry 12:
  - DEEP-VOICE size 3.96 GB (Exec Plan: ~0.55 GB);
  - the Proposal architecture figure (ASVspoof + WaveFake; "baseline + deep model" in the detector);
  - Proposal §3.2 streaming as a stretch goal (vs SRS WH-01);
  - Exec Plan's six named TTS engines (not determinable);
  - Exec Plan steps superseded by D2, D3, D6, D10.

**Stage 1 gate**
- [x] Team sign-off of `docs/dataset_audit.md` §10: signed off by Malak (PR #2, 2026-10-05); Donia author; Iheb not yet reviewed.

**Team meeting (next)**
- [x] Collaborator invitations accepted (Iheb, Malak).
- [x] Stage 1: `docs/dataset_audit.md` §10 sign-off (Malak, PR #2).
- [x] Stage 2: threat model §14 sign-off (Malak PR #3, Iheb PR #4). §13 open questions still open (see Stage 2 gate).
- [x] Assign the three tracks (0.9): done via D26 (parts A–E with owners, Entry 18).
- [ ] Confirm compute per person (0.7), set the weekly sync + channel (0.8).
- [ ] Set up W&B (0.6) before Stage 8.
- [x] Agree the PR merge order: `stage-0-git-workflow` → `stage-1-audit-run` → `stage-2-threat-model` (merged; Entry 14).

**Execution Plan phases (from the Entry 12 comparison)**
- [ ] Phase 0: compute per person (0.7); weekly sync + async channel (0.8). Random-seed policy (0.10): set in protocol v1.0 §4 (CNN seeds 0, 1, 2; split and bootstrap seed 2026; Entry 21), final (F1).
- [ ] Phase 1: GitHub Issues board per the Execution Plan (Phase 1, task 5): one issue per SRS requirement (FR / NFR / PC IDs), tagged by part (A–E) and owner (D26), grouped under one milestone per stage (Entry 13).
- [ ] Phase 2: compute benchmark before Stage 8. Train the CNN on a small subset and record GPU type, VRAM, batch size and time per epoch.

**Stage 2 gate**
- [x] Resolve the two v0.1 mismatches from Entry 10 (§3.2 REAL filenames; §10 D20 → D21) → v0.2 (Entry 11).
- [x] Team review of `docs/threat_model.md` v0.2 → **v1.0 reviewed** (§14: Malak PR #3, Iheb PR #4).
- [x] Threat model §13 Q1 (primary / secondary scenarios) and Q3 (English-only): answered in protocol v1.0 §0 (scenarios confirmed; "English speech" stated as [doc], not [verified]); final (F1, Entry 21).
- [ ] Threat model §13 Q2: access control (below).
- [ ] Decide the access-control mechanism (SRS §7.4) before the API stage (Stage 17), at the latest. The oracle-abuse residual risk (threat model §10.1) depends on it.

**Later stages (reminders)**
- [ ] `configs/robustness.yaml` → Stage 14. `docker-compose.yml` → Stage 17.
- [ ] Stage 17: verify the ModelBundle checkpoint hash at load (threat model §10, tampering recommendation).
- [ ] Stage 6: re-run the shortcut screen *after* preprocessing (silence, bandwidth).
- [ ] Stage 6: decide the silence / voice-activity detector explicitly (energy-based vs `silero-vad`) and record it as a decision; D17 depends on it.
- [ ] Phase 0 leftovers: confirm ML lead (0.9), W&B project (0.6). `requirements.txt` pinned (Entry 17); confirm the 3.13 install on the first CI run of the PR.

---

## Current status — updated 2026-10-17

**What was done:**
- **Stages 0–3 and Part A are complete.** Protocol v1.0 is frozen as F1 (PR #10).
- **Stage 4 is complete** (Entry 28): PR #12 merged as `865faf4`.
- **Stage 5 is complete and F2 is effective**: Donia approved PR #15, merged as `89b37fc`; D28 is
  approved (Entry 36).
- **Stage 7 evaluation harness, steps 1–5, is merged** as PR #13 (Entry 31).
- **Protocol clarifications D29–D31 are merged** as PR #16 (Entry 32).
- **Stage 10 risk policy, steps R1–R2, is merged** as PR #17 (Entry 33).
- **Stage 11 run registry (D31) is merged** as PR #18 (Entry 34).
- **Stage 12 prep, ModelBundle format and verifying loader, merged** as PR #19 (`8b688dc`; Entry 35).
  The **Part C prep is complete**: harness, risk policy, run registry and bundle format are built and
  tested on synthetic data; the real Part C runs wait for F3 (preprocessing profile v1).

**What we have:**
- The approved FoR split and the F2 manifests and hash ledger (D28).
- One shared, tested evaluation harness (`src/evaluation/`) for the baseline, the CNN and the
  external run, documented in `docs/evaluation_harness.md`.
- Pre-declared rules for the baseline threshold, CNN selection and the single Stage 11 run (D29–D31).
- A tested risk-policy fit (`src/tuning`) and core apply / save / load (`src/risk`), documented in
  `docs/risk_policy.md`; not yet fitted on real val-B scores.
- A tested append-only run registry (`src/evaluation/run_registry.py`) that refuses a second
  Stage 11 run for the same model set.
- A tested ModelBundle format (`src/bundle/`) whose loader verifies every part before use,
  documented in `docs/model_bundle.md`.
- Governance rules that keep the external set metadata-only until the one guarded Stage 13 run.

**Branches:**
- `main`: through PR #19 at `8b688dc`.
- `docs/stage5-closeout`: Stage 5 administrative closeout; documentation only.

**What we want:**
- A model that detects **synthetic voices**, not dataset accidents, followed by one honest external test.

**What's next:**
1. Human review of this Stage 5 closeout PR; CI must be green on Python 3.13. AI does not merge.
2. **Stage 6 is current:** preprocessing profile v1 + tests + compute benchmark, on a separate branch.
3. Khalil decides the DEEP-VOICE file-level view before Stage 13; verify the scikit-learn Colab install.
4. Then run the real Part C work on the frozen F3 preprocessing profile.
