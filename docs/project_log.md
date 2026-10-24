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

### Entry 37 — Stage briefs: structure, template and D32 (2026-10-17)

**Authorship:** proposed and specified by Donia Mabrouk. The files were written by **Claude Code**;
Donia reviews before push.

**Why:** Donia's proposal. Execution Plan v2.1 §12 asks that everyone can explain the full pipeline
end to end, not only their own part; with four owners and stages handed over between them, each
member (and the professor) must be able to answer questions on any stage.

**What:** on `docs/stage6-stage-briefs` (docs only, no code):
- `docs/stages/TEMPLATE.md`: header block (stage, owner, reviewer(s), status, date, related PRs) and
  8 sections with guidance and target lengths (2–4 pages in total).
- `docs/stages/README.md`: purpose, how briefs differ from the log, the register and the design
  notes, the process (owner writes at stage close; teach-back review; 5-minute teach-back by a
  member who did not write the stage; Stage 19 report assembled from the briefs), a Mermaid overview of Stages 0–19 with
  the four D26 hand-overs, and the table of all 20 brief files.
- D32 (proposed) in `docs/decisions/README.md`; one line each in `CONTRIBUTING.md` §7, `AGENTS.md`
  (Working mode) and the `docs/README.md` index; CHANGELOG line.

**Not done:** no stage brief is written yet. First briefs: Donia (Stages 0–3), Malak (Stages 4–5).

**Evidence [verified]:** `.venv` `python -m pytest -q`: `266 passed in 13.52s`; governance test
`3 passed in 0.23s` (docs only; no test changed).

**Gate:** PR ready for human review; D32 is approved by the team review of this PR. AI does not
approve or merge.

---

### Entry 38 — Stage briefs 00 and 01 written (2026-10-18)

**Authorship:** requested and specified by Donia Mabrouk. Both briefs were written by **Claude Code**
(Claude Opus 5.5), an AI tool, from the repository documents only; Donia reviews before push.

**What:** on `docs/stage6-briefs-00-01` (docs only, no code), from `docs/stages/TEMPLATE.md` (D32):
- `docs/stages/stage00-setup.md` (owner Donia; reviewer Iheb, repo owner; PRs #1, #6, #7).
- `docs/stages/stage01-dataset-audit.md` (owner Donia; reviewer Malak, whose split builds on the
  audit; PR #2).
- Each has all 8 sections; every number cites its file and section or log entry; section 7 answers
  the questions set by Donia (6 for Stage 0, 8 for Stage 1). No DEEP-VOICE path, slug or folder name
  is written.
- `docs/stages/README.md`: both briefs → "in review". CHANGELOG line.

**Not recorded in the sources:** where the Proposal's DEEP-VOICE counts 628 / 4,425 come from
(stated as "not recorded" in the Stage 1 brief, Q8).

**Evidence [verified]:** `.venv` `python -m pytest -q` (Python 3.14.7): `266 passed`;
`tests/audit` + `tests/governance`: `15 passed`.

**Gate:** briefs in review; teach-back review by Iheb (Stage 0) and Malak (Stage 1) (D32). AI does
not approve or merge.

---

### Entry 39 — Khalil local setup and AI-tool answer (2026-10-17)

**What we did:** Khalil prepared the Windows checkout and virtual environment on `docs/stage0-lab6-closeout`. Updated `docs/setup-report.md` with his setup evidence, command-resolution problems and Codex answer.

**Evidence:** Khalil supplied terminal screenshots showing Python 3.13.16, Git 2.56.0.windows.2 and `266 passed in 299.59s (0:04:59)` [verified from screenshots]. This is the earlier local run, not a new test run by Codex and not GitHub CI evidence.

**AI assistance:** Codex guided setup, inspected local project instructions and drafted documentation. The AI-tool question was answered in the existing conversation; fresh-session automatic context loading was not tested. Teammate review remains pending.

**Open issues:** Remaining report sections and team AI-tool responses; Iheb's audit review and explicit D14 response; passing PR CI and human review. No approval or stage closure is claimed.

**Gate:** Lab 6 closeout remains incomplete; waiting for remaining evidence and team review.

### Entry 40 — Lab 6 report evidence and remaining closeout checks (2026-10-18)

**What we did:** Codex filled the team, repository history, environment, CI, workflow and data-governance sections of `docs/setup-report.md` using the existing decision register, contribution guide, scripts and workflow. Existing member answers and local setup evidence were preserved.

**Evidence:** PR #21's GitHub UI showed one successful check for the draft ending at `faecd8c` [verified]. This is distinct from Khalil's earlier local test screenshots. The report marks historical/configuration statements as documentation and does not claim current owner settings or missing human sign-offs were verified.

**AI assistance:** Codex inspected repository files and the PR page and drafted these documentation updates. No model or dataset operation was performed.

**Open issues / gate:** Remaining member answers, Iheb's audit review and D14 answer, owner-settings confirmation, setup/access checks, the AI-test-method review and final PR checks remain pending. Lab 6 is not closed.

---

### Entry 41 — Donia AI-tool test evidence for Lab 6 (2026-10-18)

**Evidence (member-reported):** Khalil relayed Donia's fresh-session Claude Code test on `main`, dated 2026-10-18. Donia reported a passed answer on the external-data restrictions and preprocessing location, with `CLAUDE.md` importing `AGENTS.md`. The extension version was not supplied. Codex did not observe that session independently.

**What changed / AI assistance:** Codex transcribed the summary into `docs/setup-report.md` §5 and updated its outstanding checks. Donia's answer is collected; her extension version, Iheb's answer and the other Lab 6 checks remain pending. No stage closure is claimed.

---

### Entry 42 — Per-speaker analysis preparation (2026-10-18)

- **Owner:** Khalil (D26). AI assistance: Codex implemented the helper, synthetic tests and documentation; human review is pending.
- **Scope:** FR-12, D4, D19. Added `src/evaluation/speakers.py`: per-source-speaker EER/AUC, forced-binary and three-way metrics through the existing harness. All speakers use the caller's fixed threshold and band. Undefined metrics remain NaN and no speaker is silently dropped.
- **Evidence [verified]:** 25 focused synthetic tests passed in 1.53 s in a temporary workspace using the project's Python environment. After installation, the full repository suite passed: **291 passed in 20.94 s**, including the governance tests. Tests used a writable temporary directory and disabled pytest cache/bytecode writes because of the app sandbox. Git diff checks from the app shell were unavailable (Git reported that it must run in a work tree); human Git review is pending.
- **Data:** invented scores only; no audio access, model execution, threshold fitting or external run.
- **Open:** speaker-average summary, clustered CIs, filter reporting, file-level decision and external-run allowlist decision remain. Lab 6 PR #21 still awaits team evidence/review.
- **Gate:** preparation only; WAITING FOR APPROVAL. Stage 13 has not run.

---
### Entry 43 — Analysis PR and changelog correction (2026-10-18)

- Codex created draft PR #23 at Khalil's request, based on `docs/stage0-lab6-closeout` (PR #21); one analysis commit and five files at creation. Nothing merged.
- [verified] Khalil's pre-PR governance check: **3 passed in 0.22s**; token-pattern search returned no matches. Full-suite evidence is in Entry 42.
- Codex removed its accidentally duplicated speaker-analysis changelog line from the historical 2026-10-06 section, keeping the Unreleased entry. This correction changes documentation only; no code or datasets changed.
- Open: commit/push this correction, verify GitHub CI, human review and change the PR base to main after #21 merges. Stage 13 remains preparation only; WAITING FOR APPROVAL.

---

### Entry 44 — Per-speaker PR integration with main (2026-10-18)

**What changed:** PR #21 merged as `65ce1d2`; PR #23 now targets main. Codex resolved the log conflict,
kept merged Entries 1–41 intact and renumbered the unmerged analysis entries to 42–43, including
their internal reference. The changelog retains both main's additions and the per-speaker entry.

**Review follow-up:** `speakers.py` imports `_validate` through the evaluation harness re-export,
as Donia suggested. This retains the same validator. The undefined-value convention is listed as
an open pre-Stage 13 decision together with the file-level view and governance allowlist.

**Evidence [verified]:** Full suite after the resolution and import cleanup: `291 passed in 11.20s`,
including governance tests; writable temporary directory, pytest cache and bytecode writes disabled.

**AI assistance / gate:** Codex prepared these changes at Khalil's request. Commit/push and updated
CI remain pending; PR #23 is not merged. Synthetic preparation only; no external run or new
evaluation-policy approval.

---

### Entry 45 — Speaker-average helper preparation (2026-10-18)

- Owner Khalil (D26); Codex prepared the D4/D19 equal-speaker summary, 20 synthetic tests and usage documentation. Intended branch: `feat/stage13-speaker-average`, stacked on PR #23.
- Explicit caller-selected undefined policy: propagate NaN with speaker IDs/counts, or raise. No silent exclusion or zero replacement. The external report convention remains for team decision/review; frozen protocol unchanged.
- [verified] Existing repository tests plus the 20 prepared tests: **311 passed in 11.80s**. The new module was loaded from the preparation workspace; repository installation and normal full-suite verification remain pending.
- Git operations from the app failed with work-tree/permission errors. A guarded installer creates the new branch and copies the files only if the starting branch is clean and the source documents are unchanged.
- No audio or model access. No confidence intervals or real external results produced. Human review and final reporting convention pending; WAITING FOR APPROVAL.

---

### Entry 46 — Speaker-average integration with main (2026-10-18)

**What changed / AI assistance:** After PR #23 merged as `3f2f4bb`, Codex prepared the integration
of main into PR #24 at Khalil's request. Merged Entries 1–44 are preserved; the unmerged
speaker-average preparation entry is now 45. Current status and open items are refreshed.
The changelog retains both main's additions and the equal-speaker summary entry.

**Evidence:** The helper was installed and pushed as `3f88cc8`; Khalil's earlier terminal evidence
showed `311 passed in 9.77s`. The updated full suite passed: `311 passed` (including governance tests), with a writable temporary
directory and pytest cache/bytecode writes disabled. No audio or model was accessed.

**Gate:** PR #24 integration, push, base change to main and updated CI remain pending.
The report's undefined-value convention, file-level view and governance allowlist still require
explicit decisions before Stage 13; this integration approves none of them.

---

### Entry 47 — Stage briefs 02 and 03 written (2026-10-18)

**Authorship:** requested and specified by Donia Mabrouk. Both briefs were written by **Claude Code**
(Claude Opus 5.5), an AI tool, from the repository documents only; Donia reviews before push.

**What:** on `docs/stage6-briefs-02-03` (docs only, no code), from `docs/stages/TEMPLATE.md` (D32),
in the format of the merged briefs 00 and 01:
- `docs/stages/stage02-threat-model.md` (owner Donia; reviewer Khalil; PRs #3, #4, #5).
- `docs/stages/stage03-protocol.md` (owner Donia, lead; reviewers Iheb, Malak, Khalil; PRs #7, #8,
  #10, #16), with a Mermaid diagram of the freeze points F1–F4.
- Each has all 8 sections; every number cites its file and section or log entry; section 7 answers
  the 8 questions set by Donia. The threat model's base-rate numbers are stated as illustrative, not
  measured. No DEEP-VOICE path, slug or folder name is written.
- Code claims in the Stage 3 brief were checked in the code: `select_bundle_seed()`,
  `fit_risk_policy()`, `three_way_decisions()`, `RISK_LEVELS`, `RunRegistry.claim()`, `load_bundle()`.
- `docs/stages/README.md`: briefs 00 and 01 → "accepted" (merged as PR #25, `610a412`, after
  teach-back review); 02 and 03 → "in review". CHANGELOG line.
- stage00 and stage01 headers set to Accepted (PR #25).

**Value rationale:** the reasons for the main "team choice" values (10% maximum deferral, ≥ 5 points
for H5/H6, 2,000 bootstrap resamples, robustness levels) were written down on 2026-10-18 in
`stage03-protocol.md` §4 ("Why these values"), as team reasoning; values unchanged, no data seen.
The "~200 FAKE_other test clips" figure is taken from threat model §8 T5.

**Evidence [verified]:** `.venv` (Python 3.14.7): `tests/governance` → `3 passed`; seeds, run
registry, risk, tuning and bundle tests → `131 passed`; full suite in the PR report.

**Gate:** briefs 02–03 in review; teach-back review by Khalil (Stage 2) and Iheb, Malak, Khalil
(Stage 3) (D32). AI does not approve or merge.

---

### Entry 48 — Stage 13 decision proposals (2026-10-18)

**What / AI assistance:** At Khalil's request, Codex checked the decision register, frozen protocol,
metric helpers and governance scanner, and drafted proposed D33–D35 with ADR 0007. Topics:
undefined eight-speaker averages, file-level decisions using the frozen CNN policy, and two exact
future runner/config allowlist entries with runtime guard requirements. All approvals are pending.

**Evidence:** PR #24 merged as `6786c12`; PRs #21 and #23 are also merged. The current branch starts
from that main revision. This change edits documentation only; no dataset or model access,
allowlist change, scoring implementation or protocol amendment. Full suite: `311 passed in 10.91s`
(including governance tests; writable temporary directory, cache/bytecode writes disabled).
Documentation whitespace and conflict-marker checks passed.

**Gate:** Team review of each proposal before adoption. Stage 13 has not run and is not authorized.

---

### Entry 49 — Donia review corrections to proposed D33/D35 (2026-10-18)

**Source:** Khalil relayed Donia's conditional acceptance of D33–D35 with three corrections.
Codex checked the bootstrap documentation and revised ADR 0007 and the proposed register rows:
existing invalid-resample exclusion with `n_invalid` reported per interval; dataset-wide external
single-run key across all model sets; claim durable beyond Colab before audio access and registry
SHA-256/line count recorded immediately after the run. D34 is unchanged.

**Open / gate:** Donia's approval row stays pending until she confirms the revised text. Her
CNN-versus-baseline external comparison proposal is a separate decision; it is not approved here.
No protocol, runtime code or allowlist changed. Links, whitespace and log numbering checked;
D34 and approval rows verified unchanged. Full suite: `311 passed in 11.48s`, including governance
tests (writable temporary directory; pytest cache and bytecode writes disabled).

**AI assistance:** Codex prepared the review corrections and this append-only log entry.

---

### Entry 50 — Reconcile Stage 13 proposals with main (2026-10-18)

**What:** Codex resolved the project-log conflict while integrating `origin/main` at `3ecb979`
into PR #26. Preserved merged Entries 1–47, including Donia's briefs entry. Renumbered only
this PR's unmerged proposal and review-correction entries to 48–49, and updated their current
references. Retained both branches' changelog additions. Current status and branch tracking
now reflect the merged briefs and the pending stacked PR #27 update.

**Checks [verified]:** full local suite `311 passed in 11.39s` (including governance);
merged log entries preserved, sequential numbering and conflict-marker checks passed.

**Gate:** D33–D35 remain proposals; no approval is inferred from this reconciliation.
No implementation, dataset access or external run. Human review and CI remain required.

---

### Entry 51 — Synthetic reporting integration tests (2026-10-18)

**What / AI assistance:** Codex prepared six integration tests at Khalil's request, using invented
scores for eight speakers, unequal sizes, 1,870 REAL and 13,090 FAKE windows, seed 2026, and one
fixed example threshold/band. Known error/deferral counts verify the per-speaker table; an
independent count-level oracle verifies clustered accuracy intervals with 2,000 resamples.
Tests also cover macro versus pooled accuracy, artificial filter accounting and missing-speaker
roster comparison. No dataset, audio, trained model or real prediction file is used.

**Evidence:** Prepared tests: `6 passed in 1.71s`. Full installed suite: `317 passed in 11.57s`, including governance tests
(writable temporary directory; pytest cache and bytecode writes disabled).
**Limitations:** These are integration tests of existing helpers, not a production reporting
layer, eight-speaker enforcement, a file-level implementation or real model performance.
D33–D35 remain proposals in PR #26. Tests do not select their final conventions.
**Branch:** requested rename from `test/stage13-synthetic-reporting` (Codex's incorrect prefix)
to `chore/stage13-synthetic-reporting` under D23; stacked on the pending proposal branch.
**Gate:** Review, branch rename, commit/push and CI pending. No external run authorized.

---

### Entry 52 — Reconcile synthetic reporting tests with main (2026-10-18)

**What / AI assistance:** Codex resolved the project-log conflict for PR #27 after PR #26
merged as `72a9646`. Preserved merged Entries 1–50 and renumbered only the unmerged
synthetic-test entry to 51. Both changelog additions remain. The branch rename and push
recorded as pending in Entry 51 are now complete; current tracking reflects that correction.

**Checks [verified]:** full local suite `317 passed in 12.04s`, including governance.
Merged entries preserved; numbering is sequential; no conflict markers or trailing whitespace.

**Gate:** Commit/push, retarget PR #27 to main and updated CI/review remain pending.
D33–D35 remain proposals. No production code, real predictions or external data accessed.

---

### Entry 53 — Proposed D37: baseline tuning, scaling and score (2026-10-18)

**What / AI assistance:** Donia proposed D37, drafted with Claude Code (docs only). While preparing
the Stage 7 baseline code, a gap was found: Protocol §4 left the RBF-SVM grid as "team choice" and
did not fix feature scaling or which score feeds the D29 threshold. D37 fixes them: train-only
standardization of the 40-value MFCC vector (scaler stored in the baseline artifact), the 16-point
C × gamma grid (gamma 0.025 equals scikit-learn's 'scale' on the standardized inputs, written
explicitly so the grid stays ordered and reproducible), selection by lowest val-A EER with a tie rule, and the continuous `decision_function`
as the score. Recorded in `docs/decisions/README.md` and as a Protocol §14 row; §0–§13 unchanged.
Minor decision, no ADR (D24). D36 is not on main; D37 is used as requested, so D36 stays free
for the decision expected to take it.

**Evidence:** Decided before any MFCC feature or val-A result exists (F3 is not frozen; no baseline
has been trained). No code, config, dataset or model access in this change.

**Gate:** D37 is approved by team review of this PR. Grid and scaling go into `configs/` before
any baseline training run.

---

### Entry 54 — Stage 7 MFCC/SVM baseline code, steps B1–B2 (2026-10-18)

**What / AI assistance:** Donia, with Claude Code (code, tests and docs drafted by the AI; human
review pending), on `feat/stage7-baseline-code`, stacked on `docs/stage6-d37-baseline-tuning`.
- **B1** (`src/models/baseline.py`): train-only scaler, NumPy RBF scorer equal to
  `SVC.decision_function`, forced-binary label through the shared band rule (half_width 0), and a
  non-pickle artifact (`baseline.npz` read with `allow_pickle=False` + strict `baseline.json`) with
  one `artifact_sha256` for F4 (D36). Two review fixes: the npz is written with fixed zip metadata so
  equal models give identical bytes and hash; `from_sklearn` reads the public `gamma` and refuses
  `"scale"` / `"auto"` (D37 fixes numeric gamma).
- **B2** (`src/tuning/baseline_fit.py`, `configs/baseline.yaml`): the D37 config with a strict
  loader; `select_baseline` fits the 16 (C, gamma) configurations on standardized train and selects
  the lowest val-A EER (ties: smaller C, then smaller gamma), keeping all 16 rows in a report;
  `fit_baseline_threshold` sets the D29 threshold at the val-B EER point of `BaselineModel.score`;
  `fit_baseline` composes them. Documented in `docs/baseline.md`.
- **Review fixes (B1–B2):** `BaselineModel.score` scores in batches of 2048 rows, so the Stage 13
  kernel matrix stays bounded (same scores, ≤ 1e-12). The selection report records the
  scikit-learn version and the SVC settings of every fit (provenance).

**Checks [verified]:** full suite in the project venv (Python 3.14.7) `366 passed`; governance test
`3 passed`. Tests prove val-A reaches only selection and val-B only the threshold (eer spy), the
config equals D37 (drift guard), and refitting gives identical reports and artifact hash. On the
synthetic data the selected configuration is C 10, gamma 0.025 (tied with C 100 at val-A EER 0.35;
the tie rule picked the smaller C). Synthetic data only; no dataset, feature or model access.

**Gate:** human review of B1–B2; D37 must be approved (PR below) before this branch merges. Real
fitting waits for F3 features.

---

### Entry 55 — Per-window preprocessing integration requirement (2026-10-18)

**What / AI assistance:** At Khalil's request, Codex added the team follow-up from Malak's
Stage 6 work to ADR 0007's D35 implementation list: a synthetic runner integration test
must exercise the shared frozen preprocessing function for every complete 2 s window,
with trimming and centred padding per window. Direct-call comparisons and distinct silence
patterns must detect whole-recording preprocessing before splitting.

**Scope / gate:** Documentation only. The runner and integration test are not implemented
by this change; no decision acceptance, allowlist change or external execution is implied.
Review the final F3 API before implementing. Entry number must be checked against latest
main before merging this documentation update.

---

### Entry 56 — Integrate main into preprocessing-test requirement (2026-10-18)

**What / AI assistance:** Codex resolved the changelog and log conflicts against `bf8096c`.
Kept both changelog additions and all merged Entries 1–54; renumbered the unmerged
per-window test requirement to Entry 55. The D35 requirement remains in ADR 0007.
Baseline code/config/tests from main are retained. No decision approval is inferred here.

**Checks [verified]:** full local suite `366 passed in 14.86s`, including governance;
merged entries retained, sequential numbering and conflict-marker checks passed.

**Gate:** Finish the merge commit, push PR #32 and obtain updated CI/human review.
The runner integration test is still future implementation work; no external data accessed.

### Entry 57 — Analyst interface decision proposed: React instead of Streamlit (2026-10-19)

**What:** on `docs/stage18-frontend-stack` (Iheb), D38 / ADR 0008 propose replacing the Streamlit
dashboard with a React + TypeScript web app, after course-professor feedback. Only the analyst
interface changes; FastAPI, `src/` and the §11 contract are unchanged. A clickable mockup (sign-in,
Overview, Analyze with all states, History, System status) was accepted by Iheb. The Overview page
is new scope, limited to figures from stored records and built last; fallback to Streamlit by
decision if the Analyze flow is not merged by week 10.

**Evidence [verified by Iheb]:** `python -m pytest -q` passes on the branch; docs-only change.

**AI assistance:** **Claude** (claude.ai chat) drafted ADR 0008 and built the mockup, applying the
Impeccable design guidance (markdown only; its downloaded binary and hooks were not run). Iheb
reviewed both.

**Gate:** team review of this PR (all three reviewers). AI agents do not merge.

### Entry 58 — D36 proposed: baseline in the single external run (2026-10-18)

**Authorship:** requested and specified by Donia Mabrouk (Part C owner). Drafted by **Claude Code**
(Claude Opus 5.5), an AI tool, from the repository documents only; Donia reviews before push.

**What:** on `docs/stage6-d36-baseline-external` (docs only, no code):
- Gap found in the briefs review (Entry 47): Protocol §1 RQ2 compares CNN and baseline
  externally, but §8 scores only the frozen CNN bundle, so the external half of RQ2 has no run.
- Proposed **D36** with [ADR 0009](decisions/0009-baseline-in-external-run.md): both frozen models
  scored on the same windows in the one Stage 13 run, under the D35 dataset-wide claim; baseline
  forced binary at its D29 threshold plus EER/AUC; paired CNN − baseline EER difference with a
  speaker-clustered paired bootstrap CI; baseline artifact hashed in F4, not in the ModelBundle;
  C→D amended to one stored-score file per model with identical columns.
- Review fixes (Donia): baseline stored-score values defined (raw = SVM score, p = NA, prediction
  forced binary, never "uncertain"); the "drop external RQ2" alternative now cites Protocol §12.
- Register row D36 and ADR table row 0009; Protocol §14 change-log row; CHANGELOG line.
- **D33–D35 acceptance recorded:** PR #26 was approved by Donia (after her requested changes were
  incorporated) and Malak; Khalil is the author; Iheb has not approved. ADR 0007 requires all
  four, so D33–D35 stay Proposed ("accepted by Donia, Malak and Khalil (author) via PR #26,
  2026-10-18; Iheb pending"); the ADR 0007 reviewer table is filled accordingly.
- Iheb to accept or revise D33–D35 (and D36) in the review of this PR; when all four have
  accepted, set D33–D35 to Approved.

**Evidence [verified]:** `bootstrap_ci(statistic, n, *, clusters=...)` signature checked in
`src/evaluation/bootstrap.py`. Test output in the PR report. No external data read, no model run.

**Approval (2026-10-19):** Malak, Khalil and Iheb approved PR #29 on GitHub; Iheb's review answers this entry's request to accept or revise D33–D36. D33–D36 are Approved; D35 approval does not authorize the external run.

**Gate:** D33–D36 approved by all four. AI does not approve or merge.

### Entry 59 — Docs aligned with D38 (2026-10-19)

**What:** on `docs/stage18-stack-followup` (Iheb), the follow-up promised in ADR 0008:
`docs/architecture.md` (online tier, component diagram, module table, UC-01 flow), the Dashboard
row of ADR 0002 (pointer to ADR 0008) and the `AGENTS.md` roadmap row now name the React web app;
ADR 0008 status set to Accepted (PR #35, approved by Donia; Malak and Khalil informed on 2026-10-19, objections to be handled by a new decision). Open items for PR #14 and PR #35 ticked.

**Evidence [verified by Iheb]:** `python -m pytest -q` passes on the branch; docs-only change.

**AI assistance:** **Claude** (claude.ai chat) prepared the edits; Iheb reviewed the diff.

**Gate:** review of this PR. AI agents do not merge.

---

---

### Entry 59 — Stage 6 preprocessing profile implementation candidate (2026-10-18)

**What [verified locally]:** on `feat/stage6-preprocessing-profile-malak`, Malak approved the Stage 6
plan and Codex implemented the F3 candidate only. `configs/preprocessing.yaml` records the strict
profile and F2 ledger link. `src/preprocessing/` supplies one shared mono/resample, RMS edge-trim,
length-fix, log-Mel and MFCC path; the official validator verifies the frozen F2 manifest and every
sample hash before decode, re-screens RMS/silence/bandwidth, and runs D18 invariance. The benchmark
uses disposable forward/backward operations without an optimizer, weight update, checkpoint or model
metric. `notebooks/02_stage6_preprocessing_profile.ipynb` checks out an exact clean commit, performs
two validation runs, compares all four files byte-for-byte, runs the benchmark, and exports candidate
evidence. Proposed D39 records the settings and gate; it was renumbered because current main already
assigned D33–D35 and D37, while D36 is reserved by an in-flight team branch.

**Evidence [verified locally before the main update]:** focused `tests/preprocessing`: `29 passed in
1.51s`; full suite: `295 passed in 45.36s`. Tests use synthetic arrays/files only. No real audio was
opened and no model was trained. The real FoR validation, shortcut results, invariance results, GPU
measurements, selected future batch size and determinism hashes do not exist yet; they require the
clean Colab run.

**AI assistance:** **Codex** drafted the configuration, implementation, tests, notebook and proposed
decision after Malak approved the plan. Malak approved implementation commit `fdfa14f`; Donia and
Iheb review the evidence PR. AI does not approve, freeze or merge.

**Gate:** implementation candidate only. F3 is not effective; do not begin Stage 7 model work until
the clean FoR run passes, D39/F3 evidence is reviewed, and the PR is merged by a human. D32 also
requires the Stage 6 brief and teach-back review before the stage is marked done.

---

### Entry 60 — Stage 6 Colab manual-transfer fallback (2026-10-18)

**What [verified in code; official run pending]:** after repeated Colab Drive mounts failed with
`credential propagation was unsuccessful`, the Stage 6 notebook gained an explicit `manual` transfer
mode. Malak can download the two pinned FoR inputs through the normal Drive web interface, upload
`for_2sec_ff8c82c7.tar` and `for_2sec_REVISION.txt` to the temporary Colab `/content/` session, and
download `stage6_evidence.zip` when the run finishes. The original `drive` mode remains available.

**Boundary:** this changes only transport into and out of the temporary Colab runtime. It does not
change the F2 manifest or hash ledger, preprocessing settings, validators, shortcut rules, D18
invariance checks, compute benchmark or proposed D39. The notebook still checks out an exact clean
commit and validates the pinned revision before extracting the FoR archive. No dataset is made public
or committed, and DEEP-VOICE remains out of scope.

**Evidence [verified locally before the main update]:** notebook JSON parses as nbformat 4.5 with 12
cells; `git diff --check` passed; full suite: `295 passed in 9.24s`; DEEP-VOICE governance: `3 passed
in 0.22s`; token scan found no secrets. No real audio was opened locally; the official two-run FoR
evidence does not exist yet.

**AI assistance:** **Codex** prepared the manual-transfer fallback after Malak chose not to wait for
the external Colab authentication issue to be resolved. Malak approved correction commit `34e826b`.
AI does not approve, push or merge.

**Gate:** fallback committed and pushed; the official run must cite the final clean post-merge commit.
F3 remains proposed and Stage 7 model work remains blocked.

---

### Entry 61 — Stage 6 branch synchronization with current main (2026-10-18)

**What [merge candidate]:** Donia reported that the Stage 6 branch was 31 commits behind main. Malak
paused the official Colab run. `origin/main` was fetched at `bf8096c` and merged without rewriting the
published Stage 6 history. Incoming preprocessing code did not overlap; conflicts were limited to
`docs/CHANGELOG.md`, `docs/decisions/README.md` and this append-only log. Both sides were preserved.
The Stage 6 decision was renumbered from the colliding D33 to D39, and its log entries from 39–40 to
55–56. D33–D35, in-flight D36 and D37 retain their team meanings.

**Why:** the PR must be current with main, and both official Colab runs must cite the final clean
post-merge commit. The compute benchmark—not a guess—will supply the future CNN batch size required
by D30.

**Evidence [verified locally]:** full merged suite: `395 passed in 33.43s`; DEEP-VOICE governance:
`3 passed in 0.33s`; notebook JSON parses as nbformat 4.5 with 12 cells; `git diff --cached --check`
passed; token scan found no secrets; decision and project-log ID checks found no duplicates. No FoR
audio or DEEP-VOICE content was accessed during synchronization.

**AI assistance:** **Codex** fetched main, inspected the overlaps and drafted the conflict resolution.
Malak reviews and approves the merge commit. AI does not push or merge the PR.

**Gate:** do not resume the official Colab run until the resolved branch passes all checks, Malak
approves the merge commit, and the new final commit hash is pushed.

---

### Entry 62 — Stage 6 padding-shortcut correction before official run (2026-10-18)

**What [verified locally; official run pending]:** after Donia's pre-run review, Malak approved a
Stage 6-only correction that exposes the exact number of zero-padding samples added after edge
trimming. The FoR validator now screens `padding_fraction` alongside RMS, silence and bandwidth and
reports mean, median, p95 and maximum padding by class and F2 subgroup for both train-only and all-data scopes.
Synthetic tests plant a padding cue and verify that the AUC screen and subgroup summary detect it.

**Pre-declaration and boundary:** the train-only loudness AUC threshold of 0.75 was already recorded
in `configs/preprocessing.yaml` and proposed D39 in implementation commit `fdfa14f`, before any
official screen output was viewed. This correction extends proposed D39 before the official run; it
does not change that threshold or the settled feature contract: log-Mel shape `(1, 64, 197)` with no
additional CNN-feature normalization, plus 20 DCT-II, orthonormal MFCCs from the same natural-log Mel
representation summarized as mean + standard deviation = 40 values. The shared
`preprocess_waveform` function still applies trim and centred pad/crop independently to every input
window. Stage 13's guarded external-window integration and tests remain Khalil's work and are not
implemented or run here. No real FoR audio or DEEP-VOICE content was accessed locally.

**Evidence [verified locally]:** focused preprocessing suite: `30 passed in 3.76s`; full suite:
`396 passed in 26.32s`; DEEP-VOICE governance: `3 passed in 0.22s`; `git diff --check` passed. The
official two-run FoR results, determinism hashes, GPU measurements and D30 batch size remain pending
until this correction is reviewed, committed and pushed and its new exact commit is used in Colab.

**AI assistance:** **Codex** implemented the approved padding measurement, screen, summary, tests and
documentation. Malak approved the correction commit; Donia and Iheb review the evidence PR. AI does
not approve, push or merge.

**Gate:** do not run the official Colab cells from `c713111`; first review and commit this correction,
push it, and use the resulting clean commit hash in the notebook.

---

### Entry 63 — Stage 6 synchronization after PR #32 (2026-10-18)

**What [merge candidate]:** after the padding correction was pushed, `origin/main` advanced from
`bf8096c` to `638339b` through PR #32. Malak paused Colab and approved another current-main merge.
The incoming D35 documentation requires Khalil's future Stage 13 runner integration test to call the
shared frozen preprocessing function separately for every complete 2 s window. It does not move that
implementation or test into Stage 6.

**Conflict resolution:** code merged without conflict. The append-only project log preserved main's
Entries 55–56 and renumbered the still-unmerged Stage 6 Entries 55–58 to Entries 57–60. The Stage 6
padding implementation and proposed D39 remain unchanged. Current status and open items now describe
both work streams without claiming that Khalil's integration test is implemented.

**Evidence [verified locally]:** full merged suite: `396 passed in 14.85s`; DEEP-VOICE governance:
`3 passed in 0.32s`. Conflict-marker, sequential-entry, token and diff checks are completed before
the merge commit. No FoR audio or DEEP-VOICE content was accessed during synchronization.

**AI assistance:** **Codex** fetched current main, inspected the incoming scope and prepared the
append-only conflict resolution. Malak approved the merge commit. AI does not push or merge the PR.

**Gate:** do not resume the official Colab run until Malak approves this merge commit, it is pushed,
and the notebook uses its new exact hash.

---

### Entry 64 — Stage 6 first official candidate and D18 Nyquist-edge correction (2026-10-18)

**What [official candidate, not accepted as F3]:** Malak completed the FoR-only Colab validation
twice from clean implementation commit `532877d`. The four validation artifacts were byte-identical;
the pinned dataset checks, shortcut screens and no-training GPU benchmark completed. The benchmark
selected candidate batch size 256 at 83.763 clips/s. However, D18 did not pass: 57 of 180 feature
comparisons exceeded relative L2 0.01. The worst log-Mel relative L2 was 0.019605 and the worst MFCC
relative L2 was 0.012685; cosine similarity remained above its 0.999 threshold. Therefore this run is
diagnostic evidence only, batch size 256 is provisional, and F3 is not effective. The first-run
diagnostic archive was not retained; this entry's summary numbers are the available record, and no
archive hash is claimed or reconstructed.

**Root cause [verified on the 30 frozen FoR train samples only]:** the unstable comparisons were
concentrated in the final Mel band at the exact 8 kHz Nyquist edge. Changing resampler quality or
disabling trimming did not fix the issue, and padding/trim decisions were unchanged. A 7.6 kHz Mel
upper bound gave zero D18 failures with `soxr_hq`, `soxr_vhq`, polyphase and SciPy conversion probes;
the worst independent-converter log-Mel relative L2 was 0.009961. No DEEP-VOICE audio, model
training, optimizer, weight update or checkpoint was used.

**Correction [approved by Malak; commit pending]:** set `features.fmax_hz` from 8000 to 7600, keep
all other profile settings unchanged, and add a synthetic near-Nyquist regression test. Proposed D39
now records the 0–7.6 kHz range and the reason for the correction. Local verification passed: 14
focused preprocessing tests, the complete 397-test suite, and all 3 DEEP-VOICE governance tests.

**AI assistance:** **Codex** inspected the downloaded evidence, isolated the Nyquist-edge cause using
FoR-only diagnostics, proposed the 7.6 kHz correction, and implemented it after Malak's approval.
Donia and Iheb still review the corrected official evidence and D39/F3. AI does not approve, push or
merge.

**Gate:** review, commit and push this correction first. Then restart the Colab runtime, use the new
exact clean commit, rerun both validation copies and the benchmark, and accept the evidence only if
every D18 and shortcut check passes and the validation artifacts are byte-identical.

---

### Entry 65 — Stage 6 Nyquist-edge correction committed and pushed (2026-10-19)

**What [verified]:** Malak approved the Stage 6 7.6 kHz correction and committed it as `6e17185`
(`fix(stage6): stabilize Mel Nyquist edge`). She pushed it to
`origin/feat/stage6-preprocessing-profile-malak`; local HEAD and the remote branch are identical. A
fresh fetch confirms the branch is zero commits behind `origin/main` at `638339b`.

**Evidence:** the approved commit contains the profile/config correction, synthetic Nyquist-edge
regression, D39 explanation, changelog and Entry 64. Its pre-commit verification was 14 focused
preprocessing tests, the complete 397-test suite and all 3 DEEP-VOICE governance tests. The first
candidate evidence remains diagnostic only; no corrected official run or F3 approval is claimed.

**AI assistance:** **Codex** verified the remote branch and updated the append-only status after
Malak's push. AI did not push or merge.

**Gate:** use exact clean implementation commit `6e17185` for both corrected Colab validation runs
and the no-training benchmark. Do not approve D39/F3 until every required check passes, the two runs
are byte-identical, and Donia + Iheb review the evidence.

---

### Entry 66 — Corrected official Stage 6 evidence passes (2026-10-19)

**What [verified candidate evidence]:** Malak ran the complete FoR-only Stage 6 notebook from exact
clean implementation commit `6e1718547756ef04744ce0b3b1b73bb839920052`. Both validation runs
processed all 17,721 frozen FoR rows, verified every sample/F2 hash and produced byte-identical copies
of all four validation artifacts. The downloaded evidence ZIP SHA-256, recomputed from the local
9,609-byte `stage6_evidence (1).zip`, is
`5e66ceedfa446dadeb5e2d97ff388adc63c615f5214d8b3359c5bda73664ceb6` (64 hex characters); its seven extracted files
were uploaded to Drive `processed/metadata/stage6/` and copied to
`docs/audit_evidence/stage6_preprocessing/` for review.

**Validation [verified]:** all checks pass. D18 has 180/180 passing comparisons; maximum relative L2
is 0.0021238185 (limit 0.01) and minimum cosine is 0.9999977627 (limit 0.999). The train-only
FAKE-vs-REAL RMS AUC is 0.698602, so D39's pre-declared loudness rule keeps `mode: none`. Every
padding-fraction screen is low (maximum AUC 0.505879). One review flag is retained rather than hidden:
the all-split FAKE_other-vs-REAL silence-fraction AUC is 0.750071 (moderate), while the pre-declared
train view is 0.749901 (low). This is not an automated gate failure but requires explicit reviewer
attention.

**Benchmark [verified]:** Tesla T4, 15,637,086,208 bytes total VRAM, PyTorch `2.11.0+cu130`, Python
3.13.15 and 90.666 FoR clips/s preprocessing throughput. All candidate sizes 16–256 fit; D30 batch
size 256 is recorded in `configs/cnn.yaml`. Benchmark mode was forward/backward only with no optimizer,
weight update, checkpoint or model metric. DEEP-VOICE was not accessed.

**Checks [this evidence-preparation session]:** clean-branch full suite `397 passed in 40.91s`;
DEEP-VOICE governance `3 passed in 0.19s`; token grep `no token found`. Independent evidence checks
confirmed all recorded artifact hashes, exact profile/manifest hashes, row counts and thresholds.

**AI assistance:** **Codex** inspected the evidence ZIP, independently checked hashes/counts/thresholds,
copied only the seven small evidence files, recorded batch size 256 and prepared the documentation.
AI did not approve, push or merge.

**Gate:** **WAITING FOR APPROVAL.** Donia and Iheb review D39/F3, including the near-threshold
all-split silence flag. After human approval, commit/push the evidence, require green CI and merge by a
human. Only then write the Stage 6 brief/teach-back and mark Stage 6 complete.

---

### Entry 67 — Stage 6 review corrections after current-main update (2026-10-19)

**What [verified locally]:** incorporated Donia's follow-up after PRs #29 and #35 merged. Preserved
main's D38 React decision and added the Stage 6 candidate as D39; kept main's Entries 57–58 and
renumbered the eight Stage 6 history entries to 59–66. Restored D30 to the exact `origin/main` row,
corrected Entry 66's downloaded evidence ZIP SHA-256, and retained the blank line after Entry 38.
D39 contains the config SHA, profile fingerprint, implementation commit and all four artifact hashes.

**First-run evidence:** the diagnostic archive was not retained. Entry 64 preserves the initial-run
summary numbers; no archive is reconstructed and no hash is claimed. This is non-blocking per Donia.

**Checks:** `configs/preprocessing.yaml` remains unchanged and its SHA-256 is still
`7bd35bfdcb9270dd55ea9e73449c7bf6c6a359a80335b2551c0c89dde77c8630`; it contains no D-ID comment.
The ZIP hash matches the local corrected archive. The main sync and documentation conflict resolution
are prepared but not committed; documentation-only, so tests were not rerun.

**AI assistance:** **Codex** fetched current main and prepared the local merge/documentation update.
AI did not commit, push, message reviewers, approve F3 or merge the PR.

**Gate:** **WAITING FOR MALAK APPROVAL** of the local merge and review corrections, then commit/push
and require green CI. Donia will re-review. F3 remains pending human review and merge.

---

### Entry 68 — Stage 4/5 D32 briefs prepared for review (2026-10-19)

**What [doc]:** added the retrospective briefs `docs/stages/stage04-split.md` and
`docs/stages/stage05-dataset-freeze.md` from the approved D32 template. Updated the stage index and
changelog. The briefs summarize the already-closed Stage 4 split and Stage 5 F2 freeze, link to the
existing reports/configuration, and include teach-back questions and re-run commands. No dataset or
audio files were added.

**Evidence [verified against committed records]:** Stage 4 facts and hashes cite its report, config,
and Entries 25/28; Stage 5 manifest counts, hashes, checks, and determinism cite its report, D28,
and Entries 30/36. The source links are in the briefs.

**AI assistance:** Codex drafted the briefs from the D32 template and existing project records and
evidence. Malak owns review of the content; Donia's teach-back review and acceptance remain pending.
AI does not approve or merge.

**Gate:** Stages 4–6 are ready for review in the regular PR on `docs/stage4-6-briefs-malak`.
Do not mark them accepted until Donia completes the D32 teach-back.

---

### Entry 69 — Stage 6 D32 brief and D39 status recorded (2026-10-19)

**What [doc]:** added `docs/stages/stage06-preprocessing.md` alongside the Stage 4 and 5 briefs.
It summarizes the reviewed Stage 6 candidate and links to the official FoR-only evidence, config,
implementation and Colab runner. Updated the stage index and changelog so all three briefs are
listed together for review.

**Decision [verified from PR history]:** PR #34 was approved by Donia and Iheb and merged at
`4707476`; D39/F3 is therefore recorded as approved. The D32 teach-back for the Stage 6 brief, and
acceptance of all three retrospective briefs, remain separate review gates.

**Evidence:** the brief cites Entry 66 and the seven committed files under
`docs/audit_evidence/stage6_preprocessing/`. It preserves the train-only shortcut rule, reports the
all-split silence AUC only as descriptive, and records that the initial diagnostic archive was not
retained. No dataset or audio was added.

**AI assistance:** Codex drafted the Stage 6 brief and documentation updates from the D32 template,
the merged decision register, project log, config and committed reports. Malak owns review; Donia
and Iheb's teach-back review remains pending. AI does not approve or merge.

**Gate:** Stages 4, 5 and 6 briefs are prepared together on the documentation branch; the combined
regular PR and D32 teach-back reviews are still pending.

---

### Entry 70 — Factual corrections after Stage 4–6 brief review (2026-10-20)

**Record corrections:** Entry 59 was assigned twice: `Entry 59 (D38 docs)` refers to Iheb's
architecture follow-up; `Entry 59 (Stage 6)` refers to Malak's preprocessing implementation. Entry
67's statement that the blank line after Entry 38 was retained did not hold after the final merge; the
missing blank line is restored here. Stage 6 history entries are 59–67, with the collision disclosed
rather than silently renumbered.

**Stage 6 brief corrections:** batch size 256 is attributed to D39 and `configs/cnn.yaml`, not D30;
the padding means are identified as all-split alongside the train-only values; approval is cited to
Entry 69; the anti-aliasing statement distinguishes verified test scope from its hypothesized effect.

**AI assistance:** Codex applied Donia's requested documentation corrections. Malak remains
responsible for review; Donia's D32 teach-back and acceptance of the briefs remain pending. No audio
or dataset files were added.

**Gate:** corrections are prepared on `docs/stage4-6-briefs-malak`; human push/CI and Donia's D32
teach-back/re-review remain.

---

### Entry 71 — Stage 7 step 1: verified MFCC feature cache (2026-10-20)

**What:** on `feat/stage7-baseline-fit` (commits `dc3b2db`, `6f8413f`), added
`src/features/mfcc_cache.py` with `build_mfcc_cache`, `load_mfcc_cache` and a CLI, the design note
`docs/feature_cache.md`, and 35 tests in `tests/features/test_mfcc_cache.py`. Only the train, val-A
and val-B splits are allowed; the test split is refused in code before any file is read (D31). The
cache is a non-pickle, byte-deterministic npz plus strict JSON, bound to the F3 profile fingerprint
(D39) and, by default, to the approved F2 manifest hash (D28). The npz is read once, so the checked
bytes are the loaded bytes.

**Evidence [verified]:** `python -m pytest -q` → 432 passed; `tests/governance` → 3 passed (local
`.venv`, Python 3.14.7). Synthetic WAV files only; no dataset was read, no Colab run, no model fitted.

**AI assistance:** **Claude Code** wrote the code, tests and design note from Donia's specification.
**Claude (claude.ai)** reviewed both diffs with Donia and requested the single-read and F2-binding
fixes (commit `6f8413f`). Donia reviewed every change before commit. Claude Code also wrote this
entry and the CHANGELOG line; AI did not push, open or merge a PR.

**Gate:** PR #37 merged at `7538a1d`; the real feature build and baseline fit (Stage 7 steps 2–3) wait for the housekeeping PR, which records D40 (W&B used from Stage 8; baseline provenance in committed JSON reports). **WAITING FOR HUMAN REVIEW.**

---

### Entry 72 — Stage 6 closed, Stage 7 housekeeping, D40 and H7 proposed (2026-10-20)

**Numbering:** follows Entry 71 (Stage 7 step 1, PR #39).

**Stage 6 closed [doc]:** PR #37 merged at `7538a1d` with briefs 04–06. Donia's D32 teach-back was
completed with AI assistance (Claude explained the §7 answers; Donia reviewed them against the
briefs and evidence). Briefs 04–06 are marked accepted (PR #37); `AGENTS.md` marks Stage 6 done and
Stage 7 current. Briefs 02–03: teach-back completed by Khalil (Stage 2) and by Iheb, Malak and
Khalil (Stage 3), reported by Donia on 2026-10-20; confirmed in writing by their approval of PR #40;
briefs 02–03 are accepted.

**Record corrections [verified against PR history as provided]:** D29–D31 statuses and the matching
Protocol §14 row now cite PR #16 instead of "this PR"; the D37 §14 row cites PR #30 (approved by
Khalil and Malak on 2026-10-18, Donia author). D38's register status is unchanged here;
D38 was agreed by all four in person, and confirmed in writing by their approval of PR #40.
CONTRIBUTING now asks to check open PRs and remote branches before claiming a D-ID, ADR number or
log entry number. D37's register status now records its approval via PR #30.

**Proposed D40 [doc]:** W&B logging applies from Stage 8 (every CNN training and model-selection
run, including quick tests); the deterministic Stage 7 baseline fit is not logged to W&B, and its
provenance is the committed selection report, threshold report, feature-cache JSON files and
`artifact_sha256`. Waits for team approval.

**Pre-registered H7 [hypothesis]:** silence-fraction shortcut test on FoR test REAL clips, from the
stored Stage 11 per-clip outputs (no second test run, D31); tested in Stage 16. Added to the register
and Protocol §1. Requires Stage 11 to store per-clip path, group, raw score, calibrated p and
prediction.

**Colab environment check [verified by Donia, reported]:** run on 2026-10-18 at `main` `bf8096c`,
fresh CPU runtime, no data, no Drive, no external set. Python 3.13.16; `requirements.txt` installs;
scikit-learn 1.9.1, numpy 2.5.3, scipy 1.16.3, pandas 3.0.6; `python -m pytest -q` → 366 passed in
23.79s; `google.colab` and `google.colab.drive` import after the install. `pip check` on the whole
Colab environment reports 8 conflicts, all in Colab-preinstalled packages the project does not import
(google-colab wants pandas 2.2.3; diffusers, datasets, tokenizers, gradio-client want
huggingface-hub < 2.0; moviepy, pytensor, ipython): recorded, not fixed. A clean venv could not be
created (Colab lacks `ensurepip`); clean-install consistency is covered by CI on Python 3.13.
`drive.mount()` was not run. Rule for Stage 7 notebooks: mount Drive before installing
`requirements.txt`. Notebook (view-only, shared with the team):
https://colab.research.google.com/drive/1k-nBdzEfY_m9FdigsINXOaw9cqbRfcev .

**AI assistance:** Claude (claude.ai) wrote both Colab check cells and interpreted their output;
Donia ran them. Claude Code applied this entry's documentation edits from Donia's instructions on
`docs/stage7-housekeeping`. Team review is required; AI does not approve or merge.

**Gate:** approved by all four in PR #40 (2026-10-20): D40 approved, H7 pre-registration accepted, D38 confirmed, briefs 02–03 teach-backs confirmed and accepted. AI does not approve or merge.

---

### Entry 73 — Stage 7 step 2A: baseline run orchestrator and Colab notebook (2026-10-24)

**Numbering:** follows Entry 72; the number is fixed at merge time (Khalil's open PR #33 may also
need a number).

**What:** on `feat/stage7-baseline-run` (commits `f0fcc58`, `c6c3468`), added
`src/tuning/baseline_run.py` (`run` and a CLI). It loads the train, val-A and val-B MFCC caches
(F2 binding, F3 fingerprint and profile config hash checked), calls `fit_baseline` (selection on
val-A, D29 threshold on val-B; D9, D37), saves the non-pickle artifact, and writes the selection,
threshold, run (full provenance, including environment versions) and subgroup reports. The
subgroup report (REAL vs FAKE_mp3 and REAL vs FAKE_other EER on val-A and val-B) is descriptive
only and was pre-declared on 2026-10-24, before any real run. `notebooks/03_stage7_baseline.ipynb`
runs every step twice from an exact clean commit and asserts byte identity; the artifact goes to
Drive `checkpoints/` only, never to Git; existing evidence or artifact folders are never
overwritten. Design note `docs/baseline_run.md`; 13 tests in `tests/tuning/test_baseline_run.py`.

**Evidence [verified]:** `python -m pytest -q` → 445 passed; `tests/governance` → 3 passed;
nbformat validation passed (local `.venv`, Python 3.14.7). Synthetic data only; no real data, no
Colab run, no real fit.

**AI assistance:** **Claude Code** wrote the code, tests, notebook and design note from Donia's
specification. **Claude (claude.ai)** reviewed the diff with Donia and asked for three provenance
improvements (full commit, no evidence overwrite, environment versions; commit `c6c3468`). Donia
reviewed every change. Claude Code also wrote this entry and the CHANGELOG line; AI did not push,
open or merge a PR.

**Gate:** after merge, the real run uses the merge commit on `main` (Stage 7 step 2B); its results
go in a separate PR with `docs/results/baseline.md`. **WAITING FOR HUMAN REVIEW.**

---

### Entry 74 — Stage 7 steps 2B–3: official baseline evidence and validation results (2026-10-24)

**Numbering:** follows Entry 73; the number is fixed at merge time (Khalil's open PR #33 may also
need a number).

**Run:** Donia ran `notebooks/03_stage7_baseline.ipynb` on Colab on 2026-10-24 from merge commit
`297c352c91bfc9daafa2943ba04154194aa33248` (PR #41); Python 3.13.16, NumPy 2.5.3, SciPy 1.16.3,
scikit-learn 1.9.1, librosa 1.0.0, soxr 1.1.0, soundfile 0.14.0. Notebook (view-only):
<https://colab.research.google.com/drive/1AFQ4twXQmbW4SXN9iuqU37PZXakaEiGx>. Caches and fit were
each run twice with byte-identical outputs (`determinism.json`). Artifact `baseline-v1`,
`artifact_sha256` `42cc4d53a7cd881f56e4fa54fcf4797de8885b404d0a2b31c293555c868df10a`, stored in
Drive `checkpoints/baseline/baseline-v1/`, not in Git.

**What:** on `docs/stage7-baseline-results`, committed the eight evidence files (three cache JSON
files, four reports, `determinism.json`) to `docs/audit_evidence/stage7_baseline/` and wrote
`docs/results/baseline.md`. No code change; no `.npz` cache or artifact committed. Added the
Stage 7 brief docs/stages/stage07-baseline.md (D32); teach-back by Khalil and Iheb pending.

**Headline numbers [verified against the evidence files] (validation only, not test):** selected
C 10, γ 0.1 (exact val-A EER tie with C 100, γ 0.1; D37 tie rule → smaller C); val-A EER 0.149%
(`selection_report.json`); D29 threshold 0.13285804202427817 with val-B EER 0.0 on 660 bonafide /
669 spoof (`threshold_report.json`). Descriptive subgroups: val-B REAL vs FAKE_mp3 and vs
FAKE_other both 0.0; val-A 0.0 and 0.30% (`subgroup_report.json`).

**Hash verification [verified]:** SHA-256 of each committed cache JSON = `determinism.json`
`cache_sha256` (3/3); SHA-256 of each committed report = `run_sha256` (4/4); `run_report.json`
`npz_sha256` = `determinism.json` `cache_sha256` for train/val-A/val-B (3/3); `code_commit` equal
in both files. All passed. F2 hash = D28, profile config hash and fingerprint = D39,
`configs/baseline.yaml` hash = `baseline_config_sha256`, row counts = Stage 4 sizes
12,404 / 1,330 / 1,329.

**Interpretation [hypothesis]:** the near-perfect separation is too good to read as a solved task.
Likely contributors are an in-domain upper bound (shared TTS generators and probably speakers, no
speaker metadata, 0.95 grouping removes only near-duplicates), a bandwidth shortcut (FAKE median
99%-energy bandwidth about 3.3 kHz vs REAL about 5.1 kHz; FAKE_other separates too, so MP3 history
alone does not explain it; H6) and loudness/silence cues (D39: RMS AUC 0.699, FAKE_other silence
AUC about 0.75; H7). H1 may hit a ceiling but stays unchanged and will be reported as
pre-registered; the D36 external comparison becomes the main test of RQ2. No feature, grid or
threshold change after seeing the results (Protocol §12). Details and limitations:
`docs/results/baseline.md`.

**AI assistance:** **Claude Code** committed the evidence, ran the hash verification and drafted
the results document from the committed files. **Claude (claude.ai)** checked the reports'
consistency with Donia and drafted the interpretation. Donia ran the notebook and reviewed
everything. AI did not push, open or merge a PR.

**Gate:** team review of the results PR, including the Stage 7 brief and its D32 teach-back
(Khalil, Iheb). **WAITING FOR HUMAN REVIEW.**

---

### Entry 75 — Stage 7 closed (D32) (2026-10-24)

**Numbering:** follows Entry 74; the number is fixed at merge time (Malak's D41 and Khalil's PR #33
also need numbers).

**What:** the Stage 7 brief docs/stages/stage07-baseline.md was merged with PR #42. Khalil and Iheb
completed its D32 teach-back face to face, reported by Donia on 2026-10-24; their written
confirmation will be their approval of this closeout PR. On `docs/stage7-closeout`, the brief
status is set to accepted, the stage-briefs index and the `AGENTS.md` roadmap are updated, and the
brief's §8 test sentence now says the four test folders gave 295 passed on main after PR #42.

**Checks:** the four-folder test command (tests/evaluation, tests/models, tests/tuning,
tests/features) on main gave 295 passed.

**Status:** Stage 7 is done; Stages 8–9 are current.

**AI assistance:** **Claude Code** applied the edits from Donia's specification. AI did not push,
open or merge a PR.

**Gate:** approval of this PR by Khalil and Iheb is the written confirmation of the teach-back; AI
does not approve or merge. **WAITING FOR HUMAN REVIEW.**

---

### Entry 76 — Stage 8 preparation: D42 proposed, W&B set up (2026-10-24)

**Numbering:** follows Entry 75; D42 is Entry 76. D41 and the log-Mel PR may both provisionally
hold Entry 77; whichever merges second takes the next free number. Khalil's PR #33 still needs a
log number.

**What:** D42 is proposed in `docs/decisions/README.md`, with a Protocol §14 change-log row. It fixes
the CNN on the F3 log-Mel (1, 64, 197): per-Mel-bin standardization from the FoR train cache only
(stored in the checkpoint; no per-clip normalization); Conv3×3 → BatchNorm → ReLU → MaxPool blocks
with 16/32/64(/128) channels, as benchmarked for batch 256 in Stage 6; global average pooling,
dropout 0.3 and one logit as the raw score; BCEWithLogitsLoss without class weights; Adam with
constant learning rate, no weight decay, float32; early stopping on strictly lower val-A EER with
the earliest best epoch restored; val-B not scored before Stage 10; a log-Mel cache built like the
MFCC cache; safetensors checkpoint with strict JSON metadata; seeded, cudnn-deterministic training
with reproducibility reported as observed; per-epoch W&B metrics only. It is fixed before any CNN
code or result exists, so that no model choice can be tuned against results (as D37 did for the
baseline). Dependency pinning (torch, safetensors, wandb) is left to the Stage 8 code PR.

**W&B setup (D40):** done by Donia on 2026-10-24: Academic plan, organization
`voice-spoofing-detection-must-org`, team `voice-spoofing-detection-must`, private project
`voice-spoofing-detection`. Khalil, Iheb and Malak were invited (acceptance pending). The API key is
stored only in Donia's local `.env`, verified git-ignored (`git check-ignore` → `.gitignore:7`;
`git status` clean). Colab Secrets will be set with the first Stage 8 notebook. `.env.example` now
holds an empty `WANDB_API_KEY=` with its handling rule, and `docs/setup.md` has a W&B section.

**AI assistance:** **Claude** (claude.ai) drafted D42 and guided the W&B setup with Donia;
**Claude Code** wrote this commit. AI did not push, open or merge a PR.

**Gate:** D42 approved by Iheb, Malak and Khalil via PR #44 (2026-10-24); CNN code may start. AI does not approve or merge.

---

### Entry 77 — Stage 8 PR A: verified log-Mel cache (2026-10-24)

**Numbering:** follows Entry 76; the number is fixed at merge time (Malak's D41 and Khalil's PR #33
also need numbers).

**What:** `src/features/logmel_cache.py` (build, load, CLI) caches the F3 log-Mel (1, 64, 197)
float32 for train, val-A and val-B only, with the MFCC cache's guarantees: test split refused in
code before any read, sample hashes checked before decoding, all or nothing, non-pickle
byte-deterministic npz plus strict JSON, F2 binding by default, profile fingerprint, single read.
The D42 train-only normalization statistics are not stored here; the training code computes them.
Design note `docs/logmel_cache.md`; 44 tests in `tests/features/test_logmel_cache.py`.

**Evidence [verified]:** `python -m pytest -q` → 489 passed; `tests/governance` → 3 passed (local
`.venv`, Python 3.14.7). Synthetic data only; no real data, no Colab, no torch.

**AI assistance:** **Claude Code** wrote the code, tests and design note from Donia's
specification; **Claude** (claude.ai) reviewed the diff with Donia. Donia reviewed every change.
AI did not push, open or merge a PR.

**Gate:** Stage 8 PR B (CNN code, dependencies, `configs/cnn.yaml`) follows; the real cache build
runs in the Stage 8 notebook after PR B merges.

---

### Entry 78 (provisional) — Proposed D41: Stage 14 robustness runs: registry purpose key and paired clean scores (2026-10-24)

**Numbering:** follows Entry 77; 78 is provisional and will be fixed at merge time if another log
entry merges first.

- **Owner:** Malak Ben Salem; **reviewers:** all four (D24/D26; proposed major decision).
- **Amends D31.** Propose a registry key `(dataset, split, evaluation_purpose, model_set)` with a
  closed purpose allowlist implemented in code: exactly `stage11_test` and `stage14_robustness`.
  Any other value is refused; adding a purpose requires a new decision. Each allowed purpose/model
  set may claim once. This distinguishes Stage 11 test scoring from Stage 14 robustness conditions
  without opening an unbounded route to repeat test-split runs.
- **Models and scope:** Stage 11 stores per-clip clean scores for `baseline-v1` (D36/F4 artifact
  SHA-256 `42cc4d53a7cd881f56e4fa54fcf4797de8885b404d0a2b31c293555c868df10a`) and all three CNN
  seeds, including the ModelBundle CNN bundle seed. Stage 14 evaluates `baseline-v1` and the frozen
  ModelBundle CNN (bundle seed, F4), one registry claim per model set over the full Protocol §9 grid
  in one pass. Store stable clip ID, label and D14 group ID with each Stage 11 score; baseline stores
  raw score and D29 forced-binary prediction, with calibrated p not applicable.
- **Pairing and criterion:** reuse the corresponding Stage 11 clean scores (no second clean pass);
  apply each pre-declared degradation before unchanged F3. Pair clean/degraded rows by clip ID and
  compute paired 95% percentile CIs from 2,000 D14-group bootstrap resamples with seed `2026`.
  Baseline EER increase at 3.4 kHz is meaningful only if it is at least 5 absolute percentage points
  and the paired CI excludes 0. Retain H6's CNN false-synthetic-rate endpoint on REAL at 3.4 kHz and
  all existing §9 conditions. This is diagnostic evidence, not proof of a sole cause.
- **Protocol record:** proposed D41 amendment adds per-clip score storage to §7's Stage 11 run and
  the baseline model to §9; its §14 history row records that proposal. Update the frozen protocol
  text only after D41 approval.
- **Basis:** D31's existing guard permits one run per model set, while D36 defines the baseline
  artifact in F4. Main's `docs/results/baseline.md` and Entry 74 now contain hash-verified baseline
  evidence; no test-split result was accessed for this proposal.
- **AI assistance:** Codex revised the decision, ADR and records from Donia's review. No dataset,
  audio, checkpoint or test result was accessed, and no experiment was run.
- **Gate:** **PROPOSED — WAITING FOR ALL-FOUR REVIEW.** Do not access FoR test audio for this
  comparison until D41 is approved and the registry and score-storage implementation is reviewed.

---

## Open items

**Part C: external comparison (Entry 58)**
- [x] Approve D36 (all four) before F4. (PR #29, 2026-10-19)
- [x] Iheb to accept or revise D33–D35 (and D36) in the review of this PR; when all four have
  accepted, set D33–D35 to Approved (Entry 58). (PR #29, 2026-10-19)

**Khalil: Stage 13 preparation**
- [ ] Implement the D35 per-window frozen preprocessing integration test (Entry 55).
- [x] Rename synthetic-test branch to the D23-compliant `chore/stage13-synthetic-reporting`.
- [ ] Complete PR #27 integration, retarget to main, and review updated synthetic tests (Entry 51).
- [x] PR #26 merged as `72a9646`; explicit D33–D35 decision confirmations remain pending.
- [x] PR #21 merged into main as `65ce1d2`; PR #23 base changed to main.
- [x] PR #23 integrated and merged as `3f2f4bb` after passing CI (Entries 42–44).
- [x] PR #24 merged into main as `6786c12` (Entries 45–46).
- [x] Review proposed D33–D35 and ADR 0007; record explicit approval or requested revisions (Entry 48). (approved by all four, PR #29, 2026-10-19)
- [ ] Complete clustered CIs, speaker completeness checks and filter/file-level reporting.
- [x] Before Stage 13, approve and record the speaker-average undefined-value convention alongside
  the file-level view and governance allowlist decisions. No convention is approved by this update. (D33–D35 approved, PR #29, 2026-10-19)

**GitHub / repo**
- [ ] D27: confirm whether the team approved it (PR #14 merged) and update its register status (Entry 72)
- [x] Team review of PR `docs/stage18-frontend-stack` (D38, Entry 57): merged as PR #35. Follow-up docs PR `docs/stage18-stack-followup` (Entry 59) updates `docs/architecture.md`, ADR 0002 and the `AGENTS.md` roadmap row.
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
- [x] Review and merge PR `chore/stage5-ai-tooling` (Entry 27): merged as PR #14. Using Context7 is optional per member: set `CONTEXT7_API_KEY` locally (never committed) and approve the project MCP server in your own tool (ADR 0006).
- [x] Stage 13 governance: decide (new decision, before Stage 13) how the guarded external-run code is added to the DEEP-VOICE allowlist of `tests/governance/test_deepvoice_access.py`. (decided by D35, PR #29, 2026-10-19; implementation pending)
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

**Stage 6 gate**
- [x] Implement preprocessing profile v1 candidate, shared transforms, tests, validation, no-training
  compute benchmark and Colab runner (Entry 59).
- [x] Review and commit the Stage 6 implementation candidate as `fdfa14f`; branch pushed.
- [x] Review and commit the manual Colab transfer fallback as `34e826b`; branch pushed (Entry 60).
- [x] Complete, approve and push the synchronization with `origin/main` at `bf8096c` as `c713111`
  (Entry 61).
- [x] Review, commit and push the padding-shortcut correction as `becc52b` (Entry 62).
- [x] Complete, approve and push the synchronization with `origin/main` at `638339b` as `532877d`
  (Entry 63); the first official candidate cited that exact merge commit.
- [x] Diagnose the first official candidate's D18 failure and obtain Malak's approval for the 7.6 kHz
  Nyquist-edge correction (Entry 64); local implementation and all 397 tests pass.
- [x] Review, commit and push the 7.6 kHz correction as `6e17185` (Entry 65).
- [x] Run the full pinned FoR manifest twice in Colab; every sample hash/check, shortcut rule and
  D18 invariance check passed, and the four validation artifacts are byte-identical.
- [x] Record the GPU type, VRAM, PyTorch version, preprocessing throughput and selected future batch
  size; no optimizer, weight update, checkpoint or model metric is permitted in this benchmark.
- [x] Upload the seven corrected evidence files to Drive `processed/metadata/stage6/` and prepare the
  matching small repo evidence under `docs/audit_evidence/stage6_preprocessing/` (Entry 66).
- [x] Record that the first-run diagnostic archive was not retained; Entry 64 preserves its summary
  numbers and no unverified archive hash is claimed.
- [x] Approve and merge the official Stage 6 evidence and review corrections (PR #34, merged at
  `4707476`).
- [x] Record D39/F3 as approved after Donia and Iheb approved PR #34, merged at `4707476`.
- [x] Prepare `docs/stages/stage06-preprocessing.md` from the D32 template (Entry 69).
- [x] Complete the D32 teach-back for the Stage 6 brief before marking Stage 6 done. (Donia, with AI
  assistance; PR #37 merged at `7538a1d`; Stage 6 done, Entry 72)

**Stage 7 harness: protocol clarifications (Entry 31; Protocol §12, next free D-ID when decided)**
- [x] Baseline threshold: own val-B EER point, forced binary only (Donia, before Stage 7 results): D29 (Entry 32).
- [x] Exact CNN model-selection search space in `configs/` (Donia, before Stage 8): D30 (Entry 32).
- [x] Enforcement of "Stage 11 runs once" (run registry) (Donia, before Stage 11): D31 (Entry 32).
- [x] File-level decisions for DEEP-VOICE: frozen band on the file mean of p, secondary view (Khalil, before Stage 13). (D34 approved, PR #29, 2026-10-19)
- [x] Colab install of scikit-learn 1.9.1: verified in Donia's Colab environment check at `bf8096c`
  (366 passed; Entry 72). Stage 7 notebooks mount Drive before installing `requirements.txt`.
- [x] Approve D37 (team review) (Entry 53): approved via PR #30 (Khalil, Malak), 2026-10-18 (Entry 72).
- [x] Approve proposed D40 (W&B from Stage 8) in the `docs/stage7-housekeeping` review (Entry 72). (PR #40, 2026-10-20)
- [x] D38: record written confirmation (approval of the `docs/stage7-housekeeping` PR) and update
  its register status (Entry 72). (PR #40, 2026-10-20)
- [x] Write the baseline grid and scaling in `configs/` before Stage 7 training (D37): `configs/baseline.yaml` (Entry 54).
- [x] Human review of `feat/stage7-baseline-code` (B1–B2) after D37 approval (Entry 54): merged as PR #31.
- [x] Stage 7 step 1: verified MFCC cache for train/val-A/val-B (Entry 71)
- [x] Stage 7 step 2A: baseline run orchestrator and Colab notebook (Entry 73)
- [x] Stage 7 steps 2B–3: real cache build and baseline fit in Colab from the merged commit, then docs/results/baseline.md (Entry 73) (Entry 74)
- [x] Stage 7 brief docs/stages/stage07-baseline.md (D32): written; teach-back by Khalil and Iheb pending (Entry 74) (teach-back face to face, Entry 75)

**Stage 10 risk policy (Entry 33)**
- [x] Human review of `feat/stage10-risk-policy` (R1–R2); CI green on Python 3.13: merged as PR #17.
- [ ] Watch: Platt's absolute gradient-norm stop; if the real val-B fit fails to converge, propose a
  relative criterion (no silent change).

**Stage 11 run registry (Entry 34)**
- [x] Human review of `feat/stage11-run-registry`; CI green on Python 3.13: merged as PR #18.
- [ ] Stage 11: record the registry SHA-256 and line count right after the run (project-log entry
  and results PR).
- [ ] Stage 11 stores per-clip path, group, raw score, calibrated p and prediction for the FoR test split (required by H7, Entry 72)
- [ ] Report H1 as pre-registered even if both models are near 0% EER in-domain (results/baseline.md)

**Stage 12 ModelBundle (Entry 35)**
- [x] Human review of `feat/stage12-bundle-format`; CI green on Python 3.13: merged as PR #19.
- [ ] Stage 12: add the profile-version check once the F3 profile format is frozen.
- [ ] Stage 12/17: load the checkpoint from verified bytes (or re-verify at load).
- [ ] Stage 8: save checkpoints in a non-executable form (safetensors or state_dict + weights_only=True).
- [ ] F4: record manifest_sha256 in the freeze decision.

**Stage briefs (D32, Entry 37)**
- [x] Team review of `docs/stage6-stage-briefs` (approves D32): approved by all three, 2026-10-18.
- [x] Write stage briefs 00–03 (Donia). 00–01 accepted (merged as PR #25 after teach-back review);
  02–03 merged in `3ecb979` (Entry 47); confirm teach-back completion separately. (02–03 accepted, PR #40)
- [x] Prepare stage briefs 04–06 on the docs branch (Entries 68–69).
- [x] Open one regular review PR containing briefs 04–06 on `docs/stage4-6-briefs-malak`.
- [x] Donia completes the D32 teach-back and accepts each brief before the stage index marks them accepted.
  (04–06 accepted, PR #37; Entry 72)
- [x] Briefs 02–03: written confirmation of the reported teach-backs (Khalil; Iheb, Malak, Khalil)
  via approval of the `docs/stage7-housekeeping` PR, then mark them accepted (Entry 72). (PR #40, 2026-10-20)

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
  passed (Entry 22); Khalil's local setup evidence and Codex answer are recorded (Entry 39).
  Factual report sections are filled (Entry 40); Donia's extension version, Iheb's AI answer, review and D14 response,
  owner/setup verification and team acceptance are still open.

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
- [x] Set up W&B (0.6) before Stage 8: project and key handling (`.env`) before the first Stage 8
  run (D40, Entry 72). (W&B project and key handling set up, Entry 76; Colab Secrets with the first
  Stage 8 notebook)
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

**Stage 8 CNN training**
- [ ] Stage 8 brief docs/stages/stage08-cnn-training.md (D32) when the stage closes
- [x] Approve D42 (team) before any CNN code (Entry 76) (PR #44, 2026-10-24)
- [x] Stage 8 PR A: verified log-Mel cache for train/val-A/val-B (Entry 77)
- [ ] Stage 8 PR B: CNN code, dependency pins (requirements-cnn.txt; Colab uses its preinstalled torch, asserted at 2.11.0; CI installs CPU-only torch), configs/cnn.yaml (D30 grid + D42), W&B allow-list wrapper; Iheb and Khalil review
- [ ] Check that Khalil, Iheb and Malak accepted the W&B invite with the Member role

**Stage 14 robustness (D41)**
- [ ] Obtain all-four approval for proposed D41/ADR 0010; then implement and review the closed-purpose registry and Stage 11 per-clip scores before any Stage 11 or Stage 14 FoR test access.

**Later stages (reminders)**
- [ ] `configs/robustness.yaml` → Stage 14. `docker-compose.yml` → Stage 17.
- [ ] Stage 17: verify the ModelBundle checkpoint hash at load (threat model §10, tampering recommendation).
- [ ] Stage 6: re-run the shortcut screen *after* preprocessing (silence, bandwidth).
- [ ] Stage 6: decide the silence / voice-activity detector explicitly (energy-based vs `silero-vad`) and record it as a decision; D17 depends on it.
- [ ] Phase 0 leftovers: confirm ML lead (0.9), W&B project (0.6). `requirements.txt` pinned (Entry 17); confirm the 3.13 install on the first CI run of the PR.

---

## Current status — updated 2026-10-24

**What was done:**
- **Stage 8 PR A, verified log-Mel cache** on `feat/stage8-logmel-cache` (Entry 77): train/val-A/val-B
  only, bound to F3 and F2; synthetic tests only, no real features built.
- Stage 8 preparation (Entry 76): D42 (CNN design and training details) approved (PR #44); W&B project
  and key handling set up (D40), invites pending.
- Stage 7 closed (Entry 75): brief accepted after the D32 teach-back by Khalil and Iheb; Stages 8–9 current.
- **Stage 7 steps 2B–3, official baseline evidence and validation results** on
  `docs/stage7-baseline-results` (Entry 74): run from `297c352` with byte-identical reruns; C 10,
  γ 0.1; val-A EER 0.149%, val-B EER 0.0 (validation only); interpreted as an in-domain upper bound
  with possible shortcuts [hypothesis] in `docs/results/baseline.md`.
- **Stage 6 is complete; Stage 7 is current** (Entry 72): PR #37 merged at `7538a1d`; briefs 04–06
  accepted after Donia's D32 teach-back (with AI assistance). D40 (W&B from Stage 8) approved and H7
  (silence shortcut, Stage 16) pre-registered (PR #40). Donia's Colab
  environment check at `bf8096c`: Python 3.13.16, scikit-learn 1.9.1, 366 tests passed.
- Added the D35 per-window preprocessing integration-test requirement (Entry 55); implementation pending.
- **D36 approved** (ADR 0009, Entry 58): the baseline is scored with the CNN in the single external
  run, so RQ2 can be answered externally (PR #29, 2026-10-19).
- **PR #27 merged** as `3dc20b7`: synthetic reporting tests (Entries 51–52).
- **Synthetic integration tests prepared:** eight invented speakers and 14,960 windows (Entry 51); no real predictions.
- **PR #21 merged** as `65ce1d2`; its documentation is in main.
- **PR #23 merged** as `3f2f4bb`: per-speaker helper and tests (Entries 42–44).
- **PR #24 merged** as `6786c12`: speaker-average helper and tests (Entries 45–46).
- **PR #26 merged** as `72a9646`; D33–D36 were approved by all four after PR #29 (Entry 58, 2026-10-19). ADR 0007/0009 implementation remains required before Stage 13.
- **Lab 6 closeout in progress:** factual report sections and PR #21 CI evidence are documented (Entry 40); Donia's reported test is now recorded (Entry 41); her extension version, Iheb's answer and human verification/review remain pending.
- **Stages 0–3 and Part A are complete.** Protocol v1.0 is frozen as F1 (PR #10).
- **Stage 4 is complete** (Entry 28): PR #12 merged as `865faf4`.
- **Stage 5 is complete and F2 is effective**: Donia approved PR #15, merged as `89b37fc`; D28 is
  approved (Entry 36).
- **Stage 6 implementation `fdfa14f`, manual-transfer correction `34e826b`, current-main merges
  `c713111`/`532877d` and padding-shortcut correction `becc52b` are pushed** (Entries 59–63).
- **The first official FoR candidate exposed a D18 failure** despite byte-identical outputs and a
  successful no-training benchmark (Entry 64). Malak approved the 7.6 kHz Nyquist-edge correction;
  the diagnostic ZIP was not retained.
- **Correction commit `6e17185` is pushed and verified against current main** (Entry 65).
- **Official FoR evidence passed every recorded gate and was merged in PR #34** at `4707476` after
  reviewer approval (Entries 66–67). The evidence records batch size 256 in D39 and
  `configs/cnn.yaml`, plus the retained FAKE_other silence-AUC review flag.
- **Stage 6 evidence and current-main documentation corrections are merged** (PR #34): D38 remains
  the React decision and Stage 6 is D39. History entries 59–67 include the disclosed duplicate 59;
  Entry 70 records the correction and restored Entry 38 spacing. The corrected ZIP hash is recorded,
  and the first-run diagnostic archive is noted as not retained. Donia and Iheb approved PR #34;
  D39/F3 and the Stage 6 D32 brief/teach-back are complete.
- **Stage 7 is closed** after PR #43; its evidence and accepted D32 brief are recorded in Entries 74–75.
- **Stage 7 evaluation harness, steps 1–5, is merged** as PR #13 (Entry 31).
- **Protocol clarifications D29–D31 are merged** as PR #16 (Entry 32).
- **Stage 10 risk policy, steps R1–R2, is merged** as PR #17 (Entry 33).
- **Stage 11 run registry (D31) is merged** as PR #18 (Entry 34).
- **Stage 12 prep, ModelBundle format and verifying loader, merged** as PR #19 (`8b688dc`; Entry 35).
  The **Part C prep is complete**: harness, risk policy, run registry and bundle format are built and
  tested on synthetic data; the real Part C runs wait for F3 (preprocessing profile v1).
- **Stage 5 administrative closeout merged** as PR #22 (`02f53e2`; Entry 36).
- **Stage-briefs PR (D32, Entry 37)** merged as PR #20 (`567f4aa`).
- **Stage briefs 00 and 01** accepted: merged as PR #25 (`610a412`) after teach-back review.
- **Stage briefs 02 (threat model) and 03 (protocol)** written on `docs/stage6-briefs-02-03`
  (Entry 47); merged in `3ecb979`.
- **D37 recorded on main** (Entry 53): baseline grid, train-only scaling and score; Entry 53 records team approval.
- **Stage 7 baseline code, steps B1–B2** on `feat/stage7-baseline-code` (Entry 54): scorer,
  non-pickle deterministic artifact, D37 fitting and `configs/baseline.yaml`; synthetic tests only.
- **Stage 7 step 1, verified MFCC cache** on `feat/stage7-baseline-fit` (Entry 71): train/val-A/val-B
  only, bound to F3 and F2; synthetic tests only, no real features built.
- **Stage 7 step 2A, baseline run orchestrator and Colab notebook 03** on `feat/stage7-baseline-run`
  (Entry 73): reports with full provenance, descriptive pre-declared subgroup report; synthetic
  tests only, no real fit.

**What we have:**
- The approved FoR split and the F2 manifests and hash ledger (D28).
- One shared, tested evaluation harness (`src/evaluation/`) for the baseline, the CNN and the
  external run, documented in `docs/evaluation_harness.md`, plus per-speaker and speaker-average
  helpers for Stage 13.
- Pre-declared rules for the baseline threshold, CNN selection and the single Stage 11 run (D29–D31),
  and baseline tuning, scaling and score (D37, Entry 53).
- A tested MFCC/SVM baseline (`src/models/baseline.py`, `src/tuning/baseline_fit.py`); Stage 7
  validation evidence and artifact provenance are committed in `docs/results/baseline.md` and
  `docs/audit_evidence/stage7_baseline/`, with hash verification recorded in Entry 74.
- A tested risk-policy fit (`src/tuning`) and core apply / save / load (`src/risk`), documented in
  `docs/risk_policy.md`; not yet fitted on real val-B scores.
- A tested append-only run registry (`src/evaluation/run_registry.py`) that refuses a second
  Stage 11 run for the same model set.
- A tested ModelBundle format (`src/bundle/`) whose loader verifies every part before use,
  documented in `docs/model_bundle.md`.
- Governance rules that keep the external set metadata-only until the one guarded Stage 13 run.
- Corrected Stage 6 FoR evidence under `docs/audit_evidence/stage6_preprocessing/`: exact profile,
  deterministic validation, shortcut/padding screens, D18 invariance and no-training T4 benchmark.
- Stage briefs (`docs/stages/`, D32): 00–07 accepted; 02–03 accepted (PR #40), 04–06 (PR #37), and 07 (PR #43).

**Branches:**
- `origin/main`: at `2f18f60` (PR #45 merged; verified log-Mel cache, Entry 77).
- `docs/stage14-baseline-robustness-malak`: D41 proposal, updated from current main; awaiting the approval commit.
- PRs #26, #27, #29 and #35 are merged. D33–D36 are approved by all four; implementation remains a
  prerequisite before Stage 13.

**What we want:**
- A model that detects **synthetic voices**, not dataset accidents, followed by one honest external test.

**What's next:**
1. D33–D36 are approved; ADR 0007/0009 implementation (runner, both artifacts verified before any audio read) is required before Stage 13.
   Briefs 00–06 are accepted.
2. Before Stage 13, implement the approved file-level view, speaker-average convention and guarded
   runner; resolve the external-run governance allowlist.
3. Stage 7 is closed (PR #43); Stages 8–9 are current. Continue the frozen Part C work and preserve the recorded validation-only scope of Entry 74.
4. All four review proposed D41 (including ADR 0010); only after approval implement and review the closed-purpose registry and Stage 11 per-clip score storage. Do not access FoR test audio for the Stage 14 comparison before both gates are met.
5. Then run the real Part C work on the frozen F3 preprocessing profile, and set up the W&B project before the first Stage 8 run (D40).
