# Decision Log

Approved project decisions. These override the Proposal and the Execution Plan where they differ.
Any change needs a new entry with a reason and team approval.

| ID | Decision | Status |
|---|---|---|
| D1 | Write Experimental Protocol v1.0 after the dataset audit; freeze it before training | Approved |
| D2 | Fit the risk policy (thresholds, uncertain band) and calibration on FoR validation **before** the ModelBundle freeze. The DEEP-VOICE run comes after. Supersedes the Exec Plan order (Phase 5 before Phases 6/9). — [ADR 0005](0005-deep-voice-single-use.md) | Approved |
| D3 | No robustness evaluation on DEEP-VOICE. Robustness runs on the FoR test split only (FR-17). Supersedes Exec Plan Phase 9 "optionally DEEP-VOICE". — [ADR 0005](0005-deep-voice-single-use.md) | Approved |
| D4 | Metrics: score-level (EER, ROC-AUC) on the continuous score; decision-level (accuracy, precision, recall, F1) at the frozen threshold, both forced-binary and on the non-deferred subset; 3-way confusion matrix; coverage/deferral rate; class counts; FPR/FNR; 95% CIs. Positive class = spoof. | Approved |
| D5 | Evaluation unit: 2 s windows (primary), file-level mean score (secondary) | **Confirmed** by the Stage 1 audit (DEEP-VOICE files 79–600 s; `docs/dataset_audit.md` §7). Raw FoR clips need no pad/crop (all 2.0 s); whether silence trimming introduces length variation is decided in Stage 6. |
| D6 | DEEP-VOICE audit is metadata-only (counts, headers, hashes, filenames, CSV schema). No listening, spectrograms, or feature stats. Supersedes Exec Plan Phase 2 "explore both datasets". — [ADR 0005](0005-deep-voice-single-use.md) | Approved |
| D7 | Baseline = RBF-SVM on MFCC mean + std | Approved |
| D8 | Research question narrowed to cross-dataset FoR (TTS) → DEEP-VOICE (RVC) generalization. Attribution to "generation method" is a hypothesis only. | Approved |
| D9 | Split FoR validation into val-A (training decisions) and val-B (threshold, calibration) if the size permits | **Applies** (val-B ≈ 665 clips per class ≥ 500; `docs/dataset_audit.md` §7) |
| D10 | No degradation augmentation in ModelBundle v1 | Approved |
| D11 | 3 seeds on the final CNN config; bootstrap 95% CIs, clustered by source speaker for DEEP-VOICE | Approved |
| D12 | Operating threshold at the FoR-val EER point, plus a fixed deferral policy (exact criterion in Protocol v1.0) | Approved |
| D13 | Official FoR split vs re-split: decided by the pre-registered rule in `docs/dataset_audit.md` §0 | **Re-split** (no packaged split; `docs/dataset_audit.md` §7) |
| D14 | Grouping fallback = duplicate hashing + near-duplicate components, if no speaker metadata | **Groups at 0.95** (no speaker metadata; largest group 6 clips = 0.03%; sanity check passed, 0 PCM pairs; `docs/dataset_audit.md` §7) |
| D15 | Official Python version = 3.13 (Colab runtime 3.13.15; results come from Colab). Local development on Python 3.14 is allowed for editing and quick tests. `requirements.txt` must install on both 3.13 and 3.14; CI tests on 3.13. — [ADR 0002](0002-tech-stack.md) | Approved |
| D16 | FoR codec-history imbalance (85% of FAKE vs 0% of REAL clips have an MP3 step): no data changes; re-split stratified by REAL / FAKE_mp3 / FAKE_other; the imbalance is recorded as a limitation; silence is re-screened after trimming (Stage 6). | Approved (Stage 1 audit, 2026-10-04) |
| D17 | DEEP-VOICE speech-window filter using the frozen silence detector. Its threshold is fixed in Protocol v1.0, before the external run. Filtered (primary) and unfiltered (sensitivity) views are computed in the same single run. Drop counts are reported per class and per speaker. | Approved (2026-10-04) |
| D18 | Preprocessing averages channels to mono and resamples to 16 kHz, identically everywhere. Stage 6 adds a pipeline-invariance test on FoR clips converted to 40, 44.1 and 48 kHz stereo. Resampler, settings and tolerance are fixed in Stage 6. | Approved (2026-10-04) |
| D19 | External (DEEP-VOICE) results include a per-speaker table (8 rows) and a speaker average as a secondary summary. | Approved (2026-10-04) |
| D20 | Permitted DEEP-VOICE accesses before the external run are **only**: download, zip hash, unzip, listing paths / extensions / sizes / byte hashes, and `src/audit/deepvoice_metadata_audit.py`. Nothing reads audio content. Extends D6 to cover the download steps; `CLAUDE.md` hard rule 1 matches. (Since D22, this rule lives in AGENTS.md, which CLAUDE.md imports.) — [ADR 0005](0005-deep-voice-single-use.md) | Approved (2026-10-04) |
| D21 | Datasets are pinned: FoR-2sec HF revision `ff8c82c79e7bbefef811941bac9775c9328e9055` and DEEP-VOICE zip SHA-256 `8cb258530daa2bb678121f3b276313a4d56fc638280bfd276a610dd8f732bdcf` in `configs/audit.yaml`. The notebook refuses unpinned or mismatched data. Changing a pin requires a new decision entry. — [ADR 0004](0004-dataset-pinning.md) | Approved (2026-10-04) |
| D22 | `AGENTS.md` is the single AI context for all tools: Codex and Antigravity read it natively, Claude Code imports it via `CLAUDE.md`, Gemini CLI via `.gemini/settings.json`. Rules are edited only in `AGENTS.md`. — [ADR 0003](0003-agents-md-single-ai-context.md) | Approved (team, 2026-10-06) |
| D23 | Branch naming `<type>/stage<N>-<name>`, types `feat`, `fix`, `docs`, `chore` (see `CONTRIBUTING.md`). Replaces the earlier `stage-<N>-<short-name>` convention. | Approved (team, 2026-10-06) |
| D24 | Decisions = this register (every decision) + ADRs in `docs/decisions/` for major decisions only. — [ADR 0001](0001-record-architecture-decisions.md) | Approved (team, 2026-10-06) |
| D25 | MIT license for the project code only. Datasets are never redistributed (not in the repo, not in releases). | Approved (team, 2026-10-06) |
| D26 | Work distribution (Execution Plan v2.1). **Part A** (Stages 0–3): Stage 3 led by Donia. **Part B** (4–6): Malak. **Part C** (7–12): Donia. **Part D** (13–16): Khalil (13, 15, 16), Malak (14). **Part E** (17–19): Iheb. **All four:** Stage 3 sign-off, Stage 13 review, Stage 19 report. Review rule: the consumer reviews the producer. Interfaces (hand-overs, Execution Plan v2.1 §5) fixed in Stage 3: **B→C** manifest columns (path, label, subgroup, group, split, hash) and the preprocessing function's input/output (sample rate, length, feature shapes); **C→D** the ModelBundle format and `score(audio) → score`, plus the per-sample stored-score file format; **C→E** API response `{prediction, confidence, risk_level}` plus model/profile/policy versions, and how the bundle is loaded and its hash verified; **B→Stage 14** the degradation conditions and levels, applied through the same preprocessing profile. Changing an interface after Stage 3 needs a decision. | Approved (team, 2026-10-12) |
| D27 | Shared AI tooling v1: one skill (`pr-prep`, `.claude/skills/`) and one MCP server (Context7, key via `${CONTEXT7_API_KEY}`, never committed). GitHub MCP excluded until read-only is guaranteed for every member; no Filesystem, Git or Fetch MCP; no MCP in Stage 13; later additions only by decision. — [ADR 0006](0006-shared-ai-tooling.md) | Proposed (2026-10-16) |
| D28 | Dataset freeze F2: FoR manifest `badfe948fb7b833357783dd18be00f6bb36253cf8699dc4c4beb4df2b07706f8` (17,721 rows), metadata-only external manifest `7c7a3e52ef254841f53bc08018bd3f9fe7ad83d6cd84531832513c616695355f` (64 rows), and hash ledger `c9fc2e230c0883028459d0b2b8de4f6ee537676cd5a4e259e9d0f545c60943cc`, generated twice byte-identically from clean final implementation commit `df1d6ac`. After this PR is approved and merged, changing a dataset pin, path, label, subgroup, group, split or sample hash requires a new decision. | Approved (PR #15, approved by Donia, 2026-10-17) |
| D29 | Baseline threshold: the MFCC/SVM baseline gets its own operating threshold at the val-B EER point of its continuous score, forced binary only: no calibration, no uncertain band (it is evaluation-only, D7). Reason: same threshold rule and data as the CNN, so the comparison is fair, and Protocol §7 "all D4 metrics for both models" is met. | Approved (Donia, 2026-10-17; review in this PR) |
| D30 | CNN model-selection search space: 4 configurations = conv blocks {3, 4} × learning rate {1e-3, 3e-4}; fixed for all: Adam optimizer, max 30 epochs, early stopping on val-A EER with patience 5; selection = lowest val-A EER (D9). The 4 search runs use seed 0; the chosen configuration is then trained with seeds 0, 1, 2 (D11), reusing the seed-0 search run. Ties in val-A EER: 3 blocks before 4, then learning rate 1e-3 before 3e-4. Batch size is set from the Stage 6 compute benchmark and written in `configs/` before any training run. The grid is written in `configs/` before Stage 8 starts. Reason: a real but small, pre-declared selection; nothing added after results are seen. | Approved (Donia, 2026-10-17; review in this PR) |
| D31 | Stage 11 runs once: every FoR test-split evaluation is recorded in an append-only run registry (dataset, split, model versions, code commit, date); the evaluation code refuses a second test-split run for the same model set. Implemented with a test before Stage 11. Reason: the same protection the SDD gives the external test (External Test Guard), applied to the FoR test split; enforced in code, not only by rule. | Approved (Donia, 2026-10-17; review in this PR) |

## Architecture Decision Records

Major decisions also have an ADR in this folder (D24). New ADRs start from [`TEMPLATE.md`](TEMPLATE.md).

| ADR | Title | Register |
|---|---|---|
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions | D24 |
| [0002](0002-tech-stack.md) | Tech stack | D15 |
| [0003](0003-agents-md-single-ai-context.md) | AGENTS.md is the single AI context | D22 |
| [0004](0004-dataset-pinning.md) | Dataset pinning | D21 |
| [0005](0005-deep-voice-single-use.md) | DEEP-VOICE is used once, as a frozen external test | D2, D3, D6, D20 |
| [0006](0006-shared-ai-tooling.md) | Shared AI tooling v1: one skill and one documentation MCP server | D27 |

## Pre-registered hypotheses

Written before any model exists. Each is tested once, at the stage named, and reported whatever the outcome.

| ID | Hypothesis | Test | Stage |
|---|---|---|---|
| H5 | If the CNN relies on MP3 traces, its recall on FAKE_other is clearly lower than on FAKE_mp3. | Subgroup recall on the FoR test split | 11 / 16 |
| H6 | If the CNN relies on bandwidth, band-limiting REAL clips to ~3.4 kHz raises false FAKE calls. | Robustness test on FoR test REAL clips | 14 |
