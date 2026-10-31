# Changelog

All notable changes to this project. Newest at the top. Keep entries short and human: one line per
change. Based on [Keep a Changelog](https://keepachangelog.com). Detailed history with evidence is
in `project_log.md`.

Categories: **Added**, **Changed**, **Fixed**, **Removed**.

## [Unreleased]

### Added
- Midterm initial refactor and optimization audit, with the complete local baseline, source-linked API/UI and ML evidence, and proposed work awaiting team review.
- Midterm AI ethics and usage reflection grounded in the project's recorded tools, human reviews,
  corrections, data safeguards, bias limitations, licensing and academic-integrity rules.
- Canonical midterm UML package under `docs/uml/`: editable diagram sources and SVG reconstructions of SRS Figure 1 and SDD Figures 1-6, with only the approved D38, API-contract and D41 annotations.
- Midterm prioritized backlog under `docs/midterm/`, mapping the closed SRS requirement set (FR-01–FR-20, NFR-01–NFR-11 and PC-01–PC-09) to existing owners, stages, merged evidence and remaining work without adding scope.
- Stage 17 API skeleton `api/` (FastAPI): `create_app()` with routes under `/api`, `configs/api.yaml` (§9, strict loader, `VSD_API_CONFIG`), the §6 error shape for every error (400 for invalid params, 404/405/500 included, no traceback), `X-Request-ID` on every response, the §5 shapes, and `GET /api/status` with mock values; `fastapi`, `uvicorn`, `python-multipart` and `httpx` pinned; `docs/api_design.md` approved and §6 gains the 405 row.
- Stage 17 API design note `docs/api_design.md` (proposed): `/api` endpoints for status, limits, analyses, notes and stats, one error shape, storage, analyzer seam and config; §11 unchanged.
- D41 implementation (Stage 11 prep): run registry keyed by `(dataset, split, evaluation_purpose, model_set)` with the closed purpose set and a keyword-only purpose; per-clip score files `src/evaluation/clip_scores.py` (CSV + strict JSON sidecar, applicability by model role fixed from D41/D45, exact floats, canonical bytes, exclusive creation; design note `docs/clip_scores.md`); FoR test-split governance guard `tests/governance/test_for_test_access.py` (empty allow-list); synthetic tests only, no FoR test access.
- Stage 10 brief `docs/stages/stage10-risk-policy.md` (D32), draft; teach-back by Iheb, Malak and Khalil pending.
- Stage 10 official risk-policy run (`risk-v1`, commit `11c07fe`, FoR val-B, validation only): evidence under `docs/audit_evidence/stage10_risk/` (policy, run and diagnostics reports, export manifest, byte-preserving `.gitattributes`, verification README) and `docs/results/risk_policy_fit.md` (policy, derived band edges, descriptive diagnostics, Colab CPU-vs-T4 cache finding).
- Stage 10a-2 part 2: Colab notebook 05 for the official risk-policy run (full-SHA clean checkout, Python 3.13 and torch 2.11.0 asserted, no GPU, JUnit skip check, val-A/val-B log-Mel caches built twice and compared, bundle-seed checkpoint copied from Drive, `src.tuning.risk_run` invoked once under the pre-declared rerun rule, no W&B, hashed Drive export re-verified after remount), its static test `tests/governance/test_notebook_05_stage10.py`, and `tests/features/test_logmel_cache_npz_bytes.py` (cache `.npz` bytes do not depend on the code commit).
- Stage 10 risk-policy runner `src/tuning/risk_run.py`: bundle-seed checkpoint bound to the committed Stage 8 hash, provenance checks and exact val-A EER reproduction before val-B, val-B fit (CPU scores and refit checked identical, OS-independent bytes), pre-declared descriptive diagnostics; design note `docs/risk_run.md` and 15 synthetic tests; no real data.
- D45 (approved, PR #57): only the bundle-seed CNN gets a risk policy; seeds 1–2 store raw scores only (calibrated p and prediction not applicable); clarifies Protocol §7 / D41 before any val-B score.
- D43 (approved, PR #53): Stage 16 loudness diagnostic, exploratory in motivation and fully specified before test access (within-class Spearman of score vs RMS on the Stage 11 per-clip scores); implementation only at Stage 16, after the D41 storage implementation.
- Stage 8 PR D: official CNN run evidence (`docs/audit_evidence/stage8_cnn/`: three reports, export manifest, verification README), validation results `docs/results/cnn_selection.md` (4 blocks, lr 1e-3, bundle seed 0) and draft briefs 08–09.
- `docs/design/`: clickable analyst-console prototype (design reference for D38 / Stage 18, mock data only) and its README.
- Synthetic clustered EER/ROC-AUC integration tests with count-based oracles and exact invalid-resample accounting (D4/D11).
- Stage 8 PR C: Colab notebook 04 for the official CNN run (Drive only, full-SHA clean checkout, torch 2.11.0 and Python 3.13 asserted, JUnit skip check, train/val-A log-Mel caches built twice and compared, one D30 search, W&B key loaded last, hashed Drive export re-verified after remount) and its static test `tests/governance/test_notebook_04_stage8.py`.
- Stage 8 PR B2: `configs/cnn.yaml` (exactly D30 + D42) with a strict drift guard, `train_one` with D42 early stopping on val-A, an allow-list W&B logger, and the D30 search orchestrator (4 runs, tie rule, seeds 1–2, bundle seed, checkpoints and provenance reports); design note `docs/cnn_training.md`.
- Stage 8 PR B1: torch dependencies and CI (`requirements-cnn.txt`, CPU torch in CI, torch tests fail instead of skipping), D42 CNN, train-only per-Mel-bin normalization, `set_seed`, and a non-pickle safetensors checkpoint with strict JSON and `checkpoint_sha256`; design note `docs/cnn.md`.
- Stage 8 PR A: verified log-Mel cache (1, 64, 197) for train/val-A/val-B (non-pickle npz + strict JSON, bound to F3 and F2; test split refused), with design note and 44 synthetic tests.
- D42 (approved, PR #44): CNN design and training details; W&B setup documented (D40).
- Stage 7 brief `docs/stages/stage07-baseline.md` (D32), in review; teach-back by Khalil and Iheb pending.
- Stage 7 steps 2B–3: official baseline evidence (cache JSON files, selection/threshold/run/subgroup reports, determinism record) and `docs/results/baseline.md` with validation results (val-A EER 0.149%, val-B EER 0.0), interpretation [hypothesis] and limitations.
- Approved D41 (PR #46): purpose-keyed Stage 11/14 registry with a fixed purpose allowlist, Stage 11 clean per-clip scores, and paired Stage 14 robustness comparisons; no FoR test access until the implementation is reviewed.
- Stage 7 step 2A: baseline run orchestrator (train/val-A/val-B caches → fit, non-pickle artifact, selection/threshold/run/subgroup reports) and Colab notebook 03 with byte-identity checks; design note and 13 synthetic tests.
- Stage 7 step 1: verified MFCC feature cache for train/val-A/val-B (non-pickle npz + strict JSON, bound to F3 and F2; test split refused), with design note and 35 synthetic tests.
- D40 (approved, PR #40): W&B logging from Stage 8; baseline provenance via committed reports.
- H7 pre-registered: silence-fraction shortcut test on REAL clips (Stage 16).
- Stage 4 split, Stage 5 F2, and Stage 6 preprocessing retrospective briefs under `docs/stages/`; D32 teach-back reviews pending.
- Corrected official Stage 6 FoR evidence: byte-identical validation runs, passing D18/shortcut checks, and the no-training Tesla T4 benchmark selecting batch size 256.
- D38 / ADR 0008 (approved, PR #40): React + TypeScript analyst interface replaces Streamlit; Overview page as new scope.
- D39 approved after PR #34 review and merge: Stage 6 preprocessing profile v1, F3 identifiers, shortcut/D18 caveats and no-training benchmark.
- Stage 7 MFCC/SVM baseline: train-only scaler, NumPy scorer, deterministic non-pickle artifact, and D37 fitting (16-point grid on val-A, D29 threshold on val-B) with `configs/baseline.yaml`.
- D35 runner implementation requirement: synthetic integration test of shared frozen preprocessing, with trimming and centred padding per 2 s window.
- Proposed D36 (ADR 0009): the MFCC/SVM baseline is scored with the CNN in the single external run, for RQ2; approval pending.
- Deterministic eight-speaker synthetic integration tests for metric tables, averages, filter counts and clustered accuracy intervals.
- Proposed Stage 13 reporting conventions and exact guarded-run access scope (D33–D35, ADR 0007); approval pending.
- Equal-speaker rate summaries with explicit undefined-value handling and synthetic tests (D19 preparation; report convention pending review).
- Per-source-speaker metric tables reusing the evaluation harness, with synthetic tests and usage documentation (D4, D19; Stage 13 preparation).
- Evidence-backed Lab 6 report sections covering team roles, setup, CI, workflow and data rules, with remaining closeout checks listed.
- Khalil's local setup verification, troubleshooting notes and Codex rule-check answer in the Lab 6 setup report.
- Stage briefs 02 (threat model) and 03 (protocol) (`docs/stages/`, D32).
- Stage briefs 00 (setup) and 01 (dataset audit) (`docs/stages/`, D32).
- Stage briefs structure and template (`docs/stages/`, D32).
- Stage 6 preprocessing profile v1 candidate, shared waveform/log-Mel/MFCC pipeline, FoR shortcut and invariance validation, no-training compute benchmark, tests and Colab runner.
- ModelBundle format and verifying loader (hashes, versions, path safety, immutability, provenance).
- Append-only, hash-chained run registry for D31 (claim before run).
- Stage 10 risk policy: Platt calibration with smoothed targets, val-B EER threshold, uncertain band with ≤ 10% deferral, confidence, risk level, strict JSON policy file.
- Stage 7 evaluation harness (EER/AUC, decision metrics, three-way decisions, cluster bootstrap CIs, seed rules) and the scikit-learn 1.9.1 pin.
- Stage 5 metadata-only F2 freeze generator, pinned config, deterministic manifests, hash ledger and tests.
- Shared AI tooling v1 (D27, ADR 0006): `pr-prep` skill, Context7 in `.mcp.json` (key via env var); GitHub MCP excluded.
- Stage 4 FoR-only leakage-aware split code, frozen split config, automated checks and tests.
- `AGENTS.md` as the single AI context for all tools (D22); `CLAUDE.md` imports it.
- Decision register moved to `docs/decisions/README.md`, with ADRs 0001–0005 (D24).
- `CONTRIBUTING.md`, PR and issue templates aligned with the CS 488 template.
- CI (GitHub Actions): `pytest` on Python 3.13 for PRs and pushes to `main`.
- Governance test: DEEP-VOICE paths only in D20-permitted files.
- `.gitattributes`, `.editorconfig`, `.env.example`, setup / docs-check / structure scripts,
  pre-push docs reminder.
- `docs/architecture.md`, `docs/setup.md`, `docs/README.md`, `docs/setup-report.md` (skeleton).
- MIT `LICENSE` for the code only (D25).
- Khalil Dridi in the team line (`AGENTS.md`, `README.md`), `LICENSE` and `.github/CODEOWNERS`.
- D26 work distribution (Execution Plan v2.1) in the decision register.
- Owner column (D26) in the `AGENTS.md` roadmap.
- `docs/reference/Execution_Plan_v2.1.pdf`.
- `docs/experimental_protocol.md` v0.1: Stage 3 skeleton (owners, reviewers, items to decide).
- `docs/experimental_protocol.md` v1.0 frozen (F1): all sections decided from the team-accepted defaults.

### Changed
- The six PlantUML figures under `docs/uml/` now follow the SDD's visual grouping more closely:
  layered component views, report-style class/sequence legends and no implementation-status labels;
  canonical UML semantics are unchanged.
- `docs/design/`: analyst-console prototype v4 (proposed final design reference; model output moved to Technical details) and `HANDOFF.md` (developer handoff and verification report).
- `docs/design/`: analyst-console prototype v3 (second UI/UX review, aligned with `risk-v1`); README replaced (the Entry 84 copy was an unrelated file).
- Stage 10 closed: brief accepted after the D32 teach-back (Iheb, Malak, Khalil); Stage 11 current.
- Stages 8–9 closed: briefs 08–09 accepted after the D32 teach-back (Iheb, Malak, Khalil; no corrections); Stage 10 current.
- Stage 7 closed: brief accepted after the D32 teach-back (Khalil, Iheb); Stages 8–9 current.
- Stage 6 closed: briefs 04–06 accepted (PR #37); Stage 7 current. Register statuses D29–D31 and Protocol §14 rows now cite PRs #16 and #30. CONTRIBUTING: check open PRs before claiming IDs. D37 status recorded as approved (PR #30).
- Stage 6 Mel upper frequency is 7.6 kHz, keeping the final band below the 16 kHz Nyquist edge after the first official candidate exposed D18 resampling instability at 8 kHz.
- Stage 6 shortcut validation now measures post-trim zero-padding, screens its class separability and reports padding by class and F2 subgroup.
- Stage 6 Colab runner now supports a manual input/evidence transfer mode when Drive credential propagation is unavailable.
- Docs aligned with D38: `architecture.md`, ADR 0002 (Dashboard row) and the `AGENTS.md` roadmap now name the React web app; ADR 0008 accepted.
- Baseline tuning pre-declared: MFCC standardization on FoR train, RBF-SVM C × gamma grid, val-A EER selection, decision_function score (D37, proposed).
- Refined proposed D33/D35 after Donia's review: existing bootstrap convention, dataset-wide run key and durable registry evidence; approval still pending.
- Stage 5 closed after PR #15 review and merge: D28 approved, F2 effective, Stage 6 current.
- Band rule moved to the core (`src/risk/decisions.py`); the evaluation harness re-exports it.
- Protocol clarifications D29–D31 (baseline threshold, CNN search space with seed and tie rule, Stage 11 run once) and the Protocol §14 row.
- Repo transferred to `kahlanzzi/voice-spoofing-detection` (2026-10-14); repo URLs updated.
- `requirements.txt` pinned to exact versions (Python 3.13 + 3.14).
- Branch naming `<type>/stage<N>-<name>` (D23).
- Decision register header no longer limited to the readiness audit.
- `AGENTS.md`: branch-protection line (required check `checks`, branch up to date); Execution Plan pointer → v2.1.
- `Execution_Plan.md` marked superseded by v2.1 (kept for history).
- Issues tagged by part (A–E) and owner (D26) instead of track (`CONTRIBUTING.md`, feature template).
- `data/README.md`: Drive tree matches the real layout.
- D26 completed with the v2.1 §5 hand-overs; per-folder `CODEOWNERS` (owner + consumer reviewer); project log header updated.

### Fixed
- `docs/risk_policy.md` Why (d): the confidence bound for uncertain cases is `max(threshold + half_width, 1 − (threshold − half_width))`, not "about 0.5 + half_width" (0.997 for `risk-v1`, so an uncertain case can display 1.00); documentation only, no behaviour change.
- D44 (approved, PR #55): Newton-decrement stop (λ²/2 ≤ 1e-12), stall and non-descent guards in the Platt fit; tests T1–T5; reproducible pre-check in `docs/audit_evidence/stage10_d44/`; no val-B accessed.
- `tests/` is now a regular package (`tests/__init__.py`) with an import-root guard (`tests/test_import_roots.py`), so a site-packages `tests` package on Colab can no longer shadow it and break `from tests._torch import ...`.
- Stale `CLAUDE.md` mentions → `AGENTS.md` in `configs/audit.yaml` and the Stage 1 notebook intro.
- Corrected Stage 6 brief and log references after review: D39 batch-size source, padding scopes, approval entry, duplicate Entry 59 disclosure, Entry 38 spacing, and current branch/status.

## 2026-10-06 — Stages 0–2

### Added
- Repo scaffold (Stage 0), dataset audit tools and evidence (Stage 1), threat model v1.0 (Stage 2).
