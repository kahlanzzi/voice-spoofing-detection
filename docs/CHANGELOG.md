# Changelog

All notable changes to this project. Newest at the top. Keep entries short and human: one line per
change. Based on [Keep a Changelog](https://keepachangelog.com). Detailed history with evidence is
in `project_log.md`.

Categories: **Added**, **Changed**, **Fixed**, **Removed**.

## [Unreleased]

### Added
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
- Stale `CLAUDE.md` mentions → `AGENTS.md` in `configs/audit.yaml` and the Stage 1 notebook intro.

## 2026-10-06 — Stages 0–2

### Added
- Repo scaffold (Stage 0), dataset audit tools and evidence (Stage 1), threat model v1.0 (Stage 2).
