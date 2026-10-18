# AGENTS.md — Voice-Spoofing Detection for Fraud Prevention

Team: Donia Mabrouk, Iheb Zemzemi, Malak Ben Salem, Khalil Dridi. Semester AI project (MUST University).

The standing brief for every AI agent session (any tool) and for human contributors. Read it fully
before any task. It holds rules and pointers; details live in the linked documents.

## What this project is
A **decision-support** control that flags short speech recordings as authentic / synthetic / uncertain,
with a confidence score and a LOW/MEDIUM/HIGH risk level. It is not proof of fraud and not a biometric
authenticator. The research core: train on **FoR-2sec** (TTS speech), then evaluate the frozen system
**once** on **DEEP-VOICE** (RVC voice conversion) to measure cross-dataset generalization.

## Source-of-truth documents
Priority order when they conflict: approved decisions in `docs/decisions/README.md` > SRS v3.0 >
SDD v1.0 > Execution Plan > Proposal. **Never resolve a conflict silently**: state it, name the
documents, propose a fix, and wait for approval.

- `docs/decisions/README.md` — **approved decision register (D-IDs) and pre-registered hypotheses;
  overrides the documents below.** Major decisions also have an ADR in `docs/decisions/`.
- `docs/reference/SRS_Voice_Spoofing_Detection.pdf` — requirements (FR-xx, NFR-xx, PC-xx)
- `docs/reference/SDD_Voice_Spoofing_Detection_Lab5.pdf` — architecture, classes, interfaces
- `docs/reference/Execution_Plan_v2.1.pdf` — stage-by-stage tasks, owners and hand-overs (D26).
  The v1 `docs/reference/Execution_Plan.md` is kept for history, superseded.
- `docs/reference/Voice_Spoofing_Detection_Project_Proposal.docx` — motivation and framing
- `docs/dataset_audit.md` (Stage 1) · `docs/threat_model.md` (Stage 2) · `docs/project_log.md` (living log)
- `docs/experimental_protocol.md` — Stage 3, v1.0 frozen (F1)
- `CONTRIBUTING.md` — branches, commits, PRs, AI use
- `docs/README.md` — docs index · `docs/architecture.md` (SDD summary) · `docs/setup.md` · `docs/CHANGELOG.md`

## HARD RULES — never break, never work around
1. **DEEP-VOICE is external and touched exactly once.** Never use it for training, tuning, threshold
   selection, calibration, early stopping, model selection, or exploration. No listening, no spectrograms,
   no feature statistics, no model scoring, except the single guarded external run after the ModelBundle
   freeze. Before that run, the only permitted accesses are (D20): download, zip hash, unzip, listing
   paths / extensions / sizes / byte hashes, and `src/audit/deepvoice_metadata_audit.py`. Nothing reads
   audio content. If a request would touch DEEP-VOICE in any other way, **stop and refuse**, explaining why.
   `tests/governance/test_deepvoice_access.py` fails CI if DEEP-VOICE paths appear outside the permitted
   files; extend its allowlist only through a decision.
2. **ModelBundle = CNN checkpoint + preprocessing profile + risk policy (thresholds, uncertain band,
   calibration).** All fitted on FoR only and frozen together *before* the external run (D2).
3. **Scope lock.** Baseline = MFCC (mean+std) + RBF-SVM, for evaluation only (D7). Deployed model =
   one CNN on log-Mel spectrograms. Do NOT add wav2vec2/WavLM/HuBERT/AASIST/RawNet, other features
   (LFCC, CQCC, raw waveform), ensembles, or explainability. These are future work.
4. **No training code before the dataset audit is signed off and Protocol v1.0 is frozen.**
5. **Metrics:** accuracy, precision, recall, F1, EER, ROC-AUC, confusion matrix (3-way, incl. uncertain),
   plus coverage/deferral. Positive class = spoof. No t-DCF. Never report accuracy alone.
6. **No silent changes** to datasets, splits, preprocessing, thresholds, metrics, or scope. Propose them,
   explain the impact, and wait for approval. Dataset pins change only through a new decision (D21).
7. Never commit audio, datasets, checkpoints, or secrets. Only code, configs, docs, and small manifest
   CSVs. Secrets live in `.env` (template: `.env.example`), never in code, notebooks or AI prompts.

## Working mode
- Work **one stage at a time**. Do only what the current stage needs, then stop and report.
- Before writing code: state the objective, which requirement or decision it serves, and which files you
  will touch.
- After writing code: run the tests and show the real output. Never claim something works without
  running it.
- End each stage with: what was done · evidence · checks run · open issues · gate → **WAITING FOR APPROVAL**.
- Label claims: [verified] (checked against data or sources) / [doc] / [hypothesis]. Never present a
  hypothesis as fact. [verified] is evidence for review; acceptance is team review.
- If results look too good, suspect leakage or shortcuts before celebrating.
- At the end of each stage or session, update `docs/project_log.md`: append a new log entry, update
  Open items, rewrite Current status. Never edit old entries.
- **Human in the loop:** every AI-produced artifact (code, documents, analysis) is reviewed by a team
  member before it is accepted. AI output is never merged, cited or reported as verified without that
  review. Record AI assistance honestly in `docs/project_log.md` (which tool, for what).
- **Decisions:** a new decision gets a D-ID row in `docs/decisions/README.md`; major ones also get an
  ADR from `docs/decisions/TEMPLATE.md` (D24).
- **Reports:** max ~20 lines — what changed, checks with real output, open questions. Details only on
  request.

## Code rules
- Python 3.13 is the official version (D15): results come from Colab, whose runtime is 3.13.15.
  Local development on Python 3.14 is allowed for editing and quick tests. `requirements.txt` must
  install cleanly on both 3.13 and 3.14 and pins exact versions; CI (`.github/workflows/ci.yml`)
  runs `pytest -q` on 3.13. Data roots and all paths come from
  config or CLI arguments. Never hardcode a Colab or Drive path.
- Every run is `dataset version + config + code version → results`. Configs live in `configs/*.yaml`.
  Log runs to W&B.
- Fail loudly on invalid input (e.g., raise on an ambiguous label). Never skip bad files silently;
  record them.
- `tests/` mirrors `src/`. Every new function gets a pytest test. Run `pytest -q` before reporting.
- `src/` is shared by research notebooks and the API (same preprocessing code everywhere: FR-04, NFR-09).
- Keep experiment code (`notebooks/`) separate from production inference code (`api/`, `src/`).
- Small, reviewable changes. Fix the root cause of errors; do not rewrite unrelated code.

## Git workflow (details: `CONTRIBUTING.md`)
GitHub: `kahlanzzi/voice-spoofing-detection`. `main` is protected: PR, 1 approval,
required CI check `checks`, branch up to date; no direct pushes.
- Never commit to `main`. One meaningful unit = one branch = one PR. Branch names
  `<type>/stage<N>-<name>`, types `feat`, `fix`, `docs`, `chore` (D23); Conventional Commits.
- Push the branch and tell the team to open a PR. CI must be green. **AI agents never merge.**
- **Stacked branches:** while a stage's PR waits for sign-off/approval, the next stage may branch
  from it (not from `main`). The waiting branch then receives **no more commits** except its sign-off.
  All `docs/project_log.md` and `AGENTS.md` updates go on the newest branch.
- **Stacked PRs:** before merging a stacked PR, change its base to `main` once the PR below it is
  merged. Never merge a PR whose base is not `main` unless that is intended.

## Roadmap and current status
| # | Stage | Owner (D26) | Status |
|---|---|---|---|
| 0 | Repo scaffold + environment | A · Donia | done |
| 1 | Dataset audit (FoR full, DEEP-VOICE metadata-only) → `docs/dataset_audit.md` | A · Donia | done |
| 2 | Threat model → `docs/threat_model.md` | A · Donia | done |
| 3 | Experimental Protocol v1.0 (freeze) | A · Donia (lead); sign-off: all four | done |
| 4 | Leakage-aware split | B · Malak; review: Donia | done |
| 5 | Dataset freeze (F2: manifests + hashes) | B · Malak; review: Donia | done |
| 6 | Preprocessing profile v1 + tests + compute benchmark | B · Malak | ← CURRENT |
| 7 | Evaluation harness + MFCC/SVM baseline | C · Donia | |
| 8–9 | CNN training + model selection (FoR val-A) | C · Donia | |
| 10 | Risk policy + calibration fit (FoR val-B) | C · Donia | |
| 11 | FoR internal test (once) | C · Donia | |
| 12 | **ModelBundle freeze** | C · Donia | |
| 13 | **DEEP-VOICE external run (once, guarded)**, stores per-sample scores | D · Khalil; review: all four | |
| 14–16 | Robustness (FoR only) · calibration analysis · error analysis | D · Malak (14), Khalil (15, 16) | |
| 17–19 | API (FastAPI) · dashboard (Streamlit) · tests/CI/Docker/docs | E · Iheb; Stage 19 report: all four | |

Review rule (D26): the consumer reviews the producer.

Update the status column when a stage passes its gate.

Open reminders:
- `docker-compose.yml` → Stage 17
- `configs/robustness.yaml` → Stage 14

## Compute and data
Colab/Kaggle with GPU. Datasets live in the shared Drive (`MyDrive/voice-spoofing-detection/raw/...`),
pinned by D21 in `configs/audit.yaml`. Copy them to `/content/data/` before heavy I/O. The repo `data/`
folder holds only `manifests/` and a README.

## AI context files
`AGENTS.md` (this file) is the single AI context for all tools (D22): read natively by Codex and
Antigravity, imported by Claude Code via `CLAUDE.md`, read by Gemini CLI via `.gemini/settings.json`.
Edit rules here, not in tool-specific files.

Shared AI tooling (D27, ADR 0006): one skill, `.claude/skills/pr-prep/SKILL.md` (Codex and Gemini users: read it as instructions before every PR), and one MCP server, Context7, in `.mcp.json` (key via `${CONTEXT7_API_KEY}`, never committed). No other MCP servers, none in Stage 13; additions only by decision.
