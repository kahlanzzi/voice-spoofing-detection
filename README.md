# Voice-Spoofing Detection for Fraud Prevention

> A **decision-support** control that flags short speech recordings as authentic / synthetic /
> uncertain, with a confidence score and a LOW/MEDIUM/HIGH risk level.

**Team:** Donia Mabrouk, Iheb Zemzemi, Malak Ben Salem, Khalil Dridi
**Course:** Semester AI project (MUST University); repository structure from the CS 488 template
**Status (2026-11-01):** Stages 0–10 are complete; Stage 11 is current. Its runtime-plan
[PR #67](https://github.com/kahlanzzi/voice-spoofing-detection/pull/67) and runner
[PR #68](https://github.com/kahlanzzi/voice-spoofing-detection/pull/68) are under review.
The API skeleton and mock-data UI prototype are merged; production frontend/backend integration
is not yet present on `main`. See [current status](docs/project_log.md) and the
[prioritized backlog](docs/midterm/backlog.md).

---

## What it does

Fraud analysts receive voice recordings that may be synthetic. This project trains one CNN on
log-Mel spectrograms of **FoR-2sec** (TTS speech), compares it with an MFCC + RBF-SVM baseline, and
evaluates the frozen system **once** on **DEEP-VOICE** (RVC voice conversion) to measure
cross-dataset generalization.

**This is not proof of fraud and not a biometric authenticator.** Every output is a signal for a
human analyst to weigh alongside other evidence, never an automated accept/reject decision.

## Tech stack

Python 3.13 (D15) · librosa, soundfile, NumPy, pandas · PyTorch CNN on log-Mel spectrograms
(intended deployed model) and MFCC + RBF-SVM baseline (evaluation only, D7) · FastAPI (API) and
React + TypeScript (approved frontend, D38; implementation pending on `main`) ·
Colab / Kaggle GPUs · Weights & Biases · pytest + GitHub Actions. Details: ADR
[0002](docs/decisions/0002-tech-stack.md), its frontend replacement
[0008](docs/decisions/0008-web-frontend-react.md), and [architecture](docs/architecture.md).

## Getting started

Commands below use Bash (Git Bash on Windows). See [the setup guide](docs/setup.md) for
PowerShell activation and environment details.

```bash
# 1. Clone and enter the repo
git clone https://github.com/kahlanzzi/voice-spoofing-detection.git
cd voice-spoofing-detection

# 2. One-time setup (installs git hooks; Windows: run in Git Bash)
./scripts/setup.sh

# 3. Copy environment placeholders (never commit .env; tokens are not needed for tests)
cp .env.example .env

# 4. Install dependencies
python -m venv .venv
source .venv/bin/activate        # Windows (Git Bash): source .venv/Scripts/activate
pip install -r requirements.txt

# 5. Install CNN test dependencies (same CPU approach as CI)
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cnn.txt

# 6. Run the full suite; do not silently skip missing torch tests
REQUIRE_TORCH_TESTS=1 python -m pytest -q
```

Datasets are not stored in this repo. They live on the team's shared Google Drive, pinned by D21;
see [docs/setup.md](docs/setup.md) and [data/README.md](data/README.md).

## What can run from this snapshot

| Component | On merged `main` | How to inspect it |
|---|---|---|
| Research core | Preprocessing, baseline/CNN code, training, evaluation, risk policy, bundle loader and synthetic tests | [Setup](docs/setup.md), [stage briefs](docs/stages/), [recorded results](docs/results/) |
| API | Stage 17 API with config, request IDs, error handling, schemas, history, notes, stats and sign-in (`/api/session`) | Put `VSD_SIGNIN_HMAC_KEY` in `.env`, run `python -m uvicorn api.main:app --port 8001 --env-file .env`, then open `http://127.0.0.1:8001/api/health`; or `docker compose up -d` (API only, no published port yet); see [setup](docs/setup.md) and the [approved contract](docs/api_design.md) |
| UI | Self-contained HTML design prototype with mock data, no backend/model connection | Open `docs/design/analyst-console-prototype.html` in a browser; follow the [prototype guide](docs/design/README.md) |
| Production upload/inference flow | Not implemented end to end on this snapshot | Part E implementation remains in progress; the [backlog](docs/midterm/backlog.md) tracks its requirements |

The API status response explicitly contains `mock: true`; it does not establish real inference
or availability of history/statistics. The approved React `web/` implementation and Docker
Compose deployment are still pending on `main`; `dashboard/` is a legacy empty placeholder.
Iheb is working on the frontend/backend integration; unpublished work is not counted as merged.

Tests use synthetic fixtures and do not need dataset downloads or account tokens. Dataset-backed
notebooks are research workflows, not the application quick start: their existing protocol,
freeze and one-run gates still apply. Do not run the FoR test split or the external dataset to
try out the UI or measure application readiness.

## Midterm handoff

The [prioritized backlog](docs/midterm/backlog.md), [UML package](docs/uml/),
[AI ethics reflection](docs/ai-usage-reflection.md), and Khalil’s
[Part 6 report](docs/refactor-optimization-report.md) / [PDF](docs/refactor-optimization-report.pdf)
are committed. Part 6 post-merge status wording was corrected in merged
[PR #74](https://github.com/kahlanzzi/voice-spoofing-detection/pull/74).

For the team submission, independently verify the README setup, record an accessible demo,
and clearly identify real and mocked behavior. The repository does not yet demonstrate a
complete upload-to-real-result MVP. Editorial acceptance, the demo and LMS submission must be
confirmed by the team; their completion is not inferred from merged documentation.

## Project structure

<!-- STRUCTURE:START (auto-generated by scripts/update-structure.sh — do not edit by hand) -->
```
.claude/
    skills/
.editorconfig
.env.example
.gemini/
    settings.json
.gitattributes
.githooks/
    pre-push
.github/
    CODEOWNERS
    ISSUE_TEMPLATE/
    pull_request_template.md
    workflows/
.gitignore
.mcp.json
.vscode/
    extensions.json
AGENTS.md
AI_CONTEXT.md
CLAUDE.md
CONTRIBUTING.md
LICENSE
README.md
api/
    __init__.py
    config.py
    errors.py
    main.py
    middleware.py
    routers/
    schemas.py
configs/
    api.yaml
    audit.yaml
    baseline.yaml
    cnn.yaml
    freeze.yaml
    preprocessing.yaml
    split.yaml
dashboard/
    .gitkeep
data/
    README.md
    manifests/
docs/
    CHANGELOG.md
    README.md
    ai-usage-reflection.md
    api_design.md
    architecture.md
    audit_evidence/
    baseline.md
    baseline_run.md
    clip_scores.md
    cnn.md
    cnn_training.md
    dataset_audit.md
    decisions/
    design/
    evaluation_harness.md
    experimental_protocol.md
    feature_cache.md
    logmel_cache.md
    midterm/
    model_bundle.md
    project_log.md
    refactor-optimization-report.md
    refactor-optimization-report.pdf
    reference/
    results/
    risk_policy.md
    risk_run.md
    setup-report.md
    setup.md
    speaker_analysis.md
    speaker_average.md
    stages/
    synthetic_reporting.md
    threat_model.md
    uml/
notebooks/
    01_stage1_dataset_audit.ipynb
    02_stage6_preprocessing_profile.ipynb
    03_stage7_baseline.ipynb
    04_stage8_cnn.ipynb
    05_stage10_risk_policy.ipynb
pytest.ini
requirements-cnn.txt
requirements.txt
scripts/
    check-docs.sh
    setup.sh
    update-structure.sh
src/
    __init__.py
    audit/
    bundle/
    evaluation/
    features/
    freezing/
    models/
    preprocessing/
    risk/
    splitting/
    training/
    tuning/
tests/
    __init__.py
    _torch.py
    api/
    audit/
    bundle/
    evaluation/
    features/
    freezing/
    governance/
    models/
    preprocessing/
    risk/
    splitting/
    test_import_roots.py
    test_torch_guard.py
    training/
    tuning/
```
<!-- STRUCTURE:END -->

## Documentation

- [Project brief and hard rules](AGENTS.md): read first (humans and AI tools)
- [Architecture](docs/architecture.md): how the system is put together
- [Setup details](docs/setup.md): full environment setup (local, Colab, Drive)
- [Decisions](docs/decisions/README.md): register (D-IDs) and ADRs
- [Project log](docs/project_log.md): what was done, open items, current status
- [Changelog](docs/CHANGELOG.md): what changed, when
- [Contributing / team workflow](CONTRIBUTING.md): branches, commits, PRs, AI use
- [Docs index](docs/README.md): component guides, evidence, design and submission documents

## Team workflow (short version)

1. Never push to `main`. Create a branch: `git checkout -b <type>/stage<N>-<name>` (D23).
2. Commit small, with Conventional Commits messages.
3. Open a Pull Request; at least one teammate reviews before merge. AI agents never merge.
4. CI must be green before merging.

Full details in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Code: [MIT](LICENSE) (D25). Datasets are **not** part of this repository and are never
redistributed; FoR-2sec and DEEP-VOICE remain under their own terms (see
[docs/dataset_audit.md](docs/dataset_audit.md) §1).
