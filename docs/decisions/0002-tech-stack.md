# 0002. Tech stack

- **Status:** Accepted
- **Date:** 2026-10-11 (recorded; Python version decided as D15)
- **Register:** D15
- **Deciders:** Donia Mabrouk, Iheb Zemzemi, Malak Ben Salem

## Context

The SRS and SDD describe a REST API, an analyst dashboard, a CNN detector and an evaluation harness.
Results must be reproducible (`dataset version + config + code version → results`) on free compute.
The stack was set across the Execution Plan, the `AGENTS.md` roadmap and D15; this ADR collects it.

## Decision

| Area | Choice | Source |
|---|---|---|
| Language | **Python 3.13** (Colab runtime 3.13.15); local 3.14 allowed for editing and quick tests; CI on 3.13 | D15 |
| Deep learning | **PyTorch** (CNN on log-Mel spectrograms) | Execution Plan Phase 0 (0.10) and Phase 4 |
| API | **FastAPI** (`POST /analyze-audio`) | Execution Plan Phase 7; SRS FR-02; roadmap Stages 17–19 |
| Dashboard | **Streamlit** (the Execution Plan also allows React; the roadmap fixes Streamlit). **Superseded by [ADR 0008](0008-web-frontend-react.md) (D38): React + TypeScript web app.** | Execution Plan Phase 8; roadmap Stages 17–19 |
| Compute and data | **Google Colab** (GPU) + shared **Google Drive** for datasets and archives; Kaggle as fallback | Execution Plan "Data & Compute Workflow" |
| Experiment tracking | **Weights & Biases** | Execution Plan Phase 0 (0.6) |

## Consequences

- `requirements.txt` must install on 3.13 and 3.14 (D15). PyTorch, FastAPI, Streamlit and wandb are
  **not yet** in `requirements.txt`; each is added in the stage that first needs it.
- Paths come from config, never hardcoded Colab/Drive paths (`AGENTS.md`, Code rules).
- Changing a component of the stack requires a new ADR.

## Links

D15 in `README.md` · `docs/reference/Execution_Plan.md` · `AGENTS.md` (Roadmap, Code rules)
