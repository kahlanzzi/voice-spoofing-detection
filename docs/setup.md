# Full Setup Guide

Everything a new teammate (or a reviewer) needs to run the project from scratch.

## Prerequisites

- Git. **Windows:** use **Git Bash** (installed with Git for Windows) to run the `scripts/*.sh`
  files; PowerShell and cmd cannot run them.
- Python **3.13** (official, D15; Colab runtime 3.13.15; CI). Python 3.14 is allowed locally for
  editing and quick tests.
- A Google account with access to the team's shared Drive folder `voice-spoofing-detection/`.
- Optional: VS Code (recommended extensions in `.vscode/extensions.json`).

## Local steps

```bash
# 1. Clone
git clone https://github.com/kahlanzzi/voice-spoofing-detection.git
cd voice-spoofing-detection

# 2. One-time project setup (git hooks: docs reminder before push)
./scripts/setup.sh                 # Windows: run in Git Bash

# 3. Environment variables (placeholders only; never commit .env)
cp .env.example .env
# then edit .env: KAGGLE_API_TOKEN, HF_TOKEN, WANDB_API_KEY

# 4. Virtual environment + dependencies (exact pins in requirements.txt)
python -m venv .venv
source .venv/bin/activate          # Git Bash on Windows: source .venv/Scripts/activate
                                   # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 5. Run the tests (synthetic data only; no dataset needed)
pytest -q
```

## CNN dependencies (Stage 8)

The CNN code (D42) needs PyTorch, safetensors and wandb. Locally, after `requirements.txt`:

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cnn.txt
```

- `torch==2.11.0` is not in any requirements file. Colab uses its preinstalled CUDA build (the
  notebook asserts 2.11.0); CI and local development install the CPU build from the command above.
- CI sets `REQUIRE_TORCH_TESTS=1`, so a torch test fails instead of skipping if torch is missing.
  Locally without torch, those tests are skipped.

## API accounts (Stage 17, D47)

Analyst accounts exist only through the admin command (`docs/api_design.md` §4.8). It uses the
database and password limits of `configs/api.yaml` (or `$VSD_API_CONFIG`). Passwords are typed twice
at a hidden prompt, never on the command line.

```bash
python -m api.admin create-account analyst01   # also: set-password, disable, enable, unlock
```

`set-password` and `disable` end all of the account's sessions. `unlock` needs the sign-in key
`VSD_SIGNIN_HMAC_KEY` (make it with the command in `.env.example` and put it in `.env`); load `.env`
into the shell first:

```bash
set -a; . ./.env; set +a
python -m api.admin unlock analyst01
```

## Colab

1. Open a notebook from `notebooks/` in Colab (Python 3.13 runtime, GPU when training).
2. The first cell clones the repo at a given branch, prints the Python version and commit hash, and
   runs `pytest`. If the runtime lacks a package, run `pip install -r requirements.txt` first.
3. Mount Drive. Datasets live in `MyDrive/voice-spoofing-detection/raw/...`; **copy them to
   `/content/data/` before heavy I/O** (`AGENTS.md`, Compute and data).
4. Data roots and output folders come from `configs/*.yaml` or CLI arguments. Never hardcode a
   Colab or Drive path in `src/`.
5. Put tokens in Colab secrets or environment variables, never in a notebook cell.

## Weights & Biases (D40)

- W&B accounts sign in through CoreWeave Forge (`forge.coreweave.com`); the SDK and `api.wandb.ai`
  are unchanged.
- Entity `voice-spoofing-detection-must`, project `voice-spoofing-detection` (private, Academic plan).
- Each member uses their own API key, stored only in `.env` locally and in Colab Secrets (key name
  `WANDB_API_KEY`) in notebooks; never in code, cells, commits or prompts.
- W&B receives FoR training and validation metrics and configs only; never audio, test-split or
  external data.
- W&B Training, Sandboxes, Inference and the ARIA agent are not used: data must not leave Colab and
  Drive (D27 for AI tools).

## Datasets and pins

| Dataset | Role | Pin (D21, `configs/audit.yaml`) |
|---|---|---|
| FoR-2sec (Hugging Face `UncovAI/FOR-2sec`) | training, validation, internal test | HF revision `ff8c82c7…9055` |
| DEEP-VOICE (Kaggle) | external test, **touched once** (hard rule 1) | zip SHA-256 `8cb25853…bdcf` |

The audit notebook refuses to run on unpinned or changed data. Pins change only through a new
decision (D21). DEEP-VOICE: only the accesses listed in D20 before the single guarded external run;
`tests/governance/test_deepvoice_access.py` fails CI if its paths appear elsewhere. Datasets are
never committed or redistributed (D25).

## Tests and CI

- `pytest -q` from the project venv before every PR; paste the real output in the PR.
- GitHub Actions (`.github/workflows/ci.yml`) runs `pytest -q` on Python 3.13 for every PR and every
  push to `main`, then prints the docs reminder. It never downloads datasets.

## Troubleshooting

- **Problem:** `./scripts/setup.sh: Permission denied` or `bash\r: No such file or directory`.
  **Fix:** run `bash scripts/setup.sh`; line endings are fixed to LF by `.gitattributes` (re-clone
  or `git add --renormalize .` if an old checkout still has CRLF).
- **Problem:** `soundfile` cannot find `libsndfile` on Linux.
  **Fix:** `pip install --force-reinstall soundfile` (the wheel bundles the library), or
  `apt-get install libsndfile1`.
