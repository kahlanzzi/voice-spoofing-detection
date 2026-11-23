# Voice-Spoofing Detection for Fraud Prevention

> A **decision-support** control that flags short speech recordings as authentic / synthetic /
> uncertain, with a confidence score and a LOW/MEDIUM/HIGH risk level.

**Team:** Donia Mabrouk, Iheb Zemzemi, Malak Ben Salem, Khalil Dridi
**Course:** Semester AI project (MUST University); repository structure from the CS 488 template
**Status (2026-11-15):** Stage 12 (F4) is complete; Track V (model v2) is current. V1b corpus and
tool pins are closed after [PR #134](https://github.com/kahlanzzi/voice-spoofing-detection/pull/134);
V1c is next. Khalil's Stage 13 runner preparation merged in
[PR #133](https://github.com/kahlanzzi/voice-spoofing-detection/pull/133), but the external run is
not ready: the applicable model set must be frozen under D48, the exact-runtime D58 rehearsal and
durable Drive claim must be verified, and the final review gates must pass. The Docker MVP still
uses a mock analyzer; F4 integration in the API is Iheb's P1 work. See [current status](docs/project_log.md),
the [docs index](docs/README.md), and [Execution Plan v3.0](docs/reference/Execution_Plan_v3.0.md).

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
React + TypeScript (approved frontend, D38; MVP on `main`, mock analyzer) ·
Colab / Kaggle GPUs · Weights & Biases · pytest + GitHub Actions. Details: ADR
[0002](docs/decisions/0002-tech-stack.md), its frontend replacement
[0008](docs/decisions/0008-web-frontend-react.md), and [architecture](docs/architecture.md).

## MVP: install and run

The MVP runs in Docker at http://localhost:8001: sign in → upload a recording → read the verdict →
find it in History. **It uses a mock analyzer, so its verdicts are not model results**; see
[Mock results](#mock-results). To work on the code instead, see [Getting started](#getting-started).

### Prerequisites

| Needed for | Tool | Tested version |
|---|---|---|
| Running the MVP | Docker, with Compose v2 | Docker 28.5.2, Compose 2.40.3 |
| Running the MVP | Git and `openssl` | any recent version |
| Tests only | Node, exact version (`web/.nvmrc`) | 26.13.0 |
| Tests only | Python and a virtual environment | 3.13 (official, D15); 3.14 is allowed locally |

Node and Python are not needed to run the MVP. These steps were tested on Linux only.
On Windows, run the commands in Git Bash, not PowerShell.

```bash
docker --version
docker compose version
git --version
openssl version
```

### 1. Get the code

```bash
git clone https://github.com/kahlanzzi/voice-spoofing-detection.git
cd voice-spoofing-detection
```

> **Warning: the folder name decides whose data you are using.** Docker Compose names a project
> after its folder. Two clones in folders with the same name, even in different places, count as
> one project and share one database. `docker compose down -v` in either one deletes that database.
> If you already have a clone, give the new one a different folder name:
> `git clone https://github.com/kahlanzzi/voice-spoofing-detection.git vsd-second`

### 2. Create the secrets

The stack refuses to start without three random hex values of at least 32 bytes each:

| Variable | Used by | What it is |
|---|---|---|
| `VSD_SIGNIN_HMAC_KEY` | api | the key for failed sign-in counts |
| `VSD_DB_PASSWORD` | db only, at its first start | the PostgreSQL superuser; the api never gets it |
| `VSD_DB_APP_PASSWORD` | db (first start) and api | the API's own database role, `vsd_app`, not a superuser |

They live in `.env`, which git ignores; never commit them or paste them anywhere. The commands fill
only empty values, so running them again never changes a password the database already uses.

```bash
cp .env.example .env     # skip this line if you already have a .env: it would overwrite it
grep -q '^VSD_DB_PASSWORD=' .env || printf 'VSD_DB_PASSWORD=\nVSD_DB_APP_PASSWORD=\n' >> .env   # a .env from before PostgreSQL
for name in VSD_SIGNIN_HMAC_KEY VSD_DB_PASSWORD VSD_DB_APP_PASSWORD; do
  sed -i.bak "s/^$name=$/$name=$(openssl rand -hex 32)/" .env && rm .env.bak   # GNU sed (Linux, Git Bash) and BSD sed (macOS)
done
grep -cE '^(VSD_SIGNIN_HMAC_KEY|VSD_DB_PASSWORD|VSD_DB_APP_PASSWORD)=.' .env    # prints 3, without showing them
```

Every `docker compose` command needs this `.env`, `down` included. `docker compose config` prints
the values, so do not share its output. The two database passwords are read once, when the database
volume is created; to change them later, see
[Database](docs/setup.md#database-postgresql-d53).

### 3. Start

```bash
docker compose up -d --build --wait      # the first build takes a few minutes
docker compose ps                        # db and api: Up (healthy); web: Up
curl -s http://localhost:8001/api/health # {"ok":true}
```

Open http://localhost:8001. Use `localhost`, not an IP address: the session cookie needs it.

### 4. Create the first account

Accounts exist only through the admin command; there is no sign-up page. This creates the demo
account `analyst01`:

```bash
docker compose exec api python -m api.admin create-account analyst01
```

It asks for a new password twice, at a hidden prompt, then prints `Account analyst01 created.`
Whoever runs it chooses the password. It needs at least 15 characters and must not contain the
username, `vsd`, `voice` or `spoof`. Never put it in a command, a file or the repository.

### 5. Use the MVP

1. **Sign in** with Analyst ID `analyst01` and the password you chose.
2. **Upload:** on "Analyze a recording", choose a file. The page checks it against the limits below.
3. **Read the verdict:** select **Analyze recording**. The result shows authentic, synthetic or
   uncertain, a LOW/MEDIUM/HIGH risk level, a request ID and the service's notice; the model's output
   value is under **Technical details**.
4. **Open History:** the newest 10 results appear, each with its verdict and risk.

| Limit (from `GET /api/limits`, which needs a session) | Value |
|---|---|
| Accepted formats | wav, flac, mp3 |
| Maximum size | 20 MB |
| Maximum duration | 60 s |
| Audio | Deleted when processing finishes |

These are prototype values in [configs/api.yaml](configs/api.yaml), not final decisions. Only results
are stored: no audio and no original filenames. At most `analysis_workers` analyses run at once; a further
upload gets "service unavailable, try again" (mock value, not measured; replaced by the P4 load-test
decision on the staging host with the real bundle).

The repository holds no audio. Use a recording of your own, or make a 1-second test tone outside the
repository while the stack is running:

```bash
docker compose exec -T api python -c "import io,sys,numpy as n,soundfile as s;b=io.BytesIO();t=n.arange(16000)/16000;s.write(b,0.2*n.sin(2*n.pi*440*t),16000,format='WAV',subtype='PCM_16');sys.stdout.buffer.write(b.getvalue())" > /tmp/vsd-sample.wav
```

Do not use the FoR test split or the external dataset to try the app.

### Mock results

The analyzer is the `MockAnalyzer`: no model runs. The hash of the file's bytes picks the verdict, so
the same file always gives the same mock result. It says nothing about whether the speech is real.
Every such result carries a **Mock result** tag, on the result page and in History.

The real model is not connected. It waits for the team's frozen ModelBundle (Stage 12, Donia) and
its integration; FR-04 inference preprocessing comes with it.

### Not in this MVP

- Overview and Details: placeholder pages. Overview keeps its link in the side menu.
- Analyst notes: History has the column and the result mentions the note, but none can be written.
- History filters: not built.
- History pager: only the newest 10 results can be reached.
- History row links: rows do not open a result.
- Toast: no message when an answer arrives on another page.
- Drag-and-drop: choose the file with the button.
- Copy button: the request ID has none.
- Firefox e2e: not run for each PR; Chromium is.

### Clean database for a demo

The `vsd-pg` volume holds the PostgreSQL database with all accounts, sessions, sign-in lockout
counts, analyses and notes. History is shared by all analysts, so rows from tests appear for
everyone. To start from an empty History:

> **Warning: this deletes everything in `vsd-pg`: every account, session, analysis and note. It
> cannot be undone.** It keeps `vsd-var` (the SQLite store of the API before PostgreSQL), your `.env`
> and the Docker images. Check the folder warning in [step 1](#1-get-the-code) first. Run any e2e
> tests before this, never between this and the demo.

Remove only the PostgreSQL volume, as in
[Empty the database (keep `vsd-var`)](docs/setup.md#empty-the-database-keep-vsd-var):

```bash
docker compose down                                # stop the stack; both volumes are kept
docker compose config | grep -m1 '_vsd-pg$'        # the volume's real name: <project>_vsd-pg
docker volume rm voice-spoofing-detection_vsd-pg   # the name printed above
docker compose up -d --build --wait
docker compose exec api python -m api.admin create-account analyst01
```

The last command reseeds `analyst01`; choose its password again. Signed-in browsers must sign in
again, and History then shows "No analyses yet".

### Stop and reset

> **Warning: `down -v` is a full reset. It deletes `vsd-pg` (every account, session, analysis and
> note) and also `vsd-var`, the SQLite store of the API before PostgreSQL; import it first if you
> still need it ([Database](docs/setup.md#database-postgresql-d53)). It cannot be undone.** First check
> that you are in the right folder and project (the folder warning in [step 1](#1-get-the-code)).

```bash
docker compose down      # stops and removes the containers; keeps vsd-pg, so accounts and results stay
pwd && docker compose ls # the folder and the Compose project that down -v would empty
docker compose down -v   # full reset: deletes vsd-pg and vsd-var
```

### Tests

Tests are not needed to run the MVP.

- **Python:** steps 4 to 7 of [Getting started](#getting-started). The API tests need the test
  database (step 6); without it they are skipped and the run ends with a `WARNING` line.
- **Web, from `web/`:**

  ```bash
  npm ci
  npm run typecheck
  npm run lint
  npm test -- --run
  npm run build
  ```

- **End-to-end, against the running stack on :8001, from `web/`:**

  ```bash
  npm ci
  npx playwright install chromium    # once
  npm run test:e2e
  ```

  You seed nothing by hand: the run creates its own `e2e-*` accounts and never touches `analyst01`.
  It restarts `web`, stops `api` three times, and each green run adds 9 rows to History. Details are
  in [End-to-end tests](docs/setup.md#end-to-end-tests-stage-18).

### Troubleshooting

- **Compose stops with `set VSD_SIGNIN_HMAC_KEY in .env`, `set VSD_DB_PASSWORD in .env` or
  `set VSD_DB_APP_PASSWORD in .env`:** run [step 2](#2-create-the-secrets) in this folder.
- **`api` never becomes healthy after an update, and its log says the database is not reachable:**
  your `vsd-pg` volume may be older than the `vsd_app` role, or `.env` holds different database
  passwords than when the volume was made. See
  [Database troubleshooting](docs/setup.md#database-troubleshooting).
- **Port 8001 is in use:** find what holds it with `docker ps --filter publish=8001`. Often it is
  another clone's stack: run `docker compose down` there, without `-v`.
- **"Sign-in is temporarily unavailable" after several attempts:** nginx allows 10 sign-ins per
  minute per address, with a burst of 5, then answers 429. Wait a minute, or run
  `docker compose restart web`.
- **"This account is temporarily locked":** 5 failed sign-ins lock a username for 15 minutes. To
  clear it now: `docker compose exec api python -m api.admin unlock analyst01`.
- **A change does not show up:** the app and the nginx config are inside the image, so rebuild with
  `docker compose up -d --build --wait`.
- **Requests fail after the `api` container was recreated:** run `docker compose restart web`.
- **"Duration could not be read in the browser":** this does not block the analysis; the service
  checks the file itself. If it happens for every file, the web image is probably old. The output of
  `curl -sI http://localhost:8001/signin` must contain `media-src 'self' blob:`; if not, rebuild.

More detail: [Running with Docker Compose](docs/setup.md#running-with-docker-compose-stage-17),
[nginx](docs/setup.md#nginx-the-web-service), [API accounts](docs/setup.md#api-accounts-stage-17-d47)
and the [API design](docs/api_design.md).

## Getting started

Commands below use Bash (Git Bash on Windows). See [the setup guide](docs/setup.md) for
PowerShell activation and environment details.

```bash
# 1. Clone and enter the repo
git clone https://github.com/kahlanzzi/voice-spoofing-detection.git
cd voice-spoofing-detection

# 2. One-time setup (installs git hooks; Windows: run in Git Bash)
./scripts/setup.sh

# 3. Copy environment placeholders and fill the three secrets, as in step 2 of "MVP: install and run"
#    (never commit .env; the dataset tokens are not needed for tests)
cp .env.example .env
for name in VSD_SIGNIN_HMAC_KEY VSD_DB_PASSWORD VSD_DB_APP_PASSWORD; do
  sed -i.bak "s/^$name=$/$name=$(openssl rand -hex 32)/" .env && rm .env.bak
done

# 4. Install dependencies
python -m venv .venv
source .venv/bin/activate        # Windows (Git Bash): source .venv/Scripts/activate
pip install -r requirements.txt

# 5. Install CNN test dependencies (same CPU approach as CI)
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cnn.txt

# 6. Start the test database (PostgreSQL in memory, on 127.0.0.1:55432; needs the .env of step 2 above)
docker compose --profile test up -d db-test

# 7. Run the full suite; do not silently skip missing torch or database tests
REQUIRE_TORCH_TESTS=1 REQUIRE_POSTGRES_TESTS=1 VSD_TEST_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/postgres python -m pytest -q
```

Datasets are not stored in this repo. They live on the team's shared Google Drive, pinned by D21;
see [docs/setup.md](docs/setup.md) and [data/README.md](data/README.md).

## What can run from this snapshot

| Component | On merged `main` | How to inspect it |
|---|---|---|
| Research core | Preprocessing, baseline/CNN code, training, evaluation, risk policy, bundle loader and synthetic tests | [Setup](docs/setup.md), [stage briefs](docs/stages/), [recorded results](docs/results/) |
| API | Stage 17 API with config, request IDs, error handling, schemas, history, notes, stats and sign-in (`/api/session`) | Put `VSD_SIGNIN_HMAC_KEY` in `.env`, run `python -m uvicorn api.main:app --port 8001 --env-file .env`, then open `http://127.0.0.1:8001/api/health`; or `docker compose up -d` (nginx in front, http://localhost:8001); see [setup](docs/setup.md) and the [approved contract](docs/api_design.md) |
| UI | React analyst console in `web/` behind nginx, on the MockAnalyzer; the design prototype stays in `docs/design/` | [MVP: install and run](#mvp-install-and-run); [prototype guide](docs/design/README.md) |
| Production upload/inference flow | Upload to verdict works on the MockAnalyzer; no real model is connected | [MVP: install and run](#mvp-install-and-run); the [backlog](docs/midterm/backlog.md) tracks its requirements |

The API status response explicitly contains `mock: true`; it does not establish real inference.
The React `web/` app and the Docker Compose deployment are on `main`;
`dashboard/` is a legacy empty placeholder.
Unpublished work is not counted as merged.

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
and clearly identify real and mocked behavior. The MVP runs upload to a mock result; a complete
upload-to-real-result flow is not demonstrated yet. Editorial acceptance, the demo and LMS submission must be
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
    06_stage11_for_test.ipynb
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
