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

## Running the API (Stage 17)

The API refuses to start without `VSD_SIGNIN_HMAC_KEY`. Run it with `.env` loaded (`python-dotenv`,
pinned in `requirements.txt`):

```bash
python -m uvicorn api.main:app --port 8001 --env-file .env --no-access-log --log-config configs/api_logging.yaml
```

`--no-access-log` keeps client IPs out of the log (§1, §7); `configs/api_logging.yaml` shows the
`api` logger's INFO sign-in lines. `GET /api/health` needs nothing. Sign in with `POST /api/session`
and an `Origin` header from `allowed_origins` (`http://localhost:8001` in `configs/api.yaml`); see
`docs/api_design.md` §4.8.

## Running with Docker Compose (Stage 17)

`docker-compose.yml` builds the API image from `deploy/api.Dockerfile` (Python 3.13.15 slim, pinned by
digest; non-root; only `requirements-api.txt`, whose pins must equal `requirements.txt`). The `api`
service publishes **no port**; nginx (the `web` service) is the only way in, on http://localhost:8001.
Never expose the API without nginx (§4.8 rate limit). Needs Docker Compose v2 (BuildKit).

The `web` image is built from `deploy/web.Dockerfile`: `npm ci` and `npm run build` of `web/` on
`node:26.13.0-alpine`, then the `dist/` output and `deploy/nginx.conf` copied into `nginx:1.30.6-alpine`
(both pinned by digest; no Node or source in the final image). Its build context is limited by
`deploy/web.Dockerfile.dockerignore` (about 160 kB: no `node_modules`, `dist`, `.env` or data). The first
build downloads the npm packages (a few minutes). **After editing `deploy/nginx.conf` or `web/`, run
`docker compose build web && docker compose up -d web`**: the config is in the image, not mounted.

```bash
docker compose build                  # both images
docker compose up -d
docker compose ps                     # api: (healthy), from GET /api/health; web starts after it
docker compose exec web nginx -t      # config syntax check
curl -s http://localhost:8001/api/health   # {"ok":true}, through nginx
docker compose logs api               # INFO lines of the api logger; no request lines, no IPs
docker compose logs web               # time, method, path, status, bytes, duration; no IPs
docker compose exec api python -m api.admin create-account analyst01   # also unlock, disable, ...
docker compose down                   # keeps the store; down -v deletes it
```

- **Key:** Compose reads `VSD_SIGNIN_HMAC_KEY` from `.env` and passes only that variable to the
  container (not `env_file`: `.env` also holds the Kaggle, HF and W&B tokens). Empty or unset, Compose
  refuses to start. `docker compose config` prints the key: do not paste its output anywhere.
- **Store:** SQLite at `/app/var/api.sqlite` on the named volume `vsd-var` (`database_path:
  var/api.sqlite` resolves there); it survives `down`/`up`. The tables and the note-author migration
  run at every start.
- **Uploads:** `temp_dir: null`, so they go to `/tmp`, a 192 MiB tmpfs: in memory, never on disk, gone
  at restart. Each upload uses `/tmp` twice (Starlette's spooled part, then the API's temp copy), so
  192 MiB holds `x-vsd-concurrent-uploads: 4` uploads at the `configs/api.yaml` limit; the deploy tests
  fail if a larger `max_size_bytes` no longer fits. More at once fail until concurrency is limited
  (Open items).
- **Admin:** `docker compose exec` gives a TTY, so the hidden password prompt works; it uses the same
  config, key and volume as the API.
- The root filesystem is read-only, all capabilities are dropped, and FastAPI's `/docs` is reachable
  only inside the container.

### nginx (the `web` service)

- **Routes (A2):** `/api/` is proxied to `api:8000`; `/` redirects to `/analyze`; `/signin`,
  `/analyze`, `/history`, `/history/req_<32 hex>`, `/overview` and `/status` serve the app's
  `index.html` with `Cache-Control: no-cache` (a new build is picked up at the next load); `/assets/`
  serves the hashed build files with `immutable`, and a missing one is a 404, never the app. Every other
  path, including `/index.html`, `/history/foo`, `/docs`, `/redoc` and `/openapi.json`, gets a JSON
  `404 not_found`.
- **Headers (A13):** every answer, nginx's own errors and the API's included, carries `Referrer-Policy:
  strict-origin-when-cross-origin`, `X-Content-Type-Options: nosniff` and `Content-Security-Policy:
  default-src 'self'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'; object-src 'none'`.
  They live in `deploy/nginx_security_headers.conf`, included in every location (a location with its
  own `add_header` would drop inherited ones). An inline script, a `data:` image or another origin
  would need a CSP change by decision.
- **Errors:** API answers pass through unchanged (`proxy_intercept_errors` off). nginx answers itself
  only with `413 file_too_large` (body above `client_max_body_size 22m`, which sits above the API's own
  limit), `503 analysis_unavailable` (API stopped or silent for 60 s), `429 too_many_requests` and
  `404 not_found`, all in the §6 shape with `request_id: null`. An API answer carries `X-Request-ID`,
  an nginx one does not.
- **Rate limit:** `POST /api/session` 10 per minute per client IP, burst 5, `nodelay` (§4.8). `GET` and
  `DELETE /api/session` are not limited.
- **No client IP:** the access log has no address, header or query string; the error log is at `crit`
  (its lines carry a `client:` field); `X-Forwarded-For`, `X-Real-IP` and `Forwarded` are cleared towards
  the API. The IP is used only as the rate-limit key, in nginx memory.
- **Uploads:** nginx buffers each request body before passing it on, under `/var/cache/nginx`, a 96 MiB
  tmpfs (4 x 22 MiB): in memory, never on disk.
- **Restart:** nginx resolves `api` once at start. If the `api` container is recreated (not just
  stopped and started), run `docker compose restart web` too.
- **HTTPS:** plain HTTP, `localhost` only. The `Secure` session cookie works over plain HTTP only on
  `localhost`; any other host needs TLS terminated in nginx first (not part of Stage 17).

Manual check through nginx (test account and password only; `-c`/`-b` keep the session cookie):

```bash
O='Origin: http://localhost:8001'; B=http://localhost:8001; J=$(mktemp)
curl -si $B/docs                                         # 404 not_found from nginx
curl -si $B/                                             # 302, Location: /analyze
curl -si $B/history/req_00000000000000000000000000000000 # 200 text/html, no-cache, the 3 headers
curl -si $B/history/foo                                  # 404 not_found (application/json)
curl -si $B/assets/$(ls web/dist/assets | grep '^index-.*\.js$')   # 200, immutable (same build)
curl -si $B/assets/missing.js                            # 404 not_found, no Cache-Control
curl -si -c $J -H "$O" -H 'Content-Type: application/json' \
     -d '{"username":"analyst01","password":"..."}' $B/api/session   # 200, Set-Cookie vsd_session
head -c 22020096 /dev/zero > /tmp/21m.wav                # 21 MiB: the API's 413 (X-Request-ID set)
head -c 24117248 /dev/zero > /tmp/23m.wav                # 23 MiB: nginx's 413 (no X-Request-ID)
curl -si -b $J -H "$O" -F file=@/tmp/21m.wav $B/api/analyses
curl -si -b $J -H "$O" -F file=@/tmp/23m.wav $B/api/analyses
for i in $(seq 1 8); do curl -s -o /dev/null -w '%{http_code} ' -H "$O" -H 'Content-Type: application/json' \
  -d "{\"username\":\"burst$i\",\"password\":\"y\"}" $B/api/session; done   # 401s, then 429
docker compose stop api; curl -si -b $J $B/api/status    # 503 analysis_unavailable from nginx
docker compose start api
docker compose logs web api | grep -Ei 'client:|172\.|127\.0\.0\.1|192\.168\.|::1|x-forwarded'   # nothing
rm -f /tmp/21m.wav /tmp/23m.wav "$J"
```

The burst uses a different username per request so the API's lockout (5 failures per username) does not
answer `429 account_locked` first.

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
