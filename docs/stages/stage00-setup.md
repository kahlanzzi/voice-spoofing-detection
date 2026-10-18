# Stage 0 — Repository and environment setup

| | |
|---|---|
| **Stage** | 0 — Repository and environment setup |
| **Owner** | Donia Mabrouk (D26, Part A) |
| **Reviewer(s)** | Iheb Zemzemi (repo owner since 2026-10-14, `docs/project_log.md` header and Entry 20) |
| **Status** | Accepted (PR #25) |
| **Date** | 2026-10-18 |
| **Related PRs** | Stage 0: #1 (`stage-0-git-workflow`, Entry 14) · Part A repository-structure work (Execution Plan v2.1 §7, "Repository structure" row): #6 (`chore/repo-structure`, Entry 18) |

## 1. In one paragraph

Stage 0 builds the place where all later work happens: a GitHub repository with protected `main`,
a pinned Python environment, CI that runs the tests on every pull request, and one written rulebook
(`AGENTS.md`) that every person and every AI tool follows. It also fixes the habits that make
results trustworthy later: every run is "dataset version + config + code version → results",
datasets stay out of Git, and the external test set (DEEP-VOICE) is guarded by an automatic test,
not only by a promise. Without this stage, two people could train on different data, a stray
notebook could open the external test set, or an AI tool could merge unreviewed code. Nothing in
this stage touches audio or trains a model.

## 2. Inputs → outputs

| | What | From / to | Path |
|---|---|---|---|
| In | Readiness audit: 5 pre-training blockers, decisions D1–D14 | Team, before 2026-10-04 (Entry 1) | `docs/decisions/README.md` |
| In | Bootstrap package and reference documents (SRS, SDD, Execution Plan, Proposal) | Team (Entry 2) | `docs/reference/` |
| Out | Rulebook for people and AI tools | Every stage, every member | `AGENTS.md`, `CLAUDE.md`, `CONTRIBUTING.md` |
| Out | Pinned environment, CI | Every stage | `requirements.txt`, `.github/workflows/ci.yml` |
| Out | DEEP-VOICE governance test | Every PR; extended only by decision before Stage 13 (Khalil) | `tests/governance/test_deepvoice_access.py` |
| Out | Setup guide and docs index | New members, the professor | `docs/setup.md`, `docs/README.md` |
| Out | Decision register and ADRs | Every stage | `docs/decisions/` |

No D26 hand-over interface belongs to this stage; those are fixed in Stage 3 (D26).

## 3. How it works, step by step

1. **Rules before code.** The readiness audit compared the Proposal, SRS, SDD and Execution Plan
   and approved D1–D14 before any code existed (Entry 1).
2. **Repo scaffold.** Built with Claude Code from a bootstrap package; two Windows bugs in the audit
   code fixed (UTF-8 writing, `/` in manifest paths); first commit of 31 files (Entry 2).
3. **Python version fixed** (D15): 3.13 official, 3.14 allowed locally (Entry 2, ADR 0002).
4. **Git workflow.** One branch per unit, PRs only, AI tools never merge (Entry 2); stacked-branch
   and stacked-PR rules added after the PR #3 base mistake (Entries 9 and 14). Branch names follow
   D23 (`CONTRIBUTING.md` §2).
5. **One AI rulebook** (D22): `AGENTS.md` holds all rules; `CLAUDE.md` imports it; Gemini CLI reads
   it through `.gemini/settings.json` (Entry 15, ADR 0003).
6. **Hygiene, CI and the guard** (PR #6, Entries 17–18): `.gitattributes` (LF), `.env.example`
   (placeholders), pinned `requirements.txt`, CI on Python 3.13, and the governance test that fails
   if a DEEP-VOICE path appears outside the permitted files.
7. **Protection on `main`**: PR + 1 approval + required CI check `checks` + branch up to date; no
   direct pushes (Entry 18).
8. Later, before Stage 3 (PR #7): Khalil joined and the work split D26 was recorded; see the Stage 3 brief.

Details: [`docs/setup.md`](../setup.md), [`CONTRIBUTING.md`](../../CONTRIBUTING.md).

## 4. Key decisions and why

- **D15** — Python 3.13 official, 3.14 for local editing: results come from Colab, which runs 3.13.15.
- **D21** — dataset pins (FoR HF revision, DEEP-VOICE zip SHA-256) in `configs/audit.yaml`: same data
  in every run ([ADR 0004](../decisions/0004-dataset-pinning.md)). Decided in Stage 1–2, enforced by the Stage 0 habits.
- **D20** — the only DEEP-VOICE accesses before the external run: keeps the one external test honest
  ([ADR 0005](../decisions/0005-deep-voice-single-use.md)).
- **D22** — `AGENTS.md` is the single AI context: one set of rules, no drift between tools
  ([ADR 0003](../decisions/0003-agents-md-single-ai-context.md)).
- **D23** — branch naming `<type>/stage<N>-<name>`: readable history, one unit per branch.
- **D24** — register for every decision, ADR only for major ones: decisions stay findable without paperwork.
- **D25** — MIT license for code only; datasets never redistributed: respects the dataset terms.
- **D26** — work split by parts A–E and "consumer reviews producer": ownership and review are clear.

## 5. Evidence

- 5 tests passed on Windows after the two audit-code fixes [verified] — source: Entry 2.
- 15 tests passed after the governance test was added; `.env` git-ignored; `data/manifests/*.csv`
  and `docs/audit_evidence/*` not ignored [verified] — source: Entry 17.
- `pip download --only-binary=:all:` resolves the full pinned set for Python 3.13 and 3.14; local
  install verified on 3.14.7 only, 3.13 install checked by CI [verified] — source: Entry 17.
- Branch protection on `main` confirmed (PR + 1 approval + `checks` + up to date) [verified] —
  source: Entry 18.
- PR #1 approved and merged by Malak on 2026-10-05 [verified] — source: Entry 14 table.
- On this branch today: `python -m pytest -q` → `266 passed`; audit + governance tests →
  `15 passed` [verified] — source: Entry 38 (Python 3.14.7 local venv).

## 6. Limitations

- The local machine has no Python 3.13; 3.13 is checked only by CI and Colab (Entry 17). A
  3.14-only problem could pass locally and still fail in CI, or the reverse [verified absence].
- The governance test only matches four text patterns in tracked and non-ignored files: the
  DEEP-VOICE root folder name (any case), the Kaggle slug / zip name (any case), the Kaggle owner
  (any case) and the audit subfolder name as a whole word (upper case only). It skips binary files
  (`.pdf`, `.docx`, `.png`, `.jpg`, `.wav`, `.npy`, `.pt`, `.zip`, `.tar`) and allows everything under
  `src/audit/`, `docs/` and `tests/`, plus exactly `configs/audit.yaml`,
  `notebooks/01_stage1_dataset_audit.ipynb` and `data/README.md`. Code that builds the path some other
  way is not caught [verified: `PATTERNS`, `ALLOWED`, `BINARY_SUFFIXES` in the test file].
- "AI agents never merge" is a written rule; the technical block is branch protection (1 human
  approval), not a check on who presses merge [doc: `AGENTS.md`, Entry 18].
- W&B logging is required by `AGENTS.md` but not yet set up (Open items: "Set up W&B (0.6) before
  Stage 8").
- Iheb's confirmation of collaborators, protection and Actions settings after the transfer is still
  open (Open items, Entry 20).

## 7. Questions the professor may ask

1. **Q:** Why is Python 3.13 official when you develop on 3.14?
   **A:** All reported results come from Colab, whose runtime is 3.13.15, and CI tests on 3.13. 3.14 is
   only for local editing and quick tests; `requirements.txt` must install on both (D15, ADR 0002).
2. **Q:** How do you guarantee every run uses exactly the same data?
   **A:** The datasets are pinned (FoR HF revision, DEEP-VOICE zip SHA-256) in `configs/audit.yaml`,
   and the notebook stops on an empty or mismatched pin (D21, ADR 0004). From Stage 5 the manifests
   and hash ledger are frozen as F2 (D28).
3. **Q:** Why can't AI tools merge, and what protects `main`?
   **A:** Every AI output must be reviewed by a team member before it is accepted (human in the loop,
   `AGENTS.md` Working mode). `main` requires a PR, 1 approval, the green CI check `checks` and an
   up-to-date branch; no direct pushes (Entry 18).
4. **Q:** Why are datasets not in Git, and where are they?
   **A:** Hard rule 7 bans audio, datasets and checkpoints in Git, and D25 forbids redistributing the
   datasets. They live in the shared Drive under `MyDrive/voice-spoofing-detection/raw/...` and are
   copied to `/content/data/` in Colab (`AGENTS.md`, Compute and data).
5. **Q:** How is "DEEP-VOICE touched once" enforced technically, not just by a rule?
   **A:** `tests/governance/test_deepvoice_access.py` fails CI if a DEEP-VOICE path appears outside
   the permitted files (D20); the zip hash is checked before unzipping (D21); and while the metadata
   audit runs, `decoding_forbidden()` in `src/audit/deepvoice_metadata_audit.py` replaces the
   soundfile / librosa / torchaudio read and load calls with a function that raises.
6. **Q:** Why one AI rulebook (`AGENTS.md`) for every tool?
   **A:** The project is worked on with several AI tools (D22 names Codex, Antigravity, Claude Code and
   Gemini CLI). One file means one set of rules and no copies that drift apart; rules are edited only
   there (D22, ADR 0003).

## 8. How to verify or re-run it

From [`docs/setup.md`](../setup.md), local steps (no dataset needed; tests use synthetic data):

```bash
git clone https://github.com/kahlanzzi/voice-spoofing-detection.git
cd voice-spoofing-detection
./scripts/setup.sh                       # Windows: run in Git Bash; enables the pre-push docs reminder
cp .env.example .env                     # placeholders only; never commit .env
python -m venv .venv
source .venv/bin/activate                # Git Bash: source .venv/Scripts/activate · PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q                                          # full suite
python -m pytest -q tests/governance/test_deepvoice_access.py  # DEEP-VOICE guard alone
```

Needs: Git, Python 3.13 (or 3.14 for local work), no data and no tokens for the tests. Expected:
all tests pass; at this branch's commit the full suite printed `266 passed` (Entry 38). Setup
prints that hooks are enabled (`core.hooksPath = .githooks`, Entry 17). CI runs the same `pytest -q`
on Python 3.13 for every PR.
