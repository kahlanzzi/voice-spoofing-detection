# Contributing & Team Workflow

This is how our team (and any AI agent working with us) changes this repository. Everyone follows
the same rules so we don't step on each other or break `main`. Project rules and hard constraints
are in `AGENTS.md`; approved decisions are in `docs/decisions/README.md`.

## 1. Golden rules

1. **Never commit directly to `main`.** `main` is protected: PR + 1 approval, no direct pushes.
2. **One meaningful unit = one branch = one Pull Request (PR).**
3. **At least one teammate reviews** every PR before it merges.
4. **CI must pass** (green check) before merging.
5. **Never commit secrets** (`.env`, API keys) or data (audio, datasets, checkpoints). Secrets belong
   in `.env` only (see `.env.example`).
6. **DEEP-VOICE is untouched** outside the accesses permitted by D20 (`AGENTS.md` hard rule 1).

## 2. Branch naming

`<type>/stage<N>-<name>` (D23), lower-case, hyphen-separated:

```
feat/stage<N>-<name>    a new capability     e.g. feat/stage6-preprocessing-profile
fix/stage<N>-<name>     a bug fix            e.g. fix/stage7-eer-tie-handling
docs/stage<N>-<name>    documentation only   e.g. docs/stage3-experimental-protocol
chore/stage<N>-<name>   tooling/config/repo  e.g. chore/stage6-benchmark-script
```

Cross-stage housekeeping may omit the stage: `chore/repo-structure`.

## 3. Daily flow

```bash
git checkout main
git pull                                   # get everyone's latest work
git checkout -b docs/stage3-protocol       # start a fresh branch

# ... do work, commit as you go, run pytest -q ...

git push -u origin docs/stage3-protocol    # push your branch
# then open a Pull Request against main on GitHub
```

**Stacked branches:** while a stage's PR waits for sign-off, the next stage may branch from it
instead of `main`. The waiting branch then receives no more commits except its sign-off.

## 4. Commit messages

[Conventional Commits](https://www.conventionalcommits.org): `<type>(<optional scope>): <summary>`

```
docs(protocol): define val-A / val-B usage
fix(audit): fail on empty revision
feat(preprocessing): add log-Mel profile v1
chore(ci): run pytest on Python 3.13
```

**One logical change per commit.** If your message needs the word "and", it's probably two commits.

## 5. Pull Requests

- Fill in the PR template (it appears automatically). Link the issue it closes, if any (`Closes #12`).
- Run `pytest -q` before opening the PR and paste the real output.
- Keep PRs small: easier to review, fewer merge conflicts.
- If you changed behavior, update `docs/CHANGELOG.md` **in the same PR**.
- If the PR records a decision, add the D-ID row to `docs/decisions/README.md`, and an ADR for a
  major decision (D24).
- **Stacked PRs:** if your branch starts from another open branch, switch the PR base to `main` once
  the PR below it is merged, *before* merging. Never merge a PR whose base is not `main` unless that
  is intended.
- Delete the branch after merge, once its content is confirmed in `main`.

## 6. Using AI assistants responsibly

AI tools are encouraged, but the team owns every line that is committed:

- **Human in the loop:** every AI-produced artifact (code, documents, analysis) is reviewed by a team
  member before it is accepted. AI output is never merged, cited or reported as verified without
  that review. **Read and understand** AI-generated work before committing it; you will be asked to
  explain your own PRs.
- **AI agents never merge** and never push to `main`. They push a branch; a human opens and merges
  the PR.
- Disclose AI assistance in the PR (which tool, for what) and in `docs/project_log.md`.
- Keep `AGENTS.md` up to date: it is the single AI context for every tool (D22).
- Never paste secrets (API tokens, Kaggle/HF keys, passwords) or private data (personal recordings,
  anything not already in the repo) into an AI chat or prompt.
- AI agents follow `AGENTS.md`, including hard rule 1: DEEP-VOICE is never opened, played, plotted or
  scored outside the permitted accesses (D20).

## 7. Keeping docs alive (lightweight)

- Behavior changed? → add a line to `docs/CHANGELOG.md`.
- Made a real decision? → add a D-ID row to `docs/decisions/README.md` (+ an ADR if major).
- Stage or session ended? → append an entry to `docs/project_log.md`, update Open items and Current
  status.
- Stage closed? → its PR (or the next PR) includes the stage brief in `docs/stages/` (D32); the stage
  is done only when the brief is merged.
- Moved files around? → run `./scripts/update-structure.sh`.

A git hook (enabled by `./scripts/setup.sh`) reminds you if you change code without touching docs.
It only warns; it never blocks.

## 8. Issues

Features: one issue per SRS requirement (FR / NFR / PC ID), tagged by part (A–E) and owner (D26),
grouped under one milestone per stage. Use the issue templates.

## 9. Resolving disagreements

Follow the team charter's conflict rule. Default: discuss, then if still split, the person who owns
that area decides, and we revisit later. Conflicts between project documents are never resolved
silently: see the priority order in `AGENTS.md`.
