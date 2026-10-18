# Lab 6 — Setup Report (in progress)

> In progress. Recorded local results and AI-tool answers are included below; remaining TODOs and team review are still open.

## 1. Team and project
The project builds a decision-support tool that identifies recorded speech as authentic, synthetic or uncertain, with confidence and risk information for a human analyst. Its research question compares performance on FoR-2sec with one frozen external evaluation on DEEP-VOICE. It is not proof of fraud or a biometric authenticator. See [README](../README.md).

Ownership follows D26 in the [decision register](decisions/README.md):

| Member | Responsibility |
|---|---|
| Donia Mabrouk | Part A foundation and protocol lead; Part C modeling and freeze, Stages 7–12 |
| Malak Ben Salem | Part B data and preprocessing, Stages 4–6; Stage 14 robustness |
| Khalil Dridi | Stage 13 external evaluation; Stage 15 reliability analysis; Stage 16 error analysis |
| Iheb Zemzemi | Part E API, dashboard and integration, Stages 17–19 |

All four review the protocol and external evaluation, and contribute to the final report.

## 2. Repository setup
Repository: [kahlanzzi/voice-spoofing-detection](https://github.com/kahlanzzi/voice-spoofing-detection).

[doc] `AGENTS.md` and project-log Entry 18 specify protected `main`, one approving review, required CI check `checks`, and an up-to-date branch. The live PR #21 page showed one approving review required on 2026-10-18. Current force-push and bypass settings have not been independently inspected; owner confirmation remains pending.

[doc] Project-log Entries 15–17 describe adaptation of the CS 488 structure: shared `AGENTS.md` context, tool-specific pointers, contribution and PR/issue templates, decision records, repository hygiene, setup scripts, CI and the dataset-access governance test. Entry 18 records the structure PR #6 as merged. See [project log](project_log.md).

## 3. Development environment
[doc] D15 specifies Python 3.13 for official runs and CI; Python 3.14 is allowed for local editing and quick tests. Local development uses `.venv` with versions pinned in `requirements.txt`; research runs use the team's Colab environment, with Kaggle as fallback.

[doc] `scripts/setup.sh` configures `.githooks`, makes shell scripts executable where supported, and prints environment reminders. On Windows it is run in Git Bash. Running it on Khalil's checkout has not yet been verified. Credentials are supplied locally when needed and are never included in the report. See [setup guide](setup.md).

### Khalil Dridi — local setup verification

- Operating system: Windows.
- Python version: 3.13.16, verified with `py -3.13 --version`.
- Git version: 2.56.0.windows.2.
- Created the project environment with `py -3.13 -m venv .venv`.
- Installed dependencies with `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.
- Ran the tests with `.\.venv\Scripts\python.exe -m pytest -q`.
- Observed result: `266 passed in 299.59s (0:04:59)`.
- This verifies Khalil's local setup; it does not establish completion of the other team members' setup checks.

## 4. Continuous integration
[verified from configuration] [CI workflow](../.github/workflows/ci.yml) runs on pull requests and pushes to `main`. The `checks` job uses Ubuntu and Python 3.13, installs pinned dependencies, runs `pytest -q`, then runs the documentation reminder. The workflow does not download datasets.

[verified from GitHub UI, 2026-10-18] [PR #21 checks](https://github.com/kahlanzzi/voice-spoofing-detection/pull/21/checks) showed one successful check for the two-commit draft ending at `faecd8c`. Later commits require their own successful CI result.

Khalil's pre-push terminal evidence on 2026-10-17 showed `266 passed in 10.79s` for the full suite and `3 passed in 0.16s` for the governance test. These local results are separate from GitHub CI.

## 5. AI tooling
[doc] D22 defines `AGENTS.md` as the common rulebook: Codex and Antigravity use it as context, Claude Code imports it through `CLAUDE.md`, and Gemini CLI is configured through `.gemini/settings.json`. These are the configured mechanisms, not proof that every member tested automatic loading.

Each member must supply their own tool-test evidence. Khalil's, Malak's and Donia's records appear below; Iheb's answer is pending collection for this report. Human review is required before AI-produced work is accepted. Assistance is disclosed in the PR and project log; AI agents do not merge.

### Khalil Dridi

- **Tool:** Codex desktop.
- **Date:** 2026-10-17.
- **Method:** Existing conversation with supplied project documents; Codex explicitly read the local `AGENTS.md` and decision register. This was not a fresh-session automatic-context-loading test.
- **Question:** What are this project's hard rules about DEEP-VOICE, and where would preprocessing code go?
- **Answer summary:** DEEP-VOICE is reserved for the single guarded external scoring run after the complete ModelBundle is frozen using FoR only (D2). It must not guide training, tuning, model selection, thresholds or calibration; before that run only D20-permitted accesses are allowed. Robustness uses FoR only (D3), later external analyses use stored scores, and preprocessing code belongs in `src/preprocessing/`.
- **Review status:** Answer recorded for teammate review. Fresh-session automatic-context loading was not tested.

### Donia Mabrouk

- **Source:** Donia's test report, relayed by Khalil on 2026-10-18; the session was not independently observed by Codex.
- **Tool:** Claude Code, VS Code extension; version not supplied (the submitted message contains `[version]`).
- **Date:** 2026-10-18.
- **Method (reported):** Fresh session in the repository on `main`, with no files attached and no context given. Claude Code loads `CLAUDE.md`, which imports `AGENTS.md` (D22).
- **Question:** “What are this project's hard rules about DEEP-VOICE, and where would preprocessing code go?”
- **Answer summary:** DEEP-VOICE is external test data used exactly once, in the single guarded run after the ModelBundle freeze (D2); never for training, tuning, thresholds, calibration or exploration. Before that run only D20 accesses are allowed (download, zip hash, unzip, listing and metadata audit), enforced by the CI governance test. Preprocessing goes in `src/preprocessing/`, tests in `tests/preprocessing/`, and configuration in `configs/`, shared by notebooks and the API (FR-04, NFR-09).
- **Result (reported by Donia):** Passed. The tool also inspected the repository layout and labelled its claims `[doc]` / `[verified]`.

### Malak Ben Salem

- **Tool:** Codex
- **Date:** 2026-10-15
- **Result:** Passed
- **Pass basis:** The answer identified D20, the one-time external evaluation of DEEP-VOICE,
  and `src/preprocessing/`.

**Exact test question:**

```text
What are this project’s hard rules about DEEP-VOICE, and what exact repository directory contains preprocessing code? Include the applicable decision IDs and exact directory path.
```

**Exact answer:**

```markdown
[doc] DEEP-VOICE rules:

- **D2:** Use DEEP-VOICE only after the complete ModelBundle—CNN, preprocessing profile, thresholds, uncertain band, and calibration—is frozen using FoR data.
- **D3:** Do not run robustness testing on DEEP-VOICE; use the FoR test split only.
- **D6 / D20:** Before the single external evaluation, access is limited to download, ZIP hashing, unzip, and listing paths, extensions, sizes, and byte hashes, plus `src/audit/deepvoice_metadata_audit.py`. No listening, spectrograms, feature statistics, or other audio-content reads.
- **D21:** Use only the pinned DEEP-VOICE archive SHA-256 in `configs/audit.yaml`; changing it requires a new decision.
- **D17 / D19:** During the one external run, apply the frozen speech-window filter and report the required per-speaker results.
- Never use DEEP-VOICE for training, tuning, model selection, early stopping, threshold selection, calibration, or exploratory analysis. It is touched for model scoring exactly once.

The preprocessing code directory is:

`C:\Users\malak\Downloads\voice-spoofing-detection\src\preprocessing\`
```

## 6. Workflow
[doc] The [contribution guide](../CONTRIBUTING.md) requires one meaningful change per branch and PR, branch names `<type>/stage<N>-<name>` (D23), and Conventional Commits. Contributors update from `main`, work on a branch, run the checks and open a PR against `main`; a teammate reviews before merge.

The PR template asks for scope, requirements, actual test evidence, documentation, dataset restrictions and AI disclosure. Issue templates capture requirements or bug details. For stacked work, the next branch may start from a waiting branch; its PR base must be changed to `main` after the dependency merges. The shared `pr-prep` instructions require tests, governance and token checks, documentation updates, and a review diff saved outside the repository.

## 7. Data and safety
[doc] D21 pins the FoR revision and external archive hash in `configs/audit.yaml`; changing a pin requires a decision. D20 limits access before the single frozen external run to explicitly permitted download, hash, file-listing and metadata-audit operations. No training, tuning, calibration or exploratory scoring on DEEP-VOICE is allowed. The access guard is `tests/governance/test_deepvoice_access.py`; extending its allowlist requires a decision.

Audio, datasets, checkpoints and secrets are excluded from commits. `.env.example` supplies placeholders; actual credentials belong in ignored local configuration. Only small approved manifests are tracked under `data/`. The repository's code license does not authorize dataset redistribution (D25).

## 8. Problems met and fixes
- **Git command unavailable:** PowerShell initially did not recognize `git`. Installed Git for Windows and verified it with `git --version`, which returned `git version 2.56.0.windows.2`.
- **Python command unavailable:** The `python` command displayed the Windows “Python not found” message. After installation, `py -3.13 --version` returned `Python 3.13.16`, although the `python` command still did not work. Used the Python launcher to create `.venv`, then called `.\.venv\Scripts\python.exe` directly to install dependencies and run tests.
- **Verification:** The local test run completed with `266 passed in 299.59s (0:04:59)`.

## 9. Next steps
- [x] Record Donia's supplied AI-tool test summary (2026-10-18; §5).
- [ ] Confirm Donia's Claude Code extension version.
- [ ] Collect Iheb's actual AI-tool name and answer summary, or a link to existing evidence.
- [ ] Obtain Iheb's audit review/sign-off and explicit D14 yes/no, including any concern. Do not infer approval from an existing decision entry.
- [ ] Confirm live branch-protection details, including force-push/bypass policy, with the repository owner.
- [ ] Verify Khalil's one-time setup script/hooks and team resource access (Drive, compute and W&B).
- [ ] Complete the fresh-session AI-context check, or record the team's acceptance of Khalil's explicitly documented existing-session method.
- [ ] Obtain successful CI for the final PR revision and teammate review before closing Lab 6.

Khalil's later technical work is separate: prepare the file-level policy and external-run allowlist decisions and develop analysis tools on synthetic scores. The real external evaluation remains gated on the approved protocol and frozen ModelBundle.
