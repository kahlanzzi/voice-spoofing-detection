# Refactor and Optimization Report

**Owner:** Khalil Dridi

**Audit date:** 2026-10-31

**Status:** Initial audit ready for team review; full CNN-enabled baseline passed. No code refactor performed; any implementation and final human review remain pending.

**Scope:** Existing repository implementation only. No new features, model changes, frozen-evidence changes, or protected-dataset access.

## Assignment and submission context

The midterm progress submission (due Monday, 2 November 2026 at 8:00 AM) asks the team to provide a runnable MVP, a recorded demonstration, repository access, and all assignments completed so far. One team member submits the repository URL, demo link, and all team members' names; assignment files must be attached or committed to the repository.

This report is Part 6, Khalil Dridi's lead contribution. The team division assigns Iheb to provide API/UI metrics, Donia to provide ML metrics, and Malak to edit the final document. Part 3 (the UML package) was allocated jointly to Khalil and Malak; Khalil reports that Malak completed that work. This clarifies Khalil's remaining assigned deliverable as Part 6; it does not replace the team's other submission requirements.

## Baseline

The latest GitHub main tip observed for this audit is 98fd8d7e17f81ff1f9687f6ea3a83804a50fa628 (merge of PR #71). Entry 107 in the project log records the following check results:

```text
Command: python -m pytest -q
Output: 794 passed, 9 skipped in 21.99s

Recorded check: tests/governance/test_deepvoice_access.py (the exact invocation was not preserved in Entry 107)
Output: 3 passed in 0.59s

Command: git diff --check
Output: passed

Token scan (exact invocation not preserved in Entry 107)
Output: no token found
```

The results above are historical repository records. The fresh baseline below was run by Khalil on `docs/stage11-midterm-refactor-khalil` at `98fd8d7e17f81ff1f9687f6ea3a83804a50fa628`, matching GitHub main at audit time. Khalil fetched and created the branch in PowerShell. This report is prepared for that documentation branch; commit, push and PR review are pending.

Khalil subsequently ran fresh checks in his own PowerShell after installing `requirements.txt`. The pasted output and screenshot show:

```text
Command: .\.venv\Scripts\python.exe -m pip check
Output: No broken requirements found.

Command: .\.venv\Scripts\python.exe -m pytest -q -ra
Output: 794 passed, 9 skipped in 51.00s
Skipped: 8 because torch is not installed; 1 because wandb is not installed.

Command: .\.venv\Scripts\python.exe -m pytest -q tests/governance/test_deepvoice_access.py
Output: 3 passed in 0.22s
```

The first full run above lacked the CNN dependencies. Khalil then installed `torch==2.11.0` from the official CPU index (installed version `2.11.0+cpu`), followed by `requirements-cnn.txt` (`safetensors==0.8.0`, `wandb==0.31.0`). `pip check` again returned `No broken requirements found.` The complete baseline then passed:

```powershell
$env:REQUIRE_TORCH_TESTS = "1"
.\.venv\Scripts\python.exe -m pytest -q -ra
```

```text
937 passed in 89.85s (0:01:29)
```

The final summary shows no failures, errors or skips. These are user-executed results supplied by pasted output and screenshots, not assistant-executed runs. Environment: local Windows, Python 3.13.16 (project venv), CPU PyTorch. The earlier eight torch skips were module-level skip records, not eight individual test functions; installing the missing dependencies enabled additional collected tests. The initial missing-FastAPI collection error and the 794-pass run are superseded by the complete 937-pass baseline. Installing declared dependencies is environment repair, not a source-code refactor or speed improvement. The difference between the two suite durations cannot support a speed comparison because the executed tests differ.

| Measurement | Before result |
|---|---|
| Full test suite | Fresh complete user run: 937 passed in 89.85s (0:01:29), with REQUIRE_TORCH_TESTS=1 and no skips shown |
| Governance test | Baseline: 3 passed in 0.22s; final documentation check: 3 passed in 0.18s (user-run) |
| Coverage | Not measured before; no coverage dependency or project coverage configuration found |
| Lint/format | Not measured before; no Ruff, Black, Flake8, or equivalent project tool/config found. git diff --check was recorded as passed. |
| Complexity | Not measured before; no Radon or equivalent project tool/config found |
| API latency/resource use | Not measured before; status endpoint is mock, and the API design leaves the latency target open |

No DEEP-VOICE or FoR test audio was accessed for this audit.

Final documentation check supplied by Khalil after the report and documentation records were added:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/governance/test_deepvoice_access.py
```

```text
3 passed in 0.18s
```

The earlier 0.22 s governance result is retained above as a separate baseline run. The final check covers the documentation working tree. `git diff --check` passed; all relative links in the report resolve in the project checkout, no literal-token patterns were found in the four changed documents, and existing project-log entries 1–107 were confirmed unchanged.

## Existing API/UI and ML measurements collected from GitHub

The following values are historical repository evidence inspected at the pinned main SHA. They were not rerun for this audit and do not establish an improvement caused by Khalil's proposed refactor. Model-quality figures are validation results only. No additional teammate measurement is needed to reproduce these documented values in this report.

| Area | Recorded evidence | Conditions and provenance | Source / commit or PR | Limitation |
|---|---|---|---|---|
| API correctness | 50 API tests passed on Python 3.13.2. Iheb checked `/api/status` = HTTP 200 and `/api/nope` = HTTP 404 with the approved error body. | API skeleton; status response uses fixed mock values. Recorded launch command: `uvicorn api.main:app --port 8001`. The exact API-test invocation is not given in the PR. | [PR #66](https://github.com/kahlanzzi/voice-spoofing-detection/pull/66), merged as `d9e8c4a` | Response time and API memory usage: not measured before. These checks do not demonstrate working inference or a complete MVP. |
| UI verification | 49/49 automated browser checks passed, no console errors; axe-core 4.10 reported 0 violations on 12 screens. | Mock-data prototype v4, Playwright/Chromium, Africa/Tunis, 2026-10-29; widths 1440, 1024, 768, 640 and 390 px. Exact test script/invocation is not included in the handoff. | [HANDOFF §13](design/HANDOFF.md), [PR #64](https://github.com/kahlanzzi/voice-spoofing-detection/pull/64) | Design prototype evidence; manual keyboard, real screen-reader and new-user interpretation reviews remain unverified. No production React performance measurement. |
| Existing UI correction, credited to Iheb's work | Dark-mode hovered-primary-button contrast changed from 4.49:1 to 7.0:1. | Same UI element and state, as recorded in the v4 handoff/PR. | [HANDOFF §10](design/HANDOFF.md), PR #64 | Previously completed UI correction; not Khalil's refactor and not an execution-speed improvement. |
| Preprocessing throughput | 256 train clips processed in 2.8235499429997617 s; 90.66600739069045 clips/s. | Stage 6, Python 3.13.15, Linux, `preprocessing-v1`, clean code commit `6e1718547756ef04744ce0b3b1b73bb839920052`. | [benchmark.json](audit_evidence/stage6_preprocessing/benchmark.json), [PR #34](https://github.com/kahlanzzi/voice-spoofing-detection/pull/34) | Historical train-subset benchmark. Exact invocation is not recorded in this JSON; not rerun. No comparable after measurement exists. |
| Stage 6 compute dry run | Batch 256: mean step 0.05665362309996454 s, peak allocated GPU memory 901,030,912 B; total VRAM 15,637,086,208 B. | Tesla T4, torch 2.11.0+cu130; `dry_run_no_optimizer`, zero optimizer steps, no checkpoint. | Same Stage 6 benchmark and commit | Benchmark network lacks the BatchNorm later added to the CNN (project log Entry 79). Do not compare its memory against training as a refactor gain. |
| Actual CNN training resources | Selected four-block, lr 0.001, seed-0 run: peak allocated GPU memory 1,894,575,616 B; median epoch about 5.0 s; 15 epochs totaling about 75 s. | Tesla T4, Python 3.13.16, torch 2.11.0+cu130, CUDA 13.0, batch 256; code `9be2f2545520432db08a0d083a06a668cd84103a`. Timing is rounded in the results document; exact per-epoch values are in the selection report. | [CNN results §6](results/cnn_selection.md), [run_report.json](audit_evidence/stage8_cnn/run_report.json), [selection_report.json](audit_evidence/stage8_cnn/selection_report.json) | Training resources, not API inference latency. No same-conditions refactor comparison exists. Original run used notebook 04; this audit did not execute it. |
| Baseline validation quality | Selected C=10, gamma=0.1: val-A EER 0.0014925373134327957 (about 0.1493%); val-B EER 0.0 on 1,329 clips. | 12,404 train clips, 1,330 val-A clips; config selected on val-A; D29 threshold fitted on val-B. Code `297c352c91bfc9daafa2943ba04154194aa33248`. | [Baseline results](results/baseline.md), [selection report](audit_evidence/stage7_baseline/selection_report.json), [threshold report](audit_evidence/stage7_baseline/threshold_report.json) | Existing validation results; no FoR test or external generalization claim. Not a refactor outcome. |
| CNN validation quality | All three selected-configuration seeds: best val-A EER 0.0 and ROC-AUC 1.0; bundle seed 0. | Configuration selected on val-A; Stage 8 code `9be2f25`. | [CNN results §4](results/cnn_selection.md), Stage 8 selection report | Selection-set results can be optimistic; do not claim superiority or a refactor benefit. |
| Risk-policy validation | Achieved deferral 0.09932279909706546 = 132/1,329 val-B clips; threshold 0.6333105553100875, half-width 0.3636377690579665. | `risk-v1`, bundle seed 0, CPU scoring, code `11c07fedcd755dc72f16b2695f180cf094aeca04`; policy fitted on val-B. | [Risk results](results/risk_policy_fit.md), [run_report.json](audit_evidence/stage10_risk/run_report.json) | Existing fitted values are recorded for context and remain untouched. No new fit, score, or protected-data access occurred. |

The source JSON values were cross-checked against their corresponding results documents. A recorded command absent from those sources is explicitly marked unavailable; no invocation or measurement is reconstructed as if it had been executed in this audit.

Source-arithmetic checks performed in this audit (no model or benchmark execution):

```text
benchmark.preprocessing.rows / benchmark.preprocessing.seconds
= 256 / 2.8235499429997617
= 90.66600739069045 clips/s (equals the recorded value)

sum(selection_report.seeds["0"].history[*].epoch_seconds)
= 75.26422497099998 s across 15 epochs

median(selection_report.seeds["0"].history[*].epoch_seconds)
= 5.019009443000186 s
```

These calculations explain the existing results document's rounded 75 s training time and 5.0 s median epoch time. They are not new timing measurements.

## Initial code audit and proposed refactoring list

The read-only audit reviewed the feature-cache pair, preprocessing validation, Stage 17 API skeleton, per-clip score storage, and the Stage 10 risk runner at main SHA `98fd8d7e17f81ff1f9687f6ea3a83804a50fa628`. It identified the feature-cache modules' deliberate private-helper reuse as a maintainability candidate, not a demonstrated defect. Structural change in the other behavior-sensitive areas is not justified by measured maintenance or performance evidence.

| Area/file | Original problem / observation | Evidence before | Refactoring performed / proposed | Evidence after | Test result | Commit/PR | Remaining limitation |
|---|---|---|---|---|---|---|---|
| src/features/mfcc_cache.py + src/features/logmel_cache.py | The log-Mel module imports private constants and validators from the MFCC-specific module, coupling two feature caches by implementation detail. | `logmel_cache.py` imports `_check_split`, `_check_hex`, `_split_rows`, `_meta_text`, and schema constants from `mfcc_cache.py`. The matching test files define 20 MFCC test functions, 22 log-Mel test functions, and one code-commit-independent NPZ-byte test function. | Proposed for team review: move only the genuinely shared cache constants/helpers into a neutral internal module, then have both caches import it. Preserve validation order, exception text, JSON/NPZ bytes, and feature-specific schema. No speedup is claimed. | N/A; no code changed. | If approved: rerun the three cache test files, full suite, and governance test; verify byte-for-byte cache outputs remain identical. The complete baseline passed; checks after the proposed refactor are pending. | None; report branch exists, no refactor commit or PR. | The cache formats and evidence are governed; helper extraction can still change observable errors or bytes and needs review before implementation. |
| src/preprocessing/validation.py | The `run()` orchestrator coordinates manifest processing, shortcut and padding checks, invariance checks, and report assembly in one function. | `run()` is 79 lines (312–390) in a 410-line module. It already calls focused helpers; no measured complexity or maintenance failure was found. | No refactor recommended now. Keep the audit steps visible together unless reviewers identify a specific maintenance problem and agree on exact-output regression checks. | N/A; no code changed. | Existing `tests/preprocessing/test_validation.py`; no code change; included in the complete baseline. | None. | It validates the frozen preprocessing profile; changing its order or report can alter governed evidence. |
| api/config.py | Three input parameters lack annotations: `path` in `load_api_config` and both parameters of `_type_ok`. | `load_api_config(path=None) -> ApiConfig` and `_type_ok(value, expected) -> bool`; 88-line module. No static type checker/configuration is present in the current-main tree. | Lower-priority candidate: add accurate input annotations only; keep behavior and error text unchanged. No split is recommended because the loader's validation steps are cohesive. | N/A; no code changed. | Existing `tests/api/test_api_config.py` is the behavior guard; no code change; included in the complete baseline. | None; report branch exists, no refactor commit or PR. | Small maintainability gain only; no performance gain or type-check result can be claimed without a configured checker. |
| api/errors.py | Error mapping is spread across four handlers, but each covers a distinct FastAPI/Starlette condition and already uses shared error_body. | 75-line file; shared response-shape construction already exists. | No refactor recommended in this pass; combining handlers could obscure distinct header, status, and logging behavior. | Not applicable. | Existing API error and middleware tests remain the evidence if later changed. | None. | Retain current separation unless a concrete duplication issue is identified. |
| src/evaluation/clip_scores.py | Serialization, validation, integrity checks, and file creation sit in one module. | 391-line governed implementation; writer/reader already share validation and canonical-byte helpers. Exact hashes, bytes, refusal rules, and exclusive file creation are tested. | No refactor recommended in this pass. Do not split this code without a specific measured maintenance problem and byte-for-byte regression tests. | Not applicable. | tests/evaluation/test_clip_scores.py covers canonical output and refusal cases. | None. | High risk because output bytes and stored evidence are frozen. |
| src/tuning/risk_run.py | The run orchestrator performs provenance checks, cache loading, scoring, fitting, diagnostics, and output writing. | 298-line Stage 10 runner; execution order and committed evidence are governed. Repeated score and policy-fit calls are intentional determinism checks, not removable duplicate work. | No refactor recommended in this pass. Any later extraction must preserve exact call order and output bytes. | Not applicable. | tests/tuning/test_risk_run.py and committed evidence would need to remain unchanged. | None. | No timing baseline exists; do not claim faster execution or rewrite evidence. |

## Initial recommendation and team review gate

This audit does not establish a measurable performance problem. The clearest maintainability candidate is the private-helper coupling between the two feature-cache modules; it is a structural cleanup, not a performance optimization. The lower-priority `api/config.py` typing gap is optional. No code has been changed. The cache refactor should wait for team review because it touches governed schemas and deterministic evidence. Storage or run-orchestration changes are out of scope without explicit team agreement.

## Evidence and review status

- Iheb's existing API/UI checks and UI correction are documented in PRs #64 and #66 and have been included above.
- Donia's existing ML quality, timing and resource evidence is committed under `docs/results/` and `docs/audit_evidence/` and has been included above.
- API response-time/resource measurements and a before/after refactor comparison are not recorded in those sources. They remain explicitly unmeasured; the existing evidence does not justify estimating them.
- No team approval or implementation PR for the newly proposed cache-helper extraction has been verified in this audit. The initial audit must reach the team before this evidence-sensitive change is implemented, as requested in the task brief.
- Khalil's human review and Malak's final editorial review remain to be recorded.

## Closing record

- Improvements achieved by this task: repository evidence consolidated, resource and validation figures traced to their original sources, and refactor candidates prioritized. No code-performance improvement is claimed. The historical UI contrast correction belongs to Iheb's PR #64.
- Intentionally unchanged: model behavior, preprocessing rules, thresholds, API contracts, frozen evidence, governance decisions, datasets, audio, and test splits.
- Remaining technical debt: coverage, lint, complexity, and reproducible online latency baselines are not established; no refactor has yet been accepted or implemented.
- Measurement limitations: user-executed baseline results are recorded above, with no skips in the final full-suite run. The assistant could not launch the project interpreter directly. No runtime improvement or before/after refactor comparison has been measured. This is an initial audit for team review; commit/PR, any agreed implementation, after-checks and final human review remain pending.
- AI tools used: Codex for repository audit and report drafting.
- Human review: Khalil must verify the audit, review the proposed change, review the captured baseline output and run checks after any approved change, and understand this AI-assisted report before accepting or submitting it. Command execution by Khalil is recorded above; his acceptance of the report content and proposed change remains pending.
