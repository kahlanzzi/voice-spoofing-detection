# Documentation

Use this index to find setup instructions, project rules, component guides, evidence and
course deliverables. Sections below group the existing documents; file locations are unchanged.
Project rules live in [AGENTS.md](../AGENTS.md).

## Quick navigation

- [Start here](#start-here)
- [Requirements, decisions and project history](#requirements-decisions-and-project-history)
- [Implementation guides](#implementation-guides)
- [API, interface design and UML](#api-interface-design-and-uml)
- [Results and audit evidence](#results-and-audit-evidence)
- [Midterm and course deliverables](#midterm-and-course-deliverables)
- [Current implementation and review pointers](#current-implementation-and-review-pointers)

## Start here

| File | What goes here | Update when… |
|---|---|---|
| [setup.md](setup.md) | Full steps to run the project locally and in Colab | the setup steps change |
| [architecture.md](architecture.md) | Summary of the SDD: components, data flow, diagram | the structure changes |
| [stages/](stages/) | Stage briefs (D32): one short explanation per closed stage, from `stages/TEMPLATE.md` | a stage closes |

## Requirements, decisions and project history

| File | What goes here | Update when… |
|---|---|---|
| [reference/](reference/) | Version index plus current and historical SRS, SDD, Execution Plan and Proposal | a versioned document pair is added |
| [decisions/](decisions/) | Decision register (`README.md`, D-IDs, hypotheses) + ADRs for major decisions | a decision is made |
| [experimental_protocol.md](experimental_protocol.md) | Stage 3 Experimental Protocol (v1.0, frozen (F1)) | only by a new decision (§12 change control) |
| [threat_model.md](threat_model.md) | Stage 2 threat model (v1.0 reviewed) | threats or claims change |
| [project_log.md](project_log.md) | Living log: one entry per stage/session, Open items, Current status | a stage or session ends (append-only) |
| [CHANGELOG.md](CHANGELOG.md) | Human-readable history of what changed | behavior changes |

## Implementation guides

| File | What goes here | Update when… |
|---|---|---|
| [feature_cache.md](feature_cache.md) / [logmel_cache.md](logmel_cache.md) | Verified MFCC and log-Mel cache formats, provenance and deterministic output rules | cache implementation changes |
| [baseline.md](baseline.md) | MFCC/SVM baseline (Stage 7, evaluation only): scaler, NumPy scorer, non-pickle artifact | a baseline step is added |
| [baseline_run.md](baseline_run.md) / [risk_run.md](risk_run.md) | Existing research-run orchestration and reproduction instructions | an approved runner changes |
| [cnn.md](cnn.md) | D42 CNN (Stage 8): model, train-only normalization, seeding, safetensors checkpoint | a CNN step is added |
| [cnn_training.md](cnn_training.md) | CNN training (Stage 8): configs/cnn.yaml drift guard, D42 early stopping, W&B allow-list, D30 search and selection | the training loop, search or W&B logging changes |
| [risk_policy.md](risk_policy.md) | Risk policy (Stage 10): band rule in the core, Platt calibration, ... | a risk-policy step is added |
| [evaluation_harness.md](evaluation_harness.md) | Shared metrics module (EER, ROC-AUC, ...): conventions, methods, tests | a harness step is added |
| [clip_scores.md](clip_scores.md) | Per-clip score files (Stage 11, D41/D45): CSV + sidecar format, applicability per model role, FoR test-split access guard | the score format or the guard changes |
| [stage11_run.md](stage11_run.md) | Stage 11 runner (D46): the one FoR test run in two phases (claim: checks, registry claim, evidence; score: features and scores twice, per-clip files) and its val-A/val-B dry run | the Stage 11 run procedure changes |
| [stage11_report.md](stage11_report.md) | Stage 11 metrics report (D57): integrity checks on the stored run files, D4 metrics per model and subgroup, clustered CIs, H1, H5 and seed rows, JSON + Markdown | the Stage 11 reporting conventions change |
| [speaker_analysis.md](speaker_analysis.md) / [speaker_average.md](speaker_average.md) | Per-speaker metrics and speaker-average conventions | reporting implementation changes |
| [synthetic_reporting.md](synthetic_reporting.md) | Synthetic integration and discrimination checks; not real model evaluation | synthetic checks change |
| [model_bundle.md](model_bundle.md) | ModelBundle (Stage 12): format, verifying loader, tests | the bundle format changes |
| [v2_candidates.md](v2_candidates.md) | Model v2 candidates (Track V, D49, D61): cnn-aug, crnn, credited AASIST port, v2 checkpoint, configs/v2_candidates.yaml | a v2 candidate, the v2 checkpoint or its config changes |
| [v2_training.md](v2_training.md) | Model v2 training (Track V, D49, D50, D61): raw train cache, set S format, item-seed rule, training loop and early stopping on M, resumable search, batch-size benchmark | the v2 training data path, S, the search or the benchmark changes |

## API, interface design and UML

| File | What goes here | Update when… |
|---|---|---|
| [api_design.md](api_design.md) | API contract between the backend (`api/`) and the web app (`web/`), Stage 17 (approved in PR #65; implementation partial) | an endpoint, error code or stored field changes |
| [design/](design/) | HTML mock-data prototype and frontend/backend handoff; not a production React app | approved design or integration decisions change |
| [uml/](uml/) | Canonical midterm UML package: the SRS use-case model and SDD component, class and sequence figures, with editable sources and SVG exports | an approved requirement, decision or source-report figure changes |

## Results and audit evidence

| File | What goes here | Update when… |
|---|---|---|
| [dataset_audit.md](dataset_audit.md) | Stage 1 dataset audit (FoR-2sec full, DEEP-VOICE metadata only) | signed off; changes need a decision |
| [results/](results/) | Recorded baseline/CNN validation and risk-policy results, linked to immutable evidence | new approved results are recorded |
| [audit_evidence/](audit_evidence/) | Machine-generated evidence behind every audit number | the audit is re-run |

## Midterm and course deliverables

| File | What goes here | Update when… |
|---|---|---|
| [midterm/backlog.md](midterm/backlog.md) | Team midterm requirement priorities, owners, implemented and pending work | merged implementation evidence changes |
| [refactor-optimization-report.md](refactor-optimization-report.md) / [PDF](refactor-optimization-report.pdf) | Khalil’s Part 6 measured lint cleanup, before/after evidence, merged PRs and review status | evidence or approved refactoring changes |
| [ai-usage-reflection.md](ai-usage-reflection.md) | Midterm AI ethics and usage reflection: tools, human review, corrections, privacy, bias, licensing and academic integrity | AI practices, evidence or course requirements change |
| [setup-report.md](setup-report.md) | Lab 6 setup report with recorded local results and remaining TODOs | the Lab 6 report is written |

## Current implementation and review pointers

- **Snapshot (2026-11-15):** Stage 12 is closed with F4 verified and handed over (Entry 174). Track V
  is current: V1b is closed after [PR #134](https://github.com/kahlanzzi/voice-spoofing-detection/pull/134),
  and V1c is next. See [Execution Plan v3.0](reference/Execution_Plan_v3.0.md) and [current project
  status](project_log.md).
- **Stage 13:** Khalil’s guarded-run preparation merged in
  [PR #133](https://github.com/kahlanzzi/voice-spoofing-detection/pull/133). It is preparation only;
  no DEEP-VOICE audio has been opened or scored and no external-run claim exists. The run still
  depends on D48’s frozen model-set rule, an exact-runtime D58 rehearsal, verified durable Drive claim
  behavior, final checks, peer review and the Stage 13 brief. This index does not authorize the run.
- **Application:** the approved API contract and the sign-in, analysis, History, notes and stats
  routes are implemented. The React MVP supports sign-in, upload, a mock verdict and History; the
  analyzer still uses mock results pending Iheb’s F4 integration (P1). Some later interface work
  remains; see the [backlog](midterm/backlog.md) and [current project status](project_log.md).
- Part 6 report/PDF status correction merged in [PR #74](https://github.com/kahlanzzi/voice-spoofing-detection/pull/74); this documentation refresh does not duplicate it.
- These pointers describe the checked snapshot (2026-11-15). Update them as implementation and review PRs merge.

## The one rule

> If a change would surprise a teammate who only reads the docs, update the docs in the **same**
> pull request.
