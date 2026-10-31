# Documentation

This folder is the team's shared memory. Keep it current: future-you, teammates, reviewers and AI
tools all read it. Project rules live in `../AGENTS.md`.

| File | What goes here | Update when… |
|---|---|---|
| [project_log.md](project_log.md) | Living log: one entry per stage/session, Open items, Current status | a stage or session ends (append-only) |
| [decisions/](decisions/) | Decision register (`README.md`, D-IDs, hypotheses) + ADRs for major decisions | a decision is made |
| [architecture.md](architecture.md) | Summary of the SDD: components, data flow, diagram | the structure changes |
| [setup.md](setup.md) | Full steps to run the project locally and in Colab | the setup steps change |
| [CHANGELOG.md](CHANGELOG.md) | Human-readable history of what changed | behavior changes |
| [dataset_audit.md](dataset_audit.md) | Stage 1 dataset audit (FoR-2sec full, DEEP-VOICE metadata only) | signed off; changes need a decision |
| [audit_evidence/](audit_evidence/) | Machine-generated evidence behind every audit number | the audit is re-run |
| [threat_model.md](threat_model.md) | Stage 2 threat model (v1.0 reviewed) | threats or claims change |
| [experimental_protocol.md](experimental_protocol.md) | Stage 3 Experimental Protocol (v1.0, frozen (F1)) | only by a new decision (§12 change control) |
| [evaluation_harness.md](evaluation_harness.md) | Shared metrics module (EER, ROC-AUC, ...): conventions, methods, tests | a harness step is added |
| [clip_scores.md](clip_scores.md) | Per-clip score files (Stage 11, D41/D45): CSV + sidecar format, applicability per model role, FoR test-split access guard | the score format or the guard changes |
| [risk_policy.md](risk_policy.md) | Risk policy (Stage 10): band rule in the core, Platt calibration, ... | a risk-policy step is added |
| [baseline.md](baseline.md) | MFCC/SVM baseline (Stage 7, evaluation only): scaler, NumPy scorer, non-pickle artifact | a baseline step is added |
| [cnn.md](cnn.md) | D42 CNN (Stage 8): model, train-only normalization, seeding, safetensors checkpoint | a CNN step is added |
| [cnn_training.md](cnn_training.md) | CNN training (Stage 8): configs/cnn.yaml drift guard, D42 early stopping, W&B allow-list, D30 search and selection | the training loop, search or W&B logging changes |
| [stages/](stages/) | Stage briefs (D32): one short explanation per closed stage, from `stages/TEMPLATE.md` | a stage closes |
| [api_design.md](api_design.md) | API contract between the backend (`api/`) and the web app (`web/`), Stage 17 (approved in PR #65; implementation partial) | an endpoint, error code or stored field changes |
| [uml/](uml/) | Canonical midterm UML package: the SRS use-case model and SDD component, class and sequence figures, with editable sources and SVG exports | an approved requirement, decision or source-report figure changes |
| [refactor-optimization-report.md](refactor-optimization-report.md) / [PDF](refactor-optimization-report.pdf) | Khalil’s Part 6 measured lint cleanup, before/after evidence, merged PRs and review status | evidence or approved refactoring changes |
| [ai-usage-reflection.md](ai-usage-reflection.md) | Midterm AI ethics and usage reflection: tools, human review, corrections, privacy, bias, licensing and academic integrity | AI practices, evidence or course requirements change |
| [feature_cache.md](feature_cache.md) / [logmel_cache.md](logmel_cache.md) | Verified MFCC and log-Mel cache formats, provenance and deterministic output rules | cache implementation changes |
| [baseline_run.md](baseline_run.md) / [risk_run.md](risk_run.md) | Existing research-run orchestration and reproduction instructions | an approved runner changes |
| [speaker_analysis.md](speaker_analysis.md) / [speaker_average.md](speaker_average.md) | Per-speaker metrics and speaker-average conventions | reporting implementation changes |
| [synthetic_reporting.md](synthetic_reporting.md) | Synthetic integration and discrimination checks; not real model evaluation | synthetic checks change |
| [results/](results/) | Recorded baseline/CNN validation and risk-policy results, linked to immutable evidence | new approved results are recorded |
| [design/](design/) | HTML mock-data prototype and frontend/backend handoff; not a production React app | approved design or integration decisions change |
| [midterm/backlog.md](midterm/backlog.md) | Team midterm requirement priorities, owners, implemented and pending work | merged implementation evidence changes |
| [model_bundle.md](model_bundle.md) | ModelBundle (Stage 12): format, verifying loader, tests | the bundle format changes |
| [setup-report.md](setup-report.md) | Lab 6 setup report with recorded local results and remaining TODOs | the Lab 6 report is written |
| [reference/](reference/) | SRS, SDD, Execution Plan, Proposal (read-only sources) | never edited here |

## Current implementation and review pointers

- Research roadmap: Stage 11 is current; [D46 runtime plan PR #67](https://github.com/kahlanzzi/voice-spoofing-detection/pull/67) and [runner PR #68](https://github.com/kahlanzzi/voice-spoofing-detection/pull/68) remain under review. This index does not authorize a test-split run.
- API contract: approved; merged code currently exposes a mock status endpoint, not the complete analysis/history flow. Frontend/backend implementation is Iheb’s ongoing work; see the [backlog](midterm/backlog.md).
- Part 6 report/PDF status correction merged in [PR #74](https://github.com/kahlanzzi/voice-spoofing-detection/pull/74); this documentation refresh does not duplicate it.
- All pointers describe the checked snapshot (2026-11-01), not completion of unpublished work. Update them as implementation/review PRs merge.

## The one rule

> If a change would surprise a teammate who only reads the docs, update the docs in the **same**
> pull request.
