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
| [api_design.md](api_design.md) | API contract between the backend (`api/`) and the web app (`web/`), Stage 17 (proposed) | an endpoint, error code or stored field changes |
| [uml/](uml/) | Canonical midterm UML package: the SRS use-case model and SDD component, class and sequence figures, with editable sources and SVG exports | an approved requirement, decision or source-report figure changes |
| [refactor-optimization-report.md](refactor-optimization-report.md) | Midterm initial refactor audit, baseline checks, existing resource evidence and proposed work | evidence or approved refactoring changes |
| [ai-usage-reflection.md](ai-usage-reflection.md) | Midterm AI ethics and usage reflection: tools, human review, corrections, privacy, bias, licensing and academic integrity | AI practices, evidence or course requirements change |
| [model_bundle.md](model_bundle.md) | ModelBundle (Stage 12): format, verifying loader, tests | the bundle format changes |
| [setup-report.md](setup-report.md) | Lab 6 setup report (skeleton) | the Lab 6 report is written |
| [reference/](reference/) | SRS, SDD, Execution Plan, Proposal (read-only sources) | never edited here |

## The one rule

> If a change would surprise a teammate who only reads the docs, update the docs in the **same**
> pull request.
