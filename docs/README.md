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
| [risk_policy.md](risk_policy.md) | Risk policy (Stage 10): band rule in the core, Platt calibration, ... | a risk-policy step is added |
| [stages/](stages/) | Stage briefs (D32): one short explanation per closed stage, from `stages/TEMPLATE.md` | a stage closes |
| [model_bundle.md](model_bundle.md) | ModelBundle (Stage 12): format, verifying loader, tests | the bundle format changes |
| [setup-report.md](setup-report.md) | Lab 6 setup report (skeleton) | the Lab 6 report is written |
| [reference/](reference/) | SRS, SDD, Execution Plan, Proposal (read-only sources) | never edited here |

## The one rule

> If a change would surprise a teammate who only reads the docs, update the docs in the **same**
> pull request.
