# Midterm prioritized backlog

**Snapshot:** 2026-11-07

**Owners:** Donia Mabrouk, Iheb Zemzemi, Malak Ben Salem, Khalil Dridi

**Source of truth:** approved decisions in `docs/decisions/README.md` override the SRS; the SRS
v3.0 overrides the SDD, Execution Plan and Proposal (`AGENTS.md`).

This backlog reorganizes the existing SRS requirements into an implementation and submission view.
It does not create new requirements or change approved scope. Priorities are copied from SRS v3.0.
Its closed requirement list is exactly `FR-01`-`FR-20`, `NFR-01`-`NFR-11` and `PC-01`-`PC-09`.
No row may be added without an approved change to the repository requirements.

## Status legend

- **Done:** the acceptance criterion is supported by merged evidence.
- **Done (mock MVP):** the criterion is supported by the merged, clearly labelled mock application;
  any separate real-model integration requirement remains Partial.
- **Partial:** a reviewed design or reusable core exists, but the end-to-end acceptance criterion is
  not yet met.
- **In progress:** the responsible stage is actively being implemented or run.
- **Planned:** implementation has not started on `main`.
- **Blocked by gate:** work must wait for an approved protocol dependency; this is intentional, not
  an implementation failure.

## Functional requirements

| ID | Backlog item / user value | Acceptance criterion (short) | Priority | Owner / stage | Status | Evidence | Remaining work |
|---|---|---|---|---|---|---|---|
| FR-01 | As a Fraud Prevention Analyst, I can submit one supported short audio file through the web app. | A policy-compliant upload starts an analysis. | Must | Iheb / 18 | Done (mock MVP) | Upload/analyze UI and e2e, PRs #106-#108 | Replace the clearly labelled `MockAnalyzer` with the frozen ModelBundle after Stage 12; tracked separately by FR-04-FR-07. |
| FR-02 | As an API Client, I can submit one supported short audio file through the REST API. | A documented multipart request returns a request-specific structured response. | Must | Iheb / 17 | Done (mock MVP) | Approved `POST /api/analyses` contract and implementation, PRs #65 and #77 | Production inference remains gated by Stage 12; the endpoint and analyzer seam are implemented. |
| FR-03 | As a submitter, I receive a structured reason when a file is unreadable, unsupported, too large or too long. | Invalid files are rejected before inference. | Must | Iheb / 17 | Done | Intake validation and negative tests, PR #77; web pre-checks, PR #106 | Retain the negative tests when the production analyzer is connected. |
| FR-04 | As the system, I use the versioned inference preprocessing profile for every accepted file. | A known input produces the expected model-ready representation under the selected profile. | Must | Malak core / Iheb integration, 6 and 17 | Partial | F3 and Stage 6 evidence, PR #34 | Load and call F3 from the production analyzer. |
| FR-05 | As an analyst, I receive exactly one of authentic, synthetic or uncertain. | Every successful analysis returns one valid label. | Must | Donia core / Iheb integration, 8-10 and 17 | Partial | Selected CNN and risk policy, PRs #52, #60 | Integrate the frozen bundle after Stage 12 and test the API response. |
| FR-06 | As an analyst or API client, I receive the calibrated model output in the valid numeric range. | Every successful result has a value in `[0,1]`. | Must | Donia core / Iheb integration, 10 and 17 | Partial | `risk-v1`, PR #60; API schema PR #66 | Bind the production analyzer output; keep the number under Technical details in the UI. |
| FR-07 | As an analyst, I receive LOW, MEDIUM or HIGH risk from the versioned policy. | Exactly one valid risk level is returned and the policy version is recorded. | Should | Donia core / Iheb integration, 10 and 17 | Partial | `risk-v1`, PR #60 | Integrate the frozen policy and add API integration tests. |
| FR-08 | As an analyst, I see the verdict, risk, supporting model output and decision-support notice together. | The result screen contains all required information without presenting it as proof of fraud. | Must | Iheb / 18 | Partial | Result card and decision-support copy, PR #107 | The mock-MVP view is complete; verify the same view against the real ModelBundle response after Stage 12. |
| FR-09 | As the system, I store successful analysis metadata without storing raw audio. | N analyses create N retrievable metadata records with all required fields and no audio. | Must | Iheb / 17 | Done | SQLite metadata, retrieval and temp-file deletion tests, PRs #77 and #79 | Preserve metadata-only storage when the production analyzer is connected. |
| FR-10 | As an analyst, I can review stored analysis history and identify synthetic/uncertain results. | History is newest-first, supports an empty state and marks relevant results. | Should | Iheb / 17-18 | Done (MVP cut) | History API PR #79; newest-10 React History with empty/error states, PR #108 | Filters, pagination and row links are later enhancements, not part of the approved MVP cut. |
| FR-11 | As an authenticated analyst, I can annotate a result without changing the original result. | The note is saved separately; unauthorized writes are refused. | Could | Iheb / 17-18 | Partial | Authenticated note API and separate note table, PRs #80 and #85 | Add the analyst-note UI; the API and authorization are merged. |
| FR-12 | As an ML engineer, I evaluate the MFCC/SVM baseline and CNN through the same harness and FoR test split. | Both models use the same metrics module and test split. | Must | Donia / 11 | In progress | Shared harness and baseline/CNN artifacts; D46 PR #67 and runner PR #68 are open | Review/merge the pre-run PRs, then complete the single guarded Stage 11 run. |
| FR-13 | As a reviewer, I receive accuracy, precision, recall, F1, EER, ROC-AUC and the three-way confusion matrix. | Every evaluation output contains the complete required set. | Must | Donia then Khalil / 11 and 13 | Partial | `src/evaluation/` and synthetic tests | Produce the official Stage 11 and Stage 13 outputs. |
| FR-14 | As a reviewer, I can identify dataset, split and class distribution beside the metrics. | Every report states dataset/split and class counts. | Must | Donia then Khalil / 11 and 13 | Partial | Reporting helpers and result-document conventions | Complete official reports after the guarded runs. |
| FR-15 | As an ML engineer, I evaluate the frozen final model on DEEP-VOICE exactly once. | One distinct external section references F4; no second external run exists. | Must | Khalil / 13 | Blocked by gate | D20, D31, ADR 0007; run guard foundations | Finish Stage 11, freeze F4 at Stage 12, then perform the single Stage 13 run. |
| FR-16 | As a reviewer, I see the external result beside the FoR result with the gap and dataset caveats. | The report quantifies the gap and states RVC-only scope and imbalance. | Must | Khalil / 13 | Blocked by gate | Pre-registered reporting rules, ADR 0007 | Generate after FR-15 without using the external result for tuning. |
| FR-17 | As a reviewer, I see how performance changes under noise, compression and bandwidth reduction. | A clean-to-severe degradation curve is produced on FoR test data. | Could | Malak / 14 | Planned | D41 and ADR 0010 approved in PR #46; storage support PR #62 | Implement readers/config and run the whole pre-declared grid once after Stage 11. |
| FR-18 | As a reviewer, I can assess confidence calibration using reliability and ECE evidence. | Reliability diagram and ECE are reported before/after calibration. | Could | Donia core / Khalil analysis, 10 and 15 | Partial | Calibration fitted in Stage 10, PR #60 | Complete the Stage 15 test-set calibration analysis. |
| FR-19 | As a reviewer, I can request an evaluation report even before the external run. | The external section says "not yet available" before Stage 13 and shows results afterward. | Should | Khalil / 13 and 19 | Planned | Report structure documented | Implement report assembly and the placeholder/result states. |
| FR-20 | As an operations administrator, I can review version, volume, latency and error-rate logs without audio or PII. | A monitoring view exposes all four items and no raw audio. | Should | Iheb / 17-19 | Partial | Privacy-safe logging PRs #86-#87; stats API PR #81; System status PR #101 | Complete the operational view for all four required measures and perform the final privacy inspection. |

## Non-functional requirements

| ID | Quality backlog item | Acceptance criterion (short) | Priority | Owner / stage | Status | Evidence | Remaining work |
|---|---|---|---|---|---|---|---|
| NFR-01 | At least 95% of valid requests complete within the approved `L_MAX`. | Reference workload meets the latency target. | Should | Iheb + all / 19 | Planned | Stage 6 compute benchmark is not the online latency baseline | Decide `L_MAX`, define workload and run the end-to-end benchmark. |
| NFR-02 | Responses never expose weights, training paths, secrets or stack traces. | Security inspection finds none in success or error responses. | Must | Iheb / 17-19 | Partial | Structured errors PR #66; security/Origin/HTTP e2e PR #97; secret scans in CI | Repeat the final inspection with the real analyzer and release configuration. |
| NFR-03 | Uploaded raw audio is removed within the approved retention period. | A shortened-value retention test confirms deletion while metadata remains. | Must | Iheb / 17 | Done | Temp audio is deleted in `finally` on success and every error while metadata persists, PR #77 | Preserve immediate deletion when the production analyzer replaces the mock. |
| NFR-04 | Normal analysis and operational logs contain no audio or direct identifiers. | Log inspection finds neither category. | Must | Iheb / 17-19 | Partial | Metadata-only storage PR #77; privacy-safe logging/deployment PRs #86-#87 | Perform and record the final log inspection with the release configuration. |
| NFR-05 | The UI consistently frames results as decision support and never treats uncertain as fraud. | Content review finds no definitive-fraud wording. | Must | Iheb / 18 | Done (mock MVP) | Approved copy PR #64; implemented result/uncertain states PR #107 | Recheck the wording when real-model results are connected. |
| NFR-06 | Invalid input and inference failures return structured errors without successful records. | Fault and negative-input tests show no fabricated result. | Must | Iheb / 17 | Done (mock MVP) | Error middleware PR #66; intake/storage fault tests PR #77; e2e failure states PR #107 | Repeat the end-to-end fault tests with the production analyzer. |
| NFR-07 | Every successful analysis is traceable by request ID, versions, timestamp and latency. | Stored records contain all four traceability fields. | Must | Iheb / 17 | Done (mock MVP) | Analysis record, schemas and retrieval tests, PRs #77 and #79 | Bind the real ModelBundle versions after Stage 12. |
| NFR-08 | Web and API enforce the same formats and input limits. | Cross-interface tests produce identical accept/reject outcomes. | Should | Iheb / 17-18 | Done | `/api/limits` PR #78; limits store, UI checks and contract tests PR #106 | Retain conformance tests if limits change. |
| NFR-09 | Deployment binds matching model, preprocessing and risk-policy versions. | Loader/config inspection confirms and logs the three versions together. | Should | Donia then Iheb / 12 and 17 | In progress | Verifying ModelBundle loader exists; F3 and `risk-v1` exist | Complete Stage 11, freeze F4, then load it in the API. |
| NFR-10 | API and web app start as containers by following only the README. | A fresh-machine Docker run starts both and a test submission succeeds. | Must | Iheb / 19 | Partial / in verification | Docker/nginx PRs #86-#90; README PR #109; Linux fresh-clone run; Malak's Windows build, health check and account creation | Finish and record the Windows sign-in, upload, result and History flow; the current Windows check is not yet a complete test submission. |
| NFR-11 | Preprocessing, risk/confidence and API end-to-end tests run in CI. | Final CI executes all three groups and passes. | Should | Iheb + all / 19 | Partial | Large core suite and required CI; API skeleton tests merged in PR #66 | Add the real analyzer/API end-to-end tests and keep the required CI green. |

## Project and research constraints

| ID | Constraint backlog item | Priority | Owner / stage | Status | Evidence | Remaining work |
|---|---|---|---|---|---|---|
| PC-01 | Complete and document the FoR/DEEP-VOICE audit before training. | Must | Donia / 1 | Done | `docs/dataset_audit.md`, Stage 1 brief | No midterm implementation work; retain the recorded license limitation. |
| PC-02 | Use a group-aware FoR split where metadata permits; document limitations. | Must | Malak / 4 | Done | Stage 4, PR #12; `docs/stages/stage04-split.md` | Preserve F2 split hashes in all later runs. |
| PC-03 | Use FoR only for model, hyperparameter and threshold selection. | Must | Donia / 7-10 | Done | Frozen protocol and Stage 7-10 evidence | Do not use Stage 11 or DEEP-VOICE results for tuning. |
| PC-04 | Freeze the selected model and decision policy before DEEP-VOICE access. | Must | Donia / 12 | In progress | CNN selected; F3 and `risk-v1` fitted; verifying bundle loader tested | Complete the single Stage 11 run, then freeze F4. |
| PC-05 | Evaluate DEEP-VOICE exactly once and never retune from its results. | Must | Khalil / 13 | Blocked by gate | D20/D31 governance and run registry | Wait for F4, use the guarded runner once, then lock the record. |
| PC-06 | Report the full required metric set and exclude t-DCF. | Must | Donia + Khalil / 11 and 13 | Partial | Shared evaluation harness implements the required set | Produce official Stage 11/13 outputs and review them. |
| PC-07 | Retain dataset/config/code versions with every run. | Must | Every stage owner | Done | Manifests, configs, reports, hashes and project log | Continue the same provenance standard for Stages 11-19. |
| PC-08 | Keep the CNN/log-Mel model as the single required deep model. | Should | Donia / 8-9 | Done | D42 and Stage 8-9 evidence | Do not add unapproved architectures before the MVP. |
| PC-09 | Limit stretch goals to robustness and calibration. | Should | Malak + Khalil / 14-15 | Partial | D41 defines robustness; Stage 10 fitted calibration | Complete Stages 14-15 without adding further stretch goals. |

## Delivery order and dependencies

1. **Current gate:** review D46 PR #67 and runner PR #68, then complete the single FoR internal-test
   run under their approved procedure.
2. **Freeze:** assemble and verify F4 at Stage 12.
3. **Merged mock MVP:** Stage 17 API and the Stage 18 React upload/result/History flow are complete
   with the clearly labelled `MockAnalyzer`; production inference still waits for F4.
4. **External evaluation:** perform the single guarded Stage 13 run only after F4.
5. **Post-test analyses:** run Stages 14-16 according to D41/D43 and the frozen protocol.
6. **Release evidence:** satisfy NFR-10 and NFR-11 and complete the Stage 19 tests/docs/delivery work
   assigned by D26.

## Midterm review notes

- The project has strong merged research-core evidence through Stage 10 and a green required CI
  workflow.
- GitHub verification on 2026-11-07 found D46 PR #67 and the Stage 11 runner PR #68 open. Open PRs
  count as **In progress**, never Done.
- Merged `main` now contains the authenticated Stage 17 API and the Stage 18 React mock MVP:
  sign-in, upload, validation, mock result, History, System status, Docker/nginx deployment and CI.
- Mock-MVP completion does not make the real inference requirements complete. FR-04-FR-07,
  NFR-09 and NFR-11 remain gated by the Stage 11 run and Stage 12 F4 freeze.
- Malak's Windows verification on 2026-11-07 reached a healthy Docker stack, `{"ok":true}` and
  creation of `analyst01`; the browser sign-in/upload/result/History sequence is not yet recorded, so
  NFR-10 remains Partial / in verification.
- Statuses in this file must be updated from merged evidence only; open branches and verbal reports do
  not count as Done.
