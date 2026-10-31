# UML consistency and source-fidelity audit

**Audit date:** 2026-10-31

**Canonical sources:** SRS v3.0 Figure 1 and SDD v1.0 Figures 1-6

**Result:** PASS - 7 of 7 repository figures preserve the canonical model.

This audit proves more than the presence of seven diagram files. It compares the model elements and
relationships in each editable repository source with the corresponding approved report figure, then
checks that concepts shared by different UML views do not contradict one another.

## Scope and interpretation

Different UML diagram types are not expected to repeat the same boxes. A use-case diagram describes
actor goals, a component diagram describes software boundaries, a class diagram describes domain
structure, and a sequence diagram describes time-ordered collaboration. Consistency therefore means:

1. each repository figure preserves the elements and relationships of its canonical report figure;
2. a concept reused in another view keeps the same responsibility and dependency direction;
3. no repository figure silently adds a second actor, use case, component, class, domain relationship
   or message path; and
4. any approved post-report label is disclosed and does not alter the underlying relationship.

Legends, titles, layout helpers and explanatory notes are presentation metadata, not model elements.

## Figure-by-figure proof

| Repository figure | Canonical inventory | Repository inventory | Relationship result |
|---|---|---|---|
| Merged use-case model | 5 actors; UC-01-UC-10; 9 actor associations; 5 `include`; 1 `extend` | Same counts, names, directions and endpoints | Exact |
| Online component view | 1 external API client; 13 components; 4 stores; 19 dependencies | Same boundaries, provided-interface directions and store dependencies | Exact; D38 label only |
| Research component view | 1 engineer; 10 components; 3 external systems; 16 dependencies | Same nodes and dependencies; the report's representative group dependencies remain group dependencies | Exact |
| Analysis-domain classes | 10 classes + `Detector` interface; 10 relationships | Same classes/interface, attributes, operations, relationship kinds and multiplicities | Exact |
| Evaluation/governance classes | 7 classes including the Figure 3 `ModelBundle` reference; 8 relationships | Same classes, relationship kinds and multiplicities | Exact; D41 note only |
| UC-01 sequence | 9 lifelines; 21 numbered messages; one two-operand `alt` fragment | Same lifelines, ordering, calls, returns and alternatives | Exact flow; PR #65 labels only |
| UC-08 sequence | 9 lifelines; 22 numbered messages; one two-operand `alt` fragment | Same lifelines, ordering, calls, returns and alternatives | Exact |

### Use-case relationship ledger

Actor associations are unchanged:

- API Client -> UC-02.
- Fraud Prevention Analyst -> UC-01, UC-05 and UC-06.
- ML / Research Engineer -> UC-07, UC-08 and UC-09.
- Compliance Officer -> UC-09.
- Operations / Security Administrator -> UC-10.

Use-case relations are unchanged:

- UC-01 -> UC-03 `include`.
- UC-02 -> UC-03 `include`.
- UC-03 -> UC-04 `include`.
- UC-05 -> UC-04 `include`.
- UC-06 -> UC-04 `include`.
- UC-06 -> UC-05 `extend` (extension -> base).

### Online component dependency ledger

- Analyst dashboard/web app and API Client -> API Gateway.
- API Gateway -> Access Control, Evaluation Service, Analysis Service, Analysis Record Service and
  Operational Log.
- Analysis Service -> Audio Intake, Preprocessing, Detector Runtime, Risk & Confidence and Analysis
  Record services.
- Audio Intake Service and Preprocessing Service -> Temporary Audio Store.
- Detector Runtime -> Model Registry -> Model Artifact Store.
- Analysis Record Service -> Repository Layer -> Analysis Data Store.
- Operational Log -> Service Log Store.

No research-only External Test Guard or Experiment Tracker is added to the online tier.

### Research component dependency ledger

- ML / Research Engineer -> Research Notebooks.
- Research Notebooks -> Training & Tuning Pipeline, Evaluation Service and Dataset Manager (audit).
- Training & Tuning Pipeline -> shared core and research infrastructure adapter groups.
- Evaluation Service -> shared core, research infrastructure adapters and External Test Guard.
- External Test Guard -> Model Registry (frozen-bundle check) and Experiment Tracker (run-registry
  check).
- Dataset Manager -> Kaggle / Hugging Face and Google Drive.
- Model Registry -> Google Drive.
- Experiment Tracker -> Google Drive and Weights & Biases.

The shared core contains exactly Preprocessing Service, Detector Runtime and Risk & Confidence
Service, matching the corresponding online components.

### Analysis class relationship ledger

- `AudioSubmission` depends on `InputPolicy` and yields 0..1 `AnalysisRecord`.
- `AnalysisRecord` composes exactly 1 `AnalysisResult` and 0..* `Annotation` objects.
- `RiskPolicy` creates `AnalysisResult`.
- `ModelBundle` composes exactly 1 `PreprocessingProfile`, 1 `RiskPolicy` and 1 `Detector`.
- `CnnDetector` and `MfccBaselineDetector` realize `Detector`.

All attributes and operations shown in SDD Figure 3 are retained. The PC-08 legend restates the
report rule that only the CNN is deployable; it is not an additional relationship.

### Evaluation class relationship ledger

- `ExternalTestGuard` checks `ModelBundle`, consults exactly 1 `RunRegistry`, and authorizes
  `EvaluationRun`.
- One `ModelBundle` evaluates 0..* `EvaluationRun` objects.
- One `RunRegistry` aggregates/registers 0..* `EvaluationRun` objects.
- 0..* `EvaluationRun` objects use exactly 1 `DatasetSplit`.
- One `EvaluationRun` composes exactly 1 `MetricSet`.
- 1..* `MetricSet` objects are presented by exactly 1 `EvaluationReport`.

The D41 note refines the implementation registry key; the SDD relationship between the guard,
registry and run is unchanged.

### Sequence-order proof

UC-01 retains the report's order: select file -> submit -> assign request ID -> analyze -> accept ->
validate -> preprocess -> score -> decide -> record metadata -> release temporary audio -> return and
display the result. Its alternative path still returns a structured validation error without scoring.

UC-08 retains the report's order: request external test -> guard authorization -> frozen-bundle and
prior-run checks -> load frozen bundle -> load external split -> preprocess -> score -> apply frozen
decision rule -> compute metrics -> register the external run -> return results. Its alternative path
still rejects the request before loading or scoring the external split.

## Cross-view traceability proof

| SRS goal | Component realization | Class realization | Sequence evidence | Consistency |
|---|---|---|---|---|
| UC-01 / UC-02 submit audio | API Gateway -> Analysis Service and shared core | `AudioSubmission`, `InputPolicy`, `AnalysisRecord`, `AnalysisResult` | UC-01 messages 1-21 | PASS |
| UC-03 validate/analyze | Audio Intake -> Preprocessing -> Detector -> Risk | `PreprocessingProfile`, `Detector`, `RiskPolicy` | UC-01 messages 5-15 | PASS |
| UC-04 record metadata | Analysis Record Service -> Repository Layer | `AnalysisRecord` composes `AnalysisResult` | UC-01 messages 13-14 | PASS |
| UC-05 / UC-06 history and annotation | Analysis Record Service -> Repository Layer | `AnalysisRecord` composes 0..* `Annotation` | Use-case include/extend relations | PASS |
| UC-07 model evaluation | Evaluation Service and shared core | `EvaluationRun` -> `MetricSet` -> `EvaluationReport` | Same evaluation responsibilities reused by UC-08 | PASS |
| UC-08 external test | Evaluation -> Guard -> Registry/Tracker before Dataset | `ExternalTestGuard`, `RunRegistry`, `EvaluationRun` | UC-08 messages 1-22 | PASS |
| UC-09 evaluation report | Evaluation Service | `EvaluationReport` aggregates 1..* metric sets | External result returned only after metrics and registration | PASS |
| UC-10 service logs | Operational Log -> Service Log Store | Operational concern; no invented domain class | Online component dependency | PASS |

## Approved difference ledger

Exactly three post-report updates are visible, and none creates or reverses a model relationship:

1. **D38 / ADR 0008:** `Analyst Dashboard` is labelled `Analyst Web App - React, D38`. The actor,
   API-Gateway dependency and UC-01 responsibility are unchanged.
2. **PR #65:** UC-01 message 2 is labelled `POST /api/analyses` instead of `/analyze-audio`, and
   message 17 is labelled `201 AnalysisResponse (exact 8 fields)` instead of `200 JSON`. The
   participants, direction, position in the sequence and success/error branches are unchanged.
3. **D41 / ADR 0010:** a note records the later registry-key refinement. The class set and the
   `ExternalTestGuard` -> `RunRegistry` -> `EvaluationRun` relationships are unchanged.

## Final verdict

- Canonical figures checked: **7 / 7**.
- Added domain elements: **0**.
- Removed domain elements: **0**.
- Added, removed or reversed relationships: **0**.
- Multiplicity changes: **0**.
- Message-order changes: **0**.
- Approved, disclosed label/note updates: **3**.
- Undisclosed differences: **0**.

Therefore the repository package is one consistent UML model expressed through complementary views,
not seven competing versions of the system.
