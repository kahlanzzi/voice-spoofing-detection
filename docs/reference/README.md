# Reference documents

Controlling documents of the project. **Current** versions govern the work; older versions are kept
unchanged as history (project-log entries and decisions cite them by path). Priority when documents
conflict (`AGENTS.md`): decision register > SRS > SDD > Execution Plan > Proposal.

Convention: each new version is a pair `<Document>_v<major.minor>.md` + `.pdf` generated from one
source; old files are never edited, moved or renamed. Update this index in the PR that adds a version.

| Document | Current | History | Notes |
|---|---|---|---|
| SRS | `SRS_v4.1.md` + `.pdf` | `SRS_v4.0.md` + `.pdf`; `SRS_Voice_Spoofing_Detection.pdf` (v3.0) | v4.1 carries v4.0 forward and incorporates approved D63 probability/policy/window requirements and D65 Direction D. |
| SDD | `SDD_v2.0.md` + `.pdf` | `SDD_Voice_Spoofing_Detection_Lab5.pdf` (v1.0) | v2.0 aligns the implemented React/FastAPI MVP with the approved target design for live analysis, process-pool inference, PostgreSQL, environments and extension security under D38, D41 and D47-D57. |
| Execution Plan | `Execution_Plan_v3.0.md` + `.pdf` | `Execution_Plan_v2.1.pdf` (D26); `Execution_Plan_v2.2.md` + `.pdf` (draft, never adopted); `Execution_Plan.md` (v1) | v3.0 adopted in PR #113. |
| Proposal | `Voice_Spoofing_Detection_Project_Proposal.docx` | - | Motivation and framing; lowest priority. |
