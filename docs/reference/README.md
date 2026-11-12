# Reference documents

Controlling documents of the project. **Current** versions govern the work; older versions are kept
unchanged as history (project-log entries and decisions cite them by path). Priority when documents
conflict (`AGENTS.md`): decision register > SRS > SDD > Execution Plan > Proposal.

Convention: each new version is a pair `<Document>_v<major.minor>.md` + `.pdf` generated from one
source; old files are never edited, moved or renamed. Update this index in the PR that adds a version.

| Document | Current | History | Notes |
|---|---|---|---|
| SRS | `SRS_v4.0.md` + `.pdf` | `SRS_Voice_Spoofing_Detection.pdf` (v3.0) | v4.0 incorporates the current platform plus approved live-analysis, model-v2, reporting and infrastructure decisions D47-D57. |
| SDD | `SDD_Voice_Spoofing_Detection_Lab5.pdf` (v1.0) | - | v2.0 in preparation (Iheb, W3): streaming, process pool, PostgreSQL, environments and extension security. |
| Execution Plan | `Execution_Plan_v3.0.md` + `.pdf` | `Execution_Plan_v2.1.pdf` (D26); `Execution_Plan_v2.2.md` + `.pdf` (draft, never adopted); `Execution_Plan.md` (v1) | v3.0 adopted in PR #113. |
| Proposal | `Voice_Spoofing_Detection_Project_Proposal.docx` | - | Motivation and framing; lowest priority. |

When SDD v2.0 merges, move the old SDD file to History in this table, put the new pair in Current,
and update `AGENTS.md` in the same PR.
