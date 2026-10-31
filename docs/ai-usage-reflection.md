# AI ethics and usage reflection

**Team:** Donia Mabrouk, Iheb Zemzemi, Malak Ben Salem and Khalil Dridi

**Owner:** Malak Ben Salem

**Status:** Content approved by Donia; team AI-tool inventory confirmed; ready for PR

**Evidence cutoff:** 2026-10-31

**Team confirmation:** the members confirmed that no additional AI tool was used beyond the tools
documented below and in the project log.

## 1. How the team used AI

AI assisted this project, but it did not replace the team's decisions, experiments, reviews or
responsibility. The project record attributes substantial work to three tools:

| Tool | Documented uses | Human responsibility |
|---|---|---|
| Claude Code | Drafting code, tests, notebooks and documentation from team specifications | A named owner reviewed the diff and real outputs before commit or PR approval |
| Claude (chat) | Reviewing diffs, explaining results, proposing corrections and preparing implementation prompts | Team members selected the proposal, ran official commands and accepted or rejected changes |
| Codex | Repository audits, implementation, tests, troubleshooting, documentation, UML work and PR preparation | The task owner reviewed the work; humans opened, approved and merged PRs |

The repository also supports Gemini/Antigravity through the shared `AGENTS.md` context, but the
project log does not identify either as the author of an accepted artifact. Context7 is configured as
an optional documentation MCP server. When it rejected its key during the API work, the limitation
was recorded and the team verified behaviour against installed source and tests instead of claiming
that the MCP result had been used.

AI was most useful for repetitive implementation, synthetic test generation, cross-document checks,
debugging and preparing reviewable drafts. It was not allowed to approve governance decisions or its
own output, merge PRs, interpret a model result as proof of fraud, or authorize access to protected
data.

## 2. Human review and verification

The project's human-in-the-loop rule is explicit in [`AGENTS.md`](../AGENTS.md) and
[`CONTRIBUTING.md`](../CONTRIBUTING.md): every AI-produced artifact must be understood and reviewed
by a team member before acceptance. AI assistance is disclosed in the PR and in
[`project_log.md`](project_log.md). Protected `main`, required CI, CODEOWNERS and PR approval provide
additional checks, but the team does not treat an automated check as a substitute for understanding
the change.

Verification depends on the claim:

- Code changes require real tests and their exact output; synthetic tests are clearly distinguished
  from real-data evidence.
- Experimental results are tied to the dataset version, configuration and exact code commit.
- Claims are labelled `[verified]`, `[doc]` or `[hypothesis]`; a verified result is still evidence for
  human review, not automatic acceptance.
- Official Colab runs and data transfers are performed by a human. AI may help diagnose the recorded
  output but does not silently repeat, tune or reinterpret a governed run.
- PR preparation includes the DEEP-VOICE governance test, changed-file review, token scan and checks
  that audio, datasets, checkpoints and secrets are absent.

## 3. Examples where AI output required correction

The project history demonstrates why AI output was never accepted uncritically:

1. **Review-status overstatement:** an early threat-model entry could be read as team-reviewed even
   though Claude had drafted it and review was still pending. Entry 12 corrected the record, and Entry
   13 formalized the difference between `[verified]` evidence and human acceptance.
2. **Incorrect repository convention:** Codex initially proposed the unsupported branch prefix
   `test/` for synthetic reporting. The branch was renamed to the D23-compliant `chore/` prefix and
   the mistake was recorded in Entry 51.
3. **Inaccurate technical wording:** Claude-assisted D42 wording implied that the Stage 6 benchmark
   network matched the CNN, but the benchmark had no BatchNorm. The discrepancy was recorded rather
   than hidden, and the real Stage 8 run was required to measure memory instead of relying on the
   assumption.
4. **Parallel UML interpretation:** the first Codex UML draft represented an implementation/planning
   view rather than the canonical SRS/SDD figures. It was replaced with source-faithful editable
   diagrams and visually rechecked before PR #70.

A related scientific example was the first Stage 6 candidate at an 8 kHz Mel upper bound. Although
the run was deterministic, it failed the pre-declared D18 invariance gate. The team did not weaken the
gate or call the candidate final. FoR-only diagnostics supported a 7.6 kHz correction, followed by new
tests and a complete rerun. This illustrates that reproducible output can still be unsuitable and must
be evaluated against pre-declared criteria.

## 4. Privacy, security and data governance

Voice recordings are personal data. The repository therefore contains no audio, dataset archives,
checkpoints, personal recordings or literal secrets. The approved design requires uploaded audio to
be temporary and analysis records to contain result metadata rather than raw audio. Secrets use local
environment or Colab secret storage and must never be pasted into an AI prompt.

DEEP-VOICE receives stronger protection because it is the one external generalization test. Before
the single guarded run, D20 permits only download, archive hashing, extraction, metadata listing and
the approved metadata audit; no listening, plotting, feature extraction or scoring is allowed. A
governance test scans the repository for prohibited paths. The shared-tooling design also excludes
Filesystem, Git, Fetch and GitHub-write MCP servers, and no MCP server is permitted during Stage 13.
These restrictions reduce the risk that an AI tool could bypass the repository-level data guard
through Drive, Colab or a network connector.

## 5. Bias, harm and responsible interpretation

This system is decision support, not proof of fraud and not biometric authentication. A false
synthetic label could cause a genuine customer to be treated as suspicious, while a missed spoof could
enable financial harm. For that reason, the interface must expose confidence, risk and an uncertain
outcome, and an analyst must verify consequential cases through an independent channel.

The data also limits what the model can claim. The FoR package lacks speaker metadata, contains a
large codec-history imbalance and may contain bandwidth, silence or padding shortcuts. DEEP-VOICE is
small, imbalanced and based on public figures; a FoR-to-DEEP-VOICE performance gap confounds speakers,
recording conditions and generation method. The team pre-registered shortcut checks, group-aware
reporting, uncertainty and robustness analysis instead of treating high validation scores as general
proof of fairness or real-world reliability. Results must be reported with these limitations and with
class-aware metrics, not accuracy alone.

## 6. Copyright, licensing and attribution

The repository's MIT licence applies to project code only. It does not grant permission to redistribute
the datasets. The FoR Hugging Face repackager did not state a licence, so the original dataset terms
remain relevant. The DEEP-VOICE Kaggle description states MIT and the project records its paper and
authors, but that licence does not necessarily resolve rights in the original public-figure speeches.
Both datasets are restricted to academic use in this project, and their audio is never committed or
released.

AI-generated suggestions do not remove the obligation to respect third-party licences or identify
sources. The team uses official project references and approved decisions as the source of claims,
does not present AI prose as an external scholarly source, and keeps attribution and limitations in
the dataset audit.

## 7. Academic integrity and accountability

The team discloses which AI tool assisted each substantial change and what it did. AI-generated code
and text are part of the team's submission only after a member reads, understands, tests and can
explain them. The team remains responsible for requirements, decisions, experiment execution,
interpretation, citations and the final repository. An AI tool is not treated as an accountable human
author, reviewer or approver, and its confidence is not evidence.

Our main lesson is that AI is valuable as a fast drafting and checking partner, but its speed increases
the need for traceability. The strongest safeguards in this project are not statements that the tools
are accurate; they are frozen requirements, pre-declared gates, exact provenance, synthetic tests,
governance checks, append-only corrections, independent review and honest disclosure when a tool or
assumption fails.

## Evidence used

- [`AGENTS.md`](../AGENTS.md): standing AI, data, testing and human-review rules.
- [`CONTRIBUTING.md`](../CONTRIBUTING.md): responsible AI use and disclosure requirements.
- [`project_log.md`](project_log.md): tool attribution, review history, corrections and real checks.
- [`decisions/README.md`](decisions/README.md): D20, D22, D25 and D27.
- [`dataset_audit.md`](dataset_audit.md): provenance, licensing, bias and rights limitations.
- [`threat_model.md`](threat_model.md): personal-data, false-positive and analyst-overreliance risks.
- [`decisions/0006-shared-ai-tooling.md`](decisions/0006-shared-ai-tooling.md): MCP scope and exclusions.
