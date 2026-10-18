# Stage NN — <Stage name>

<!-- Copy this file to docs/stages/stageNN-short-name.md (name from the table in README.md).
     Whole brief: 2–4 pages. Plain words. Link to existing docs instead of copying them.
     Label claims as AGENTS.md does: [verified] / [doc] / [hypothesis]. Delete these comments. -->

| | |
|---|---|
| **Stage** | NN — <Stage name> |
| **Owner** | <name> (D26) |
| **Reviewer(s)** | <name(s)> (D26: the consumer reviews the producer) |
| **Status** | Draft / In review / Accepted |
| **Date** | YYYY-MM-DD |
| **Related PRs** | #NN, #NN |

## 1. In one paragraph

*What the stage does and why the pipeline needs it. Write it so a classmate outside the project
understands it. Target: one paragraph (4–6 sentences).*

## 2. Inputs → outputs

*What the stage receives (from which stage and owner) and what it hands over (to whom), with file
paths. Name the hand-over if it is one of the four D26 interfaces. Target: a short table or 4–8
bullets.*

| | What | From / to | Path |
|---|---|---|---|
| In | | Stage NN (<owner>) | |
| Out | | Stage NN (<owner>) | |

## 3. How it works, step by step

*The steps in plain words, in the order they run. A small Mermaid diagram if it helps. Link to the
design note or protocol section for the details. Target: 5–10 numbered steps, under one page.*

1. …

## 4. Key decisions and why

*One line each: the decision, its D-ID, and the reason in a few words. Only the decisions this stage
depends on or produced. Target: 3–8 lines.*

- **D-NN** — <decision>: <why>.

## 5. Evidence

*Where to look: tests, reports, hashes, logs. Give real numbers with their source file (and commit or
log entry). No number without a source. Target: 4–8 bullets.*

- <number or fact> — source: `<path>` (Entry NN / commit `abcdef0`).

## 6. Limitations

*What we cannot claim from this stage, and why. Include known weak points the reviewer should
know about. Target: 3–6 bullets.*

- …

## 7. Questions the professor may ask

*5–8 questions with short answers (1–3 lines each). The reviewer must be able to answer these
without asking the author (teach-back review, D32).*

1. **Q:** …
   **A:** …

## 8. How to verify or re-run it

*The exact commands (local and/or Colab), what they need (data, config, commit), and the output to
expect. Target: one code block plus 2–4 lines.*

```bash
python -m pytest -q tests/<area>/
```
