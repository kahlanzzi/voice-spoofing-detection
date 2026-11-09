# Part 6 software baseline

These are new midterm software-quality measurements, not frozen model or dataset evidence.
Khalil executed the commands in the project Python 3.13.16 environment and supplied their outputs.
The source under `api/` and `src/` is unchanged from `98fd8d7`; documentation commit `7c0e243`
and uncommitted report corrections were present. No source refactor had occurred.

- `coverage.json`: raw Coverage.py 7.17.0 output, branch measurement, scope `api,src`.
- `complexity.json`: raw Radon 6.0.1 `cc -j api src` output.
- `lint.json`: raw Ruff 0.17.3 output, isolated configuration, Python 3.13 target,
  selected rules E4/E7/E9/F, no automatic fixes; exit code 1 indicates two findings.
- `tool-versions.txt`: installed audit-tool version capture, with trailing line whitespace removed and UTF-8/LF used for Git; values unchanged. Original and stored SHA-256 values are in the summary; the original capture is retained outside the repo.
- `summary.json`: readable summary, precise aggregation definitions, source provenance and
  SHA-256 values for the three raw JSON files and original/normalized tool-version captures. The JSON files are copied byte-for-byte from the user output directory.

The instrumented test run passed 937 tests in 73.21 s. The earlier uninstrumented run passed
937 in 89.85 s; this difference is not evidence of a performance optimization.

No `.coverage` database, audio, checkpoint, model or protected data is included. Coverage of
code paths using synthetic fixtures does not imply an authorized real-data run. The coverage
denominator excludes notebooks and tests. No after-refactor evidence exists yet.

Commands and exact reported outputs are in `docs/refactor-optimization-report.md`.

The local `.gitattributes` disables Git line-ending conversion for the three raw JSON files.
Their whitespace rule recognizes the original CRLF endings as line endings, without rewriting them.
Verify their staged blob hashes against `summary.json` before committing; the readable summary
and normalized tool-version capture continue to use the normal repository text rules.
