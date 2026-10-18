# Per-speaker analysis: first preparation step

Owner: Khalil (D26). Requirements: FR-12, D4, D19.

`src/evaluation/speakers.py` groups in-memory scores by **source speaker** and
calls the existing evaluation harness. It does not load audio, run a model,
fit calibration, choose a threshold, or perform the external evaluation.

## Synthetic example

Run from the repository root, using the project's Python environment:

```python
from src.evaluation.speakers import per_speaker_metrics

rows = per_speaker_metrics(
    labels=[0, 1, 0, 1],
    scores=[0.1, 0.9, 0.8, 0.2],
    source_speakers=["example_A", "example_A", "example_B", "example_B"],
    threshold=0.5,
    half_width=0.1,
)
for row in rows:
    print(row.source_speaker, row.n, row.eer, row.roc_auc,
          row.binary.accuracy, row.three_way.coverage)
```

Expected output (invented data, not model results):

```text
example_A 2 0.0 1.0 1.0 1.0
example_B 2 1.0 0.0 0.0 1.0
```

## Input and result contract

- One aligned input row per evaluation unit, normally a two-second window.
  Labels: 0 = bonafide, 1 = spoof. Higher scores indicate spoof.
- The caller supplies the threshold and band on the same scale as the scores.
  For the calibrated CNN policy, supply calibrated p, not raw logits.
- Every speaker uses the same supplied frozen policy. The diagnostic threshold
  returned by the harness EER calculation is discarded.
- Results are sorted by source-speaker ID. IDs must be nonempty strings with no
  surrounding whitespace. Case is preserved. Target speakers are not used.
- Each row contains sample count, EER, AUC, forced-binary metrics, and three-way
  metrics including confusion counts, coverage, class-specific deferral and
  non-deferred metrics. These use the existing harness definitions unchanged.
- A speaker with only one class remains in the table; EER/AUC are NaN because
  they are undefined. Other undefined rates also remain NaN, never zero.
- Choose filtered or unfiltered rows before calling. This helper does not
  silently filter, aggregate windows into files, or drop missing speaker IDs.

## Remaining Stage 13 work

This is a tested building block, not a complete Stage 13 report or approval to run.
The final report still needs the eight-speaker completeness check, the secondary
speaker average (with an explicit undefined-metric reporting convention),
source-speaker clustered confidence intervals, filter drop counts, file-level
secondary view and provenance. The file-level policy and external-run governance
allowlist require the team's decisions before the guarded external run. The
ModelBundle must be frozen first. No real dataset is needed for these unit tests.
