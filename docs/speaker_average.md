# Speaker-average summary (Stage 13 preparation)

Owner: Khalil (D26); D4, D19, FR-12. This helper summarizes in-memory
`SpeakerMetrics` rows. It reads no audio and runs no model.

```python
from src.evaluation.speakers import per_speaker_metrics
from src.evaluation.speaker_average import speaker_average

rows = per_speaker_metrics(
    [0, 1] * 50 + [0, 1],
    [0.1, 0.9] * 50 + [0.9, 0.1],
    ["A"] * 100 + ["B"] * 2,
    threshold=0.5, half_width=0,
)
summary = speaker_average(rows, undefined_policy="propagate")
print(summary["binary.accuracy"])
# SpeakerMean(value=0.5, n_speakers=2, n_defined=2, undefined_speakers=())
```

These are invented scores. A has 100 correct windows, B has two incorrect
windows: the equal-speaker mean is 50%, whereas pooled accuracy is 100/102.
The secondary speaker average must not replace the primary pooled report.

The output has 18 named rate summaries: EER, ROC-AUC, six forced-binary rates,
six non-deferred rates, coverage and three deferral rates. Each includes the
total and defined speaker counts and the sorted IDs with undefined values.
Counts and confusion matrices stay in the individual speaker rows; they are
not averaged. Pass rows from one view and one common frozen policy only.
The helper cannot verify policy provenance from SpeakerMetrics alone.

## Explicit undefined policy — team choice still pending

D19 and Protocol section 8 require a speaker average, but do not specify
undefined-value handling. This development API therefore has no default:

- `propagate`: a metric's mean is NaN if any speaker's value is NaN. Its
  defined count and missing speaker IDs explain why; other metrics still work.
- `raise`: fail with the metric name and affected speakers instead.

Neither option silently excludes speakers or replaces undefined rates with
zero. These are explicit development controls, not approval of a final report
convention. Team review must select and record that convention before the real
external report. No frozen protocol or decision register is changed here.

An empty table, duplicate or invalid speaker IDs, invalid rates, or unknown
policy raises ValueError. Omitting the required policy raises TypeError.

The final report still needs eight-speaker completeness checks (including
speakers lost entirely to filtering), clustered CIs, drop counts, file-level
reporting, approved governance and the frozen ModelBundle. Stage 13 has not run.
