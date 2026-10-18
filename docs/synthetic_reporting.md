# Synthetic reporting integration checks

These tests exercise approved D4/D11/D17/D19 capabilities with invented numbers only.
They do not measure model accuracy or authorize the external run.

Run: `python -m pytest -q tests/evaluation/test_synthetic_reporting.py`.

The fixed seed is 2026. Eight fictional IDs have REAL counts 50, 80, 120, 170, 230,
300, 400 and 520; each has seven times as many FAKE windows. Totals are 1,870 REAL
and 13,090 FAKE, 87.5% FAKE. These are chosen test dimensions, not newly verified
external-dataset counts. Speaker error fractions range from zero to 0.5. Scores
are invented probabilities, not fitted or calibrated model outputs. Threshold 0.5
and half-width 0.1 are test constants. Artificial keep flags drop every tenth row
within each speaker/class; no speech detector is run. Generation performs no I/O.

Six tests cover reproducibility and dimensions; known per-speaker errors/deferrals;
equal-speaker versus pooled accuracy; filter counts; a missing-speaker roster check;
and a source-speaker clustered accuracy interval against a count-based oracle.
The interval uses existing defaults: 2,000 draws, seed 2026, 95% percentile CI.
All accuracy resamples are defined, so n_invalid is zero; existing bootstrap unit
tests cover invalid resamples. This does not implement macro-mean CIs or EER/AUC CIs.

The missing-speaker test documents that the current helper reports observed IDs
only: the separate production completeness/reporting layer is still required.
The speaker-average test uses the existing explicit `raise` development option
on fully defined metrics; no final D33 reporting policy is selected. File means,
zero-window report serialization and guarded execution remain separate work.
No synthetic CSV, audio, model checkpoint or dataset is committed.
