# Stage 14 robustness v1 — preparation

Status: implementation under review; no FoR test audio has been opened by this work.

Stage 14 measures the frozen v1 baseline and CNN bundle seed under the robustness grid in Protocol
section 9. It is a new `stage14_robustness` purpose under D41, not another Stage 11 clean run.

## Frozen condition contract

`configs/robustness.yaml` is strict. Its ordered grid is:

- white Gaussian noise at 30, 20 and 10 dB SNR;
- MP3 at 128, 64 and 32 kbps through the FFmpeg version and library pins already recorded in
  `configs/augmentation.yaml`;
- an eighth-order Butterworth low-pass at 6,000, 4,000 and 3,400 Hz, applied with second-order
  sections and forward/backward filtering.

White-noise power is mean square. Each realization uses NumPy's `default_rng` with the little-endian
unsigned 64-bit value formed from the first eight bytes of
`SHA-256("<global seed>\n<clip id>\n<condition id>")`; the global seed is 2026. This makes each
clip/condition realization deterministic without sharing one noise vector between clips. These
implementation details are explicit review points before any test access.

All inputs and outputs are non-empty, one-dimensional, finite float32 at 16 kHz with unchanged
length. A zero-power waveform is refused for an SNR condition. Degradation happens before unchanged
F3 feature extraction.

## Clean-score hand-over and paired statistics

Stage 14 reuses the Stage 11 clean per-clip files. `stage14_run.read_stage11_clean_scores` requires:

- exact F2 manifest bytes;
- a `stage11_test` sidecar with the F2 hash;
- baseline or frozen CNN bundle role; and
- exactly one matching completed FoR test `stage11_test` registry record containing that model.

It therefore cannot silently use an unregistered, incomplete, wrong-manifest or non-bundle clean
file. The Stage 14 runner is the third exact file in the FoR test-split governance allow-list; no
directory is allowed.

Condition comparisons require identical clip IDs, labels, D14 groups, model identity, artifact and
manifest. EER increases and the CNN's REAL false-synthetic-rate increases use paired 2,000-resample
D14-group percentile bootstrap intervals with seed 2026. At 3.4 kHz, an increase is meaningful only
when it is at least 0.05 and the interval excludes zero.

## Remaining before the official run

The same reviewed runner file still needs the two-phase claim/score flow, degraded score writer,
unchanged F3/model integration, D58 gate before baseline scoring, report/export commands and
synthetic end-to-end tests. Then the team reviews the complete runner and notebook before the single
official FoR test access. DEEP-VOICE is outside Stage 14 and must not be opened.
