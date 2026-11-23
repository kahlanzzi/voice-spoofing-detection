# Stage 14 robustness v1 — reviewed-run preparation

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
- exactly one matching completed FoR test `stage11_test` registry record for the full v1 set; and
- the Stage 11 run report's exact model identity, row count, CSV hash and sidecar hash.

It therefore cannot silently use an unregistered, incomplete, wrong-manifest or non-bundle clean
file, or a mutually edited CSV/sidecar pair. The completed registry record and run report must name
the exact four-model Stage 11 v1 set, not merely a
partial set containing the requested model. The Stage 14 runner is the third exact file in the FoR
test-split governance allow-list; no directory is allowed.

Condition comparisons require identical clip IDs, labels, D14 groups, model identity, artifact and
manifest. EER increases and the CNN's REAL false-synthetic-rate increases use paired 2,000-resample
D14-group percentile bootstrap intervals with seed 2026. At 3.4 kHz, an increase is meaningful only
when it is at least 0.05 and the interval excludes zero.

## Official runner contract

`python -m src.evaluation.stage14_run` has read-only `check` plus separate `claim` and `score`
commands. Run them once for
`--model-kind baseline` and once for `--model-kind cnn_bundle`; each registry key contains exactly
one model. Both commands require the reviewed Colab 2026.10 Tesla T4 runtime (scoring itself remains
on CPU), the pinned FFmpeg build, exact F2/F3 and Stage 7/8/10 evidence, and the completed Stage 11
clean-score files. The CNN path verifies F4 against the byte-identical committed F4 record and
anchors its checkpoint, profile and policy hashes. The baseline path verifies the exact D41 artifact
and repeats D58's two-build MFCC/reference gate before the claim and before scoring.

`check` performs the complete preflight without creating an output folder or registry record, so the
notebook can be reviewed before irreversible access. `claim` repeats those checks. It refuses an existing
output folder or any prior started/completed Stage 14 record for that model, appends the registry
claim, and writes the durable claim/environment files. `score` checks the claimed run and commit,
creates `scoring_started.json` exclusively before the first test-audio read, then repeats preflight.
An interrupted or failed score is not retried without a new decision.

For every condition, the runner reads each F2 test clip through the hash-verifying reader, applies
the degradation to the decoded 16 kHz float32 waveform, and only then calls unchanged F3. It records
canonical degraded-waveform and feature hashes. Each frozen model scores twice on CPU and the raw
score bytes must be identical; baseline and policy decisions are independently checked. Per-clip
condition files are re-read before the paired metrics are calculated. The final files are
`metrics_report.json`, `run_report.json`, and nine condition score folders; only then is the registry
record completed.

## Remaining before official access

No FoR test audio or DEEP-VOICE audio was opened during implementation. The complete runner and its
notebook hand-off still require team review. After approval, use a fresh Colab 2026.10 T4 session,
run the two claim/score lifecycles in the reviewed order, keep per-clip files on Drive, and commit
only the permitted evidence copies. The v2 model remains the later Stage 14 v2 run after its own
Stage 11 clean scores exist.
