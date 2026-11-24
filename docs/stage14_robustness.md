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
  sections and `sosfiltfilt`. This is zero-phase forward/backward filtering, so its effective filter
  order is doubled; the configured design order remains eight.

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

### H6 endpoint clarification (requires explicit all-four approval)

The H6 endpoint is the forced-binary false-synthetic rate on REAL clips at the frozen F4 policy
threshold: calibrated probability `p >= threshold` is synthetic, including clips whose stored
three-way decision is `uncertain`. Clean and degraded three-way synthetic and uncertain rates remain
in the report as descriptive values only. They do not determine H6. This definition must be named
explicitly in the all-four approval record before any FoR test access.

## Official runner contract

`python -m src.evaluation.stage14_run` has read-only `check` and `time-condition` commands plus
separate `claim` and `score` commands. D41 is read as two independent single-model claims: run one
lifecycle for `--model-kind baseline` (`baseline-v1`) and a second lifecycle for
`--model-kind cnn_bundle` (the frozen F4 bundle seed). Each registry key contains exactly one model;
the two models are not one combined claim. All commands require the reviewed Colab 2026.10 Tesla T4
runtime (scoring itself remains
on CPU), the pinned FFmpeg build, exact F2/F3 and Stage 7/8/10 evidence, and the completed Stage 11
clean-score files. The CNN path verifies F4 against the byte-identical committed F4 record and
anchors its checkpoint, profile and policy hashes. The baseline path verifies the exact D41 artifact
and repeats D58's two-build MFCC/reference gate before the claim and before scoring.

`check` performs the complete preflight without creating an output folder or registry record, so the
notebook can be reviewed before irreversible access. Notebook 11 extracts val-A only in this mode;
FoR test members are extracted only in official mode. `time-condition` measures feature extraction
and two-pass CPU scoring for `low_pass_3400hz` on all 1,330 val-A rows, separately for the baseline
and bundle. The expected two-lifecycle scoring time is the sum of nine times each observed condition
time. The first reviewed check-mode run must copy the printed numeric estimate into Entry 216 before
official approval; no value is invented in advance.

`claim` repeats the preflight and requires `docs/audit_evidence/stage14_v1_approval.json`, copied from
current `main` before the exact approved commit is checked out. That record must bind the commit,
the H6 definition above and explicit evidence references from Donia, Iheb, Khalil and Malak. It refuses an existing
output folder or any prior started/completed Stage 14 record for that model, appends the registry
claim, and writes the durable claim/environment files. `score` checks the claimed run and commit,
creates `scoring_started.json` exclusively before the first test-audio read, then repeats preflight.
An interrupted or failed score is not retried without a new decision. The two official lifecycles
must therefore run as one attended, uninterrupted session sized from the check-mode timing. They
must not overlap another registry writer. The current schedule places Stage 14 after the v2 Stage 11
notebook 09 run on 2 December; an earlier run would require its updated registry evidence merged by
1 December.

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
