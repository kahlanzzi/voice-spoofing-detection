# Stage 13 — Guarded external evaluation

| | |
|---|---|
| **Stage** | 13 — DEEP-VOICE external evaluation |
| **Owner** | Khalil Dridi (D26) |
| **Reviewers** | All four team members |
| **Status** | Draft preparation brief; stage open, official execution not authorized |
| **Date** | 2026-11-25 |
| **Related PRs** | #133, #138, #147, #149 |

## 1. In one paragraph

[doc] Stage 13 measures how frozen models trained on FoR TTS speech generalize to RVC voice
conversion in DEEP-VOICE. Khalil runs the frozen v1 bundle, frozen v2 bundle and baseline together
under one dataset-wide claim. D48 permits the v1-plus-baseline fallback only after its cutoff if
F5 was not verified in time. Stored scores support the primary speech-filtered results, unfiltered
sensitivity results, speaker summaries and secondary file-level results without further inference.
This draft explains preparation; it contains no external performance results and does not close
the stage. See [Execution Plan v3.0 §10](../reference/Execution_Plan_v3.0.md).

## 2. Inputs → outputs

| | Artifact | Source / consumer |
|---|---|---|
| In | Verified F4 bundle and frozen baseline | Donia → Khalil; [F4 evidence](../audit_evidence/stage12_bundle/README.md) |
| In | Verified F5 bundle and freeze record, or D48 cutoff exclusion | Donia → Khalil; [D48](../decisions/0012-model-v2-track-and-joint-external-run.md) |
| In | Hash-pinned metadata manifest | [Frozen manifest](../../data/manifests/deepvoice_frozen_manifest.csv) |
| In | Passing, hash-pinned D58 evidence matching the run runtime | [Config template](../../configs/external_run.yaml) and runner verification |
| In | Reviewed paths, approved durable registry/output target and execution readiness | Team → Khalil |
| Out | One CSV and integrity sidecar per model, with common row columns | Stage 13 reporting; Khalil's Stage 15/16 analysis |
| Out | Report and finalized dataset-wide claim, with hashes and line count | All-four review; project log and closing evidence PR |

[doc] Paths come from the reviewed configuration. The placeholders in the config template are
not executable defaults. A successful dummy Drive persistence check does not create or authorize
the official claim.

## 3. How it works, in execution order

1. Complete code review and synthetic checks. Before claiming, independently verify the approved
   Drive target's persistence using harmless dummy data and a new runtime. Record the exact paths,
   hashes and approvals; do not create the official claim during this rehearsal.
2. Verify the frozen metadata-manifest hash, F4 and baseline identities, the D48 model set and real
   F5 record when applicable. Use the architecture-aware bundle loader and its verified in-memory
   profile/policy. Recheck D58 against the actual run runtime.
3. `prepare_guarded_run` builds the scorers, creates the dataset-wide claim and verifies its
   persisted bytes **before any external audio read or model inference**. Calling it is an
   irreversible run step, not a harmless readiness check.
4. `score_guarded_dataset` consumes that prepared claim. Each complete, non-overlapping 2-second
   window receives frozen Stage 6 preprocessing separately; the short remainder is dropped.
   Store all complete-window scores and the frozen D17 filter flag so both views use one pass.
5. Validate score/run/artifact identities and complete window membership. Write per-model CSVs,
   sidecars and reports. Finalize the claim as succeeded, or retain an incomplete claim on failure.
   Never automatically retry or resume a partially executed external run.
6. Record final registry SHA-256 and line count immediately, preserve the output bytes and obtain
   all-four review. Downstream external-data analyses read these stored scores, not the audio.

[verified from source at `48aeeb1`] The preparation and scoring order above is implemented in
[`external_run.py`](../../src/evaluation/external_run.py). The actual storage/runtime behavior must
still be demonstrated at the run gate. [ADR 0007](../decisions/0007-stage13-reporting-and-access.md)
defines the durability and failure requirements.

## 4. Key decisions and reporting rules

- **D17/D19:** speech-filtered primary view, unfiltered sensitivity view, per-source-speaker tables.
  Filter rejection is distinct from model deferral.
- **D33:** retain all eight expected source speakers. If any speaker's rate is undefined, its
  equal-speaker mean is undefined with affected IDs/reasons; never average only the defined subset.
- **D34:** average calibrated window probabilities within each file/view, then apply the frozen
  threshold and band. Zero retained windows means no mean and no prediction. File CIs retain
  source-speaker clustering. Baseline raw scores are not calibrated probabilities.
- **D35/D36/D48:** one claim for the dataset across all model sets; one stored-score file per model.
  F5 deadline: **4 December 2026, 23:59 Africa/Tunis**. A different run ID or directory never permits
  another external run. The plan targets the joint run for 5–6 December.
- **D58:** baseline MFCC reproduction and exact-runtime evidence precede the claim.
- **D52:** prepare the descriptive live warm-up analysis on simulated val-B calls; retain the
  approved three-kept-window warm-up rule. Report external end-of-call accuracy using D34.

[doc] Report the full D4 metric set, coverage/deferral, counts and clustered uncertainty, not
accuracy alone. Cover H2, H8 (paired v1–v2), RQ2 (CNN versus baseline), filter/drop counts and file
exclusions. State the dataset's documented **87.5% FAKE share and RVC-only scope**; these are not
new measurements from this preparation. Sources: [Protocol](../experimental_protocol.md),
[decision register](../decisions/README.md) and [plan §10](../reference/Execution_Plan_v3.0.md).

## 5. Evidence and remaining gates

- [verified from committed record] F4 manifest SHA-256:
  `edf4a205e12fc948a68575ebaa929145481aaa535d34e3353ef9535f92840efc`.
- [verified from committed record] Baseline artifact SHA-256:
  `42cc4d53a7cd881f56e4fa54fcf4797de8885b404d0a2b31c293555c868df10a`.
  Both identities are in [f4_record.json](../audit_evidence/stage12_bundle/f4_record.json).
- [doc] #133 supplies guarded scoring/reporting; #147 corrects optional-Torch synthetic tests.
  #138 supplies the architecture-aware loader and synthetic handover test (Entry 186).
  Code availability does not establish verification of the future real F5 artifacts.
- [doc] Entry 198 / #149 records scratch runtime and personal-Drive read-back evidence. It does
  not establish completion of the approved-target persistence and sharing requirements.
- [doc] Remaining: reviewed invocation with real F5 or the applicable cutoff exclusion; D58
  confirmation in the run session; approved-target durability/access evidence; D52 val-B warm-up;
  final runner review and execution readiness. After execution, add score/report/claim hashes,
  results PR and D32 teach-back evidence before accepting this brief.

## 6. Limitations

- No official external run or claim was made for this draft. No external performance is established.
- Synthetic checks exercise behavior, not generalization or real Drive durability.
- The text-based governance allowlist is not a security sandbox; review actual I/O ordering too.
- Eight source speakers and one RVC dataset do not establish performance on all speakers, codecs,
  spoofing methods or fraud scenarios. File means are secondary scores, not newly calibrated risks.
- A failed run retains its claim. Further action requires team review; this brief grants no rerun.

## 7. Questions for the D32 teach-back

1. **Why wait for F5?** All eligible frozen models must share the single untouched external run;
   running v1 early would consume the dataset-wide opportunity (D48).
2. **What must happen before decoding?** Verify artifacts, metadata, runtime and approvals, then
   durably create and read back the claim. Failure before verified persistence prevents scoring.
3. **Is an empty filtered file authentic?** No. It has no mean or prediction; report it separately
   and exclude it from the relevant metric denominator (D34).
4. **How is a file verdict calculated?** Average stored calibrated probabilities for the chosen
   view and apply the frozen band once; do not recalibrate the mean or vote over labels.
5. **What happens after a disconnect?** Preserve the claim and evidence and stop for review;
   changing the output folder or model set does not authorize another attempt.
6. **Can results choose the deployed model?** No. D48 fixes deployment independently of test results.
7. **What does Stage 15 receive?** Integrity-checked per-model scores and provenance from the
   reviewed run. Its FoR-only preparation can proceed separately under its own rules.

## 8. How to verify

From the repository root with its documented test dependencies, run synthetic checks only:

```bash
python -m pytest -q tests/evaluation/test_external_run.py tests/governance/test_deepvoice_access.py
python -m pytest -q tests/bundle/test_model_loader_v2.py
python -m pytest -q
git diff --check
```

[doc] Record actual pass/skip/failure output, Python/dependency versions and tested commit in the
PR. Follow [setup](../setup.md) for Torch and PostgreSQL requirements; skipped groups are not passes.
There is deliberately no official scoring command in this draft: the reviewed invocation and
approved paths must accompany the real run gate. See the [runner](../../src/evaluation/external_run.py)
and [config template](../../configs/external_run.yaml). Subsequent verification uses stored outputs;
the official external evaluation is not rerunnable.
