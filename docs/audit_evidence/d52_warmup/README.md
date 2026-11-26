# D52 descriptive val-B warm-up — 26 November 2026

**Owner:** Khalil Dridi. **Acceptance: pending human review.**
This note records observed output, not independent reproduction or Stage 13 approval.
It serves D52 (9) and Execution Plan v3.0 §10 (Part D).

## Preserved evidence

- Notebook: [D52 val-B warm-up — Khalil](https://colab.research.google.com/drive/1y5Ffl0XPdaaj-KmHhwfaLFODi4aDR6L_).
- Drive folder: `MyDrive/voice-spoofing-detection-d52-khalil-output/20261126T223751Z/`.
- Files: `d52-call-plan.json`, `d52-report.json`, `environment.txt`,
  `invocation.json`, `scoring.log`, `verification-summary.json`, `SHA256SUMS.txt`.
  Keep the generated files unchanged; reviewer access must be confirmed separately.
- Reviewed code: `b9fb99027fb7c59f64b9f57a6e62f5bb3ae33cb3` (merged PR #177).
  The notebook checked HEAD and a clean checkout before scoring.
- Report SHA-256: `810d7d6f832bf68c14d8b35da043a1fed7a2c8e586e9ece51ba143ffc62c41e5`.
- Plan SHA-256: `3fc1e898a30eb34b5f1cde9f2354fecc00b3fa463b2cb4902e15135c8a9d79a9`.
- F4 manifest SHA-256: `edf4a205e12fc948a68575ebaa929145481aaa535d34e3353ef9535f92840efc`.
- Input archive SHA-256: `9bddb83503ab442e9c7f54a1b7b9c02bc0ce32ed93f7863f28de5f59b6540b46`.

[verified, observed in Colab] The export exited with code 0. A separate cell read
the saved report bytes back and checked its digest. A further cell verified hashes
for the six evidence files before writing `SHA256SUMS.txt`. These checks were in
the **same runtime**, not an independent fresh-runtime durability rehearsal.
No generated score arrays, audio or checkpoints are committed with this note.

## Method and actual runtime

The seed-2026 plan uses 44 FAKE and 44 REAL constructed calls, each containing
15 same-label, two-second val-B clips (30 seconds). The 1,320 planned clips were
extracted selectively and individually hash-checked against the frozen manifest;
nine FAKE leftovers were excluded. The scorer verifies val-B membership, applies
D17 and the frozen profile, calibrates each retained window once, then applies
the frozen policy to the mean of all kept windows at checkpoints 3, 5 and 10.
This does not use the last-five-window trend as a verdict.

The loaded model was `cnn-v1`, profile `preprocessing-v1`, policy `risk-v1`.
Threshold: `0.6333105553100875`; half-width: `0.3636377690579665`.
No parameter was fitted or changed.

- Start: `2026-11-26T22:38:56.832525+00:00`.
- Finish: `2026-11-26T22:39:27.792455+00:00` (about 31 seconds for the export).
- Device: CPU; Python **3.13.18**; platform `Linux-6.6.159+-x86_64-with-glibc2.39`.
- Torch `2.11.0+cpu`, safetensors `0.8.0`, NumPy `2.5.3`, SciPy `1.18.1`,
  librosa `1.0.0`, soundfile `0.14.0`; full dependency listing in `environment.txt`.
- The Colab runtime selector was set to `2026.10`, but the observed Python was
  **3.13.18**, not the **3.13.15** named in `AGENTS.md`. Do not describe this as
  an exact reference-runtime match. Donia must review this discrepancy before
  accepting the evidence; this note does not amend D15 or satisfy D58.

## Observed checkpoint results

All 88 calls reached every checkpoint; none was excluded as unreached.
Verdict accuracy below is correct non-deferred verdicts / **all 88 reached calls**.
Uncertain verdicts are not counted as correct.

| Kept windows | Correct verdicts | Uncertain | Verdict accuracy | Coverage | Deferral |
|---|---:|---:|---:|---:|---:|
| 3 | 75 / 88 | 13 | 85.23% | 85.23% | 14.77% |
| 5 | 74 / 88 | 14 | 84.09% | 84.09% | 15.91% |
| 10 | 65 / 88 | 23 | 73.86% | 73.86% | 26.14% |

All 44 REAL calls received authentic verdicts at each checkpoint. FAKE calls had
31 / 30 / 21 synthetic verdicts and 13 / 14 / 23 uncertain verdicts respectively;
no opposite-class verdict occurred. Thus the three-way confusion counts are
REAL: `(44 authentic, 0 synthetic, 0 uncertain)` at each checkpoint and
FAKE: `(0, 31, 13)`, `(0, 30, 14)`, `(0, 21, 23)`.

Forced-binary metrics at the frozen central threshold were accuracy, precision,
recall and F1 = 1.0, with TN=44, TP=44, FP=0, FN=0; EER=0 and ROC-AUC=1 at each
checkpoint. Non-deferred precision, recall, F1 and accuracy were also 1.0, on
75 / 74 / 65 calls respectively. The report retains the full confusion and
class-specific deferral rates. These perfect metrics require scrutiny, not a
claim of perfect detection.

## Interpretation and review gate

This is **descriptive val-B evidence**. Val-B was used for policy/calibration fit;
these same-label constructed sequences are not held-out natural calls, independent
speaker samples, or external generalization evidence. No confidence intervals or
statistical significance claims are made. A reviewer should inspect membership,
grouping and the stored scores for shortcuts before accepting the interpretation.

More windows did not monotonically improve verdict accuracy here: more FAKE calls
fell into the frozen uncertain band at 10 kept windows. This is an observation,
not permission to tune the band, select a checkpoint, or change D52's fixed minimum
of **three kept windows**. The run does not validate microphone capture, streaming
transport, live latency, v2 performance or the final external file-level rule.

Review requested from Donia: independently verify the saved file hashes and full
report, the runtime discrepancy, denominators and limitations. Record the review
on the evidence PR. No DEEP-VOICE audio was accessed and no external claim was
created. Official Stage 13 still requires F5 or the D48 cutoff, the applicable
model set, D58 recheck, approved durable claim storage and all-four run review.

AI assistance: Codex operated the reviewed val-B export in Colab, checked its
visible output and read-back checks, and drafted this note. Human acceptance is pending.
