# Stage 6 — Preprocessing profile v1 (F3)

| | |
|---|---|
| **Stage** | 6 — Preprocessing profile v1 |
| **Owner** | Malak Ben Salem (D26, Part B) |
| **Reviewer(s)** | Donia Mabrouk and Iheb Zemzemi (D26 / PR #34) |
| **Status** | Accepted (PR #37) |
| **Date** | 2026-10-19 |
| **Related PRs** | #34 (Stage 6 evidence and review corrections; merged at `4707476`) |

## 1. In one paragraph

Stage 6 defines one deterministic way to turn every two-second FoR recording into the waveform and
features that later model stages consume. It starts from the exact F2 manifest and checks each
sample hash before reading audio; it then validates preprocessing invariance, re-screens possible
class shortcuts, and measures compute without training a model. The reviewed candidate, D39/F3,
uses a 16 kHz mono waveform and fixed 64-bin log-Mel features for the CNN, plus 20 MFCCs summarized
for the baseline. Its corrected FoR-only evidence passed the recorded gates and was reviewed and
merged in PR #34; the first-run diagnostic archive was not retained, so no unverified hash is
claimed. This brief records the method for the D32 teach-back; completing that review is still
required before marking the stage done.

## 2. Inputs → outputs

| | What | From / to | Path |
|---|---|---|---|
| In | Frozen 17,721-row FoR manifest and F2 hash ledger | Stage 5 (Malak) | `data/manifests/for2sec_frozen_manifest.csv`; ledger hash pinned in `configs/preprocessing.yaml` |
| In | Frozen profile settings and reviewed implementation | Stage 6 | `configs/preprocessing.yaml`; implementation commit `6e1718547756ef04744ce0b3b1b73bb839920052` |
| Out | Shared waveform preprocessing and CNN log-Mel / baseline MFCC features | Stage 7 (Donia), then later consumers | `src/preprocessing/`; B→C interface in D26 |
| Out | Validation, shortcut/padding, invariance, determinism, and dry-run benchmark evidence | Reviewers and later stages | `docs/audit_evidence/stage6_preprocessing/` |

The official validation reads FoR only. DEEP-VOICE is not an input and was not accessed.

## 3. How it works, step by step

1. Load the F2 manifest and verify its pinned SHA-256, exact columns and row count before reading
   audio. Verify each row's sample hash before decoding its recording.
2. Average channels to mono, resample with `soxr_hq` to 16 kHz, and deterministically trim only
   leading/trailing frames below −50 dBFS using 20 ms frame-RMS windows. Preserve all-silent input.
3. Centre-pad or crop the result to exactly 32,000 samples (2 s). Record the amount of zero-padding
   per item; the screen reports it by class and subgroup.
4. For CNN input, calculate 64 natural-log Mel bands from 0 to 7.6 kHz using a 512-point FFT,
   400-sample Hann window, 160-sample hop, no centring, power 1, and log epsilon `1e-6`. The
   resulting shape is `(1, 64, 197)`; no additional feature normalization is applied.
5. For the MFCC baseline, calculate 20 DCT-II coefficients (`norm=ortho`, `lifter=0`) and summarize
   each by its mean and standard deviation, giving 40 values. Train-only standardization is a
   separate baseline decision (D37), not part of this profile.
6. Re-screen waveform RMS, silence fraction, bandwidth and padding separation. Loudness mode is
   `none` because the pre-registered train-only RMS AUC is below 0.75. Also report class/subgroup
   padding and the all-split descriptive screen without using test rows as a gate.
7. Test D18 by converting 10 clips per subgroup to 40, 44.1 and 48 kHz stereo, processing them
   through the same pipeline, and comparing the outputs against the 16 kHz references.
8. Run two clean validations and compare artifacts byte-for-byte. Separately benchmark candidate
   batches on the Tesla T4 with forward/backward only; no optimizer, weight update or checkpoint is
   allowed. The largest candidate within 80% of available VRAM was selected; batch size 256 is
   recorded in D39 and `configs/cnn.yaml`.

Detailed settings and implementation are in [`configs/preprocessing.yaml`](../../configs/preprocessing.yaml),
[`src/preprocessing/`](../../src/preprocessing/), and the
[Stage 6 notebook](../../notebooks/02_stage6_preprocessing_profile.ipynb).

## 4. Key decisions and why

- **D18** — mono averaging and 16 kHz resampling are shared everywhere; the measured tolerance is
  fixed so preprocessing cannot vary silently across components.
- **D39** — approve the waveform, trimming, feature, loudness and compute profile; the 8 kHz Mel
  edge failed the first D18 candidate, so the corrected 7.6 kHz edge is used. The benchmark's
  selected batch size is recorded in D39 and `configs/cnn.yaml`.
- **D26** — Stage 6's consumer receives the frozen profile I/O through the B→C hand-over; the
  downstream runner must apply it consistently to each complete 2 s window.

D39/F3 was approved by Donia and Iheb in PR #34 and merged at `4707476` (Entry 69). The brief's
D32 teach-back is a separate closeout requirement.

## 5. Evidence

- [verified] All 17,721 F2 rows and sample hashes were checked; all recorded validation gates passed
  — source: `docs/audit_evidence/stage6_preprocessing/profile_report.md` and `.json` (Entry 66,
  implementation commit `6e1718547756ef04744ce0b3b1b73bb839920052`).
- [verified] D18 covered **180 comparisons**; maximum relative L2 error was **0.0021238185** (limit
  0.01) and minimum cosine similarity was **0.9999977627** (minimum 0.999) — source: `invariance.csv`
  and `profile_report.json`.
- [verified] Train-only FAKE-vs-REAL RMS AUC was **0.698602**, below the pre-registered 0.75
  threshold, so loudness normalization remains off. Train-only FAKE_other-vs-REAL silence AUC was
  **0.749901 (LOW)**; the **0.750071 MODERATE** all-split value is descriptive, includes test, and is
  not a gate — source: `shortcut_screen.csv`, D39 and Entry 66.
- [verified] All-split mean post-trim padding was **0.003747** of a 2 s window for FAKE and
  **0.004567** for REAL; the train-only means were **0.003688** and **0.004619**, respectively. The
  report also gives per-subgroup summaries — source: `profile_report.md`.
- [verified] The final config SHA-256 is
  `7bd35bfdcb9270dd55ea9e73449c7bf6c6a359a80335b2551c0c89dde77c8630`; its profile fingerprint is
  `5b4f5849a9bb63f9016dd79969ff5bfb9e37006acbff1e26bbfee4ee2bad68f2` — source: D39 and
  `profile_report.json`.
- [verified] The Tesla T4 benchmark processed 256 train clips at **90.666 clips/s**, selected batch
  size **256** with **15,637,086,208 bytes** total VRAM, and performed zero optimizer steps or
  checkpoint writes — source: `benchmark.md` and `benchmark.json`; the selected batch size is
  recorded in D39 and `configs/cnn.yaml`.
- [verified] The two corrected validation runs were byte-identical. The initial 8 kHz diagnostic
  archive was not retained; Entry 64 records its summary and does not claim an archive hash — source:
  Entries 64–67 and `determinism.json`.

## 6. Limitations

- [verified limitation] The 7.6 kHz edge was selected after examining the same 30 converted clips
  used by D18; the invariance check is therefore not an independent test of that choice.
- [verified scope; effect [hypothesis]] D18 starts from 16 kHz FoR audio and upsamples it to test
  rates. It does not test anti-aliasing when downsampling natively wideband audio; the effect of
  that untested case is unknown.
- [verified limitation] Edge trimming removes only leading and trailing low-energy frames; internal
  pauses remain. Carry this limitation into Stage 16; do not alter the frozen profile after seeing
  model results.
- [verified limitation] The all-split FAKE_other silence AUC includes test data and is descriptive
  only. It must not be used for tuning, threshold selection, or changing D39.
- [doc limitation] The benchmark is compute-only. It establishes a candidate batch size, not CNN
  training speed, convergence, accuracy or model quality.
- [verified limitation] No initial-candidate archive/hash is available. The official evidence is
  FoR-only, and DEEP-VOICE was not accessed.

## 7. Questions the professor may ask

1. **Q:** Why does the final profile use 7.6 kHz instead of 8 kHz?
   **A:** The 8 kHz Mel edge was unstable on the first D18 resampling run. The correction moved the
   edge below Nyquist; it passed the declared tolerance, with the same-clip selection limitation
   disclosed above.
2. **Q:** Why is there no loudness normalization?
   **A:** The pre-registered train-only RMS separability AUC was 0.698602, below 0.75. The rule
   therefore selects `mode: none`; it was not changed after looking at test results.
3. **Q:** Does the 0.750071 silence AUC fail the shortcut gate?
   **A:** No. It is an all-split descriptive FAKE_other comparison that includes test. The gate uses
   train only, where the AUC was 0.749901 and flagged LOW.
4. **Q:** What does the D18 check show, and what does it not show?
   **A:** It shows close agreement after converting 30 FoR clips to three sample rates and stereo.
   Since the clips began at 16 kHz, it does not establish anti-alias behavior for native wideband
   downsampling.
5. **Q:** Why report padding by subgroup?
   **A:** Trimming changes the remaining waveform length before fixed-window padding. Per-class and
   per-subgroup counts expose whether added zeros differ systematically; padding also gets its own
   separability screen.
6. **Q:** How did the benchmark choose 256?
   **A:** It measured candidate batches and selected the largest successful one within 80% of the
   Tesla T4's VRAM. The run used forward/backward only, with no optimizer or saved checkpoint.
7. **Q:** What is frozen as F3, and what remains to close Stage 6?
   **A:** D39 records the profile settings, IDs and evidence references; PR #34 approved and merged
   the candidate. The D32 teach-back for this brief still has to be completed before marking the
   stage done.

## 8. How to verify or re-run it

Use the clean implementation commit and pinned FoR F2 inputs only. In Colab, the checked-in
[`notebooks/02_stage6_preprocessing_profile.ipynb`](../../notebooks/02_stage6_preprocessing_profile.ipynb)
is the official runner; the extracted FoR archive and its revision marker are supplied separately
and are not committed. Use fresh output directories for each run.

```bash
python -m pytest -q tests/preprocessing/
python -m src.preprocessing.validation \
  --root <extracted_for2sec_root> \
  --manifest data/manifests/for2sec_frozen_manifest.csv \
  --profile configs/preprocessing.yaml \
  --out <fresh_validation_output>
python -m src.preprocessing.benchmark \
  --root <extracted_for2sec_root> \
  --manifest data/manifests/for2sec_frozen_manifest.csv \
  --profile configs/preprocessing.yaml \
  --out <fresh_benchmark_output> \
  --device cuda
```

The validation should report 17,721 rows and all checks passing; compare both validation outputs
byte-for-byte. The benchmark requires a CUDA GPU and should report batch size 256 for the recorded
environment. Do not supply DEEP-VOICE or overwrite committed evidence. See Entry 66 for the official
run provenance and Entry 69 for the review/merge record.
