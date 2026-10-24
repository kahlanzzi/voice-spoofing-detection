# Experimental Protocol — Voice-Spoofing Detection for Fraud Prevention

**Version:** 1.0
**Freezes at v1.0 = F1 (D1); after F1, changes need a decision (§12, Change control).**
**Authors:** Donia Mabrouk (lead), Malak Ben Salem, Khalil Dridi, Iheb Zemzemi (D26).

This protocol fixes every experimental rule before any model exists. Each section lists what is
decided, one bullet per rule, with its basis (D-ID, FR / NFR, H-ID or source). Rules required by an
approved decision are copied from it. Rules marked "(team choice, 2026-10-14)" are one defensible value
among several, accepted by all four members on 2026-10-14. Sources: decision register
`docs/decisions/README.md` (D-IDs, H5, H6), Execution Plan v2.1 (`docs/reference/Execution_Plan_v2.1.pdf`,
"v2.1" below), SRS v3.0, `docs/threat_model.md`, `docs/project_log.md` Open items. All values are fitted
on FoR only; nothing here touches DEEP-VOICE before Stage 13 (AGENTS.md hard rule 1).

---

## 0. Purpose, scope, deployment scenario

- **Owner:** Donia Mabrouk
- **Reviewer:** all four
- **Decided:**
  - Purpose: fix every rule before any model exists; this protocol at v1.0 is freeze F1 (D1).
  - Scope: FoR-2sec (TTS) → DEEP-VOICE (RVC) generalization; the cause of any gap (e.g. the generation
    method) stays a hypothesis (D8).
  - Deployment scenarios: primary = bank call-center triage of high-risk requests; secondary =
    executive voice message, as in the threat model §3 (threat model §13 Q1) (team choice, 2026-10-14).
  - Language: "English speech" is stated as [doc] (dataset papers), not [verified]; DEEP-VOICE audio may
    not be inspected (threat model §13 Q3; D20) (team choice, 2026-10-14).
- **Status:** Decided

## 1. Research questions and hypotheses

- **Owner:** Donia Mabrouk
- **Reviewer:** all four
- **Decided:**
  - RQ1–RQ4 as in v2.1 Stage 3: RQ1 FoR → DEEP-VOICE generalization; RQ2 CNN vs baseline, in-domain
    and externally; RQ3 robustness under degradation; RQ4 calibration under shift (v2.1) (team choice,
    2026-10-14).
  - H1: CNN EER < baseline EER on the FoR test, and the 95% CI of the difference excludes 0; tested in
    Stage 11 (v2.1) (team choice, 2026-10-14).
  - H2: external EER (DEEP-VOICE) > in-domain EER (FoR test) for the ModelBundle; tested in Stage 13
    (v2.1) (team choice, 2026-10-14).
  - H3: ECE on the stored DEEP-VOICE scores > ECE on the FoR test; tested in Stage 15 (v2.1) (team
    choice, 2026-10-14).
  - H4: EER increases from clean to severe for each degradation; tested in Stage 14 (v2.1) (team
    choice, 2026-10-14).
  - H5 and H6 copied unchanged from the register (below); "clearly lower" and "raises" are made
    operational in §7 and §9 (register).

  | ID | Hypothesis | Test | Stage |
  |---|---|---|---|
  | H5 | If the CNN relies on MP3 traces, its recall on FAKE_other is clearly lower than on FAKE_mp3. | Subgroup recall on the FoR test split | 11 / 16 |
  | H6 | If the CNN relies on bandwidth, band-limiting REAL clips to ~3.4 kHz raises false FAKE calls. | Robustness test on FoR test REAL clips | 14 |
  | H7 | If the CNN relies on pauses (silence), genuine REAL clips with many pauses are flagged as synthetic more often than REAL clips with few. | On the FoR test split, from the stored Stage 11 per-clip outputs (no second test run, D31): forced-binary false-synthetic rate of the bundle CNN at its frozen threshold on REAL clips in the top quartile of silence fraction vs the bottom quartile; quartile boundaries from REAL clips of the train split only; silence fraction from the frozen Stage 6 shortcut_stats on the F3 waveform. 'Raises' = at least 5 points higher and the 95% CI of the difference excludes 0 (bootstrap, 2,000 resamples, seed 2026, clustered by near-duplicate group; Protocol §5). Reported descriptively: Spearman correlation of calibrated p with silence fraction within FAKE_other, and the same comparison for the baseline. Requires Stage 11 to store per-clip path, group, raw score, calibrated p and prediction. | 16 |
- **Status:** Decided

## 2. Data and split

- **Owner:** Malak Ben Salem
- **Reviewer:** Donia Mabrouk
- **Decided:**
  - FoR-2sec is re-split; no packaged split is used (D13).
  - Grouping: near-duplicate components at 0.95; no group crosses splits (D14).
  - Stratification by REAL / FAKE_mp3 / FAKE_other (D16).
  - Ratios 70 / 15 / 15 (train / validation / test), group-aware (`docs/dataset_audit.md` §0).
  - Validation halved 50 / 50 into val-A (training decisions) and val-B (threshold, calibration), same
    stratification (val-B ≈ 665 clips per class) (D9).
  - Split seed 2026, the shared project seed (§4; log Open item 0.10) (team choice, 2026-10-14).
  - Automated split checks: no group in two splits; class and subgroup proportions within ±1 point of
    the full set in every split; split sizes reported (v2.1 Stage 4) (team choice, 2026-10-14).
  - DEEP-VOICE file manifest: file-level metadata only (path, label, source / target speaker, sample
    rate, channels, duration, SHA-256) (D20).
  - Freeze F2 contains the FoR manifests (path, label, subgroup, group, split, hash) and the DEEP-VOICE
    file manifest, each with its own hash (D26 B→C).
- **Status:** Decided

## 3. Preprocessing profile

- **Owner:** Malak Ben Salem
- **Reviewer:** Donia Mabrouk, Iheb Zemzemi
- **Decided:**
  - Channels averaged to mono; resampled to 16 kHz, identically everywhere (D18).
  - Fixed input length 2 s = 32,000 samples (D5).
  - Log-Mel for the CNN: 64 mel bands, 25 ms window, 10 ms hop, log magnitude (team choice,
    2026-10-14).
  - MFCC for the baseline: 20 coefficients; mean + std over time = 40 values (D7; count: team choice,
    2026-10-14).
  - Silence detector: energy-based recommended (simple, deterministic, no extra model); the final
    choice and settings are recorded as a decision in Stage 6 and frozen in F3; the D17 filter uses the
    same detector (D17; log Entry 12) (team choice, 2026-10-14).
  - Loudness normalization: decided in Stage 6, after the shortcut re-screen (`docs/dataset_audit.md`
    §3.3; D16).
  - Pipeline-invariance test: FoR clips converted to 40 / 44.1 / 48 kHz stereo; resampler and tolerance
    are fixed in Stage 6 (D18).
  - Shortcut re-screen (silence, bandwidth, loudness) re-run after preprocessing (D16).
  - Freeze F3 contains the versioned profile config, the code commit, the feature settings and the
    detector settings (v2.1 Stage 6) (format: team choice, 2026-10-14).
- **Status:** Decided

## 4. Models and training plan

- **Owner:** Donia Mabrouk
- **Reviewer:** Khalil Dridi
- **Decided:**
  - Baseline: RBF-SVM on MFCC mean + std, evaluation only (D7); C and gamma grid-searched on val-A
    (grid: team choice, 2026-10-14).
  - Deployed model: one small CNN on log-Mel spectrograms (3–4 conv blocks); early stopping on val-A
    EER; no degradation augmentation (AGENTS.md hard rule 3; D10) (size: team choice, 2026-10-14).
  - Model selection on val-A only; test split and DEEP-VOICE untouched (D9).
  - Seed policy: CNN seeds 0, 1, 2 on the final configuration; split seed and bootstrap seed 2026 (D11;
    log Open item 0.10) (team choice, 2026-10-14).
  - Every run logged to W&B with its config; compute benchmark before Stage 8 (AGENTS.md code rules;
    v2.1).
- **Status:** Decided

## 5. Metrics and statistics

- **Owner:** Donia Mabrouk
- **Reviewer:** Khalil Dridi
- **Decided:**
  - Core metrics: D4 in full (accuracy, precision, recall, F1, EER, ROC-AUC, 3-way confusion matrix,
    coverage / deferral, class counts, FPR / FNR); positive class = spoof; no t-DCF (D4; AGENTS.md hard
    rule 5).
  - Seed reporting: mean ± SD of EER and ROC-AUC across the 3 seeds on the FoR test, at score level
    only (D11) (team choice, 2026-10-14).
  - Bundle seed: the seed with the median val-A EER; tie → lower seed number (D11; scope lock) (team
    choice, 2026-10-14).
  - CIs: bootstrap 95% percentile CIs, 2,000 resamples, on the bundle seed; the seed SD is reported as
    a separate row (D11) (team choice, 2026-10-14).
  - Resampling unit: FoR = near-duplicate groups; DEEP-VOICE = source speakers (D11; D14) (FoR unit:
    team choice, 2026-10-14).
- **Status:** Decided

## 6. Risk policy and calibration fit

- **Owner:** Donia Mabrouk
- **Reviewer:** Khalil Dridi, Iheb Zemzemi
- **Decided:**
  - Fitted on val-B only, for the bundle seed, before F4 (D2; D9).
  - Calibration: Platt scaling (logistic, 2 parameters) on val-B (FR-18) (team choice, 2026-10-14).
  - Operating threshold: the val-B EER point on the calibrated probability p (D12; D9).
  - Uncertain band: symmetric around the threshold; width set so that 10% of val-B is deferred (D12)
    (team choice, 2026-10-14).
  - Maximum deferral rate: 10% on val-B (D12) (team choice, 2026-10-14).
  - Confidence: max(p, 1 − p), rounded to 2 decimals, in [0.00, 1.00] (FR-06) (team choice,
    2026-10-14).
  - Risk level: authentic → LOW, uncertain → MEDIUM, synthetic → HIGH (FR-07) (team choice,
    2026-10-14).
  - Freeze F4 contains the CNN checkpoint + preprocessing profile config + policy file (threshold, band,
    calibration parameters, version) + a manifest with the SHA-256 of each (D2; AGENTS.md hard rule 2)
    (format: team choice, 2026-10-14).
- **Status:** Decided

## 7. FoR internal test (Stage 11)

- **Owner:** Donia Mabrouk
- **Reviewer:** Khalil Dridi
- **Decided:**
  - Run once: the baseline and all 3 CNN seeds on the same FoR test split; this is the single Stage 11
    run (FR-12; D11) (team choice, 2026-10-14).
  - Reporting: all D4 metrics with class counts, per subgroup (D4; FR-14).
  - H5 "clearly lower": FAKE_other recall ≥ 5 points below FAKE_mp3 recall, and the 95% CI of the
    difference excludes 0 (H5) (team choice, 2026-10-14).
- **Status:** Decided

## 8. External run (Stage 13)

- **Owner:** Khalil Dridi
- **Reviewer:** all four
- **Decided:**
  - Windows: non-overlapping 2 s windows from file start; a remainder < 2 s is dropped (D5).
  - Evaluation unit: window (primary); file-level mean of p (secondary) (D5).
  - Speech-window filter: a window is kept if ≥ 50% of its frames are speech according to the detector
    frozen in F3 (D17) (team choice, 2026-10-14).
  - Views: filtered (primary) and unfiltered (sensitivity) from the same single run; drop counts per
    class and per speaker (D17).
  - Per-speaker table: 8 rows + speaker average (D19).
  - Stored scores: one CSV row per window with file, window index, source speaker, target speaker,
    label, raw score, calibrated p, prediction, kept_by_filter; plus bundle version and run ID (v2.1
    Stage 13; D26 C→D) (columns: team choice, 2026-10-14).
  - Single-run rule: exactly one run of the frozen bundle, registered once (AGENTS.md hard rule 1;
    FR-15).
  - Reporting: next to the in-domain result, with the gap, the RVC-only scope and the 87.5% FAKE share
    stated (FR-16; D8).
  - Governance: a new decision, before Stage 13, on adding the external-run code to the allowlist of
    `tests/governance/test_deepvoice_access.py` (log Open items) (open; to be decided before Stage 13).
- **Status:** Decided

## 9. Robustness (Stage 14)

- **Owner:** Malak Ben Salem
- **Reviewer:** Khalil Dridi
- **Decided:**
  - Data: FoR test split only; no DEEP-VOICE (D3).
  - Noise: white Gaussian noise at SNR 30 / 20 / 10 dB (mild / moderate / severe) (FR-17) (team
    choice, 2026-10-14).
  - Compression: MP3 at 128 / 64 / 32 kbps (FR-17) (team choice, 2026-10-14).
  - Bandwidth: low-pass at 6 / 4 / 3.4 kHz; 3.4 kHz = telephone band (FR-17; H6) (team choice,
    2026-10-14).
  - H6 "raises": false-synthetic rate on REAL clips at 3.4 kHz ≥ 5 points above clean, and the 95% CI
    excludes 0 (H6) (team choice, 2026-10-14).
  - Pipeline: the degradation is applied to the waveform, then the same frozen preprocessing profile
    (D26 B→Stage 14).
- **Note:** FAKE clips already carry MP3 history (D16), so the compression curve also informs H5;
  it is reported, not avoided.
- **Status:** Decided

## 10. Calibration and error analysis plans (Stages 15–16)

- **Owner:** Khalil Dridi
- **Reviewer:** Donia Mabrouk
- **Decided:**
  - Reliability diagram and ECE with 10 equal-width bins, before and after calibration (FR-18) (bins:
    team choice, 2026-10-14).
  - Data: the FoR test split and the stored DEEP-VOICE scores; no re-run (v2.1 Stage 15).
  - Uncertain band under shift: the band still means something if the error rate inside it is higher
    than outside; the DEEP-VOICE deferral rate is also compared with the 10% on val-B (RQ4) (team
    choice, 2026-10-14).
  - Error analysis by subgroup, speaker, confidence bin and deferral; each finding linked to threats
    T1–T7 (v2.1 Stage 16).
- **Status:** Decided

## 11. Interfaces

- **Owner:** Iheb Zemzemi
- **Reviewer:** Donia Mabrouk, Malak Ben Salem, Khalil Dridi
- **Decided:**
  - B→C manifests: CSV columns path, label (REAL / FAKE), subgroup (REAL / FAKE_mp3 / FAKE_other),
    group, split (train / val-A / val-B / test), sha256 (D26).
  - B→C preprocessing: input = waveform, any sample rate and channel count; output = mono 16 kHz
    float32, 32,000 samples; log-Mel array (1 × 64 × frames) for the CNN; 40-value MFCC vector for the
    baseline (D18; §3) (shapes follow §3; team choice, 2026-10-14).
  - C→D scoring: `score(audio) → raw score` (one float, higher = more likely spoof); the risk policy
    turns it into p, prediction, confidence and risk level (D26) (team choice, 2026-10-14).
  - C→D stored scores: the per-window CSV defined in §8 (D26; §8).
  - C→E API response: {prediction, confidence, risk_level, model_version, profile_version,
    policy_version, request_id, notice} (D26; FR-05–FR-07; NFR-07) (extra fields: team choice,
    2026-10-14).
  - C→E bundle loading: load the bundle folder; verify the SHA-256 of each part against its manifest
    and that all three versions match; refuse to start otherwise (NFR-09; threat model §10) (team
    choice, 2026-10-14).
  - B→Stage 14 degradations: conditions and levels as in §9, applied to the waveform before the frozen
    preprocessing profile (D26; §9).
- **Status:** Decided

## 12. Freeze points F1–F4

- **Owner:** Donia Mabrouk
- **Reviewer:** all four
- **Decided:**
  - F1 = this protocol at v1.0 (Stage 3); F2 = split + manifests (Stage 5); F3 = preprocessing profile
    (Stage 6); F4 = ModelBundle (Stage 12). After a freeze, the frozen item changes only through a new
    decision (D1; v2.1 §6; AGENTS.md hard rule 6).
- **Change control:** (1) Items deferred by design (silence detector, loudness normalization,
  resampler and invariance tolerance) are decided in Stage 6 and frozen in F3. (2) Any other change
  before the affected data is seen needs a new decision (D27 onwards) with its reason, logged before
  the affected step runs. (3) No change after results are seen; any extra analysis is labelled
  exploratory.
- **Status:** Decided

## 13. Sign-off

| Member | Role | Date | PR |
|---|---|---|---|
| Donia Mabrouk | Lead (author); §0, 1, 4–7, 12 | 2026-10-14 | #10 |
| Malak Ben Salem | §2, 3, 9 | 2026-10-14 | #10 |
| Khalil Dridi | §8, 10 | 2026-10-14 | #10 |
| Iheb Zemzemi | §11 | 2026-10-14 | #10 |

## 14. Change log

| Version | Date | Change | PR |
|---|---|---|---|
| 0.1 | 2026-10-12 | Skeleton: sections, owners, reviewers, "Must decide" items. No content. | |
| 1.0 | 2026-10-14 | All sections decided (team-accepted defaults). | #10 |
| 1.0 | 2026-10-17 | Clarifications D29 (baseline threshold), D30 (search space), D31 (Stage 11 run once); no existing rule changed; fills gaps the protocol left open | #16 |
| 1.0 | 2026-10-18 | D36: baseline scored in the single external run (RQ2); C→D stored scores per model | #29 |
| 1.0 | 2026-10-18 | D37: baseline grid, scaling and score fixed (completes §4 'grid: team choice'); no existing rule changed | #30 |
| 1.0 | 2026-10-20 | D40 narrows §4's 'every run logged to W&B' to Stage 8 onwards (§4 text unchanged; D40 governs); H7 pre-registered (silence, Stage 16) | #40 |
| 1.0 (proposed amendment) | 2026-10-24 | D41 proposes per-clip clean-score storage in §7 and adding `baseline-v1` beside the bundle CNN in §9; requires all-four approval before protocol text changes or FoR test access | pending |
| 1.0 | 2026-10-24 | D42: CNN design and training details fixed (completes §4 'size: team choice'); no existing rule changed | #44 |
