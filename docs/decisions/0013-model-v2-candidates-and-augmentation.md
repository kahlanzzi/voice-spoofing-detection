# 0013. Model v2: candidates, robust selection and training augmentation

- **Status:** Approved (all four; PR #111, 2026-11-10)
- **Date:** 2026-11-09
- **Register:** D49 (amends AGENTS.md hard rule 3 and Execution Plan v2.2 §1.3 for model v2 only),
  D50 (supersedes D10 for model v2 only); H9
- **Deciders:** Donia Mabrouk (Part C, author of D49), Malak Ben Salem (Part B preprocessing, owner of
  the D50 pipeline), Iheb Zemzemi, Khalil Dridi
- **AI assistance:** Claude (claude.ai, Claude Opus 5.5) drafted this ADR; Donia and Malak review it.

## Context

- v1 is one CNN on F3 log-Mel features (D42), selected on clean val-A (D30) with **val-A EER 0 for all
  three seeds** (Entry 86). A selection rule based on clean val-A EER alone can no longer tell
  candidates apart: they tie at zero.
- FoR is clean studio TTS speech. The target use (call-centre triage, voice messages; Protocol §0)
  brings codecs (Opus for WhatsApp, narrowband telephony), background noise, rooms and level changes.
  D10 excluded augmentation from v1 on purpose.
- FoR has a codec shortcut risk: 85% of FAKE clips and 0% of REAL clips carry an MP3 step (D16; H5).
- The professor asked for LSTM-type or voice-specific models. The spoofing literature uses purpose-built
  graph-attention models (AASIST) and large self-supervised front-ends (wav2vec 2.0 / XLS-R); the
  latter are strong but heavy for a free T4 and for CPU inference in live mode (D51, D53).
- Stage 14 (§9) degrades the FoR test split with white Gaussian noise (SNR 30/20/10 dB), MP3
  (128/64/32 kbps) and low-pass filtering (6/4/3.4 kHz). If training used the same conditions, Stage 14
  would measure memorised conditions, not robustness.

## Decision

- **D49:** three pre-declared candidates trained on FoR train with D50 augmentation and the unchanged
  F3 profile: `cnn-aug` (v1 architecture, the control that separates data from architecture), `crnn`
  (D42 conv blocks + BiLSTM, the temporal model) and `aasist` (raw-waveform graph attention). XLS-R is
  excluded and recorded as future work. Selection by the lowest mean val-A EER over a fixed
  clean-plus-degraded condition set; the winner gets seeds 0–2, a val-B risk policy by the unchanged
  Protocol §6 method, and is frozen as F5. One deployed model.
- **D50:** on-the-fly waveform augmentation of the training split only, REAL and FAKE alike, five
  families, with parameters disjoint from the Stage 14 grid; white Gaussian noise is held out entirely
  so that Stage 14 keeps one unseen degradation family (H9). FoR stays the only speech data.

Full texts: D49 and D50 in [`README.md`](README.md).

## Consequences

- **Easier:** the v1 → `cnn-aug` comparison isolates the effect of data; `cnn-aug` → `crnn` / `aasist`
  isolates architecture (reported at validation level for all three; only the selected model reaches
  the test sets). Codec augmentation on REAL clips weakens the MP3-only-on-FAKE association (D16).
- **Committed:** exact candidate configurations, batch sizes (from a short compute benchmark, as D30),
  noise/RIR corpus pins and file splits, codec tool versions and the RawBoost implementation commit
  are written in `configs/` and the Protocol addendum **before** any v2 training run (D48 ordering).
- **Committed:** Stage 14 labels each §9 condition for v2 as *unseen family* (white noise), *seen family,
  unseen parameters* (MP3) or *seen family* (bandwidth, because narrowband telephony is a training
  family). H6 remains a v1 hypothesis. H4 and the D41 paired method apply to v2 unchanged.
- **Committed:** new third-party data (noise and room-impulse corpora) gets a licence and hash record
  before use, like the Stage 1 audit and D21 pins.
- **Harder:** three training pipelines instead of one (two input types: log-Mel and waveform); degraded
  val-A copies must be built deterministically once and reused at every epoch; W&B logging (D40) for
  every run.
- **Risk:** training time on a free T4. Mitigation: one seed per candidate during selection, a 30-epoch
  cap with patience 5, and a timing benchmark in week 1; the D48 cut-off bounds the schedule.

## Options considered

1. **XLS-R 300M (+ AASIST or linear head).** Rejected for this semester: fine-tuning a 300M-parameter
   front-end on a free T4 within the schedule is uncertain, and CPU inference would be too slow for
   live mode (D53's 1 s window budget). Future work.
2. **LSTM alone on frame features.** Rejected: the CRNN keeps the convolutional front-end that already
   works on F3 log-Mel and adds the temporal model the professor asked for.
3. **Pre-generated augmented copies on disk.** Rejected: fixed variety, extra storage, and copies of
   FoR audio outside the pinned dataset.
4. **Augment FAKE clips only, or add white noise to training.** Rejected: the first creates a new
   shortcut; the second removes Stage 14's only unseen family.
5. **Add extra speech datasets (e.g. multilingual deepfake corpora).** Rejected for v2: licence and audit
   work, and it would mix "more data" into the v1–v2 comparison. Future work.

## Links

- [Decision register](README.md): D9, D10, D11, D16, D30, D39, D40, D41, D42, D44, D45, D48–D50; H4–H6,
  H9.
- [ADR 0012](0012-model-v2-track-and-joint-external-run.md).
- [Experimental Protocol](../experimental_protocol.md): §§4, 6, 9, 12.
- [CNN design](../cnn.md), [feature caches](../feature_cache.md), [log-Mel cache](../logmel_cache.md).
