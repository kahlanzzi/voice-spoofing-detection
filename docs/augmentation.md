# Model v2 waveform augmentation (V1a)

Track V step V1a implements the train-only waveform augmentation contract approved in D50 and
amended by D56. This step uses synthetic audio only. It does not download MUSAN or RIRS_NOISES,
read project audio, build the corpus split, train a model or inspect any validation/test result.
Those corpus pins and manifests belong to V1b.

## Contract

`configs/augmentation.yaml` is loaded by `src.augmentation.load_augmentation_config`. The loader is
deliberately strict: unknown, missing or changed values are refused rather than silently accepted.
The approved policy is:

- 16 kHz waveforms and seed 2026;
- augmentation on `train` only, with probability 0.8 for both REAL and FAKE;
- exactly one uniformly selected family when augmentation is applied: recorded noise,
  reverberation, codec round-trip, gain or RawBoost algorithm 1;
- seven codec options sampled uniformly (probability 1/7 each): MP3 24/48/96 kbps, Opus
  8/16/24 kbps and narrowband mu-law telephony;
- white Gaussian noise, MP3 32/64/128 kbps and plain low-pass filtering remain held out for the
  Stage 14 robustness test.

`Augmenter.augment(waveform, split=..., label=..., seed=...)` returns a float32 waveform with the
same number of samples and records the selected family and parameters. It refuses non-training
splits, unknown labels, invalid seeds, non-vector input, NaN/infinity and any transform that breaks
the shape/finite-value contract. The caller supplies only the already-audited training half of the
noise and RIR resources.

## RawBoost algorithm 1

`src/augmentation/rawboost.py` is an independent implementation of the published algorithm 1
equations and the D56 parameter pin. It does not copy the reference implementation. The official
MIT-licensed reference is pinned to commit
`4f161a8b4d0d9f4a8431509ddd86c645869ef6c4`; the synthetic parity test compares the float32 output
for a fixed two-tone input and seed against a fixture produced from that pinned reference. No
reference repository or data is vendored.

## Codec and corpus gates

`src/augmentation/codec.py` performs an ffmpeg encode/decode round trip. It intentionally refuses
to run until V1b records the exact ffmpeg version and codec-library pins in the configuration.

`python -m src.audit.augmentation_corpora` audits local corpus folders without downloading them. It
checks inventory, SHA-256 and audio headers, creates a deterministic 50/50 by-file split with seed
2026, verifies the configured archive/licence pins and writes only small manifests/reports. V1b
must fill the MUSAN and RIRS_NOISES archive/licence hashes before the audit can be accepted.

## Verification

V1a is covered by synthetic tests for the strict configuration, family and codec-option sampling,
train-only and label-symmetric behavior, deterministic transforms, RawBoost reference parity,
corpus split/audit behavior and refusal while the V1b pins are absent.

Run the repository suite with torch mandatory:

```powershell
$env:REQUIRE_TORCH_TESTS='1'
python -m pytest -q
```

The V1a implementation record is Entry 166 in `docs/project_log.md`.
