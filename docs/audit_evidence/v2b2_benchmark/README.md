# V2b-2 — D49 batch-size benchmark (Colab T4)

Evidence for the batch sizes in `configs/v2_candidates.yaml` (D49; D61 (7): the benchmark writes
them before any v2 training run). Project log Entry 173. Tool: `src/tuning/v2_benchmark.py`
(`docs/v2_training.md`, "Batch-size benchmark"). No dataset was read and no weights were saved
(`data_read: false`, `weights_saved: false` in every file).

## Runtime and commit

| Item | Value |
|---|---|
| Code | clean detached checkout of `9c46bafba7e220867f3b5fdd7b4d66ebc0573991` (main after PR #129, V2b-1) |
| Colab runtime | 2026.10, T4 GPU |
| GPU | Tesla T4, 15360 MiB (`nvidia-smi`); `total_vram_bytes` = 15,637,086,208 B (torch) |
| CPUs | 2 |
| Python / torch | 3.13.16 / 2.11.0+cu130 (CUDA 13.0, cuDNN 92700) |
| numpy / safetensors / wandb | 2.5.3 / 0.8.0 / 0.31.0 |
| `v2_config_sha256` | `527c368183e676f2f156cd90033e938fc2a644cdd83e9aa6144b51c36bd46841` (the null-batch file at 9c46baf) |
| `augmentation_config_sha256` | `512653b3af02b1a9ff31c09a39b7597784f1fcb099c827075df656ba330cec84` |
| `profile_fingerprint` | `5b4f5849a9bb63f9016dd79969ff5bfb9e37006acbff1e26bbfee4ee2bad68f2` (F3) |

## How it was run

The exact cells are in [`colab_cells.md`](colab_cells.md). In order:

1. Clone, detach at the commit, assert a clean worktree (re-checked after every cell).
2. Environment gate: T4, torch 2.11.0+cu130, python 3.13.16; `pip install -r requirements.txt -r
   requirements-cnn.txt`.
3. `pytest tests/training tests/tuning/test_v2_benchmark.py tests/models` with
   `REQUIRE_TORCH_TESTS=1`: **428 passed, 1 skipped** (the AASIST faithfulness test, `AASIST_REF_DIR`).
4. GPU, twice (`gpu_run1`, `gpu_run2`):
   `python -m src.tuning.v2_benchmark --out <dir> --device cuda --throughput-items 0`.
   The cell asserts both runs choose the same sizes and that no choice errored.
5. CPU throughput, once (`cpu_throughput`):
   `python -m src.tuning.v2_benchmark --out <dir> --device cpu --candidates cnn-aug --batch-sizes 16
   --throughput-items 20 --codec-unpinned-timing`.
6. Copy each `benchmark.json` to Drive as `<run>.json` (hash checked after the copy), download.

## GPU rows (both runs)

Benchmark settings: candidates 16–256, 3 warm-up + 10 measured steps, `optimizer_step: true`,
`max_vram_fraction: 0.8`. "Fits" = peak ≤ the 80 % budget below.

| Candidate | Batch | Status | Peak allocated (B), both runs | Fits | Mean step s, run 1 | Mean step s, run 2 |
|---|---:|---|---:|---|---:|---:|
| `cnn-aug` | 16 | ok | 232,463,360 | yes | 0.0103 | 0.0105 |
| `cnn-aug` | 32 | ok | 324,660,224 | yes | 0.0146 | 0.0143 |
| `cnn-aug` | 64 | ok | 525,933,568 | yes | 0.0211 | 0.0217 |
| `cnn-aug` | 128 | ok | 981,625,856 | yes | 0.0410 | 0.0423 |
| `cnn-aug` | 256 | ok | 1,893,527,040 | yes | 0.0862 | 0.0882 |
| `crnn` | 16 | ok | 236,527,104 | yes | 0.0087 | 0.0077 |
| `crnn` | 32 | ok | 328,375,808 | yes | 0.0123 | 0.0124 |
| `crnn` | 64 | ok | 529,350,144 | yes | 0.0214 | 0.0219 |
| `crnn` | 128 | ok | 987,622,912 | yes | 0.0425 | 0.0436 |
| `crnn` | 256 | ok | 1,897,988,096 | yes | 0.0879 | 0.0901 |
| `aasist` | 16 | ok | 3,539,112,960 | yes | 0.4957 | 0.5177 |
| `aasist` | 32 | ok | 6,985,602,048 | yes | 0.9438 | 0.9823 |
| `aasist` | 64 | ok | 13,895,315,968 | **no** | 1.8368 | 1.9251 |
| `aasist` | 128 | oom | — | — | — | — |
| `aasist` | 256 | oom | — | — | — | — |

## 80 % budget

budget = `total_vram_bytes` × 0.80 = 15,637,086,208 × 0.80 = **12,509,668,966.4 B**.
The rule (`choose_batch_size`, Stage 6): the largest batch with status `ok` and peak ≤ budget.

- `cnn-aug` 256: 1,893,527,040 ≤ budget (12.1 % of VRAM); 256 is the largest candidate size.
- `crnn` 256: 1,897,988,096 ≤ budget (12.1 %); 256 is the largest candidate size.
- `aasist` 32: 6,985,602,048 ≤ budget (44.7 %); 64 peaks at 13,895,315,968 > budget (88.9 %);
  128 and 256 run out of memory.

## Chosen batch sizes

| Candidate | `batch_size` |
|---|---:|
| `cnn-aug` | **256** |
| `crnn` | **256** |
| `aasist` | **32** |

Both runs record `chosen_batch_size` = `{aasist: 32, cnn-aug: 256, crnn: 256}` and empty
`choice_errors`. Written to `configs/v2_candidates.yaml`; `tests/training/test_v2_batch_size_evidence.py`
checks the config against these files.

## Why the two GPU files have different hashes

The two runs differ **only in `mean_step_seconds`** (wall-clock timing, which varies run to run). Every
other field is identical, including every row's status and `peak_allocated_bytes` (byte-identical) and
the chosen sizes. The choice depends on status and peak only, so the timing difference cannot change it.

## Difference from Stage 6 (optimizer step)

Stage 6 (D39) measured forward + backward only, without an optimizer. This benchmark adds each
candidate's **real optimizer step** (Adam keeps two state tensors per parameter, which is real
training memory). The selection rule is unchanged and the measured peak can only be ≥ Stage 6's
measure, so the chosen batch is at least as safe (`docs/v2_training.md`).

## CPU throughput — **unpinned, planning only**

`codec_timing: "unpinned"`: ffmpeg was **not** verified against a pin (V1b fills the pin); the run
recorded `ffmpeg version 6.1.1-3ubuntu5` instead. These numbers are for planning only and decide
nothing. Synthetic 2 s audio and synthetic noise/RIR resources; 20 items per family (190 draws).

Per family (augment step):

| Family | Share | Mean s | p95 s |
|---|---:|---:|---:|
| clean | 0.20 | 0.0005 | 0.0006 |
| codec | 0.16 | **0.4215** | 0.5083 |
| gain | 0.16 | 0.0008 | 0.0009 |
| rawboost | 0.16 | 0.0346 | 0.0535 |
| recorded_noise | 0.16 | 0.0010 | 0.0011 |
| reverberation | 0.16 | 0.0038 | 0.0043 |

Per stage: augment mean 0.0770 s (p95 0.4522), F3 0.0007 s, log-Mel + stats 0.0072 s; per item
0.0819 s (log-Mel candidates) and 0.0747 s (waveform). **The codec (0.42 s/item) dominates**:
0.16 × 0.4215 ≈ 0.067 s of the 0.077 s mean augment time. `FFmpegCodecBackend.round_trip` runs
`ffmpeg -version` on every call and that cost is included.

Estimated augmentation time per epoch (12,404 train clips):

| Workers | log-Mel (`cnn-aug`, `crnn`) | waveform (`aasist`) |
|---:|---:|---:|
| 1 | 1015 s (16.9 min) | 926 s (15.4 min) |
| 2 | 508 s (8.5 min) | 463 s (7.7 min) |
| 4 | 254 s (4.2 min) | 232 s (3.9 min) |

**Caveat:** the estimate assumes per-item time scales linearly with workers; Colab's 2 vCPUs (often
one physical core) may not give that, so 2 and 4 workers are optimistic. GPU step time and the S
scoring time are not included. Real RIR lengths change the reverberation cost. The single CPU
benchmark row (`cnn-aug` 16, 0.1699 s/step on CPU, no peak) is not used.

## File hashes (SHA-256)

| File | SHA-256 |
|---|---|
| `gpu_run1.json` | `74f8024a9d4cd52ece24534b7eadf8286c3361a79484673a9db4845d06695de0` |
| `gpu_run2.json` | `2efeb019ee74a65554fe423daa1ecfd694a2db2c2c92334a6681f25d66cb5fa9` |
| `cpu_throughput.json` | `039d07793f5bf7c0653468f45126f2a92d85241d1d92c7c1ecbcba1ff1493f68` |

The JSON files are byte-for-byte copies of the Colab output (not reformatted).
