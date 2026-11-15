"""D49 batch-size benchmark per v2 candidate and CPU per-item throughput (Track V, PR V2b-1).

GPU part (the Stage 6 rule, choose_batch_size, with fraction, candidates, warm-up and measured
steps read from configs/preprocessing.yaml `benchmark`): per candidate and batch size, set_seed(0),
build_model, train mode, a synthetic input of the candidate's shape ((b, 1, 64, 197) or (b, 32000)),
random 0/1 targets, BCEWithLogitsLoss, forward + backward + the candidate's real optimizer step;
per batch: status ok/oom, mean step seconds, peak allocated bytes. Difference from Stage 6 (D39 text:
forward/backward only, no optimizer): the optimizer step is included, because Adam's state (two
tensors per parameter) is real training memory; the selection rule is unchanged and the measured peak
can only be >= Stage 6's, so the choice is at least as safe. No weights saved, no data read.

CPU part: per-item transform time on synthetic 2 s audio (fixed seed) with synthetic noise/RIR
resources: per family (clean, recorded_noise, reverberation, codec, gain, rawboost) the augment
time, and per stage (augment / F3 / log-Mel + stats) mean and p95 seconds; then an estimated epoch
time = 12,404 x (D50 mix: clean share + equal family shares) / workers, which assumes linear
scaling with workers (Colab's 2 vCPUs, often one physical core, may not give it). The S scoring time
is not estimated (GPU). Codec: FFmpegCodecBackend if configs/augmentation.yaml pins ffmpeg; otherwise only
with --codec-unpinned-timing, through UnpinnedTimingCodecBackend (defined here, timing only), which
records the ffmpeg version instead of verifying it and marks the result codec_timing "unpinned".
FFmpegCodecBackend.round_trip runs `ffmpeg -version` on every call; that cost is measured as part of
the codec time and not changed here.

Training code never imports this module (tested), so the unpinned backend cannot reach a run.

Output: benchmark.json in a new --out folder outside the repository. The config is not edited:
PR V2b-2 writes the chosen batch sizes into configs/v2_candidates.yaml.

CLI: python -m src.tuning.v2_benchmark --out DIR --device cuda|cpu [--candidates ...]
     [--batch-sizes ...] [--throughput-items N] [--codec-unpinned-timing]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.augmentation import AugmentationResources, Augmenter, load_augmentation_config
from src.augmentation.codec import FFmpegCodecBackend
from src.features.raw_train_cache import check_outside_repo
from src.models.v2_checkpoint import ARCHITECTURES, INPUT_SHAPES, INPUTS, build_model
from src.preprocessing.benchmark import choose_batch_size
from src.preprocessing.features import log_mel_spectrogram
from src.preprocessing.profile import PreprocessingError, load_profile
from src.preprocessing.validation import code_version, sha256_file
from src.preprocessing.waveform import preprocess_waveform
from src.training.determinism import set_seed
from src.training.normalization import N_MELS, apply_bin_stats
from src.training.train_v2 import build_optimizer
from src.training.v2_config import load_v2_config
from src.training.v2_data import COUNT_FAMILIES
from src.tuning.cnn_search import _json_text, environment

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_CLIPS = 12404  # FoR train (docs/dataset_audit.md; F2)
WORKER_COUNTS = (1, 2, 4)
SAMPLE_RATE = 16000
SYNTHETIC_SECONDS = 2.0
SYNTHETIC_SEED = 2026
NOTES = (
    "GPU rows include the candidate's real optimizer step (Stage 6 measured forward/backward only); "
    "the peak can only be higher, so the chosen batch is at least as safe.",
    "FFmpegCodecBackend.round_trip runs `ffmpeg -version` on every call; its cost is included in the "
    "codec time.",
    "CPU throughput uses synthetic audio and synthetic noise/RIR resources; real RIR lengths change "
    "the reverberation cost.",
    "Selection-set scoring time is not estimated (GPU).",
    "The epoch estimate assumes the per-item time scales linearly with the number of workers; "
    "Colab's 2 vCPUs (often one physical core) may not give that.",
)


class UnpinnedTimingCodecBackend(FFmpegCodecBackend):
    """Timing only: records the ffmpeg version instead of verifying the V1b pin. Never for training."""

    observed_version: str | None = None

    def verify_pin(self) -> None:
        text = self.version_text()
        self.observed_version = text.splitlines()[0] if text else ""


def codec_backend_for(aug_config, *, unpinned_timing: bool):
    """(backend, "pinned" | "unpinned"); an unpinned ffmpeg is refused without the explicit flag."""
    if aug_config.ffmpeg_pin_complete:
        backend = FFmpegCodecBackend(aug_config.ffmpeg, aug_config.sample_rate)
        backend.verify_pin()
        return backend, "pinned"
    if not unpinned_timing:
        raise ValueError("ffmpeg is not pinned in configs/augmentation.yaml (V1b); codec timing "
                         "needs --codec-unpinned-timing, which marks the result unpinned")
    return UnpinnedTimingCodecBackend(aug_config.ffmpeg, aug_config.sample_rate), "unpinned"


def benchmark_settings(profile) -> dict[str, Any]:
    b = profile.benchmark
    return {"candidate_batch_sizes": [int(x) for x in b["candidate_batch_sizes"]],
            "max_vram_fraction": float(b["max_vram_fraction"]),
            "warmup_steps": int(b["warmup_steps"]), "measured_steps": int(b["measured_steps"])}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def gpu_rows(candidate: str, settings: dict, batch_sizes, bench: dict, device) -> list[dict]:
    """One row per batch size: forward + backward + optimizer step on synthetic input."""
    device = torch.device(device)
    shape = INPUT_SHAPES[INPUTS[candidate]]
    rows = []
    for batch in batch_sizes:
        model = optimizer = x = y = loss = None
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            set_seed(0)
            model = build_model(candidate).to(device).train()
            optimizer = build_optimizer(model, settings)
            loss_fn = nn.BCEWithLogitsLoss()
            generator = torch.Generator().manual_seed(0)
            x = torch.randn((int(batch), *shape), generator=generator).to(device)
            y = torch.randint(0, 2, (int(batch),), generator=generator).float().to(device)
            times = []
            for step in range(bench["warmup_steps"] + bench["measured_steps"]):
                optimizer.zero_grad(set_to_none=True)
                _sync(device)
                start = time.perf_counter()
                loss = loss_fn(model(x), y)
                loss.backward()
                optimizer.step()
                _sync(device)
                if step >= bench["warmup_steps"]:
                    times.append(time.perf_counter() - start)
            peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            rows.append({"candidate": candidate, "batch_size": int(batch), "status": "ok",
                         "mean_step_seconds": float(np.mean(times)), "peak_allocated_bytes": peak})
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            rows.append({"candidate": candidate, "batch_size": int(batch), "status": "oom",
                         "mean_step_seconds": None, "peak_allocated_bytes": None})
        finally:
            del model, optimizer, x, y, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return rows


def _stats(values) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    return {"n": int(a.size), "mean_seconds": float(a.mean()), "p95_seconds": float(np.quantile(a, 0.95))}


def synthetic_resources(seed: int = SYNTHETIC_SEED) -> AugmentationResources:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(3 * SAMPLE_RATE).astype(np.float32) * np.float32(0.1)
    t = np.arange(SAMPLE_RATE // 2) / SAMPLE_RATE
    rir = (rng.standard_normal(t.size) * np.exp(-t / 0.08)).astype(np.float32)
    rir[0] = 1.0
    return AugmentationResources(noise_waveforms=(noise,), rir_waveforms=(rir,))


def synthetic_clips(n: int = 8, seed: int = SYNTHETIC_SEED) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(int(SYNTHETIC_SECONDS * SAMPLE_RATE)) / SAMPLE_RATE
    return [(0.2 * np.sin(2 * np.pi * rng.uniform(120, 900) * t)
             + 0.02 * rng.standard_normal(t.size)).astype(np.float32) for _ in range(n)]


def throughput(augmenter: Augmenter, profile, items: int, *, max_draws: int | None = None) -> dict:
    """Per-family augment time and per-stage time over `items` items per family (synthetic audio)."""
    clips = synthetic_clips()
    mean, std = np.zeros(N_MELS, np.float32), np.ones(N_MELS, np.float32)
    families = {f: [] for f in COUNT_FAMILIES}
    f3_times, logmel_times = [], []
    max_draws = max_draws or 200 * items
    seed = 0
    while any(len(v) < items for v in families.values()):
        if seed >= max_draws:
            raise RuntimeError(f"throughput: {max_draws} draws did not give {items} items per family")
        clip = clips[seed % len(clips)]
        start = time.perf_counter()
        result = augmenter.augment(clip, split="train", label=("FAKE", "REAL")[seed % 2], seed=seed)
        augmented = time.perf_counter()
        seed += 1
        if len(families[result.family]) >= items:
            continue
        families[result.family].append(augmented - start)
        start = time.perf_counter()
        wave = preprocess_waveform(result.waveform, SAMPLE_RATE, profile).waveform
        f3 = time.perf_counter()
        apply_bin_stats(log_mel_spectrogram(wave, profile)[None], mean, std)
        done = time.perf_counter()
        f3_times.append(f3 - start)
        logmel_times.append(done - f3)
    family_stats = {f: _stats(v) for f, v in families.items()}
    p = float(augmenter.config.training["apply_probability"])
    share = {"clean": 1.0 - p, **{f: p / len(augmenter.config.families)
                                   for f in augmenter.config.families}}
    augment_mix = sum(share[f] * family_stats[f]["mean_seconds"] for f in COUNT_FAMILIES)
    f3_mean = float(np.mean(f3_times))
    logmel_mean = float(np.mean(logmel_times))
    per_item = {"waveform": augment_mix + f3_mean, "logmel": augment_mix + f3_mean + logmel_mean}
    return {
        "items_per_family": items, "draws": seed, "family_share": share,
        "families": family_stats,
        "stages": {"augment": _stats([t for v in families.values() for t in v]),
                   "f3": _stats(f3_times), "logmel_stats": _stats(logmel_times)},
        "per_item_mean_seconds": per_item,
        "estimated_epoch_seconds": {
            kind: {str(w): TRAIN_CLIPS * value / w for w in WORKER_COUNTS}
            for kind, value in per_item.items()},
        "train_clips": TRAIN_CLIPS,
    }


def run_benchmark(out, *, device: str, candidates=None, batch_sizes=None, throughput_items: int = 20,
                  codec_unpinned_timing: bool = False,
                  v2_config_path=REPO_ROOT / "configs" / "v2_candidates.yaml",
                  augmentation_config_path=REPO_ROOT / "configs" / "augmentation.yaml",
                  profile_path=REPO_ROOT / "configs" / "preprocessing.yaml",
                  codec_backend=None) -> dict[str, Any]:
    """Run both parts and write benchmark.json into a new folder; return the report.

    codec_backend replaces the configured backend (synthetic tests only); then codec_timing is
    "injected".
    """
    version = code_version()  # before creating outputs
    out = Path(out)
    check_outside_repo(out, "benchmark output")
    if out.exists():
        raise ValueError(f"benchmark output already exists: {out}")
    if device not in ("cuda", "cpu"):
        raise ValueError(f"device must be cuda or cpu, got {device!r}")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA benchmark requested but no CUDA GPU is available")
    if isinstance(throughput_items, bool) or not isinstance(throughput_items, int) or throughput_items < 0:
        raise ValueError(f"throughput_items must be an int >= 0, got {throughput_items!r}")
    config = load_v2_config(Path(v2_config_path))
    profile = load_profile(Path(profile_path))
    aug_config = load_augmentation_config(Path(augmentation_config_path))
    bench = benchmark_settings(profile)
    candidates = list(candidates or config.candidate_order)
    unknown = [c for c in candidates if c not in ARCHITECTURES]
    if unknown:
        raise ValueError(f"unknown candidates {unknown}; candidates are {ARCHITECTURES}")
    sizes = [int(b) for b in (batch_sizes or bench["candidate_batch_sizes"])]
    if not sizes or any(b < 1 for b in sizes):
        raise ValueError(f"batch sizes must be ints >= 1, got {sizes}")

    codec_timing, ffmpeg_observed = None, None
    augmenter = None
    if throughput_items > 0:
        if codec_backend is None:
            codec_backend, codec_timing = codec_backend_for(
                aug_config, unpinned_timing=codec_unpinned_timing)
        else:
            codec_timing = "injected"
        augmenter = Augmenter(aug_config, synthetic_resources(), codec_backend)

    rows, chosen, choice_errors = [], {}, {}
    total_vram = None
    if device == "cuda":
        total_vram = int(torch.cuda.get_device_properties(torch.device("cuda")).total_memory)
    for candidate in candidates:
        candidate_rows = gpu_rows(candidate, config.candidate(candidate), sizes, bench, device)
        rows.extend(candidate_rows)
        chosen[candidate] = None
        if device == "cuda":
            try:
                chosen[candidate] = choose_batch_size(candidate_rows, total_vram,
                                                      bench["max_vram_fraction"])
            except PreprocessingError as exc:
                choice_errors[candidate] = str(exc)

    table = None
    if augmenter is not None:
        table = throughput(augmenter, profile, throughput_items)
        ffmpeg_observed = getattr(augmenter.codec_backend, "observed_version", None)

    report = {
        "schema_version": SCHEMA_VERSION,
        "code_commit": version["git_commit"],
        "environment": {**environment(device), "platform": platform.platform(),
                        "cpu_count": str(os.cpu_count())},
        "device": device,
        "benchmark": {**bench, "candidate_batch_sizes": sizes, "optimizer_step": True},
        "total_vram_bytes": total_vram,
        "rows": rows,
        "chosen_batch_size": chosen,
        "choice_errors": choice_errors,
        "throughput": table,
        "codec_timing": codec_timing,
        "ffmpeg_version_observed": ffmpeg_observed,
        "profile_fingerprint": profile.fingerprint(),
        "v2_config_sha256": sha256_file(Path(v2_config_path)),
        "augmentation_config_sha256": sha256_file(Path(augmentation_config_path)),
        "notes": list(NOTES),
        "weights_saved": False,
        "data_read": False,
    }
    out.mkdir(parents=True)
    (out / "benchmark.json").write_text(_json_text(report), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path, help="New output folder outside the repo")
    parser.add_argument("--device", required=True, choices=("cuda", "cpu"))
    parser.add_argument("--candidates", nargs="+", choices=ARCHITECTURES)
    parser.add_argument("--batch-sizes", nargs="+", type=int)
    parser.add_argument("--throughput-items", type=int, default=20,
                        help="Items per augmentation family for the CPU part (0 skips it)")
    parser.add_argument("--codec-unpinned-timing", action="store_true",
                        help="Time the codec family with an unpinned ffmpeg (result marked unpinned)")
    args = parser.parse_args(argv)
    report = run_benchmark(args.out, device=args.device, candidates=args.candidates,
                           batch_sizes=args.batch_sizes, throughput_items=args.throughput_items,
                           codec_unpinned_timing=args.codec_unpinned_timing)
    print(json.dumps({"chosen_batch_size": report["chosen_batch_size"],
                      "codec_timing": report["codec_timing"], "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
