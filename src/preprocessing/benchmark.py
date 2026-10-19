"""Stage 6 compute-only benchmark; no optimizer, fitting, checkpoint or metric."""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from .profile import PreprocessingError, load_profile
from .validation import code_version, load_f2_manifest, read_verified_audio, sha256_file
from .features import extract_features


def choose_batch_size(rows: list[dict[str, Any]], total_vram_bytes: int, fraction: float) -> int:
    """Choose the largest successful batch whose peak allocation stays within the VRAM budget."""
    if total_vram_bytes <= 0 or not 0 < fraction <= 1:
        raise PreprocessingError("VRAM size and fraction must be positive")
    budget = total_vram_bytes * fraction
    eligible = [
        int(row["batch_size"]) for row in rows
        if row.get("status") == "ok" and float(row["peak_allocated_bytes"]) <= budget
    ]
    if not eligible:
        raise PreprocessingError("No candidate batch size fits the approved VRAM budget")
    return max(eligible)


def _gpu_dry_run(profile, device: str) -> dict[str, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise PreprocessingError(
            "The Colab benchmark requires PyTorch; record and approve the installed version before F3"
        ) from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise PreprocessingError("CUDA benchmark requested but no CUDA GPU is available")
    actual_device = torch.device(device)

    # Benchmark-only worst-case four-block CNN. No optimizer is constructed and
    # no weights are updated or saved; Stage 8 owns the real architecture/training.
    layers = []
    channels = 1
    for out_channels in (16, 32, 64, 128):
        layers.extend((
            nn.Conv2d(channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
        ))
        channels = out_channels
    model = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 1))
    model.to(actual_device)
    config = profile.benchmark
    rows = []
    for batch in config["candidate_batch_sizes"]:
        try:
            if actual_device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(actual_device)
            x = torch.randn(
                int(batch), 1, int(profile.features["n_mels"]),
                profile.expected_log_mel_frames, device=actual_device,
            )
            times = []
            for step in range(int(config["warmup_steps"]) + int(config["measured_steps"])):
                model.zero_grad(set_to_none=True)
                if actual_device.type == "cuda":
                    torch.cuda.synchronize(actual_device)
                start = time.perf_counter()
                loss = model(x).square().mean()
                loss.backward()
                if actual_device.type == "cuda":
                    torch.cuda.synchronize(actual_device)
                if step >= int(config["warmup_steps"]):
                    times.append(time.perf_counter() - start)
            peak = (
                int(torch.cuda.max_memory_allocated(actual_device))
                if actual_device.type == "cuda" else 0
            )
            rows.append({
                "batch_size": int(batch), "status": "ok",
                "mean_step_seconds": float(np.mean(times)),
                "peak_allocated_bytes": peak,
            })
            del x, loss
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            rows.append({
                "batch_size": int(batch), "status": "oom",
                "mean_step_seconds": None, "peak_allocated_bytes": None,
            })
            if actual_device.type == "cuda":
                torch.cuda.empty_cache()

    if actual_device.type == "cuda":
        props = torch.cuda.get_device_properties(actual_device)
        total_vram = int(props.total_memory)
        chosen = choose_batch_size(rows, total_vram, float(config["max_vram_fraction"]))
        gpu_name = props.name
    else:
        total_vram, chosen, gpu_name = 0, None, "CPU dry-run"
    return {
        "torch_version": torch.__version__, "device": str(actual_device),
        "gpu_name": gpu_name, "total_vram_bytes": total_vram,
        "results": rows, "chosen_batch_size": chosen,
        "weights_updated": False, "checkpoint_written": False,
    }


def run(root: Path, manifest_path: Path, profile_path: Path, out: Path, device: str) -> dict[str, Any]:
    version = code_version()
    profile = load_profile(profile_path)
    manifest = load_f2_manifest(manifest_path)
    out = Path(out)
    if out.exists():
        raise PreprocessingError(f"Benchmark output already exists: {out}")

    sample_count = int(profile.benchmark["preprocessing_samples"])
    sample = manifest[manifest.split == "train"].sort_values(["hash", "path"], kind="stable").head(sample_count)
    started = time.perf_counter()
    for _, row in sample.iterrows():
        waveform, sample_rate = read_verified_audio(root, row)
        extract_features(waveform, sample_rate, profile)
    preprocess_seconds = time.perf_counter() - started
    gpu = _gpu_dry_run(profile, device)
    report = {
        "stage": 6, "mode": profile.benchmark["mode"], "code_version": version,
        "profile_version": profile.profile_version, "profile_fingerprint": profile.fingerprint(),
        "profile_config_sha256": sha256_file(Path(profile_path)),
        "f2_manifest_sha256": sha256_file(Path(manifest_path)),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "preprocessing": {
            "split": "train", "rows": len(sample), "seconds": preprocess_seconds,
            "clips_per_second": len(sample) / preprocess_seconds,
        },
        "compute": gpu,
        "audio_scope": "FoR train subset only; DEEP-VOICE not accessed",
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "benchmark.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    md = [
        "# Stage 6 compute benchmark", "",
        f"- Code commit: `{version['git_commit']}`", f"- Profile: `{profile.profile_version}`",
        f"- FoR train clips preprocessed: {len(sample)}",
        f"- Preprocessing throughput: {report['preprocessing']['clips_per_second']:.3f} clips/s",
        f"- Device: {gpu['gpu_name']}", f"- PyTorch: `{gpu['torch_version']}`",
        f"- Selected future batch size: {gpu['chosen_batch_size']}",
        "- Optimizer steps: 0", "- Checkpoints written: 0",
        "- DEEP-VOICE accessed: no", "",
        "Detailed candidate measurements are in `benchmark.json`.",
    ]
    (out / "benchmark.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    report = run(args.root, args.manifest, args.profile, args.out, args.device)
    print(json.dumps({
        "chosen_batch_size": report["compute"]["chosen_batch_size"],
        "weights_updated": report["compute"]["weights_updated"],
        "out": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
