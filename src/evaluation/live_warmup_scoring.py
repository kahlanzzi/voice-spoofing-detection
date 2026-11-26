"""D52 val-B inference from a pinned plan and verified frozen bundle."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from src.evaluation.live_warmup_plan import build_call_plan
from src.evaluation.live_warmup_report import report_from_plan
from src.preprocessing.validation import read_verified_audio
from src.preprocessing.waveform import preprocess_waveform


def frozen_logit(loaded, waveform) -> float:
    """Score one already-preprocessed window, without applying calibration."""
    import torch
    from src.preprocessing.features import log_mel_spectrogram
    from src.training.normalization import apply_bin_stats

    if loaded.model.training or any(p.device.type != "cpu" for p in loaded.model.parameters()):
        raise ValueError("D52 requires the loaded CPU model in eval mode")
    if loaded.architecture == "aasist":
        features = np.asarray(waveform, dtype=np.float32)[None]
    elif loaded.architecture in {"cnn", "cnn-aug", "crnn"}:
        features = log_mel_spectrogram(waveform, loaded.bundle.profile)[None]
        features = apply_bin_stats(features, loaded.mean, loaded.std)
    else:
        raise ValueError("unsupported frozen architecture")
    with torch.inference_mode():
        scores = loaded.model(torch.from_numpy(features)).detach().cpu().numpy()
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size != 1 or not np.isfinite(values[0]):
        raise ValueError("model must return one finite raw logit")
    return float(values[0])


def score_call_plan(plan_path: Path, manifest_path: Path, audio_root: Path,
                    bundle_dir: Path, *, expected_plan_sha256: str,
                    expected_bundle_sha256: str, code_commit: str) -> dict:
    """Validate membership before audio; filter with D17 and calibrate exactly once.

    This function does not authorize execution, create a claim or write files.
    The caller must supply the reviewed clean code SHA and approved bundle pin.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", code_commit):
        raise ValueError("code_commit must be a full lowercase Git SHA")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_bundle_sha256):
        raise ValueError("expected_bundle_sha256 must be a lowercase SHA-256")
    plan = build_call_plan(manifest_path)
    members = {p for group in plan["plans"].values() for call in group["calls"] for p in call}
    # The shared report validator checks the exported bytes, schema and ordering.
    report_from_plan(plan_path, manifest_path, dict.fromkeys(members),
                     expected_plan_sha256=expected_plan_sha256, threshold=0.5, half_width=0.0)
    from src.bundle.model_loader import load_bundle_model
    loaded = load_bundle_model(bundle_dir)
    if loaded.bundle.manifest_sha256 != expected_bundle_sha256:
        raise ValueError("bundle manifest SHA-256 differs from the approved pin")
    profile, policy = loaded.bundle.profile, loaded.bundle.policy
    minimum = float(profile.silence["external_min_speech_fraction"])
    if not np.isfinite(minimum) or not 0 <= minimum <= 1:
        raise ValueError("invalid D17 speech fraction")
    probabilities, slots = {}, []
    for row in plan["clips"]:
        if row["path"] not in members:
            continue
        waveform, rate = read_verified_audio(audio_root, row)
        if waveform.ndim != 1 or len(waveform) != 2 * rate:
            raise ValueError("D52 slots require exactly two seconds of mono audio")
        prepared = preprocess_waveform(waveform, rate, profile)
        fraction = float(prepared.speech_fraction)
        if not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("invalid observed speech fraction")
        kept = fraction >= minimum
        raw, probability = None, None
        if kept:
            raw = frozen_logit(loaded, prepared.waveform)
            calibrated = np.asarray(policy.calibration.transform([raw]), dtype=np.float64).reshape(-1)
            if calibrated.size != 1 or not np.isfinite(calibrated[0]) or not 0 <= calibrated[0] <= 1:
                raise ValueError("calibration must return one finite probability")
            probability = float(calibrated[0])
        probabilities[row["path"]] = probability
        slots.append(dict(path=row["path"], sample_sha256=row["hash"],
                          speech_fraction=fraction, kept=kept, raw_logit=raw, p=probability))
    result = report_from_plan(plan_path, manifest_path, probabilities,
                              expected_plan_sha256=expected_plan_sha256,
                              threshold=policy.threshold, half_width=policy.half_width)
    result["provenance"] = dict(code_commit=code_commit,
        bundle_manifest_sha256=expected_bundle_sha256, architecture=loaded.architecture,
        component_versions={key: value.version for key, value in loaded.bundle.manifest.components.items()},
        d17_min_speech_fraction=minimum)
    result["slots"] = slots
    return result
