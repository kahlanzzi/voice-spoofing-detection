"""Load the model of a verified ModelBundle (D59 (5), D59 (7), D61 (6); SRS v4.0 FR-29; SDD v2.0 §3.5).

load_bundle (src.bundle.bundle, torch-free) verifies the folder, every hash and the cross-checks.
load_bundle_model then loads the checkpoint with a reviewed non-pickle loader, passing the
manifest's combined checkpoint_sha256 as expected_sha256: the loader hashes the bytes it loads, so a
file replaced after load_bundle's check is refused (closes check-then-use for the checkpoint).
Finally the loaded metadata version must equal the manifest model version.

Dispatch on the checkpoint format (D61 (6)). F4 (model v1) and F5 (model v2) bundles share the
layout and the hash rule (D59 (7)); only checkpoint.json differs. The v2 schema has an
"architecture" key (cnn-aug, crnn or aasist, D49); the v1 (D42) schema has none:
- v1 -> cnn_checkpoint.load_checkpoint, exactly as for F4; architecture "cnn";
- v2 -> v2_checkpoint.load_v2_checkpoint; the loaded architecture must be the one used to
  dispatch, the model must be on CPU in eval mode, and mean/std are float32 (64,) for the log-Mel
  models (cnn-aug, crnn) and None for aasist (waveform input).
The checkpoint.json bytes read for the dispatch are hashed against the manifest's per-file value
before they are parsed, and the chosen loader re-hashes everything it loads.

Used by the API (Stage 17) and Stage 13. Uses torch; must not import src.evaluation or src.tuning
(research code).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.bundle.bundle import CHECKPOINT_JSON, LoadedBundle, load_bundle
from src.models import cnn_checkpoint, v2_checkpoint

V1_ARCHITECTURE = "cnn"  # the D42 model v1 (F4)


@dataclass(frozen=True)
class BundleModel:
    """A verified bundle and its model (eval mode, CPU), normalization statistics and metadata.

    architecture is "cnn" for a v1 (D42) checkpoint, else the v2 candidate name (D49). mean and std
    are float32 (64,) arrays for log-Mel models and None for aasist only.
    """

    bundle: LoadedBundle
    model: object  # src.models.cnn.CNN, src.models.crnn.CRNN or src.models.aasist.Model
    mean: np.ndarray | None
    std: np.ndarray | None
    metadata: dict
    architecture: str = V1_ARCHITECTURE


def _checkpoint_format(bundle: LoadedBundle) -> str:
    """V1_ARCHITECTURE for a D42 checkpoint, else the declared v2 architecture (checked bytes only)."""
    data = (bundle.checkpoint_dir / CHECKPOINT_JSON).read_bytes()
    recorded = bundle.manifest.components["checkpoint"].files[CHECKPOINT_JSON]
    actual = hashlib.sha256(data).hexdigest()
    if actual != recorded:
        raise ValueError(
            f"checkpoint: SHA-256 mismatch for {CHECKPOINT_JSON} (manifest {recorded}, file {actual})"
        )
    try:
        meta = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"checkpoint: {CHECKPOINT_JSON} is not valid JSON: {exc}") from None
    if not isinstance(meta, dict):
        raise ValueError(f"checkpoint: {CHECKPOINT_JSON} must be a JSON object, got {type(meta).__name__}")
    if "architecture" not in meta:
        return V1_ARCHITECTURE
    architecture = meta["architecture"]
    if architecture not in v2_checkpoint.ARCHITECTURES:
        raise ValueError(
            f"checkpoint: architecture must be one of {v2_checkpoint.ARCHITECTURES}, got {architecture!r}"
        )
    return architecture


def _check_v2(architecture: str, model, mean, std, metadata: dict) -> None:
    """The loaded v2 model is the dispatched one, ready to score on CPU, with the right statistics."""
    if metadata.get("architecture") != architecture:
        raise ValueError(
            f"checkpoint: architecture mismatch (dispatched {architecture!r}, "
            f"loaded {metadata.get('architecture')!r})"
        )
    if getattr(model, "training", None) is not False:
        raise ValueError(f"checkpoint: {architecture} model must be in eval mode")
    parameters = list(model.parameters())
    if not parameters or any(p.device.type != "cpu" for p in parameters):
        raise ValueError(f"checkpoint: {architecture} model must be on CPU")
    if v2_checkpoint.INPUTS[architecture] == "logmel":
        for name, value in (("mean", mean), ("std", std)):
            if not (isinstance(value, np.ndarray) and value.dtype == np.float32 and value.shape == (64,)):
                raise ValueError(f"checkpoint: {architecture} needs {name} as float32 (64,), got {value!r}")
    elif mean is not None or std is not None:
        raise ValueError(f"checkpoint: {architecture} takes the waveform; mean and std must be None")


def load_bundle_model(directory) -> BundleModel:
    """Verify a bundle, then load its model from bytes checked against the manifest anchor."""
    bundle = load_bundle(Path(directory))
    architecture = _checkpoint_format(bundle)
    if architecture == V1_ARCHITECTURE:
        model, mean, std, metadata = cnn_checkpoint.load_checkpoint(
            bundle.checkpoint_dir, expected_sha256=bundle.checkpoint_sha256
        )
    else:
        model, mean, std, metadata = v2_checkpoint.load_v2_checkpoint(
            bundle.checkpoint_dir, expected_sha256=bundle.checkpoint_sha256
        )
    manifest_version = bundle.manifest.components["checkpoint"].version
    if metadata["version"] != manifest_version:
        raise ValueError(
            f"checkpoint: version mismatch (manifest {manifest_version!r}, loaded {metadata['version']!r})"
        )
    if architecture != V1_ARCHITECTURE:
        _check_v2(architecture, model, mean, std, metadata)
    return BundleModel(
        bundle=bundle, model=model, mean=mean, std=std, metadata=metadata, architecture=architecture
    )
