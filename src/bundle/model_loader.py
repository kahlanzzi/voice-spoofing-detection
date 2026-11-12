"""Load the CNN of a verified ModelBundle (D59 (5); SRS v4.0 FR-29; SDD v2.0 §3.5).

load_bundle (src.bundle.bundle, torch-free) verifies the folder, every hash and the cross-checks.
load_bundle_model then loads the checkpoint with the reviewed non-pickle loader,
cnn_checkpoint.load_checkpoint, passing the manifest's combined checkpoint_sha256 as expected_sha256:
load_checkpoint hashes the bytes it loads, so a file replaced after load_bundle's check is refused
(closes check-then-use for the checkpoint). Finally the loaded metadata version must equal the
manifest model version.

Used by the API (Stage 17) and Stage 13. Uses torch; must not import src.evaluation or src.tuning
(research code).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.bundle.bundle import LoadedBundle, load_bundle
from src.models import cnn_checkpoint


@dataclass(frozen=True)
class BundleModel:
    """A verified bundle and its CNN (eval mode), normalization statistics and checkpoint metadata."""

    bundle: LoadedBundle
    model: object  # src.models.cnn.CNN
    mean: np.ndarray
    std: np.ndarray
    metadata: dict


def load_bundle_model(directory) -> BundleModel:
    """Verify a bundle, then load its CNN from bytes checked against the manifest anchor."""
    bundle = load_bundle(Path(directory))
    model, mean, std, metadata = cnn_checkpoint.load_checkpoint(
        bundle.checkpoint_dir, expected_sha256=bundle.checkpoint_sha256
    )
    manifest_version = bundle.manifest.components["checkpoint"].version
    if metadata["version"] != manifest_version:
        raise ValueError(
            f"checkpoint: version mismatch (manifest {manifest_version!r}, loaded {metadata['version']!r})"
        )
    return BundleModel(bundle=bundle, model=model, mean=mean, std=std, metadata=metadata)
