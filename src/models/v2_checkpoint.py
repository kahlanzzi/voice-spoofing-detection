"""Non-pickle model v2 checkpoint for the three D49 candidates (D59 (7), D61 (6)).

Same layout and the same combined-hash rule as src/models/cnn_checkpoint.py, so the D59 bundle
code (src/bundle/bundle.py, src/bundle/f4.py) can hold a v2 checkpoint unchanged:
- weights.safetensors: the model state_dict, plus "norm.mean" / "norm.std" (float32 (64,), std > 0)
  if and only if the model takes the log-Mel (cnn-aug, crnn; D61 (5)); aasist takes the waveform
  and has no norm tensors;
- checkpoint.json: the strict v2 metadata (METADATA_KEYS exactly), sorted keys, no NaN.

The file reading, the per-file hashing and the combined hash are the cnn_checkpoint functions
themselves (imported, not copied), so v2_checkpoint_sha256 cannot drift from checkpoint_sha256.
cnn_checkpoint (F4's loader) is not modified. Safetensors only: this module never calls torch.load.

Layering: Stage 13, F5 and the API load checkpoints through this module, so it must not import
src.evaluation or src.tuning (research code).
"""
from __future__ import annotations

import json
import math
import re
import shutil
from functools import lru_cache
from numbers import Real
from pathlib import Path

import torch
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

from src.models.aasist import NB_SAMP, build_aasist
from src.models.aasist import Model as AASIST
from src.models.cnn import CNN, build_cnn
from src.models.cnn import INPUT_SHAPE as LOGMEL_SHAPE
from src.models.cnn_checkpoint import (
    FILES,
    JSON_NAME,
    NORM_KEYS,
    WEIGHTS_NAME,
    _norm_stat,
    _read_files,
    _sha256_lines,
)
from src.models.crnn import CRNN, build_crnn

SCHEMA_VERSION = 1
MAX_EPOCHS = 30  # D49
CNN_AUG_BLOCKS = 4  # D49 (C1): D30's selected configuration
ARCHITECTURES = ("cnn-aug", "crnn", "aasist")  # D49 candidate order
BUILDERS = {
    "cnn-aug": lambda: build_cnn(CNN_AUG_BLOCKS),
    "crnn": build_crnn,
    "aasist": build_aasist,
}
INPUTS = {"cnn-aug": "logmel", "crnn": "logmel", "aasist": "waveform"}
INPUT_SHAPES = {"logmel": list(LOGMEL_SHAPE), "waveform": [NB_SAMP]}

METADATA_KEYS = frozenset({
    "schema_version", "architecture", "version", "input", "input_shape", "seed", "best_epoch",
    "selection_m", "code_commit", "environment", "profile_fingerprint", "f2_manifest_sha256",
    "augmentation_config_sha256", "selection_set_sha256", "train_stats_npz_sha256",
    "determinism", "trainable_parameters",
})
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")

__all__ = [
    "ARCHITECTURES", "BUILDERS", "FILES", "JSON_NAME", "METADATA_KEYS", "WEIGHTS_NAME",
    "build_model", "load_v2_checkpoint", "save_v2_checkpoint", "trainable_parameters",
    "v2_checkpoint_sha256", "validate_v2_metadata",
]


def trainable_parameters(model: torch.nn.Module) -> int:
    """Number of trainable (requires_grad) parameter values; D49's tie rule puts fewer first."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(architecture: str) -> torch.nn.Module:
    """A freshly initialized candidate model for one of ARCHITECTURES; anything else raises."""
    if architecture not in BUILDERS:
        raise ValueError(f"architecture must be one of {ARCHITECTURES}, got {architecture!r}")
    return BUILDERS[architecture]()


@lru_cache(maxsize=None)
def _expected_parameters(architecture: str) -> int:
    return trainable_parameters(build_model(architecture))


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(f"{JSON_NAME}: {message}")


def validate_v2_metadata(metadata) -> dict:
    """Check v2 metadata against the strict schema; return a plain-dict copy, or raise ValueError."""
    _require(isinstance(metadata, dict), f"metadata must be a JSON object, got {type(metadata).__name__}")
    missing = sorted(METADATA_KEYS - set(metadata))
    unknown = sorted(set(metadata) - METADATA_KEYS, key=str)
    _require(not missing and not unknown, f"missing keys {missing}, unknown keys {unknown}")
    m = metadata
    _require(_is_int(m["schema_version"]) and m["schema_version"] == SCHEMA_VERSION,
             f"unsupported schema_version {m['schema_version']!r}")
    arch = m["architecture"]
    _require(isinstance(arch, str) and arch in ARCHITECTURES,
             f"architecture must be one of {ARCHITECTURES}, got {arch!r}")
    _require(isinstance(m["version"], str) and m["version"].strip() != "",
             f"version must be a non-empty string, got {m['version']!r}")
    _require(m["input"] == INPUTS[arch],
             f"input must be {INPUTS[arch]!r} for {arch}, got {m['input']!r}")
    shape = INPUT_SHAPES[INPUTS[arch]]
    _require(isinstance(m["input_shape"], list) and all(_is_int(d) for d in m["input_shape"])
             and m["input_shape"] == shape,
             f"input_shape must be {shape} for {arch}, got {m['input_shape']!r}")
    _require(_is_int(m["seed"]) and 0 <= m["seed"] < 2**32,
             f"seed must be an int in [0, 2**32), got {m['seed']!r}")
    _require(_is_int(m["best_epoch"]) and 1 <= m["best_epoch"] <= MAX_EPOCHS,
             f"best_epoch must be an int in [1, {MAX_EPOCHS}], got {m['best_epoch']!r}")
    _require(_is_number(m["selection_m"]) and 0 <= m["selection_m"] <= 1,
             f"selection_m must be a finite number in [0, 1], got {m['selection_m']!r}")
    _require(isinstance(m["code_commit"], str) and _HEX40.fullmatch(m["code_commit"]) is not None,
             f"code_commit must be 40 lowercase hex, got {m['code_commit']!r}")
    for key in ("profile_fingerprint", "f2_manifest_sha256", "augmentation_config_sha256",
                "selection_set_sha256"):
        _require(isinstance(m[key], str) and _HEX64.fullmatch(m[key]) is not None,
                 f"{key} must be 64 lowercase hex, got {m[key]!r}")
    stats = m["train_stats_npz_sha256"]
    if INPUTS[arch] == "logmel":
        _require(isinstance(stats, str) and _HEX64.fullmatch(stats) is not None,
                 f"train_stats_npz_sha256 must be 64 lowercase hex for {arch}, got {stats!r}")
    else:
        _require(stats is None, f"train_stats_npz_sha256 must be null for {arch}, got {stats!r}")
    env = m["environment"]
    _require(isinstance(env, dict) and env
             and all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()),
             f"environment must be a non-empty object of strings, got {env!r}")
    det = m["determinism"]
    _require(isinstance(det, dict) and all(isinstance(k, str) for k in det),
             f"determinism must be a JSON object, got {det!r}")
    _require(det.get("seed") == m["seed"] and _is_int(det.get("seed")),
             f"determinism.seed must equal seed {m['seed']!r}, got {det.get('seed')!r}")
    expected_params = _expected_parameters(arch)
    _require(_is_int(m["trainable_parameters"]) and m["trainable_parameters"] == expected_params,
             f"trainable_parameters must be {expected_params} for {arch}, "
             f"got {m['trainable_parameters']!r}")
    # A JSON round trip gives a plain copy and refuses NaN and non-JSON values anywhere.
    try:
        return json.loads(json.dumps(m, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{JSON_NAME}: metadata is not strict JSON: {exc}") from exc


def _architecture_of(model) -> str:
    """The candidate name of a built model; any other model raises."""
    if isinstance(model, CNN) and model.n_blocks == CNN_AUG_BLOCKS:
        return "cnn-aug"
    if isinstance(model, CRNN):
        return "crnn"
    if isinstance(model, AASIST):
        return "aasist"
    raise ValueError(f"model must come from {ARCHITECTURES} builders, got {type(model).__name__}")


def _encode(model, metadata, mean, std) -> tuple[bytes, bytes]:
    """Both files' bytes, every check done before anything is written."""
    meta = validate_v2_metadata(metadata)
    arch = _architecture_of(model)
    if meta["architecture"] != arch:
        raise ValueError(f"metadata architecture {meta['architecture']!r} != model {arch!r}")
    if trainable_parameters(model) != meta["trainable_parameters"]:
        raise ValueError(f"model has {trainable_parameters(model)} trainable parameters, "
                         f"metadata says {meta['trainable_parameters']}")
    tensors = {k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()}
    if INPUTS[arch] == "logmel":
        if mean is None or std is None:
            raise ValueError(f"{arch} takes the log-Mel: mean and std are required (D61 (5))")
        tensors["norm.mean"] = _norm_stat("mean", mean)
        tensors["norm.std"] = _norm_stat("std", std)
    elif mean is not None or std is not None:
        raise ValueError(f"{arch} takes the waveform: mean and std must be None")
    weights = save_tensors(tensors)
    text = json.dumps(meta, indent=2, sort_keys=True, allow_nan=False) + "\n"
    return text.encode("utf-8"), weights


def save_v2_checkpoint(model, metadata, out_dir, *, mean=None, std=None) -> Path:
    """Write weights.safetensors and checkpoint.json into a new or empty folder; return the folder.

    A non-empty folder (or a file) is refused, so a checkpoint is never overwritten. If writing
    fails, the files written are removed, and the folder too if this call created it.
    """
    out_dir = Path(out_dir)
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise ValueError(f"{out_dir} must be a new or empty folder")
    json_bytes, weights_bytes = _encode(model, metadata, mean, std)
    created = not out_dir.exists()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    try:
        for name, data in ((WEIGHTS_NAME, weights_bytes), (JSON_NAME, json_bytes)):
            path = out_dir / name
            written.append(path)
            path.write_bytes(data)
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        if created:
            shutil.rmtree(out_dir, ignore_errors=True)
        raise
    return out_dir


def v2_checkpoint_sha256(out_dir) -> str:
    """The combined checkpoint SHA-256, by the same rule as cnn_checkpoint.checkpoint_sha256 (D59)."""
    return _sha256_lines(_read_files(Path(out_dir)))


def load_v2_checkpoint(out_dir, *, expected_sha256: str | None = None):
    """Verify and load a v2 checkpoint; return (model, mean | None, std | None, metadata).

    The model is in eval mode; mean and std are float32 (64,) arrays for log-Mel models and None
    for aasist. Refuses extra or missing files, invalid metadata, norm tensors on aasist or none on
    a log-Mel model, unexpected, missing or mistyped tensors, and a state_dict that does not match
    the architecture's builder exactly (strict=True). With expected_sha256, the hash of the bytes
    read here must match it. Safetensors only, never torch.load.
    """
    out_dir = Path(out_dir)
    data = _read_files(out_dir)
    if expected_sha256 is not None:
        actual = _sha256_lines(data)
        if actual != expected_sha256:
            raise ValueError(f"checkpoint_sha256 mismatch: expected {expected_sha256}, got {actual}")
    try:
        meta = json.loads(data[JSON_NAME].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{JSON_NAME} is not valid JSON: {exc}") from exc
    meta = validate_v2_metadata(meta)
    try:
        tensors = load_tensors(data[WEIGHTS_NAME])
    except Exception as exc:  # safetensors raises its own SafetensorError on a corrupt file
        raise ValueError(f"{WEIGHTS_NAME} is not a readable safetensors file: {exc}") from exc

    arch = meta["architecture"]
    present = [k for k in NORM_KEYS if k in tensors]
    mean = std = None
    if INPUTS[arch] == "logmel":
        if len(present) != len(NORM_KEYS):
            raise ValueError(f"{WEIGHTS_NAME}: {arch} needs tensors {list(NORM_KEYS)}, found {present}")
        mean = _norm_stat("mean", tensors.pop("norm.mean").numpy()).numpy()
        std = _norm_stat("std", tensors.pop("norm.std").numpy()).numpy()
    elif present:
        raise ValueError(f"{WEIGHTS_NAME}: {arch} takes the waveform and must not hold {present}")

    model = build_model(arch)
    expected = model.state_dict()
    missing = sorted(set(expected) - set(tensors))
    unknown = sorted(set(tensors) - set(expected))
    if missing or unknown:
        raise ValueError(f"{WEIGHTS_NAME} does not match {arch}: missing {missing}, unknown {unknown}")
    for key, ref in expected.items():
        t = tensors[key]
        if t.dtype != ref.dtype or t.shape != ref.shape:
            raise ValueError(f"{WEIGHTS_NAME}[{key}] must be {ref.dtype} {tuple(ref.shape)}, "
                             f"got {t.dtype} {tuple(t.shape)}")
        if t.is_floating_point() and not torch.isfinite(t).all():
            raise ValueError(f"{WEIGHTS_NAME}[{key}] contains NaN or infinite values")
    try:
        model.load_state_dict(tensors, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"{WEIGHTS_NAME} does not load into {arch}: {exc}") from exc
    model.eval()
    return model, mean, std, meta
