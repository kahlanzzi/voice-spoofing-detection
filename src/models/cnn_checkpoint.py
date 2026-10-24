"""Non-pickle CNN checkpoint: safetensors weights + strict JSON metadata (D42 "Checkpoint").

A checkpoint is one folder holding exactly two files:
- weights.safetensors: the model state_dict plus the D42 normalization statistics as the tensors
  "norm.mean" and "norm.std" (float32, shape (64,)), written with safetensors.
- checkpoint.json: the metadata (METADATA_KEYS exactly), sorted keys, no NaN.

Why no pickle: torch.save / torch.load use pickle, and a pickled checkpoint can execute code when it
is loaded (docs/model_bundle.md, "Unsafe deserialization"). safetensors stores only a JSON header
and raw tensor bytes, so loading never runs code. This module never calls torch.load.

checkpoint_sha256 is the one value the F4 record stores, built like the baseline artifact hash
(src/models/baseline.py). load_checkpoint reads each file once, so the bytes that are hashed are the
bytes that are loaded.

Layering: Stage 12, Stage 13 and the API load checkpoints through this module, so it must not
import src.evaluation or src.tuning (research code).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from numbers import Real
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

from src.models.cnn import CHANNELS, CNN, DROPOUT, INPUT_SHAPE, N_BLOCKS, build_cnn
from src.training.normalization import N_MELS

SCHEMA_VERSION = 1
WEIGHTS_NAME = "weights.safetensors"
JSON_NAME = "checkpoint.json"
FILES = (JSON_NAME, WEIGHTS_NAME)  # fixed order for checkpoint_sha256
NORM_KEYS = ("norm.mean", "norm.std")
MAX_EPOCHS = 30  # D30

METADATA_KEYS = frozenset({
    "schema_version", "version", "n_blocks", "channels", "dropout", "input_shape",
    "learning_rate", "seed", "best_epoch", "val_a_eer", "code_commit", "environment",
    "profile_fingerprint", "f2_manifest_sha256", "train_cache_npz_sha256",
    "val_a_cache_npz_sha256", "determinism",
})
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(f"{JSON_NAME}: {message}")


def validate_metadata(metadata) -> dict:
    """Check the metadata against the strict schema; return a plain-dict copy, or raise ValueError."""
    _require(isinstance(metadata, dict), f"metadata must be a JSON object, got {type(metadata).__name__}")
    missing = sorted(METADATA_KEYS - set(metadata))
    unknown = sorted(set(metadata) - METADATA_KEYS, key=str)
    _require(not missing and not unknown, f"missing keys {missing}, unknown keys {unknown}")
    m = metadata
    _require(_is_int(m["schema_version"]) and m["schema_version"] == SCHEMA_VERSION,
             f"unsupported schema_version {m['schema_version']!r}")
    _require(isinstance(m["version"], str) and m["version"].strip() != "",
             f"version must be a non-empty string, got {m['version']!r}")
    _require(_is_int(m["n_blocks"]) and m["n_blocks"] in N_BLOCKS,
             f"n_blocks must be one of {N_BLOCKS}, got {m['n_blocks']!r}")
    expected_channels = list(CHANNELS[:m["n_blocks"]])
    _require(isinstance(m["channels"], list) and all(_is_int(c) for c in m["channels"])
             and m["channels"] == expected_channels,
             f"channels must be {expected_channels} for {m['n_blocks']} blocks, got {m['channels']!r}")
    _require(_is_number(m["dropout"]) and m["dropout"] == DROPOUT,
             f"dropout must be {DROPOUT}, got {m['dropout']!r}")
    _require(isinstance(m["input_shape"], list) and all(_is_int(d) for d in m["input_shape"])
             and m["input_shape"] == list(INPUT_SHAPE),
             f"input_shape must be {list(INPUT_SHAPE)}, got {m['input_shape']!r}")
    _require(_is_number(m["learning_rate"]) and m["learning_rate"] > 0,
             f"learning_rate must be a finite number > 0, got {m['learning_rate']!r}")
    _require(_is_int(m["seed"]) and 0 <= m["seed"] < 2**32,
             f"seed must be an int in [0, 2**32), got {m['seed']!r}")
    _require(_is_int(m["best_epoch"]) and 1 <= m["best_epoch"] <= MAX_EPOCHS,
             f"best_epoch must be an int in [1, {MAX_EPOCHS}], got {m['best_epoch']!r}")
    _require(_is_number(m["val_a_eer"]) and 0 <= m["val_a_eer"] <= 1,
             f"val_a_eer must be a number in [0, 1], got {m['val_a_eer']!r}")
    _require(isinstance(m["code_commit"], str) and _HEX40.fullmatch(m["code_commit"]) is not None,
             f"code_commit must be 40 lowercase hex, got {m['code_commit']!r}")
    for key in ("profile_fingerprint", "f2_manifest_sha256", "train_cache_npz_sha256",
                "val_a_cache_npz_sha256"):
        _require(isinstance(m[key], str) and _HEX64.fullmatch(m[key]) is not None,
                 f"{key} must be 64 lowercase hex, got {m[key]!r}")
    env = m["environment"]
    _require(isinstance(env, dict) and env
             and all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()),
             f"environment must be a non-empty object of strings, got {env!r}")
    det = m["determinism"]
    _require(isinstance(det, dict) and all(isinstance(k, str) for k in det),
             f"determinism must be a JSON object, got {det!r}")
    _require(det.get("seed") == m["seed"] and _is_int(det.get("seed")),
             f"determinism.seed must equal seed {m['seed']!r}, got {det.get('seed')!r}")
    # A JSON round trip gives a plain copy and refuses NaN and non-JSON values anywhere.
    try:
        return json.loads(json.dumps(m, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{JSON_NAME}: metadata is not strict JSON: {exc}") from exc


def _norm_stat(name: str, value) -> torch.Tensor:
    a = np.asarray(value)
    if a.dtype != np.float32 or a.shape != (N_MELS,) or not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite float32 with shape ({N_MELS},), got {a.dtype} {a.shape}")
    if name == "std" and (a <= 0).any():
        raise ValueError("std must be > 0 for every Mel bin")
    return torch.from_numpy(a.copy())


def _encode(model: CNN, mean, std, metadata) -> tuple[bytes, bytes]:
    """Both files' bytes, every check done before anything is written."""
    if not isinstance(model, CNN):
        raise ValueError(f"model must be a CNN from build_cnn, got {type(model).__name__}")
    meta = validate_metadata(metadata)
    if meta["n_blocks"] != model.n_blocks:
        raise ValueError(f"metadata n_blocks {meta['n_blocks']} != model n_blocks {model.n_blocks}")
    tensors = {k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()}
    tensors["norm.mean"] = _norm_stat("mean", mean)
    tensors["norm.std"] = _norm_stat("std", std)
    weights = save_tensors(tensors)
    text = json.dumps(meta, indent=2, sort_keys=True, allow_nan=False) + "\n"
    return text.encode("utf-8"), weights


def save_checkpoint(model: CNN, mean, std, metadata, out_dir) -> Path:
    """Write weights.safetensors and checkpoint.json into a new or empty folder; return the folder.

    A non-empty folder (or a file) is refused, so a checkpoint is never overwritten. If writing
    fails, the files written are removed, and the folder too if this call created it.
    """
    out_dir = Path(out_dir)
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise ValueError(f"{out_dir} must be a new or empty folder")
    json_bytes, weights_bytes = _encode(model, mean, std, metadata)
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


def _read_files(out_dir: Path) -> dict[str, bytes]:
    """The bytes of exactly the two checkpoint files (regular files, no symlinks), each read once."""
    if not out_dir.is_dir():
        raise ValueError(f"{out_dir} is not a checkpoint folder")
    names = sorted(p.name for p in out_dir.iterdir())
    if names != sorted(FILES):
        raise ValueError(f"{out_dir} must hold exactly {sorted(FILES)}, found {names}")
    data = {}
    for name in FILES:
        path = out_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{path} must be a regular file")
        data[name] = path.read_bytes()
    return data


def _sha256_lines(data: dict[str, bytes]) -> str:
    lines = "".join(f"{name}:{hashlib.sha256(data[name]).hexdigest()}\n" for name in FILES)
    return hashlib.sha256(lines.encode("ascii")).hexdigest()


def checkpoint_sha256(out_dir) -> str:
    """One SHA-256 for the checkpoint, for the F4 record (as baseline.artifact_sha256, D36).

    It hashes the ASCII text "checkpoint.json:<sha256>\\nweights.safetensors:<sha256>\\n", so a change
    to either file changes it.
    """
    return _sha256_lines(_read_files(Path(out_dir)))


def load_checkpoint(out_dir, *, expected_sha256: str | None = None):
    """Verify and load a checkpoint; return (model, mean, std, metadata), the model in eval mode.

    Refuses extra or missing files, invalid metadata, unexpected, missing or mistyped tensors, and a
    state_dict that does not match build_cnn(n_blocks) exactly (strict=True). Tensors are read with
    safetensors only, never torch.load or pickle.

    A file cannot hold its own hash, so a changed weight value is detected only against an external
    anchor: with expected_sha256 (the recorded checkpoint_sha256), the hash of the bytes read here
    must match, else ValueError. Hash and load use the same bytes.
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
    meta = validate_metadata(meta)
    try:
        tensors = load_tensors(data[WEIGHTS_NAME])
    except Exception as exc:  # safetensors raises its own SafetensorError on a corrupt file
        raise ValueError(f"{WEIGHTS_NAME} is not a readable safetensors file: {exc}") from exc

    model = build_cnn(meta["n_blocks"])
    expected = model.state_dict()
    for key in NORM_KEYS:
        if key not in tensors:
            raise ValueError(f"{WEIGHTS_NAME}: missing tensor {key!r}")
    mean = tensors.pop("norm.mean")
    std = tensors.pop("norm.std")
    missing = sorted(set(expected) - set(tensors))
    unknown = sorted(set(tensors) - set(expected))
    if missing or unknown:
        raise ValueError(f"{WEIGHTS_NAME} does not match a {meta['n_blocks']}-block CNN: "
                         f"missing {missing}, unknown {unknown}")
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
        raise ValueError(f"{WEIGHTS_NAME} does not load into the CNN: {exc}") from exc
    model.eval()

    mean_np = _norm_stat("mean", mean.numpy()).numpy()
    std_np = _norm_stat("std", std.numpy()).numpy()
    return model, mean_np, std_np, meta
