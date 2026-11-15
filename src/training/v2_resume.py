"""Per-epoch resume state of one model v2 run (Execution Plan v3.0 §9 V3; D42, D49, D61).

After every completed epoch (scored on S, best/patience updated) train_v2 writes the full training
state into `<run>.partial/resume/`, so a Colab session that ends mid-run loses at most the epoch in
progress. No pickle and no torch.save (D42 spirit): tensors in safetensors, the rest in strict JSON.

- `state.safetensors`: `model.*` (current weights), `best.*` (the best epoch's weights, absent while
  no epoch improved), `opt.<param_index>.<key>` (the flattened Adam state: exp_avg, exp_avg_sq, step),
  `rng.torch_cpu`, `rng.torch_cuda` (only on CUDA), `rng.sampler` (the EpochSampler generator).
- `state.json` (sorted keys, no NaN, canonical text): STATE_KEYS below, including the SHA-256 of
  `state.safetensors`.

Atomic write: the new state is written completely into `resume.tmp/`; then `resume/` is renamed to
`resume.old/`, `resume.tmp/` to `resume/`, and `resume.old/` is removed. A crash at any point leaves
one complete state: `resume/`, or (between the two renames) `resume.old/`, which recover() renames
back. `resume.tmp/` alone is always incomplete and is discarded.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

SCHEMA_VERSION = 1
RESUME_DIR = "resume"
TENSORS_NAME = "state.safetensors"
JSON_NAME = "state.json"
TMP_SUFFIX = ".tmp"
OLD_SUFFIX = ".old"
STATE_KEYS = frozenset({
    "schema_version", "candidate", "run_seed", "last_epoch", "global_step", "best_selection_m",
    "best_epoch", "patience_counter", "history", "numpy_random_state", "python_random_state",
    "determinism", "wandb_run_id", "provenance", "settings", "peak_gpu_memory_bytes",
    "state_safetensors_sha256",
})
SETTINGS_KEYS = frozenset({"max_epochs", "patience", "batch_size", "steps_per_epoch", "n_train",
                           "device_type"})
REQUIRED_RNG = ("rng.torch_cpu", "rng.sampler")
CUDA_RNG = "rng.torch_cuda"


def json_text(data: dict) -> str:
    """The canonical state.json text: sorted keys, indent 2, NaN and infinity refused."""
    return json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _siblings(resume_dir: Path) -> tuple[Path, Path]:
    return (resume_dir.with_name(resume_dir.name + TMP_SUFFIX),
            resume_dir.with_name(resume_dir.name + OLD_SUFFIX))


def state_folders(resume_dir) -> tuple[Path, Path, Path]:
    """resume/, resume.tmp/ and resume.old/ for one run folder's resume path."""
    resume_dir = Path(resume_dir)
    return (resume_dir, *_siblings(resume_dir))


def recover(resume_dir) -> None:
    """Finish an interrupted swap: resume.old/ back to resume/ if resume/ is missing; drop resume.tmp/."""
    resume_dir = Path(resume_dir)
    tmp, old = _siblings(resume_dir)
    if not resume_dir.exists() and old.is_dir():
        old.rename(resume_dir)
    if resume_dir.is_dir() and old.exists():
        shutil.rmtree(old)
    if tmp.exists():
        shutil.rmtree(tmp)


def has_state(resume_dir) -> bool:
    """True if a (possibly invalid) state was ever completed: resume/ or resume.old/ exists."""
    resume_dir = Path(resume_dir)
    return resume_dir.exists() or _siblings(resume_dir)[1].exists()


def remove_state(resume_dir) -> None:
    """Delete resume/, resume.tmp/ and resume.old/ (only once the run folder is complete and verified)."""
    for folder in state_folders(resume_dir):
        if folder.exists():
            shutil.rmtree(folder)


# --- RNG and optimizer state ---------------------------------------------------------------------

def numpy_state_json() -> dict:
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return {"name": str(name), "keys": [int(k) for k in keys], "pos": int(pos),
            "has_gauss": int(has_gauss), "cached_gaussian": float(cached)}


def set_numpy_state(data: dict) -> None:
    np.random.set_state((data["name"], np.asarray(data["keys"], dtype=np.uint32), int(data["pos"]),
                         int(data["has_gauss"]), float(data["cached_gaussian"])))


def python_state_json() -> dict:
    version, internal, gauss_next = random.getstate()
    return {"version": int(version), "internal": [int(v) for v in internal],
            "gauss_next": gauss_next}


def set_python_state(data: dict) -> None:
    random.setstate((int(data["version"]), tuple(int(v) for v in data["internal"]),
                     data["gauss_next"]))


def flatten_optimizer(optimizer: torch.optim.Optimizer) -> dict[str, torch.Tensor]:
    """The optimizer's per-parameter state as `opt.<param_index>.<key>` CPU tensors."""
    out = {}
    for index, entry in optimizer.state_dict()["state"].items():
        for key, value in entry.items():
            if not torch.is_tensor(value):
                raise ValueError(f"optimizer state {index}.{key} is not a tensor ({type(value).__name__})")
            out[f"opt.{index}.{key}"] = _cpu(value)
    return out


def restore_optimizer(optimizer: torch.optim.Optimizer, tensors: dict[str, torch.Tensor]) -> None:
    """Load the flattened `opt.*` state into a freshly built optimizer of the same parameters."""
    fresh = optimizer.state_dict()
    n_params = sum(len(group["params"]) for group in fresh["param_groups"])
    amsgrad = any(group.get("amsgrad") for group in fresh["param_groups"])
    expected = {"exp_avg", "exp_avg_sq", "step"} | ({"max_exp_avg_sq"} if amsgrad else set())
    state: dict[int, dict] = {}
    for name, value in tensors.items():
        parts = name.split(".")
        if len(parts) != 3 or parts[0] != "opt" or not parts[1].isdigit():
            raise ValueError(f"resume state: bad optimizer key {name!r}")
        index = int(parts[1])
        if index >= n_params:
            raise ValueError(f"resume state: optimizer index {index} >= {n_params} parameters")
        state.setdefault(index, {})[parts[2]] = value
    for index, entry in state.items():
        if set(entry) != expected:
            raise ValueError(f"resume state: optimizer entry {index} holds {sorted(entry)}, "
                             f"expected {sorted(expected)}")
    optimizer.load_state_dict({"state": state, "param_groups": fresh["param_groups"]})


def _cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to("cpu", copy=True).contiguous()


# --- write / read --------------------------------------------------------------------------------

def write_state(resume_dir, meta: dict, tensors: dict[str, torch.Tensor]) -> Path:
    """Write one complete state atomically (module docstring); meta without the tensor hash."""
    resume_dir = Path(resume_dir)
    tmp, old = _siblings(resume_dir)
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    data = save_tensors({k: _cpu(v) for k, v in tensors.items()})
    digest = hashlib.sha256(data).hexdigest()
    meta = {**meta, "schema_version": SCHEMA_VERSION, "state_safetensors_sha256": digest}
    _check_keys(meta)
    text = json_text(meta)
    (tmp / TENSORS_NAME).write_bytes(data)
    (tmp / JSON_NAME).write_text(text, encoding="utf-8")
    if resume_dir.exists():
        if old.exists():
            shutil.rmtree(old)
        resume_dir.rename(old)
    tmp.rename(resume_dir)
    if old.exists():
        shutil.rmtree(old)
    return resume_dir


def _no_constant(name: str):
    raise ValueError(f"state.json holds {name}; NaN and infinity are refused")


def _check_keys(meta: dict) -> None:
    missing, unknown = sorted(STATE_KEYS - set(meta)), sorted(set(meta) - STATE_KEYS)
    if missing or unknown:
        raise ValueError(f"state.json keys: missing {missing}, unknown {unknown}")
    settings = meta["settings"]
    if not isinstance(settings, dict) or set(settings) != SETTINGS_KEYS:
        raise ValueError(f"state.json settings must hold exactly {sorted(SETTINGS_KEYS)}")


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def read_state(resume_dir, *, candidate: str, run_seed: int,
               provenance: dict) -> tuple[dict, dict[str, torch.Tensor]]:
    """Recover, read and verify one state; return (state.json dict, tensors). Raise ValueError.

    Checks: both files present and nothing else; canonical strict JSON without NaN; schema version;
    the tensor file's SHA-256; candidate, run seed and provenance equal to the caller's; counters
    consistent; the RNG tensors present. Model and optimizer keys are checked when loaded.
    """
    resume_dir = Path(resume_dir)
    recover(resume_dir)
    if not resume_dir.is_dir():
        raise ValueError(f"{resume_dir}: no resume state")
    names = sorted(p.name for p in resume_dir.iterdir())
    if names != sorted([TENSORS_NAME, JSON_NAME]):
        raise ValueError(f"{resume_dir}: a resume state holds exactly {JSON_NAME} and "
                         f"{TENSORS_NAME}, found {names}")
    text = (resume_dir / JSON_NAME).read_text(encoding="utf-8")
    try:
        meta = json.loads(text, parse_constant=_no_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{resume_dir / JSON_NAME} is not valid JSON: {exc}") from exc
    if not isinstance(meta, dict):
        raise ValueError(f"{resume_dir / JSON_NAME} must hold a JSON object")
    _check_keys(meta)
    if json_text(meta) != text:
        raise ValueError(f"{resume_dir / JSON_NAME} is not in canonical form (sorted keys, indent 2)")
    if meta["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"resume state schema_version {meta['schema_version']!r}, expected "
                         f"{SCHEMA_VERSION}")
    data = (resume_dir / TENSORS_NAME).read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != meta["state_safetensors_sha256"]:
        raise ValueError(f"{resume_dir / TENSORS_NAME}: SHA-256 {actual} differs from {JSON_NAME} "
                         f"{meta['state_safetensors_sha256']}")
    for key, expected in (("candidate", candidate), ("run_seed", run_seed),
                          ("provenance", provenance)):
        if meta[key] != expected:
            raise ValueError(f"resume state {key} differs from the current run: recorded "
                             f"{meta[key]!r}, current {expected!r}")
    epoch, bad, step = meta["last_epoch"], meta["patience_counter"], meta["global_step"]
    settings = meta["settings"]
    if not all(_is_int(v) for v in (epoch, bad, step)) or epoch < 1 or bad < 0:
        raise ValueError("resume state: last_epoch, patience_counter and global_step must be ints "
                         "(last_epoch >= 1)")
    if step != epoch * settings["steps_per_epoch"] or len(meta["history"]) != epoch:
        raise ValueError(f"resume state: global_step {step} and {len(meta['history'])} history "
                         f"records do not match epoch {epoch}")
    if (meta["best_epoch"] is None) != (meta["best_selection_m"] is None):
        raise ValueError("resume state: best_epoch and best_selection_m must both be set or null")
    if meta["best_selection_m"] is not None and not math.isfinite(meta["best_selection_m"]):
        raise ValueError("resume state: best_selection_m must be finite")
    try:
        tensors = load_tensors(data)
    except Exception as exc:  # safetensors raises its own SafetensorError
        raise ValueError(f"{resume_dir / TENSORS_NAME} is not a readable safetensors file: "
                         f"{exc}") from exc
    rng = [*REQUIRED_RNG, *([CUDA_RNG] if settings["device_type"] == "cuda" else [])]
    missing = [k for k in rng if k not in tensors]
    if missing or (CUDA_RNG in tensors and settings["device_type"] != "cuda"):
        raise ValueError(f"resume state: RNG tensors missing {missing} or unexpected for device "
                         f"{settings['device_type']!r}")
    has_best = any(k.startswith("best.") for k in tensors)
    if has_best != (meta["best_epoch"] is not None):
        raise ValueError("resume state: best.* tensors must exist exactly when best_epoch is set")
    known = ("model.", "best.", "opt.")
    unknown = [k for k in tensors if not k.startswith(known) and k not in (*REQUIRED_RNG, CUDA_RNG)]
    if unknown:
        raise ValueError(f"resume state: unknown tensors {unknown[:5]}")
    return meta, tensors


def split(tensors: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """The tensors under `prefix.` with the prefix removed."""
    cut = len(prefix) + 1
    return {k[cut:]: v for k, v in tensors.items() if k.startswith(prefix + ".")}


__all__ = [
    "JSON_NAME", "RESUME_DIR", "SCHEMA_VERSION", "STATE_KEYS", "TENSORS_NAME", "flatten_optimizer",
    "has_state", "json_text", "numpy_state_json", "python_state_json", "read_state", "recover",
    "remove_state", "restore_optimizer", "set_numpy_state", "set_python_state", "split",
    "state_folders", "write_state",
]
