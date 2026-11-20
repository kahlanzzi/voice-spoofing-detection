"""Model v2 rehearsal on the training runtime, before the search (V2b-4; D42, D49, D58, D64).

Run once on the Colab T4 (runtime 2026.10) from the same fixed commit as the search, into its own
new folder. It reads the same inputs as the search and changes no config or frozen setting: every
epoch limit below is a constant of this module, passed to train_v2, never written to configs/.

Order: require_versions() (D64) before any work; clean worktree (code_version); the folder checks
(never a search folder; a new folder; the marker file written first); the inputs, verified as
run_search verifies them, plus the V2b-3 raw-cache pin and the V1b resource check of the CLI; then

(a) clean check: check_clean_transform on a fixed, seeded set of CLEAN_CHECK_PER_CLASS train indices
    per class (raw -> F3 -> log-Mel equals the Stage 8 train cache byte for byte);
(b) timing: one epoch per candidate, seed 0, with the configured batch sizes (seconds per epoch,
    train items per second, peak GPU memory);
(c) determinism: crnn, seed 0, DETERMINISM_EPOCHS epochs, run twice in this process; whether
    best_selection_m and the SHA-256 of the best weights are equal is reported as observed (D42: not
    promised on GPU);
(d) forced resume: the same crnn run with a resume folder; an exception is injected during epoch 2,
    after its training pass and before its state is written (as a disconnect), so the state of epoch
    1 is on disk; train_v2 is then called again on that folder and the result is compared with (c);
(e) cut line: worst case of the 5 search runs at max_epochs with the measured epoch times, against
    the selection date (Fri 27 Nov 23:59 Tunis); if they do not fit, the plan drops aasist first
    (Execution Plan v3.0 §7), which is a team decision, not something this code does.

rehearsal_report.json is written once (exclusive create) and printed. No W&B: the rehearsal runs are
not search runs. Only train and val-A (through S) are read (D9).
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from src.augmentation import Augmenter, load_augmentation_config
from src.augmentation.codec import FFmpegCodecBackend
from src.augmentation.version_guard import require_versions
from src.features.logmel_cache import load_logmel_cache
from src.features.raw_train_cache import check_outside_repo, load_raw_train_cache
from src.models.v2_checkpoint import INPUTS
from src.preprocessing.profile import load_profile
from src.preprocessing.validation import F2_MANIFEST_SHA256, code_version, load_f2_manifest, sha256_file
from src.training.normalization import fit_bin_stats
from src.training.selection_set import load_selection_set
from src.training.train_v2 import train_v2
from src.training.v2_config import load_v2_config
from src.training.v2_data import check_clean_transform
from src.training.v2_resume import RESUME_DIR, read_state
from src.tuning import v2_search as vs
from src.tuning.cnn_search import _check_paths, _json_text, environment

SCHEMA_VERSION = 1
CLEAN_CHECK_PER_CLASS = 16  # 32 train indices in all
CLEAN_CHECK_SEED = 2026
TIMING_EPOCHS = 1
TIMING_SEED = 0
DETERMINISM_CANDIDATE = "crnn"
DETERMINISM_SEED = 0
DETERMINISM_EPOCHS = 2
CRASH_EPOCH = 2  # the exception is raised during this epoch, before its state is written
FORCED_RESUME_DIR = "forced_resume"
SELECTION_DEADLINE_UTC = datetime(2026, 11, 27, 22, 59, tzinfo=timezone.utc)  # 23:59 Tunis (UTC+1)
SEARCH_RUNS = 5  # each candidate once with seed 0, then the winner's seeds 1 and 2 (D49)
DROPPED_FIRST = "aasist"  # Execution Plan v3.0 §7 cut line


class InjectedCrash(RuntimeError):
    """Raised by the rehearsal's logger to stop a run as a disconnect would."""


class _Logger:
    """No W&B. Optionally raises InjectedCrash when epoch `crash_epoch` is logged (before its state
    is written by train_v2)."""

    run_id = None

    def __init__(self, crash_epoch: int | None = None) -> None:
        self.crash_epoch = crash_epoch
        self.epochs: list[int] = []

    def log_epoch(self, record: dict) -> None:
        if record["epoch"] == self.crash_epoch:
            raise InjectedCrash(f"rehearsal: injected crash during epoch {record['epoch']}")
        self.epochs.append(record["epoch"])

    def log_summary(self, summary: dict) -> None:
        pass

    def finish(self) -> None:
        pass


def clean_check_indices(labels: np.ndarray, per_class: int = CLEAN_CHECK_PER_CLASS,
                        seed: int = CLEAN_CHECK_SEED) -> list[int]:
    """`per_class` train indices of each label, drawn without replacement with a fixed seed, sorted."""
    rng = np.random.default_rng(seed)
    chosen = []
    for code in (0, 1):
        pool = np.flatnonzero(np.asarray(labels) == code)
        if len(pool) < per_class:
            raise ValueError(f"clean check needs {per_class} clips of label {code}, found {len(pool)}")
        chosen += rng.choice(pool, size=per_class, replace=False).tolist()
    return sorted(int(i) for i in chosen)


def state_dict_sha256(state_dict: dict) -> str:
    """SHA-256 over the sorted parameter names, dtypes, shapes and bytes of a state_dict."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(f"{tensor.dtype}{tuple(tensor.shape)}".encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _without_seconds(history) -> list[dict]:
    return [{k: v for k, v in row.items() if not k.endswith("_seconds")} for row in history]


def _summary(result) -> dict[str, Any]:
    return {"best_epoch": result.best_epoch, "best_selection_m": result.best_selection_m,
            "epochs_run": result.epochs_run, "weights_sha256": state_dict_sha256(result.state_dict),
            "peak_gpu_memory_bytes": result.peak_gpu_memory_bytes}


def cut_line(epoch_seconds: dict[str, float], max_epochs: int, order, now: datetime,
             deadline: datetime = SELECTION_DEADLINE_UTC) -> dict[str, Any]:
    """Worst case of the search at max_epochs: every candidate once, then the slowest one's 2 seeds."""
    def worst(candidates) -> float:
        slowest = max(epoch_seconds[c] for c in candidates)
        return (sum(epoch_seconds[c] for c in candidates) + 2 * slowest) * max_epochs / 3600

    available = (deadline - now).total_seconds() / 3600
    full = worst(order)
    reduced = worst([c for c in order if c != DROPPED_FIRST])
    return {
        "runs": SEARCH_RUNS, "max_epochs": max_epochs,
        "epoch_seconds": {c: epoch_seconds[c] for c in order},
        "worst_case_hours": full,
        f"worst_case_hours_without_{DROPPED_FIRST}": reduced,
        "now_utc": now.isoformat(timespec="seconds"),
        "deadline_utc": deadline.isoformat(timespec="seconds"),
        "hours_until_deadline": available,
        "fits": full <= available,
        f"fits_without_{DROPPED_FIRST}": reduced <= available,
        "rule": (f"if the {SEARCH_RUNS} runs do not fit, drop {DROPPED_FIRST} first (Execution Plan "
                 f"v3.0 §7); that is a team decision. Wall-clock hours, no Colab session gaps."),
    }


def _check_out(out_dir: Path) -> None:
    check_outside_repo(out_dir, "rehearsal output")
    if out_dir.exists():
        if not out_dir.is_dir():
            raise ValueError(f"{out_dir} exists and is not a folder")
        if (out_dir / vs.RUNS_DIR).exists() or any((out_dir / r).exists() for r in vs.REPORTS):
            raise ValueError(f"{out_dir} is a search folder ({vs.RUNS_DIR}/ or a search report "
                             f"present); the rehearsal writes only to its own new folder")
        if any(out_dir.iterdir()):
            raise ValueError(f"{out_dir} is not empty; the rehearsal writes to a new folder")


def rehearse(*, raw_cache_dir, stage8_train_logmel_dir, stage8_val_a_logmel_dir, selection_set_dir,
             resources_dir, out_dir, device: str, num_workers: int,
             profile_path=None, v2_config_path=None, augmentation_config_path=None,
             manifest_path=None, version: Callable[[str], str] = importlib.metadata.version,
             now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> dict[str, Any]:
    """The rehearsal (module docstring); return the report written to rehearsal_report.json.

    The profile, config and manifest paths default to the committed files (the CLI passes none);
    `version` is the package-version lookup given to require_versions.
    """
    versions = require_versions(version=version)  # D64: before any other work
    commit = code_version()["git_commit"]
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError(f"num_workers must be an int >= 0, got {num_workers!r}")
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"device {device} requested but CUDA is not available")
    out_dir = Path(out_dir)
    _check_out(out_dir)
    profile_path = Path(profile_path or vs.PROFILE_PATH)
    v2_config_path = Path(v2_config_path or vs.V2_CONFIG_PATH)
    augmentation_config_path = Path(augmentation_config_path or vs.AUGMENTATION_CONFIG_PATH)
    manifest_path = Path(manifest_path or vs.F2_MANIFEST_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / vs.REHEARSAL_MARKER).write_text(_json_text({
        "kind": "v2_rehearsal", "code_commit": commit,
        "note": "rehearsal output; never a search folder (src.tuning.v2_search refuses it)",
    }), encoding="utf-8")

    # Inputs: the CLI checks, then the run_search checks (same functions, same order of meaning).
    raw_npz_sha256 = vs.check_raw_cache_pin(raw_cache_dir)
    resources, resources_meta = vs.load_v2_resources(resources_dir)
    config = load_v2_config(v2_config_path)
    batch_sizes = config.require_batch_sizes()
    aug_config = load_augmentation_config(augmentation_config_path)
    if not aug_config.corpus_pins_complete or not aug_config.ffmpeg_pin_complete:
        raise ValueError("configs/augmentation.yaml corpus and ffmpeg pins must be complete (V1b)")
    codec = FFmpegCodecBackend(aug_config.ffmpeg, aug_config.sample_rate)
    codec.verify_pin()
    augmenter = Augmenter(aug_config, resources, codec)
    profile = load_profile(profile_path)
    manifest = load_f2_manifest(manifest_path)
    raw = load_raw_train_cache(raw_cache_dir, expected_f2_manifest_sha256=F2_MANIFEST_SHA256)
    if raw.meta["n_all_zero_clips"] > 0:
        raise ValueError(f"raw train cache holds {raw.meta['n_all_zero_clips']} all-zero clip(s): "
                         f"decision D62 needed before training (docs/v2_training.md)")
    _check_paths(manifest, vs.TRAIN_SPLIT, raw.paths, raw.labels)
    groups = manifest.set_index("path").loc[raw.paths.tolist(), "group"].astype(str).to_numpy()
    if not np.array_equal(groups, raw.groups.astype(object)):
        raise ValueError("raw train cache groups differ from the F2 manifest")
    profile_config_sha256 = sha256_file(profile_path)
    stage8_train = load_logmel_cache(Path(stage8_train_logmel_dir), vs.TRAIN_SPLIT, profile,
                                     expected_f2_manifest_sha256=F2_MANIFEST_SHA256)
    if stage8_train[3]["profile_config_sha256"] != profile_config_sha256:
        raise ValueError("Stage 8 train cache was built from another profile config")

    # (a) clean check, before the train features are released
    clean = check_clean_transform(raw, stage8_train, clean_check_indices(raw.labels), profile)
    labels_checked = raw.labels[clean["indices"]]
    clean.update({"n_real": int((labels_checked == 0).sum()), "n_fake": int((labels_checked == 1).sum()),
                  "seed": CLEAN_CHECK_SEED, "passed": True})
    mean, std = fit_bin_stats(stage8_train[0])
    stats_sha256 = stage8_train[3]["npz_sha256"]
    del stage8_train

    val_a = load_logmel_cache(Path(stage8_val_a_logmel_dir), vs.VAL_A_SPLIT, profile,
                              expected_f2_manifest_sha256=F2_MANIFEST_SHA256)
    if val_a[3]["profile_config_sha256"] != profile_config_sha256:
        raise ValueError("Stage 8 val-A cache was built from another profile config")
    s_cache: dict[str, Any] = {}

    def selection_for(form: str):
        if form not in s_cache:
            s_cache.clear()  # one form in memory at a time, as run_search
            s_cache[form] = load_selection_set(selection_set_dir, profile, form=form,
                                               f2_manifest=manifest, clean_reference_logmel=val_a)
        return s_cache[form]

    s_manifest = selection_for("logmel").manifest
    aug_sha256, f2_sha256 = sha256_file(augmentation_config_path), sha256_file(manifest_path)
    for key, value in (("augmentation_config_sha256", aug_sha256), ("f2_manifest_sha256", f2_sha256),
                       ("profile_config_sha256", profile_config_sha256)):
        if s_manifest[key] != value:
            raise ValueError(f"selection set {key} {s_manifest[key]} differs from the current {value}")
    inputs = {
        "code_commit": commit, "v2_config_sha256": sha256_file(v2_config_path),
        "augmentation_config_sha256": aug_sha256, "profile_fingerprint": profile.fingerprint(),
        "profile_config_sha256": profile_config_sha256, "f2_manifest_sha256": f2_sha256,
        "raw_train_cache_npz_sha256": raw_npz_sha256,
        "selection_set_sha256": selection_for("logmel").sha256,
        "val_a_cache_npz_sha256": val_a[3]["npz_sha256"], "train_stats_npz_sha256": stats_sha256,
        "resource_cache_npz_sha256": resources_meta["npz_sha256"],
    }
    patience = config.settings["patience"]

    def train(candidate: str, seed: int, epochs: int, logger: _Logger, **resume):
        logmel = INPUTS[candidate] == "logmel"
        return train_v2(candidate, config=config, run_seed=seed, raw_cache=raw,
                        selection=selection_for(vs.S_FORMS[INPUTS[candidate]]), augmenter=augmenter,
                        profile=profile, mean=mean if logmel else None, std=std if logmel else None,
                        max_epochs=epochs, patience=patience, batch_size=batch_sizes[candidate],
                        num_workers=num_workers, logger=logger, device=device, **resume)

    # (b) one epoch per candidate
    timing = {}
    for candidate in config.candidate_order:
        result = train(candidate, TIMING_SEED, TIMING_EPOCHS, _Logger())
        row = result.history[0]
        timing[candidate] = {
            "batch_size": batch_sizes[candidate], "epoch_seconds": row["epoch_seconds"],
            "train_seconds": row["train_seconds"], "selection_seconds": row["selection_seconds"],
            "items_per_second": len(raw) / row["train_seconds"],
            "peak_gpu_memory_bytes": result.peak_gpu_memory_bytes, "selection_m": row["selection_m"],
        }
        print(f"v2_rehearsal: {candidate} epoch {row['epoch_seconds']:.1f} s "
              f"({timing[candidate]['items_per_second']:.1f} items/s)")

    # (c) the same short crnn run twice
    runs = [train(DETERMINISM_CANDIDATE, DETERMINISM_SEED, DETERMINISM_EPOCHS, _Logger())
            for _ in range(2)]
    first, second = (_summary(r) for r in runs)
    determinism = {
        "candidate": DETERMINISM_CANDIDATE, "seed": DETERMINISM_SEED, "epochs": DETERMINISM_EPOCHS,
        "run_1": first, "run_2": second,
        "best_selection_m_equal": first["best_selection_m"] == second["best_selection_m"],
        "weights_sha256_equal": first["weights_sha256"] == second["weights_sha256"],
        "history_equal_without_seconds": _without_seconds(runs[0].history) == _without_seconds(runs[1].history),
        "determinism": runs[0].determinism,
        "note": "observed only; D42: bit-reproducibility is not promised on GPU",
    }

    # (d) forced resume
    resume_dir = out_dir / FORCED_RESUME_DIR / RESUME_DIR
    provenance = {**inputs, "rehearsal": True}
    try:
        train(DETERMINISM_CANDIDATE, DETERMINISM_SEED, DETERMINISM_EPOCHS, _Logger(CRASH_EPOCH),
              resume_dir=resume_dir, provenance=provenance)
    except InjectedCrash:
        pass
    else:
        raise RuntimeError("rehearsal: the injected crash did not happen")
    saved, _ = read_state(resume_dir, candidate=DETERMINISM_CANDIDATE, run_seed=DETERMINISM_SEED,
                          provenance=provenance)
    resumed_logger = _Logger()
    resumed = train(DETERMINISM_CANDIDATE, DETERMINISM_SEED, DETERMINISM_EPOCHS, resumed_logger,
                    resume_dir=resume_dir, provenance=provenance)
    after = _summary(resumed)
    forced_resume = {
        "crash": f"during epoch {CRASH_EPOCH}, after its training pass, before its state was written",
        "state_last_epoch_at_crash": saved["last_epoch"],
        "epochs_trained_after_resume": resumed_logger.epochs,
        "resumed": after,
        "best_selection_m_equal_to_run_1": after["best_selection_m"] == first["best_selection_m"],
        "weights_sha256_equal_to_run_1": after["weights_sha256"] == first["weights_sha256"],
        "history_equal_to_run_1_without_seconds":
            _without_seconds(resumed.history) == _without_seconds(runs[0].history),
        "note": "observed only (D42); on CPU a resumed run equals an uninterrupted one (tested)",
    }

    report = {
        "schema_version": SCHEMA_VERSION, "kind": "v2_rehearsal",
        "require_versions": versions, "environment": environment(device),
        "device": str(device), "num_workers": num_workers, "batch_sizes": batch_sizes,
        "inputs": inputs, "n_train": len(raw),
        "clean_transform": clean, "epoch_timing": timing, "determinism": determinism,
        "forced_resume": forced_resume,
        "cut_line": cut_line({c: t["epoch_seconds"] for c, t in timing.items()},
                             config.settings["max_epochs"], config.candidate_order, now()),
    }
    text = _json_text(report)
    with open(out_dir / vs.REHEARSAL_REPORT, "x", encoding="utf-8") as stream:
        stream.write(text)
    line = report["cut_line"]
    print(f"v2_rehearsal: cut line: worst case {line['worst_case_hours']:.1f} h for {SEARCH_RUNS} "
          f"runs x {line['max_epochs']} epochs; {line['hours_until_deadline']:.1f} h until "
          f"{line['deadline_utc']}; fits: {line['fits']}; without {DROPPED_FIRST}: "
          f"{line[f'worst_case_hours_without_{DROPPED_FIRST}']:.1f} h, "
          f"fits: {line[f'fits_without_{DROPPED_FIRST}']}")
    return report
