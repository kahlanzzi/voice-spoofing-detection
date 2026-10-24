"""D30 CNN model-selection search (Stages 8-9; D9, D11, D30, D40, D42): train and val-A only.

Reads the verified train and val-A log-Mel caches (src/features/logmel_cache.py), trains the 4
configurations of configs/cnn.yaml with the search seed, selects one on val-A EER, trains it with the
other final seeds and writes into a new folder:
- checkpoints/seed_<s>/: the selected configuration's checkpoint for every final seed
  (src/models/cnn_checkpoint.py, non-pickle).
- search_report.json: the 4 search runs with their per-epoch history.
- selection_report.json: the selected configuration, whether the D30 tie rule decided it, the seed
  results and the bundle seed (select_bundle_seed, D11).
- run_report.json: provenance binding the checkpoints to their inputs (code commit, environment,
  F3 profile, F2 manifest, cnn.yaml, cache hashes) and the checkpoint hashes.

Selection (D30): the lowest best val-A EER wins; ties go to fewer blocks, then the higher learning
rate (sort key (eer, n_blocks, -learning_rate)). The seed-0 search run of the selected configuration
is reused as seed 0, so only seeds 1 and 2 are trained again.

Only train and val-A are read: val-B is not loaded or scored in Stages 8-9 (D42), and there is no
test or external path. Every run is logged through its own allow-list RunLogger (D40). Nothing is
written unless every step succeeds.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.evaluation.seeds import select_bundle_seed
from src.features.logmel_cache import LABEL_CODES, load_logmel_cache
from src.models.cnn import CHANNELS, DROPOUT, INPUT_SHAPE, build_cnn
from src.models.cnn_checkpoint import SCHEMA_VERSION as CHECKPOINT_SCHEMA_VERSION
from src.models.cnn_checkpoint import checkpoint_sha256, save_checkpoint
from src.preprocessing.profile import load_profile
from src.preprocessing.validation import (
    F2_MANIFEST_SHA256,
    code_version,
    load_f2_manifest,
    sha256_file,
)
from src.training.cnn_config import CNNConfig, load_cnn_config
from src.training.train_cnn import TrainResult, train_one
from src.training.wandb_logger import MODES, RunLogger

SCHEMA_VERSION = 1
SEARCH_SPLITS = ("train", "val-A")  # the only splits this module reads (D42)
CHECKPOINT_DIR = "checkpoints"
SEARCH_REPORT = "search_report.json"
SELECTION_REPORT = "selection_report.json"
RUN_REPORT = "run_report.json"


def environment(device: str) -> dict[str, str]:
    """Versions and hardware for run_report.json and the checkpoints (all values strings)."""
    cuda = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "safetensors": importlib.metadata.version("safetensors"),
        "wandb": importlib.metadata.version("wandb"),  # metadata only: wandb is not imported here
        "cuda_available": str(cuda).lower(),
        "gpu_name": torch.cuda.get_device_name(0) if cuda else "none",
        "device": str(device),
    }


def _json_text(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"


def run_name(version: str, n_blocks: int, learning_rate: float, seed: int) -> str:
    """W&B run name, e.g. cnn-v1-b3-lr0.001-s0."""
    return f"{version}-b{n_blocks}-lr{learning_rate:g}-s{seed}"


def select_configuration(eers: dict[tuple[int, float], float]) -> tuple[tuple[int, float], list]:
    """The (n_blocks, learning_rate) with the lowest EER and the configurations tied with it.

    Ties follow the D30 tie_rule: n_blocks ascending, then learning_rate descending. The tied list
    (in tie-rule order) has more than one entry exactly when the tie rule decided the selection.
    """
    best = min(eers.values())
    tied = sorted((cfg for cfg, value in eers.items() if value == best),
                  key=lambda cfg: (cfg[0], -cfg[1]))
    return tied[0], tied


def _row(result: TrainResult) -> dict[str, Any]:
    return {
        "best_epoch": result.best_epoch,
        "best_val_a_eer": result.best_val_a_eer,
        "epochs_run": result.epochs_run,
        "history": [asdict(r) for r in result.history],
    }


def search_and_select(train_x, train_y, val_a_x, val_a_y, config: CNNConfig, make_logger, device):
    """Run the D30 search and the final seeds; return (search runs, selected, tied, seed runs, names).

    make_logger(n_blocks, learning_rate, seed) returns (RunLogger, run name). search runs map
    (n_blocks, learning_rate) to the TrainResult of the search seed; seed runs map each final seed to
    the selected configuration's TrainResult, the search-seed run reused (D30).
    """
    def train(n_blocks, learning_rate, seed):
        logger, name = make_logger(n_blocks, learning_rate, seed)
        try:
            result = train_one(
                train_x, train_y, val_a_x, val_a_y, n_blocks=n_blocks, learning_rate=learning_rate,
                seed=seed, max_epochs=config.max_epochs, patience=config.patience,
                batch_size=config.batch_size, logger=logger, device=device,
            )
        finally:
            logger.finish()
        names[(n_blocks, learning_rate, seed)] = name
        return result

    names: dict[tuple[int, float, int], str] = {}
    search = {(n, lr): train(n, lr, config.search_seed) for n, lr in config.grid()}
    selected, tied = select_configuration({cfg: r.best_val_a_eer for cfg, r in search.items()})
    seeds = {s: search[selected] if s == config.search_seed else train(*selected, s)
             for s in config.final_seeds}
    return search, selected, tied, seeds, names


def _check_paths(manifest, split: str, paths: np.ndarray, labels: np.ndarray) -> None:
    """Every cached path is an F2 row of this split with the same label, or raise."""
    rows = manifest.set_index("path")
    missing = [p for p in paths.tolist() if p not in rows.index]
    if missing:
        raise ValueError(f"{split}: {len(missing)} cache paths are not in the F2 manifest, "
                         f"first {missing[0]!r}")
    found = rows.loc[paths.tolist()]
    if (found["split"] != split).any():
        raise ValueError(f"{split}: cache paths belong to another F2 split")
    if not np.array_equal(found["label"].map(LABEL_CODES).to_numpy(), labels):
        raise ValueError(f"{split}: cache labels differ from the F2 manifest labels")


def run_search(cache_dir, profile_path, config_path, manifest_path, out_dir, *, version: str,
               logger_mode: str, device: str) -> dict[str, Any]:
    """Search, select, train the final seeds and write checkpoints and reports; return run_report.

    cache_dir holds train/ and val-A/ (one load_logmel_cache folder each). out_dir must not exist;
    on any failure nothing is left in it.
    """
    commit = code_version()["git_commit"]  # before reading inputs or creating outputs
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise ValueError(f"output folder already exists: {out_dir}")
    if logger_mode not in MODES:
        raise ValueError(f"logger_mode must be one of {MODES}, got {logger_mode!r}")
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f"version must be a non-empty string, got {version!r}")
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"device {device} requested but CUDA is not available")
    profile_path, config_path, manifest_path = Path(profile_path), Path(config_path), Path(manifest_path)
    config = load_cnn_config(config_path)
    profile = load_profile(profile_path)
    manifest = load_f2_manifest(manifest_path)
    profile_config_sha256 = sha256_file(profile_path)

    caches = {}
    for split in SEARCH_SPLITS:
        features, labels, paths, meta = load_logmel_cache(
            Path(cache_dir) / split, split, profile, expected_f2_manifest_sha256=F2_MANIFEST_SHA256
        )
        if meta["profile_config_sha256"] != profile_config_sha256:
            raise ValueError(
                f"{split}: cache was built from profile config {meta['profile_config_sha256']}, "
                f"profile file is {profile_config_sha256}"
            )
        _check_paths(manifest, split, paths, labels)
        caches[split] = (features, labels, meta)

    provenance = {
        "code_commit": commit,
        "profile_fingerprint": profile.fingerprint(),
        "f2_manifest_sha256": sha256_file(manifest_path),
        "train_cache_npz_sha256": caches["train"][2]["npz_sha256"],
        "val_a_cache_npz_sha256": caches["val-A"][2]["npz_sha256"],
    }

    def make_logger(n_blocks, learning_rate, seed):
        name = run_name(version, n_blocks, learning_rate, seed)
        wandb_config = {**config.to_dict(), **provenance, "seed": seed, "n_blocks": n_blocks,
                        "learning_rate": learning_rate}
        return RunLogger(logger_mode, wandb_config, name), name

    search, selected, tied, seeds, names = search_and_select(
        caches["train"][0], caches["train"][1], caches["val-A"][0], caches["val-A"][1],
        config, make_logger, device,
    )
    n_blocks, learning_rate = selected
    bundle_seed = select_bundle_seed({s: r.best_val_a_eer for s, r in seeds.items()})
    env = environment(device)
    # One entry per training run (6): the 4 search runs, then the selected configuration's new seeds.
    peaks = {names[(n, lr, config.search_seed)]: r.peak_gpu_memory_bytes
             for (n, lr), r in search.items()}
    peaks.update({names[(*selected, s)]: r.peak_gpu_memory_bytes for s, r in seeds.items()})

    out_dir.mkdir(parents=True)
    try:
        checkpoint_hashes = {}
        for seed, result in seeds.items():
            model = build_cnn(n_blocks)
            model.load_state_dict(result.state_dict, strict=True)
            metadata = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION, "version": version, "n_blocks": n_blocks,
                "channels": list(CHANNELS[:n_blocks]), "dropout": DROPOUT,
                "input_shape": list(INPUT_SHAPE), "learning_rate": learning_rate, "seed": seed,
                "best_epoch": result.best_epoch, "val_a_eer": result.best_val_a_eer,
                "environment": env, "determinism": result.determinism, **provenance,
            }
            folder = save_checkpoint(model, result.mean, result.std, metadata,
                                     out_dir / CHECKPOINT_DIR / f"seed_{seed}")
            checkpoint_hashes[str(seed)] = checkpoint_sha256(folder)

        search_report = {
            "search_seed": config.search_seed,
            "rows": [{"n_blocks": n, "learning_rate": lr, **_row(r)} for (n, lr), r in search.items()],
        }
        selection_report = {
            "selected": {"n_blocks": n_blocks, "learning_rate": learning_rate},
            "selected_val_a_eer": search[selected].best_val_a_eer,
            "tie_rule": list(config.tie_rule),
            "tie_rule_decided": len(tied) > 1,
            "tied_rows": [{"n_blocks": n, "learning_rate": lr} for n, lr in tied],
            "seeds": {str(s): {**_row(r), "reused_search_run": s == config.search_seed}
                      for s, r in seeds.items()},
            "bundle_seed": bundle_seed,
        }
        run_report = {
            "schema_version": SCHEMA_VERSION,
            "version": version,
            "environment": env,
            "profile_config_sha256": profile_config_sha256,
            "cnn_config_sha256": sha256_file(config_path),
            "caches": {split: {"npz_sha256": meta["npz_sha256"], "n_rows": meta["n_rows"]}
                       for split, (_, _, meta) in caches.items()},
            "checkpoint_sha256": checkpoint_hashes,
            "peak_gpu_memory_bytes": peaks,
            "wandb_mode": logger_mode,
            "wandb_runs": sorted(names.values()),
            "selected": {"n_blocks": n_blocks, "learning_rate": learning_rate},
            "bundle_seed": bundle_seed,
            **provenance,
        }
        texts = {
            SEARCH_REPORT: _json_text(search_report),
            SELECTION_REPORT: _json_text(selection_report),
            RUN_REPORT: _json_text(run_report),
        }
        for name, text in texts.items():
            (out_dir / name).write_text(text, encoding="utf-8")
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    return run_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", required=True, type=Path,
                        help="Folder holding the train and val-A log-Mel caches")
    parser.add_argument("--profile", required=True, type=Path, help="F3 preprocessing profile YAML")
    parser.add_argument("--config", required=True, type=Path, help="configs/cnn.yaml")
    parser.add_argument("--manifest", required=True, type=Path, help="Approved F2 FoR manifest")
    parser.add_argument("--out", required=True, type=Path, help="New output folder")
    parser.add_argument("--version", required=True, help="CNN version, e.g. cnn-v1")
    parser.add_argument("--wandb-mode", required=True, choices=MODES, help="W&B mode (D40)")
    parser.add_argument("--device", required=True, help="torch device, e.g. cuda or cpu")
    args = parser.parse_args(argv)
    report = run_search(args.cache_dir, args.profile, args.config, args.manifest, args.out,
                        version=args.version, logger_mode=args.wandb_mode, device=args.device)
    print(_json_text(report), end="")


if __name__ == "__main__":
    main()
