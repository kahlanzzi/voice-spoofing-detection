"""Model v2 search and selection (D49, D50, D61): train and val-A (through S) only; resumable.

Order (D49): each candidate once with the search seed (0) -> the lowest best M wins; ties: fewer
trainable parameters, then candidate order (cnn-aug, crnn, aasist) -> the winner is trained with
seeds 1 and 2 -> bundle seed = select_bundle_seed over the three M values (median, ties to the lower
seed). Every run is logged through its own allow-list V2RunLogger (D40).

Inputs: the raw train cache (src.features.raw_train_cache), the Stage 8 train log-Mel cache (only
for the clean-train statistics of D61 (5) and as a cross-check of paths and labels), the Stage 8
val-A log-Mel cache (the known answer: S's clean condition must equal it byte for byte), the selection
set S (src.training.selection_set), the F2 manifest, the F3 profile, configs/v2_candidates.yaml
(batch sizes required) and configs/augmentation.yaml (corpus and ffmpeg pins required, V1b).
`resources` (the audited training halves of the noise and RIR corpora) is passed in because the
corpus loader belongs to V1b. There is no CLI `main` in this PR: it is wired in V2b-4, once the V1b
loader exists.

Resumable per run: out_dir/runs/<candidate>_seed<k>/ is either complete (checkpoint/ written by
save_v2_checkpoint + run.json) or absent. A run is built in <name>.partial/ and renamed only when
complete. At start, every *.partial folder is removed (the count is printed), every complete run's
checkpoint hash is verified, and a complete run is reused only if its provenance equals the current
one; otherwise the search refuses. The final reports (search_report.json, selection_report.json,
run_report.json) are written once; if they exist they are verified and returned, never overwritten.

Only train and val-A are read; there is no val-B, held-out or external path (D9).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from src.augmentation import AugmentationResources, Augmenter, load_augmentation_config
from src.augmentation.codec import FFmpegCodecBackend
from src.evaluation.seeds import select_bundle_seed
from src.features.logmel_cache import load_logmel_cache
from src.features.raw_train_cache import check_outside_repo, load_raw_train_cache
from src.models.v2_checkpoint import (
    INPUT_SHAPES,
    INPUTS,
    SCHEMA_VERSION as CHECKPOINT_SCHEMA_VERSION,
    build_model,
    load_v2_checkpoint,
    save_v2_checkpoint,
    trainable_parameters,
    v2_checkpoint_sha256,
)
from src.preprocessing.profile import load_profile
from src.preprocessing.validation import (
    F2_MANIFEST_SHA256,
    code_version,
    load_f2_manifest,
    sha256_file,
)
from src.training.normalization import fit_bin_stats
from src.training.selection_set import load_selection_set
from src.training.train_v2 import V2TrainResult, train_v2
from src.training.v2_config import load_v2_config
from src.training.v2_wandb_logger import MODES, V2RunLogger
from src.tuning.cnn_search import _check_paths, _json_text, environment

SCHEMA_VERSION = 1
TRAIN_SPLIT = "train"
VAL_A_SPLIT = "val-A"
RUNS_DIR = "runs"
PARTIAL_SUFFIX = ".partial"
CHECKPOINT_DIR = "checkpoint"
RUN_JSON = "run.json"
SEARCH_REPORT = "search_report.json"
SELECTION_REPORT = "selection_report.json"
RUN_REPORT = "run_report.json"
REPORTS = (SEARCH_REPORT, SELECTION_REPORT, RUN_REPORT)
S_FORMS = {"logmel": "logmel", "waveform": "waveform"}


def run_dir_name(candidate: str, seed: int) -> str:
    return f"{candidate}_seed{seed}"


def run_name(version: str, candidate: str, seed: int) -> str:
    """W&B run name, e.g. cnn-v2-crnn-s0."""
    return f"{version}-{candidate}-s{seed}"


def select_candidate(ms: dict[str, float], params: dict[str, int],
                     order: tuple[str, ...]) -> tuple[str, list[str]]:
    """The candidate with the lowest M and the candidates tied with it, in tie-rule order (D49)."""
    best = min(ms.values())
    tied = sorted((c for c, m in ms.items() if m == best), key=lambda c: (params[c], order.index(c)))
    return tied[0], tied


def search_and_select(config, run_one: Callable[[str, int], dict]) -> dict[str, Any]:
    """Run the D49 search with run_one(candidate, seed) -> run record; return the selection."""
    search_seed = config.settings["search_seed"]
    order = config.candidate_order
    search = {c: run_one(c, search_seed) for c in order}
    winner, tied = select_candidate({c: r["best_selection_m"] for c, r in search.items()},
                                    {c: r["trainable_parameters"] for c, r in search.items()}, order)
    seeds = {s: search[winner] if s == search_seed else run_one(winner, s)
             for s in config.settings["final_seeds"]}
    bundle_seed = select_bundle_seed({s: r["best_selection_m"] for s, r in seeds.items()})
    return {"search": search, "winner": winner, "tied": tied, "seeds": seeds,
            "bundle_seed": bundle_seed}


def _row(record: dict) -> dict[str, Any]:
    return {k: record[k] for k in ("best_epoch", "best_selection_m", "epochs_run", "history")}


def _reports(config, selection: dict, base: dict, run_report_extra: dict) -> dict[str, dict]:
    search_seed = config.settings["search_seed"]
    winner = selection["winner"]
    search_report = {
        "search_seed": search_seed,
        "rows": [{"candidate": c, "trainable_parameters": r["trainable_parameters"], **_row(r)}
                 for c, r in selection["search"].items()],
    }
    selection_report = {
        "selected": winner,
        "selected_selection_m": selection["search"][winner]["best_selection_m"],
        "tie_rule": list(config.settings["tie_rule"]),
        "tie_rule_decided": len(selection["tied"]) > 1,
        "tied_candidates": list(selection["tied"]),
        "seeds": {str(s): {**{k: r[k] for k in ("best_epoch", "best_selection_m", "epochs_run")},
                           "reused_search_run": s == search_seed}
                  for s, r in selection["seeds"].items()},
        "bundle_seed": selection["bundle_seed"],
    }
    records = [*selection["search"].values(),
               *(r for s, r in selection["seeds"].items() if s != search_seed)]
    run_report = {
        "schema_version": SCHEMA_VERSION,
        **base,
        **run_report_extra,
        "checkpoint_sha256": {run_dir_name(r["candidate"], r["seed"]): r["checkpoint_sha256"]
                              for r in records},
        "peak_gpu_memory_bytes": {r["run_name"]: r["peak_gpu_memory_bytes"] for r in records},
        "wandb_runs": sorted(r["run_name"] for r in records),
        "train_stats_npz_sha256": {run_dir_name(r["candidate"], r["seed"]):
                                   r["provenance"]["train_stats_npz_sha256"] for r in records},
        "selected": winner,
        "tie_rule_decided": len(selection["tied"]) > 1,
        "bundle_seed": selection["bundle_seed"],
    }
    return {SEARCH_REPORT: search_report, SELECTION_REPORT: selection_report, RUN_REPORT: run_report}


def _write_reports(out_dir: Path, reports: dict[str, dict]) -> dict:
    """Write the reports once (run_report last); if they exist, verify and return; never overwrite."""
    texts = {name: _json_text(data) for name, data in reports.items()}
    final = out_dir / RUN_REPORT
    if final.exists():
        existing = json.loads(final.read_text(encoding="utf-8"))
        for name in (SEARCH_REPORT, SELECTION_REPORT):
            path = out_dir / name
            if not path.is_file() or path.read_text(encoding="utf-8") != texts[name]:
                raise ValueError(f"{path} differs from the completed runs; reports are never "
                                 f"overwritten")
        expected = reports[RUN_REPORT]
        for key in ("checkpoint_sha256", "selected", "bundle_seed", "code_commit",
                    "v2_config_sha256", "augmentation_config_sha256", "selection_set_sha256",
                    "raw_train_cache_npz_sha256", "f2_manifest_sha256", "profile_fingerprint",
                    "train_stats_npz_sha256", "val_a_cache_npz_sha256"):
            if existing.get(key) != expected[key]:
                raise ValueError(f"{final}: {key} differs from the completed runs; reports are "
                                 f"never overwritten")
        return existing
    for name in (SEARCH_REPORT, SELECTION_REPORT):
        path = out_dir / name
        if path.exists():
            if path.read_text(encoding="utf-8") != texts[name]:
                raise ValueError(f"{path} exists and differs from the completed runs; reports "
                                 f"are never overwritten")
            continue
        _write_new(path, texts[name])
    _write_new(final, texts[RUN_REPORT])
    return reports[RUN_REPORT]


def _write_new(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + PARTIAL_SUFFIX)
    tmp.write_text(text, encoding="utf-8")
    tmp.rename(path)


def _clean_partials(runs: Path, out_dir: Path) -> int:
    removed = 0
    for folder in (runs, out_dir):
        for path in sorted(folder.iterdir()):
            if path.name.endswith(PARTIAL_SUFFIX):
                shutil.rmtree(path) if path.is_dir() else path.unlink()
                removed += 1
    return removed


def _load_complete_run(folder: Path, expected_provenance: dict) -> dict:
    """Verify one complete run folder; return its run.json, or raise."""
    names = sorted(p.name for p in folder.iterdir())
    if names != sorted([CHECKPOINT_DIR, RUN_JSON]):
        raise ValueError(f"{folder}: a complete run holds exactly {CHECKPOINT_DIR}/ and {RUN_JSON}, "
                         f"found {names}")
    record = json.loads((folder / RUN_JSON).read_text(encoding="utf-8"))
    actual = v2_checkpoint_sha256(folder / CHECKPOINT_DIR)
    if actual != record.get("checkpoint_sha256"):
        raise ValueError(f"{folder}: checkpoint SHA-256 {actual} differs from {RUN_JSON} "
                         f"{record.get('checkpoint_sha256')}")
    _, _, _, meta = load_v2_checkpoint(folder / CHECKPOINT_DIR, expected_sha256=actual)
    if record.get("provenance") != expected_provenance:
        raise ValueError(f"{folder}: provenance differs from the current inputs; a completed run is "
                         f"reused only with the same code, configs, S, val-A cache, raw cache, stats, F2 and F3 "
                         f"(recorded {record.get('provenance')}, current {expected_provenance})")
    if (meta["architecture"], meta["seed"]) != (record["candidate"], record["seed"]):
        raise ValueError(f"{folder}: checkpoint metadata does not match {RUN_JSON}")
    return record


def run_search(*, raw_cache_dir, stage8_train_logmel_dir, stage8_val_a_logmel_dir,
               selection_set_dir, profile_path,
               v2_config_path, augmentation_config_path, manifest_path,
               resources: AugmentationResources, out_dir, version: str, logger_mode: str,
               device: str, num_workers: int) -> dict[str, Any]:
    """Search, select, train the winner's seeds and write the reports; return run_report."""
    commit = code_version()["git_commit"]  # before reading inputs or creating outputs
    if logger_mode not in MODES:
        raise ValueError(f"logger_mode must be one of {MODES}, got {logger_mode!r}")
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f"version must be a non-empty string, got {version!r}")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError(f"num_workers must be an int >= 0, got {num_workers!r}")
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"device {device} requested but CUDA is not available")
    if not isinstance(resources, AugmentationResources) or not resources.noise_waveforms \
            or not resources.rir_waveforms:
        raise ValueError("resources must be AugmentationResources with the audited training halves "
                         "of the noise and RIR corpora (V1b)")
    out_dir = Path(out_dir)
    check_outside_repo(out_dir, "output")
    v2_config_path, augmentation_config_path = Path(v2_config_path), Path(augmentation_config_path)
    profile_path, manifest_path = Path(profile_path), Path(manifest_path)
    config = load_v2_config(v2_config_path)
    batch_sizes = config.require_batch_sizes()

    aug_config = load_augmentation_config(augmentation_config_path)
    if not aug_config.corpus_pins_complete or not aug_config.ffmpeg_pin_complete:
        raise ValueError("configs/augmentation.yaml corpus and ffmpeg pins must be complete (V1b, "
                         "D50 (5)) before v2 training; the codec family is never skipped")
    codec = FFmpegCodecBackend(aug_config.ffmpeg, aug_config.sample_rate)
    codec.verify_pin()
    augmenter = Augmenter(aug_config, resources, codec)

    profile = load_profile(profile_path)
    manifest = load_f2_manifest(manifest_path)
    raw = load_raw_train_cache(raw_cache_dir, expected_f2_manifest_sha256=F2_MANIFEST_SHA256)
    n_zero = raw.meta["n_all_zero_clips"]
    if n_zero > 0:
        raise ValueError(f"raw train cache holds {n_zero} all-zero clip(s) (Augmenter's "
                         f"recorded_noise refuses a silent clip): decision D62 needed before "
                         f"training (docs/v2_training.md)")
    _check_paths(manifest, TRAIN_SPLIT, raw.paths, raw.labels)
    groups = manifest.set_index("path").loc[raw.paths.tolist(), "group"].astype(str).to_numpy()
    if not np.array_equal(groups, raw.groups.astype(object)):
        raise ValueError("raw train cache groups differ from the F2 manifest")
    profile_config_sha256 = sha256_file(profile_path)
    features, labels, paths, meta = load_logmel_cache(
        Path(stage8_train_logmel_dir), TRAIN_SPLIT, profile,
        expected_f2_manifest_sha256=F2_MANIFEST_SHA256)
    if meta["profile_config_sha256"] != profile_config_sha256:
        raise ValueError(f"Stage 8 train cache was built from profile config "
                         f"{meta['profile_config_sha256']}, profile file is {profile_config_sha256}")
    if not np.array_equal(paths, raw.paths) or not np.array_equal(labels, raw.labels):
        raise ValueError("raw train cache paths or labels differ from the Stage 8 train cache")
    mean, std = fit_bin_stats(features)  # clean train (D61 (5))
    del features
    stats_sha256 = meta["npz_sha256"]
    val_a = load_logmel_cache(Path(stage8_val_a_logmel_dir), VAL_A_SPLIT, profile,
                              expected_f2_manifest_sha256=F2_MANIFEST_SHA256)
    if val_a[3]["profile_config_sha256"] != profile_config_sha256:
        raise ValueError(f"Stage 8 val-A cache was built from profile config "
                         f"{val_a[3]['profile_config_sha256']}, profile file is "
                         f"{profile_config_sha256}")

    f2_sha256 = sha256_file(manifest_path)
    aug_sha256 = sha256_file(augmentation_config_path)
    s_cache: dict[str, Any] = {}

    def selection_for(form: str):
        if form not in s_cache:
            s_cache.clear()  # one form in memory at a time
            s_cache[form] = load_selection_set(selection_set_dir, profile, form=form,
                                               f2_manifest=manifest, clean_reference_logmel=val_a)
        return s_cache[form]

    first = selection_for("logmel")
    s_manifest = first.manifest
    for key, value in (("augmentation_config_sha256", aug_sha256), ("f2_manifest_sha256", f2_sha256),
                       ("profile_config_sha256", profile_config_sha256)):
        if s_manifest[key] != value:
            raise ValueError(f"selection set {key} {s_manifest[key]} differs from the current {value}")
    s_sha256 = first.sha256

    base = {
        "code_commit": commit,
        "v2_config_sha256": sha256_file(v2_config_path),
        "augmentation_config_sha256": aug_sha256,
        "profile_fingerprint": profile.fingerprint(),
        "profile_config_sha256": profile_config_sha256,
        "f2_manifest_sha256": f2_sha256,
        "raw_train_cache_npz_sha256": raw.meta["npz_sha256"],
        "selection_set_sha256": s_sha256,
        "val_a_cache_npz_sha256": val_a[3]["npz_sha256"],
    }

    def provenance(candidate: str) -> dict:
        logmel = INPUTS[candidate] == "logmel"
        return {**base, "train_stats_npz_sha256": stats_sha256 if logmel else None}

    runs = out_dir / RUNS_DIR
    runs.mkdir(parents=True, exist_ok=True)
    removed = _clean_partials(runs, out_dir)
    print(f"v2_search: removed {removed} incomplete *{PARTIAL_SUFFIX} folder(s)")
    allowed = {run_dir_name(c, s) for c in config.candidate_order for s in config.settings["final_seeds"]}
    complete = {}
    for folder in sorted(runs.iterdir()):
        if folder.name not in allowed or not folder.is_dir():
            raise ValueError(f"{folder}: unexpected entry in {runs}")
        candidate = folder.name.rsplit("_seed", 1)[0]
        complete[folder.name] = _load_complete_run(folder, provenance(candidate))
    env = environment(device)

    def run_one(candidate: str, seed: int) -> dict:
        name = run_dir_name(candidate, seed)
        if name in complete:
            return complete[name]
        logmel = INPUTS[candidate] == "logmel"
        wandb_name = run_name(version, candidate, seed)
        prov = provenance(candidate)
        wandb_config = {
            **config.to_dict(), "candidate": candidate, "seed": seed,
            "batch_size": batch_sizes[candidate], "num_workers": num_workers,
            **{k: prov[k] for k in ("code_commit", "profile_fingerprint", "f2_manifest_sha256",
                                    "raw_train_cache_npz_sha256", "selection_set_sha256",
                                    "augmentation_config_sha256", "train_stats_npz_sha256")},
        }
        partial = runs / f"{name}{PARTIAL_SUFFIX}"
        partial.mkdir()
        try:
            logger = V2RunLogger(logger_mode, wandb_config, wandb_name)
            try:
                result: V2TrainResult = train_v2(
                    candidate, config=config, run_seed=seed, raw_cache=raw,
                    selection=selection_for(S_FORMS[INPUTS[candidate]]), augmenter=augmenter,
                    profile=profile, mean=mean if logmel else None, std=std if logmel else None,
                    max_epochs=config.settings["max_epochs"], patience=config.settings["patience"],
                    batch_size=batch_sizes[candidate], num_workers=num_workers, logger=logger,
                    device=device,
                )
            finally:
                logger.finish()
            model = build_model(candidate)
            model.load_state_dict(result.state_dict, strict=True)
            metadata = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION, "architecture": candidate,
                "version": version, "input": INPUTS[candidate],
                "input_shape": list(INPUT_SHAPES[INPUTS[candidate]]), "seed": seed,
                "best_epoch": result.best_epoch, "selection_m": result.best_selection_m,
                "code_commit": commit, "environment": env,
                "profile_fingerprint": prov["profile_fingerprint"],
                "f2_manifest_sha256": prov["f2_manifest_sha256"],
                "augmentation_config_sha256": prov["augmentation_config_sha256"],
                "selection_set_sha256": prov["selection_set_sha256"],
                "train_stats_npz_sha256": prov["train_stats_npz_sha256"],
                "determinism": result.determinism,
                "trainable_parameters": trainable_parameters(model),
            }
            folder = save_v2_checkpoint(model, metadata, partial / CHECKPOINT_DIR,
                                        mean=mean if logmel else None, std=std if logmel else None)
            record = {
                "schema_version": SCHEMA_VERSION, "candidate": candidate, "seed": seed,
                "version": version, "run_name": wandb_name, "batch_size": batch_sizes[candidate],
                "num_workers": num_workers, "best_epoch": result.best_epoch,
                "best_selection_m": result.best_selection_m, "epochs_run": result.epochs_run,
                "history": list(result.history),
                "peak_gpu_memory_bytes": result.peak_gpu_memory_bytes,
                "trainable_parameters": result.trainable_parameters,
                "checkpoint_sha256": v2_checkpoint_sha256(folder),
                "determinism": result.determinism, "environment": env,
                "wandb_mode": logger_mode, "provenance": prov,
            }
            (partial / RUN_JSON).write_text(_json_text(record), encoding="utf-8")
            partial.rename(runs / name)
        except BaseException:
            shutil.rmtree(partial, ignore_errors=True)
            raise
        record = json.loads((runs / name / RUN_JSON).read_text(encoding="utf-8"))
        complete[name] = record
        return record

    selection = search_and_select(config, run_one)
    extra = {"version": version, "environment": env, "wandb_mode": logger_mode,
             "batch_sizes": batch_sizes, "num_workers": num_workers}
    return _write_reports(out_dir, _reports(config, selection, base, extra))
