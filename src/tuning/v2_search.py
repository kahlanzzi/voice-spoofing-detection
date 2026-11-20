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
`resources` (the audited training halves of the noise and RIR corpora) is passed in; run_search
itself does not load them.

CLI (V2b-4, `main`): `search` and `rehearse`, every data path explicit; the F3 profile, both configs
and the F2 manifest are the committed repository files. Before run_search, `search` refuses a
rehearsal folder as output, a raw train cache whose npz differs from the committed V2b-3 evidence
(check_raw_cache_pin) and a resource cache whose inventory or archive hashes differ from the
committed V1b inventories and configs/augmentation.yaml pins (load_v2_resources). `rehearse` is in
src.tuning.v2_rehearsal; its output can never be used as a search output.

Resumable per run and per epoch: out_dir/runs/<candidate>_seed<k>/ is either complete (checkpoint/
written by save_v2_checkpoint + run.json) or absent. A run is built in <name>.partial/, where
train_v2 keeps its per-epoch resume state (resume/, src.training.v2_resume), and renamed only when
complete; on any exception the .partial folder is kept. At start: a .partial with a valid resume state
of the same candidate, seed and provenance is resumed at its next epoch; a .partial without one
(stopped before its first epoch ended) is removed (the count is printed); a .partial whose state is
invalid or has another provenance is refused and kept (trained epochs are never deleted silently).
Every complete run's checkpoint hash is verified, and a complete run is reused only if its provenance
equals the current one; otherwise the search refuses. resume/ is deleted only once the renamed run
folder is complete and verified. The final reports (search_report.json, selection_report.json,
run_report.json) are written once; if they exist they are verified and returned, never overwritten.

Only train and val-A are read; there is no val-B, held-out or external path (D9).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from src.augmentation import AugmentationResources, Augmenter, load_augmentation_config
from src.augmentation.codec import FFmpegCodecBackend
from src.augmentation.resource_cache import NOISE_COUNT, RIR_COUNT, load_training_resources
from src.augmentation.version_guard import require_versions
from src.evaluation.seeds import select_bundle_seed
from src.features.logmel_cache import load_logmel_cache
from src.features.raw_train_cache import JSON_NAME as RAW_JSON_NAME
from src.features.raw_train_cache import NPZ_NAME as RAW_NPZ_NAME
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
from src.training.v2_resume import RESUME_DIR, has_state, read_state, remove_state, state_folders
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

# The committed files the CLI uses (not arguments), read at call time.
REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = REPO_ROOT / "configs" / "preprocessing.yaml"
V2_CONFIG_PATH = REPO_ROOT / "configs" / "v2_candidates.yaml"
AUGMENTATION_CONFIG_PATH = REPO_ROOT / "configs" / "augmentation.yaml"
F2_MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "for2sec_frozen_manifest.csv"
NOISE_INVENTORY_PATH = REPO_ROOT / "data" / "manifests" / "augmentation" / "musan_inventory.csv"
RIR_INVENTORY_PATH = REPO_ROOT / "data" / "manifests" / "augmentation" / "rirs_noises_inventory.csv"
RAW_CACHE_EVIDENCE = REPO_ROOT / "docs" / "audit_evidence" / "v2b3_raw_train_cache" / RAW_JSON_NAME
# A rehearsal output folder holds this marker (written first) and its report; search refuses both.
REHEARSAL_MARKER = "REHEARSAL_ONLY.json"
REHEARSAL_REPORT = "rehearsal_report.json"


def run_dir_name(candidate: str, seed: int) -> str:
    return f"{candidate}_seed{seed}"


def run_name(version: str, candidate: str, seed: int) -> str:
    """W&B run name, e.g. model-v2-crnn-s0."""
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


def _remove(path: Path) -> None:
    shutil.rmtree(path) if path.is_dir() else path.unlink()


def _scan_partials(runs: Path, out_dir: Path, allowed: set[str],
                   provenance: Callable[[str], dict]) -> tuple[int, dict[str, dict]]:
    """Remove the .partial entries that hold no trained epoch; return (count, resumable states).

    A run's .partial with a resume state must be valid for its candidate, seed and the current
    provenance; otherwise the search refuses and the folder is kept.
    """
    removed, resumable = 0, {}
    for path in sorted(out_dir.iterdir()):  # unfinished report files
        if path.name.endswith(PARTIAL_SUFFIX):
            _remove(path)
            removed += 1
    for path in sorted(runs.iterdir()):
        if not path.name.endswith(PARTIAL_SUFFIX):
            continue
        name = path.name[:-len(PARTIAL_SUFFIX)]
        resume = path / RESUME_DIR
        if not path.is_dir() or not has_state(resume):
            _remove(path)
            removed += 1
            continue
        if name not in allowed:
            raise ValueError(f"{path}: holds a resume state but is not a run of this search; it is "
                             f"kept: inspect it and remove it by hand")
        candidate, seed = name.rsplit("_seed", 1)
        try:
            meta, _ = read_state(resume, candidate=candidate, run_seed=int(seed),
                                 provenance=provenance(candidate))
        except ValueError as exc:
            raise ValueError(f"{path}: holds trained epochs but its resume state cannot be resumed "
                             f"({exc}); it is kept, never deleted silently: inspect it and remove "
                             f"it by hand to retrain the run") from exc
        resumable[name] = meta
    return removed, resumable


def _load_complete_run(folder: Path, expected_provenance: dict) -> dict:
    """Verify one complete run folder; return its run.json, or raise.

    A resume state left over from training (the process stopped between the rename and its
    deletion) is deleted only after the run verifies.
    """
    leftovers = {p.name for p in state_folders(folder / RESUME_DIR)}
    names = sorted(p.name for p in folder.iterdir() if p.name not in leftovers)
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
    remove_state(folder / RESUME_DIR)
    return record


def run_search(*, raw_cache_dir, stage8_train_logmel_dir, stage8_val_a_logmel_dir,
               selection_set_dir, profile_path,
               v2_config_path, augmentation_config_path, manifest_path,
               resources: AugmentationResources, out_dir, version: str, logger_mode: str,
               device: str, num_workers: int) -> dict[str, Any]:
    """Search, select, train the winner's seeds and write the reports; return run_report."""
    require_versions()
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
    allowed = {run_dir_name(c, s) for c in config.candidate_order for s in config.settings["final_seeds"]}
    removed, resumable = _scan_partials(runs, out_dir, allowed, provenance)
    print(f"v2_search: removed {removed} incomplete *{PARTIAL_SUFFIX} folder(s)")
    for name, state in sorted(resumable.items()):
        print(f"v2_search: {name} resumes after epoch {state['last_epoch']}")
    complete = {}
    for folder in sorted(runs.iterdir()):
        if folder.name.endswith(PARTIAL_SUFFIX) and folder.name[:-len(PARTIAL_SUFFIX)] in resumable:
            continue
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
        state = resumable.pop(name, None)  # verified at start
        partial.mkdir(exist_ok=state is not None)
        # On any exception the .partial folder is kept: its resume state (if an epoch finished)
        # is resumed by the next run_search; without one it is removed then.
        logger = V2RunLogger(logger_mode, wandb_config, wandb_name,
                             resume_id=None if state is None else state["wandb_run_id"])
        try:
            result: V2TrainResult = train_v2(
                candidate, config=config, run_seed=seed, raw_cache=raw,
                selection=selection_for(S_FORMS[INPUTS[candidate]]), augmenter=augmenter,
                profile=profile, mean=mean if logmel else None, std=std if logmel else None,
                max_epochs=config.settings["max_epochs"], patience=config.settings["patience"],
                batch_size=batch_sizes[candidate], num_workers=num_workers, logger=logger,
                device=device, resume_dir=partial / RESUME_DIR, provenance=prov,
            )
        finally:
            logger.finish()
        for stale in (partial / CHECKPOINT_DIR, partial / RUN_JSON):
            if stale.exists():  # written by a session that stopped before the rename
                _remove(stale)
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
        record = _load_complete_run(runs / name, prov)  # verifies, then deletes resume/
        complete[name] = record
        return record

    selection = search_and_select(config, run_one)
    extra = {"version": version, "environment": env, "wandb_mode": logger_mode,
             "batch_sizes": batch_sizes, "num_workers": num_workers}
    return _write_reports(out_dir, _reports(config, selection, base, extra))


# --- CLI checks (V2b-4) --------------------------------------------------------------------------

def load_v2_resources(directory, *, noise_inventory_path=None, rir_inventory_path=None,
                      augmentation_config_path=None, expected_noise_count: int = NOISE_COUNT,
                      expected_rir_count: int = RIR_COUNT) -> tuple[AugmentationResources, dict]:
    """load_training_resources, then the cache's provenance must equal the committed V1b files.

    load_training_resources checks the cache against its own metadata only. Here the recorded
    noise_inventory_sha256 / rir_inventory_sha256 must equal the SHA-256 of the committed
    musan_inventory.csv / rirs_noises_inventory.csv, and noise_archive_sha256 / rir_archive_sha256
    the configs/augmentation.yaml pins. Every mismatch is named with the expected and actual value.
    """
    noise_inventory_path = Path(noise_inventory_path or NOISE_INVENTORY_PATH)
    rir_inventory_path = Path(rir_inventory_path or RIR_INVENTORY_PATH)
    corpora = load_augmentation_config(Path(augmentation_config_path or AUGMENTATION_CONFIG_PATH)).corpora
    resources, meta = load_training_resources(Path(directory), expected_noise_count=expected_noise_count,
                                              expected_rir_count=expected_rir_count)
    expected = {
        "noise_inventory_sha256": sha256_file(noise_inventory_path),
        "rir_inventory_sha256": sha256_file(rir_inventory_path),
        "noise_archive_sha256": str(corpora["noise"]["archive_sha256"]),
        "rir_archive_sha256": str(corpora["rir"]["archive_sha256"]),
    }
    wrong = [f"{field}: expected {value}, got {meta[field]}" for field, value in expected.items()
             if meta[field] != value]
    if wrong:
        raise ValueError(f"training resource cache {directory} was not built from the committed V1b "
                         f"inventories and pins: " + "; ".join(wrong))
    return resources, meta


def check_raw_cache_pin(raw_cache_dir, evidence_path=None) -> str:
    """The raw train cache must be the one recorded in the committed V2b-3 evidence; return its npz SHA-256.

    The npz bytes (streamed) must hash to the evidence's npz_sha256 and train_raw.json must equal the
    evidence file byte for byte.
    """
    evidence_path = Path(evidence_path or RAW_CACHE_EVIDENCE)
    expected = json.loads(evidence_path.read_text(encoding="utf-8"))["npz_sha256"]
    folder = Path(raw_cache_dir)
    npz, meta = folder / RAW_NPZ_NAME, folder / RAW_JSON_NAME
    for path in (npz, meta):
        if not path.is_file():
            raise ValueError(f"raw train cache file {path} is missing")
    actual = sha256_file(npz)
    if actual != expected:
        raise ValueError(f"raw train cache {npz}: npz SHA-256 expected {expected} (V2b-3 evidence "
                         f"{evidence_path.name}), got {actual}")
    expected_json, actual_json = sha256_file(evidence_path), sha256_file(meta)
    if actual_json != expected_json:
        raise ValueError(f"raw train cache {meta}: SHA-256 expected {expected_json} (the committed "
                         f"V2b-3 evidence file), got {actual_json}")
    return actual


def refuse_rehearsal_folder(out_dir) -> None:
    """A rehearsal output is never reused or resumed by the search."""
    out_dir = Path(out_dir)
    for name in (REHEARSAL_MARKER, REHEARSAL_REPORT):
        if (out_dir / name).exists():
            raise ValueError(f"{out_dir} is a rehearsal folder ({name} present); a rehearsal output "
                             f"is never used by the search: choose the search folder")


def search_cli(*, raw_cache_dir, stage8_train_logmel_dir, stage8_val_a_logmel_dir,
               selection_set_dir, resources_dir, out_dir, version: str, logger_mode: str,
               device: str, num_workers: int) -> dict[str, Any]:
    """The `search` subcommand: the CLI checks above, then run_search with the committed files."""
    require_versions()
    refuse_rehearsal_folder(out_dir)
    check_raw_cache_pin(raw_cache_dir)
    resources, _ = load_v2_resources(resources_dir)
    return run_search(
        raw_cache_dir=raw_cache_dir, stage8_train_logmel_dir=stage8_train_logmel_dir,
        stage8_val_a_logmel_dir=stage8_val_a_logmel_dir, selection_set_dir=selection_set_dir,
        profile_path=PROFILE_PATH, v2_config_path=V2_CONFIG_PATH,
        augmentation_config_path=AUGMENTATION_CONFIG_PATH, manifest_path=F2_MANIFEST_PATH,
        resources=resources, out_dir=out_dir, version=version, logger_mode=logger_mode,
        device=device, num_workers=num_workers,
    )


def _inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--raw-cache-dir", required=True, type=Path,
                        help="Raw train cache folder (train_raw.npz + .json, V2b-3)")
    parser.add_argument("--stage8-train-logmel-dir", required=True, type=Path,
                        help="Stage 8 train log-Mel cache folder (clean-train statistics)")
    parser.add_argument("--stage8-val-a-logmel-dir", required=True, type=Path,
                        help="Stage 8 val-A log-Mel cache folder (S known answer)")
    parser.add_argument("--selection-set-dir", required=True, type=Path, help="Selection set S (V1c)")
    parser.add_argument("--resources-dir", required=True, type=Path,
                        help="Training-half resource cache (V1c)")
    parser.add_argument("--device", required=True, help="torch device, e.g. cuda or cpu")
    parser.add_argument("--num-workers", required=True, type=int, help="DataLoader workers (>= 0)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    search = sub.add_parser("search", help="Run (or resume) the D49 search; prints run_report.json")
    _inputs(search)
    search.add_argument("--out", required=True, type=Path,
                        help="Search folder (runs/ and the reports); re-run the same command to resume")
    search.add_argument("--version", required=True, help="Model version, e.g. model-v2 (the F5 model version)")
    search.add_argument("--wandb-mode", required=True, choices=MODES, help="W&B mode (D40)")
    rehearse = sub.add_parser("rehearse", help="T4 rehearsal before the search; no W&B")
    _inputs(rehearse)
    rehearse.add_argument("--out", required=True, type=Path,
                          help="New rehearsal folder, never the search folder")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    common = dict(raw_cache_dir=args.raw_cache_dir,
                  stage8_train_logmel_dir=args.stage8_train_logmel_dir,
                  stage8_val_a_logmel_dir=args.stage8_val_a_logmel_dir,
                  selection_set_dir=args.selection_set_dir, resources_dir=args.resources_dir,
                  out_dir=args.out, device=args.device, num_workers=args.num_workers)
    if args.command == "search":
        report = search_cli(**common, version=args.version, logger_mode=args.wandb_mode)
    else:
        from src.tuning.v2_rehearsal import rehearse  # imports this module

        report = rehearse(**common)
    print(_json_text(report), end="")


if __name__ == "__main__":
    main()
