"""Materialize the frozen V1b val-A recipe as the model-v2 training selection set S."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.augmentation.codec import FFmpegCodecBackend
from src.augmentation.config import load_augmentation_config
from src.augmentation.selection_set_recipe import (
    CONDITIONS,
    apply_condition,
    derived_seed,
    waveform_sha256,
)
from src.features.logmel_cache import load_logmel_cache
from src.features.raw_train_cache import check_outside_repo
from src.preprocessing.features import log_mel_spectrogram
from src.preprocessing.profile import load_profile
from src.preprocessing.validation import (
    code_version,
    load_f2_manifest,
    read_verified_audio,
    sha256_file,
)
from src.preprocessing.waveform import preprocess_waveform
from src.training.selection_set import load_selection_set, write_selection_set

RECIPE_SHA256 = "71a340f009b5b0a20e295f4031c15fe2471b1174444b4b0fa2d3014f34c72533"
EXPECTED_SOURCES = 1330
EXPECTED_RECORDS = EXPECTED_SOURCES * len(CONDITIONS)
EXPECTED_COUNTS = {"REAL": 660, "FAKE": 670}
STAGE8_VAL_A_NPZ_SHA256 = "4502eea2f77cfd83d7e2b1d777e392fa460a13e805a936faa2788df8e7862c81"
COLAB_RUNTIME_VERSION = "2026.10"
PYTHON_VERSION = "3.13.16"
TORCH_VERSION = "2.11.0+cu130"
RECIPE_COLUMNS = (
    "source_path", "label", "subgroup", "group", "source_file_sha256", "condition",
    "condition_seed", "resource_path", "resource_file_sha256", "resource_channel",
    "parameters_json", "sample_rate_hz", "samples", "waveform_float32_sha256",
)


def _records(path: Path, half: str) -> list[dict[str, Any]]:
    frame = pd.read_csv(path, dtype={"resource_channel": "string"}, keep_default_na=False)
    required = {"rel_path", "sha256", "half"}
    if not required.issubset(frame.columns):
        raise ValueError(f"inventory {path} is missing {sorted(required - set(frame.columns))}")
    return frame.loc[frame["half"] == half].sort_values("rel_path", kind="stable").to_dict("records")


def load_recipe(path: Path, *, expected_sources: int = EXPECTED_SOURCES) -> pd.DataFrame:
    """Verify the frozen recipe structure, split semantics, order and deterministic seeds."""
    path = Path(path)
    actual = sha256_file(path)
    if expected_sources == EXPECTED_SOURCES and actual != RECIPE_SHA256:
        raise ValueError(f"selection recipe SHA-256 {actual} differs from frozen {RECIPE_SHA256}")
    recipe = pd.read_csv(path, keep_default_na=False)
    if tuple(recipe.columns) != RECIPE_COLUMNS:
        raise ValueError(f"selection recipe columns differ from {list(RECIPE_COLUMNS)}")
    if len(recipe) != expected_sources * len(CONDITIONS):
        raise ValueError(f"selection recipe needs {expected_sources * len(CONDITIONS)} rows, found {len(recipe)}")
    grouped = recipe.groupby("source_path", sort=False)
    if grouped.ngroups != expected_sources:
        raise ValueError(f"selection recipe needs {expected_sources} unique sources")
    if expected_sources == EXPECTED_SOURCES:
        source_rows = recipe.drop_duplicates("source_path", keep="first")
        counts = source_rows["label"].value_counts().to_dict()
        if counts != EXPECTED_COUNTS:
            raise ValueError(f"selection recipe label counts {counts} differ from {EXPECTED_COUNTS}")
    previous = None
    for source_path, rows in grouped:
        if previous is not None and source_path <= previous:
            raise ValueError("selection recipe source paths must be unique and sorted")
        previous = source_path
        if rows["condition"].tolist() != list(CONDITIONS):
            raise ValueError(f"{source_path}: conditions/order differ from {list(CONDITIONS)}")
        for _, row in rows.iterrows():
            expected_seed = derived_seed(source_path, str(row["condition"]))
            if int(row["condition_seed"]) != expected_seed:
                raise ValueError(f"{source_path}/{row['condition']}: condition seed differs")
            if int(row["sample_rate_hz"]) != 16000 or int(row["samples"]) != 32000:
                raise ValueError(f"{source_path}/{row['condition']}: waveform shape metadata differs")
    return recipe


def materialize_arrays(
    for_root: Path,
    f2_manifest: pd.DataFrame,
    recipe: pd.DataFrame,
    *,
    noise_inventory: list[dict[str, Any]],
    rir_inventory: list[dict[str, Any]],
    musan_noise_root: Path,
    rirs_root: Path,
    codec_backend,
    codec_options,
    profile,
    expected_sources: int = EXPECTED_SOURCES,
) -> dict[str, dict[str, np.ndarray]]:
    """Regenerate recipe waveforms, check the recorded hashes, then apply unchanged F3."""
    rows = f2_manifest.loc[f2_manifest["split"] == "val-A"].sort_values("path", kind="stable")
    if len(rows) != expected_sources:
        raise ValueError(f"F2 val-A needs {expected_sources} rows, found {len(rows)}")
    recipe_by_source = {name: part for name, part in recipe.groupby("source_path", sort=False)}
    if set(recipe_by_source) != set(rows["path"].astype(str)):
        raise ValueError("selection recipe sources differ from the F2 val-A rows")
    arrays = {
        name: {
            "waveforms": np.empty((expected_sources, 32000), dtype=np.float32),
            "log_mel": np.empty((expected_sources, 1, 64, 197), dtype=np.float32),
        }
        for name in CONDITIONS
    }
    labels = rows["label"].map({"REAL": 0, "FAKE": 1}).to_numpy(dtype=np.int8)
    paths_list = rows["path"].astype(str).tolist()
    paths = np.asarray(paths_list, dtype=f"<U{max(map(len, paths_list))}")
    groups_list = rows["group"].astype(str).tolist()
    groups = np.asarray(groups_list, dtype=f"<U{max(map(len, groups_list))}")

    for i, (_, source_row) in enumerate(rows.iterrows()):
        source_path = str(source_row["path"])
        source, sample_rate = read_verified_audio(Path(for_root), source_row)
        if sample_rate != 16000 or source.ndim != 1 or source.size != 32000:
            raise ValueError(f"{source_path}: expected a 16 kHz, 32,000-sample mono source")
        recipe_rows = recipe_by_source[source_path]
        for condition in CONDITIONS:
            recipe_row = recipe_rows.loc[recipe_rows["condition"] == condition].iloc[0]
            for field in ("label", "subgroup", "group"):
                if str(recipe_row[field]) != str(source_row[field]):
                    raise ValueError(f"{source_path}/{condition}: {field} differs from F2")
            if str(recipe_row["source_file_sha256"]) != str(source_row["hash"]):
                raise ValueError(f"{source_path}/{condition}: source hash differs from F2")
            generated = apply_condition(
                source, source_path, condition,
                noise_inventory=noise_inventory, rir_inventory=rir_inventory,
                noise_root=musan_noise_root, rirs_root=rirs_root,
                codec_backend=codec_backend, codec_options=codec_options,
            )
            for key in ("resource_path", "resource_file_sha256"):
                if str(generated[key]) != str(recipe_row[key]):
                    raise ValueError(f"{source_path}/{condition}: {key} differs from recipe")
            expected_channel = str(recipe_row["resource_channel"])
            actual_channel = "" if generated["resource_channel"] == "" else str(generated["resource_channel"])
            if actual_channel != expected_channel:
                raise ValueError(f"{source_path}/{condition}: resource_channel differs from recipe")
            params = json.dumps(generated["parameters"], sort_keys=True, separators=(",", ":"))
            if params != str(recipe_row["parameters_json"]):
                raise ValueError(f"{source_path}/{condition}: parameters_json differs from recipe")
            actual_waveform_hash = waveform_sha256(generated["output"])
            if actual_waveform_hash != str(recipe_row["waveform_float32_sha256"]):
                raise ValueError(
                    f"{source_path}/{condition}: waveform SHA-256 {actual_waveform_hash} differs "
                    f"from recipe {recipe_row['waveform_float32_sha256']}"
                )
            prepared = preprocess_waveform(generated["output"], 16000, profile).waveform
            arrays[condition]["waveforms"][i] = prepared
            arrays[condition]["log_mel"][i] = log_mel_spectrogram(prepared, profile)
    for condition in CONDITIONS:
        arrays[condition].update({"labels": labels.copy(), "paths": paths.copy(), "groups": groups.copy()})
    return arrays


def _condition_fields() -> list[dict[str, Any]]:
    recipe = {"recipe_file_sha256": RECIPE_SHA256}
    return [
        {"name": "clean", "parameters": dict(recipe)},
        {"name": "recorded_noise_10db", "parameters": {
            "snr_db": 10.0, "corpus_half": "selection", **recipe,
        }},
        {"name": "reverberation", "parameters": {"convolution": "full_then_crop_to_input",
                                                    "corpus_half": "selection",
                                                    "channel_policy": "seeded_single_channel",
                                                    **recipe}},
        {"name": "opus_16", "parameters": {"bitrate_kbps": 16, "vbr": False,
                                              "application": "voip", "codec_option": "opus_16",
                                              **recipe}},
        {"name": "narrowband_mulaw", "parameters": {"intermediate_sample_rate_hz": 8000,
                                                       "output_sample_rate_hz": 16000,
                                                       "codec_option": "narrowband_mulaw", **recipe}},
    ]


def check_stage8_reference(reference):
    """Require the approved Stage 8 val-A cache before clean-condition comparison."""
    if reference[3]["npz_sha256"] != STAGE8_VAL_A_NPZ_SHA256:
        raise ValueError(
            "Stage 8 val-A cache SHA-256 "
            f"{reference[3]['npz_sha256']} differs from approved {STAGE8_VAL_A_NPZ_SHA256}"
        )
    return reference


def official_runtime_record(codec_backend, colab_runtime_version: str) -> dict[str, str]:
    """Assert and record the D58 Colab T4 runtime used for the official V1c build."""
    import torch

    python_version = platform.python_version()
    torch_version = str(torch.__version__)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    checks = {
        "Colab runtime": (colab_runtime_version, COLAB_RUNTIME_VERSION),
        "Python": (python_version, PYTHON_VERSION),
        "torch": (torch_version, TORCH_VERSION),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"{name} must be {expected!r}, got {actual!r}")
    if "T4" not in gpu:
        raise ValueError(f"GPU must be a T4, got {gpu!r}")
    return {
        "colab_runtime_version": colab_runtime_version,
        "gpu": gpu,
        "python": python_version,
        "numpy": np.__version__,
        "librosa": importlib.metadata.version("librosa"),
        "torch": torch_version,
        "ffmpeg_version_line": codec_backend.version_text().splitlines()[0],
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
    }


def run(
    for_root: Path,
    f2_manifest_path: Path,
    recipe_path: Path,
    musan_noise_root: Path,
    rirs_root: Path,
    out: Path,
    clean_reference_logmel: Path,
    colab_runtime_version: str,
) -> str:
    """Official clean-worktree V1c build; return the selection-set manifest SHA-256."""
    version = code_version()
    check_outside_repo(Path(out), "selection set")
    repo = Path(__file__).resolve().parents[2]
    profile_path, aug_path = repo / "configs" / "preprocessing.yaml", repo / "configs" / "augmentation.yaml"
    profile = load_profile(profile_path)
    config = load_augmentation_config(aug_path)
    if not config.corpus_pins_complete or not config.ffmpeg_pin_complete:
        raise ValueError("V1b corpus and FFmpeg pins must be complete")
    codec = FFmpegCodecBackend(config.ffmpeg, config.sample_rate)
    codec.verify_pin()
    f2 = load_f2_manifest(Path(f2_manifest_path))
    recipe = load_recipe(Path(recipe_path))
    manifest_dir = repo / "data" / "manifests" / "augmentation"
    arrays = materialize_arrays(
        for_root, f2, recipe,
        noise_inventory=_records(manifest_dir / "musan_inventory.csv", "selection"),
        rir_inventory=_records(manifest_dir / "rirs_noises_inventory.csv", "selection"),
        musan_noise_root=musan_noise_root, rirs_root=rirs_root,
        codec_backend=codec, codec_options={x["id"]: x for x in config.codec_options}, profile=profile,
    )
    runtime = official_runtime_record(codec, colab_runtime_version)
    fields = {
        "conditions": _condition_fields(),
        "f2_manifest_sha256": sha256_file(Path(f2_manifest_path)),
        "profile_fingerprint": profile.fingerprint(),
        "profile_config_sha256": sha256_file(profile_path),
        "augmentation_config_sha256": sha256_file(aug_path),
        "code_commit": version["git_commit"], "runtime": runtime,
    }
    selection_sha = write_selection_set(Path(out), arrays, fields, profile=profile)
    clean_reference = check_stage8_reference(
        load_logmel_cache(Path(clean_reference_logmel), "val-A", profile)
    )
    for form in ("waveform", "logmel"):
        loaded = load_selection_set(
            Path(out), profile, form=form, f2_manifest=f2,
            clean_reference_logmel=clean_reference,
        )
        if loaded.sha256 != selection_sha:
            raise ValueError(f"{form}: selection-set manifest hash changed during verification")
    return selection_sha


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--for-root", required=True, type=Path)
    parser.add_argument("--f2-manifest", required=True, type=Path)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--musan-noise-root", required=True, type=Path)
    parser.add_argument("--rirs-root", required=True, type=Path)
    parser.add_argument("--clean-reference-logmel", required=True, type=Path)
    parser.add_argument("--colab-runtime-version", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    digest = run(
        args.for_root, args.f2_manifest, args.recipe, args.musan_noise_root, args.rirs_root,
        args.out, args.clean_reference_logmel, args.colab_runtime_version,
    )
    print(json.dumps({"selection_set_sha256": digest, "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
