"""Regenerate the D64 selection recipe and prove that only derived hashes changed."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.augmentation.codec import FFmpegCodecBackend
from src.augmentation.config import load_augmentation_config
from src.augmentation.selection_set_build import _records
from src.augmentation.selection_set_build import official_runtime_record
from src.augmentation.selection_set_recipe import (
    CONDITIONS,
    RECIPE_COLUMNS,
    apply_condition,
    waveform_sha256,
)
from src.augmentation.version_guard import require_versions
from src.features.raw_train_cache import check_outside_repo
from src.preprocessing.validation import (
    code_version,
    load_f2_manifest,
    read_verified_audio,
    sha256_file,
)

HASH_COLUMN = "waveform_float32_sha256"
KEY_COLUMNS = ("source_path", "condition")
INVARIANT_COLUMNS = tuple(column for column in RECIPE_COLUMNS if column != HASH_COLUMN)


def _semantic_sha256(frame: pd.DataFrame) -> str:
    rows = frame.astype(str).to_dict("records")
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def load_committed_old(repo: Path) -> tuple[Path, Path, pd.DataFrame, dict[str, Any]]:
    """Load and cryptographically bind the old side to the committed V1b recipe and report."""
    manifest_dir = Path(repo) / "data" / "manifests" / "augmentation"
    recipe_path = manifest_dir / "selection_set_s.csv"
    report_path = manifest_dir / "selection_set_s_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    recipe = pd.read_csv(recipe_path, dtype=str, keep_default_na=False)
    actual_file = sha256_file(recipe_path)
    expected_file = str(report.get("manifest_file_sha256", ""))
    if actual_file != expected_file:
        raise ValueError(
            f"committed old recipe SHA-256 {actual_file} differs from report {expected_file}"
        )
    actual_semantic = _semantic_sha256(recipe)
    expected_semantic = str(report.get("manifest_semantic_sha256", ""))
    if actual_semantic != expected_semantic:
        raise ValueError(
            f"committed old semantic SHA-256 {actual_semantic} differs from report {expected_semantic}"
        )
    return recipe_path, report_path, recipe, report


def _validate_inventory_hashes(manifest_dir: Path, report: dict[str, Any]) -> None:
    """Bind the inventory hashes carried by the new report to committed CSV bytes."""
    inventory_files = {
        "musan_inventory_sha256": "musan_inventory.csv",
        "rirs_inventory_sha256": "rirs_noises_inventory.csv",
    }
    for report_key, filename in inventory_files.items():
        path = Path(manifest_dir) / filename
        actual = sha256_file(path)
        expected = str(report.get(report_key, ""))
        if actual != expected:
            raise ValueError(
                f"committed {filename} SHA-256 {actual} differs from report {expected}"
            )


def make_new_report(
    old_report: dict[str, Any],
    *,
    recipe_sha256: str,
    semantic_sha256: str,
    commit: str,
    versions: dict[str, str],
    runtime: dict[str, str],
) -> dict[str, Any]:
    """Create the D64 report without carrying forward the old run provenance."""
    report = dict(old_report)
    report.update({
        "manifest_file_sha256": recipe_sha256,
        "manifest_semantic_sha256": semantic_sha256,
        "repo_base_commit": commit,
        "d64_refreeze_commit": commit,
        "decisions": ["D49", "D50", "D56", "D64"],
        "library_versions": dict(versions),
        "runtime": dict(runtime),
    })
    return report


def _resource_halves(frame: pd.DataFrame, noise_rows, rir_rows) -> dict[str, str]:
    lookup = {
        **{str(row["rel_path"]): str(row["half"]) for row in noise_rows},
        **{str(row["rel_path"]): str(row["half"]) for row in rir_rows},
    }
    observed = {
        lookup[str(path)]
        for path in frame["resource_path"].astype(str)
        if str(path)
    }
    if observed != {"selection"}:
        raise ValueError(f"D64 recipe resources must all use the selection half, got {observed}")
    return {"recorded_noise_10db": "selection", "reverberation": "selection"}


def compare_recipes(
    old: pd.DataFrame,
    new: pd.DataFrame,
    *,
    noise_rows: list[dict[str, Any]],
    rir_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Prove D64 invariants and return the complete old/new hash delta."""
    if tuple(old.columns) != RECIPE_COLUMNS or tuple(new.columns) != RECIPE_COLUMNS:
        raise ValueError("old and new recipes must use the frozen recipe columns")
    old = old.astype(str)
    new = new.astype(str)
    if len(old) != len(new):
        raise ValueError(f"row count changed from {len(old)} to {len(new)}")
    old_keys = list(old.loc[:, KEY_COLUMNS].itertuples(index=False, name=None))
    new_keys = list(new.loc[:, KEY_COLUMNS].itertuples(index=False, name=None))
    if old_keys != new_keys or len(set(old_keys)) != len(old_keys):
        raise ValueError("source/condition row set or ordering changed")
    unequal = old.loc[:, INVARIANT_COLUMNS].ne(new.loc[:, INVARIANT_COLUMNS])
    if unequal.any().any():
        row_index, column_index = unequal.to_numpy().nonzero()
        row = int(row_index[0])
        column = INVARIANT_COLUMNS[int(column_index[0])]
        raise ValueError(
            f"D64 invariant changed at {old_keys[row][0]}/{old_keys[row][1]}: {column}"
        )
    halves = _resource_halves(new, noise_rows, rir_rows)
    changed = old[HASH_COLUMN] != new[HASH_COLUMN]
    changes = [
        {
            "source_path": str(old.iloc[index]["source_path"]),
            "condition": str(old.iloc[index]["condition"]),
            "old_waveform_float32_sha256": str(old.iloc[index][HASH_COLUMN]),
            "new_waveform_float32_sha256": str(new.iloc[index][HASH_COLUMN]),
        }
        for index in old.index[changed]
    ]
    changed_by_condition = Counter(item["condition"] for item in changes)
    return {
        "schema_version": 1,
        "result": "passed",
        "rows_identical": True,
        "sources_identical": True,
        "conditions_identical": True,
        "parameters_identical": True,
        "seeds_identical": True,
        "resources_identical": True,
        "resource_halves_identical": True,
        "resource_halves": halves,
        "row_count": len(old),
        "source_count": int(old["source_path"].nunique()),
        "changed_waveform_hash_count": len(changes),
        "unchanged_waveform_hash_count": int((~changed).sum()),
        "changed_waveform_hashes_by_condition": dict(sorted(changed_by_condition.items())),
        "changed_waveform_hashes": changes,
    }


def regenerate(
    for_root: Path,
    f2_manifest_path: Path,
    musan_noise_root: Path,
    rirs_root: Path,
    out: Path,
    colab_runtime_version: str,
) -> dict[str, Any]:
    """Run D64 from one clean reviewed commit and retain old/new recipe evidence."""
    versions = require_versions()
    version = code_version()
    out = Path(out)
    check_outside_repo(out, "D64 recipe refreeze")
    if out.exists():
        raise ValueError(f"D64 output already exists: {out}")
    repo = Path(__file__).resolve().parents[2]
    old_recipe_path, old_report_path, old, old_report = load_committed_old(repo)
    config = load_augmentation_config(repo / "configs" / "augmentation.yaml")
    if not config.corpus_pins_complete or not config.ffmpeg_pin_complete:
        raise ValueError("V1b corpus and FFmpeg pins must be complete")
    codec = FFmpegCodecBackend(config.ffmpeg, config.sample_rate)
    codec.verify_pin()
    runtime = official_runtime_record(codec, colab_runtime_version)
    manifest_dir = repo / "data" / "manifests" / "augmentation"
    _validate_inventory_hashes(manifest_dir, old_report)
    noise_rows = _records(manifest_dir / "musan_inventory.csv", "selection")
    rir_rows = _records(manifest_dir / "rirs_noises_inventory.csv", "selection")
    f2 = load_f2_manifest(Path(f2_manifest_path))
    sources = f2.loc[f2["split"] == "val-A"].sort_values("path", kind="stable")
    old_by_key = old.set_index(list(KEY_COLUMNS), drop=False)
    rows: list[dict[str, Any]] = []
    options = {item["id"]: item for item in config.codec_options}
    for _, source_row in sources.iterrows():
        source_path = str(source_row["path"])
        source, sample_rate = read_verified_audio(Path(for_root), source_row)
        if sample_rate != 16000 or source.ndim != 1 or source.size != 32000:
            raise ValueError(f"{source_path}: expected 16 kHz, 32,000-sample mono source")
        for condition in CONDITIONS:
            if (source_path, condition) not in old_by_key.index:
                raise ValueError(f"old recipe is missing {source_path}/{condition}")
            generated = apply_condition(
                source,
                source_path,
                condition,
                noise_inventory=noise_rows,
                rir_inventory=rir_rows,
                noise_root=Path(musan_noise_root),
                rirs_root=Path(rirs_root),
                codec_backend=codec,
                codec_options=options,
            )
            row = {
                "source_path": source_path,
                "label": str(source_row["label"]),
                "subgroup": str(source_row["subgroup"]),
                "group": str(source_row["group"]),
                "source_file_sha256": str(source_row["hash"]),
                "condition": condition,
                "condition_seed": str(generated["condition_seed"]),
                "resource_path": str(generated["resource_path"]),
                "resource_file_sha256": str(generated["resource_file_sha256"]),
                "resource_channel": str(generated["resource_channel"]),
                "parameters_json": json.dumps(
                    generated["parameters"], sort_keys=True, separators=(",", ":")
                ),
                "sample_rate_hz": str(sample_rate),
                "samples": str(source.size),
                HASH_COLUMN: waveform_sha256(generated["output"]),
            }
            rows.append(row)
    new = pd.DataFrame(rows, columns=RECIPE_COLUMNS)
    comparison = compare_recipes(old, new, noise_rows=noise_rows, rir_rows=rir_rows)
    # Every fail-closed check above completes before the output directory or any output file exists.
    out.mkdir(parents=True)
    old_copy = out / "selection_set_s.old.csv"
    new_path = out / "selection_set_s.csv"
    old_report_copy = out / "selection_set_s_report.old.json"
    new_report_path = out / "selection_set_s_report.json"
    comparison_path = out / "selection_recipe_refreeze_comparison.json"
    old_copy.write_bytes(Path(old_recipe_path).read_bytes())
    old_report_copy.write_bytes(old_report_path.read_bytes())
    _write_csv(new, new_path)
    new_report = make_new_report(
        old_report,
        recipe_sha256=sha256_file(new_path),
        semantic_sha256=_semantic_sha256(new),
        commit=version["git_commit"],
        versions=versions,
        runtime=runtime,
    )
    new_report_path.write_text(json.dumps(new_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    comparison.update({
        "decision": "D64",
        "code_commit": version["git_commit"],
        "library_versions": versions,
        "runtime": runtime,
        "old_recipe_sha256": sha256_file(old_copy),
        "new_recipe_sha256": sha256_file(new_path),
        "old_recipe_semantic_sha256": _semantic_sha256(old),
        "new_recipe_semantic_sha256": _semantic_sha256(new),
        "old_report_sha256": sha256_file(old_report_copy),
        "new_report_sha256": sha256_file(new_report_path),
    })
    comparison_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return comparison


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--for-root", required=True, type=Path)
    parser.add_argument("--f2-manifest", required=True, type=Path)
    parser.add_argument("--musan-noise-root", required=True, type=Path)
    parser.add_argument("--rirs-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--colab-runtime-version", required=True)
    args = parser.parse_args(argv)
    comparison = regenerate(
        args.for_root,
        args.f2_manifest,
        args.musan_noise_root,
        args.rirs_root,
        args.out,
        args.colab_runtime_version,
    )
    print(json.dumps({key: value for key, value in comparison.items() if key != "changed_waveform_hashes"}, indent=2))


if __name__ == "__main__":
    main()
