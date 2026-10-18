"""FoR-only Stage 6 validation: shortcut re-screen and pipeline invariance.

The official run verifies every F2 sample hash before decoding it. It never accepts
or references the external dataset. Run only from a clean implementation commit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from src.audit.common import auc_separability, df_to_md

from .features import extract_features
from .profile import PreprocessingError, PreprocessingProfile, load_profile
from .waveform import preprocess_waveform, speech_mask


F2_MANIFEST_SHA256 = "badfe948fb7b833357783dd18be00f6bb36253cf8699dc4c4beb4df2b07706f8"
F2_ROWS = 17721
F2_COLUMNS = ("path", "label", "subgroup", "group", "split", "hash")
LABELS = {"REAL", "FAKE"}
SUBGROUPS = {"REAL", "FAKE_mp3", "FAKE_other"}
SPLITS = {"train", "val-A", "val-B", "test"}
FEATURES = ("rms_dbfs", "silence_frac", "bw99_hz", "padding_fraction")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def code_version() -> dict[str, Any]:
    """Refuse an official run unless Git identifies a clean code commit."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
            text=True, timeout=5,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True,
            text=True, timeout=5,
        ).stdout.strip())
    except (OSError, subprocess.SubprocessError) as exc:
        raise PreprocessingError(f"Cannot determine code version from Git: {exc}") from exc
    if not commit:
        raise PreprocessingError("Git returned an empty code version")
    if dirty:
        raise PreprocessingError("Official Stage 6 validation requires a clean Git worktree")
    return {"git_commit": commit, "git_dirty": False}


def _safe_relative(value: str) -> str:
    normalized = value.replace("\\", "/")
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath(value)
    if (
        not normalized.strip() or posix.is_absolute() or windows.drive or windows.root
        or ".." in posix.parts
    ):
        raise PreprocessingError(f"F2 manifest contains an unsafe path: {value!r}")
    return normalized


def load_f2_manifest(path: Path) -> pd.DataFrame:
    """Load the exact approved FoR F2 manifest and validate its frozen schema."""
    path = Path(path)
    if not path.is_file():
        raise PreprocessingError(f"F2 FoR manifest does not exist: {path}")
    actual = sha256_file(path)
    if actual != F2_MANIFEST_SHA256:
        raise PreprocessingError(
            f"F2 FoR manifest SHA-256 mismatch: expected {F2_MANIFEST_SHA256}, got {actual}"
        )
    try:
        manifest = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.ParserError) as exc:
        raise PreprocessingError(f"Cannot read F2 FoR manifest {path}: {exc}") from exc
    if tuple(manifest.columns) != F2_COLUMNS:
        raise PreprocessingError(f"F2 columns must be exactly {list(F2_COLUMNS)}")
    if len(manifest) != F2_ROWS:
        raise PreprocessingError(f"F2 manifest must contain {F2_ROWS} rows, got {len(manifest)}")
    if manifest["path"].duplicated().any():
        raise PreprocessingError("F2 manifest contains duplicate paths")
    manifest["path"] = manifest["path"].map(_safe_relative)
    if set(manifest["label"]) != LABELS:
        raise PreprocessingError(f"F2 labels must be exactly {sorted(LABELS)}")
    if set(manifest["subgroup"]) != SUBGROUPS:
        raise PreprocessingError(f"F2 subgroups must be exactly {sorted(SUBGROUPS)}")
    if set(manifest["split"]) != SPLITS:
        raise PreprocessingError(f"F2 splits must be exactly {sorted(SPLITS)}")
    if not manifest["hash"].str.fullmatch(r"[0-9a-f]{64}").all():
        raise PreprocessingError("F2 manifest contains an invalid sample SHA-256")
    if (manifest.loc[manifest.label == "REAL", "subgroup"] != "REAL").any():
        raise PreprocessingError("F2 REAL rows must use subgroup REAL")
    if manifest.groupby("group")["split"].nunique().gt(1).any():
        raise PreprocessingError("F2 manifest has a group crossing splits")
    return manifest


def resolve_audio(root: Path, relative: str) -> Path:
    root = Path(root).resolve()
    candidate = (root / _safe_relative(relative)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise PreprocessingError(f"Manifest path escapes the FoR root: {relative!r}") from None
    if not candidate.is_file():
        raise PreprocessingError(f"FoR sample is missing: {relative}")
    return candidate


def read_verified_audio(root: Path, row: pd.Series) -> tuple[np.ndarray, int]:
    path = resolve_audio(root, str(row["path"]))
    actual = sha256_file(path)
    if actual != row["hash"]:
        raise PreprocessingError(
            f"FoR sample SHA-256 mismatch for {row['path']}: expected {row['hash']}, got {actual}"
        )
    try:
        waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    except (OSError, RuntimeError) as exc:
        raise PreprocessingError(f"Cannot decode FoR sample {row['path']}: {exc}") from exc
    if np.asarray(waveform).size == 0:
        raise PreprocessingError(f"Decoded zero samples from {row['path']}")
    return np.asarray(waveform), int(sample_rate)


def shortcut_stats(waveform: np.ndarray, profile: PreprocessingProfile) -> dict[str, float]:
    data = np.asarray(waveform, dtype=np.float32)
    eps = 1.0e-12
    rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)) + eps)
    spec = np.abs(np.fft.rfft(data.astype(np.float64))) ** 2
    cumulative = np.cumsum(spec)
    bandwidth = (
        float(np.searchsorted(cumulative, 0.99 * cumulative[-1]) * profile.sample_rate / data.size)
        if cumulative[-1] > 0 else 0.0
    )
    return {
        "rms_dbfs": float(20.0 * np.log10(rms)),
        "silence_frac": float(1.0 - speech_mask(data, profile).mean()),
        "bw99_hz": bandwidth,
    }


def shortcut_screen(rows: pd.DataFrame, profile: PreprocessingProfile) -> pd.DataFrame:
    """Compute the pre-registered numeric shortcut screen by scope and subgroup."""
    output = []
    comparisons = {
        "FAKE_vs_REAL": {"REAL", "FAKE"},
        "FAKE_mp3_vs_REAL": {"REAL", "FAKE_mp3"},
        "FAKE_other_vs_REAL": {"REAL", "FAKE_other"},
    }
    for scope, scoped in (("train", rows[rows.split == "train"]), ("all", rows)):
        for comparison, values in comparisons.items():
            if comparison == "FAKE_vs_REAL":
                part = scoped.copy()
                labels = part["label"]
            else:
                part = scoped[scoped.subgroup.isin(values)].copy()
                labels = part["subgroup"].replace({"FAKE_mp3": "FAKE", "FAKE_other": "FAKE"})
            for feature in FEATURES:
                auc = auc_separability(part[feature], labels)
                high = float(profile.validation["shortcut_auc_high"])
                flag = float(profile.validation["shortcut_auc_flag"])
                level = "HIGH" if auc >= high else "MODERATE" if auc >= flag else "LOW"
                output.append({
                    "scope": scope, "comparison": comparison, "feature": feature,
                    "rows": len(part), "auc_separability": auc, "flag": level,
                })
    return pd.DataFrame(output)


def padding_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize post-trim zero-padding by split scope, class and F2 subgroup."""
    output = []
    for scope, scoped in (("train", rows[rows.split == "train"]), ("all", rows)):
        for grouping, values in (("label", sorted(LABELS)), ("subgroup", sorted(SUBGROUPS))):
            for value in values:
                part = scoped[scoped[grouping] == value]
                if part.empty:
                    raise PreprocessingError(
                        f"Cannot summarize padding: {scope} scope has no {grouping}={value} rows"
                    )
                samples = part["padding_samples"].to_numpy(dtype=np.int64)
                fractions = part["padding_fraction"].to_numpy(dtype=np.float64)
                output.append({
                    "scope": scope,
                    "grouping": grouping,
                    "value": value,
                    "rows": int(len(part)),
                    "mean_padding_samples": float(samples.mean()),
                    "median_padding_samples": float(np.median(samples)),
                    "p95_padding_samples": float(np.quantile(samples, 0.95)),
                    "max_padding_samples": int(samples.max()),
                    "mean_padding_fraction": float(fractions.mean()),
                })
    return pd.DataFrame(output)


def relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = float(np.linalg.norm(expected))
    difference = float(np.linalg.norm(actual - expected))
    return difference / denominator if denominator > 0 else (0.0 if difference == 0 else float("inf"))


def cosine_similarity(actual: np.ndarray, expected: np.ndarray) -> float:
    a, b = actual.ravel().astype(np.float64), expected.ravel().astype(np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.dot(a, b) / denominator)


def converted_stereo(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    data = np.asarray(waveform, dtype=np.float32)
    mono = data.mean(axis=1) if data.ndim == 2 else data
    converted = librosa.resample(
        mono, orig_sr=source_rate, target_sr=target_rate, res_type="soxr_hq",
        fix=True, scale=False,
    ).astype(np.float32)
    return np.column_stack((converted, converted))


def invariance_rows(
    root: Path,
    manifest: pd.DataFrame,
    profile: PreprocessingProfile,
) -> pd.DataFrame:
    """Run D18 on a deterministic train-only sample (lowest sample hash per subgroup)."""
    count = int(profile.validation["invariance_sample_per_subgroup"])
    selected = (
        manifest[manifest.split == "train"].sort_values(["subgroup", "hash", "path"], kind="stable")
        .groupby("subgroup", sort=True).head(count)
    )
    if selected.groupby("subgroup").size().to_dict() != {name: count for name in sorted(SUBGROUPS)}:
        raise PreprocessingError("F2 train split cannot supply the frozen invariance sample")
    results = []
    for _, row in selected.iterrows():
        waveform, sample_rate = read_verified_audio(root, row)
        _, base_mel, base_mfcc = extract_features(waveform, sample_rate, profile)
        for target_rate in profile.validation["invariance_sample_rates_hz"]:
            stereo = converted_stereo(waveform, sample_rate, int(target_rate))
            _, mel, mfcc = extract_features(stereo, int(target_rate), profile)
            for feature, actual, expected in (
                ("log_mel", mel, base_mel), ("mfcc", mfcc, base_mfcc)
            ):
                rel = relative_l2(actual, expected)
                cosine = cosine_similarity(actual, expected)
                results.append({
                    "path": row["path"], "subgroup": row["subgroup"],
                    "converted_rate_hz": int(target_rate), "feature": feature,
                    "relative_l2_error": rel, "cosine_similarity": cosine,
                    "pass": rel <= float(profile.validation["max_relative_l2_error"])
                    and cosine >= float(profile.validation["min_cosine_similarity"]),
                })
    return pd.DataFrame(results)


def _write_report(
    out: Path,
    report: dict[str, Any],
    screen: pd.DataFrame,
    padding: pd.DataFrame,
    invariance: pd.DataFrame,
) -> None:
    out.mkdir(parents=True, exist_ok=False)
    (out / "shortcut_screen.csv").write_text(screen.to_csv(index=False), encoding="utf-8", newline="")
    (out / "invariance.csv").write_text(invariance.to_csv(index=False), encoding="utf-8", newline="")
    (out / "profile_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    checks = report["checks"]
    md = [
        "# Stage 6 preprocessing profile validation",
        "",
        f"- Profile: `{report['profile']['version']}`",
        f"- Profile fingerprint: `{report['profile']['fingerprint']}`",
        f"- Code commit: `{report['code_version']['git_commit']}`",
        f"- Git dirty: `{str(report['code_version']['git_dirty']).lower()}`",
        f"- FoR rows processed: {report['for2sec_rows']}",
        "- Audio scope: FoR only; DEEP-VOICE not accessed",
        "",
        "## Checks",
        *[f"- {name}: {value}" for name, value in checks.items()],
        "",
        "## Shortcut re-screen",
        df_to_md(screen),
        "",
        "## Post-trim padding by class and subgroup",
        df_to_md(padding),
        "",
        "## Invariance summary",
        df_to_md(pd.DataFrame([report["invariance"]])),
        "",
        f"All checks pass: **{str(report['all_checks_pass']).lower()}**.",
    ]
    (out / "profile_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def run(root: Path, manifest_path: Path, profile_path: Path, out: Path) -> dict[str, Any]:
    """Execute the official FoR-only Stage 6 validation from a clean worktree."""
    version = code_version()  # before reading inputs or creating outputs
    profile = load_profile(profile_path)
    manifest = load_f2_manifest(manifest_path)
    out = Path(out)
    if out.exists():
        raise PreprocessingError(f"Output directory already exists: {out}")

    processed_rows = []
    for _, row in manifest.iterrows():
        waveform, sample_rate = read_verified_audio(root, row)
        prepared = preprocess_waveform(waveform, sample_rate, profile)
        processed_rows.append({
            "path": row["path"], "label": row["label"], "subgroup": row["subgroup"],
            "split": row["split"],
            "padding_samples": prepared.padding_samples,
            "padding_fraction": prepared.padding_samples / profile.samples,
            **shortcut_stats(prepared.waveform, profile),
        })
    processed = pd.DataFrame(processed_rows)
    screen = shortcut_screen(processed, profile)
    padding = padding_summary(processed)
    invariance = invariance_rows(root, manifest, profile)

    train_rms = screen[
        (screen.scope == "train") & (screen.comparison == "FAKE_vs_REAL")
        & (screen.feature == "rms_dbfs")
    ].iloc[0]
    loudness_rule_pass = (
        profile.loudness["mode"] == "speech_rms"
        or float(train_rms.auc_separability) < float(profile.loudness["separability_auc_threshold"])
    )
    padding_measurements_valid = bool(
        processed["padding_samples"].between(0, profile.samples).all()
        and processed["padding_samples"].map(lambda value: isinstance(value, (int, np.integer))).all()
        and np.isclose(
            processed["padding_fraction"],
            processed["padding_samples"] / profile.samples,
        ).all()
    )
    checks = {
        "f2_manifest_verified": True,
        "all_sample_hashes_verified": True,
        "all_rows_processed": len(processed) == F2_ROWS,
        "outputs_finite": bool(processed[list(FEATURES)].apply(np.isfinite).all().all()),
        "padding_measurements_valid": padding_measurements_valid,
        "loudness_decision_rule_pass": bool(loudness_rule_pass),
        "invariance_pass": bool(invariance["pass"].all()),
        "invariance_rows_complete": len(invariance) == 3 * 10 * 3 * 2,
    }
    report = {
        "freeze_id": "F3-candidate",
        "protocol_version": profile.protocol_version,
        "code_version": version,
        "f2": {
            "manifest_sha256": F2_MANIFEST_SHA256,
            "ledger_sha256": profile.f2_ledger_sha256,
        },
        "profile": {
            "version": profile.profile_version,
            "fingerprint": profile.fingerprint(),
            "config_sha256": sha256_file(Path(profile_path)),
            "settings": profile.canonical_dict(),
        },
        "for2sec_rows": len(processed),
        "audio_scope": "FoR only; DEEP-VOICE not accessed",
        "shortcut_flags": screen[screen.flag != "LOW"].to_dict(orient="records"),
        "padding_summary": padding.to_dict(orient="records"),
        "invariance": {
            "rows": len(invariance),
            "max_relative_l2_error": float(invariance.relative_l2_error.max()),
            "min_cosine_similarity": float(invariance.cosine_similarity.min()),
        },
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }
    _write_report(out, report, screen, padding, invariance)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Local extracted FoR root")
    parser.add_argument("--manifest", required=True, type=Path, help="Approved F2 FoR manifest")
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = run(args.root, args.manifest, args.profile, args.out)
    print(json.dumps({
        "all_checks_pass": report["all_checks_pass"],
        "rows": report["for2sec_rows"],
        "profile_fingerprint": report["profile"]["fingerprint"],
        "out": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
