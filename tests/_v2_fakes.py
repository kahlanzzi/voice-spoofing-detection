"""Synthetic fakes for the model v2 training tests (Track V, PR V2b-1). Tests only; no real data.

Module-level classes, so a dataset holding them pickles for spawn DataLoader workers.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from src.augmentation import AugmentationResources, load_augmentation_config
from src.features.logmel_cache import build_logmel_cache
from src.features.raw_train_cache import build_raw_train_cache
from src.preprocessing.features import log_mel_spectrogram
from src.preprocessing.profile import load_profile
from src.preprocessing.validation import read_verified_audio
from src.preprocessing.waveform import preprocess_waveform

REPO = Path(__file__).resolve().parents[1]
PROFILE_PATH = REPO / "configs" / "preprocessing.yaml"
V2_CONFIG_PATH = REPO / "configs" / "v2_candidates.yaml"
AUG_CONFIG_PATH = REPO / "configs" / "augmentation.yaml"
COMMIT = "c" * 40
SIZES = {"train": 24, "val-A": 10}
S_COUNTS = {"REAL": 5, "FAKE": 5}


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FakeCodec:
    """Deterministic codec stand-in: a fixed per-option gain and a 2-tap smoothing."""

    GAINS = {"mp3_24": 0.5, "mp3_48": 0.55, "mp3_96": 0.6, "opus_8": 0.65, "opus_16": 0.7,
             "opus_24": 0.75, "narrowband_mulaw": 0.8}

    def __init__(self):
        self.verified = 0

    def verify_pin(self):
        self.verified += 1

    def round_trip(self, waveform, option):
        x = np.asarray(waveform, dtype=np.float32)
        y = np.convolve(x, np.array([0.5, 0.5], np.float32), mode="same")
        return (y * np.float32(self.GAINS[option["id"]])).astype(np.float32)


def fake_resources(seed: int = 7) -> AugmentationResources:
    rng = np.random.default_rng(seed)
    noise = (0.1 * rng.standard_normal(20000)).astype(np.float32)
    rir = np.array([1.0, 0.4, 0.2, 0.1, 0.05], dtype=np.float32)
    return AugmentationResources(noise_waveforms=(noise,), rir_waveforms=(rir,))


def write_dataset(base: Path, sizes=SIZES, seed: int = 2026, rate: int = 16000, channels: int = 1,
                  zero_clips=()):
    """Synthetic FoR-like WAVs (random lengths 1.5-2.5 s) and an F2-like manifest; return paths."""
    rng = np.random.default_rng(seed)
    root = base / "for"
    rows = []
    specs = [(split, i % 2) for split, n in sizes.items() for i in range(n)]
    for i, (split, spoof) in enumerate(specs):
        t = np.arange(int(rate * rng.uniform(1.5, 2.5))) / rate
        tone = np.sin(2 * np.pi * rng.uniform(150, 900) * t)
        mix = (0.6 if spoof else 0.1) + rng.uniform(-0.05, 0.05)
        data = (0.3 * ((1 - mix) * tone + mix * rng.standard_normal(len(t)))).astype(np.float32)
        if i in zero_clips:
            data = np.zeros_like(data)  # digital silence
        if channels == 2:
            data = np.stack([data, data], axis=1)
        rel = f"{split}/{'FAKE' if spoof else 'REAL'}/{i:02d}.wav"
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, data, rate, subtype="PCM_16")
        rows.append({"path": rel, "label": "FAKE" if spoof else "REAL",
                     "subgroup": "FAKE_other" if spoof else "REAL", "group": f"g{i // 2}",
                     "split": split, "hash": sha(path)})
    manifest = pd.DataFrame(rows, dtype=str)
    manifest_path = base / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return root, manifest, manifest_path


def build_caches(base: Path, root, manifest, manifest_path, profile):
    """Raw train cache, Stage 8-like train and val-A log-Mel caches."""
    f2 = sha(manifest_path)
    raw_dir = base / "raw"
    build_raw_train_cache(root, manifest, "train", raw_dir, f2_manifest_sha256=f2, code_commit=COMMIT)
    lm = base / "logmel"
    for split in ("train", "val-A"):
        build_logmel_cache(root, manifest, profile, split, lm / split, f2_manifest_sha256=f2,
                           profile_config_sha256=sha(PROFILE_PATH), code_commit=COMMIT)
    return raw_dir, lm


def selection_arrays(root, manifest, profile) -> dict:
    """The five S conditions from the synthetic val-A clips (stand-in degradations, then F3)."""
    from src.training.selection_set import CONDITIONS

    rows = manifest[manifest.split == "val-A"].sort_values("path", kind="stable")
    decoded = [read_verified_audio(root, row)[0] for _, row in rows.iterrows()]
    rng = np.random.default_rng(99)
    noise = [(0.02 * rng.standard_normal(x.size)).astype(np.float32) for x in decoded]
    degrade = {
        "clean": lambda x, i: x,
        "recorded_noise_10db": lambda x, i: x + noise[i],
        "reverberation": lambda x, i: np.convolve(x, [1.0, 0.0, 0.3], mode="same").astype(np.float32),
        "opus_16": lambda x, i: (0.7 * x).astype(np.float32),
        "narrowband_mulaw": lambda x, i: np.convolve(x, [0.5, 0.5], mode="same").astype(np.float32),
    }
    labels = rows["label"].map({"FAKE": 1, "REAL": 0}).to_numpy(dtype=np.int8)
    paths = np.array(rows["path"].tolist())
    groups = np.array(rows["group"].tolist())
    out = {}
    for name in CONDITIONS:
        waves = np.stack([preprocess_waveform(degrade[name](x, i), 16000, profile).waveform
                          for i, x in enumerate(decoded)]).astype(np.float32)
        log_mel = np.stack([log_mel_spectrogram(w, profile) for w in waves]).astype(np.float32)
        out[name] = {"waveforms": waves, "log_mel": log_mel, "labels": labels.copy(),
                     "paths": paths.copy(), "groups": groups.copy()}
    return out


def manifest_fields(profile, f2_sha: str, aug_sha: str) -> dict:
    from src.training.selection_set import CONDITIONS

    return {
        "conditions": [{"name": c, "parameters": {"stand_in": True, "index": i}}
                       for i, c in enumerate(CONDITIONS)],
        "f2_manifest_sha256": f2_sha,
        "profile_fingerprint": profile.fingerprint(),
        "profile_config_sha256": sha(PROFILE_PATH),
        "augmentation_config_sha256": aug_sha,
        "code_commit": COMMIT,
        "runtime": {"colab_runtime": "synthetic", "python": "3.x"},
    }


def write_s(out_dir: Path, arrays, profile, f2_sha: str, aug_sha: str, **kw) -> str:
    from src.training.selection_set import write_selection_set

    return write_selection_set(out_dir, arrays, manifest_fields(profile, f2_sha, aug_sha),
                               profile=profile, **kw)


NULL_BATCH_SIZES = {"cnn-aug": None, "crnn": None, "aasist": None}  # the file before V2b-2


def v2_config_copy(path: Path, sizes=None) -> Path:
    """configs/v2_candidates.yaml with the given batch sizes (default 8 each, so the tiny tests stay
    small; the committed file holds the D49 benchmark values). A None size writes null."""
    sizes = sizes or {"cnn-aug": 8, "crnn": 8, "aasist": 8}
    lines, order, replaced = [], iter(["cnn-aug", "crnn", "aasist"]), 0
    for line in V2_CONFIG_PATH.read_text(encoding="utf-8").splitlines():
        match = re.match(r"(\s+batch_size: )(\S+)", line)
        if match:
            size = sizes[next(order)]
            value = "null" if size is None else str(size)
            line = match.group(1) + value + line[match.end():]
            replaced += 1
        lines.append(line)
    assert replaced == 3, f"expected 3 batch_size lines in {V2_CONFIG_PATH}, found {replaced}"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def aug_config_copy(path: Path, *, pinned: bool = True) -> Path:
    """Copy the real pins, or explicitly remove every V1b pin for refusal tests."""
    data = yaml.safe_load(AUG_CONFIG_PATH.read_text(encoding="utf-8"))
    if not pinned:
        for corpus in ("noise", "rir"):
            data["corpora"][corpus]["archive_sha256"] = None
            data["corpora"][corpus]["license_sha256"] = None
        data["ffmpeg"]["expected_version"] = None
        data["ffmpeg"]["expected_codec_libraries"] = {}
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    loaded = load_augmentation_config(path)
    assert loaded.corpus_pins_complete is pinned
    assert loaded.ffmpeg_pin_complete is pinned
    return path


def profile():
    return load_profile(PROFILE_PATH)
