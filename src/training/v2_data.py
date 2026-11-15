"""Model v2 training data: per-item seed, transform, dataset, sampler and collate (D49, D50, D61 (5)).

Order of one training item (D50 (1)): raw train waveform (src.features.raw_train_cache) ->
Augmenter.augment(split="train", label, seed) -> unchanged F3 preprocess_waveform -> the candidate
input: aasist takes the F3 waveform (32000,); cnn-aug and crnn take log_mel_spectrogram, then
apply_bin_stats with the clean-train statistics (D61 (5)).

Reproducibility: the augmentation of a clip depends only on (run seed, epoch, clip index). The
sampler yields (epoch, clip_index) pairs, one torch.randperm per epoch from a generator seeded with
the run seed (as train_one), and the dataset derives the item seed itself with item_seed. Nothing
reads global NumPy / Python / torch RNG state, worker ids or timing, so the batches are the same for
any num_workers (fork on Colab, spawn on Windows) and the same item regenerated alone is identical.

The dataset must pickle for spawn workers. PreprocessingProfile and AugmentationConfig hold
MappingProxyType fields, which do not pickle; __getstate__ stores them as plain dicts and
__setstate__ rebuilds the same frozen objects (src/preprocessing and src/augmentation unchanged).
"""
from __future__ import annotations

import dataclasses
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from src.augmentation import Augmenter
from src.augmentation.config import CODEC_OPTIONS, FAMILIES
from src.preprocessing.features import log_mel_spectrogram
from src.preprocessing.waveform import preprocess_waveform
from src.training.normalization import apply_bin_stats

SAMPLE_RATE = 16000  # FoR train clips (checked by the raw cache)
INPUT_KINDS = ("logmel", "waveform")
LABEL_NAMES = {1: "FAKE", 0: "REAL"}
COUNT_FAMILIES = ("clean", *FAMILIES)  # clean, recorded_noise, reverberation, codec, gain, rawboost
CODEC_OPTION_IDS = tuple(option_id for option_id, _kind, _rate in CODEC_OPTIONS)


def _nonneg_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise ValueError(f"{name} must be an int >= 0, got {value!r}")
    return int(value)


def item_seed(aug_seed: int, run_seed: int, epoch: int, clip_index: int) -> int:
    """The augmentation seed of one item: a function of its four inputs only."""
    entropy = [_nonneg_int(n, v) for n, v in (("aug_seed", aug_seed), ("run_seed", run_seed),
                                               ("epoch", epoch), ("clip_index", clip_index))]
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])


def transform_item(raw_waveform, label_code, seed, augmenter, profile, input_kind, mean, std):
    """One training input from one raw clip; return (x float32, family, codec option id or None).

    x is (32000,) for input_kind "waveform" and (1, 64, 197) standardized for "logmel".
    """
    if input_kind not in INPUT_KINDS:
        raise ValueError(f"input_kind must be one of {INPUT_KINDS}, got {input_kind!r}")
    if label_code not in LABEL_NAMES:
        raise ValueError(f"label_code must be 0 or 1, got {label_code!r}")
    if input_kind == "waveform" and (mean is not None or std is not None):
        raise ValueError("the waveform input takes no mean/std")
    if input_kind == "logmel" and (mean is None or std is None):
        raise ValueError("the log-Mel input needs the clean-train mean and std (D61 (5))")
    result = augmenter.augment(raw_waveform, split="train", label=LABEL_NAMES[int(label_code)],
                               seed=seed)
    option = result.parameters["option"] if result.family == "codec" else None
    wave = preprocess_waveform(result.waveform, SAMPLE_RATE, profile).waveform
    if input_kind == "waveform":
        x = np.asarray(wave, dtype=np.float32)
    else:
        x = apply_bin_stats(log_mel_spectrogram(wave, profile)[None], mean, std)[0]
    return x, result.family, option


# --- pickling of frozen configs ------------------------------------------------------------------

class _Proxy(dict):
    """Marker for a MappingProxyType turned into a plain dict for pickling."""


def _thaw(value):
    if isinstance(value, MappingProxyType):
        return _Proxy({k: _thaw(v) for k, v in value.items()})
    if isinstance(value, dict):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_thaw(v) for v in value)
    return value


def _refreeze(value):
    if isinstance(value, _Proxy):
        return MappingProxyType({k: _refreeze(v) for k, v in value.items()})
    if isinstance(value, dict):
        return {k: _refreeze(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_refreeze(v) for v in value)
    return value


def _freeze_dataclass(obj) -> tuple[type, dict]:
    return type(obj), {f.name: _thaw(getattr(obj, f.name)) for f in dataclasses.fields(obj)}


def _rebuild_dataclass(state: tuple[type, dict]):
    cls, fields = state
    return cls(**{name: _refreeze(value) for name, value in fields.items()})


class V2TrainDataset(torch.utils.data.Dataset):
    """Item (epoch, clip_index) -> (x tensor, label, family, codec option id or None)."""

    def __init__(self, raw_cache, augmenter: Augmenter, profile, input_kind: str, run_seed: int,
                 mean=None, std=None) -> None:
        if input_kind not in INPUT_KINDS:
            raise ValueError(f"input_kind must be one of {INPUT_KINDS}, got {input_kind!r}")
        self.raw_cache = raw_cache
        self.augmenter = augmenter
        self.profile = profile
        self.input_kind = input_kind
        self.run_seed = _nonneg_int("run_seed", run_seed)
        self.aug_seed = int(augmenter.config.seed)  # 2026, enforced by load_augmentation_config
        self.mean, self.std = mean, std

    def __len__(self) -> int:
        return len(self.raw_cache)

    def __getitem__(self, key):
        epoch, clip_index = key
        seed = item_seed(self.aug_seed, self.run_seed, epoch, clip_index)
        label = int(self.raw_cache.labels[clip_index])
        x, family, option = transform_item(self.raw_cache.clip(clip_index), label, seed,
                                           self.augmenter, self.profile, self.input_kind,
                                           self.mean, self.std)
        return torch.from_numpy(x), label, family, option

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        aug = state.pop("augmenter")
        state["_augmenter"] = (type(aug), _freeze_dataclass(aug.config), aug.resources,
                               aug.codec_backend)
        state["profile"] = _freeze_dataclass(state["profile"])
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        state = dict(state)
        cls, config, resources, codec = state.pop("_augmenter")
        state["augmenter"] = cls(_rebuild_dataclass(config), resources, codec)
        state["profile"] = _rebuild_dataclass(state["profile"])
        self.__dict__.update(state)


class EpochSampler(torch.utils.data.Sampler):
    """Each iteration is the next epoch: one randperm from a generator seeded with the run seed."""

    def __init__(self, n: int, run_seed: int) -> None:
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError(f"n must be an int >= 1, got {n!r}")
        self.n = n
        self.generator = torch.Generator().manual_seed(_nonneg_int("run_seed", run_seed))
        self.epoch = 0

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        self.epoch += 1
        epoch = self.epoch
        order = torch.randperm(self.n, generator=self.generator).tolist()
        return iter([(epoch, i) for i in order])


def collate(items):
    """Stack a batch; return (x, y float32, families tuple, codec options tuple)."""
    xs, labels, families, options = zip(*items)
    return (torch.stack(xs), torch.tensor(labels, dtype=torch.float32), tuple(families),
            tuple(options))


def make_loader(dataset: V2TrainDataset, sampler: EpochSampler, batch_size: int,
                num_workers: int) -> torch.utils.data.DataLoader:
    """The training DataLoader: sampler order, in-order batches, last batch may be smaller."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size must be an int >= 1, got {batch_size!r}")
    workers = _nonneg_int("num_workers", num_workers)
    # A private generator: each new DataLoader iterator draws its worker base seed from it, not from
    # the global torch RNG, so weight init and dropout see the same global RNG stream whatever
    # num_workers is (persistent workers draw once, num_workers=0 draws every epoch). The base seed
    # only seeds worker-local RNG state, which nothing here uses.
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate,
        num_workers=workers, drop_last=False, persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(0),
    )


def empty_counts() -> dict[str, int]:
    """Per-epoch counters: family_<family> x 6 and codec_<option> x 7, all zero."""
    return {**{f"family_{f}": 0 for f in COUNT_FAMILIES},
            **{f"codec_{o}": 0 for o in CODEC_OPTION_IDS}}


def add_counts(counts: dict[str, int], families, options) -> None:
    """Add one batch's families and codec options to counts; unknown values raise."""
    for family, option in zip(families, options):
        key = f"family_{family}"
        if key not in counts:
            raise ValueError(f"unknown augmentation family {family!r}")
        counts[key] += 1
        if family == "codec":
            if f"codec_{option}" not in counts:
                raise ValueError(f"unknown codec option {option!r}")
            counts[f"codec_{option}"] += 1
        elif option is not None:
            raise ValueError(f"family {family!r} must not carry a codec option, got {option!r}")


def check_clean_transform(raw_cache, stage8_train_logmel, indices, profile) -> dict[str, Any]:
    """The un-augmented path raw -> F3 -> log-Mel equals the Stage 8 train cache rows byte for byte.

    stage8_train_logmel: the load_logmel_cache(train) result (features, labels, paths, meta). Paths
    and labels must be equal for the whole split; the log-Mel is compared for `indices`.
    """
    features, labels, paths = stage8_train_logmel[:3]
    if not np.array_equal(paths, raw_cache.paths) or paths.dtype.kind != "U":
        raise ValueError("clean transform check: paths differ between the raw cache and Stage 8")
    if labels.dtype != raw_cache.labels.dtype or not np.array_equal(labels, raw_cache.labels):
        raise ValueError("clean transform check: labels differ between the raw cache and Stage 8")
    checked = []
    for i in indices:
        i = _nonneg_int("index", i)
        wave = preprocess_waveform(raw_cache.clip(i), SAMPLE_RATE, profile).waveform
        log_mel = log_mel_spectrogram(wave, profile)
        if log_mel.dtype != features.dtype or log_mel.tobytes() != features[i].tobytes():
            raise ValueError(f"clean transform check: row {i} ({paths[i]}) differs from the Stage 8 "
                             f"train cache")
        checked.append(i)
    return {"n_checked": len(checked), "indices": checked, "n_rows": int(len(paths))}


__all__ = [
    "CODEC_OPTION_IDS", "COUNT_FAMILIES", "EpochSampler", "V2TrainDataset", "add_counts",
    "check_clean_transform", "collate", "empty_counts", "item_seed", "make_loader",
    "transform_item",
]
