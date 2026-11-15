"""Model v2 training data (D50 (1), D61 (5)): item seeds, transform, worker independence. Synthetic only."""
from __future__ import annotations

import pickle

import numpy as np
import pytest

from tests._torch import require_torch

torch = require_torch()

from src.augmentation import Augmenter, load_augmentation_config  # noqa: E402
from src.features.logmel_cache import load_logmel_cache  # noqa: E402
from src.features.raw_train_cache import load_raw_train_cache  # noqa: E402
from src.preprocessing.features import extract_features  # noqa: E402
from src.training.v2_data import (  # noqa: E402
    CODEC_OPTION_IDS,
    COUNT_FAMILIES,
    EpochSampler,
    V2TrainDataset,
    add_counts,
    check_clean_transform,
    empty_counts,
    item_seed,
    make_loader,
    transform_item,
)
from tests._v2_fakes import (  # noqa: E402
    AUG_CONFIG_PATH,
    FakeCodec,
    build_caches,
    fake_resources,
    profile as load_test_profile,
    sha,
    write_dataset,
)

ZERO, ONE = np.zeros(64, np.float32), np.ones(64, np.float32)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    base = tmp_path_factory.mktemp("v2data")
    prof = load_test_profile()
    root, manifest, manifest_path = write_dataset(base)
    raw_dir, lm = build_caches(base, root, manifest, manifest_path, prof)
    f2 = sha(manifest_path)
    raw = load_raw_train_cache(raw_dir, expected_f2_manifest_sha256=f2)
    stage8 = load_logmel_cache(lm / "train", "train", prof, expected_f2_manifest_sha256=f2)
    augmenter = Augmenter(load_augmentation_config(AUG_CONFIG_PATH), fake_resources(), FakeCodec())
    return {"profile": prof, "raw": raw, "stage8": stage8, "augmenter": augmenter, "root": root,
            "manifest": manifest}


def test_item_seed_rule():
    assert item_seed(2026, 0, 1, 5) == item_seed(2026, 0, 1, 5)
    expected = int(np.random.SeedSequence([2026, 0, 1, 5]).generate_state(1)[0])
    assert item_seed(2026, 0, 1, 5) == expected
    seeds = {item_seed(2026, r, e, i) for r in (0, 1) for e in (1, 2) for i in (0, 1)}
    assert len(seeds) == 8  # run seed, epoch and clip index each change the seed
    for bad in (-1, True, 1.0, "1"):
        with pytest.raises(ValueError):
            item_seed(2026, bad, 1, 0)


def _seed_with_family(augmenter, raw, family):
    for seed in range(500):
        if augmenter.augment(raw, split="train", label="REAL", seed=seed).family == family:
            return seed
    raise AssertionError(family)


def test_clean_path_equals_extract_features(env):
    raw = env["raw"].clip(3)
    seed = _seed_with_family(env["augmenter"], raw, "clean")
    x, family, option = transform_item(raw, 1, seed, env["augmenter"], env["profile"], "logmel", ZERO, ONE)
    assert (family, option) == ("clean", None)
    expected = extract_features(raw, 16000, env["profile"])[1]
    assert x.dtype == np.float32 and x.shape == (1, 64, 197) and x.tobytes() == expected.tobytes()
    w, _, _ = transform_item(raw, 1, seed, env["augmenter"], env["profile"], "waveform", None, None)
    assert w.shape == (32000,) and w.tobytes() == extract_features(raw, 16000, env["profile"])[0].waveform.tobytes()


def test_codec_option_is_reported_and_standardization_applied(env):
    raw = env["raw"].clip(0)
    seed = _seed_with_family(env["augmenter"], raw, "codec")
    mean, std = np.full(64, 2.0, np.float32), np.full(64, 4.0, np.float32)
    x, family, option = transform_item(raw, 0, seed, env["augmenter"], env["profile"], "logmel", mean, std)
    plain, _, _ = transform_item(raw, 0, seed, env["augmenter"], env["profile"], "logmel", ZERO, ONE)
    assert family == "codec" and option in CODEC_OPTION_IDS
    assert np.allclose(x, (plain - 2.0) / 4.0)


def test_transform_refusals(env):
    raw, aug, prof = env["raw"].clip(0), env["augmenter"], env["profile"]
    with pytest.raises(ValueError, match="no mean/std"):
        transform_item(raw, 0, 1, aug, prof, "waveform", ZERO, ONE)
    with pytest.raises(ValueError, match="clean-train mean and std"):
        transform_item(raw, 0, 1, aug, prof, "logmel", None, None)
    with pytest.raises(ValueError, match="label_code"):
        transform_item(raw, 2, 1, aug, prof, "logmel", ZERO, ONE)
    with pytest.raises(ValueError, match="input_kind"):
        transform_item(raw, 0, 1, aug, prof, "mfcc", None, None)


def test_check_clean_transform_known_answer(env):
    out = check_clean_transform(env["raw"], env["stage8"], range(24), env["profile"])
    assert out == {"n_checked": 24, "indices": list(range(24)), "n_rows": 24}
    features, labels, paths, meta = env["stage8"]
    changed = features.copy()
    changed[5, 0, 1, 2] += np.float32(1e-3)
    with pytest.raises(ValueError, match="row 5"):
        check_clean_transform(env["raw"], (changed, labels, paths, meta), [5], env["profile"])
    with pytest.raises(ValueError, match="labels differ"):
        check_clean_transform(env["raw"], (features, 1 - labels, paths, meta), [0], env["profile"])
    with pytest.raises(ValueError, match="paths differ"):
        check_clean_transform(env["raw"], (features, labels, paths[::-1], meta), [0], env["profile"])


def _dataset(env, kind="logmel", run_seed=0):
    mean, std = (ZERO, ONE) if kind == "logmel" else (None, None)
    return V2TrainDataset(env["raw"], env["augmenter"], env["profile"], kind, run_seed, mean, std)


def _epoch(env, batch_size, workers, epochs=1, run_seed=0):
    ds = _dataset(env, run_seed=run_seed)
    sampler = EpochSampler(len(ds), run_seed)
    loader = make_loader(ds, sampler, batch_size, workers)
    out = []
    for _ in range(epochs):
        xs, ys, fams, opts = [], [], [], []
        for x, y, f, o in loader:
            xs.append(x)
            ys.append(y)
            fams += f
            opts += o
        out.append((torch.cat(xs), torch.cat(ys), fams, opts))
    return out


def test_workers_and_batch_size_do_not_change_items(env):
    reference = _epoch(env, 8, 0, epochs=2)
    other = _epoch(env, 5, 2, epochs=2)
    for (x0, y0, f0, o0), (x1, y1, f1, o1) in zip(reference, other):
        assert x0.numpy().tobytes() == x1.numpy().tobytes()
        assert torch.equal(y0, y1) and f0 == f1 and o0 == o1
    assert reference[0][0].numpy().tobytes() != reference[1][0].numpy().tobytes()  # new epoch


def test_item_alone_equals_loader_item(env):
    ds = _dataset(env)
    sampler = EpochSampler(len(ds), 0)
    order = [i for _, i in sampler]
    batch = next(iter(make_loader(ds, EpochSampler(len(ds), 0), 4, 0)))
    for k in range(4):
        x, label, family, option = ds[(1, order[k])]
        assert x.numpy().tobytes() == batch[0][k].numpy().tobytes()
        assert (float(label), family, option) == (float(batch[1][k]), batch[2][k], batch[3][k])


def test_sampler_order_depends_on_run_seed_and_epoch():
    a, b = EpochSampler(24, 0), EpochSampler(24, 1)
    first_a, second_a = list(a), list(a)
    assert [e for e, _ in first_a] == [1] * 24 and [e for e, _ in second_a] == [2] * 24
    assert [i for _, i in first_a] != [i for _, i in second_a]
    assert [i for _, i in first_a] != [i for _, i in list(b)]
    gen = torch.Generator().manual_seed(0)
    assert [i for _, i in first_a] == torch.randperm(24, generator=gen).tolist()  # as train_one


def test_dataset_pickles_and_rebuilds_frozen_configs(env):
    ds = _dataset(env, "waveform", run_seed=3)
    again = pickle.loads(pickle.dumps(ds))
    assert again.profile == ds.profile and again.profile.fingerprint() == ds.profile.fingerprint()
    assert again.augmenter.config == ds.augmenter.config
    assert type(again.augmenter.config.training).__name__ == "mappingproxy"
    x0, *rest0 = ds[(2, 7)]
    x1, *rest1 = again[(2, 7)]
    assert x0.numpy().tobytes() == x1.numpy().tobytes() and rest0 == rest1


def test_counts_add_up(env):
    x, y, families, options = _epoch(env, 8, 0)[0]
    counts = empty_counts()
    add_counts(counts, families, options)
    assert list(counts) == [f"family_{f}" for f in COUNT_FAMILIES] + [f"codec_{o}" for o in CODEC_OPTION_IDS]
    assert sum(counts[f"family_{f}"] for f in COUNT_FAMILIES) == 24 == len(y)
    assert sum(counts[f"codec_{o}"] for o in CODEC_OPTION_IDS) == counts["family_codec"]
    with pytest.raises(ValueError, match="unknown augmentation family"):
        add_counts(counts, ["white_noise"], [None])
    with pytest.raises(ValueError, match="unknown codec option"):
        add_counts(counts, ["codec"], ["mp3_128"])
    with pytest.raises(ValueError, match="must not carry"):
        add_counts(counts, ["gain"], ["mp3_24"])
