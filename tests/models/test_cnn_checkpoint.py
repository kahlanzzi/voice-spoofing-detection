"""Non-pickle CNN checkpoint (D42): safetensors + strict JSON. Synthetic tensors only, seed 2026."""
from __future__ import annotations

import ast
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from tests._torch import require_torch

torch = require_torch()
from safetensors.torch import load_file, save_file  # noqa: E402

import src.models.cnn_checkpoint as ckpt_module  # noqa: E402
from src.models.cnn import INPUT_SHAPE, build_cnn  # noqa: E402
from src.models.cnn_checkpoint import (  # noqa: E402
    JSON_NAME,
    METADATA_KEYS,
    WEIGHTS_NAME,
    checkpoint_sha256,
    load_checkpoint,
    save_checkpoint,
)

CKPT_FILE = Path(__file__).resolve().parents[2] / "src" / "models" / "cnn_checkpoint.py"


def _meta(n_blocks: int = 3, **changes) -> dict:
    meta = {
        "schema_version": 1,
        "version": "cnn-test",
        "n_blocks": n_blocks,
        "channels": [16, 32, 64, 128][:n_blocks],
        "dropout": 0.3,
        "input_shape": [1, 64, 197],
        "learning_rate": 0.001,
        "seed": 0,
        "best_epoch": 7,
        "val_a_eer": 0.0125,
        "code_commit": "a" * 40,
        "environment": {"python": "3.13.15", "torch": "2.11.0+cpu"},
        "profile_fingerprint": "b" * 64,
        "f2_manifest_sha256": "c" * 64,
        "train_cache_npz_sha256": "d" * 64,
        "val_a_cache_npz_sha256": "e" * 64,
        "determinism": {"seed": 0, "cudnn_deterministic": True},
    }
    meta.update(changes)
    return meta


def _stats():
    rng = np.random.default_rng(2026)
    mean = rng.normal(-6.0, 2.0, 64).astype(np.float32)
    std = rng.uniform(0.5, 3.0, 64).astype(np.float32)
    return mean, std


def _model(n_blocks: int = 3):
    """A CNN with non-default BatchNorm running statistics, so the round trip covers buffers too."""
    torch.manual_seed(2026)
    model = build_cnn(n_blocks).train()
    with torch.no_grad():
        for _ in range(2):
            model(torch.randn(4, *INPUT_SHAPE))
    return model.eval()


def _saved(tmp_path, n_blocks: int = 3, name: str = "ckpt"):
    model = _model(n_blocks)
    mean, std = _stats()
    out = save_checkpoint(model, mean, std, _meta(n_blocks), tmp_path / name)
    return model, mean, std, out


def _inputs():
    return torch.from_numpy(np.random.default_rng(7).normal(size=(6, *INPUT_SHAPE)).astype(np.float32))


# Round trip and determinism --------------------------------------------------------------------

@pytest.mark.parametrize("n_blocks", [3, 4])
def test_round_trip_gives_identical_logits_and_stats(tmp_path, n_blocks):
    model, mean, std, out = _saved(tmp_path, n_blocks)
    assert sorted(p.name for p in out.iterdir()) == [JSON_NAME, WEIGHTS_NAME]
    loaded, mean2, std2, meta = load_checkpoint(out)
    assert not loaded.training  # returned ready to score
    x = _inputs()
    with torch.no_grad():
        assert torch.equal(model(x), loaded(x))
    np.testing.assert_array_equal(mean2, mean)
    np.testing.assert_array_equal(std2, std)
    assert mean2.dtype == std2.dtype == np.float32
    assert meta == _meta(n_blocks)
    for key, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)


def test_double_save_is_byte_identical_with_same_hash(tmp_path):
    _, _, _, a = _saved(tmp_path, name="a")
    _, _, _, b = _saved(tmp_path, name="b")
    for name in (JSON_NAME, WEIGHTS_NAME):
        assert (a / name).read_bytes() == (b / name).read_bytes()
    assert checkpoint_sha256(a) == checkpoint_sha256(b)
    # The hash is the documented two-line text over the two file hashes.
    lines = "".join(f"{n}:{hashlib.sha256((a / n).read_bytes()).hexdigest()}\n"
                    for n in (JSON_NAME, WEIGHTS_NAME))
    assert checkpoint_sha256(a) == hashlib.sha256(lines.encode("ascii")).hexdigest()


def test_json_is_sorted_strict_and_has_exact_keys(tmp_path):
    _, _, _, out = _saved(tmp_path)
    text = (out / JSON_NAME).read_text(encoding="utf-8")
    meta = json.loads(text)
    assert set(meta) == METADATA_KEYS
    assert text == json.dumps(meta, indent=2, sort_keys=True, allow_nan=False) + "\n"


def test_expected_sha256_accepts_the_recorded_hash(tmp_path):
    _, _, _, out = _saved(tmp_path)
    load_checkpoint(out, expected_sha256=checkpoint_sha256(out))


# Tampering ------------------------------------------------------------------------------------

def _flip_last_byte(path: Path):
    data = bytearray(path.read_bytes())
    data[-1] ^= 0x01  # inside the tensor data, so the file stays a valid safetensors file
    path.write_bytes(bytes(data))


def test_tampered_weight_value_changes_hash_and_is_refused_against_record(tmp_path):
    _, _, _, out = _saved(tmp_path)
    recorded = checkpoint_sha256(out)
    _flip_last_byte(out / WEIGHTS_NAME)
    assert checkpoint_sha256(out) != recorded
    with pytest.raises(ValueError, match="checkpoint_sha256 mismatch"):
        load_checkpoint(out, expected_sha256=recorded)


def test_tampered_json_value_changes_hash_and_is_refused_against_record(tmp_path):
    _, _, _, out = _saved(tmp_path)
    recorded = checkpoint_sha256(out)
    path = out / JSON_NAME
    path.write_text(path.read_text(encoding="utf-8").replace("0.0125", "0.0100"), encoding="utf-8")
    assert checkpoint_sha256(out) != recorded
    with pytest.raises(ValueError, match="checkpoint_sha256 mismatch"):
        load_checkpoint(out, expected_sha256=recorded)


def test_corrupt_weights_file_is_refused(tmp_path):
    _, _, _, out = _saved(tmp_path)
    path = out / WEIGHTS_NAME
    path.write_bytes(path.read_bytes()[:-100])  # truncated
    with pytest.raises(ValueError, match="safetensors"):
        load_checkpoint(out)
    path.write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{garbage")  # broken header
    with pytest.raises(ValueError, match="safetensors"):
        load_checkpoint(out)


def _rewrite_weights(out: Path, edit):
    tensors = load_file(out / WEIGHTS_NAME)
    edit(tensors)
    (out / WEIGHTS_NAME).unlink()
    save_file(tensors, out / WEIGHTS_NAME)


@pytest.mark.parametrize("edit, match", [
    (lambda t: t.pop("norm.mean"), "missing tensor 'norm.mean'"),
    (lambda t: t.pop("head.bias"), "does not match"),
    (lambda t: t.update({"extra.weight": torch.zeros(1)}), "does not match"),
    (lambda t: t.update({"head.weight": t["head.weight"].double()}), "head.weight"),
    (lambda t: t.update({"head.weight": torch.zeros(1, 32)}), "head.weight"),
    (lambda t: t["head.weight"].__setitem__((0, 0), float("nan")), "NaN"),
    (lambda t: t["norm.std"].__setitem__(3, 0.0), "std must be > 0"),
    (lambda t: t.update({"norm.mean": t["norm.mean"][:32]}), "mean"),
])
def test_invalid_tensors_are_refused(tmp_path, edit, match):
    _, _, _, out = _saved(tmp_path)
    _rewrite_weights(out, edit)
    with pytest.raises(ValueError, match=match):
        load_checkpoint(out)


@pytest.mark.parametrize("changes", [
    {"schema_version": 2}, {"schema_version": True}, {"version": " "},
    {"n_blocks": 5}, {"channels": [16, 32, 32]}, {"dropout": 0.5}, {"dropout": "0.3"},
    {"input_shape": [1, 64, 196]}, {"learning_rate": 0}, {"learning_rate": "1e-3"},
    {"seed": -1}, {"seed": True}, {"best_epoch": 0}, {"best_epoch": 31}, {"best_epoch": 7.0},
    {"val_a_eer": 1.5}, {"val_a_eer": None}, {"code_commit": "A" * 40}, {"code_commit": "a" * 39},
    {"profile_fingerprint": "g" * 64}, {"f2_manifest_sha256": "c" * 63},
    {"train_cache_npz_sha256": None}, {"val_a_cache_npz_sha256": 1},
    {"environment": {}}, {"environment": {"python": 3.13}}, {"determinism": []},
    {"determinism": {"seed": 1}}, {"determinism": {}},
])
def test_invalid_json_fields_are_refused(tmp_path, changes):
    _, _, _, out = _saved(tmp_path)
    (out / JSON_NAME).write_text(json.dumps(_meta(**changes)), encoding="utf-8")
    with pytest.raises(ValueError, match=JSON_NAME):
        load_checkpoint(out)


@pytest.mark.parametrize("edit", [
    lambda m: m.pop("best_epoch"),
    lambda m: m.update({"extra": 1}),
])
def test_missing_or_unknown_json_key_is_refused(tmp_path, edit):
    _, _, _, out = _saved(tmp_path)
    meta = _meta()
    edit(meta)
    (out / JSON_NAME).write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="missing keys|unknown keys"):
        load_checkpoint(out)


@pytest.mark.parametrize("text", ["{not json", "[1, 2]", '{"val_a_eer": NaN}', "\xff"])
def test_unparseable_json_is_refused(tmp_path, text):
    _, _, _, out = _saved(tmp_path)
    (out / JSON_NAME).write_bytes(text.encode("latin-1"))
    with pytest.raises(ValueError):
        load_checkpoint(out)


def test_nan_inside_metadata_is_refused(tmp_path):
    _, _, _, out = _saved(tmp_path)
    text = json.dumps(_meta(determinism={"seed": 0, "x": 1.0})).replace("1.0", "NaN")
    (out / JSON_NAME).write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="strict JSON"):
        load_checkpoint(out)


def test_four_block_weights_cannot_load_as_three_blocks(tmp_path):
    _, _, _, out = _saved(tmp_path, n_blocks=4)
    (out / JSON_NAME).write_text(json.dumps(_meta(3)), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match a 3-block CNN"):
        load_checkpoint(out)


# Folder contents ------------------------------------------------------------------------------

def test_extra_file_is_refused(tmp_path):
    _, _, _, out = _saved(tmp_path)
    (out / "notes.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly"):
        load_checkpoint(out)
    with pytest.raises(ValueError, match="exactly"):
        checkpoint_sha256(out)


@pytest.mark.parametrize("name", [JSON_NAME, WEIGHTS_NAME])
def test_missing_file_is_refused(tmp_path, name):
    _, _, _, out = _saved(tmp_path)
    (out / name).unlink()
    with pytest.raises(ValueError, match="exactly"):
        load_checkpoint(out)


def test_missing_folder_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not a checkpoint folder"):
        load_checkpoint(tmp_path / "nothing")


# Saving ---------------------------------------------------------------------------------------

def test_save_refuses_non_empty_folder_and_leaves_it_unchanged(tmp_path):
    out = tmp_path / "ckpt"
    out.mkdir()
    (out / "keep.txt").write_text("keep", encoding="utf-8")
    mean, std = _stats()
    with pytest.raises(ValueError, match="new or empty"):
        save_checkpoint(_model(), mean, std, _meta(), out)
    assert [p.name for p in out.iterdir()] == ["keep.txt"]


def test_save_into_empty_folder_is_allowed(tmp_path):
    out = tmp_path / "ckpt"
    out.mkdir()
    mean, std = _stats()
    save_checkpoint(_model(), mean, std, _meta(), out)
    load_checkpoint(out)


@pytest.mark.parametrize("meta, match", [
    (_meta(4), "n_blocks 4 != model n_blocks 3"),
    (_meta(dropout=0.2), "dropout"),
    ({"seed": 0}, "missing keys"),
])
def test_save_validates_before_writing(tmp_path, meta, match):
    mean, std = _stats()
    with pytest.raises(ValueError, match=match):
        save_checkpoint(_model(3), mean, std, meta, tmp_path / "ckpt")
    assert not (tmp_path / "ckpt").exists()


def test_save_refuses_bad_stats_or_model(tmp_path):
    mean, std = _stats()
    zero = std.copy()
    zero[0] = 0
    for args in [(mean.astype(np.float64), std), (mean, zero), (mean[:10], std)]:
        with pytest.raises(ValueError):
            save_checkpoint(_model(), *args, _meta(), tmp_path / "ckpt")
    with pytest.raises(ValueError, match="CNN"):
        save_checkpoint(torch.nn.Linear(2, 1), mean, std, _meta(), tmp_path / "ckpt")
    assert not (tmp_path / "ckpt").exists()


def test_failed_save_removes_partial_output(tmp_path, monkeypatch):
    real_write = Path.write_bytes
    calls = []

    def flaky(self, data):
        calls.append(self.name)
        if len(calls) == 2:  # the second file fails after the first was written
            raise OSError("disk full")
        return real_write(self, data)

    monkeypatch.setattr(Path, "write_bytes", flaky)
    mean, std = _stats()
    created = tmp_path / "new"
    with pytest.raises(OSError, match="disk full"):
        save_checkpoint(_model(), mean, std, _meta(), created)
    assert not created.exists()  # created by this call, so removed

    calls.clear()
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(OSError, match="disk full"):
        save_checkpoint(_model(), mean, std, _meta(), existing)
    assert existing.is_dir() and list(existing.iterdir()) == []  # pre-existing folder left empty


# No pickle, layering ---------------------------------------------------------------------------

def test_load_never_uses_torch_load_or_pickle(tmp_path, monkeypatch):
    _, _, _, out = _saved(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("pickle-based loading must never be called")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(pickle, "load", forbidden)
    monkeypatch.setattr(pickle, "loads", forbidden)
    monkeypatch.setattr(pickle, "Unpickler", forbidden)
    load_checkpoint(out)


def test_checkpoint_module_imports_no_pickle_or_research_code():
    tree = ast.parse(CKPT_FILE.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Attribute) and node.attr in {"load", "save"}:
            assert not (isinstance(node.value, ast.Name) and node.value.id == "torch"), \
                "torch.load / torch.save must not be used"
    forbidden = {"pickle", "joblib", "dill", "cloudpickle"}
    assert not {m for m in imported if m.split(".")[0] in forbidden}
    assert not {m for m in imported if m.startswith(("src.evaluation", "src.tuning"))}
    assert ckpt_module.FILES == (JSON_NAME, WEIGHTS_NAME)
