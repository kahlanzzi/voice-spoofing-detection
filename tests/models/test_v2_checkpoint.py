"""Model v2 checkpoint (D59 (7), D61 (6)): safetensors + strict v2 JSON for the three D49 candidates.

Synthetic tensors only, seed 2026. The layout and combined hash must be the D59 ones, so the
bundle code can hold a v2 checkpoint unchanged.
"""
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

import src.models.v2_checkpoint as v2  # noqa: E402
from src.bundle.bundle import CHECKPOINT_FILES, checkpoint_dir_sha256  # noqa: E402
from src.models import cnn_checkpoint  # noqa: E402
from src.models.v2_checkpoint import (  # noqa: E402
    ARCHITECTURES,
    JSON_NAME,
    METADATA_KEYS,
    WEIGHTS_NAME,
    build_model,
    load_v2_checkpoint,
    save_v2_checkpoint,
    trainable_parameters,
    v2_checkpoint_sha256,
)

ROOT = Path(__file__).resolve().parents[2]
V2_FILE = ROOT / "src" / "models" / "v2_checkpoint.py"
NEW_MODEL_FILES = [ROOT / "src" / "models" / n for n in ("crnn.py", "aasist.py", "v2_checkpoint.py")]
# Pinned trainable parameters (Entry 170): D49's tie rule orders candidates by these.
PARAMETERS = {"cnn-aug": 97_761, "crnn": 319_585, "aasist": 297_705}
INPUT_SHAPES = {"cnn-aug": (1, 64, 197), "crnn": (1, 64, 197), "aasist": (32000,)}


def _meta(arch: str = "crnn", **changes) -> dict:
    logmel = arch != "aasist"
    meta = {
        "schema_version": 1,
        "architecture": arch,
        "version": f"{arch}-test",
        "input": "logmel" if logmel else "waveform",
        "input_shape": [1, 64, 197] if logmel else [32000],
        "seed": 0,
        "best_epoch": 9,
        "selection_m": 0.0425,
        "code_commit": "a" * 40,
        "environment": {"python": "3.13.15", "torch": "2.11.0+cpu"},
        "profile_fingerprint": "b" * 64,
        "f2_manifest_sha256": "c" * 64,
        "augmentation_config_sha256": "d" * 64,
        "selection_set_sha256": "e" * 64,
        "train_stats_npz_sha256": "f" * 64 if logmel else None,
        "determinism": {"seed": 0, "cudnn_deterministic": True},
        "trainable_parameters": PARAMETERS[arch],
    }
    meta.update(changes)
    return meta


def _stats():
    rng = np.random.default_rng(2026)
    return rng.normal(-6.0, 2.0, 64).astype(np.float32), rng.uniform(0.5, 3.0, 64).astype(np.float32)


def _inputs(arch: str, n: int = 2):
    rng = np.random.default_rng(7)
    return torch.from_numpy(rng.normal(size=(n, *INPUT_SHAPES[arch])).astype(np.float32))


def _model(arch: str):
    """A model with non-default BatchNorm running statistics, so the round trip covers buffers."""
    torch.manual_seed(2026)
    model = build_model(arch).train()
    with torch.no_grad():
        model(_inputs(arch, 3))
    return model.eval()


def _saved(tmp_path, arch: str = "crnn", name: str = "ckpt"):
    model = _model(arch)
    mean, std = _stats() if arch != "aasist" else (None, None)
    out = save_v2_checkpoint(model, _meta(arch), tmp_path / name, mean=mean, std=std)
    return model, mean, std, out


# Round trip, hash rule, parameters ---------------------------------------------------------------

@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_round_trip_for_each_architecture(tmp_path, arch):
    # Weights, buffers, norm stats and metadata come back exactly; the model is ready to score.
    model, mean, std, out = _saved(tmp_path, arch)
    assert sorted(p.name for p in out.iterdir()) == sorted([JSON_NAME, WEIGHTS_NAME])
    loaded, mean2, std2, meta = load_v2_checkpoint(out)
    assert not loaded.training
    assert type(loaded) is type(model)
    for key, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)
    x = _inputs(arch)
    with torch.no_grad():
        assert torch.equal(model(x), loaded(x))
    if arch == "aasist":
        assert mean2 is None and std2 is None
    else:
        np.testing.assert_array_equal(mean2, mean)
        np.testing.assert_array_equal(std2, std)
        assert mean2.dtype == std2.dtype == np.float32
    assert meta == _meta(arch)


def test_trainable_parameters_are_pinned():
    # These numbers decide D49's tie rule (fewer trainable parameters first); recorded in Entry 170.
    counts = {arch: trainable_parameters(build_model(arch)) for arch in ARCHITECTURES}
    print(f"trainable parameters: {counts}")
    assert counts == PARAMETERS


def test_builders_are_d49_candidates():
    # C1 is exactly v1's selected 4-block CNN (D49 (C1)); unknown names are refused.
    from src.models.aasist import Model
    from src.models.cnn import CNN
    from src.models.crnn import CRNN

    assert ARCHITECTURES == ("cnn-aug", "crnn", "aasist")
    cnn = build_model("cnn-aug")
    assert isinstance(cnn, CNN) and cnn.n_blocks == 4
    assert isinstance(build_model("crnn"), CRNN)
    assert isinstance(build_model("aasist"), Model)
    with pytest.raises(ValueError, match="architecture"):
        build_model("wav2vec2")


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_hash_equals_the_d59_bundle_rule(tmp_path, arch):
    # F5 must fit D59 unchanged: same files in the same order, same combined hash as
    # cnn_checkpoint.checkpoint_sha256 and src.bundle.bundle.checkpoint_dir_sha256.
    _, _, _, out = _saved(tmp_path, arch)
    assert v2.FILES == cnn_checkpoint.FILES == tuple(CHECKPOINT_FILES)
    per_file = {n: hashlib.sha256((out / n).read_bytes()).hexdigest() for n in CHECKPOINT_FILES}
    assert v2_checkpoint_sha256(out) == checkpoint_dir_sha256(per_file)
    assert v2_checkpoint_sha256(out) == cnn_checkpoint.checkpoint_sha256(out)


def test_double_save_is_byte_identical(tmp_path):
    _, _, _, a = _saved(tmp_path, name="a")
    _, _, _, b = _saved(tmp_path, name="b")
    for name in (JSON_NAME, WEIGHTS_NAME):
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_json_is_sorted_strict_and_has_exact_keys(tmp_path):
    _, _, _, out = _saved(tmp_path, "aasist")
    text = (out / JSON_NAME).read_text(encoding="utf-8")
    meta = json.loads(text)
    assert set(meta) == METADATA_KEYS
    assert meta["train_stats_npz_sha256"] is None  # JSON null for the waveform model
    assert text == json.dumps(meta, indent=2, sort_keys=True, allow_nan=False) + "\n"


def test_v1_loader_refuses_a_v2_checkpoint(tmp_path):
    # Same layout, different schema: F4's loader must not accept a v2 checkpoint by accident.
    _, _, _, out = _saved(tmp_path, "cnn-aug")
    with pytest.raises(ValueError, match="keys"):
        cnn_checkpoint.load_checkpoint(out)


# Metadata refusals -------------------------------------------------------------------------------

def _rewrite_json(out: Path, meta: dict) -> None:
    (out / JSON_NAME).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@pytest.mark.parametrize("arch,changes,match", [
    ("crnn", {"architecture": "rawnet"}, "architecture"),
    ("crnn", {"schema_version": 2}, "schema_version"),
    ("crnn", {"version": " "}, "version"),
    ("crnn", {"input": "waveform"}, "input must be"),          # input does not match architecture
    ("aasist", {"input": "logmel"}, "input must be"),
    ("crnn", {"input_shape": [32000]}, "input_shape"),         # shape does not match architecture
    ("aasist", {"input_shape": [64600]}, "input_shape"),
    ("crnn", {"seed": -1}, "seed"),
    ("crnn", {"seed": True}, "seed"),
    ("crnn", {"best_epoch": 0}, "best_epoch"),
    ("crnn", {"best_epoch": 31}, "best_epoch"),
    ("crnn", {"selection_m": 1.5}, "selection_m"),
    ("crnn", {"selection_m": float("inf")}, "selection_m"),
    ("crnn", {"code_commit": "A" * 40}, "code_commit"),
    ("crnn", {"augmentation_config_sha256": "0" * 63}, "augmentation_config_sha256"),
    ("crnn", {"selection_set_sha256": None}, "selection_set_sha256"),
    ("crnn", {"train_stats_npz_sha256": None}, "train_stats_npz_sha256"),   # logmel needs stats hash
    ("aasist", {"train_stats_npz_sha256": "f" * 64}, "train_stats_npz_sha256"),  # waveform: null
    ("crnn", {"environment": {}}, "environment"),
    ("crnn", {"determinism": {"seed": 1}}, "determinism.seed"),
    ("crnn", {"trainable_parameters": 319_584}, "trainable_parameters"),
    ("aasist", {"trainable_parameters": PARAMETERS["crnn"]}, "trainable_parameters"),
    ("crnn", {"trainable_parameters": 319_585.0}, "trainable_parameters"),
])
def test_invalid_metadata_is_refused_on_load(tmp_path, arch, changes, match):
    _, _, _, out = _saved(tmp_path, arch)
    _rewrite_json(out, _meta(arch, **changes))
    with pytest.raises(ValueError, match=match):
        load_v2_checkpoint(out)


@pytest.mark.parametrize("edit", ["missing", "unknown"])
def test_missing_or_unknown_key_is_refused(tmp_path, edit):
    _, _, _, out = _saved(tmp_path)
    meta = _meta()
    if edit == "missing":
        del meta["selection_m"]
    else:
        meta["val_a_eer"] = 0.0  # the v1 key is not part of the v2 schema
    _rewrite_json(out, meta)
    with pytest.raises(ValueError, match="keys"):
        load_v2_checkpoint(out)


def test_nan_in_metadata_is_refused(tmp_path):
    _, _, _, out = _saved(tmp_path)
    meta = _meta(determinism={"seed": 0, "x": float("nan")})
    (out / JSON_NAME).write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="strict JSON"):
        load_v2_checkpoint(out)


# Tensor refusals ---------------------------------------------------------------------------------

def _edit_weights(out: Path, edit) -> None:
    tensors = load_file(out / WEIGHTS_NAME)
    edit(tensors)
    save_file(tensors, out / WEIGHTS_NAME)


def test_norm_tensors_on_aasist_are_refused(tmp_path):
    _, _, _, out = _saved(tmp_path, "aasist")
    mean, std = _stats()
    _edit_weights(out, lambda t: t.update({"norm.mean": torch.from_numpy(mean),
                                           "norm.std": torch.from_numpy(std)}))
    with pytest.raises(ValueError, match="waveform"):
        load_v2_checkpoint(out)


@pytest.mark.parametrize("drop", [("norm.mean", "norm.std"), ("norm.std",)])
def test_logmel_checkpoint_without_norm_tensors_is_refused(tmp_path, drop):
    _, _, _, out = _saved(tmp_path, "cnn-aug")
    _edit_weights(out, lambda t: [t.pop(k) for k in drop])
    with pytest.raises(ValueError, match="norm"):
        load_v2_checkpoint(out)


@pytest.mark.parametrize("edit,match", [
    (lambda t: t.pop("head.weight"), "missing"),
    (lambda t: t.update({"extra": torch.zeros(1)}), "unknown"),
    (lambda t: t.update({"head.weight": t["head.weight"].double()}), "must be"),
    (lambda t: t.update({"head.bias": torch.zeros(2)}), "must be"),
    (lambda t: t.update({"head.bias": torch.tensor([float("nan")])}), "NaN"),
    (lambda t: t.update({"norm.std": torch.zeros(64)}), "std must be > 0"),
])
def test_invalid_tensors_are_refused(tmp_path, edit, match):
    _, _, _, out = _saved(tmp_path, "crnn")
    _edit_weights(out, edit)
    with pytest.raises(ValueError, match=match):
        load_v2_checkpoint(out)


def test_crnn_weights_cannot_load_as_another_architecture(tmp_path):
    # The metadata names the builder; weights of another architecture must not load into it.
    _, _, _, out = _saved(tmp_path, "crnn")
    _rewrite_json(out, _meta("cnn-aug"))
    with pytest.raises(ValueError, match="does not match cnn-aug"):
        load_v2_checkpoint(out)


def test_tampered_weights_are_refused_against_the_recorded_hash(tmp_path):
    _, _, _, out = _saved(tmp_path)
    recorded = v2_checkpoint_sha256(out)
    load_v2_checkpoint(out, expected_sha256=recorded)  # the recorded hash is accepted
    _edit_weights(out, lambda t: t.update({"head.bias": t["head.bias"] + 1.0}))
    load_v2_checkpoint(out)  # still a valid checkpoint on its own ...
    with pytest.raises(ValueError, match="checkpoint_sha256 mismatch"):
        load_v2_checkpoint(out, expected_sha256=recorded)  # ... but not the recorded one


# Files and folders -------------------------------------------------------------------------------

def test_extra_file_is_refused(tmp_path):
    _, _, _, out = _saved(tmp_path)
    (out / "notes.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly"):
        load_v2_checkpoint(out)


@pytest.mark.parametrize("name", [JSON_NAME, WEIGHTS_NAME])
def test_missing_file_is_refused(tmp_path, name):
    _, _, _, out = _saved(tmp_path)
    (out / name).unlink()
    with pytest.raises(ValueError, match="exactly"):
        load_v2_checkpoint(out)


def test_save_refuses_non_empty_folder_and_leaves_it_unchanged(tmp_path):
    out = tmp_path / "ckpt"
    out.mkdir()
    (out / "keep.txt").write_text("x", encoding="utf-8")
    mean, std = _stats()
    with pytest.raises(ValueError, match="new or empty"):
        save_v2_checkpoint(_model("crnn"), _meta(), out, mean=mean, std=std)
    assert [p.name for p in out.iterdir()] == ["keep.txt"]


@pytest.mark.parametrize("arch,kwargs,meta_arch,match", [
    ("crnn", {}, "crnn", "mean and std are required"),                # logmel without stats
    ("aasist", "stats", "aasist", "must be None"),                    # waveform with stats
    ("crnn", "stats", "cnn-aug", "architecture"),                     # model / metadata mismatch
])
def test_save_validates_before_writing(tmp_path, arch, kwargs, meta_arch, match):
    if kwargs == "stats":
        mean, std = _stats()
        kwargs = {"mean": mean, "std": std}
    out = tmp_path / "ckpt"
    with pytest.raises(ValueError, match=match):
        save_v2_checkpoint(_model(arch), _meta(meta_arch), out, **kwargs)
    assert not out.exists()


def test_save_refuses_a_model_that_is_not_a_candidate(tmp_path):
    from src.models.cnn import build_cnn

    mean, std = _stats()
    with pytest.raises(ValueError, match="builders"):  # 3-block CNN is v1's search space, not C1
        save_v2_checkpoint(build_cnn(3), _meta("cnn-aug"), tmp_path / "ckpt", mean=mean, std=std)


def test_save_refuses_frozen_parameters(tmp_path):
    # trainable_parameters must describe the saved model; freezing a layer changes the count.
    model = _model("crnn")
    model.head.weight.requires_grad_(False)
    mean, std = _stats()
    with pytest.raises(ValueError, match="trainable parameters"):
        save_v2_checkpoint(model, _meta(), tmp_path / "ckpt", mean=mean, std=std)


def test_failed_save_removes_partial_output(tmp_path, monkeypatch):
    out = tmp_path / "ckpt"
    real = Path.write_bytes

    def fail_on_json(self, data):
        if self.name == JSON_NAME:
            raise OSError("disk full")
        return real(self, data)

    monkeypatch.setattr(Path, "write_bytes", fail_on_json)
    mean, std = _stats()
    with pytest.raises(OSError):
        save_v2_checkpoint(_model("crnn"), _meta(), out, mean=mean, std=std)
    assert not out.exists()


# No pickle, layering -----------------------------------------------------------------------------

def test_load_never_uses_torch_load_or_pickle(tmp_path, monkeypatch):
    outs = [_saved(tmp_path, arch, name=arch)[3] for arch in ARCHITECTURES]

    def forbidden(*args, **kwargs):
        raise AssertionError("pickle-based loading must never be called")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(pickle, "load", forbidden)
    monkeypatch.setattr(pickle, "loads", forbidden)
    monkeypatch.setattr(pickle, "Unpickler", forbidden)
    for out in outs:
        load_v2_checkpoint(out)


@pytest.mark.parametrize("path", NEW_MODEL_FILES, ids=lambda p: p.name)
def test_new_model_files_import_no_pickle_or_research_code(path):
    # Stage 13, F5 and the API import these files: no research code, no pickle, no torch.load/save.
    tree = ast.parse(path.read_text(encoding="utf-8"))
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
