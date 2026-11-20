"""V2b-4 CLI of the model v2 search: resource check, raw-cache pin, argument parsing, wiring. Synthetic."""
from __future__ import annotations

import json
import shutil

import pandas as pd
import pytest

from tests._torch import require_torch

torch = require_torch()

import src.tuning.v2_search as vs  # noqa: E402
from src.augmentation.resource_cache import JSON_NAME, build_training_resources  # noqa: E402
from src.features.raw_train_cache import build_raw_train_cache  # noqa: E402
from tests._v2_fakes import AUG_CONFIG_PATH, COMMIT, aug_config_copy, sha, write_dataset  # noqa: E402
from tests.augmentation.test_resource_cache import _inputs as resource_inputs  # noqa: E402

FIELDS = ("noise_inventory_sha256", "rir_inventory_sha256", "noise_archive_sha256",
          "rir_archive_sha256")


# --- load_v2_resources ---------------------------------------------------------------------------

@pytest.fixture()
def resource_cache(tmp_path):
    """A 2 + 2 training-half cache built from inventories and pins that the check reads back."""
    noise_root, rir_root, noise, rirs = resource_inputs(tmp_path / "corpus")
    noise_csv, rir_csv = tmp_path / "musan_inventory.csv", tmp_path / "rirs_noises_inventory.csv"
    noise.to_csv(noise_csv, index=False)
    rirs.to_csv(rir_csv, index=False)
    aug = aug_config_copy(tmp_path / "augmentation.yaml")
    corpora = vs.load_augmentation_config(aug).corpora
    out = tmp_path / "resources"
    build_training_resources(
        noise_root, rir_root, noise, rirs, out,
        noise_inventory_sha256=sha(noise_csv), rir_inventory_sha256=sha(rir_csv),
        noise_archive_sha256=corpora["noise"]["archive_sha256"],
        rir_archive_sha256=corpora["rir"]["archive_sha256"], code_commit=COMMIT,
        expected_noise_count=2, expected_rir_count=2)
    kwargs = dict(noise_inventory_path=noise_csv, rir_inventory_path=rir_csv,
                  augmentation_config_path=aug, expected_noise_count=2, expected_rir_count=2)
    return out, kwargs


def test_load_v2_resources_accepts_the_committed_provenance(resource_cache):
    out, kwargs = resource_cache
    resources, meta = vs.load_v2_resources(out, **kwargs)
    assert len(resources.noise_waveforms) == len(resources.rir_waveforms) == 2
    assert meta["noise_archive_sha256"] == "86d1061c7e15b5c9e906777685c519701df51bfde3001e1070dcc9ffac955ee1"


@pytest.mark.parametrize("field", FIELDS)
def test_load_v2_resources_refuses_each_mismatch_naming_both_values(resource_cache, field):
    out, kwargs = resource_cache
    path = out / JSON_NAME
    meta = json.loads(path.read_text(encoding="utf-8"))
    expected = meta[field]
    meta[field] = "f" * 64  # still valid for load_training_resources (format only)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=rf"{field}: expected {expected}, got {'f' * 64}"):
        vs.load_v2_resources(out, **kwargs)


def test_load_v2_resources_defaults_are_the_committed_files():
    assert vs.NOISE_INVENTORY_PATH.is_file() and vs.RIR_INVENTORY_PATH.is_file()
    assert vs.AUGMENTATION_CONFIG_PATH == AUG_CONFIG_PATH
    assert vs.RAW_CACHE_EVIDENCE.is_file() and vs.F2_MANIFEST_PATH.is_file()
    assert vs.PROFILE_PATH.is_file() and vs.V2_CONFIG_PATH.is_file()


# --- raw-cache pin -------------------------------------------------------------------------------

@pytest.fixture()
def raw_cache(tmp_path):
    root, manifest, manifest_path = write_dataset(tmp_path / "data", sizes={"train": 4})
    out = tmp_path / "raw"
    build_raw_train_cache(root, manifest, "train", out, f2_manifest_sha256=sha(manifest_path),
                          code_commit=COMMIT)
    evidence = tmp_path / "evidence" / "train_raw.json"
    evidence.parent.mkdir()
    shutil.copy2(out / "train_raw.json", evidence)
    return out, evidence


def test_raw_cache_pin_accepts_the_evidence_cache(raw_cache):
    out, evidence = raw_cache
    assert vs.check_raw_cache_pin(out, evidence) == sha(out / "train_raw.npz")


def test_raw_cache_pin_refuses_another_npz(raw_cache):
    out, evidence = raw_cache
    data = json.loads(evidence.read_text(encoding="utf-8"))
    data["npz_sha256"] = "0" * 64
    evidence.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=rf"npz SHA-256 expected {'0' * 64} \(V2b-3 evidence"):
        vs.check_raw_cache_pin(out, evidence)


def test_raw_cache_pin_refuses_other_metadata_bytes_and_missing_files(raw_cache, tmp_path):
    out, evidence = raw_cache
    evidence.write_text(evidence.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="the committed V2b-3 evidence file"):
        vs.check_raw_cache_pin(out, evidence)
    with pytest.raises(ValueError, match="is missing"):
        vs.check_raw_cache_pin(tmp_path / "nowhere", evidence)


# --- arguments -----------------------------------------------------------------------------------

INPUT_ARGS = ["--raw-cache-dir", "r", "--stage8-train-logmel-dir", "t", "--stage8-val-a-logmel-dir", "v",
              "--selection-set-dir", "s", "--resources-dir", "a", "--out", "o", "--device", "cpu",
              "--num-workers", "2"]


def test_parser_search_subcommand():
    args = vs.build_parser().parse_args(["search", *INPUT_ARGS, "--version", "cnn-v2",
                                         "--wandb-mode", "online"])
    assert (args.command, args.version, args.wandb_mode, args.num_workers) == ("search", "cnn-v2", "online", 2)
    assert [str(p) for p in (args.raw_cache_dir, args.stage8_train_logmel_dir, args.stage8_val_a_logmel_dir,
                             args.selection_set_dir, args.resources_dir, args.out)] == list("rtvsao")


def test_parser_rehearse_subcommand_takes_no_wandb_or_version():
    args = vs.build_parser().parse_args(["rehearse", *INPUT_ARGS])
    assert args.command == "rehearse" and not hasattr(args, "wandb_mode")
    for extra in (["--wandb-mode", "online"], ["--version", "cnn-v2"]):
        with pytest.raises(SystemExit):
            vs.build_parser().parse_args(["rehearse", *INPUT_ARGS, *extra])


@pytest.mark.parametrize("drop", ["--raw-cache-dir", "--resources-dir", "--out", "--num-workers", "--device"])
def test_parser_requires_every_input(drop):
    i = INPUT_ARGS.index(drop)
    with pytest.raises(SystemExit):
        vs.build_parser().parse_args(["search", *INPUT_ARGS[:i], *INPUT_ARGS[i + 2:], "--version", "x",
                                      "--wandb-mode", "disabled"])
    with pytest.raises(SystemExit):
        vs.build_parser().parse_args(["search", *INPUT_ARGS, "--version", "x", "--wandb-mode", "loud"])


# --- wiring --------------------------------------------------------------------------------------

@pytest.fixture()
def recorded(monkeypatch):
    calls = []
    resources = object()
    monkeypatch.setattr(vs, "require_versions", lambda: calls.append("require_versions"))
    monkeypatch.setattr(vs, "check_raw_cache_pin", lambda d: calls.append(("pin", d)))
    monkeypatch.setattr(vs, "load_v2_resources", lambda d: calls.append(("resources", d)) or (resources, {}))

    def run_search(**kw):
        calls.append(("run_search", kw))
        return {"selected": "crnn"}
    monkeypatch.setattr(vs, "run_search", run_search)
    return calls, resources


def test_main_search_checks_then_runs_with_the_committed_files(recorded, tmp_path, capsys):
    calls, resources = recorded
    vs.main(["search", *INPUT_ARGS[:-6], "--out", str(tmp_path / "o"), "--device", "cpu",
             "--num-workers", "0", "--version", "cnn-v2", "--wandb-mode", "disabled"])
    assert calls[0] == "require_versions"
    assert [c[0] for c in calls[1:]] == ["pin", "resources", "run_search"]
    kw = calls[-1][1]
    assert kw["resources"] is resources
    assert (kw["profile_path"], kw["v2_config_path"], kw["augmentation_config_path"], kw["manifest_path"]) == (
        vs.PROFILE_PATH, vs.V2_CONFIG_PATH, vs.AUGMENTATION_CONFIG_PATH, vs.F2_MANIFEST_PATH)
    assert (kw["version"], kw["logger_mode"], kw["device"], kw["num_workers"]) == ("cnn-v2", "disabled", "cpu", 0)
    assert json.loads(capsys.readouterr().out) == {"selected": "crnn"}


@pytest.mark.parametrize("name", [vs.REHEARSAL_MARKER, vs.REHEARSAL_REPORT])
def test_search_refuses_a_rehearsal_folder_before_reading_inputs(recorded, tmp_path, name):
    calls, _ = recorded
    out = tmp_path / "rehearsal"
    out.mkdir()
    (out / name).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="is a rehearsal folder"):
        vs.search_cli(raw_cache_dir="r", stage8_train_logmel_dir="t", stage8_val_a_logmel_dir="v",
                      selection_set_dir="s", resources_dir="a", out_dir=out, version="cnn-v2",
                      logger_mode="disabled", device="cpu", num_workers=0)
    assert calls == ["require_versions"]


def test_committed_manifest_is_the_f2_manifest():
    assert sha(vs.F2_MANIFEST_PATH) == vs.F2_MANIFEST_SHA256
    assert list(pd.read_csv(vs.F2_MANIFEST_PATH, nrows=1).columns)[:2] == ["path", "label"]
