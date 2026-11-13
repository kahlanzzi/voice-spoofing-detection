"""ModelBundle format and verifying loader tests (D59), on dummy files in tmp_path only.

No dataset access and no PyTorch: the checkpoint folder holds a small valid checkpoint.json (version
and profile_fingerprint) and a few dummy weight bytes, the profile is a tmp copy of
configs/preprocessing.yaml, and the policy is a tmp file with a hand-made RiskPolicy.to_json().
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import src.bundle.bundle as bundle_module
from src.bundle.bundle import (
    CHECKPOINT_DIR,
    CHECKPOINT_FILES,
    COMPONENTS,
    MANIFEST_NAME,
    SCHEMA_VERSION,
    BundleManifest,
    build_bundle,
    checkpoint_dir_sha256,
    load_bundle,
)
from src.preprocessing.profile import load_profile
from src.risk.calibration import PlattCalibration
from src.risk.policy import RiskPolicy

ROOT = Path(__file__).resolve().parents[2]
BUNDLE_DIR = ROOT / "src" / "bundle"
PROFILE_SRC = ROOT / "configs" / "preprocessing.yaml"

COMMIT = "0123456789abcdef0123456789abcdef01234567"
LEDGER = "c9fc2e230c0883028459d0b2b8de4f6ee537676cd5a4e259e9d0f545c60943cc"
CREATED_AT = "2026-10-17T12:00:00Z"
PROFILE_VERSION = "preprocessing-v1"
FINGERPRINT = load_profile(PROFILE_SRC).fingerprint()
PROFILE_NAME = "preprocessing.yaml"
CKPT_JSON, WEIGHTS = CHECKPOINT_FILES
TOP_FILES = (PROFILE_NAME, "policy.json")


def _policy(version: str = "policy-v1") -> RiskPolicy:
    return RiskPolicy(
        version=version,
        calibration=PlattCalibration(a=1.0, b=0.0, n_spoof=10, n_bonafide=12),
        threshold=0.5,
        half_width=0.1,
        target_deferral=0.10,
        n_spoof_fit=10,
        n_bonafide_fit=12,
        deferral_on_fit=0.0909090909090909,
    )


def _ckpt_json(version: str = "cnn-v1", fingerprint: str = FINGERPRINT) -> bytes:
    return (json.dumps({"profile_fingerprint": fingerprint, "version": version}, sort_keys=True) + "\n").encode()


def _sha(data_or_path) -> str:
    data = data_or_path.read_bytes() if isinstance(data_or_path, Path) else data_or_path
    return hashlib.sha256(data).hexdigest()


def _combined(folder: Path) -> str:
    return checkpoint_dir_sha256({name: _sha(folder / name) for name in CHECKPOINT_FILES})


def _sources(root: Path, *, ckpt_json: bytes | None = None, policy_bytes: bytes | None = None) -> dict:
    """A source checkpoint folder, profile and policy file under root; returns their paths."""
    ckpt = root / "seed_0"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / CKPT_JSON).write_bytes(_ckpt_json() if ckpt_json is None else ckpt_json)
    (ckpt / WEIGHTS).write_bytes(b"dummy weights \x00\x01\x02\xff")
    profile = root / PROFILE_NAME
    profile.write_bytes(PROFILE_SRC.read_bytes())
    policy = root / "risk_policy.json"
    policy.write_bytes(_policy().to_json().encode("utf-8") if policy_bytes is None else policy_bytes)
    return {"checkpoint_dir": ckpt, "profile": profile, "policy": policy}


def _kwargs(src: dict, **overrides) -> dict:
    kwargs = dict(
        checkpoint_dir=src["checkpoint_dir"],
        profile=src["profile"],
        policy=src["policy"],
        bundle_version="bundle-v1",
        model_version="cnn-v1",
        profile_version=PROFILE_VERSION,
        code_commit=COMMIT,
        protocol_version="1.0",
        f2_ledger_sha256=LEDGER,
        created_at=CREATED_AT,
    )
    kwargs.update(overrides)
    if "expected_checkpoint_sha256" not in kwargs:  # computed only when the test does not set it
        kwargs["expected_checkpoint_sha256"] = _combined(src["checkpoint_dir"])
    return kwargs


def _build(directory: Path, root: Path, **overrides) -> Path:
    return build_bundle(directory, **_kwargs(_sources(root), **overrides))


@pytest.fixture
def bundle_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "bundle"
    _build(directory, tmp_path / "src")
    return directory


def _edit_manifest(directory: Path, mutate) -> None:
    path = directory / MANIFEST_NAME
    data = json.loads(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def _rehash_manifest(directory: Path) -> None:
    """Rewrite every hash in the manifest from the files on disk, so only the cross-checks remain."""
    def mutate(d):
        files = {n: _sha(directory / CHECKPOINT_DIR / n) for n in CHECKPOINT_FILES}
        d["components"]["checkpoint"].update(files=files, sha256=checkpoint_dir_sha256(files))
        for name in ("profile", "policy"):
            d["components"][name]["sha256"] = _sha(directory / d["components"][name]["file"])
    _edit_manifest(directory, mutate)


def _tree(directory: Path) -> dict:
    return {str(p.relative_to(directory)): (p.read_bytes() if p.is_file() else None)
            for p in sorted(directory.rglob("*"))}


# 1. Round trip.
def test_build_then_load_round_trip(tmp_path: Path):
    # The loaded bundle must report exactly what was built: each file's hash is the SHA-256 of its
    # source bytes (copied unchanged), the combined checkpoint hash follows the D42 rule and is
    # returned as checkpoint_sha256, the policy hash equals the source policy file's SHA-256 (so the
    # real build reproduces Stage 10's value), the versions are the inputs, the profile is the parsed
    # profile, and manifest_sha256 is the SHA-256 of the manifest bytes (the F4 record).
    directory = tmp_path / "bundle"
    src = _sources(tmp_path / "src")
    manifest_path = build_bundle(directory, **_kwargs(src))
    assert manifest_path == directory / MANIFEST_NAME

    loaded = load_bundle(directory)
    m = loaded.manifest
    assert m.schema_version == SCHEMA_VERSION == 2
    assert (m.bundle_version, m.created_at, m.code_commit, m.protocol_version, m.f2_ledger_sha256) == (
        "bundle-v1", CREATED_AT, COMMIT, "1.0", LEDGER,
    )
    assert tuple(m.components) == COMPONENTS
    ck = m.components["checkpoint"]
    assert ck.dir == CHECKPOINT_DIR
    assert dict(ck.files) == {n: _sha(src["checkpoint_dir"] / n) for n in CHECKPOINT_FILES}
    assert ck.sha256 == _combined(src["checkpoint_dir"]) == loaded.checkpoint_sha256
    assert m.components["profile"].sha256 == _sha(src["profile"])
    assert m.components["policy"].sha256 == _sha(src["policy"])
    assert [m.components[c].version for c in COMPONENTS] == ["cnn-v1", PROFILE_VERSION, "policy-v1"]
    assert loaded.policy == _policy()
    assert loaded.profile == load_profile(PROFILE_SRC)
    assert loaded.checkpoint_dir == directory / CHECKPOINT_DIR
    assert loaded.profile_path == directory / PROFILE_NAME
    # Bundle bytes == source bytes for all four files.
    for name in CHECKPOINT_FILES:
        assert (directory / CHECKPOINT_DIR / name).read_bytes() == (src["checkpoint_dir"] / name).read_bytes()
    assert (directory / PROFILE_NAME).read_bytes() == src["profile"].read_bytes()
    assert (directory / "policy.json").read_bytes() == src["policy"].read_bytes()
    assert loaded.manifest_sha256 == _sha(manifest_path)
    # Strict JSON round trip of the manifest itself.
    assert BundleManifest.from_json(m.to_json()) == m


def test_manifest_checkpoint_entry_shape(bundle_dir: Path):
    # Schema 2 (D59 (2)): the checkpoint entry is {dir, files, sha256, version}; profile and policy
    # keep {file, sha256, version}.
    data = json.loads((bundle_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert set(data["components"]["checkpoint"]) == {"dir", "files", "sha256", "version"}
    assert list(data["components"]["checkpoint"]["files"]) == sorted(CHECKPOINT_FILES)
    assert set(data["components"]["profile"]) == {"file", "sha256", "version"}


def test_checkpoint_dir_sha256_rule():
    # The D42 rule spelled out: SHA-256 of "checkpoint.json:<h>\nweights.safetensors:<h>\n".
    files = {CKPT_JSON: "a" * 64, WEIGHTS: "b" * 64}
    text = f"checkpoint.json:{'a' * 64}\nweights.safetensors:{'b' * 64}\n"
    assert checkpoint_dir_sha256(files) == _sha(text.encode("ascii"))
    with pytest.raises(ValueError, match="checkpoint files must be exactly"):
        checkpoint_dir_sha256({CKPT_JSON: "a" * 64})
    with pytest.raises(ValueError, match="64 lowercase hex"):
        checkpoint_dir_sha256({CKPT_JSON: "A" * 64, WEIGHTS: "b" * 64})


# 2. Build refusals: every check runs before anything is written.
def _refused(tmp_path: Path, match: str, *, sources: dict | None = None, **overrides) -> None:
    directory = tmp_path / "bundle"
    src = sources if sources is not None else _sources(tmp_path / "src")
    kwargs = _kwargs(src, **overrides)
    with pytest.raises(ValueError, match=match):
        build_bundle(directory, **kwargs)
    assert not directory.exists()


def test_build_refuses_wrong_expected_checkpoint_sha256(tmp_path: Path):
    # The caller passes the Stage 8 recorded value; a wrong seed or a changed file gives another
    # combined hash, so the build must stop.
    _refused(tmp_path, "checkpoint_sha256 mismatch", expected_checkpoint_sha256="0" * 64)


@pytest.mark.parametrize("change", ["extra", "missing"])
def test_build_refuses_checkpoint_folder_contents(tmp_path: Path, change: str):
    # A D42 checkpoint is exactly two files; anything else is not a checkpoint we recorded.
    src = _sources(tmp_path / "src")
    if change == "extra":
        (src["checkpoint_dir"] / "notes.txt").write_text("x", encoding="utf-8")
    else:
        (src["checkpoint_dir"] / WEIGHTS).unlink()
    _refused(tmp_path, "must hold exactly", sources=src, expected_checkpoint_sha256="0" * 64)


def test_build_refuses_symlinked_checkpoint_file(tmp_path: Path):
    # A symlink could point anywhere and change after the build; only regular files are copied.
    src = _sources(tmp_path / "src")
    target = tmp_path / "elsewhere.safetensors"
    target.write_bytes((src["checkpoint_dir"] / WEIGHTS).read_bytes())
    (src["checkpoint_dir"] / WEIGHTS).unlink()
    try:
        os.symlink(target, src["checkpoint_dir"] / WEIGHTS)
    except (OSError, NotImplementedError) as e:
        pytest.skip(f"symlinks unavailable here (Windows needs Developer Mode or admin): {e}")
    _refused(tmp_path, "must be a regular file", sources=src, expected_checkpoint_sha256="0" * 64)


@pytest.mark.parametrize("overrides, match", [
    ({"profile_version": "preprocessing-v2"}, "profile: profile_version mismatch"),
    ({"protocol_version": "1.1"}, "profile: protocol_version mismatch"),
    ({"f2_ledger_sha256": "0" * 64}, "profile: f2_ledger_sha256 mismatch"),
    ({"model_version": "cnn-v2"}, "checkpoint: version mismatch"),
])
def test_build_refuses_declared_value_mismatch(tmp_path: Path, overrides: dict, match: str):
    # The declared versions and ledger must be what the files say (FR-04, NFR-09), so the manifest
    # cannot record a value the bundled profile or checkpoint contradicts.
    _refused(tmp_path, match, **overrides)


def test_build_refuses_checkpoint_profile_fingerprint_mismatch(tmp_path: Path):
    # The checkpoint was trained with a profile whose fingerprint differs from the bundled one.
    src = _sources(tmp_path / "src", ckpt_json=_ckpt_json(fingerprint="f" * 64))
    _refused(tmp_path, "profile_fingerprint mismatch", sources=src,
             expected_checkpoint_sha256=_combined(src["checkpoint_dir"]))


def test_build_refuses_invalid_policy_file(tmp_path: Path):
    # The policy file must load as a RiskPolicy before anything is written.
    _refused(tmp_path, "policy: invalid policy file",
             sources=_sources(tmp_path / "src", policy_bytes=b'{"schema_version": 1}'))


# 3. Tampering with a file after the build.
@pytest.mark.parametrize("component, rel", [
    ("checkpoint", f"{CHECKPOINT_DIR}/{CKPT_JSON}"),
    ("checkpoint", f"{CHECKPOINT_DIR}/{WEIGHTS}"),
    ("profile", PROFILE_NAME),
    ("policy", "policy.json"),
])
def test_tampered_file_raises(bundle_dir: Path, component: str, rel: str):
    # Flipping one bit of one byte changes the SHA-256 (no collision is known for SHA-256), so the
    # loader must refuse and name the part and file. Hashes are checked before anything is parsed,
    # so a tampered but still valid file is caught too.
    path = bundle_dir / rel
    data = bytearray(path.read_bytes())
    data[0] ^= 0x01
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match=rf"{component}: SHA-256 mismatch for {rel}"):
        load_bundle(bundle_dir)


# 4. Edited manifest.
def test_edited_manifest_sha256_raises(bundle_dir: Path):
    # A changed recorded per-file hash no longer matches the (untouched) file; the combined value is
    # updated too so the manifest still parses and the file check is what catches it.
    def mutate(d):
        files = {**d["components"]["checkpoint"]["files"], WEIGHTS: "0" * 64}
        d["components"]["checkpoint"].update(files=files, sha256=checkpoint_dir_sha256(files))
    _edit_manifest(bundle_dir, mutate)
    with pytest.raises(ValueError, match=f"checkpoint: SHA-256 mismatch for {CHECKPOINT_DIR}/{WEIGHTS}"):
        load_bundle(bundle_dir)


def test_edited_manifest_policy_version_raises(bundle_dir: Path):
    # The policy file still says "policy-v1"; the manifest now claims "policy-v2".
    _edit_manifest(bundle_dir, lambda d: d["components"]["policy"].update(version="policy-v2"))
    with pytest.raises(ValueError, match="policy: version mismatch"):
        load_bundle(bundle_dir)


def test_edited_manifest_profile_version_raises(bundle_dir: Path):
    # Closes Limitation 1 of the format step: the profile file says preprocessing-v1.
    _edit_manifest(bundle_dir, lambda d: d["components"]["profile"].update(version="preprocessing-v2"))
    with pytest.raises(ValueError, match="profile: profile_version mismatch"):
        load_bundle(bundle_dir)


def test_edited_manifest_checkpoint_version_raises(bundle_dir: Path):
    # checkpoint.json says cnn-v1; the manifest now claims cnn-v2.
    _edit_manifest(bundle_dir, lambda d: d["components"]["checkpoint"].update(version="cnn-v2"))
    with pytest.raises(ValueError, match="checkpoint: version mismatch"):
        load_bundle(bundle_dir)


def test_edited_manifest_changes_manifest_sha256(bundle_dir: Path):
    # A field no file confirms (here the bundle version) still changes the manifest bytes, so it no
    # longer matches the manifest_sha256 recorded in the F4 decision.
    before = load_bundle(bundle_dir).manifest_sha256
    _edit_manifest(bundle_dir, lambda d: d.update(bundle_version="bundle-v2"))
    assert load_bundle(bundle_dir).manifest_sha256 != before


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(schema_version=1),                          # retired schema (D59 (2))
    lambda d: d.update(schema_version=3),                          # unknown schema
    lambda d: d.pop("code_commit"),                                # missing key
    lambda d: d.update(extra=1),                                   # unknown key
    lambda d: d["components"].pop("profile"),                      # missing component
    lambda d: d["components"].update(extra=d["components"]["profile"]),  # unknown component
    lambda d: d["components"]["policy"].update(size=1),            # unknown component key
    lambda d: d["components"]["checkpoint"].update(file="x"),      # unknown checkpoint key
    lambda d: d["components"]["checkpoint"]["files"].update(extra="0" * 64),   # files keys wrong
    lambda d: d["components"]["checkpoint"]["files"].pop(WEIGHTS),             # files keys wrong
    lambda d: d["components"]["checkpoint"].update(files=["checkpoint.json"]), # files not an object
    lambda d: d["components"]["checkpoint"].update(sha256="0" * 64),  # combined != files
    lambda d: d["components"]["checkpoint"].update(dir="ckpt"),    # dir other than checkpoint
    lambda d: d["components"]["profile"].update(file="checkpoint"),  # reserved name
    lambda d: d.update(code_commit="ABC"),                         # not 40 lowercase hex
    lambda d: d.update(created_at="2026-10-17 12:00:00"),          # not ISO 8601 UTC
    lambda d: d.update(protocol_version="v1"),                     # not like "1.0"
])
def test_invalid_manifest_raises(bundle_dir: Path, mutate):
    # Strict schema at every level: anything not written by to_json is refused at parse.
    _edit_manifest(bundle_dir, mutate)
    with pytest.raises(ValueError, match="manifest"):
        load_bundle(bundle_dir)


def test_schema_1_manifest_is_named_as_retired(bundle_dir: Path):
    # A schema-1 manifest (old flat layout) must fail on the schema version, not on a key list.
    _edit_manifest(bundle_dir, lambda d: d.update(schema_version=1))
    with pytest.raises(ValueError, match="schema_version must be 2, got 1"):
        load_bundle(bundle_dir)


# 5. Cross-checks inside a consistently re-hashed bundle.
def test_fingerprint_mismatch_in_rehashed_bundle_raises(bundle_dir: Path):
    # checkpoint.json is replaced by one naming another profile fingerprint, and the manifest is
    # rebuilt with correct hashes, so no hash check fails: only the cross-check can catch it.
    (bundle_dir / CHECKPOINT_DIR / CKPT_JSON).write_bytes(_ckpt_json(fingerprint="f" * 64))
    _rehash_manifest(bundle_dir)
    with pytest.raises(ValueError, match="checkpoint: profile_fingerprint mismatch"):
        load_bundle(bundle_dir)


def test_invalid_checkpoint_json_in_rehashed_bundle_raises(bundle_dir: Path):
    # checkpoint.json must be a JSON object with string version and profile_fingerprint.
    (bundle_dir / CHECKPOINT_DIR / CKPT_JSON).write_bytes(b'{"version": 1}')
    _rehash_manifest(bundle_dir)
    with pytest.raises(ValueError, match="checkpoint: checkpoint.json version must be a string"):
        load_bundle(bundle_dir)


def test_invalid_profile_in_rehashed_bundle_raises(bundle_dir: Path):
    # A re-hashed profile that is not a valid Stage 6 profile is refused when parsed.
    path = bundle_dir / PROFILE_NAME
    path.write_bytes(path.read_bytes().replace(b"n_mels: 64", b"n_mels: 80"))
    _rehash_manifest(bundle_dir)
    with pytest.raises(ValueError, match="profile: invalid profile file"):
        load_bundle(bundle_dir)


# 6. Path safety.
@pytest.mark.parametrize("bad_name", ["../x", "/abs/x", "C:\\abs\\x", "sub/x", "sub\\x", "..", "."])
def test_unsafe_component_path_raises_and_reads_nothing(tmp_path: Path, bundle_dir: Path, monkeypatch, bad_name):
    # A file named "x" exists both outside the bundle and in a sub-folder, so a loader that joined
    # the path would find something to read. The plain-name check runs while the manifest is parsed,
    # before any component file is opened, so nothing is hashed and no file is read.
    (tmp_path / "x").write_bytes(b"outside")
    (bundle_dir / "sub").mkdir()
    (bundle_dir / "sub" / "x").write_bytes(b"nested")
    _edit_manifest(bundle_dir, lambda d: d["components"]["profile"].update(file=bad_name))
    hashed, reads = [], []
    real_read_bytes = Path.read_bytes
    monkeypatch.setattr(bundle_module, "sha256_file", lambda p: hashed.append(p) or "0" * 64)
    monkeypatch.setattr(Path, "read_bytes", lambda self: reads.append(self) or real_read_bytes(self))
    with pytest.raises(ValueError, match="plain file name"):
        load_bundle(bundle_dir)
    assert hashed == []
    assert reads == [bundle_dir / MANIFEST_NAME]


# 7. Folder contents.
def test_extra_top_file_raises(bundle_dir: Path):
    # Only the manifest, the profile, the policy and checkpoint/ may be present.
    (bundle_dir / "notes.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match=r"extra \['notes.txt'\]"):
        load_bundle(bundle_dir)


@pytest.mark.parametrize("name", [PROFILE_NAME, "policy.json"])
def test_missing_top_file_raises(bundle_dir: Path, name: str):
    (bundle_dir / name).unlink()
    with pytest.raises(ValueError, match=rf"missing \['{name}'\]"):
        load_bundle(bundle_dir)


def test_missing_manifest_raises(bundle_dir: Path):
    (bundle_dir / MANIFEST_NAME).unlink()
    with pytest.raises(ValueError, match="manifest"):
        load_bundle(bundle_dir)


def test_extra_file_in_checkpoint_dir_raises(bundle_dir: Path):
    # load_checkpoint also refuses this, but the bundle check must not depend on the caller.
    (bundle_dir / CHECKPOINT_DIR / "extra.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match=r"checkpoint: checkpoint/ contents .* extra \['extra.bin'\]"):
        load_bundle(bundle_dir)


@pytest.mark.parametrize("name", CHECKPOINT_FILES)
def test_missing_file_in_checkpoint_dir_raises(bundle_dir: Path, name: str):
    (bundle_dir / CHECKPOINT_DIR / name).unlink()
    with pytest.raises(ValueError, match=rf"checkpoint: checkpoint/ contents .* missing \['{name}'\]"):
        load_bundle(bundle_dir)


def test_checkpoint_dir_replaced_by_file_raises(bundle_dir: Path):
    # The name is present, so the top-level listing passes; it must be a real directory.
    for name in CHECKPOINT_FILES:
        (bundle_dir / CHECKPOINT_DIR / name).unlink()
    (bundle_dir / CHECKPOINT_DIR).rmdir()
    (bundle_dir / CHECKPOINT_DIR).write_bytes(b"not a folder")
    with pytest.raises(ValueError, match="checkpoint: checkpoint is not a real directory"):
        load_bundle(bundle_dir)


# 8. Policy.
def test_policy_version_mismatch_at_load_raises(bundle_dir: Path):
    # Replace policy.json by a valid policy with another version and update its hash, so only the
    # version check can catch it.
    other = _policy("policy-v2").to_json().encode("utf-8")
    (bundle_dir / "policy.json").write_bytes(other)
    _edit_manifest(bundle_dir, lambda d: d["components"]["policy"].update(sha256=_sha(other)))
    with pytest.raises(ValueError, match=r"policy: version mismatch \(manifest 'policy-v1', file 'policy-v2'\)"):
        load_bundle(bundle_dir)


def test_invalid_policy_at_load_raises(bundle_dir: Path):
    # Hash updated, so the parse step is what must fail (NaN threshold is rejected by RiskPolicy).
    bad = _policy().to_json().replace('"threshold": 0.5', '"threshold": NaN').encode("utf-8")
    (bundle_dir / "policy.json").write_bytes(bad)
    _edit_manifest(bundle_dir, lambda d: d["components"]["policy"].update(sha256=_sha(bad)))
    with pytest.raises(ValueError, match="policy: invalid policy file"):
        load_bundle(bundle_dir)


# 9. Single read.
def _swap_after_read(monkeypatch, swaps: dict[Path, bytes]) -> list[Path]:
    """Path.read_bytes that overwrites the file with other bytes right after reading it."""
    real_read_bytes, real_write_bytes = Path.read_bytes, Path.write_bytes
    reads = []

    def read_bytes(self):
        data = real_read_bytes(self)
        if self in swaps:
            reads.append(self)
            real_write_bytes(self, swaps[self])
        return data

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    return reads


def test_policy_is_parsed_from_the_hashed_bytes(bundle_dir: Path, monkeypatch):
    # policy.json must be read once: the bytes that are hashed are the bytes that are parsed. Right
    # after that read, the file is swapped for a valid policy-v2; a loader that read the file again
    # to parse it would return (or reject) policy-v2. Reading once, it returns the verified policy-v1.
    policy_path = bundle_dir / "policy.json"
    reads = _swap_after_read(monkeypatch, {policy_path: _policy("policy-v2").to_json().encode("utf-8")})
    hashed = []
    real_sha256_file = bundle_module.sha256_file
    monkeypatch.setattr(bundle_module, "sha256_file", lambda p: hashed.append(p) or real_sha256_file(p))

    loaded = load_bundle(bundle_dir)
    assert loaded.policy == _policy("policy-v1")
    assert reads == [policy_path]
    assert policy_path not in hashed


def test_profile_and_checkpoint_json_are_parsed_from_the_hashed_bytes(bundle_dir: Path, monkeypatch):
    # Same technique for the two other parsed files. The profile is swapped for one that fails to
    # parse, checkpoint.json for one with another version and fingerprint; a loader that re-read
    # either file would raise. Reading once, it returns the verified profile and passes the
    # cross-checks against the verified checkpoint.json.
    profile_path = bundle_dir / PROFILE_NAME
    ckpt_json_path = bundle_dir / CHECKPOINT_DIR / CKPT_JSON
    reads = _swap_after_read(monkeypatch, {
        profile_path: b"not: [a, valid, profile",
        ckpt_json_path: _ckpt_json(version="cnn-v2", fingerprint="f" * 64),
    })
    hashed = []
    real_sha256_file = bundle_module.sha256_file
    monkeypatch.setattr(bundle_module, "sha256_file", lambda p: hashed.append(p) or real_sha256_file(p))

    loaded = load_bundle(bundle_dir)
    assert loaded.profile == load_profile(PROFILE_SRC)
    assert sorted(reads) == sorted([profile_path, ckpt_json_path])
    assert profile_path not in hashed
    assert ckpt_json_path not in hashed
    # The weights are hashed streamed, not parsed, here.
    assert hashed == [bundle_dir / CHECKPOINT_DIR / WEIGHTS]


# 10. Immutability.
def test_build_into_non_empty_directory_raises(tmp_path: Path, bundle_dir: Path):
    # Bundles are never overwritten: the existing folder keeps exactly its bytes.
    before = _tree(bundle_dir)
    with pytest.raises(ValueError, match="not empty"):
        _build(bundle_dir, tmp_path / "src2", bundle_version="bundle-v2")
    assert _tree(bundle_dir) == before


def test_loaded_bundle_is_immutable(bundle_dir: Path):
    # The frozen manifest and its component maps cannot be changed through the loaded object.
    loaded = load_bundle(bundle_dir)
    with pytest.raises(AttributeError):
        loaded.checkpoint_sha256 = "0" * 64
    with pytest.raises(TypeError):
        loaded.manifest.components["checkpoint"] = None
    with pytest.raises(TypeError):
        loaded.manifest.components["checkpoint"].files[WEIGHTS] = "0" * 64


@pytest.mark.parametrize("pre_create", [False, True])
@pytest.mark.parametrize("fail_at", [1, 2, 3, 4, 5])
def test_failed_build_removes_what_it_wrote(tmp_path: Path, monkeypatch, pre_create: bool, fail_at: int):
    # Writes are: profile, policy.json, checkpoint/checkpoint.json, checkpoint/weights.safetensors,
    # manifest. Whichever fails (after writing part of the file), no half-built bundle may remain: a
    # folder build_bundle created (the bundle folder, checkpoint/) is removed, a pre-existing empty
    # bundle folder is empty again, and the original error is re-raised.
    directory = tmp_path / "bundle"
    if pre_create:
        directory.mkdir()
    src = _sources(tmp_path / "src")
    real_write = bundle_module._write_file
    calls = []

    def write(path, data):
        calls.append(path)
        if len(calls) == fail_at:
            Path(path).write_bytes(b"half")  # a partly written file must be removed too
            raise OSError("disk full")
        return real_write(path, data)

    kwargs = _kwargs(src)
    monkeypatch.setattr(bundle_module, "_write_file", write)
    with pytest.raises(OSError, match="disk full"):
        build_bundle(directory, **kwargs)
    assert len(calls) == fail_at
    if pre_create:
        assert directory.is_dir()
        assert list(directory.iterdir()) == []
    else:
        assert not directory.exists()


def test_build_into_empty_directory_is_allowed(tmp_path: Path):
    directory = tmp_path / "bundle"
    directory.mkdir()
    _build(directory, tmp_path / "src")
    assert load_bundle(directory).manifest.bundle_version == "bundle-v1"


# 11. Determinism.
def test_build_is_deterministic(tmp_path: Path):
    # Same inputs and created_at -> same manifest bytes (sorted keys, fixed layout), so the manifest
    # SHA-256 recorded in F4 can be reproduced.
    a = _build(tmp_path / "a", tmp_path / "src")
    b = _build(tmp_path / "b", tmp_path / "src")
    assert a.read_bytes() == b.read_bytes()
    assert _tree(tmp_path / "a") == _tree(tmp_path / "b")


# 12. Layering.
def _imported_modules(path: Path) -> list[str]:
    """Absolute module names imported by a source file (relative imports resolved inside src.bundle)."""
    names = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # from .x / ..x: resolve against the src.bundle package
                base = ["src", "bundle"][: max(0, 2 - (node.level - 1))]
                names.append(".".join(base + ([node.module] if node.module else [])))
            else:
                names.append(node.module or "")
    return names


def _offending(path: Path, banned: tuple[str, ...]) -> list[str]:
    return [n for n in _imported_modules(path) for b in banned if n == b or n.startswith(f"{b}.")]


def test_bundle_does_not_import_research_code():
    # The API loads bundles but has no Evaluation or Tuning service (SDD Fig. 1).
    files = sorted(BUNDLE_DIR.glob("*.py"))
    assert files, f"no source files found in {BUNDLE_DIR}"
    assert {f.name: _offending(f, ("src.evaluation", "src.tuning", "evaluation", "tuning"))
            for f in files} == {f.name: [] for f in files}


def test_bundle_module_is_torch_free():
    # load_bundle stays torch-free (D59 (5)): no direct import, and none pulled in indirectly.
    assert _offending(BUNDLE_DIR / "bundle.py",
                      ("torch", "safetensors", "src.models", "src.evaluation", "src.tuning")) == []
    code = ("import sys, src.bundle.bundle; "
            "print([m for m in ('torch', 'safetensors', 'src.models') if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
