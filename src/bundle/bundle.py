"""ModelBundle format and verifying loader (Protocol v1.0 §6 and §11; D2; D59; AGENTS.md hard rule 2).

A ModelBundle is one folder (D59 (1)):

    <bundle>/
      bundle_manifest.json
      policy.json                 bytes of the source policy file, unchanged
      <profile file name>         e.g. preprocessing.yaml, bytes unchanged
      checkpoint/
        checkpoint.json           bytes unchanged (Stage 8 D42 checkpoint)
        weights.safetensors       bytes unchanged

The manifest (schema 2) records the SHA-256 and version of each part, the code commit, the protocol
version and the F2 hash ledger (D28). For the checkpoint it records both file hashes and the
combined D42 checkpoint_sha256, the value Stage 8 recorded. It is frozen in F4 (Stage 12) and is
what the API loads (SDD v2.0 §3.5 and §6.4, SRS v4.0 FR-29).

- Verification before use (NFR-09; threat model §10, tampering): load_bundle checks the manifest,
  the folder contents, every SHA-256, the policy version and the cross-checks between the parts
  before it returns anything, so a replaced or altered part fails loudly instead of running.
- Cross-checks (D59 (4); FR-04, NFR-09): the profile's version, protocol version and F2 ledger equal
  the manifest's; checkpoint.json's version equals the manifest model version and its
  profile_fingerprint equals the bundled profile's fingerprint.
- Path safety: profile and policy are plain names inside the bundle folder, the checkpoint is the
  fixed folder "checkpoint/" with two fixed names; a manifest naming "../x", an absolute path or
  "sub/x" is rejected before any component file is opened.
- Immutability: build_bundle never writes into a non-empty folder, and load_bundle only reads.
- The manifest cannot hash itself, so load_bundle returns its SHA-256 (manifest_sha256) and the F4
  freeze decision records it; that external anchor also covers fields no file can confirm.
- Single read: the manifest, policy.json, the profile and checkpoint.json are read once; the bytes
  that are hashed are the bytes that are parsed, so a file swapped between the check and the parse
  cannot slip through.
- Partial build cleanup: if build_bundle fails after it starts writing, every file and folder it
  wrote is removed before the error is re-raised.

Closed by D59 (were the limitations of the format step):
- The profile version is checked: the profile is parsed from its hashed bytes and compared with the
  manifest (LoadedBundle.profile is that parsed profile).
- Check-then-use for the checkpoint: src.bundle.model_loader.load_bundle_model loads it with
  load_checkpoint(..., expected_sha256=<manifest value>), which hashes the bytes it loads.

What remains:
- profile_path is still a path a caller could re-read after the check; the API must use
  LoadedBundle.profile, never re-load the profile from profile_path.
- weights.safetensors is hashed here but not parsed (no torch here); its content is checked by
  load_checkpoint in model_loader.

Part of the shared core (SDD Fig. 1): the API loads bundles, so this module must not import
src.evaluation or src.tuning (research code), and it stays torch-free: no torch, safetensors or
src.models import.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Mapping

from src.preprocessing.profile import PreprocessingError, PreprocessingProfile, parse_profile
from src.risk.policy import RiskPolicy

MANIFEST_NAME = "bundle_manifest.json"
SCHEMA_VERSION = 2
COMPONENTS = ("checkpoint", "profile", "policy")
POLICY_FILE = "policy.json"
CHECKPOINT_DIR = "checkpoint"
# Same names and order as src.models.cnn_checkpoint.FILES (not imported: this module is torch-free).
CHECKPOINT_FILES = ("checkpoint.json", "weights.safetensors")
CHECKPOINT_JSON = CHECKPOINT_FILES[0]

_CHUNK = 1 << 20  # 1 MiB: checkpoints can be large, so hashes are streamed
_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEX40 = re.compile(r"[0-9a-f]{40}")
_PROTOCOL = re.compile(r"[0-9]+\.[0-9]+")
_UTC = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]{1,6})?Z")

_COMPONENT_KEYS = {"file", "sha256", "version"}
_CHECKPOINT_KEYS = {"dir", "files", "sha256", "version"}
_MANIFEST_KEYS = {
    "schema_version", "bundle_version", "created_at", "code_commit", "protocol_version",
    "f2_ledger_sha256", "components",
}


def _require_nonempty_str(name: str, value) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")


def _require_match(name: str, pattern: re.Pattern, value, what: str) -> None:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"{name} must be {what}, got {value!r}")


def _require_plain_name(name: str, value) -> None:
    """A file name with no directory part: no separators, not '.', '..', absolute or a drive."""
    _require_nonempty_str(name, value)
    if (
        value in {".", ".."}
        or PurePosixPath(value).name != value
        or PureWindowsPath(value).name != value
        or PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
    ):
        raise ValueError(f"{name} must be a plain file name inside the bundle, got {value!r}")


def _check_keys(where: str, data: dict, expected: set[str]) -> None:
    missing = sorted(expected - data.keys())
    unknown = sorted(data.keys() - expected)
    if missing or unknown:
        raise ValueError(f"{where}: missing keys {missing}, unknown keys {unknown}")


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's bytes, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checkpoint_dir_sha256(files: Mapping[str, str]) -> str:
    """The combined D42 checkpoint hash from the two per-file SHA-256 values.

    Same rule as src.models.cnn_checkpoint.checkpoint_sha256 (its _sha256_lines): SHA-256 of the
    ASCII text "checkpoint.json:<sha256>\\nweights.safetensors:<sha256>\\n". It is restated here so
    this module stays torch-free; tests/bundle/test_model_loader.py proves both give the same value
    on a real checkpoint.
    """
    if not isinstance(files, Mapping) or set(files) != set(CHECKPOINT_FILES):
        keys = sorted(files) if isinstance(files, Mapping) else files
        raise ValueError(f"checkpoint files must be exactly {list(CHECKPOINT_FILES)}, got {keys!r}")
    for name in CHECKPOINT_FILES:
        _require_match(f"files[{name!r}]", _HEX64, files[name], "64 lowercase hex characters")
    lines = "".join(f"{name}:{files[name]}\n" for name in CHECKPOINT_FILES)
    return _sha256(lines.encode("ascii"))


@dataclass(frozen=True)
class Component:
    """One single-file bundle part (profile, policy): plain file name, SHA-256 of its bytes, version."""

    file: str
    sha256: str
    version: str

    def __post_init__(self) -> None:
        _require_plain_name("file", self.file)
        if self.file in {MANIFEST_NAME, CHECKPOINT_DIR}:
            raise ValueError(f"file must not be {MANIFEST_NAME!r} or {CHECKPOINT_DIR!r}, got {self.file!r}")
        _require_match("sha256", _HEX64, self.sha256, "64 lowercase hex characters")
        _require_nonempty_str("version", self.version)


@dataclass(frozen=True)
class CheckpointComponent:
    """The checkpoint part (D59 (2)): the fixed folder, both file hashes, the combined D42 hash, version.

    sha256 must equal checkpoint_dir_sha256(files), so a manifest whose combined value does not
    follow from its per-file values is refused when it is parsed.
    """

    dir: str
    files: Mapping[str, str]
    sha256: str
    version: str

    def __post_init__(self) -> None:
        if self.dir != CHECKPOINT_DIR:
            raise ValueError(f"dir must be {CHECKPOINT_DIR!r}, got {self.dir!r}")
        expected = checkpoint_dir_sha256(self.files)  # also checks the keys and every per-file hash
        _require_match("sha256", _HEX64, self.sha256, "64 lowercase hex characters")
        if self.sha256 != expected:
            raise ValueError(
                f"sha256 {self.sha256} is not the combined hash of files (expected {expected})"
            )
        _require_nonempty_str("version", self.version)
        # Read-only copy in the fixed order, so the frozen component cannot be changed through it.
        object.__setattr__(
            self, "files", MappingProxyType({name: self.files[name] for name in CHECKPOINT_FILES})
        )


@dataclass(frozen=True)
class BundleManifest:
    """Contents of bundle_manifest.json. components maps exactly COMPONENTS to their parts."""

    schema_version: int
    bundle_version: str
    created_at: str
    code_commit: str
    protocol_version: str
    f2_ledger_sha256: str
    components: Mapping[str, Component | CheckpointComponent]

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}, got {self.schema_version!r}")
        _require_nonempty_str("bundle_version", self.bundle_version)
        _require_match("created_at", _UTC, self.created_at, "ISO 8601 UTC (YYYY-MM-DDTHH:MM:SSZ)")
        try:
            datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(f"created_at is not a valid date and time, got {self.created_at!r}") from None
        _require_match("code_commit", _HEX40, self.code_commit, "40 lowercase hex characters")
        _require_match("protocol_version", _PROTOCOL, self.protocol_version, "like '1.0'")
        _require_match("f2_ledger_sha256", _HEX64, self.f2_ledger_sha256, "64 lowercase hex characters")
        if not isinstance(self.components, Mapping) or set(self.components) != set(COMPONENTS):
            keys = sorted(self.components) if isinstance(self.components, Mapping) else self.components
            raise ValueError(f"components must have exactly the keys {list(COMPONENTS)}, got {keys!r}")
        if not isinstance(self.components["checkpoint"], CheckpointComponent):
            raise ValueError("components['checkpoint'] must be a CheckpointComponent")
        for name in ("profile", "policy"):
            if not isinstance(self.components[name], Component):
                raise ValueError(f"components[{name!r}] must be a Component")
        files = [self.components["profile"].file, self.components["policy"].file]
        if len(set(files)) != len(files):
            raise ValueError(f"component files must be distinct, got {files}")
        # Read-only copy in a fixed order, so the frozen manifest cannot be changed through it.
        object.__setattr__(
            self, "components", MappingProxyType({name: self.components[name] for name in COMPONENTS})
        )

    def to_json(self) -> str:
        """Sorted keys, 2-space indent, trailing newline: same manifest -> same bytes."""
        checkpoint = self.components["checkpoint"]
        payload = {
            "schema_version": self.schema_version,
            "bundle_version": self.bundle_version,
            "created_at": self.created_at,
            "code_commit": self.code_commit,
            "protocol_version": self.protocol_version,
            "f2_ledger_sha256": self.f2_ledger_sha256,
            "components": {
                "checkpoint": {
                    "dir": checkpoint.dir,
                    "files": dict(checkpoint.files),
                    "sha256": checkpoint.sha256,
                    "version": checkpoint.version,
                },
                **{
                    name: {"file": c.file, "sha256": c.sha256, "version": c.version}
                    for name, c in self.components.items()
                    if name != "checkpoint"
                },
            },
        }
        return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, text: str) -> BundleManifest:
        """Load a manifest written by to_json; ValueError on any schema or value problem."""
        data = json.loads(text)  # json.JSONDecodeError is a ValueError
        if not isinstance(data, dict):
            raise ValueError(f"manifest JSON must be an object, got {type(data).__name__}")
        _check_keys("manifest JSON", data, _MANIFEST_KEYS)
        # Checked first, so a retired schema is named as such rather than as a key error below.
        schema = data["schema_version"]
        if isinstance(schema, bool) or schema != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}, got {schema!r}")
        components = data["components"]
        if not isinstance(components, dict):
            raise ValueError(f"components must be an object, got {type(components).__name__}")
        _check_keys("manifest components", components, set(COMPONENTS))
        parsed = {}
        for name in COMPONENTS:
            entry = components[name]
            if not isinstance(entry, dict):
                raise ValueError(f"components.{name} must be an object, got {type(entry).__name__}")
            if name == "checkpoint":
                _check_keys("manifest components.checkpoint", entry, _CHECKPOINT_KEYS)
                if not isinstance(entry["files"], dict):
                    raise ValueError(
                        f"components.checkpoint.files must be an object, got {type(entry['files']).__name__}"
                    )
                _check_keys("manifest components.checkpoint.files", entry["files"], set(CHECKPOINT_FILES))
                make = CheckpointComponent
            else:
                _check_keys(f"manifest components.{name}", entry, _COMPONENT_KEYS)
                make = Component
            try:
                parsed[name] = make(**entry)
            except ValueError as e:
                raise ValueError(f"components.{name}: {e}") from None
        return cls(**{**data, "components": parsed})


@dataclass(frozen=True)
class LoadedBundle:
    """A bundle that passed every check in load_bundle.

    profile is the profile parsed from the verified bytes: use it, not profile_path (D59 (4)).
    checkpoint_sha256 is the manifest's combined hash: pass it to load_checkpoint as expected_sha256
    (src.bundle.model_loader does this).
    """

    directory: Path
    manifest: BundleManifest
    checkpoint_dir: Path
    checkpoint_sha256: str
    profile_path: Path
    profile: PreprocessingProfile
    policy: RiskPolicy
    manifest_sha256: str


def _parse_profile_bytes(data: bytes, source: str) -> PreprocessingProfile:
    try:
        return parse_profile(data.decode("utf-8"), source=source)
    except (PreprocessingError, UnicodeDecodeError) as e:
        raise ValueError(f"profile: invalid profile file {source}: {e}") from None


def _parse_checkpoint_json(data: bytes) -> dict:
    """checkpoint.json as a JSON object with string version and profile_fingerprint (no torch).

    The full metadata schema is checked by load_checkpoint; here only what the cross-checks need.
    """
    try:
        meta = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"checkpoint: {CHECKPOINT_JSON} is not valid JSON: {e}") from None
    if not isinstance(meta, dict):
        raise ValueError(f"checkpoint: {CHECKPOINT_JSON} must be a JSON object, got {type(meta).__name__}")
    for key in ("version", "profile_fingerprint"):
        if not isinstance(meta.get(key), str):
            raise ValueError(f"checkpoint: {CHECKPOINT_JSON} {key} must be a string, got {meta.get(key)!r}")
    return meta


def _cross_check(meta: dict, profile: PreprocessingProfile, *, model_version: str) -> None:
    """checkpoint.json against the model version and the profile it was trained with (D59 (4))."""
    if meta["version"] != model_version:
        raise ValueError(
            f"checkpoint: version mismatch (manifest {model_version!r}, {CHECKPOINT_JSON} {meta['version']!r})"
        )
    fingerprint = profile.fingerprint()
    if meta["profile_fingerprint"] != fingerprint:
        raise ValueError(
            f"checkpoint: profile_fingerprint mismatch ({CHECKPOINT_JSON} {meta['profile_fingerprint']}, "
            f"bundled profile {fingerprint})"
        )


def _profile_check(profile: PreprocessingProfile, *, version: str, protocol: str, ledger: str, where: str) -> None:
    for field, actual, expected in (
        ("profile_version", profile.profile_version, version),
        ("protocol_version", profile.protocol_version, protocol),
        ("f2_ledger_sha256", profile.f2_ledger_sha256, ledger),
    ):
        if actual != expected:
            raise ValueError(f"profile: {field} mismatch ({where} {expected!r}, profile {actual!r})")


def _read_checkpoint_source(checkpoint_dir: Path) -> dict[str, bytes]:
    """The bytes of exactly CHECKPOINT_FILES (regular files, no symlinks), each read once."""
    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise ValueError(f"checkpoint: source {checkpoint_dir} is not a folder")
    names = sorted(p.name for p in checkpoint_dir.iterdir())
    if names != sorted(CHECKPOINT_FILES):
        raise ValueError(f"checkpoint: source {checkpoint_dir} must hold exactly {sorted(CHECKPOINT_FILES)}, found {names}")
    data = {}
    for name in CHECKPOINT_FILES:
        path = checkpoint_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"checkpoint: source {path} must be a regular file")
        data[name] = path.read_bytes()
    return data


def _read_source(name: str, path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name}: source {path} is not a regular file")
    return path.read_bytes()


def _write_file(path: Path, data: bytes) -> None:
    """One bundle file write (a seam for the failed-build tests)."""
    path.write_bytes(data)


def build_bundle(
    directory,
    *,
    checkpoint_dir,
    expected_checkpoint_sha256: str,
    profile,
    policy,
    bundle_version: str,
    model_version: str,
    profile_version: str,
    code_commit: str,
    protocol_version: str,
    f2_ledger_sha256: str,
    created_at: str,
) -> Path:
    """Write a new bundle folder and return the manifest path (D59 (1)-(3)).

    checkpoint_dir is a D42 checkpoint folder; its combined hash must equal
    expected_checkpoint_sha256 (the value Stage 8 recorded). profile and policy are file paths; all
    four files are copied byte for byte. Every input is read once and checked before anything is
    written; the manifest is written last.
    """
    directory = Path(directory)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError(f"bundle directory {directory} exists and is not empty; bundles are never overwritten")
    _require_match("expected_checkpoint_sha256", _HEX64, expected_checkpoint_sha256, "64 lowercase hex characters")

    # 1. Checkpoint: exactly the two files, and the combined hash is the recorded one.
    checkpoint_bytes = _read_checkpoint_source(Path(checkpoint_dir))
    checkpoint_files = {name: _sha256(data) for name, data in checkpoint_bytes.items()}
    combined = checkpoint_dir_sha256(checkpoint_files)
    if combined != expected_checkpoint_sha256:
        raise ValueError(
            f"checkpoint: checkpoint_sha256 mismatch (expected {expected_checkpoint_sha256}, source {combined})"
        )

    # 2. Policy: the file's own bytes, copied unchanged, so its hash is the source file's.
    policy_bytes = _read_source("policy", Path(policy))
    try:
        risk_policy = RiskPolicy.from_json(policy_bytes.decode("utf-8"))
    except (ValueError, TypeError) as e:  # UnicodeDecodeError is a ValueError
        raise ValueError(f"policy: invalid policy file {policy}: {e}") from None

    # 3. Profile: parsed from the bytes that are copied, and matching the declared versions.
    profile = Path(profile)
    profile_bytes = _read_source("profile", profile)
    parsed_profile = _parse_profile_bytes(profile_bytes, str(profile))
    _profile_check(parsed_profile, version=profile_version, protocol=protocol_version,
                   ledger=f2_ledger_sha256, where="build argument")

    # 4. checkpoint.json: the model version and the profile it was trained with.
    _cross_check(_parse_checkpoint_json(checkpoint_bytes[CHECKPOINT_JSON]), parsed_profile,
                 model_version=model_version)

    manifest = BundleManifest(
        schema_version=SCHEMA_VERSION,
        bundle_version=bundle_version,
        created_at=created_at,
        code_commit=code_commit,
        protocol_version=protocol_version,
        f2_ledger_sha256=f2_ledger_sha256,
        components={
            "checkpoint": CheckpointComponent(
                dir=CHECKPOINT_DIR, files=checkpoint_files, sha256=combined, version=model_version
            ),
            "profile": Component(file=profile.name, sha256=_sha256(profile_bytes), version=profile_version),
            "policy": Component(file=POLICY_FILE, sha256=_sha256(policy_bytes), version=risk_policy.version),
        },
    )

    # 5. Write: the bytes read above, each copy re-hashed, the manifest last.
    outputs = [
        ("profile", directory / profile.name, profile_bytes, manifest.components["profile"].sha256),
        ("policy", directory / POLICY_FILE, policy_bytes, manifest.components["policy"].sha256),
        *[
            ("checkpoint", directory / CHECKPOINT_DIR / name, checkpoint_bytes[name], checkpoint_files[name])
            for name in CHECKPOINT_FILES
        ],
    ]
    created_dir = not directory.exists()
    created_checkpoint_dir = False
    written: list[Path] = []
    try:
        directory.mkdir(parents=True, exist_ok=True)
        for name, dest, data, sha in outputs:
            if dest.parent != directory and not dest.parent.exists():
                dest.parent.mkdir()
                created_checkpoint_dir = True
            written.append(dest)  # before the write, so a half-written file is removed too
            _write_file(dest, data)
            if sha256_file(dest) != sha:
                raise ValueError(f"{name}: written file {dest} does not match its source bytes")
        manifest_path = directory / MANIFEST_NAME
        written.append(manifest_path)
        _write_file(manifest_path, manifest.to_json().encode("utf-8"))
    except BaseException:
        # No half-built bundle is left behind: remove what this call wrote, then re-raise.
        for path in written:
            path.unlink(missing_ok=True)
        if created_checkpoint_dir and (directory / CHECKPOINT_DIR).is_dir():
            (directory / CHECKPOINT_DIR).rmdir()
        if created_dir and directory.is_dir():
            directory.rmdir()
        raise
    return manifest_path


def load_bundle(directory) -> LoadedBundle:
    """Verify a bundle folder and return it; read-only. ValueError naming the first failed check."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"bundle directory {directory} does not exist or is not a directory")

    # 1. Manifest, read once; manifest_sha256 comes from these bytes.
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"manifest {MANIFEST_NAME} is missing in {directory}")
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = BundleManifest.from_json(manifest_bytes.decode("utf-8"))
    except (ValueError, TypeError) as e:  # UnicodeDecodeError is a ValueError
        raise ValueError(f"manifest {MANIFEST_NAME}: {e}") from None
    checkpoint = manifest.components["checkpoint"]
    profile_component = manifest.components["profile"]
    policy_component = manifest.components["policy"]

    # 2. Folder contents. Plain names were checked by the manifest parse, so every path stays inside.
    expected = {MANIFEST_NAME, CHECKPOINT_DIR, profile_component.file, policy_component.file}
    present = {p.name for p in directory.iterdir()}
    if present != expected:
        raise ValueError(
            f"bundle contents differ from the manifest: missing {sorted(expected - present)}, "
            f"extra {sorted(present - expected)}"
        )
    checkpoint_dir = directory / CHECKPOINT_DIR
    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise ValueError(f"checkpoint: {CHECKPOINT_DIR} is not a real directory")
    present = {p.name for p in checkpoint_dir.iterdir()}
    if present != set(CHECKPOINT_FILES):
        raise ValueError(
            f"checkpoint: {CHECKPOINT_DIR}/ contents differ from the manifest: "
            f"missing {sorted(set(CHECKPOINT_FILES) - present)}, extra {sorted(present - set(CHECKPOINT_FILES))}"
        )

    # 3. Hashes. Files that are parsed below are read once here: hashed bytes = parsed bytes.
    def check(name: str, rel: str, path: Path, recorded: str, *, keep: bool) -> tuple[bytes | None, str]:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{name}: {rel} is not a regular file")
        data = path.read_bytes() if keep else None
        actual = _sha256(data) if keep else sha256_file(path)
        if actual != recorded:
            raise ValueError(f"{name}: SHA-256 mismatch for {rel} (manifest {recorded}, file {actual})")
        return data, actual

    actual_files = {}
    checkpoint_json = b""
    for file_name in CHECKPOINT_FILES:
        rel = f"{CHECKPOINT_DIR}/{file_name}"
        data, actual_files[file_name] = check(
            "checkpoint", rel, checkpoint_dir / file_name, checkpoint.files[file_name],
            keep=file_name == CHECKPOINT_JSON,
        )
        if data is not None:
            checkpoint_json = data
    # Redundant with the per-file checks and the parse-time rule, kept as the explicit D42 anchor.
    actual_combined = checkpoint_dir_sha256(actual_files)
    if actual_combined != checkpoint.sha256:
        raise ValueError(
            f"checkpoint: checkpoint_sha256 mismatch (manifest {checkpoint.sha256}, files {actual_combined})"
        )
    profile_bytes, _ = check("profile", profile_component.file, directory / profile_component.file,
                             profile_component.sha256, keep=True)
    policy_bytes, _ = check("policy", policy_component.file, directory / policy_component.file,
                            policy_component.sha256, keep=True)

    # 4. Policy, from its hashed bytes.
    try:
        policy = RiskPolicy.from_json(policy_bytes.decode("utf-8"))
    except (ValueError, TypeError) as e:
        raise ValueError(f"policy: invalid policy file {policy_component.file}: {e}") from None
    if policy.version != policy_component.version:
        raise ValueError(
            f"policy: version mismatch (manifest {policy_component.version!r}, file {policy.version!r})"
        )

    # 5. Profile, from its hashed bytes, against the manifest (FR-04, NFR-09).
    profile = _parse_profile_bytes(profile_bytes, profile_component.file)
    _profile_check(profile, version=profile_component.version, protocol=manifest.protocol_version,
                   ledger=manifest.f2_ledger_sha256, where="manifest")

    # 6. checkpoint.json, from its hashed bytes, against the manifest and the bundled profile.
    _cross_check(_parse_checkpoint_json(checkpoint_json), profile, model_version=checkpoint.version)

    return LoadedBundle(
        directory=directory,
        manifest=manifest,
        checkpoint_dir=checkpoint_dir,
        checkpoint_sha256=checkpoint.sha256,
        profile_path=directory / profile_component.file,
        profile=profile,
        policy=policy,
        manifest_sha256=_sha256(manifest_bytes),
    )
