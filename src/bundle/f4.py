"""F4 build and verify: ModelBundle v1 from committed evidence (Stage 12; D2, D36, D59; Protocol v1.0 §12).

F4 is the load_bundle-verified bundle of the Stage 8 bundle-seed checkpoint, the F3 preprocessing
profile and risk-v1 (D59), with baseline-v1's artifact hash recorded beside it (D36). Every expected
value is read in code from committed evidence, never typed:

    bundle seed             stage8 and stage10 run_report.json  bundle_seed (must be equal)
    checkpoint_sha256       stage8 run_report.json  checkpoint_sha256[str(seed)] (= stage10's)
    per-file hashes, sizes  stage8 stage8_export_manifest.json  files[] for seed_<seed>/
    model version           stage8 run_report.json  version
    policy file             stage10 risk_policy.json; SHA-256 = stage10 risk_policy_sha256
    profile file            configs/preprocessing.yaml; SHA-256 = stage8 profile_config_sha256,
                            fingerprint = stage8 profile_fingerprint
    F2 ledger               stage6 profile_report.json  f2.ledger_sha256
    baseline-v1             stage7 run_report.json  version, artifact_sha256

build() checks provenance (clean tree, HEAD), the evidence, the checkpoint folder against the
export manifest, calls build_bundle, loads the result with load_bundle_model and compares every
manifest value with the evidence. It then writes the F4 record (f4_record.json) and a copy of the
manifest next to the bundle folder, never inside it. verify() is what the API (Iheb) and Stage 13
(Khalil) run on their copy: load_bundle_model plus a comparison with the record.

The build copies and hashes bytes only: no features, no scoring, no dataset access.

Part of the shared core: must not import src.evaluation or src.tuning. torch is imported lazily
(inside build and verify) so expected_from_evidence runs without torch.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from src.bundle.bundle import (
    CHECKPOINT_DIR,
    CHECKPOINT_FILES,
    MANIFEST_NAME,
    POLICY_FILE,
    build_bundle,
    checkpoint_dir_sha256,
    sha256_file,
)
from src.preprocessing.profile import PreprocessingError, parse_profile
from src.risk.policy import RiskPolicy

BUNDLE_VERSION = "bundle-v1"
PROTOCOL_VERSION = "1.0"
RECORD_NAME = "f4_record.json"
RECORD_SCHEMA_VERSION = 1
MANIFEST_COPY_NAME = MANIFEST_NAME

STAGE8_REPORT = "docs/audit_evidence/stage8_cnn/run_report.json"
STAGE8_EXPORT = "docs/audit_evidence/stage8_cnn/stage8_export_manifest.json"
STAGE10_REPORT = "docs/audit_evidence/stage10_risk/run_report.json"
POLICY_SOURCE = "docs/audit_evidence/stage10_risk/risk_policy.json"
PROFILE_SOURCE = "configs/preprocessing.yaml"
STAGE6_REPORT = "docs/audit_evidence/stage6_preprocessing/profile_report.json"
STAGE7_REPORT = "docs/audit_evidence/stage7_baseline/run_report.json"

DEFAULT_ROOT = Path(__file__).resolve().parents[2]

_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEX40 = re.compile(r"[0-9a-f]{40}")
_CREATED_AT = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")

GitState = Callable[[Path], tuple[str, str]]


def git_state(root: Path) -> tuple[str, str]:
    """(HEAD, `git status --porcelain` output) of the repository at root."""
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=root, check=True, capture_output=True, text=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as e:
            raise ValueError(f"provenance: cannot run git {' '.join(args)} in {root}: {e}") from None
    return run("rev-parse", "HEAD"), run("status", "--porcelain")


def _provenance(root: Path, git: GitState) -> str:
    """HEAD of a clean tree: it becomes the manifest's code_commit."""
    head, status = git(root)
    if not isinstance(head, str) or not _HEX40.fullmatch(head):
        raise ValueError(f"provenance: HEAD must be 40 lowercase hex characters, got {head!r}")
    if status:
        raise ValueError(f"provenance: the working tree is dirty; F4 is built from a clean commit:\n{status}")
    return head


class _Checks:
    """Ordered list of {name, expected, actual, source, ok}; a failed check raises at once."""

    def __init__(self) -> None:
        self.items: list[dict] = []

    def equal(self, name: str, expected, actual, source: str) -> None:
        ok = expected == actual and type(expected) is type(actual)
        self.items.append({"name": name, "expected": expected, "actual": actual, "source": source, "ok": ok})
        if not ok:
            raise ValueError(f"{name}: mismatch ({source}): expected {expected!r}, actual {actual!r}")


class _Evidence:
    """Reads each evidence file once (relative to root) and keeps its SHA-256."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.sha256: dict[str, str] = {}

    def read(self, rel: str) -> bytes:
        if rel in self.sha256:
            raise AssertionError(f"evidence file {rel} read twice")
        path = self.root / rel
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"evidence: {rel} is missing or not a regular file (root {self.root})")
        data = path.read_bytes()
        self.sha256[rel] = hashlib.sha256(data).hexdigest()
        return data

    def json(self, rel: str) -> dict:
        try:
            data = json.loads(self.read(rel).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"evidence: {rel} is not valid JSON: {e}") from None
        if not isinstance(data, dict):
            raise ValueError(f"evidence: {rel} must be a JSON object")
        return data


def _field(data: dict, rel: str, *keys: str):
    value = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"evidence: {rel} has no field {'.'.join(keys)}")
        value = value[key]
    return value


def _hex64(value, rel: str, field: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise ValueError(f"evidence: {rel} {field} must be 64 lowercase hex characters, got {value!r}")
    return value


def _nonempty(value, rel: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"evidence: {rel} {field} must be a non-empty string, got {value!r}")
    return value


def expected_from_evidence(root=DEFAULT_ROOT) -> dict:
    """Every F4 expected value from the committed evidence, after the checks between sources.

    ValueError naming both files and fields on any disagreement.
    """
    root = Path(root)
    ev = _Evidence(root)
    checks = _Checks()
    s8 = ev.json(STAGE8_REPORT)
    s8_export = ev.json(STAGE8_EXPORT)
    s10 = ev.json(STAGE10_REPORT)
    policy_bytes = ev.read(POLICY_SOURCE)
    profile_bytes = ev.read(PROFILE_SOURCE)
    s6 = ev.json(STAGE6_REPORT)
    s7 = ev.json(STAGE7_REPORT)

    # Bundle seed: one value in both run reports (an int, not a bool).
    seed = _field(s8, STAGE8_REPORT, "bundle_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"evidence: {STAGE8_REPORT} bundle_seed must be a non-negative int, got {seed!r}")
    checks.equal("bundle_seed", seed, _field(s10, STAGE10_REPORT, "bundle_seed"),
                  f"{STAGE8_REPORT} bundle_seed vs {STAGE10_REPORT} bundle_seed")

    # Combined checkpoint hash of that seed: the one Stage 10 fitted the policy on.
    s8_hashes = _field(s8, STAGE8_REPORT, "checkpoint_sha256")
    if not isinstance(s8_hashes, dict) or str(seed) not in s8_hashes:
        raise ValueError(f"evidence: {STAGE8_REPORT} checkpoint_sha256 has no entry for seed {seed}")
    combined = _hex64(s8_hashes[str(seed)], STAGE8_REPORT, f"checkpoint_sha256[{str(seed)!r}]")
    checks.equal("checkpoint_sha256", combined,
                 _hex64(_field(s10, STAGE10_REPORT, "checkpoint_sha256"), STAGE10_REPORT, "checkpoint_sha256"),
                 f"{STAGE8_REPORT} checkpoint_sha256[{str(seed)!r}] vs {STAGE10_REPORT} checkpoint_sha256")

    model_version = _nonempty(_field(s8, STAGE8_REPORT, "version"), STAGE8_REPORT, "version")

    # Per-file hashes and sizes from the Stage 8 export manifest; they must give the combined hash.
    prefix = f"checkpoints/cnn/{model_version}/seed_{seed}"
    entries = _field(s8_export, STAGE8_EXPORT, "files")
    if not isinstance(entries, list):
        raise ValueError(f"evidence: {STAGE8_EXPORT} files must be a list")
    files: dict[str, dict] = {}
    for name in CHECKPOINT_FILES:
        path = f"{prefix}/{name}"
        found = [e for e in entries if isinstance(e, dict) and e.get("path") == path]
        if len(found) != 1:
            raise ValueError(f"evidence: {STAGE8_EXPORT} must have exactly one files[] entry {path}, found {len(found)}")
        sha = _hex64(found[0].get("sha256"), STAGE8_EXPORT, f"files[{path}].sha256")
        size = found[0].get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"evidence: {STAGE8_EXPORT} files[{path}].size must be a positive int, got {size!r}")
        files[name] = {"sha256": sha, "size": size}
    checks.equal("checkpoint_sha256 from per-file hashes", combined,
                 checkpoint_dir_sha256({n: files[n]["sha256"] for n in CHECKPOINT_FILES}),
                 f"{STAGE8_REPORT} checkpoint_sha256[{str(seed)!r}] vs {STAGE8_EXPORT} files[{prefix}/*]")

    # Policy: the committed file is the one Stage 10 wrote.
    policy_sha = hashlib.sha256(policy_bytes).hexdigest()
    checks.equal("policy sha256",
                 _hex64(_field(s10, STAGE10_REPORT, "risk_policy_sha256"), STAGE10_REPORT, "risk_policy_sha256"),
                 policy_sha, f"{STAGE10_REPORT} risk_policy_sha256 vs SHA-256 of {POLICY_SOURCE}")
    try:
        policy = RiskPolicy.from_json(policy_bytes.decode("utf-8"))
    except (ValueError, TypeError) as e:
        raise ValueError(f"evidence: {POLICY_SOURCE} is not a valid risk policy: {e}") from None
    checks.equal("policy version", _field(s10, STAGE10_REPORT, "version"), policy.version,
                 f"{STAGE10_REPORT} version vs {POLICY_SOURCE} version")

    # Profile: the file Stage 8 trained with (config hash and fingerprint).
    profile_sha = hashlib.sha256(profile_bytes).hexdigest()
    checks.equal("profile sha256", _field(s8, STAGE8_REPORT, "profile_config_sha256"), profile_sha,
                 f"{STAGE8_REPORT} profile_config_sha256 vs SHA-256 of {PROFILE_SOURCE}")
    try:
        profile = parse_profile(profile_bytes.decode("utf-8"), source=PROFILE_SOURCE)
    except (PreprocessingError, UnicodeDecodeError) as e:
        raise ValueError(f"evidence: {PROFILE_SOURCE} is not a valid profile: {e}") from None
    fingerprint = profile.fingerprint()
    checks.equal("profile fingerprint", _field(s8, STAGE8_REPORT, "profile_fingerprint"), fingerprint,
                 f"{STAGE8_REPORT} profile_fingerprint vs fingerprint of {PROFILE_SOURCE}")
    checks.equal("profile protocol_version", PROTOCOL_VERSION, profile.protocol_version,
                 f"PROTOCOL_VERSION vs {PROFILE_SOURCE} protocol_version")

    # F2 ledger (D28): Stage 6 recorded it; the profile carries the same value.
    ledger = _hex64(_field(s6, STAGE6_REPORT, "f2", "ledger_sha256"), STAGE6_REPORT, "f2.ledger_sha256")
    checks.equal("f2_ledger_sha256", ledger, profile.f2_ledger_sha256,
                 f"{STAGE6_REPORT} f2.ledger_sha256 vs {PROFILE_SOURCE} f2_ledger_sha256")

    baseline = {
        "version": _nonempty(_field(s7, STAGE7_REPORT, "version"), STAGE7_REPORT, "version"),
        "artifact_sha256": _hex64(_field(s7, STAGE7_REPORT, "artifact_sha256"), STAGE7_REPORT, "artifact_sha256"),
        "source": STAGE7_REPORT,
    }

    return {
        "bundle_seed": seed,
        "checkpoint": {"source": prefix, "files": files, "sha256": combined, "version": model_version},
        "profile": {"source": PROFILE_SOURCE, "sha256": profile_sha, "version": profile.profile_version,
                    "fingerprint": fingerprint},
        "policy": {"source": POLICY_SOURCE, "sha256": policy_sha, "version": policy.version},
        "f2_ledger_sha256": ledger,
        "protocol_version": PROTOCOL_VERSION,
        "baseline": baseline,
        "evidence_sha256": dict(sorted(ev.sha256.items())),
        "checks": checks.items,
    }


def _check_checkpoint_dir(checkpoint_dir: Path, expected: dict, checks: _Checks) -> None:
    """Exactly the two D42 files, each with the size and SHA-256 of the Stage 8 export manifest."""
    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise ValueError(f"checkpoint: {checkpoint_dir} is not a folder")
    names = sorted(p.name for p in checkpoint_dir.iterdir())
    if names != sorted(CHECKPOINT_FILES):
        raise ValueError(f"checkpoint: {checkpoint_dir} must hold exactly {sorted(CHECKPOINT_FILES)}, found {names}")
    # Drive-relative only: this text is written to the committed record, so no local path.
    source = f"{STAGE8_EXPORT} files[{expected['source']}/<name>] vs downloaded copy of {expected['source']}"
    for name in CHECKPOINT_FILES:
        path = checkpoint_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"checkpoint: {path} must be a regular file")
        want = expected["files"][name]
        checks.equal(f"checkpoint file {name} size", want["size"], path.stat().st_size, source)
        checks.equal(f"checkpoint file {name} sha256", want["sha256"], sha256_file(path), source)


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _environment() -> dict:
    import numpy
    import torch
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "safetensors": importlib.metadata.version("safetensors"),
    }


def _json_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _manifest_checks(bundle_model, expected: dict, checks: _Checks, *, code_commit: str, created_at: str) -> None:
    """Every manifest value of the loaded bundle against the evidence (and the build arguments)."""
    manifest = bundle_model.bundle.manifest
    ck, prof, pol = (manifest.components[n] for n in ("checkpoint", "profile", "policy"))
    src = MANIFEST_NAME
    checks.equal("manifest bundle_version", BUNDLE_VERSION, manifest.bundle_version, f"BUNDLE_VERSION vs {src}")
    checks.equal("manifest protocol_version", expected["protocol_version"], manifest.protocol_version,
                 f"evidence vs {src}")
    checks.equal("manifest f2_ledger_sha256", expected["f2_ledger_sha256"], manifest.f2_ledger_sha256,
                 f"{STAGE6_REPORT} vs {src}")
    checks.equal("manifest code_commit", code_commit, manifest.code_commit, f"git HEAD vs {src}")
    checks.equal("manifest created_at", created_at, manifest.created_at, f"build argument vs {src}")
    checks.equal("manifest checkpoint sha256", expected["checkpoint"]["sha256"], ck.sha256, f"{STAGE8_REPORT} vs {src}")
    for name in CHECKPOINT_FILES:
        checks.equal(f"manifest checkpoint file {name} sha256", expected["checkpoint"]["files"][name]["sha256"],
                     ck.files[name], f"{STAGE8_EXPORT} vs {src}")
    checks.equal("manifest checkpoint version", expected["checkpoint"]["version"], ck.version,
                 f"{STAGE8_REPORT} vs {src}")
    checks.equal("loaded checkpoint version", expected["checkpoint"]["version"], bundle_model.metadata["version"],
                 f"{STAGE8_REPORT} vs loaded checkpoint.json")
    checks.equal("manifest profile sha256", expected["profile"]["sha256"], prof.sha256, f"{STAGE8_REPORT} vs {src}")
    checks.equal("manifest profile version", expected["profile"]["version"], prof.version, f"{PROFILE_SOURCE} vs {src}")
    checks.equal("bundled profile fingerprint", expected["profile"]["fingerprint"],
                 bundle_model.bundle.profile.fingerprint(), f"{STAGE8_REPORT} vs bundled profile")
    checks.equal("manifest policy sha256", expected["policy"]["sha256"], pol.sha256, f"{STAGE10_REPORT} vs {src}")
    checks.equal("manifest policy version", expected["policy"]["version"], pol.version, f"{POLICY_SOURCE} vs {src}")


def _clean(out_dir: Path, created: bool) -> None:
    """Remove what a failed build left in out_dir (the folder too if this call created it)."""
    if not out_dir.is_dir():
        return
    if created:
        shutil.rmtree(out_dir)
        return
    for child in out_dir.iterdir():
        shutil.rmtree(child) if child.is_dir() and not child.is_symlink() else child.unlink()


def build(checkpoint_dir, out_dir, *, root=DEFAULT_ROOT, created_at: str | None = None,
          git: GitState = git_state) -> dict:
    """Build F4 into out_dir, verify it, write the F4 record and a manifest copy beside it.

    Returns the record. out_dir, its parent and checkpoint_dir must be outside root. Any failed
    check raises ValueError; then nothing is left in out_dir and no record or manifest copy is
    written.
    """
    from src.bundle.model_loader import load_bundle_model  # torch: imported only when building

    root, checkpoint_dir, out_dir = Path(root), Path(checkpoint_dir), Path(out_dir)
    # Outside the repository only (AGENTS.md rule 7): neither the bundle, its record and manifest
    # copy, nor the downloaded Stage 8 files may sit in the working tree, where they could be committed.
    repo = root.resolve()
    for what, path in (("output", out_dir), ("output", out_dir.parent), ("checkpoint", checkpoint_dir)):
        if path.resolve().is_relative_to(repo):
            raise ValueError(f"{what}: {path} is inside the repository; F4 is built outside it "
                             f"(AGENTS.md rule 7: no checkpoints in Git)")
    record_path = out_dir.parent / RECORD_NAME
    manifest_copy = out_dir.parent / MANIFEST_COPY_NAME
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise ValueError(f"output: {out_dir} exists and is not empty; bundles are never overwritten")
    for path in (record_path, manifest_copy):
        if path.exists():
            raise ValueError(f"output: {path} already exists; the F4 record is never overwritten")
    if created_at is None:
        created_at = _now_utc()
    if not isinstance(created_at, str) or not _CREATED_AT.fullmatch(created_at):
        raise ValueError(f"created_at must be YYYY-MM-DDTHH:MM:SSZ (UTC, seconds), got {created_at!r}")

    # 1-2. Provenance, evidence and the checkpoint folder; nothing is written before they pass.
    code_commit = _provenance(root, git)
    expected = expected_from_evidence(root)
    checks = _Checks()
    checks.items.extend(expected["checks"])
    _check_checkpoint_dir(checkpoint_dir, expected["checkpoint"], checks)

    # 3. Build (build_bundle cleans up after itself on failure).
    created = not out_dir.exists()
    manifest_path = build_bundle(
        out_dir,
        checkpoint_dir=checkpoint_dir,
        expected_checkpoint_sha256=expected["checkpoint"]["sha256"],
        profile=root / PROFILE_SOURCE,
        policy=root / POLICY_SOURCE,
        bundle_version=BUNDLE_VERSION,
        model_version=expected["checkpoint"]["version"],
        profile_version=expected["profile"]["version"],
        code_commit=code_commit,
        protocol_version=expected["protocol_version"],
        f2_ledger_sha256=expected["f2_ledger_sha256"],
        created_at=created_at,
    )
    written: list[Path] = []
    try:
        # 4. Verify through the API's loader; every manifest value equals the evidence.
        bundle_model = load_bundle_model(out_dir)
        _manifest_checks(bundle_model, expected, checks, code_commit=code_commit, created_at=created_at)
        manifest_bytes = manifest_path.read_bytes()
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        checks.equal("manifest_sha256", manifest_sha256, bundle_model.bundle.manifest_sha256,
                     f"SHA-256 of {MANIFEST_NAME} vs load_bundle")

        # 5. The F4 record and the manifest copy, outside the bundle, never overwritten ("x").
        record = {
            "schema_version": RECORD_SCHEMA_VERSION,
            "bundle_version": BUNDLE_VERSION,
            "manifest_sha256": manifest_sha256,
            "created_at": created_at,
            "code_commit": code_commit,
            "components": {
                "checkpoint": {"dir": CHECKPOINT_DIR, **expected["checkpoint"]},
                "profile": {"file": (root / PROFILE_SOURCE).name, **expected["profile"]},
                "policy": {"file": POLICY_FILE, **expected["policy"]},
            },
            "baseline": expected["baseline"],
            "bundle_seed": expected["bundle_seed"],
            "f2_ledger_sha256": expected["f2_ledger_sha256"],
            "protocol_version": expected["protocol_version"],
            "environment": _environment(),
            "checks": checks.items,
            "evidence_sha256": expected["evidence_sha256"],
        }
        for path, data in ((manifest_copy, manifest_bytes), (record_path, _json_bytes(record))):
            with open(path, "xb") as f:
                written.append(path)
                f.write(data)
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        _clean(out_dir, created)
        raise
    return record


def _load_record(record) -> dict:
    try:
        data = json.loads(Path(record).read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"record: cannot read {record}: {e}") from None
    if not isinstance(data, dict) or data.get("schema_version") != RECORD_SCHEMA_VERSION:
        raise ValueError(f"record: {record} is not an F4 record of schema {RECORD_SCHEMA_VERSION}")
    return data


def verify(bundle_dir, record) -> dict:
    """Verify a copy of F4 against its record: load_bundle_model, manifest_sha256, every component.

    ValueError on any difference.
    """
    from src.bundle.model_loader import load_bundle_model  # torch: imported only when verifying

    rec = _load_record(record)
    try:
        comps = rec["components"]
        ck, prof, pol = comps["checkpoint"], comps["profile"], comps["policy"]
        bm = load_bundle_model(Path(bundle_dir))
        manifest = bm.bundle.manifest
        mck, mprof, mpol = (manifest.components[n] for n in ("checkpoint", "profile", "policy"))
        checks = _Checks()
        src = f"record {Path(record).name} vs bundle {Path(bundle_dir)}"
        checks.equal("manifest_sha256", rec["manifest_sha256"], bm.bundle.manifest_sha256, src)
        checks.equal("bundle_version", rec["bundle_version"], manifest.bundle_version, src)
        checks.equal("checkpoint sha256", ck["sha256"], mck.sha256, src)
        for name in CHECKPOINT_FILES:
            checks.equal(f"checkpoint file {name} sha256", ck["files"][name]["sha256"], mck.files[name], src)
        checks.equal("checkpoint version", ck["version"], mck.version, src)
        checks.equal("profile sha256", prof["sha256"], mprof.sha256, src)
        checks.equal("profile version", prof["version"], mprof.version, src)
        checks.equal("profile fingerprint", prof["fingerprint"], bm.bundle.profile.fingerprint(), src)
        checks.equal("policy sha256", pol["sha256"], mpol.sha256, src)
        checks.equal("policy version", pol["version"], mpol.version, src)
    except (KeyError, TypeError) as e:
        raise ValueError(f"record: {record} is missing a field: {e}") from None
    return {"bundle_version": manifest.bundle_version, "manifest_sha256": bm.bundle.manifest_sha256,
            "checks": checks.items}


def main(argv=None, *, root=DEFAULT_ROOT, git: GitState = git_state) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.bundle.f4", description="F4 ModelBundle build and verify")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="build F4 from the Stage 8 checkpoint folder and committed evidence")
    b.add_argument("--checkpoint-dir", required=True, type=Path)
    b.add_argument("--out", required=True, type=Path)
    b.add_argument("--created-at", default=None)
    v = sub.add_parser("verify", help="verify a copy of F4 against its record")
    v.add_argument("--bundle", required=True, type=Path)
    v.add_argument("--record", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            record = build(args.checkpoint_dir, args.out, root=root, created_at=args.created_at, git=git)
            print(f"record: {args.out.parent / RECORD_NAME}")
            print(f"manifest_sha256: {record['manifest_sha256']}")
            print("F4 build OK")
        else:
            report = verify(args.bundle, args.record)
            print(f"manifest_sha256: {report['manifest_sha256']}")
            print("F4 verify OK")
    except (ValueError, OSError) as e:
        print(f"F4 {args.command} FAILED: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
