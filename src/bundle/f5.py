"""F5 build and verify: ModelBundle v2 from committed evidence (Track V, V5; D48 (4), (7); D59 (7); D61 (6)).

F5 is the load_bundle-verified bundle of the v2 search winner's bundle-seed checkpoint, the F3
preprocessing profile and risk-v2, in the D59 layout (the v2 counterpart of src/bundle/f4.py).
Every expected value is read in code from committed evidence, never typed:

    winner, bundle seed     v2_search run_report.json and selection_report.json  selected,
                            bundle_seed (must agree); v2_risk run_report.json architecture,
                            bundle_seed (must equal them)
    checkpoint_sha256       v2_search run_report.json  checkpoint_sha256[<winner>_seed<seed>]
                            (= v2_risk checkpoint_sha256)
    model version           v2_search run_report.json  version
    selection               v2_search selection_report.json  seeds.<k>.best_selection_m, k = 0, 1, 2
    policy file             v2_risk risk_policy.json; SHA-256 = v2_risk risk_policy_sha256,
                            its version = v2_risk version; v2_risk v2_search_evidence = SHA-256
                            of the two search reports
    D48 (7) order           stage11_v2_run run_report.json and run_registry.jsonl: a completed
                            ("for", "test", "stage11_test") record of the v2 model set
                            (<winner>-v2-seed<k>) with the report's run_id; the report's
                            architecture, bundle_seed and model_set artifact hashes equal the
                            search's. F5 is refused without it.
    profile file            configs/preprocessing.yaml; SHA-256 = v2_search profile_config_sha256,
                            fingerprint = v2_search profile_fingerprint (= v2_risk's)
    F2                      stage6 profile_report.json  f2.ledger_sha256 (= the profile's) and
                            f2.manifest_sha256 (= v2_search f2_manifest_sha256 = v2_risk's)

There is no per-file export manifest for v2: the checkpoint folder's two per-file hashes and sizes
are recorded as observed, and their combined hash must equal the search's checkpoint_sha256.

build() checks provenance (clean tree, HEAD), the evidence, the checkpoint folder (exactly the two
D59 files, combined hash, checkpoint.json architecture / seed / version / provenance), calls
build_bundle, loads the result with load_bundle_model (v2 dispatch: the winner's architecture, CPU,
eval) and compares every manifest value with the evidence. It then writes the F5 record
(f5_record.json) and a copy of the manifest next to the bundle folder, never inside it. verify() is
what the API (Iheb) and Stage 13 (Khalil) run on their copy; Stage 13 then passes the bundle and the
record to external_run.verify_f5_artifacts.

The build copies and hashes bytes only: no features, no scoring, no dataset access.

Part of the shared core: must not import src.evaluation or src.tuning. The names those packages
define for the same files (run folder, model IDs, report names, seeds) are restated here and
tests/bundle/test_f5.py asserts they are equal. torch is imported lazily (inside build and verify),
so expected_from_evidence runs without torch. The F4 helpers are imported from f4, not copied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from src.bundle.bundle import (
    CHECKPOINT_DIR,
    CHECKPOINT_FILES,
    CHECKPOINT_JSON,
    MANIFEST_NAME,
    POLICY_FILE,
    build_bundle,
    checkpoint_dir_sha256,
    sha256_file,
)
from src.bundle.f4 import (
    _CREATED_AT,
    DEFAULT_ROOT,
    PROFILE_SOURCE,
    PROTOCOL_VERSION,
    STAGE6_REPORT,
    GitState,
    _Checks,
    _clean,
    _environment,
    _Evidence,
    _field,
    _hex64,
    _json_bytes,
    _nonempty,
    _now_utc,
    _provenance,
    git_state,
)
from src.preprocessing.profile import PreprocessingError, parse_profile
from src.risk.policy import RiskPolicy

BUNDLE_VERSION = "bundle-v2"
RECORD_NAME = "f5_record.json"
RECORD_SCHEMA_VERSION = 1
MANIFEST_COPY_NAME = MANIFEST_NAME

# D49 candidates and final seeds; equal to src.models.v2_checkpoint.ARCHITECTURES and
# src.evaluation.stage11_v2_run.SEEDS (asserted in the tests; not imported: torch / research code).
ARCHITECTURES = ("cnn-aug", "crnn", "aasist")
SEEDS = (0, 1, 2)

# File names written by src.tuning.v2_search, src.tuning.v2_risk_run (src.tuning.risk_run names) and
# src.evaluation.stage11_v2_run; the evidence folders are where their outputs are committed.
SEARCH_RUN_REPORT_NAME = "run_report.json"
SEARCH_SELECTION_REPORT_NAME = "selection_report.json"
RISK_RUN_REPORT_NAME = "run_report.json"
RISK_POLICY_NAME = "risk_policy.json"
STAGE11_RUN_REPORT_NAME = "run_report.json"

V2_SEARCH_DIR = "docs/audit_evidence/v2_search"
V2_RISK_DIR = "docs/audit_evidence/v2_risk"
STAGE11_V2_DIR = "docs/audit_evidence/stage11_v2_run"
SEARCH_REPORT = f"{V2_SEARCH_DIR}/{SEARCH_RUN_REPORT_NAME}"
SELECTION_REPORT = f"{V2_SEARCH_DIR}/{SEARCH_SELECTION_REPORT_NAME}"
RISK_REPORT = f"{V2_RISK_DIR}/{RISK_RUN_REPORT_NAME}"
POLICY_SOURCE = f"{V2_RISK_DIR}/{RISK_POLICY_NAME}"
STAGE11_V2_REPORT = f"{STAGE11_V2_DIR}/{STAGE11_RUN_REPORT_NAME}"
RUN_REGISTRY = "docs/audit_evidence/run_registry.jsonl"

# The registry key of the v2 Stage 11 claim (src.evaluation.run_registry, stage11_run).
STAGE11_DATASET, STAGE11_SPLIT, STAGE11_PURPOSE = "for", "test", "stage11_test"
REGISTRY_STARTED, REGISTRY_COMPLETED = "started", "completed"
REGISTRY_GENESIS = "0" * 64

# Where the checkpoint folder sits in the v2 search output (v2_search RUNS_DIR / run / CHECKPOINT_DIR).
SEARCH_RUNS_DIR = "runs"

_HEX32 = re.compile(r"[0-9a-f]{32}")


def run_dir_name(candidate: str, seed: int) -> str:
    """The v2 search run folder and checkpoint_sha256 key, e.g. crnn_seed1 (= v2_search.run_dir_name)."""
    return f"{candidate}_seed{seed}"


def model_ids_v2(winner: str) -> tuple[str, ...]:
    """The v2 Stage 11 model set (D48 (7)), `<winner>-v2-seed<k>` (= stage11_v2_run.model_ids_v2)."""
    if winner not in ARCHITECTURES:
        raise ValueError(f"winner must be one of {ARCHITECTURES}, got {winner!r}")
    return tuple(f"{winner}-v2-seed{k}" for k in SEEDS)


def _seed(value, rel: str, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in SEEDS:
        raise ValueError(f"evidence: {rel} {field} must be one of {SEEDS}, got {value!r}")
    return value


def _unit(value, rel: str, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError(f"evidence: {rel} {field} must be a number in [0, 1], got {value!r}")
    return value


def _registry_records(data: bytes) -> list[dict]:
    """The registry's JSON Lines, with the hash chain checked (src.evaluation.run_registry rule).

    Parsed with json only (shared core): each line is an object whose prev_sha256 is the SHA-256 of
    the previous line's bytes (with its "\\n"); the first has "0" * 64.
    """
    if data and not data.endswith(b"\n"):
        raise ValueError(f"evidence: {RUN_REGISTRY} must end with a newline")
    records, prev = [], REGISTRY_GENESIS
    for number, line in enumerate(data.splitlines(keepends=True), start=1):
        try:
            record = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"evidence: {RUN_REGISTRY} line {number} is not valid JSON: {e}") from None
        if not isinstance(record, dict):
            raise ValueError(f"evidence: {RUN_REGISTRY} line {number} must be a JSON object")
        if record.get("prev_sha256") != prev:
            raise ValueError(f"evidence: {RUN_REGISTRY} line {number} breaks the hash chain "
                             f"(prev_sha256 {record.get('prev_sha256')!r}, expected {prev})")
        prev = hashlib.sha256(line).hexdigest()
        records.append(record)
    return records


def expected_from_evidence(root=DEFAULT_ROOT) -> dict:
    """Every F5 expected value from the committed evidence, after the checks between sources.

    ValueError naming both files and fields on any disagreement.
    """
    root = Path(root)
    ev = _Evidence(root)
    checks = _Checks()
    search = ev.json(SEARCH_REPORT)
    selection = ev.json(SELECTION_REPORT)
    risk = ev.json(RISK_REPORT)
    policy_bytes = ev.read(POLICY_SOURCE)
    stage11 = ev.json(STAGE11_V2_REPORT)
    registry_bytes = ev.read(RUN_REGISTRY)
    profile_bytes = ev.read(PROFILE_SOURCE)
    s6 = ev.json(STAGE6_REPORT)

    # Winner and bundle seed: one value in both search reports (v2_risk_run.bundle_checkpoint's rule).
    winner = _field(search, SEARCH_REPORT, "selected")
    if winner not in ARCHITECTURES:
        raise ValueError(f"evidence: {SEARCH_REPORT} selected must be one of {ARCHITECTURES}, got {winner!r}")
    seed = _seed(_field(search, SEARCH_REPORT, "bundle_seed"), SEARCH_REPORT, "bundle_seed")
    checks.equal("selected", winner, _field(selection, SELECTION_REPORT, "selected"),
                 f"{SEARCH_REPORT} selected vs {SELECTION_REPORT} selected")
    checks.equal("bundle_seed", seed, _field(selection, SELECTION_REPORT, "bundle_seed"),
                 f"{SEARCH_REPORT} bundle_seed vs {SELECTION_REPORT} bundle_seed")

    # The combined hash of every winner seed (the three v2 Stage 11 models); the bundle seed's is F5's.
    hashes = _field(search, SEARCH_REPORT, "checkpoint_sha256")
    if not isinstance(hashes, dict):
        raise ValueError(f"evidence: {SEARCH_REPORT} checkpoint_sha256 must be an object")
    seed_hashes = {}
    for k in SEEDS:
        name = run_dir_name(winner, k)
        if name not in hashes:
            raise ValueError(f"evidence: {SEARCH_REPORT} checkpoint_sha256 has no entry {name}")
        seed_hashes[k] = _hex64(hashes[name], SEARCH_REPORT, f"checkpoint_sha256[{name!r}]")
    combined = seed_hashes[seed]
    run_name = run_dir_name(winner, seed)
    model_version = _nonempty(_field(search, SEARCH_REPORT, "version"), SEARCH_REPORT, "version")

    seeds = _field(selection, SELECTION_REPORT, "seeds")
    if not isinstance(seeds, dict) or sorted(seeds) != sorted(str(k) for k in SEEDS):
        raise ValueError(f"evidence: {SELECTION_REPORT} seeds must have exactly the keys "
                         f"{[str(k) for k in SEEDS]}, got {sorted(seeds) if isinstance(seeds, dict) else seeds!r}")
    best_m = {str(k): _unit(_field(selection, SELECTION_REPORT, "seeds", str(k), "best_selection_m"),
                            SELECTION_REPORT, f"seeds.{k}.best_selection_m") for k in SEEDS}

    search_values = {key: _hex64(_field(search, SEARCH_REPORT, key), SEARCH_REPORT, key)
                     for key in ("profile_config_sha256", "profile_fingerprint", "f2_manifest_sha256",
                                 "selection_set_sha256")}

    # risk-v2: fitted on that checkpoint, bound to these search reports.
    checks.equal("risk architecture", winner, _field(risk, RISK_REPORT, "architecture"),
                 f"{SEARCH_REPORT} selected vs {RISK_REPORT} architecture")
    checks.equal("risk bundle_seed", seed, _field(risk, RISK_REPORT, "bundle_seed"),
                 f"{SEARCH_REPORT} bundle_seed vs {RISK_REPORT} bundle_seed")
    checks.equal("risk checkpoint_sha256", combined,
                 _hex64(_field(risk, RISK_REPORT, "checkpoint_sha256"), RISK_REPORT, "checkpoint_sha256"),
                 f"{SEARCH_REPORT} checkpoint_sha256[{run_name!r}] vs {RISK_REPORT} checkpoint_sha256")
    for key in ("profile_fingerprint", "profile_config_sha256", "f2_manifest_sha256"):
        checks.equal(f"risk {key}", search_values[key], _field(risk, RISK_REPORT, key),
                     f"{SEARCH_REPORT} {key} vs {RISK_REPORT} {key}")
    checks.equal("risk v2_search_evidence",
                 {SEARCH_RUN_REPORT_NAME: ev.sha256[SEARCH_REPORT],
                  SEARCH_SELECTION_REPORT_NAME: ev.sha256[SELECTION_REPORT]},
                 _field(risk, RISK_REPORT, "v2_search_evidence"),
                 f"SHA-256 of {SEARCH_REPORT} and {SELECTION_REPORT} vs {RISK_REPORT} v2_search_evidence")

    # Policy: the committed file is the one the risk-v2 run wrote.
    policy_sha = hashlib.sha256(policy_bytes).hexdigest()
    checks.equal("policy sha256",
                 _hex64(_field(risk, RISK_REPORT, "risk_policy_sha256"), RISK_REPORT, "risk_policy_sha256"),
                 policy_sha, f"{RISK_REPORT} risk_policy_sha256 vs SHA-256 of {POLICY_SOURCE}")
    try:
        policy = RiskPolicy.from_json(policy_bytes.decode("utf-8"))
    except (ValueError, TypeError) as e:
        raise ValueError(f"evidence: {POLICY_SOURCE} is not a valid risk policy: {e}") from None
    checks.equal("policy version", _field(risk, RISK_REPORT, "version"), policy.version,
                 f"{RISK_REPORT} version vs {POLICY_SOURCE} version")

    # D48 (7): the v2 Stage 11 run is completed in the registry before F5.
    model_ids = model_ids_v2(winner)
    bundle_id = model_ids[SEEDS.index(seed)]
    run_id = _field(stage11, STAGE11_V2_REPORT, "run_id")
    if not isinstance(run_id, str) or not _HEX32.fullmatch(run_id):
        raise ValueError(f"evidence: {STAGE11_V2_REPORT} run_id must be 32 lowercase hex, got {run_id!r}")
    for key, want in (("dataset", STAGE11_DATASET), ("split", STAGE11_SPLIT),
                      ("evaluation_purpose", STAGE11_PURPOSE)):
        checks.equal(f"stage11_v2 {key}", want, _field(stage11, STAGE11_V2_REPORT, key),
                     f"the v2 Stage 11 claim vs {STAGE11_V2_REPORT} {key}")
    checks.equal("stage11_v2 architecture", winner, _field(stage11, STAGE11_V2_REPORT, "architecture"),
                 f"{SEARCH_REPORT} selected vs {STAGE11_V2_REPORT} architecture")
    checks.equal("stage11_v2 bundle_seed", seed, _field(stage11, STAGE11_V2_REPORT, "bundle_seed"),
                 f"{SEARCH_REPORT} bundle_seed vs {STAGE11_V2_REPORT} bundle_seed")
    model_set = _field(stage11, STAGE11_V2_REPORT, "model_set")
    if not isinstance(model_set, dict):
        raise ValueError(f"evidence: {STAGE11_V2_REPORT} model_set must be an object")
    checks.equal("stage11_v2 model_set", sorted(model_ids), sorted(model_set),
                 f"model IDs of {SEARCH_REPORT} selected vs {STAGE11_V2_REPORT} model_set")
    checks.equal("stage11_v2 bundle artifact_sha256", combined,
                 _field(stage11, STAGE11_V2_REPORT, "model_set", bundle_id, "artifact_sha256"),
                 f"{SEARCH_REPORT} checkpoint_sha256[{run_name!r}] vs "
                 f"{STAGE11_V2_REPORT} model_set[{bundle_id!r}].artifact_sha256")
    for k, model_id in zip(SEEDS, model_ids):
        if k != seed:
            checks.equal(f"stage11_v2 {model_id} artifact_sha256", seed_hashes[k],
                         _field(stage11, STAGE11_V2_REPORT, "model_set", model_id, "artifact_sha256"),
                         f"{SEARCH_REPORT} checkpoint_sha256[{run_dir_name(winner, k)!r}] vs "
                         f"{STAGE11_V2_REPORT} model_set[{model_id!r}].artifact_sha256")
    key = [STAGE11_DATASET, STAGE11_SPLIT, STAGE11_PURPOSE, sorted(model_ids)]
    found = [[r.get("status"), r.get("run_id")] for r in _registry_records(registry_bytes)
             if [r.get("dataset"), r.get("split"), r.get("evaluation_purpose"),
                 sorted(r.get("model_set") or [])] == key]
    checks.equal("registry v2 stage11_test records", [[REGISTRY_STARTED, run_id], [REGISTRY_COMPLETED, run_id]],
                 found, f"{STAGE11_V2_REPORT} run_id vs {RUN_REGISTRY} records {key} "
                        f"(D48 (7): a completed v2 Stage 11 run before F5)")

    # Profile: the file the v2 checkpoints were trained with (config hash and fingerprint).
    profile_sha = hashlib.sha256(profile_bytes).hexdigest()
    checks.equal("profile sha256", search_values["profile_config_sha256"], profile_sha,
                 f"{SEARCH_REPORT} profile_config_sha256 vs SHA-256 of {PROFILE_SOURCE}")
    try:
        profile = parse_profile(profile_bytes.decode("utf-8"), source=PROFILE_SOURCE)
    except (PreprocessingError, UnicodeDecodeError) as e:
        raise ValueError(f"evidence: {PROFILE_SOURCE} is not a valid profile: {e}") from None
    fingerprint = profile.fingerprint()
    checks.equal("profile fingerprint", search_values["profile_fingerprint"], fingerprint,
                 f"{SEARCH_REPORT} profile_fingerprint vs fingerprint of {PROFILE_SOURCE}")
    checks.equal("profile protocol_version", PROTOCOL_VERSION, profile.protocol_version,
                 f"PROTOCOL_VERSION vs {PROFILE_SOURCE} protocol_version")

    # F2 (D28): Stage 6 recorded the ledger (the profile carries it) and the manifest hash.
    ledger = _hex64(_field(s6, STAGE6_REPORT, "f2", "ledger_sha256"), STAGE6_REPORT, "f2.ledger_sha256")
    checks.equal("f2_ledger_sha256", ledger, profile.f2_ledger_sha256,
                 f"{STAGE6_REPORT} f2.ledger_sha256 vs {PROFILE_SOURCE} f2_ledger_sha256")
    checks.equal("f2_manifest_sha256", _field(s6, STAGE6_REPORT, "f2", "manifest_sha256"),
                 search_values["f2_manifest_sha256"],
                 f"{STAGE6_REPORT} f2.manifest_sha256 vs {SEARCH_REPORT} f2_manifest_sha256")

    return {
        "architecture": winner,
        "bundle_seed": seed,
        "checkpoint": {"source": f"{SEARCH_RUNS_DIR}/{run_name}/{CHECKPOINT_DIR}", "sha256": combined,
                       "version": model_version},
        "profile": {"source": PROFILE_SOURCE, "sha256": profile_sha, "version": profile.profile_version,
                    "fingerprint": fingerprint},
        "policy": {"source": POLICY_SOURCE, "sha256": policy_sha, "version": policy.version},
        "selection": {"winner": winner, "best_selection_m": best_m},
        "provenance": {key: search_values[key] for key in ("f2_manifest_sha256", "selection_set_sha256")},
        "stage11_v2": {"run_id": run_id, "model_set": sorted(model_ids), "source": STAGE11_V2_REPORT},
        "f2_ledger_sha256": ledger,
        "protocol_version": PROTOCOL_VERSION,
        "evidence_sha256": dict(sorted(ev.sha256.items())),
        "checks": checks.items,
    }


def _check_checkpoint_dir(checkpoint_dir: Path, expected: dict, checks: _Checks) -> dict:
    """Exactly the two D59 files; their combined hash and checkpoint.json against the evidence.

    Returns the observed per-file {sha256, size} (no v2 export manifest exists to compare them with).
    """
    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise ValueError(f"checkpoint: {checkpoint_dir} is not a folder")
    names = sorted(p.name for p in checkpoint_dir.iterdir())
    if names != sorted(CHECKPOINT_FILES):
        raise ValueError(f"checkpoint: {checkpoint_dir} must hold exactly {sorted(CHECKPOINT_FILES)}, found {names}")
    files, meta_bytes = {}, b""
    for name in CHECKPOINT_FILES:
        path = checkpoint_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"checkpoint: {path} must be a regular file")
        if name == CHECKPOINT_JSON:  # one read: the hashed bytes are the parsed bytes
            meta_bytes = path.read_bytes()
            files[name] = {"sha256": hashlib.sha256(meta_bytes).hexdigest(), "size": len(meta_bytes)}
        else:
            files[name] = {"sha256": sha256_file(path), "size": path.stat().st_size}
    # Drive-relative only: this text is written to the committed record, so no local path.
    ck = expected["checkpoint"]
    checks.equal("checkpoint_sha256 of the checkpoint folder", ck["sha256"],
                 checkpoint_dir_sha256({n: files[n]["sha256"] for n in CHECKPOINT_FILES}),
                 f"{SEARCH_REPORT} checkpoint_sha256 vs downloaded copy of {ck['source']}")
    try:
        meta = json.loads(meta_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"checkpoint: {CHECKPOINT_JSON} is not valid JSON: {e}") from None
    if not isinstance(meta, dict):
        raise ValueError(f"checkpoint: {CHECKPOINT_JSON} must be a JSON object")
    src = f"{SEARCH_REPORT} vs {ck['source']}/{CHECKPOINT_JSON}"
    checks.equal(f"{CHECKPOINT_JSON} architecture", expected["architecture"], meta.get("architecture"),
                 f"{SEARCH_REPORT} selected vs {ck['source']}/{CHECKPOINT_JSON} architecture")
    checks.equal(f"{CHECKPOINT_JSON} seed", expected["bundle_seed"], meta.get("seed"),
                 f"{SEARCH_REPORT} bundle_seed vs {ck['source']}/{CHECKPOINT_JSON} seed")
    checks.equal(f"{CHECKPOINT_JSON} version", ck["version"], meta.get("version"), src)
    checks.equal(f"{CHECKPOINT_JSON} profile_fingerprint", expected["profile"]["fingerprint"],
                 meta.get("profile_fingerprint"), src)
    for key, value in expected["provenance"].items():
        checks.equal(f"{CHECKPOINT_JSON} {key}", value, meta.get(key), src)
    return files


def _loaded_model_checks(bundle_model, expected: dict, checks: _Checks) -> None:
    """The loader dispatched to v2: the winner's architecture, the bundle seed, on CPU, in eval mode."""
    src = "evidence vs load_bundle_model"
    checks.equal("loaded architecture", expected["architecture"], bundle_model.architecture, src)
    checks.equal("loaded checkpoint architecture", expected["architecture"],
                 bundle_model.metadata.get("architecture"), src)
    checks.equal("loaded checkpoint seed", expected["bundle_seed"], bundle_model.metadata.get("seed"), src)
    checks.equal("loaded checkpoint version", expected["checkpoint"]["version"],
                 bundle_model.metadata.get("version"), f"{SEARCH_REPORT} vs loaded {CHECKPOINT_JSON}")
    checks.equal("loaded model eval mode", False, getattr(bundle_model.model, "training", None), src)
    devices = sorted({p.device.type for p in bundle_model.model.parameters()})
    checks.equal("loaded model device", ["cpu"], devices, src)


def _manifest_checks(bundle_model, expected: dict, files: dict, checks: _Checks, *,
                     code_commit: str, created_at: str) -> None:
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
    checks.equal("manifest checkpoint sha256", expected["checkpoint"]["sha256"], ck.sha256, f"{SEARCH_REPORT} vs {src}")
    for name in CHECKPOINT_FILES:
        checks.equal(f"manifest checkpoint file {name} sha256", files[name]["sha256"], ck.files[name],
                     f"checkpoint folder vs {src}")
    checks.equal("manifest checkpoint version", expected["checkpoint"]["version"], ck.version,
                 f"{SEARCH_REPORT} vs {src}")
    checks.equal("manifest profile sha256", expected["profile"]["sha256"], prof.sha256, f"{SEARCH_REPORT} vs {src}")
    checks.equal("manifest profile version", expected["profile"]["version"], prof.version, f"{PROFILE_SOURCE} vs {src}")
    checks.equal("bundled profile fingerprint", expected["profile"]["fingerprint"],
                 bundle_model.bundle.profile.fingerprint(), f"{SEARCH_REPORT} vs bundled profile")
    checks.equal("manifest policy sha256", expected["policy"]["sha256"], pol.sha256, f"{RISK_REPORT} vs {src}")
    checks.equal("manifest policy version", expected["policy"]["version"], pol.version, f"{POLICY_SOURCE} vs {src}")


def build(checkpoint_dir, out_dir, *, root=DEFAULT_ROOT, created_at: str | None = None,
          git: GitState = git_state) -> dict:
    """Build F5 into out_dir, verify it, write the F5 record and a manifest copy beside it.

    Returns the record. out_dir, its parent and checkpoint_dir must be outside root. Any failed
    check raises ValueError; then nothing is left in out_dir and no record or manifest copy is
    written.
    """
    from src.bundle.model_loader import load_bundle_model  # torch: imported only when building

    root, checkpoint_dir, out_dir = Path(root), Path(checkpoint_dir), Path(out_dir)
    # Outside the repository only (AGENTS.md rule 7): neither the bundle, its record and manifest
    # copy, nor the downloaded v2 checkpoint may sit in the working tree, where they could be committed.
    repo = root.resolve()
    for what, path in (("output", out_dir), ("output", out_dir.parent), ("checkpoint", checkpoint_dir)):
        if path.resolve().is_relative_to(repo):
            raise ValueError(f"{what}: {path} is inside the repository; F5 is built outside it "
                             f"(AGENTS.md rule 7: no checkpoints in Git)")
    record_path = out_dir.parent / RECORD_NAME
    manifest_copy = out_dir.parent / MANIFEST_COPY_NAME
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise ValueError(f"output: {out_dir} exists and is not empty; bundles are never overwritten")
    for path in (record_path, manifest_copy):
        if path.exists():
            raise ValueError(f"output: {path} already exists; the F5 record is never overwritten")
    if created_at is None:
        created_at = _now_utc()
    if not isinstance(created_at, str) or not _CREATED_AT.fullmatch(created_at):
        raise ValueError(f"created_at must be YYYY-MM-DDTHH:MM:SSZ (UTC, seconds), got {created_at!r}")

    # 1-2. Provenance, evidence and the checkpoint folder; nothing is written before they pass.
    code_commit = _provenance(root, git)
    expected = expected_from_evidence(root)
    checks = _Checks()
    checks.items.extend(expected["checks"])
    files = _check_checkpoint_dir(checkpoint_dir, expected, checks)

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
        # 4. Verify through the API's loader (v2 dispatch); every manifest value equals the evidence.
        bundle_model = load_bundle_model(out_dir)
        _loaded_model_checks(bundle_model, expected, checks)
        _manifest_checks(bundle_model, expected, files, checks, code_commit=code_commit, created_at=created_at)
        manifest_bytes = manifest_path.read_bytes()
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        checks.equal("manifest_sha256", manifest_sha256, bundle_model.bundle.manifest_sha256,
                     f"SHA-256 of {MANIFEST_NAME} vs load_bundle")

        # 5. The F5 record and the manifest copy, outside the bundle, never overwritten ("x").
        record = {
            "schema_version": RECORD_SCHEMA_VERSION,
            "bundle_version": BUNDLE_VERSION,
            "manifest_sha256": manifest_sha256,
            "created_at": created_at,
            "code_commit": code_commit,
            "components": {
                "checkpoint": {"dir": CHECKPOINT_DIR, **expected["checkpoint"], "files": files},
                "profile": {"file": (root / PROFILE_SOURCE).name, **expected["profile"]},
                "policy": {"file": POLICY_FILE, **expected["policy"]},
            },
            "architecture": expected["architecture"],
            "bundle_seed": expected["bundle_seed"],
            "selection": expected["selection"],
            "provenance": expected["provenance"],
            "stage11_v2": expected["stage11_v2"],
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
    if not isinstance(data, dict) or data.get("schema_version") != RECORD_SCHEMA_VERSION \
            or data.get("bundle_version") != BUNDLE_VERSION:
        raise ValueError(f"record: {record} is not an F5 record ({BUNDLE_VERSION}, schema {RECORD_SCHEMA_VERSION})")
    return data


def verify(bundle_dir, record) -> dict:
    """Verify a copy of F5 against its record: load_bundle_model, manifest_sha256, every component,
    the architecture and bundle seed. ValueError on any difference.
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
        checks.equal("architecture", rec["architecture"], bm.architecture, src)
        checks.equal("checkpoint seed", rec["bundle_seed"], bm.metadata.get("seed"), src)
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
            "architecture": bm.architecture, "checks": checks.items}


def main(argv=None, *, root=DEFAULT_ROOT, git: GitState = git_state) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.bundle.f5", description="F5 ModelBundle v2 build and verify")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="build F5 from the v2 bundle-seed checkpoint folder and committed evidence")
    b.add_argument("--checkpoint-dir", required=True, type=Path,
                   help="the v2 search runs/<winner>_seed<bundle seed>/checkpoint/ folder (outside the repo)")
    b.add_argument("--out", required=True, type=Path)
    b.add_argument("--created-at", default=None)
    v = sub.add_parser("verify", help="verify a copy of F5 against its record")
    v.add_argument("--bundle", required=True, type=Path)
    v.add_argument("--record", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            record = build(args.checkpoint_dir, args.out, root=root, created_at=args.created_at, git=git)
            print(f"record: {args.out.parent / RECORD_NAME}")
            print(f"architecture: {record['architecture']} (bundle seed {record['bundle_seed']})")
            print(f"manifest_sha256: {record['manifest_sha256']}")
            print("F5 build OK")
        else:
            report = verify(args.bundle, args.record)
            print(f"architecture: {report['architecture']}")
            print(f"manifest_sha256: {report['manifest_sha256']}")
            print("F5 verify OK")
    except (ValueError, OSError) as e:
        print(f"F5 {args.command} FAILED: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
