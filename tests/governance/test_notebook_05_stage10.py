"""Static checks for the Stage 10 Colab notebook (notebooks/05_stage10_risk_policy.ipynb; D9, D40, D45).

The notebook is committed with outputs cleared and is only run in Colab, so these tests read its JSON
and parse each code cell with ast; they never execute it. They pin what the official run must and
must not do: only val-A and val-B are cached, the risk run is invoked once with the pre-declared
rerun rule above it, nothing reads the fitted values, no GPU is required, W&B is absent, outputs are
never overwritten, and the steps run in order.
"""
import ast
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO_ROOT / "notebooks" / "05_stage10_risk_policy.ipynb"
CLONE = "/content/voice-spoofing-detection"  # where the notebook clones the repo in Colab
RISK_RUN = "src.tuning.risk_run"
RERUN_RULE = (
    "Rerun rule (pre-declared 2026-10-28): if the runner raises a check failure (checkpoint hash, "
    "seed, provenance, val-A reproduction, completeness, determinism or fit), Stage 10 stops and the "
    "failure is reported; proceeding requires a logged decision. Only an infrastructure failure that "
    "happens before this cell starts (Drive, clone, pip, archive) may be rerun from the same COMMIT. "
    "Nothing in this notebook reads or reacts to the fitted values."
)


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _cells() -> list[dict]:
    return _notebook()["cells"]


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _code() -> list[str]:
    return [_source(c) for c in _cells() if c["cell_type"] == "code"]


def _trees() -> list[ast.Module]:
    return [ast.parse(src) for src in _code()]


def _strings(node: ast.AST) -> list[str]:
    return [n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def _cell_tree(marker: str) -> ast.Module:
    (tree,) = [t for t, src in zip(_trees(), _code()) if marker in src]
    return tree


def _asserted(tree: ast.Module) -> list[str]:
    return [ast.unparse(n.test) for n in ast.walk(tree) if isinstance(n, ast.Assert)]


def _calls(tree: ast.AST, name: str) -> list[ast.Call]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == name]


def _risk_run_call() -> list[ast.expr]:
    (call,) = [n for tree in _trees() for n in ast.walk(tree)
               if isinstance(n, ast.List) and RISK_RUN in _strings(n)]
    return call.elts


def test_outputs_and_execution_counts_are_cleared():
    for cell in _cells():
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None, cell["id"]


def test_every_code_cell_follows_a_markdown_cell():
    cells = _cells()
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            assert i > 0 and cells[i - 1]["cell_type"] == "markdown", cell["id"]


def test_no_shell_magics_so_every_cell_is_plain_python():
    for src in _code():
        assert not any(line.lstrip().startswith(("!", "%")) for line in src.splitlines())
        ast.parse(src)


def test_parameters_are_the_placeholder_commit_and_fixed_versions():
    (params,) = [c for c in _cells() if c["cell_type"] == "code"][:1]
    values = {t.id: ast.literal_eval(n.value) for n in ast.parse(_source(params)).body
              if isinstance(n, ast.Assign) for t in n.targets}
    assert values == {
        "REPO_URL": "https://github.com/kahlanzzi/voice-spoofing-detection.git",
        "COMMIT": "REPLACE_WITH_APPROVED_COMMIT",
        "VERSION": "risk-v1",
        "CNN_VERSION": "cnn-v1",
        "DRIVE_ROOT": "/content/drive/MyDrive/voice-spoofing-detection",
    }
    assert "len(COMMIT) == 40" in _asserted(_cell_tree("'git', 'clone'"))


def test_no_wandb_and_no_secrets():
    for cell in _cells():
        text = _source(cell)
        assert not re.search(r"wandb|W&B", text, re.IGNORECASE), cell["id"]
        assert "userdata" not in text and "getenv" not in text, cell["id"]
        assert not re.search(r"[0-9a-fA-F]{40}", text), cell["id"]
    for tree in _trees():
        for value in _strings(tree):
            assert not re.fullmatch(r"[A-Za-z0-9_\-]{32,}", value), value


def test_torch_is_never_installed():
    text = "\n".join(_source(c) for c in _cells())
    assert not re.search(r"(pip|uv|conda)\b.*\binstall\b.*torch", text, re.IGNORECASE)
    installs = [n for tree in _trees() for n in ast.walk(tree)
                if isinstance(n, ast.List) and "install" in _strings(n)]
    assert len(installs) == 1
    words = _strings(installs[0])
    assert not any("torch" in w.lower() for w in words), words
    assert {"requirements.txt", "requirements-cnn.txt"} <= set(words)


def test_python_3_13_and_torch_2_11_are_asserted_but_no_gpu():
    asserted = _asserted(_cell_tree("'pip', 'install'"))
    assert "ENV['python'].startswith('3.13.')" in asserted  # D15
    assert "torch.__version__.startswith('2.11.0')" in asserted
    for cell in _cells():
        if cell["cell_type"] == "code":
            assert not re.search(r"cuda|gpu", _source(cell), re.IGNORECASE), cell["id"]
    metadata = _notebook()["metadata"]
    assert "accelerator" not in metadata and "gpuType" not in metadata.get("colab", {})


def test_only_val_a_and_val_b_are_cached():
    strings = [s for tree in _trees() for s in _strings(tree)]
    for forbidden in ("train", "test"):
        assert forbidden not in strings, forbidden
    cache_cells = [t for t, src in zip(_trees(), _code()) if "src.features.logmel_cache" in src]
    assert len(cache_cells) == 1
    splits = [n.value for n in ast.walk(cache_cells[0]) if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "CACHE_SPLITS" for t in n.targets)]
    assert len(splits) == 1 and ast.literal_eval(splits[0]) == ("val-A", "val-B")
    assert "assert CACHE_HASHES_B == CACHE_HASHES_A, 'the two cache builds differ'" in "\n".join(_code())


def test_checkpoint_is_copied_outside_the_clone_and_never_loaded_here():
    code = "\n".join(_code())
    assert "load_checkpoint" not in code and "safetensors" not in code
    assert "DRIVE / 'checkpoints' / 'cnn' / CNN_VERSION / 'seed_0'" in code
    tree = _cell_tree("shutil.copytree(")
    (local,) = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "CHECKPOINT_LOCAL" for t in n.targets)]
    assert ast.unparse(local) == "Path('/content/stage8_checkpoint') / 'seed_0'", ast.unparse(local)
    assert not "/content/stage8_checkpoint/".startswith(CLONE + "/")


def test_risk_run_is_invoked_exactly_once_outside_loops_and_retries():
    assert "\n".join(_code()).count(RISK_RUN) == 1
    for tree in _trees():
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While, ast.AsyncFor, ast.Try)):
                assert RISK_RUN not in _strings(node)
            if isinstance(node, (ast.FunctionDef, ast.Lambda)):
                assert RISK_RUN not in _strings(node)


def test_risk_run_arguments():
    elts = _risk_run_call()
    words = [ast.unparse(e) for e in elts]
    args = dict(zip(words[3::2], words[4::2]))
    assert words[:3] == ["sys.executable", "'-m'", f"'{RISK_RUN}'"], words
    assert args == {
        "'--cache-dir'": "str(CACHE_A)",
        "'--checkpoint'": "str(CHECKPOINT_LOCAL)",
        "'--profile'": "str(PROFILE)",
        "'--config'": "str(REPO / 'configs/cnn.yaml')",
        "'--manifest'": "str(MANIFEST)",
        "'--stage8-evidence'": "str(REPO / 'docs/audit_evidence/stage8_cnn')",
        "'--out'": "str(OUT)",
        "'--version'": "VERSION",
    }
    assert "OUT = Path('/content/stage10_out')" in _code()[[RISK_RUN in s for s in _code()].index(True)]


def test_rerun_rule_is_in_the_markdown_right_before_the_run():
    cells = _cells()
    (i,) = [k for k, c in enumerate(cells) if c["cell_type"] == "code" and RISK_RUN in _source(c)]
    assert cells[i - 1]["cell_type"] == "markdown"
    assert RERUN_RULE in _source(cells[i - 1])


def test_assert_clean_runs_right_before_and_right_after_the_run():
    code = _code()
    i = [RISK_RUN in s for s in code].index(True)
    run = code[i]
    assert -1 < run.rfind("assert_clean()") < run.find(RISK_RUN)
    assert code[i + 1].startswith("assert_clean()\n")


def test_post_run_checks_bind_the_report_to_the_run():
    asserted = _asserted(_cell_tree("RUN_REPORT = "))
    for check in ("RUN_REPORT['code_commit'] == COMMIT",
                  "RUN_REPORT['profile_fingerprint'] == F3['fingerprint']",
                  "RUN_REPORT['profile_config_sha256'] == F3['config_sha256']",
                  "RUN_REPORT['bundle_seed'] == 0"):
        assert check in asserted, check


def test_nothing_reads_the_fitted_values():
    code = "\n".join(_code())
    for key in ("headline", "threshold", "half_width", "deferral", "eer", "calibration",
                "risk_policy_sha256", "RiskPolicy"):
        assert key not in code, key
    assert "json.loads((OUT / 'diagnostics_report.json')" not in code  # printed as text only


def test_existing_outputs_are_refused_before_the_run_and_before_export():
    code = "\n".join(_code())
    calls = [m.start() for m in re.finditer(r"^refuse_existing_outputs\(\)$", code, re.MULTILINE)]
    run = code.find(RISK_RUN)
    export = code.find("CHECKPOINT_DEST.mkdir(")
    assert len(calls) == 2 and calls[0] < run < calls[1] < export, (calls, run, export)


def test_export_folders_are_created_with_exist_ok_false():
    tree = _cell_tree("EXPORT_MANIFEST = ")
    made = {}
    for call in ast.walk(tree):
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "mkdir"
                and isinstance(call.func.value, ast.Name)):
            made[call.func.value.id] = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
    assert made == {"CHECKPOINT_DEST": {"exist_ok": False}, "EVIDENCE_DEST": {"exist_ok": False}}


def test_export_destinations():
    code = "\n".join(_code())
    assert "CHECKPOINT_DEST = DRIVE / 'checkpoints' / 'risk' / VERSION" in code
    assert "EVIDENCE_DEST = DRIVE / 'processed' / 'metadata' / 'stage10'" in code
    (exports,) = [n.value for n in ast.walk(_cell_tree("EXPORT_MANIFEST = ")) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "EXPORTS" for t in n.targets)]
    assert {ast.literal_eval(k): ast.unparse(v) for k, v in zip(exports.keys, exports.values)} == {
        "risk_policy.json": "CHECKPOINT_DEST",
        "val_b_scores.npz": "CHECKPOINT_DEST",
        "run_report.json": "EVIDENCE_DEST",
        "diagnostics_report.json": "EVIDENCE_DEST",
    }
    assert "EVIDENCE_DEST / 'stage10_export_manifest.json'" in code


def test_export_manifest_records_the_run_evidence():
    (manifest,) = [n.value for n in ast.walk(_cell_tree("EXPORT_MANIFEST = ")) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "EXPORT_MANIFEST" for t in n.targets)]
    keys = {k.value for k in manifest.keys}
    assert keys == {"commit", "version", "cnn_version", "python", "numpy", "torch", "created_utc",
                    "for_revision", "for_archive_sha256_observed", "pytest_counts", "cache_sha256",
                    "f3", "files"}


def test_export_is_rehashed_after_flush_and_remount():
    code = "\n".join(_code())
    flush = code.find("drive.flush_and_unmount()")
    remount = code.find("drive.mount('/content/drive', force_remount=True)")
    assert -1 < code.find("EXPORT_MANIFEST = ") < flush < remount, (flush, remount)
    after = code[remount:]
    for check in ("sha256(DRIVE / path)", "missing", "extra"):
        assert check in after, check


def test_final_markdown_names_the_committed_files_and_keeps_scores_on_drive():
    last = _cells()[-1]
    assert last["cell_type"] == "markdown"
    text = _source(last)
    assert "docs/audit_evidence/stage10_risk/" in text
    for name in ("risk_policy.json", "run_report.json", "diagnostics_report.json",
                 "stage10_export_manifest.json", "val_b_scores.npz"):
        assert name in text, name
    assert "stays on Drive" in text


def test_steps_run_in_order():
    code = "\n".join(_code())
    markers = ["drive.mount(", "'git', 'clone'", "assert_clean()", "'pip', 'install'", "'pytest'",
               "src.features.logmel_cache", "shutil.copytree(", RISK_RUN, "RUN_REPORT = ",
               "CHECKPOINT_DEST.mkdir(", "drive.flush_and_unmount()", "force_remount=True"]
    positions = [code.find(m) for m in markers]
    assert -1 not in positions, dict(zip(markers, positions))
    assert positions == sorted(positions), dict(zip(markers, positions))
