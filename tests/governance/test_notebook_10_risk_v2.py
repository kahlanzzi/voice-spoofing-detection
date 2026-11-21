"""Static checks for the risk-v2 Colab notebook (notebooks/10_trackV_risk_v2.ipynb; D9, D40, D45, D49, D58, D64).

The notebook is committed with outputs cleared and is only run in Colab, so these tests read its JSON
and parse each code cell with ast; they never execute it. They pin what the official run must and
must not do: the winner and bundle seed come only from v2_risk_run.bundle_checkpoint, the exact
runtime (T4 2026.10, Python 3.13.16, torch 2.11.0+cu130) with require_versions in a fresh process after
the install, only val-B is cached (log-Mel winners only, twice, compared, equal to the Stage 10 hash),
both val-B input branches pass exactly one of --cache-dir / --root, the runner is invoked once with
the pre-declared rerun rule above it and no device argument, outputs go to risk/risk-v2 and
metadata/v2_risk only (never overwritten), nothing reads the fitted values, and the steps run in order.

The helpers mirror tests/governance/test_notebook_05_stage10.py and test_notebook_08_trackV_search.py.
"""
import ast
import json
import re
from pathlib import Path

from tests.governance.test_for_test_access import file_hits

REPO_ROOT = Path(__file__).resolve().parents[2]
REL = "notebooks/10_trackV_risk_v2.ipynb"
NOTEBOOK = REPO_ROOT / REL
NOTEBOOK_08 = REPO_ROOT / "notebooks" / "08_trackV_v2_search.ipynb"
CLONE = "/content/voice-spoofing-detection"  # where the notebook clones the repo in Colab
RISK_RUN = "src.tuning.v2_risk_run"
RUN_MARKER = f"'-m', '{RISK_RUN}'"  # the CLI call (the winner probe imports the module)
CANDIDATES = ("cnn-aug", "crnn", "aasist")
RERUN_RULE = (
    "Rerun rule (pre-declared for risk-v2, adapted from notebook 05): if the runner raises a check "
    "failure (search evidence, checkpoint hash, architecture, seed, provenance, M reproduction, "
    "completeness, determinism or fit), the risk-v2 run stops and the failure is reported; proceeding "
    "requires a logged decision. Only an infrastructure failure that happens before this cell starts "
    "(Drive, clone, pip, archive, copies) may be rerun from the same COMMIT. Nothing in this notebook "
    "reads or reacts to the fitted values."
)


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _cells() -> list[dict]:
    return _notebook()["cells"]


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _code() -> list[str]:
    return [_source(c) for c in _cells() if c["cell_type"] == "code"]


def _markdown() -> str:
    return "\n".join(_source(c) for c in _cells() if c["cell_type"] == "markdown")


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
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == name]


def _assigned(tree: ast.AST, name: str) -> list[str]:
    return [ast.unparse(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]


def _assigned_anywhere(name: str) -> list[str]:
    return [value for tree in _trees() for value in _assigned(tree, name)]


def _guarding_if(tree: ast.Module, node: ast.AST) -> str | None:
    """The unparsed test of the innermost `if` whose body (not orelse) contains node."""
    found = None
    for candidate in ast.walk(tree):
        if isinstance(candidate, ast.If) and any(node is n for s in candidate.body for n in ast.walk(s)):
            found = ast.unparse(candidate.test)
    return found


def _risk_run_call() -> list[ast.expr]:
    (call,) = [n for tree in _trees() for n in ast.walk(tree)
               if isinstance(n, ast.List) and RISK_RUN in _strings(n)]
    return call.elts


def _drive_chains() -> list[list[str]]:
    """Every maximal `DRIVE / a / b ...` chain, as its unparsed parts after DRIVE."""
    chains = []
    for tree in _trees():
        inner = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                if isinstance(node.left, ast.BinOp):
                    inner.add(id(node.left))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)) or id(node) in inner:
                continue
            parts, left = [], node
            while isinstance(left, ast.BinOp) and isinstance(left.op, ast.Div):
                parts.insert(0, ast.unparse(left.right))
                left = left.left
            if isinstance(left, ast.Name) and left.id == "DRIVE":
                chains.append(parts)
    return chains


# --- shape ---------------------------------------------------------------------------------------

def test_outputs_and_execution_counts_are_cleared():
    data = _notebook()
    assert data["nbformat"] == 4
    for cell in data["cells"]:
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
        "VERSION": "risk-v2",
        "SEARCH_VERSION": "model-v2",
        "DRIVE_ROOT": "/content/drive/MyDrive/voice-spoofing-detection",
    }
    clone = _cell_tree("'git', 'clone'")
    assert "len(COMMIT) == 40" in _asserted(clone)
    assert "'--detach', COMMIT" in ast.unparse(clone) and "assert head == COMMIT" in ast.unparse(clone)


def test_search_version_is_notebook_08s_constant():
    cells = json.loads(NOTEBOOK_08.read_text(encoding="utf-8"))["cells"]
    params_08 = ast.parse(_source([c for c in cells if c["cell_type"] == "code"][0]))
    (value_08,) = [ast.literal_eval(n.value) for n in params_08.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "SEARCH_VERSION" for t in n.targets)]
    assert _assigned(ast.parse(_code()[0]), "SEARCH_VERSION") == [repr(value_08)]
    assert "SEARCH_REPORT['version'] == SEARCH_VERSION" in _asserted(_cell_tree("bundle_checkpoint"))


def test_no_wandb_and_no_secrets():
    for cell in _cells():
        text = _source(cell)
        assert not re.search(r"wandb|W&B", text, re.IGNORECASE), cell["id"]
        assert "userdata" not in text and "getenv" not in text, cell["id"]
        assert not re.search(r"[0-9a-fA-F]{40}", text), cell["id"]
    for tree in _trees():
        for value in _strings(tree):
            assert not re.fullmatch(r"[A-Za-z0-9_\-]{32,}", value), value


# --- runtime -------------------------------------------------------------------------------------

def test_torch_is_never_installed():
    text = "\n".join(_source(c) for c in _cells())
    assert not re.search(r"(pip|uv|conda)\b.*\binstall\b.*torch", text, re.IGNORECASE)
    installs = [n for tree in _trees() for n in ast.walk(tree)
                if isinstance(n, ast.List) and "install" in _strings(n)]
    assert len(installs) == 1
    words = _strings(installs[0])
    assert not any("torch" in w.lower() for w in words), words
    assert {"requirements.txt", "requirements-cnn.txt"} <= set(words)


def test_exact_python_torch_and_t4_runtime():
    tree = _cell_tree("'pip', 'install'")
    asserted = _asserted(tree)
    assert "torch.__version__ == '2.11.0+cu130'" in asserted  # in the kernel, before the install
    assert "ENV['python'] == '3.13.16'" in asserted  # in a fresh process, after the install
    assert "ENV['torch'] == '2.11.0+cu130'" in asserted
    assert "'T4' in GPU_NAME" in asserted
    assert "nvidia-smi" in _strings(tree)
    metadata = _notebook()["metadata"]
    assert metadata["accelerator"] == "GPU" and metadata["colab"]["gpuType"] == "T4"
    assert "2026.10" in _markdown() and "3.13.16" in _markdown() and "2.11.0+cu130" in _markdown()


def test_require_versions_in_a_fresh_process_after_the_install():
    tree = _cell_tree("'pip', 'install'")
    (install,) = [n for n in _calls(tree, "subprocess.run") if "'pip', 'install'" in ast.unparse(n)]
    assert "'-r', 'requirements.txt', '-r', 'requirements-cnn.txt'" in ast.unparse(install)
    probes = [n for n in _calls(tree, "probe") if "require_versions()" in ast.unparse(n)]
    assert len(probes) == 1 and probes[0].lineno > install.end_lineno
    assert "from src.augmentation.version_guard import require_versions" in ast.unparse(probes[0])
    assert _assigned(tree, "VERSIONS") == ["ENV.pop('require_versions')"]
    # no second version helper and no in-kernel import of the guard
    for tree in _trees():
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                assert "version_guard" not in ast.unparse(node)
            if isinstance(node, ast.FunctionDef):
                assert "version" not in node.name


def test_probe_runs_in_a_fresh_process_inside_the_clone():
    (probe,) = [n for n in ast.walk(_cell_tree("def probe")) if isinstance(n, ast.FunctionDef) and n.name == "probe"]
    text = ast.unparse(probe)
    assert "[sys.executable, '-c', script], cwd=REPO" in text and "json.loads(result.stdout)" in text


def test_the_gpu_is_never_used_for_scoring():
    for src in _code():
        assert not re.search(r"\bcuda\b|torch\.cuda|--device|\.to\(", src), src[:80]
    assert "--device" not in [ast.unparse(e) for e in _risk_run_call()]


# --- winner and inputs ---------------------------------------------------------------------------

def test_winner_and_bundle_seed_come_only_from_bundle_checkpoint():
    tree = _cell_tree("bundle_checkpoint")
    assert _assigned(tree, "SEARCH_EVIDENCE") == ["REPO / 'docs/audit_evidence/v2_search'"]
    (probe,) = _calls(tree, "probe")
    text = ast.unparse(probe)
    assert "from src.tuning.v2_risk_run import bundle_checkpoint" in text
    assert "bundle_checkpoint(Path({str(SEARCH_EVIDENCE)!r}))" in text
    assert "from src.tuning.v2_search import run_dir_name" in text and "run_dir_name(winner, seed)" in text
    assert "from src.models.v2_checkpoint import INPUTS" in text and "INPUTS[winner]" in text
    assert _assigned_anywhere("BUNDLE") == [ast.unparse(probe)]
    winner_assign = [n for tr in _trees() for n in ast.walk(tr) if isinstance(n, ast.Assign)
                     and "WINNER" in ast.unparse(n.targets[0])]
    assert [ast.unparse(n) for n in winner_assign] == [
        "WINNER, BUNDLE_SEED = (BUNDLE['winner'], BUNDLE['bundle_seed'])"]
    assert _assigned_anywhere("BUNDLE_SEED") == []  # only in the tuple above
    assert "INPUT_KIND in ('logmel', 'waveform')" in _asserted(tree)
    # no duplicated selection rule: the notebook names no candidate and reads no selection field itself
    for t in _trees():
        for value in _strings(t):
            assert value not in CANDIDATES and value not in ("selected", "selection_report.json"), value


def test_val_b_cache_only_built_twice_compared_and_equal_to_stage10():
    strings = [s for tree in _trees() for s in _strings(tree)]
    for forbidden in ("train", "test", "val-A"):
        assert forbidden not in strings, forbidden
    tree = _cell_tree("src.features.logmel_cache")
    assert _assigned(tree, "CACHE_SPLITS") == ["('val-B',)"]
    (command,) = [n for n in ast.walk(tree) if isinstance(n, ast.List) and "src.features.logmel_cache" in _strings(n)]
    assert _guarding_if(tree, command) == "INPUT_KIND == 'logmel'"
    text = ast.unparse(tree)
    assert "for cache in (CACHE_A, CACHE_B):" in text
    assert "CACHE_A, CACHE_B = (Path('/content/cache_a'), Path('/content/cache_b'))" in text
    asserted = _asserted(tree)
    assert "CACHE_HASHES_B == CACHE_HASHES_A" in asserted
    for check in ("meta['profile_config_sha256'] == F3['config_sha256']",
                  "meta['profile_fingerprint'] == F3['fingerprint']",
                  "meta['code_commit'] == COMMIT",
                  "meta['npz_sha256'] == VAL_B_NPZ == VAL_B_NPZ_EXPECTED"):
        assert check in asserted, check
    assert "docs/audit_evidence/stage10_risk/run_report.json" in text
    assert _assigned(tree, "VAL_B_NPZ_EXPECTED") == ["STAGE10['caches']['val-B']['npz_sha256']"]
    assert "sha256(PROFILE) == F3['config_sha256']" in _asserted(_cell_tree("bundle_checkpoint"))


def test_both_val_b_input_branches_pass_exactly_one_of_cache_dir_and_root():
    tree = _cell_tree(RUN_MARKER)
    (branch,) = [n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == "INPUT_KIND == 'logmel'"]
    assert [ast.unparse(s) for s in branch.body] == ["VAL_B_ARGS = ['--cache-dir', str(CACHE_A)]"]
    assert [ast.unparse(s) for s in branch.orelse] == ["VAL_B_ARGS = ['--root', str(AUDIO_ROOT)]"]
    words = [ast.unparse(e) for e in _risk_run_call()]
    assert words.count("*VAL_B_ARGS") == 1
    assert "'--cache-dir'" not in words and "'--root'" not in words
    assert "\n".join(_code()).count("'--cache-dir'") == 1  # only in the log-Mel branch


def test_checkpoint_and_s_are_copied_outside_the_clone_and_never_loaded_here():
    code = "\n".join(_code())
    for loader in ("load_v2_checkpoint", "load_checkpoint", "safetensors", "load_selection_set"):
        assert loader not in code, loader
    tree = _cell_tree("shutil.copytree(CHECKPOINT_DRIVE")
    assert _assigned(tree, "CHECKPOINT_DRIVE") == ["SEARCH_OUT / 'runs' / RUN_DIR / 'checkpoint'"]
    assert _assigned(tree, "INPUTS_LOCAL") == ["Path('/content/v2_risk_inputs')"]
    assert _assigned(tree, "CHECKPOINT_LOCAL") == ["INPUTS_LOCAL / RUN_DIR / 'checkpoint'"]
    assert _assigned(tree, "S_LOCAL") == ["INPUTS_LOCAL / 'selection_set_s'"]
    text = ast.unparse(tree)
    assert "shutil.copytree(CHECKPOINT_DRIVE, CHECKPOINT_LOCAL)" in text
    assert "shutil.copytree(SELECTION_SET_DRIVE, S_LOCAL)" in text
    assert not "/content/v2_risk_inputs/".startswith(CLONE + "/")


# --- Drive ---------------------------------------------------------------------------------------

def test_drive_layout_constants():
    tree = _cell_tree("drive.mount('/content/drive')\n")
    assert _assigned(tree, "SEARCH_OUT") == ["DRIVE / 'checkpoints' / 'v2_search' / SEARCH_VERSION"]
    assert _assigned(tree, "SELECTION_SET_DRIVE") == ["DRIVE / 'processed' / 'v2' / 'selection_set_s'"]
    assert _assigned(tree, "CHECKPOINT_DEST") == ["DRIVE / 'checkpoints' / 'risk' / VERSION"]
    assert _assigned(tree, "EVIDENCE_DEST") == ["DRIVE / 'processed' / 'metadata' / 'v2_risk'"]


def test_outputs_under_risk_v2_and_metadata_v2_risk_only_never_v1():
    allowed = {
        ("'checkpoints'", "'v2_search'", "SEARCH_VERSION"),  # read
        ("'processed'", "'v2'", "'selection_set_s'"),  # read
        ("'checkpoints'", "'risk'", "VERSION"),  # written
        ("'processed'", "'metadata'", "'v2_risk'"),  # written
        ("'raw'",), ("'raw'", "'for_2sec_ff8c82c7.tar'"), ("'raw'", "'for_2sec_REVISION.txt'"),  # read
        ("path",),  # the re-verification after the remount (manifest paths under the two export folders)
    }
    chains = {tuple(c) for c in _drive_chains()}
    assert chains <= allowed, chains - allowed
    strings = [s for tree in _trees() for s in _strings(tree)]
    for v1 in ("risk-v1", "stage10", "stage8", "stage11", "cnn"):
        assert v1 not in strings, v1
    # every write goes to the two export folders, which are created exclusively
    tree = _cell_tree("EXPORT_MANIFEST = ")
    copies = [ast.unparse(c.args[1]) for c in _calls(tree, "shutil.copy2")]
    assert copies == ["dest / name"]


def test_existing_outputs_are_refused_before_the_run_and_before_export():
    code = "\n".join(_code())
    calls = [m.start() for m in re.finditer(r"^refuse_existing_outputs\(\)$", code, re.MULTILINE)]
    run = code.find(RUN_MARKER)
    export = code.find("CHECKPOINT_DEST.mkdir(")
    clone = code.find("'git', 'clone'")
    assert len(calls) == 2 and calls[0] < clone < run < calls[1] < export, (calls, run, export)


def test_no_output_inside_the_clone():
    allowed_content = ("/content/drive", CLONE, "/content/data", "/content/cache_a", "/content/cache_b",
                       "/content/v2_risk_inputs", "/content/v2_risk_out", "/content/pytest_v2_risk.xml")
    for tree in _trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and ast.unparse(node.left) == "REPO":
                assert ast.literal_eval(node.right).startswith(("docs/", "configs/", "data/manifests/")), ast.unparse(node)
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("/content"):
                assert node.value.startswith(allowed_content), node.value
    assert _assigned(_cell_tree("'git', 'clone'"), "REPO") == [f"Path('{CLONE}')"]


# --- run -----------------------------------------------------------------------------------------

def test_risk_run_is_invoked_exactly_once_outside_loops_branches_and_retries():
    assert "\n".join(_code()).count(f"'{RISK_RUN}'") == 1  # the CLI call; the winner probe only imports it
    for tree in _trees():
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While, ast.AsyncFor, ast.Try, ast.If)):
                assert RISK_RUN not in _strings(node)
            if isinstance(node, (ast.FunctionDef, ast.Lambda)):
                assert RISK_RUN not in _strings(node)


def test_risk_run_arguments():
    words = [ast.unparse(e) for e in _risk_run_call()]
    assert words[:3] == ["sys.executable", "'-m'", f"'{RISK_RUN}'"], words
    rest = [w for w in words[3:] if w != "*VAL_B_ARGS"]
    args = dict(zip(rest[0::2], rest[1::2]))
    assert len(rest) == 2 * len(args)
    assert args == {
        "'--checkpoint'": "str(CHECKPOINT_LOCAL)",
        "'--selection-set'": "str(S_LOCAL)",
        "'--profile'": "str(PROFILE)",
        "'--config'": "str(REPO / 'configs/v2_candidates.yaml')",
        "'--manifest'": "str(MANIFEST)",
        "'--out'": "str(OUT)",
        "'--version'": "VERSION",
    }
    assert _assigned(_cell_tree(RUN_MARKER), "OUT") == ["Path('/content/v2_risk_out')"]
    assert _assigned(_cell_tree("bundle_checkpoint"), "MANIFEST") == [
        "REPO / 'data/manifests/for2sec_frozen_manifest.csv'"]
    assert _assigned(_cell_tree("bundle_checkpoint"), "PROFILE") == ["REPO / 'configs/preprocessing.yaml'"]


def test_rerun_rule_is_in_the_markdown_right_before_the_run():
    cells = _cells()
    (i,) = [k for k, c in enumerate(cells) if c["cell_type"] == "code" and RUN_MARKER in _source(c)]
    assert cells[i - 1]["cell_type"] == "markdown"
    assert RERUN_RULE in _source(cells[i - 1])


def test_assert_clean_runs_right_before_and_right_after_the_run():
    code = _code()
    i = [RUN_MARKER in s for s in code].index(True)
    run = code[i]
    assert -1 < run.rfind("assert_clean()") < run.find(RUN_MARKER)
    assert code[i + 1].startswith("assert_clean()\n")


def test_head_and_clean_worktree_before_and_after_every_long_cell():
    long_cells = 0
    for src in _code():
        tree = ast.parse(src)
        long_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and (
            ast.unparse(n.func) == "stream"
            or (ast.unparse(n.func) == "subprocess.run" and ("'pip', 'install'" in ast.unparse(n)
                                                           or "src.features.logmel_cache" in ast.unparse(n)))
            or ast.unparse(n.func) in ("archive.extractall", "probe"))]
        if not long_calls:
            continue
        long_cells += 1
        clean = [n.lineno for n in _calls(tree, "assert_clean")]
        assert clean and min(clean) < min(c.lineno for c in long_calls), src[:80]
        if RUN_MARKER not in src:  # the run cell's closing check is the next cell's first line
            assert max(clean) > max(c.end_lineno for c in long_calls), src[:80]
    assert long_cells >= 6  # install, winner probe, archive, pytest, cache, run


def test_post_run_identity_checks_only():
    tree = _cell_tree("RUN_REPORT = ")
    asserted = _asserted(tree)
    for check in ("RUN_REPORT['code_commit'] == COMMIT",
                  "RUN_REPORT['version'] == VERSION",
                  "RUN_REPORT['architecture'] == WINNER",
                  "RUN_REPORT['bundle_seed'] == BUNDLE_SEED",
                  "RUN_REPORT['checkpoint_sha256'] == BUNDLE['checkpoint_sha256']",
                  "RUN_REPORT['val_b_input']['form'] == INPUT_KIND",
                  "RUN_REPORT['scoring']['device'] == 'cpu'",
                  "RUN_REPORT['profile_fingerprint'] == F3['fingerprint']",
                  "RUN_REPORT['profile_config_sha256'] == F3['config_sha256']"):
        assert check in asserted, check
    assert len(asserted) == 9, asserted


def test_nothing_reads_the_fitted_values():
    code = "\n".join(_code())
    for key in ("headline", "threshold", "half_width", "deferral", "eer", "calibration", "selection_m",
                "risk_policy_sha256", "RiskPolicy"):
        assert key not in code, key
    assert "json.loads((OUT / 'diagnostics_report.json')" not in code  # printed as text only


# --- export --------------------------------------------------------------------------------------

def test_export_folders_are_created_with_exist_ok_false():
    tree = _cell_tree("EXPORT_MANIFEST = ")
    made = {}
    for call in ast.walk(tree):
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "mkdir"
                and isinstance(call.func.value, ast.Name)):
            made[call.func.value.id] = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
    assert made == {"CHECKPOINT_DEST": {"exist_ok": False}, "EVIDENCE_DEST": {"exist_ok": False}}


def test_export_destinations():
    (exports,) = [n.value for n in ast.walk(_cell_tree("EXPORT_MANIFEST = ")) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "EXPORTS" for t in n.targets)]
    assert {ast.literal_eval(k): ast.unparse(v) for k, v in zip(exports.keys, exports.values)} == {
        "risk_policy.json": "CHECKPOINT_DEST",
        "val_b_scores.npz": "CHECKPOINT_DEST",
        "run_report.json": "EVIDENCE_DEST",
        "diagnostics_report.json": "EVIDENCE_DEST",
    }
    assert _assigned(_cell_tree("EXPORT_MANIFEST = "), "MANIFEST_PATH") == [
        "EVIDENCE_DEST / 'v2_risk_export_manifest.json'"]


def test_export_manifest_records_the_run_evidence():
    (manifest,) = [n.value for n in ast.walk(_cell_tree("EXPORT_MANIFEST = ")) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "EXPORT_MANIFEST" for t in n.targets)]
    entries = {k.value: ast.unparse(v) for k, v in zip(manifest.keys, manifest.values)}
    assert set(entries) == {"commit", "version", "search_version", "architecture", "bundle_seed", "environment",
                            "require_versions", "created_utc", "for_revision", "for_archive_sha256_observed",
                            "pytest_counts", "cache_sha256", "f3", "files"}
    assert entries["require_versions"] == "VERSIONS" and entries["environment"] == "ENV"
    assert entries["cache_sha256"] == "{'cache_a': CACHE_HASHES_A, 'cache_b': CACHE_HASHES_B}"


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
    assert "docs/audit_evidence/v2_risk/" in text
    for name in ("risk_policy.json", "run_report.json", "diagnostics_report.json",
                 "v2_risk_export_manifest.json", "val_b_scores.npz", "val_b_scores_sha256", "notebook 09"):
        assert name in text, name
    assert "stays on Drive only" in text and "AGENTS.md rule 7" in text
    assert text.index("results PR") < text.index("notebook 09")


# --- guards and order ----------------------------------------------------------------------------

def test_for_guard_finds_nothing():
    assert file_hits(REL, NOTEBOOK.read_text(encoding="utf-8")) == []


def test_steps_run_in_order():
    code = "\n".join(_code())
    markers = ["drive.mount(", "refuse_existing_outputs()", "'git', 'clone'", "assert_clean()", "nvidia-smi",
               "'pip', 'install'", "require_versions()", "bundle_checkpoint(", "for_2sec_ff8c82c7.tar", "'pytest'",
               "src.features.logmel_cache", "shutil.copytree(CHECKPOINT_DRIVE", RUN_MARKER, "RUN_REPORT = ",
               "CHECKPOINT_DEST.mkdir(", "drive.flush_and_unmount()", "force_remount=True"]
    positions = [code.find(m) for m in markers]
    assert -1 not in positions, dict(zip(markers, positions))
    assert positions == sorted(positions), dict(zip(markers, positions))
