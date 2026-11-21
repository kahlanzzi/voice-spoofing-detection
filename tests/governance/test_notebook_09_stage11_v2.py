"""Static checks for the v2 Stage 11 Colab notebook (notebooks/09_stage11_v2_for_test.ipynb; D46, D48 (7), D57, D58, D64).

The notebook is committed with outputs cleared and is only run in Colab, so these tests read its JSON
and parse each code cell with ast; they never execute it. They pin notebook 06's D46 order and its
guards, adapted to the v2 model set: the T4 runtime (version 2026.10, D58 (1)) with CPU scoring, the
environment gate with require_versions in a fresh process after the install (D64), the winner, its
input kind and the bundle seed read only from the committed evidence, log-Mel caches (val-A and
val-B only) for a log-Mel winner and none (no --cache-* argument) for aasist, no MFCC, D58 reference
or baseline, the read-only registry pre-check before the rehearsal, the rehearsal before the claim,
the claim cell and every later cell refusing the dry-run mode, outputs under stage11_v2/ only (v1's
stage11/run/ never written, the shared registry never written by the notebook), the Drive flush and
remount between claim and score, run_id read from claim.json, one rehearse, claim, score and report,
6 clip-score files, no metric read before the results view, and the FoR held-out guard.

The helpers mirror tests/governance/test_notebook_06_stage11.py and test_notebook_10_risk_v2.py:
there they are private and bound to those notebooks' paths, so they are copied here rather than
imported.
"""
import ast
import json
import re
from pathlib import Path

from tests.governance.test_for_test_access import file_hits

REPO_ROOT = Path(__file__).resolve().parents[2]
REL = "notebooks/09_stage11_v2_for_test.ipynb"
NOTEBOOK = REPO_ROOT / REL
NOTEBOOK_08 = REPO_ROOT / "notebooks" / "08_trackV_v2_search.ipynb"
CLONE = "/content/voice-spoofing-detection"  # where the notebook clones the repo in Colab
RUNNER = "src.evaluation.stage11_v2_run"
REPORT = "src.evaluation.stage11_v2_report"
CANDIDATES = ("cnn-aug", "crnn", "aasist")
OFFICIAL = "assert MODE == 'official', 'dry run: stop before the claim'"
RERUN_RULE = (
    "Rerun rule (D46 (8) and D48 (7), pre-declared for the v2 claim): before the claim, an infrastructure "
    "failure (Drive, clone, pip, archive, artifact copies) may be rerun from the same COMMIT; an "
    "environment-gate failure (including require_versions), a reproduction-gate failure, a registry "
    "pre-check mismatch, or a rehearsal that ends with a failed check (failed_check not null) stops the v2 "
    "Stage 11 run until a logged decision; a rehearsal that crashes on an infrastructure error "
    "(failed_check null) may be rerun from the same COMMIT. After the claim, there is no rerun ever without "
    "a new decision, including a crash before any score exists; everything written stays for team review. "
    "The v2 claim cannot be repeated. Nothing in this notebook reads or reacts to the test results before "
    "they are exported."
)
RUNNER_FILES = ("claim.json", "environment.json", "pip_freeze.txt", "scoring_started.json",
                "environment_score.json", "features.json", "run_report.json")


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


def _cell_index(marker: str) -> int:
    (i,) = [k for k, src in enumerate(_code()) if marker in src]
    return i


def _cell_tree(marker: str) -> ast.Module:
    return _trees()[_cell_index(marker)]


def _asserted(tree: ast.AST) -> list[str]:
    return [ast.unparse(n.test) for n in ast.walk(tree) if isinstance(n, ast.Assert)]


def _assigned(tree: ast.AST, name: str) -> list[ast.expr]:
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]


def _calls(tree: ast.AST, name: str) -> list[ast.Call]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == name]


def _guarding_if(tree: ast.Module, node: ast.AST) -> str | None:
    """The unparsed test of the innermost `if` whose body (not orelse) contains node."""
    found = None
    for candidate in ast.walk(tree):
        if isinstance(candidate, ast.If) and any(node is n for s in candidate.body for n in ast.walk(s)):
            found = ast.unparse(candidate.test)
    return found


def _runner_calls(subcommand: str) -> list[ast.List]:
    """Command lists [sys.executable, '-m', RUNNER, <subcommand>, ...] (RUNNER by name)."""
    calls = []
    for tree in _trees():
        for node in ast.walk(tree):
            if (isinstance(node, ast.List) and len(node.elts) > 3 and ast.unparse(node.elts[2]) == "RUNNER"
                    and isinstance(node.elts[3], ast.Constant) and node.elts[3].value == subcommand):
                calls.append(node)
    return calls


def _call_args(call: ast.List) -> dict[str, str]:
    words = [ast.unparse(e) for e in call.elts[4:]]
    args, i = {}, 0
    while i < len(words):
        if words[i].startswith("*"):
            args[words[i]] = ""
            i += 1
        else:
            args[words[i]] = words[i + 1]
            i += 2
    return args


def _drive_chains() -> list[list[str]]:
    """Every maximal `DRIVE / a / b ...` chain, as its unparsed parts after DRIVE."""
    chains = []
    for tree in _trees():
        inner = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and isinstance(node.left, ast.BinOp):
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


def _results_cell() -> int:
    return _cell_index("display(Markdown(")


# --- Notebook shape ---------------------------------------------------------------------------------

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


def test_parameters_are_the_placeholder_commit_dry_run_and_fixed_versions():
    params = _trees()[0]
    values = {t.id: ast.literal_eval(n.value) for n in params.body
              if isinstance(n, ast.Assign) for t in n.targets}
    assert values == {
        "REPO_URL": "https://github.com/kahlanzzi/voice-spoofing-detection.git",
        "COMMIT": "REPLACE_WITH_APPROVED_COMMIT",
        "MODE": "dry_run",
        "RISK_VERSION": "risk-v2",
        "SEARCH_VERSION": "model-v2",
        "DRIVE_ROOT": "/content/drive/MyDrive/voice-spoofing-detection",
        "RESULTS_REF": "docs/results/stage11_v2_for_test.md",
    }
    assert _asserted(params) == ["MODE in ('dry_run', 'official')"]
    clone = _cell_tree("'git', 'clone'")
    assert "len(COMMIT) == 40" in _asserted(clone)
    assert "'--detach', COMMIT" in ast.unparse(clone) and "assert head == COMMIT" in ast.unparse(clone)


def test_search_version_is_notebook_08s_constant():
    cells = json.loads(NOTEBOOK_08.read_text(encoding="utf-8"))["cells"]
    params_08 = ast.parse(_source([c for c in cells if c["cell_type"] == "code"][0]))
    (value_08,) = [ast.literal_eval(n.value) for n in params_08.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "SEARCH_VERSION" for t in n.targets)]
    assert [ast.literal_eval(v) for v in _assigned(_trees()[0], "SEARCH_VERSION")] == [value_08]


def test_no_wandb_and_no_secrets():
    for cell in _cells():
        text = _source(cell)
        assert not re.search(r"wandb|W&B", text, re.IGNORECASE), cell["id"]
        assert "userdata" not in text and "getenv" not in text, cell["id"]
        assert not re.search(r"[0-9a-fA-F]{32}", text), cell["id"]  # no SHA, no typed run_id
    for tree in _trees():
        for value in _strings(tree):
            assert not re.fullmatch(r"[A-Za-z0-9_\-]{32,}", value), value


def test_the_for_held_out_guard_finds_nothing():
    assert file_hits(REL, NOTEBOOK.read_text(encoding="utf-8")) == []


# --- Environment gate (D46 (1), (2); D58 (1); D64) --------------------------------------------------

def test_torch_is_never_installed():
    text = "\n".join(_source(c) for c in _cells())
    assert not re.search(r"(pip|uv|conda)\b.*\binstall\b.*torch", text, re.IGNORECASE)
    installs = [n for tree in _trees() for n in ast.walk(tree)
                if isinstance(n, ast.List) and "install" in _strings(n)]
    assert len(installs) == 1
    words = _strings(installs[0])
    assert not any("torch" in w.lower() for w in words), words
    assert {"requirements.txt", "requirements-cnn.txt"} <= set(words)


def test_python_torch_and_t4_are_asserted():
    tree = _cell_tree("'pip', 'install'")
    asserted = _asserted(tree)
    assert "ENV['python'] == '3.13.16'" in asserted  # D15, D58 (1): runtime 2026.10
    assert "ENV['torch'] == '2.11.0+cu130'" in asserted  # D58 (1)
    assert "torch.__version__ == '2.11.0+cu130'" in asserted  # before the install
    assert "ENV['torch'] == torch.__version__" in asserted
    assert not any("startswith('3.13" in a or "startswith('2.11" in a for a in asserted), asserted
    assert "'T4' in GPU_NAME" in asserted  # D46 (1)
    (gpu,) = _assigned(tree, "GPU_NAME")
    (command,) = [n for n in ast.walk(gpu) if isinstance(n, ast.List)]
    assert ast.literal_eval(command) == ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    metadata = _notebook()["metadata"]
    assert metadata["accelerator"] == "GPU" and metadata["colab"]["gpuType"] == "T4"


def test_markdown_names_runtime_version_2026_10():
    # D58 (1): the runtime version is chosen in the Colab UI, so the markdown must say which.
    cells = _cells()
    for marker in ("'pip', 'install'", "REPO_URL = "):
        (i,) = [k for k, c in enumerate(cells) if c["cell_type"] == "code" and marker in _source(c)]
        text = _source(cells[i - 1])
        assert "2026.10" in text and "T4" in text, marker
        assert "Version d'exécution / Runtime version" in text, marker


def test_require_versions_in_a_fresh_process_after_the_install():
    tree = _cell_tree("'pip', 'install'")
    (install,) = [n for n in _calls(tree, "subprocess.run") if "'pip', 'install'" in ast.unparse(n)]
    assert "'-r', 'requirements.txt', '-r', 'requirements-cnn.txt'" in ast.unparse(install)
    probes = [n for n in _calls(tree, "probe") if "require_versions()" in ast.unparse(n)]
    assert len(probes) == 1 and probes[0].lineno > install.end_lineno
    assert "from src.augmentation.version_guard import require_versions" in ast.unparse(probes[0])
    assert [ast.unparse(v) for v in _assigned(tree, "VERSIONS")] == ["ENV.pop('require_versions')"]
    # the result goes into the manifest; no second version helper and no in-kernel import of the guard
    assert "'require_versions': VERSIONS" in _code()[_cell_index("def base_manifest(")]
    for t in _trees():
        for node in ast.walk(t):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                assert "version_guard" not in ast.unparse(node)
            if isinstance(node, ast.FunctionDef):
                assert "version" not in node.name


def test_probe_runs_in_a_fresh_process_inside_the_clone():
    (probe,) = [n for n in ast.walk(_cell_tree("def probe")) if isinstance(n, ast.FunctionDef) and n.name == "probe"]
    text = ast.unparse(probe)
    assert "[sys.executable, '-c', script], cwd=REPO" in text and "json.loads(result.stdout)" in text


def test_nothing_runs_on_the_gpu():
    code = "\n".join(_code())
    assert "--device" not in code
    assert not re.search(r"\.cuda\(|torch\.cuda|\.to\(|['\"]cuda", code), "no model or tensor moves to a GPU"
    for subcommand in ("rehearse", "claim", "score"):
        for call in _runner_calls(subcommand):
            assert not any("device" in s or "cuda" in s.lower() for s in _strings(call)), subcommand


def test_pytest_counts_come_from_junit_and_must_be_clean():
    tree = _cell_tree("'pytest'")
    asserted = _asserted(tree)
    assert "COUNTS['failures'] == COUNTS['errors'] == COUNTS['skipped'] == 0" in asserted
    assert "COUNTS['tests'] > 0" in asserted
    src = _code()[_cell_index("'pytest'")]
    assert "'REQUIRE_TORCH_TESTS': '1'" in src and "--junitxml=" in src


# --- Winner from evidence ---------------------------------------------------------------------------

def test_winner_and_bundle_seed_come_only_from_the_committed_evidence():
    tree = _cell_tree("bundle_checkpoint")
    assert [ast.unparse(v) for v in _assigned(tree, "SEARCH_EVIDENCE")] == ["REPO / 'docs/audit_evidence/v2_search'"]
    assert [ast.unparse(v) for v in _assigned(tree, "RISK_EVIDENCE")] == ["REPO / 'docs/audit_evidence/v2_risk'"]
    (probe,) = _calls(tree, "probe")
    text = ast.unparse(probe)
    for needed in ("from src.tuning.v2_risk_run import bundle_checkpoint",
                   "bundle_checkpoint(Path({str(SEARCH_EVIDENCE)!r}))",
                   "from src.models.v2_checkpoint import INPUTS", "INPUTS[winner]",
                   "from src.tuning.v2_search import run_dir_name",
                   "from src.evaluation.stage11_v2_run import SEEDS", "run_dir_name(winner, k) for k in SEEDS"):
        assert needed in text, needed
    bundles = [n for t in _trees() for n in ast.walk(t) if isinstance(n, ast.Assign)
               and any(isinstance(x, ast.Name) and x.id == "BUNDLE" for x in n.targets)]
    assert [ast.unparse(n.value) for n in bundles] == [ast.unparse(probe)]
    # WINNER and BUNDLE_SEED are assigned once, from the probe's output, never from a literal.
    targets = [n for t in _trees() for n in ast.walk(t) if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign))
               and re.search(r"\b(WINNER|BUNDLE_SEED|INPUT_KIND|RUN_DIRS)\b",
                             ast.unparse(n.targets[0] if isinstance(n, ast.Assign) else n.target))]
    assert sorted(ast.unparse(n) for n in targets) == sorted([
        "WINNER, BUNDLE_SEED = (BUNDLE['winner'], BUNDLE['bundle_seed'])",
        "INPUT_KIND, RUN_DIRS = (BUNDLE['input_kind'], BUNDLE['run_dirs'])"])
    asserted = _asserted(tree)
    assert "INPUT_KIND in ('logmel', 'waveform')" in asserted
    assert "len(set(RUN_DIRS)) == len(RUN_DIRS) == 3" in asserted
    # no duplicated selection rule: the notebook names no candidate and reads no selection field itself
    for t in _trees():
        for value in _strings(t):
            assert value not in CANDIDATES and value not in ("selected", "selection_report.json"), value


def test_search_and_risk_versions_are_bound_to_the_committed_reports():
    tree = _cell_tree("bundle_checkpoint")
    asserted = _asserted(tree)
    assert "SEARCH_REPORT['version'] == SEARCH_VERSION" in asserted
    assert "RISK_REPORT['version'] == RISK_VERSION" in asserted
    assert [ast.unparse(v) for v in _assigned(tree, "RISK_REPORT")] == [
        "json.loads((RISK_EVIDENCE / 'run_report.json').read_text(encoding='utf-8'))"]
    assert "sha256(PROFILE) == F3['config_sha256']" in asserted
    printed = [ast.unparse(c) for c in _calls(tree, "print")]
    assert any("SEARCH_REPORT['version']" in p and "RISK_REPORT['version']" in p for p in printed), printed


# --- Reproduction-gate caches (D46 (4)), log-Mel winners only ---------------------------------------

def test_only_val_a_and_val_b_are_cached_and_cache_b_holds_val_a_only():
    strings = [s for tree in _trees() for s in _strings(tree)]
    for forbidden in ("train", "test"):
        assert forbidden not in strings, forbidden
    tree = _cell_tree("CACHE_PLAN = ")
    (plan,) = _assigned(tree, "CACHE_PLAN")
    assert ast.unparse(plan) == "((CACHE_A, ('val-A', 'val-B')), (CACHE_B, ('val-A',)))"
    src = _code()[_cell_index("CACHE_PLAN = ")]
    assert "CACHE_A, CACHE_B = Path('/content/cache_a'), Path('/content/cache_b')" in src
    assert "'--out', str(cache / 'log_mel' / split)" in src
    assert "src.features.mfcc_cache" not in src


def test_both_cache_branches_are_present_and_only_log_mel_builds():
    tree = _cell_tree("CACHE_PLAN = ")
    (command,) = [n for n in ast.walk(tree) if isinstance(n, ast.List) and "src.features.logmel_cache" in _strings(n)]
    assert _guarding_if(tree, command) == "INPUT_KIND == 'logmel'"
    (branch,) = [n for n in tree.body if isinstance(n, ast.If)]
    assert ast.unparse(branch.test) == "INPUT_KIND == 'logmel'"
    # aasist: nothing built, nothing hashed (the manifest records null)
    assert [ast.unparse(s) for s in branch.orelse] == [
        "print(f'{WINNER} takes the F3 waveform: no cache is built and no --cache-* argument is passed')"]
    assert "CACHE_HASHES_A = CACHE_HASHES_B = None" in [ast.unparse(s) for s in tree.body]


def test_cache_builds_are_compared_byte_for_byte_and_bound_to_f3_and_commit():
    asserted = _asserted(_cell_tree("CACHE_PLAN = "))
    for check in ("CACHE_HASHES_B == CACHE_A_VAL_A", "built == planned",
                  "meta['profile_config_sha256'] == F3['config_sha256']",
                  "meta['profile_fingerprint'] == F3['fingerprint']", "meta['code_commit'] == COMMIT"):
        assert check in asserted, check


def test_no_mfcc_no_d58_reference_and_no_baseline_in_the_code():
    # The v2 model set has no baseline (scored in the v1 claim), so no MFCC and no D58 reference.
    for src in _code():
        assert not re.search(r"mfcc|baseline", src, re.IGNORECASE), src[:80]
    for name in ("--mfcc-reference", "mfcc_reference", "MFCC_REFERENCE", "baseline_version", "--baseline",
                 "--stage7-evidence"):
        assert name not in "\n".join(_source(c) for c in _cells()), name


# --- Artifacts --------------------------------------------------------------------------------------

def test_artifacts_are_copied_outside_the_clone_and_never_loaded_here():
    code = "\n".join(_code())
    for loader in ("load_v2_checkpoint", "load_checkpoint", "safetensors", "RiskPolicy", "load_selection_set",
                   "np.load"):
        assert loader not in code, loader
    tree = _cell_tree("ARTIFACTS = ")
    expected = {
        "ARTIFACTS": "Path('/content/stage11_v2_artifacts')",
        "CHECKPOINTS_LOCAL": "ARTIFACTS / 'runs'",
        "S_LOCAL": "ARTIFACTS / 'selection_set_s'",
        "RISK_LOCAL": "ARTIFACTS / 'risk' / RISK_VERSION",
        "source": "SEARCH_OUT / 'runs' / run_dir / 'checkpoint'",
    }
    for name, value in expected.items():
        assert [ast.unparse(v) for v in _assigned(tree, name)] == [value], name
    text = ast.unparse(tree)
    copies = [n for n in tree.body if isinstance(n, ast.For) and "copytree" in ast.unparse(n)]
    (loop,) = copies
    assert ast.unparse(loop.iter) == "RUN_DIRS"  # the winner's three seeds only
    assert "shutil.copytree(source, CHECKPOINTS_LOCAL / run_dir / 'checkpoint')" in ast.unparse(loop)
    assert "shutil.copytree(SELECTION_SET_DRIVE, S_LOCAL)" in text
    assert "shutil.copy2(RISK_DRIVE / 'val_b_scores.npz', RISK_LOCAL / 'val_b_scores.npz')" in text
    assert "risk_policy.json" not in _strings(tree)  # the runner reads the committed policy
    asserted = _asserted(tree)
    assert "sorted((p.name for p in CHECKPOINTS_LOCAL.iterdir())) == sorted(RUN_DIRS)" in asserted
    assert "not ARTIFACTS.resolve().is_relative_to(REPO.resolve())" in asserted
    assert not "/content/stage11_v2_artifacts/".startswith(CLONE + "/")


def test_no_output_inside_the_clone():
    allowed_content = ("/content/drive", CLONE, "/content/data", "/content/cache_a", "/content/cache_b",
                       "/content/stage11_v2_artifacts", "/content/stage11_v2_rehearsal",
                       "/content/stage11_v2_report", "/content/pytest_stage11_v2.xml")
    for tree in _trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and ast.unparse(node.left) == "REPO":
                assert ast.literal_eval(node.right).startswith(("docs/", "configs/", "data/manifests/")), ast.unparse(node)
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("/content"):
                assert node.value.startswith(allowed_content), node.value
    assert [ast.unparse(v) for v in _assigned(_cell_tree("'git', 'clone'"), "REPO")] == [f"Path('{CLONE}')"]


# --- Drive layout -----------------------------------------------------------------------------------

def test_drive_layout_constants():
    tree = _cell_tree("STAGE11_V2 = ")
    expected = {
        "REGISTRY": "DRIVE / 'processed' / 'metadata' / 'stage11' / 'run_registry.jsonl'",
        "STAGE11_V2": "DRIVE / 'processed' / 'metadata' / 'stage11_v2'",
        "RUN_DEST": "STAGE11_V2 / 'run'",
        "DRY_RUN_DEST": "STAGE11_V2 / 'dry_run' / DRY_RUN_STAMP",
        "SEARCH_OUT": "DRIVE / 'checkpoints' / 'v2_search' / SEARCH_VERSION",
        "SELECTION_SET_DRIVE": "DRIVE / 'processed' / 'v2' / 'selection_set_s'",
        "RISK_DRIVE": "DRIVE / 'checkpoints' / 'risk' / RISK_VERSION",
    }
    for name, value in expected.items():
        assert [ast.unparse(v) for v in _assigned(tree, name)] == [value], name


def test_output_root_is_stage11_v2_and_v1_stage11_run_is_never_written():
    allowed = {
        ("'processed'", "'metadata'", "'stage11'", "'run_registry.jsonl'"),  # the shared registry, read here
        ("'processed'", "'metadata'", "'stage11_v2'"),  # every output
        ("'checkpoints'", "'v2_search'", "SEARCH_VERSION"),  # read
        ("'processed'", "'v2'", "'selection_set_s'"),  # read
        ("'checkpoints'", "'risk'", "RISK_VERSION"),  # read
        ("'raw'",), ("'raw'", "'for_2sec_ff8c82c7.tar'"), ("'raw'", "'for_2sec_REVISION.txt'"),  # read
        ("path",),  # the re-verification after the remount (manifest paths under RUN_DEST / DRY_RUN_DEST)
    }
    chains = {tuple(c) for c in _drive_chains()}
    assert chains <= allowed, chains - allowed
    strings = [s for tree in _trees() for s in _strings(tree)]
    assert strings.count("stage11") == 1  # only in the REGISTRY path
    for value in strings:
        if "stage11" in value and value != "stage11":
            assert "stage11_v2" in value, value
    for v1 in ("cnn-v1", "risk-v1", "baseline-v1", "src.evaluation.stage11_run", "src.evaluation.stage11_report"):
        assert v1 not in strings, v1
    # every Drive write goes under RUN_DEST or DRY_RUN_DEST
    writes = []
    for tree in _trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in (
                    "mkdir", "write_text", "write_bytes"):
                writes.append(ast.unparse(node.func.value))
    assert sorted(set(writes)) == sorted({
        "cache", "RISK_LOCAL", "DATA_ROOT", "DRY_RUN_DEST", "DRY_RUN_MANIFEST_PATH", "RUN_DEST / 'rehearsal'",
        "MANIFEST_PATH"}), writes
    assert [ast.unparse(v) for v in _assigned(_cell_tree("EXPORT_MANIFEST = "), "MANIFEST_PATH")] == [
        "RUN_DEST / 'stage11_v2_export_manifest.json'"]
    assert [ast.unparse(v) for v in _assigned(_cell_tree("DRY_RUN_MANIFEST = "), "DRY_RUN_MANIFEST_PATH")] == [
        "DRY_RUN_DEST / 'stage11_v2_dry_run_manifest.json'"]


def test_existing_outputs_are_refused_at_mount_before_the_rehearsal_and_before_the_claim():
    code = "\n".join(_code())
    calls = [m.start() for m in re.finditer(r"^ *refuse_existing_outputs\(\)$", code, re.MULTILINE)]
    clone, rehearse, claim = code.find("'git', 'clone'"), code.find("'rehearse'"), code.find("'claim', '--root'")
    assert len(calls) == 3 and calls[0] < clone < calls[1] < rehearse < calls[2] < claim, calls


# --- Registry (D46 (6), D48 (7)) --------------------------------------------------------------------

def test_registry_pre_check_is_read_only_against_the_committed_copy_before_the_rehearsal():
    code = _code()
    pre = _cell_index("REGISTRY_PRE_CHECK = {")
    assert pre < _cell_index("'rehearse'") < _cell_index("'claim', '--root'")
    tree = _trees()[pre]
    assert [ast.unparse(v) for v in _assigned(tree, "COMMITTED_REGISTRY")] == [
        "REPO / 'docs/audit_evidence/run_registry.jsonl'"]
    (check,) = _assigned(tree, "REGISTRY_PRE_CHECK")
    assert ast.unparse(check) == (
        "{'drive': {'sha256': sha256(REGISTRY), 'lines': len(REGISTRY.read_bytes().splitlines())}, "
        "'committed': {'sha256': sha256(COMMITTED_REGISTRY), "
        "'lines': len(COMMITTED_REGISTRY.read_bytes().splitlines())}}")
    (branch,) = [n for n in tree.body if isinstance(n, ast.If)]
    assert ast.unparse(branch.test) == "REGISTRY_PRE_CHECK['drive'] != REGISTRY_PRE_CHECK['committed']"
    assert len(branch.body) == 1 and isinstance(branch.body[0], ast.Raise)
    assert "REGISTRY_PRE_CHECK" in ast.unparse(branch.body[0])  # both hashes and line counts in the message
    uses = {ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Call)
            and any(isinstance(a, ast.Name) and a.id == "REGISTRY" for a in ast.walk(n.func if isinstance(
                n.func, ast.Attribute) else n))}
    assert uses <= {"sha256(REGISTRY)", "REGISTRY.read_bytes()", "REGISTRY.read_bytes().splitlines()",
                    "len(REGISTRY.read_bytes().splitlines())"}, uses
    assert "'registry_pre_check': REGISTRY_PRE_CHECK" in code[_cell_index("def base_manifest(")]
    claim = code[_cell_index("'claim', '--root'")]
    at = claim.find("assert sha256(REGISTRY) == REGISTRY_PRE_CHECK['drive']['sha256']")
    assert -1 < at < claim.find("'claim', '--root'")


def test_registry_is_on_drive_and_never_rewritten_by_the_notebook():
    writers = {"unlink", "rmtree", "rename", "replace", "write_text", "write_bytes", "touch", "move", "remove",
               "copy", "copy2", "copyfile", "copytree", "truncate", "mkdir"}
    for tree in _trees():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            touched = [n for n in ast.walk(node) if isinstance(n, ast.Name)
                       and n.id in ("REGISTRY", "COMMITTED_REGISTRY")]
            if name in writers:
                assert not touched, ast.unparse(node)
            if name == "open" and touched:
                modes = [ast.literal_eval(a) for a in node.args[1:2]] + [
                    ast.literal_eval(k.value) for k in node.keywords if k.arg == "mode"]
                assert all(not set(m) & set("wax+") for m in modes), ast.unparse(node)


def test_registry_is_not_used_before_the_pre_check():
    code = _code()
    pre = _cell_index("REGISTRY_PRE_CHECK = {")
    for i, src in enumerate(code[:pre]):
        uses = [line for line in src.splitlines() if "REGISTRY" in line]
        assert uses in ([], ["REGISTRY = DRIVE / 'processed' / 'metadata' / 'stage11' / 'run_registry.jsonl'"]), (i, uses)


# --- Runner calls -----------------------------------------------------------------------------------

def test_each_runner_subcommand_is_invoked_exactly_once_outside_loops_branches_and_functions():
    for subcommand in ("rehearse", "claim", "score"):
        assert len(_runner_calls(subcommand)) == 1, subcommand
    assert "\n".join(_code()).count(f"'{REPORT}'") == 1
    for tree in _trees():
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While, ast.AsyncFor, ast.Try, ast.If, ast.FunctionDef, ast.Lambda)):
                inner = _strings(node)
                assert not {"rehearse", "claim", "score", REPORT} & set(inner), ast.unparse(node)[:80]


def test_shared_inputs_are_the_runner_arguments():
    tree = _cell_tree("INPUTS = [")
    (inputs,) = _assigned(tree, "INPUTS")
    words = [ast.unparse(e) for e in inputs.elts]
    assert words[-1] == "*CACHE_ARGS"
    assert dict(zip(words[:-1:2], words[1:-1:2])) == {
        "'--checkpoints'": "str(CHECKPOINTS_LOCAL)",
        "'--selection-set'": "str(S_LOCAL)",
        "'--profile'": "str(PROFILE)",
        "'--config'": "str(REPO / 'configs/v2_candidates.yaml')",
        "'--manifest'": "str(MANIFEST)",
    }
    assert len(words) == 11
    assert "RUNNER = 'src.evaluation.stage11_v2_run'" in _code()[_cell_index("INPUTS = [")]
    bundle = _cell_tree("bundle_checkpoint")
    assert [ast.unparse(v) for v in _assigned(bundle, "MANIFEST")] == ["REPO / 'data/manifests/for2sec_frozen_manifest.csv'"]
    assert [ast.unparse(v) for v in _assigned(bundle, "PROFILE")] == ["REPO / 'configs/preprocessing.yaml'"]


def test_cache_arguments_only_for_a_log_mel_winner_and_none_for_aasist():
    tree = _cell_tree("INPUTS = [")
    (branch,) = [n for n in tree.body if isinstance(n, ast.If) and ast.unparse(n.test) == "INPUT_KIND == 'logmel'"]
    assert [ast.unparse(s) for s in branch.body] == ["CACHE_ARGS = ['--cache-a', str(CACHE_A), '--cache-b', str(CACHE_B)]"]
    assert [ast.unparse(s) for s in branch.orelse] == ["CACHE_ARGS = []"]
    strings = [s for t in _trees() for s in _strings(t)]
    assert strings.count("--cache-a") == strings.count("--cache-b") == 1  # only in the log-Mel branch
    for subcommand in ("rehearse", "claim", "score"):
        (call,) = _runner_calls(subcommand)
        assert not {"--cache-a", "--cache-b"} & set(_strings(call)), subcommand  # only through INPUTS


def test_runner_arguments_per_subcommand():
    (rehearse,), (claim,), (score,) = (_runner_calls(s) for s in ("rehearse", "claim", "score"))
    assert _call_args(rehearse) == {"'--root'": "str(AUDIO_ROOT)", "*INPUTS": "",
                                    "'--risk-scores'": "str(RISK_LOCAL / 'val_b_scores.npz')",
                                    "'--out'": "str(REHEARSAL_OUT)"}
    assert _call_args(claim) == {"'--root'": "str(AUDIO_ROOT)", "*INPUTS": "", "'--registry'": "str(REGISTRY)",
                                 "'--results-ref'": "RESULTS_REF", "'--out'": "str(CLAIM_OUT)"}
    assert _call_args(score) == {"'--root'": "str(AUDIO_ROOT)", "*INPUTS": "", "'--registry'": "str(REGISTRY)",
                                 "'--run-id'": "RUN_ID", "'--out'": "str(CLAIM_OUT)"}
    for call in (rehearse, claim, score):
        assert "--split" not in _strings(call)
        assert "--device" not in _strings(call)
    code = _code()
    assert "REPORT_OUT = Path('/content/stage11_v2_report')" in code[_cell_index(REPORT)]  # the report stays local


def test_rehearsal_writes_to_drive_in_dry_run_and_locally_in_official_mode():
    tree = _cell_tree("'rehearse'")
    (branch,) = [n for n in tree.body if isinstance(n, ast.If) and ast.unparse(n.test) == "MODE == 'dry_run'"]
    assert [ast.unparse(n) for n in branch.body] == [
        "refuse_existing_outputs()",
        "DRY_RUN_DEST.mkdir(parents=True, exist_ok=False)",
        "REHEARSAL_OUT = DRY_RUN_DEST / 'rehearsal'",
    ]
    assert [ast.unparse(n) for n in branch.orelse] == ["REHEARSAL_OUT = Path('/content/stage11_v2_rehearsal')"]
    assert tree.body.index(branch) < [ast.unparse(n).startswith("stream(") for n in tree.body].index(True)


def test_runner_output_is_on_drive_under_run_dest():
    code = _code()
    claim = code[_cell_index("'claim', '--root'")]
    lines = claim.splitlines()
    # The runner creates run/runner/ (and run/) by an exclusive mkdir only after the claim, so a
    # pre-claim failure leaves no folder that would block a permitted rerun.
    assert lines[1:5] == ["CLAIM_OUT = RUN_DEST / 'runner'", "refuse_existing_outputs()",
                          "assert sha256(REGISTRY) == REGISTRY_PRE_CHECK['drive']['sha256'], "
                          "'the Drive registry changed since the pre-check'", "assert_clean()"]
    assert "'claim', '--root'" in lines[5]
    (hashes,) = _assigned(ast.parse(claim), "CLAIM_HASHES")
    assert ast.unparse(hashes) == "{name: sha256(CLAIM_OUT / name) for name in CLAIM_FILES}"
    assert claim.find("'claim', '--root'") < claim.find("CLAIM_HASHES = ")
    assert len([s for s in code if "CLAIM_OUT = " in s]) == 1
    mkdirs = [n for n in ast.walk(ast.parse(claim)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and n.func.attr in ("mkdir", "makedirs")]
    assert mkdirs == [], "the claim cell creates no folder; the runner does, after the claim"


def test_no_claim_copy_and_runner_files_are_hashed_in_place():
    code = "\n".join(_code())
    assert "'claim'" not in code.replace("'claim', '--root'", "")  # no run/claim/ folder
    export = _code()[_cell_index("EXPORT_MANIFEST = ")]
    assert "files = [drive_entry(CLAIM_OUT / rel) for rel in runner_files]" in export
    assert "export_file(CLAIM_OUT" not in export
    assert "assert {name: sha256(CLAIM_OUT / name) for name in CLAIM_FILES} == CLAIM_HASHES" in export
    assert "assert sorted(p.name for p in RUN_DEST.iterdir()) == ['runner']" in export


def test_report_arguments_and_called_once():
    (call,) = [n for tree in _trees() for n in ast.walk(tree) if isinstance(n, ast.List) and REPORT in _strings(n)]
    words = [ast.unparse(e) for e in call.elts]
    assert words[:3] == ["sys.executable", "'-m'", f"'{REPORT}'"]
    assert len(words) == 15
    assert dict(zip(words[3::2], words[4::2])) == {
        "'--run-dir'": "str(CLAIM_OUT)",
        "'--registry'": "str(REGISTRY)",
        "'--manifest'": "str(MANIFEST)",
        "'--search-evidence'": "str(SEARCH_EVIDENCE)",
        "'--risk-evidence'": "str(RISK_EVIDENCE)",
        "'--out'": "str(REPORT_OUT)",
    }
    src = _code()[_cell_index(REPORT)]
    assert "if REPORT_OUT.exists():\n    raise FileExistsError" in src


def test_assert_clean_brackets_every_runner_and_report_call():
    for marker in ("'rehearse'", "'claim', '--root'", "'score', '--root'", REPORT):
        src = _code()[_cell_index(marker)]
        at = src.find(marker)
        assert -1 < src.rfind("assert_clean()", 0, at) < at < src.find("assert_clean()", at), marker


def test_head_and_clean_worktree_before_and_after_every_long_cell():
    long_cells = 0
    for src in _code():
        tree = ast.parse(src)
        long_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and (
            ast.unparse(n.func) in ("stream", "probe", "archive.extractall", "shutil.copytree")
            or (ast.unparse(n.func) == "subprocess.run" and ("'pip', 'install'" in ast.unparse(n)
                                                           or "src.features.logmel_cache" in ast.unparse(n))))]
        if not long_calls:
            continue
        long_cells += 1
        clean = [n.lineno for n in _calls(tree, "assert_clean")]
        assert clean and min(clean) < min(c.lineno for c in long_calls), src[:80]
        assert max(clean) > max(c.end_lineno for c in long_calls), src[:80]
    # install, winner probe, archive, pytest, caches, copies, rehearse, claim, score, report
    assert long_cells == 10


# --- Rehearsal and dry-run stop ---------------------------------------------------------------------

def test_rehearsal_must_pass_and_only_passed_is_read():
    tree = _cell_tree("'rehearse'")
    assert "REHEARSAL_PASSED is True" in _asserted(tree)
    (passed,) = _assigned(tree, "REHEARSAL_PASSED")
    assert ast.unparse(passed) == (
        "json.loads((REHEARSAL_OUT / 'rehearsal_report.json').read_text(encoding='utf-8'))['passed']")


def test_dry_run_export_runs_only_in_dry_run_mode_and_ends_with_the_stop_line():
    tree = _cell_tree("DRY_RUN_MANIFEST = ")
    (branch,) = [n for n in tree.body if isinstance(n, ast.If)]
    assert ast.unparse(branch.test) == "MODE == 'dry_run'" and not branch.orelse
    assert ast.unparse(tree.body[-1]) == ast.unparse(branch)  # the branch is the cell's last statement
    assert ast.unparse(branch.body[-1]) == "print('Dry run complete. Stop here.')"
    body = "\n".join(ast.unparse(n) for n in branch.body)
    for step in ("assert REHEARSAL_OUT == DRY_RUN_DEST / 'rehearsal'",
                 "assert stored == ['rehearsal/pip_freeze.txt', 'rehearsal/rehearsal_report.json']",
                 "files = [drive_entry(DRY_RUN_DEST / rel) for rel in stored]",
                 "drive.flush_and_unmount()", "drive.mount('/content/drive', force_remount=True)",
                 "verify_after_remount(DRY_RUN_DEST, checked)"):
        assert step in body, step
    assert "export_file(" not in body and "mkdir(" not in body  # hashed in place, nothing copied
    assert body.find("flush_and_unmount") < body.find("force_remount") < body.find("verify_after_remount")


def test_claim_cell_and_every_later_code_cell_start_with_the_official_assert():
    code = _code()
    claim = _cell_index("'claim', '--root'")
    for i, src in enumerate(code):
        if i >= claim:
            assert src.splitlines()[0] == OFFICIAL, i
        else:
            assert OFFICIAL not in src, i
            assert "'claim'" not in src and "'score'" not in src, i


def test_rerun_rule_is_in_the_markdown_right_before_the_claim():
    cells = _cells()
    (i,) = [k for k, c in enumerate(cells) if c["cell_type"] == "code" and "'claim', '--root'" in _source(c)]
    assert cells[i - 1]["cell_type"] == "markdown"
    assert RERUN_RULE in _source(cells[i - 1])


# --- Claim, durability, score ----------------------------------------------------------------------

def test_run_id_comes_from_claim_json_never_a_constant():
    tree = _cell_tree("RUN_ID = ")
    (run_id,) = _assigned(tree, "RUN_ID")
    assert ast.unparse(run_id) == "CLAIM['run_id']"
    (claim,) = _assigned(tree, "CLAIM")
    assert ast.unparse(claim) == "json.loads((CLAIM_OUT / 'claim.json').read_text(encoding='utf-8'))"
    assignments = [n for t in _trees() for n in ast.walk(t) if isinstance(n, ast.Assign)
                   and any(isinstance(x, ast.Name) and x.id in ("RUN_ID", "CLAIM") for x in n.targets)]
    assert len(assignments) == 2
    (score,) = _runner_calls("score")
    words = [ast.unparse(e) for e in score.elts]
    after = score.elts[words.index("'--run-id'") + 1]
    assert isinstance(after, ast.Name) and after.id == "RUN_ID"


def test_claim_is_verified_on_drive_after_flush_and_remount_with_the_v2_model_set():
    src = _code()[_cell_index("RUN_ID = ")]
    lines = src.splitlines()
    assert lines[1:3] == ["drive.flush_and_unmount()", "drive.mount('/content/drive', force_remount=True)"]
    asserted = _asserted(ast.parse(src))
    for check in ("sha256(REGISTRY) == CLAIM['registry_sha256']",
                  "LAST['run_id'] == RUN_ID and LAST['status'] == 'started'",
                  "LAST['code_commit'] == COMMIT and LAST['split'] == CLAIM['split']",
                  "LAST['model_set'] == CLAIM['model_ids']",
                  "CLAIM['architecture'] == WINNER and CLAIM['bundle_seed'] == BUNDLE_SEED",
                  "len(CLAIM['model_ids']) == 3",
                  "REGISTRY_BYTES.startswith(COMMITTED_REGISTRY.read_bytes())",
                  "len(REGISTRY_BYTES.splitlines()) == REGISTRY_PRE_CHECK['drive']['lines'] + 1",
                  "CLAIM['code_commit'] == COMMIT", "CLAIM['results_ref'] == RESULTS_REF",
                  "{name: sha256(CLAIM_OUT / name) for name in CLAIM_FILES} == CLAIM_HASHES"):
        assert check in asserted, check


def test_score_checks_identity_only():
    tree = _cell_tree("'score', '--root'")
    asserted = _asserted(tree)
    for check in ("RUN_REPORT['code_commit'] == COMMIT", "RUN_REPORT['run_id'] == RUN_ID",
                  "DONE['run_id'] == RUN_ID and DONE['status'] == 'completed'",
                  "DONE['results_ref'] == RESULTS_REF"):
        assert check in asserted, check
    keys = {n.slice.value for n in ast.walk(tree) if isinstance(n, ast.Subscript)
            and isinstance(n.value, ast.Name) and n.value.id == "RUN_REPORT" and isinstance(n.slice, ast.Constant)}
    assert keys == {"code_commit", "run_id"}, keys


# --- Export and results view ------------------------------------------------------------------------

def test_official_export_files_six_clip_scores_and_manifest():
    tree = _cell_tree("EXPORT_MANIFEST = ")
    src = _code()[_cell_index("EXPORT_MANIFEST = ")]
    (score_files,) = _assigned(tree, "SCORE_FILES")
    assert ast.literal_eval(score_files) == ("scoring_started.json", "environment_score.json", "features.json",
                                             "run_report.json")
    (claim_files,) = _assigned(_cell_tree("'claim', '--root'"), "CLAIM_FILES")
    assert sorted([*ast.literal_eval(claim_files), *ast.literal_eval(score_files)]) == sorted(RUNNER_FILES)
    assert "assert len(clip_files) == 6, clip_files" in src  # 3 models x (csv + json)
    assert ("assert clip_files == sorted(f'{model_id}.{ext}' for model_id in CLAIM['model_ids'] "
            "for ext in ('csv', 'json')), clip_files") in src
    for name in ("'metrics_report.json', 'metrics_report.md'", "'rehearsal_report.json', 'pip_freeze.txt'"):
        assert name in src, name
    assert "(RUN_DEST / 'rehearsal').mkdir(exist_ok=False)" in src
    assert "assert not dest.exists()" in _code()[_cell_index("def export_file(")]
    (manifest,) = _assigned(tree, "EXPORT_MANIFEST")
    assert {k.value for k in manifest.keys if k is not None} == {"run_id", "results_ref", "registry", "files"}
    assert [ast.unparse(v) for k, v in zip(manifest.keys, manifest.values) if k is None] == ["base_manifest()"]
    (base,) = [n for n in _cell_tree("def base_manifest(").body if isinstance(n, ast.FunctionDef)
               and n.name == "base_manifest"]
    (ret,) = [n.value for n in ast.walk(base) if isinstance(n, ast.Return)]
    assert {k.value for k in ret.keys} == {
        "commit", "mode", "search_version", "risk_version", "winner", "input_kind", "bundle_seed", "python",
        "numpy", "torch", "require_versions", "gpu_name", "created_utc", "for_revision",
        "for_archive_sha256_observed", "pytest_counts", "cache_sha256", "f3", "registry_pre_check",
        "rehearsal_passed"}
    assert "'registry': {'path': REGISTRY.relative_to(DRIVE).as_posix(), 'sha256': REGISTRY_SHA256," in src
    assert "REGISTRY_LINES = len(REGISTRY.read_text(encoding='utf-8').splitlines())" in src


def test_official_export_is_rehashed_after_flush_and_remount():
    src = _code()[_cell_index("EXPORT_MANIFEST = ")]
    flush = src.find("drive.flush_and_unmount()")
    remount = src.find("drive.mount('/content/drive', force_remount=True)")
    verify = src.find("verify_after_remount(RUN_DEST, checked)")
    registry = src.find("assert sha256(REGISTRY) == REGISTRY_SHA256")
    assert -1 < src.find("MANIFEST_PATH.write_text(") < flush < remount < verify < registry
    helper = _code()[_cell_index("def verify_after_remount(")]
    for check in ("sha256(DRIVE / path)", "missing", "extra"):
        assert check in helper, check


def test_no_metric_is_read_before_the_results_view():
    results = _results_cell()
    readers = {"read_text", "read_bytes", "open", "loads", "load", "Markdown", "display"}
    for i, tree in enumerate(_trees()):
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                if name in readers and any("metrics_report" in s or "clip_scores" in s for s in _strings(node)):
                    assert i == results, (i, ast.unparse(node))
    assert "display(Markdown((REPORT_OUT / 'metrics_report.md').read_text(encoding='utf-8')))" in _code()[results]
    assert results == len(_code()) - 1


def test_steps_run_in_order():
    code = "\n".join(_code())
    markers = ["drive.mount(", "'git', 'clone'", "'pip', 'install'", "require_versions()", "bundle_checkpoint(",
               "archive.extractall", "'pytest'", "CACHE_PLAN = ", "shutil.copytree(source",
               "REGISTRY_PRE_CHECK = {", "'rehearse'", "Dry run complete. Stop here.", "'claim', '--root'",
               "drive.flush_and_unmount()", "force_remount=True", "RUN_ID = CLAIM['run_id']", "'score', '--root'",
               REPORT, "EXPORT_MANIFEST = {", "drive.flush_and_unmount()", "force_remount=True",
               "verify_after_remount(RUN_DEST, checked)", "display(Markdown("]
    position = 0
    for marker in markers:
        found = code.find(marker, position)
        assert found >= position, (marker, position)
        position = found + len(marker)


def test_final_markdown_names_the_committed_files_and_keeps_clip_scores_on_drive():
    last = _cells()[-1]
    assert last["cell_type"] == "markdown"
    text = _source(last)
    for name in ("docs/audit_evidence/run_registry.jsonl", "docs/audit_evidence/stage11_v2_run/", *RUNNER_FILES,
                 "rehearsal_report.json", "metrics_report.json", "stage11_v2_export_manifest.json",
                 "docs/results/stage11_v2_for_test.md", "stage11_v2/run/runner/clip_scores/",
                 "SHA-256 and line count", "src.bundle.f5"):
        assert name in text, name
    assert "stay on Drive only" in text
    assert text.index("What is committed") < text.index("F5")
    assert "only after this results PR is merged" in text
