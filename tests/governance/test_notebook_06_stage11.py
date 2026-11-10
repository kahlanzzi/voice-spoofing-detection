"""Static checks for the Stage 11 Colab notebook (notebooks/06_stage11_for_test.ipynb; D31, D40, D46, D57).

The notebook is committed with outputs cleared and is only run in Colab, so these tests read its JSON
and parse each code cell with ast; they never execute it. They pin the D46 order and its guards:
the T4 runtime with CPU scoring, the environment gate, only val-A and val-B cached, the rehearsal
before the claim, the claim cell and every later cell refusing the dry-run mode, the Drive flush and
remount between claim and score, run_id read from claim.json, the registry never rewritten, one
claim and one score, no metric read before the results view, and the FoR held-out guard.

The helpers mirror tests/governance/test_notebook_05_stage10.py: there they are private and bound to
notebook 05's path, so they are copied here rather than imported.
"""
import ast
import json
import re
from pathlib import Path

from tests.governance.test_for_test_access import file_hits

REPO_ROOT = Path(__file__).resolve().parents[2]
REL = "notebooks/06_stage11_for_test.ipynb"
NOTEBOOK = REPO_ROOT / REL
CLONE = "/content/voice-spoofing-detection"  # where the notebook clones the repo in Colab
RUNNER = "src.evaluation.stage11_run"
REPORT = "src.evaluation.stage11_report"
OFFICIAL = "assert MODE == 'official', 'dry run: stop before the claim'"
RERUN_RULE = (
    "Rerun rule (D46 (8), pre-declared): before the claim, an infrastructure failure (Drive, clone, pip, "
    "archive) may be rerun from the same COMMIT; an environment-gate, reproduction-gate or rehearsal "
    "failure stops Stage 11 until a logged decision. After the claim, there is no rerun ever without a "
    "new decision, including a crash before any score exists; everything written stays for team review. "
    "Nothing in this notebook reads or reacts to the test results before they are exported."
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


def _results_cell() -> int:
    return _cell_index("display(Markdown(")


# --- Notebook shape ---------------------------------------------------------------------------------

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


def test_parameters_are_the_placeholder_commit_dry_run_and_fixed_versions():
    params = _trees()[0]
    values = {t.id: ast.literal_eval(n.value) for n in params.body
              if isinstance(n, ast.Assign) for t in n.targets}
    assert values == {
        "REPO_URL": "https://github.com/kahlanzzi/voice-spoofing-detection.git",
        "COMMIT": "REPLACE_WITH_APPROVED_COMMIT",
        "MODE": "dry_run",
        "CNN_VERSION": "cnn-v1",
        "BASELINE_VERSION": "baseline-v1",
        "RISK_VERSION": "risk-v1",
        "DRIVE_ROOT": "/content/drive/MyDrive/voice-spoofing-detection",
        "RESULTS_REF": "docs/results/stage11_for_test.md",
    }
    assert _asserted(params) == ["MODE in ('dry_run', 'official')"]
    assert "len(COMMIT) == 40" in _asserted(_cell_tree("'git', 'clone'"))


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


# --- Environment gate (D46 (1), (2)) ----------------------------------------------------------------

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
    assert "ENV['python'].startswith('3.13.')" in asserted  # D15
    assert "torch.__version__.startswith('2.11.0')" in asserted
    assert "'T4' in GPU_NAME" in asserted  # D46 (1)
    (gpu,) = _assigned(tree, "GPU_NAME")
    (command,) = [n for n in ast.walk(gpu) if isinstance(n, ast.List)]
    assert ast.literal_eval(command) == ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    metadata = _notebook()["metadata"]
    assert metadata["accelerator"] == "GPU" and metadata["colab"]["gpuType"] == "T4"


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


# --- Reproduction-gate caches (D46 (4)) -------------------------------------------------------------

def test_only_val_a_and_val_b_are_cached_and_cache_b_holds_val_a_only():
    strings = [s for tree in _trees() for s in _strings(tree)]
    for forbidden in ("train", "test"):
        assert forbidden not in strings, forbidden
    tree = _cell_tree("CACHE_PLAN = ")
    (plan,) = _assigned(tree, "CACHE_PLAN")
    assert ast.unparse(plan) == "((CACHE_A, ('val-A', 'val-B')), (CACHE_B, ('val-A',)))"
    (modules,) = _assigned(tree, "FEATURE_MODULES")
    assert ast.literal_eval(modules) == {"log_mel": "src.features.logmel_cache", "mfcc": "src.features.mfcc_cache"}
    src = _code()[_cell_index("CACHE_PLAN = ")]
    assert "CACHE_A, CACHE_B = Path('/content/cache_a'), Path('/content/cache_b')" in src
    assert "'--out', str(cache / feature / split)" in src


def test_cache_builds_are_compared_byte_for_byte_and_bound_to_f3_and_commit():
    asserted = _asserted(_cell_tree("CACHE_PLAN = "))
    for check in ("CACHE_HASHES_B == CACHE_A_VAL_A", "built == planned",
                  "meta['profile_config_sha256'] == F3['config_sha256']",
                  "meta['profile_fingerprint'] == F3['fingerprint']", "meta['code_commit'] == COMMIT",
                  "sha256(PROFILE) == F3['config_sha256']"):
        assert check in asserted, check


# --- Artifacts --------------------------------------------------------------------------------------

def test_frozen_artifacts_are_copied_outside_the_clone_and_never_loaded_here():
    code = "\n".join(_code())
    assert "load_checkpoint" not in code and "safetensors" not in code and "RiskPolicy" not in code
    tree = _cell_tree("ARTIFACTS = ")
    expected = {
        "ARTIFACTS": "Path('/content/stage11_artifacts')",
        "CNN_DRIVE": "DRIVE / 'checkpoints' / 'cnn' / CNN_VERSION",
        "BASELINE_DRIVE": "DRIVE / 'checkpoints' / 'baseline' / BASELINE_VERSION",
        "RISK_DRIVE": "DRIVE / 'checkpoints' / 'risk' / RISK_VERSION",
    }
    for name, value in expected.items():
        (node,) = _assigned(tree, name)
        assert ast.unparse(node) == value, name
    src = _code()[_cell_index("ARTIFACTS = ")]
    assert "for seed in (0, 1, 2):" in src
    assert "for name in ('risk_policy.json', 'val_b_scores.npz'):" in src
    assert not "/content/stage11_artifacts/".startswith(CLONE + "/")


# --- Runner calls -----------------------------------------------------------------------------------

def test_each_runner_subcommand_is_invoked_exactly_once_outside_loops_and_functions():
    for subcommand in ("rehearse", "claim", "score"):
        assert len(_runner_calls(subcommand)) == 1, subcommand
    assert "\n".join(_code()).count(REPORT) == 1
    for tree in _trees():
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While, ast.AsyncFor, ast.Try, ast.FunctionDef, ast.Lambda)):
                inner = _strings(node)
                assert not {"rehearse", "claim", "score", REPORT} & set(inner), ast.unparse(node)[:80]


def test_shared_inputs_are_the_runner_arguments():
    (inputs,) = _assigned(_cell_tree("INPUTS = ["), "INPUTS")
    words = [ast.unparse(e) for e in inputs.elts]
    assert dict(zip(words[::2], words[1::2])) == {
        "'--checkpoints'": "str(CHECKPOINTS_LOCAL)",
        "'--baseline'": "str(BASELINE_LOCAL)",
        "'--risk-policy'": "str(RISK_LOCAL / 'risk_policy.json')",
        "'--profile'": "str(PROFILE)",
        "'--config'": "str(REPO / 'configs/cnn.yaml')",
        "'--manifest'": "str(MANIFEST)",
        "'--stage7-evidence'": "str(REPO / 'docs/audit_evidence/stage7_baseline')",
        "'--stage8-evidence'": "str(REPO / 'docs/audit_evidence/stage8_cnn')",
        "'--stage10-evidence'": "str(REPO / 'docs/audit_evidence/stage10_risk')",
        "'--cache-a'": "str(CACHE_A)",
        "'--cache-b'": "str(CACHE_B)",
    }
    assert "RUNNER = 'src.evaluation.stage11_run'" in _code()[_cell_index("INPUTS = [")]


def test_runner_arguments_per_subcommand():
    (rehearse,), (claim,), (score,) = (_runner_calls(s) for s in ("rehearse", "claim", "score"))
    assert _call_args(rehearse) == {"'--root'": "str(AUDIO_ROOT)", "*INPUTS": "",
                                    "'--stage10-scores'": "str(RISK_LOCAL / 'val_b_scores.npz')",
                                    "'--out'": "str(REHEARSAL_OUT)"}
    assert _call_args(claim) == {"*INPUTS": "", "'--registry'": "str(REGISTRY)",
                                 "'--results-ref'": "RESULTS_REF", "'--out'": "str(CLAIM_OUT)"}
    assert _call_args(score) == {"'--root'": "str(AUDIO_ROOT)", "*INPUTS": "", "'--registry'": "str(REGISTRY)",
                                 "'--run-id'": "RUN_ID", "'--out'": "str(CLAIM_OUT)"}
    for call in (rehearse, claim, score):
        assert "--split" not in _strings(call)
    code = _code()
    assert "REHEARSAL_OUT = Path('/content/stage11_rehearsal')" in code[_cell_index("'rehearse'")]
    assert "CLAIM_OUT = Path('/content/stage11_run')" in code[_cell_index("'claim', *INPUTS")]
    assert "REPORT_OUT = Path('/content/stage11_report')" in code[_cell_index(REPORT)]


def test_report_arguments():
    (call,) = [n for tree in _trees() for n in ast.walk(tree) if isinstance(n, ast.List) and REPORT in _strings(n)]
    words = [ast.unparse(e) for e in call.elts]
    assert words[:3] == ["sys.executable", "'-m'", f"'{REPORT}'"]
    assert dict(zip(words[3::2], words[4::2])) == {
        "'--run-dir'": "str(CLAIM_OUT)",
        "'--registry'": "str(REGISTRY)",
        "'--manifest'": "str(MANIFEST)",
        "'--risk-policy'": "str(RISK_LOCAL / 'risk_policy.json')",
        "'--stage7-evidence'": "str(REPO / 'docs/audit_evidence/stage7_baseline')",
        "'--stage10-evidence'": "str(REPO / 'docs/audit_evidence/stage10_risk')",
        "'--out'": "str(REPORT_OUT)",
    }


def test_assert_clean_brackets_every_runner_and_report_call():
    for marker in ("'rehearse'", "'claim', *INPUTS", "'score', '--root'", REPORT):
        src = _code()[_cell_index(marker)]
        at = src.find(marker)
        assert -1 < src.rfind("assert_clean()", 0, at) < at < src.find("assert_clean()", at), marker


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
    for step in ("refuse_existing_outputs()", "DRY_RUN_DEST.mkdir(parents=True, exist_ok=False)",
                 "drive.flush_and_unmount()", "drive.mount('/content/drive', force_remount=True)",
                 "verify_after_remount(DRY_RUN_DEST, checked)"):
        assert step in body, step
    assert body.find("flush_and_unmount") < body.find("force_remount") < body.find("verify_after_remount")
    assert "STAGE11 / 'dry_run' / DRY_RUN_STAMP" in _code()[_cell_index("DRY_RUN_DEST = ")]


def test_claim_cell_and_every_later_code_cell_start_with_the_official_assert():
    code = _code()
    claim = _cell_index("'claim', *INPUTS")
    for i, src in enumerate(code):
        if i >= claim:
            assert src.splitlines()[0] == OFFICIAL, i
        else:
            assert OFFICIAL not in src, i
            assert "'claim'" not in src and "'score'" not in src, i


def test_rerun_rule_is_in_the_markdown_right_before_the_claim():
    cells = _cells()
    (i,) = [k for k, c in enumerate(cells) if c["cell_type"] == "code" and "'claim', *INPUTS" in _source(c)]
    assert cells[i - 1]["cell_type"] == "markdown"
    assert RERUN_RULE in _source(cells[i - 1])


# --- Claim, durability, score ----------------------------------------------------------------------

def test_registry_is_on_drive_and_never_rewritten_by_the_notebook():
    code = "\n".join(_code())
    assert "STAGE11 = DRIVE / 'processed' / 'metadata' / 'stage11'" in code
    assert "REGISTRY = STAGE11 / 'run_registry.jsonl'" in code
    writers = {"unlink", "rmtree", "rename", "replace", "write_text", "write_bytes", "touch", "move", "remove",
               "copy", "copy2", "copyfile", "truncate"}
    for tree in _trees():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            touched = [n for n in ast.walk(node) if isinstance(n, ast.Name) and n.id == "REGISTRY"]
            if name in writers:
                assert not touched, ast.unparse(node)
            if name == "open" and touched:
                modes = [ast.literal_eval(a) for a in node.args[1:2]] + [
                    ast.literal_eval(k.value) for k in node.keywords if k.arg == "mode"]
                assert all(not set(m) & set("wax+") for m in modes), ast.unparse(node)


def test_registry_is_not_used_before_the_claim():
    code = _code()
    claim = _cell_index("'claim', *INPUTS")
    for i, src in enumerate(code[:claim]):
        uses = [line for line in src.splitlines() if "REGISTRY" in line]
        assert uses in ([], ["REGISTRY = STAGE11 / 'run_registry.jsonl'"]), (i, uses)


def test_run_dest_and_claim_dest_are_created_with_exist_ok_false():
    tree = _cell_tree("'claim', *INPUTS")
    made = {}
    for call in ast.walk(tree):
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "mkdir"
                and isinstance(call.func.value, ast.Name)):
            made[call.func.value.id] = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
    assert made["RUN_DEST"] == {"exist_ok": False} and made["CLAIM_DEST"] == {"exist_ok": False}
    assert "CLAIM_DEST = RUN_DEST / 'claim'" in _code()[_cell_index("'claim', *INPUTS")]
    assert "RUN_DEST = STAGE11 / 'run'" in "\n".join(_code())
    export = _code()[_cell_index("EXPORT_MANIFEST = ")]
    for sub in ("(RUN_DEST / 'clip_scores').mkdir(exist_ok=False)", "(RUN_DEST / 'rehearsal').mkdir(exist_ok=False)"):
        assert sub in export, sub
    assert "assert not dest.exists()" in _code()[_cell_index("def export_file(")]


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


def test_claim_is_verified_on_drive_after_flush_and_remount():
    src = _code()[_cell_index("RUN_ID = ")]
    lines = src.splitlines()
    assert lines[1:3] == ["drive.flush_and_unmount()", "drive.mount('/content/drive', force_remount=True)"]
    asserted = _asserted(ast.parse(src))
    for check in ("sha256(REGISTRY) == CLAIM['registry_sha256']",
                  "LAST['run_id'] == RUN_ID and LAST['status'] == 'started'",
                  "CLAIM['code_commit'] == COMMIT", "CLAIM['results_ref'] == RESULTS_REF",
                  "sha256(DRIVE / entry['path']) == entry['sha256']"):
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

def test_official_export_files_and_manifest():
    tree = _cell_tree("EXPORT_MANIFEST = ")
    src = _code()[_cell_index("EXPORT_MANIFEST = ")]
    (score_files,) = _assigned(tree, "SCORE_FILES")
    assert ast.literal_eval(score_files) == ("scoring_started.json", "environment_score.json", "run_report.json")
    assert "assert len(clip_files) == 8, clip_files" in src
    for name in ("'metrics_report.json', 'metrics_report.md'", "'rehearsal_report.json', 'pip_freeze.txt'"):
        assert name in src, name
    assert "MANIFEST_PATH = RUN_DEST / 'stage11_export_manifest.json'" in src
    (manifest,) = _assigned(tree, "EXPORT_MANIFEST")
    assert {k.value for k in manifest.keys if k is not None} == {"run_id", "results_ref", "registry", "files"}
    assert [ast.unparse(v) for k, v in zip(manifest.keys, manifest.values) if k is None] == ["base_manifest()"]
    (base,) = [n for n in _cell_tree("def base_manifest(").body if isinstance(n, ast.FunctionDef)
               and n.name == "base_manifest"]
    (ret,) = [n.value for n in ast.walk(base) if isinstance(n, ast.Return)]
    assert {k.value for k in ret.keys} == {
        "commit", "mode", "cnn_version", "baseline_version", "risk_version", "python", "numpy", "torch",
        "gpu_name", "created_utc", "for_revision", "for_archive_sha256_observed", "pytest_counts",
        "cache_sha256", "f3", "rehearsal_passed"}
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
                if name in readers and any("metrics_report" in s for s in _strings(node)):
                    assert i == results, (i, ast.unparse(node))
    assert "display(Markdown((REPORT_OUT / 'metrics_report.md').read_text(encoding='utf-8')))" in _code()[results]
    assert results == len(_code()) - 1


def test_steps_run_in_order():
    code = "\n".join(_code())
    markers = ["drive.mount(", "'git', 'clone'", "'pip', 'install'", "'pytest'", "CACHE_PLAN = ",
               "shutil.copytree(", "'rehearse'", "Dry run complete. Stop here.", "'claim', *INPUTS",
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
    for name in ("docs/audit_evidence/run_registry.jsonl", "docs/audit_evidence/stage11_run/", "claim.json",
                 "environment.json", "scoring_started.json", "environment_score.json", "run_report.json",
                 "rehearsal_report.json", "metrics_report.json", "stage11_export_manifest.json",
                 "docs/results/stage11_for_test.md", "run/clip_scores/", "SHA-256 and line count"):
        assert name in text, name
    assert "stay on Drive only" in text
