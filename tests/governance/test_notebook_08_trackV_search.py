"""Static checks for the Track V search notebook (notebooks/08_trackV_v2_search.ipynb; D9, D40, D49, D58, D64).

The notebook is committed with outputs cleared and only run in Colab, so these tests read its JSON and
parse each code cell with ast; they never execute it. They pin: the fixed commit (refuse empty, 40
characters, HEAD and a clean worktree before and after every long cell), require_versions in a fresh
process after the install, the Track V Drive layout (no v1 folder written), the V1c builders called with
exactly their arguments and only when their Drive folder is absent, archive hashes checked before
extraction, the rehearsal and search CLI calls, no output inside the clone, and the FoR guard.

The small helpers mirror tests/governance/test_notebook_06_stage11.py (private there, so copied).
"""
import ast
import json
from pathlib import Path

from tests.governance.test_for_test_access import file_hits

REPO_ROOT = Path(__file__).resolve().parents[2]
REL = "notebooks/08_trackV_v2_search.ipynb"
NOTEBOOK = REPO_ROOT / REL
LONG_MODULES = ("src.features.logmel_cache", "src.features.raw_train_cache", "src.augmentation.selection_set_build",
                "src.augmentation.resource_cache", "src.tuning.v2_search")


def _cells() -> list[dict]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _code() -> list[str]:
    return [_source(c) for c in _cells() if c["cell_type"] == "code"]


def _markdown() -> str:
    return "\n".join(_source(c) for c in _cells() if c["cell_type"] == "markdown")


def _tree(marker: str) -> ast.Module:
    (src,) = [s for s in _code() if marker in s]
    return ast.parse(src)


def _calls(tree: ast.AST, name: str) -> list[ast.Call]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == name]


def _command_lists(tree: ast.AST, module: str) -> list[ast.List]:
    """[sys.executable, '-m', <module>, ...] list literals."""
    return [n for n in ast.walk(tree) if isinstance(n, ast.List) and len(n.elts) > 2
            and ast.unparse(n.elts[0]) == "sys.executable" and isinstance(n.elts[2], ast.Constant)
            and n.elts[2].value == module]


def _flags(command: ast.List, start: int) -> dict[str, str]:
    words = [ast.unparse(e) for e in command.elts[start:]]
    out, i = {}, 0
    while i < len(words):
        if words[i].startswith("*"):
            out[words[i]] = ""
            i += 1
        else:
            out[ast.literal_eval(words[i])] = words[i + 1]
            i += 2
    return out


def _guarding_if(tree: ast.Module, node: ast.AST) -> str | None:
    """The unparsed test of the innermost `if` whose body (not orelse) contains node."""
    found = None
    for candidate in ast.walk(tree):
        if isinstance(candidate, ast.If) and any(node is n for s in candidate.body for n in ast.walk(s)):
            found = ast.unparse(candidate.test)
    return found


def _assigned(tree: ast.AST, name: str) -> list[str]:
    return [ast.unparse(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]


# --- shape ---------------------------------------------------------------------------------------

def test_valid_json_and_outputs_cleared():
    data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert data["nbformat"] == 4 and data["metadata"]["colab"]["gpuType"] == "T4"
    for cell in data["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None, cell["id"]


def test_every_code_cell_follows_a_markdown_cell_and_no_shell_magics():
    cells = _cells()
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            assert i > 0 and cells[i - 1]["cell_type"] == "markdown", cell["id"]
            src = _source(cell)
            assert not any(line.lstrip().startswith(("!", "%")) for line in src.splitlines())
            ast.parse(src)


def test_parameters_commit_placeholder_and_refusals():
    params = ast.parse(_code()[0])
    values = {t.id: ast.literal_eval(n.value) for n in params.body if isinstance(n, ast.Assign) for t in n.targets}
    assert values == {
        "REPO_URL": "https://github.com/kahlanzzi/voice-spoofing-detection.git",
        "COMMIT": "REPLACE_WITH_MERGED_MAIN_COMMIT",
        "SEARCH_VERSION": "model-v2",
        "NUM_WORKERS": 2,
        "COLAB_RUNTIME_VERSION": "2026.10",
        "DRIVE_ROOT": "/content/drive/MyDrive/voice-spoofing-detection",
        "CORPUS_DIR": "/content/drive/MyDrive/voice-spoofing-detection-v1b-checkpoint",
    }
    asserted = [ast.unparse(n.test) for n in ast.walk(params) if isinstance(n, ast.Assert)]
    assert "COMMIT and COMMIT != 'REPLACE_WITH_MERGED_MAIN_COMMIT'" in asserted
    assert "len(COMMIT) == 40" in asserted
    clone = _tree("'git', 'clone'")
    assert "len(COMMIT) == 40" in [ast.unparse(n.test) for n in ast.walk(clone) if isinstance(n, ast.Assert)]
    assert "'--detach', COMMIT" in ast.unparse(clone)
    assert "assert head == COMMIT" in ast.unparse(clone)
    assert "voice-spoofing-detection-v1b-checkpoint" in _markdown() and "shortcut" in _markdown()


def test_head_and_clean_worktree_before_and_after_every_long_cell():
    long_cells = 0
    for src in _code():
        tree = ast.parse(src)
        long_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and (
            (ast.unparse(n.func) == "stream")
            or (ast.unparse(n.func) == "subprocess.run" and "'pip', 'install'" in ast.unparse(n)))]
        if not long_calls:
            continue
        long_cells += 1
        clean = [n.lineno for n in _calls(tree, "assert_clean")]
        assert clean and min(clean) < min(c.lineno for c in long_calls), src[:80]
        assert max(clean) > max(c.end_lineno for c in long_calls), src[:80]
    assert long_cells >= 7  # install, raw verify, log-Mel, S, resources, rehearsal, search


def test_every_long_module_is_run_through_stream_from_the_clone():
    for module in LONG_MODULES:
        found = [c for src in _code() for c in _command_lists(ast.parse(src), module)]
        assert found, module


# --- runtime -------------------------------------------------------------------------------------

def test_runtime_gate_and_require_versions_in_a_fresh_process_after_the_install():
    src = [s for s in _code() if "'pip', 'install'" in s][0]
    tree = ast.parse(src)
    install = [n for n in _calls(tree, "subprocess.run") if "'pip', 'install'" in ast.unparse(n)]
    (install,) = install
    assert "'-r', 'requirements.txt', '-r', 'requirements-cnn.txt'" in ast.unparse(install)
    probes = [n for n in _calls(tree, "probe") if "require_versions()" in ast.unparse(n)]
    assert len(probes) == 1 and probes[0].lineno > install.end_lineno
    assert "from src.augmentation.version_guard import require_versions" in ast.unparse(probes[0])
    text = ast.unparse(tree)
    assert "torch.__version__ == '2.11.0+cu130'" in text and "ENV['python'] == '3.13.16'" in text
    assert "'T4' in GPU_NAME" in text
    # no second version helper and no in-kernel import of the guard
    for code in _code():
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                assert "version_guard" not in ast.unparse(node)
            if isinstance(node, ast.FunctionDef):
                assert "version" not in node.name


def test_pinned_ffmpeg_installed_and_verified():
    tree = _tree("FFMPEG_PACKAGE")
    text = ast.unparse(tree)
    assert _assigned(tree, "FFMPEG_PACKAGE") == ["'7:6.1.1-3ubuntu5'"]
    assert "f'ffmpeg={FFMPEG_PACKAGE}'" in text and "codec.verify_pin()" in text


# --- inputs --------------------------------------------------------------------------------------

def test_drive_layout_constants():
    tree = _tree("drive.mount('/content/drive')\n")
    want = {
        "V2": "DRIVE / 'processed' / 'v2'",
        "RAW_CACHE_DRIVE": "V2 / 'raw_train_cache'",
        "STAGE8_LOGMEL_DRIVE": "V2 / 'stage8_logmel'",
        "SELECTION_SET_DRIVE": "V2 / 'selection_set_s'",
        "RESOURCES_DRIVE": "V2 / 'aug_resources'",
        "V1C_EVIDENCE_DRIVE": "V2 / 'v1c_evidence'",
        "SEARCH_ROOT": "DRIVE / 'checkpoints' / 'v2_search'",
        "SEARCH_OUT": "SEARCH_ROOT / SEARCH_VERSION",
        "LOCAL": "Path('/content/v2_inputs')",
    }
    for name, value in want.items():
        assert _assigned(tree, name) == [value], name
    assert _assigned(_tree("REHEARSAL_OUT = "), "REHEARSAL_OUT") == [
        "SEARCH_ROOT / 'rehearsal' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')"]


def test_no_write_to_v1_folders():
    """Every DRIVE / ... path is under processed/v2, checkpoints/v2_search or the read-only raw/."""
    for src in _code():
        for node in ast.walk(ast.parse(src)):
            if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)):
                continue
            parts, left = [], node
            while isinstance(left, ast.BinOp) and isinstance(left.op, ast.Div):
                parts.insert(0, left.right)
                left = left.left
            if not (isinstance(left, ast.Name) and left.id == "DRIVE"):
                continue
            first = ast.literal_eval(parts[0])
            assert first in ("processed", "checkpoints", "raw"), ast.unparse(node)
            if len(parts) == 1:
                continue  # the inner `DRIVE / 'x'` of a longer chain; the full chain is checked too
            if first == "processed":
                assert ast.literal_eval(parts[1]) == "v2", ast.unparse(node)
            if first == "checkpoints":
                assert ast.literal_eval(parts[1]) == "v2_search", ast.unparse(node)
    text = "\n".join(_code())
    for v1 in ("'cnn'", "'stage8'", "'stage10'", "'stage11'", "'risk'", "'metadata'"):
        assert v1 not in text, v1


def test_no_output_inside_the_clone_and_no_dataset_path_in_the_repo():
    for src in _code():
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and ast.unparse(node.left) == "REPO":
                assert ast.literal_eval(node.right).startswith(("docs/", "configs/", "data/manifests/")), ast.unparse(node)
            if isinstance(node, ast.List):
                words = [ast.unparse(e) for e in node.elts]
                for flag, value in zip(words, words[1:]):
                    if flag == "'--out'":
                        assert value in ("str(local_dir)", "str(LOCAL_S)", "str(LOCAL_RESOURCES)",
                                         "str(REHEARSAL_OUT)", "str(SEARCH_OUT)"), value
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("/content"):
                assert node.value == "/content" or node.value.startswith(("/content/drive", "/content/voice-spoofing-detection",
                                              "/content/v2_inputs", "/content/v2_corpora", "/content/data",
                                              "/content/wandb_runtime")), node.value
    assert _assigned(_tree("'git', 'clone'"), "REPO") == ["Path('/content/voice-spoofing-detection')"]


def test_raw_cache_checked_against_the_v2b3_evidence():
    text = ast.unparse(_tree("RAW_EVIDENCE"))
    assert "docs/audit_evidence/v2b3_raw_train_cache/train_raw.json" in text
    assert "RAW_NPZ_SHA256 == RAW_EVIDENCE['npz_sha256']" in text
    assert "'verify', '--dir', str(LOCAL_RAW)" in text


def test_stage8_logmel_built_only_if_absent_and_checked_against_stage8():
    tree = _tree("STAGE8_EXPECTED = ")
    text = ast.unparse(tree)
    assert "docs/audit_evidence/stage8_cnn/run_report.json" in text
    assert "STAGE8['train_cache_npz_sha256']" in text and "STAGE8['val_a_cache_npz_sha256']" in text
    (command,) = _command_lists(tree, "src.features.logmel_cache")
    assert _flags(command, 3) == {"--root": "str(AUDIO_ROOT)", "--manifest": "str(MANIFEST)",
                                  "--profile": "str(PROFILE)", "--split": "split", "--out": "str(local_dir)"}
    assert _guarding_if(tree, command) == "LOGMEL_BUILT[split]"
    assert "LOGMEL_BUILT[split] = not drive_dir.exists()" in text
    assert "meta['npz_sha256'] == actual == expected" in text
    copies = [c for c in _calls(tree, "copy_exclusive")]
    assert len(copies) == 1 and _guarding_if(tree, copies[0]) == "LOGMEL_BUILT[split]"


def test_corpus_archives_hash_checked_before_extraction():
    tree = _tree("NEED_CORPORA")
    text = ast.unparse(tree)
    assert "NEED_CORPORA = not SELECTION_SET_DRIVE.exists() or not RESOURCES_DRIVE.exists()" in text
    assert "(('noise', 'musan.tar.gz'), ('rir', 'rirs_noises.zip'))" in text
    check = [n for n in ast.walk(tree) if isinstance(n, ast.Assert) and "actual == expected" in ast.unparse(n.test)]
    assert len(check) == 1
    extract_lines = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and ast.unparse(n.func).endswith(("extractall", "extract"))]
    assert extract_lines and check[0].lineno < min(extract_lines)
    assert "archive.extract(member, path=CORPUS_WORK / 'musan', filter='data')" in text
    assert "member.name.startswith('musan/noise/')" in text and "archive.testzip() is None" in text


def test_selection_set_builder_called_unchanged_and_only_when_absent():
    tree = _tree("S_BUILT = not SELECTION_SET_DRIVE")
    (command,) = _command_lists(tree, "src.augmentation.selection_set_build")
    assert _flags(command, 3) == {
        "--for-root": "str(AUDIO_ROOT)", "--f2-manifest": "str(MANIFEST)",
        "--recipe": "str(REPO / 'data/manifests/augmentation/selection_set_s.csv')",
        "--musan-noise-root": "str(MUSAN_NOISE_ROOT)", "--rirs-root": "str(RIRS_ROOT)",
        "--clean-reference-logmel": "str(LOCAL_LOGMEL / 'val-A')",
        "--colab-runtime-version": "COLAB_RUNTIME_VERSION", "--out": "str(LOCAL_S)"}
    assert _assigned(tree, "S_BUILT") == ["not SELECTION_SET_DRIVE.exists()"]
    assert _guarding_if(tree, command) == "S_BUILT"
    (copy,) = _calls(tree, "copy_exclusive")
    assert ast.unparse(copy) == "copy_exclusive(LOCAL_S, SELECTION_SET_DRIVE)" and _guarding_if(tree, copy) == "S_BUILT"
    text = ast.unparse(tree)
    assert "for form in ('waveform', 'logmel')" in text and "clean_reference_logmel=reference" in text
    assert "check_stage8_reference" in text and "print('selection_set_sha256:'" in text
    # the verification runs in every session: it is not inside the `if S_BUILT` body
    (verify,) = _calls(tree, "probe")
    assert _guarding_if(tree, verify) is None


def test_resource_cache_builder_called_unchanged_and_only_when_absent():
    tree = _tree("RESOURCES_BUILT = not")
    commands = _command_lists(tree, "src.augmentation.resource_cache")
    assert [ast.literal_eval(c.elts[3]) for c in commands] == ["build", "verify"]
    assert _flags(commands[0], 4) == {"--musan-noise-root": "str(MUSAN_NOISE_ROOT)", "--rirs-root": "str(RIRS_ROOT)",
                                      "--out": "str(LOCAL_RESOURCES)"}
    assert _flags(commands[1], 4) == {"--dir": "str(LOCAL_RESOURCES)"}
    assert _assigned(tree, "RESOURCES_BUILT") == ["not RESOURCES_DRIVE.exists()"]
    assert all(_guarding_if(tree, c) == "RESOURCES_BUILT" for c in commands)
    (copy,) = _calls(tree, "copy_exclusive")
    assert _guarding_if(tree, copy) == "RESOURCES_BUILT"
    (load,) = _calls(tree, "probe")
    assert "from src.tuning.v2_search import load_v2_resources" in ast.unparse(load)
    assert _guarding_if(tree, load) is None


def test_v1c_evidence_written_once_and_checked_later():
    tree = _tree("v1c_build_report.json")
    text = ast.unparse(tree)
    assert "open(REPORT_PATH, 'x'" in text and "V1C_EVIDENCE_DRIVE.mkdir(parents=True, exist_ok=False)" in text
    for key in ("'selection_set_sha256': SELECTION_SET_SHA256", "'code_commit': COMMIT",
                "'require_versions': VERSIONS", "'resource_cache_npz_sha256'"):
        assert key in text, key
    assert "existing[key] == value" in text


def test_copy_helpers_never_overwrite():
    tree = _tree("def copy_exclusive")
    text = ast.unparse(tree)
    assert "if dest.exists():\n        raise FileExistsError" in text
    assert "assert actual == expected" in text


# --- rehearsal and search ------------------------------------------------------------------------

def test_rehearsal_then_search_cli_calls():
    rehearse_tree = _tree("'rehearse'")
    (rehearse,) = _command_lists(rehearse_tree, "src.tuning.v2_search")
    assert ast.literal_eval(rehearse.elts[3]) == "rehearse"
    assert _flags(rehearse, 4) == {"*INPUT_ARGS": "", "--out": "str(REHEARSAL_OUT)"}
    assert "assert not REHEARSAL_OUT.exists()" in ast.unparse(rehearse_tree)
    search_tree = _tree("'search', *INPUT_ARGS")
    (search,) = _command_lists(search_tree, "src.tuning.v2_search")
    assert _flags(search, 4) == {"*INPUT_ARGS": "", "--out": "str(SEARCH_OUT)", "--version": "SEARCH_VERSION",
                                 "--wandb-mode": "'online'"}
    code = _code()
    i_rehearse = [i for i, s in enumerate(code) if "'rehearse'" in s][0]
    i_wandb = [i for i, s in enumerate(code) if "WANDB_API_KEY" in s][0]
    i_search = [i for i, s in enumerate(code) if "'search', *INPUT_ARGS" in s][0]
    assert i_rehearse < i_wandb < i_search
    (args,) = _assigned(_tree("INPUT_ARGS = "), "INPUT_ARGS")
    for flag in ("--raw-cache-dir", "--stage8-train-logmel-dir", "--stage8-val-a-logmel-dir",
                 "--selection-set-dir", "--resources-dir", "--device", "--num-workers"):
        assert f"'{flag}'" in args, flag
    assert "'cuda'" in args


def test_reports_rehashed_after_drive_flush():
    text = ast.unparse(_tree("flush_and_unmount"))
    for name in ("'search_report.json'", "'selection_report.json'", "'run_report.json'"):
        assert name in text
    assert text.index("drive.flush_and_unmount()") < text.index("force_remount=True") < text.index(
        "sha256(path) == digest")
    assert "RUN_REPORT['code_commit'] == COMMIT" in text


def test_for_guard_finds_nothing():
    assert file_hits(REL, NOTEBOOK.read_text(encoding="utf-8")) == []
