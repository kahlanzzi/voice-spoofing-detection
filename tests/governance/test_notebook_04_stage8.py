"""Static checks for the Stage 8 Colab notebook (notebooks/04_stage8_cnn.ipynb; D30, D40, D42).

The notebook is committed with outputs cleared and is only run in Colab, so these tests read its JSON
and parse each code cell with ast; they never execute it. They pin what the official run must and
must not do: torch is never installed, the W&B key comes only from Colab secrets, only train and
val-A are cached, the search runs once, W&B writes outside the clone, and the steps run in order.
"""
import ast
import json
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO_ROOT / "notebooks" / "04_stage8_cnn.ipynb"
CLONE = "/content/voice-spoofing-detection"  # where the notebook clones the repo in Colab
WANDB_DIRS = ("WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR")


def _cells() -> list[dict]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _code() -> list[str]:
    return [_source(c) for c in _cells() if c["cell_type"] == "code"]


def _trees() -> list[ast.Module]:
    return [ast.parse(src) for src in _code()]


def _strings(node: ast.AST) -> list[str]:
    return [n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def _is_environ(node: ast.AST) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr == "environ"
            and isinstance(node.value, ast.Name) and node.value.id == "os")


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


def test_torch_is_never_installed():
    text = "\n".join(_source(c) for c in _cells())
    assert not re.search(r"(pip|uv|conda)\b.*\binstall\b.*torch", text, re.IGNORECASE)
    installs = [n for tree in _trees() for n in ast.walk(tree)
                if isinstance(n, ast.List) and "install" in _strings(n)]
    assert len(installs) == 1  # the only install: the two requirements files, after the torch check
    words = _strings(installs[0])
    assert not any("torch" in w.lower() for w in words), words
    assert {"requirements.txt", "requirements-cnn.txt"} <= set(words)


def test_wandb_key_comes_only_from_colab_userdata():
    code = "\n".join(_code())
    assert "from google.colab import userdata" in code
    assert "wandb.login" not in code and "getenv" not in code
    reads = writes = 0
    for tree in _trees():
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get" and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "userdata" and _strings(node) == ["WANDB_API_KEY"]):
                reads += 1
            elif (isinstance(node, ast.Subscript) and _is_environ(node.value)
                  and isinstance(node.ctx, ast.Store) and _strings(node.slice) == ["WANDB_API_KEY"]):
                writes += 1
    assert (reads, writes) == (1, 1)
    assert code.count("WANDB_API_KEY") == 2  # those two places only: never read back or printed


def test_no_key_like_literals():
    for cell in _cells():
        assert not re.search(r"[0-9a-fA-F]{40}", _source(cell)), cell["id"]  # also W&B keys
    for tree in _trees():
        for value in _strings(tree):
            assert not re.fullmatch(r"[A-Za-z0-9_\-]{32,}", value), value


def test_only_train_and_val_a_are_cached():
    strings = [s for tree in _trees() for s in _strings(tree)]
    for forbidden in ("test", "val-B", "val_b", "valB", "val-b"):
        assert forbidden not in strings, forbidden
    cache_cells = [t for t, src in zip(_trees(), _code()) if "src.features.logmel_cache" in src]
    assert len(cache_cells) == 1
    splits = [n.value for n in ast.walk(cache_cells[0]) if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "CACHE_SPLITS" for t in n.targets)]
    assert len(splits) == 1 and ast.literal_eval(splits[0]) == ("train", "val-A")


def test_cnn_search_is_invoked_exactly_once_outside_loops():
    assert "\n".join(_code()).count("src.tuning.cnn_search") == 1
    for tree in _trees():
        for loop in ast.walk(tree):
            if isinstance(loop, (ast.For, ast.While, ast.AsyncFor)):
                assert "src.tuning.cnn_search" not in _strings(loop)
    (call,) = [n for tree in _trees() for n in ast.walk(tree)
               if isinstance(n, ast.List) and "src.tuning.cnn_search" in _strings(n)]
    words = _strings(call)
    for flag, value in (("--wandb-mode", "online"), ("--device", "cuda")):
        assert words[words.index(flag) + 1] == value


def test_wandb_dirs_are_under_content_and_outside_the_clone():
    found = {}
    for tree in _trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if not (isinstance(target, ast.Subscript) and _is_environ(target.value)):
                    continue
                name = next(iter(_strings(target.slice)), None)
                if name in WANDB_DIRS:
                    assert isinstance(node.value, ast.Constant), "W&B folders must be literal paths"
                    found[name] = node.value.value
    assert set(found) == set(WANDB_DIRS)
    for name, path in found.items():
        assert path.startswith("/content/") and not (path + "/").startswith(CLONE + "/"), (name, path)


def _cell_tree(marker: str) -> ast.Module:
    (tree,) = [t for t, src in zip(_trees(), _code()) if marker in src]
    return tree


def _asserted(tree: ast.Module) -> list[str]:
    return [ast.unparse(n.test) for n in ast.walk(tree) if isinstance(n, ast.Assert)]


def test_python_3_13_is_asserted_after_the_install():
    assert "ENV['python'].startswith('3.13.')" in _asserted(_cell_tree("'pip', 'install'"))  # D15


def test_export_requires_exactly_the_final_seed_checkpoints():
    seeds = yaml.safe_load((REPO_ROOT / "configs" / "cnn.yaml").read_text(encoding="utf-8"))["final_seeds"]
    expected = f"[d.name for d in SEED_DIRS] == {[f'seed_{s}' for s in seeds]!r}"
    assert expected in _asserted(_cell_tree("shutil.copytree("))


def test_export_manifest_records_the_run_evidence():
    (manifest,) = [n.value for n in ast.walk(_cell_tree("shutil.copytree(")) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "EXPORT_MANIFEST" for t in n.targets)]
    keys = {k.value for k in manifest.keys}
    assert keys == {"commit", "version", "python", "torch", "cuda_version", "cudnn_version", "gpu_name",
                    "created_utc", "for_revision", "for_archive_sha256_observed", "pytest_counts",
                    "cache_sha256", "f3", "files"}


def test_export_is_rehashed_after_flush_and_remount():
    (export,) = [src for src in _code() if "shutil.copytree(" in src]
    flush = export.find("drive.flush_and_unmount()")
    remount = export.find("drive.mount('/content/drive', force_remount=True)")
    assert -1 < flush < remount, (flush, remount)
    after = export[remount:]
    for check in ("sha256(DRIVE / path)", "missing", "extra"):
        assert check in after, check


def test_steps_run_in_order():
    code = "\n".join(_code())
    markers = ["drive.mount(", "'git', 'clone'", "assert_clean()", "'pip', 'install'", "'pytest'",
               "src.features.logmel_cache", "userdata.get(", "src.tuning.cnn_search", "shutil.copytree("]
    positions = [code.find(m) for m in markers]
    assert -1 not in positions, dict(zip(markers, positions))
    assert positions == sorted(positions), dict(zip(markers, positions))
