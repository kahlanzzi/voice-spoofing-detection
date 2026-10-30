"""Governance guard for D31/D41: no code requests the FoR test split before its reviewed runner.

The FoR test split is evaluated once (Stage 11, D31) and again only under the D41 purpose
stage14_robustness, through code reviewed for that purpose. This test scans src/, notebooks/ and
configs/ for code that asks for the test split, so a stray loader, cache build or notebook cell is
caught in CI. ALLOWED is empty: the Stage 11 runner adds itself in its own reviewed PR.
ALLOWED_LINES exempts one rule on one line that is not data access; an entry names the file, the
exact stripped line text, the rule and the reason. Any other rule on that line is still reported,
and any edit to the line text ends the exemption.

What counts as access (Python files and notebook code cells are parsed with ast, so comments,
docstrings and plain mentions in text never count, except for the cli rule, a line-by-line text
match that also fires in a comment, fail-closed; YAML is parsed with yaml.safe_load and JSON files
under configs/ with json.loads; a file that does not parse is reported, never skipped):
- keyword:    a keyword argument whose value is the string "test"     f(split="test")
- argument:   a call argument that is the string "test"               load(cache_dir, "test")
- compare:    == "test", or `in` / .isin() with a literal holding "test"  df[df.split == "test"]
- assign:     the string "test" assigned to a name or attribute        EVAL_SPLIT = "test"
- subscript:  a subscript index equal to "test"                        caches["test"]
- path:       "test" joined with /, or a string constant (not a docstring) with a path segment,
              split on / or \\, equal to "test" or "test.<ext>"   "cache/test.npz", "data/test"
- dict:       a dict literal entry whose key contains "split" (any case) and whose value is
              "test"                                                  {"eval_split": "test"}
- query:      a string argument containing split == 'test'            df.query("split == 'test'")
- cli:        --split test / --split=test in any line (incl. notebook ! lines), or "--split",
              "test" next to each other in a list or tuple literal
- yaml, json: a YAML or configs/ JSON value (not a key) equal to "test"   split: test
The string match ignores case and surrounding spaces.

What does not count, so existing refusal code passes:
- != "test" and `not in` (exclusion), and any comparison inside the condition of an `if` whose
  body only raises (refusal: `if split == "test": raise ValueError(...)`).
- "test" inside a set, tuple, list or dict literal that is not compared or passed (the valid-split
  sets of src/preprocessing/validation.py and src/splitting/for2sec.py, and the audit's folder-name
  map in src/audit/common.py, whose keys do not contain "split"). An allow-list of splits that
  leaves "test" out (CACHE_SPLITS) contains no "test" at all.
- The rule named by an ALLOWED_LINES entry, on its exact line (today only the subscript rule for
  the Stage 4 split-ratio read ratios["test"]).
- Words and path segments that only contain "test" (pytest, tests, testsuite, testing,
  "tests/x.py", "pytest.ini", "data/testing/x"), and docstrings.

Limits (it is a guard, not a sandbox; ADR 0007): it reads text, it does not run code. It cannot see
a split name built at run time ("te" + "st", f"{name}", a variable read from a file or CLI, a
config key looked up indirectly), a subscript through a variable (caches[name]), a dict whose key
does not say "split" ({"which": "test"}), JSON outside configs/, code outside the three folders
(api/, scripts, a Colab cell typed by hand and not committed), or a reviewed module called with a
test argument from an unscanned place. An ALLOWED_LINES entry is added only after review. The
code-level refusals (mfcc_cache / logmel_cache split checks, the run registry) and team review
remain the real controls.

This test reads repository text files only. It never touches a dataset.
"""
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCANNED = ("src/", "notebooks/", "configs/")
# Prefixes (directories end with "/") and exact files allowed to request the FoR test split.
# Empty until the Stage 11 runner adds itself in its own reviewed PR (D41).
ALLOWED: tuple[str, ...] = ()
# (file, exact stripped line text, rule, reason): one rule on one line that is not data access;
# any other rule on that line is still reported.
ALLOWED_LINES: tuple[tuple[str, str, str, str], ...] = (
    (
        "src/splitting/for2sec.py",
        'if not np.isclose(ratios["train"], 0.70) or not np.isclose(ratios["test"], 0.15):',
        "subscript",
        "Stage 4 config check: reads the protocol's test-split ratio from configs/split.yaml, no clip",
    ),
)

_CLI = re.compile(r"--split(?:\s+|\s*=\s*)[\"']?test\b(?![-\w])", re.IGNORECASE)
_QUERY = re.compile(r"\bsplit\s*==\s*[\"']test[\"']", re.IGNORECASE)
_TEST_FILE = re.compile(r"test\.[A-Za-z0-9]+", re.IGNORECASE)  # test.npz, test.csv, ...
_SEPARATORS = re.compile(r"[/\\]")


def _is_test(node) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.strip().lower() == "test"


def _literal_has_test(node) -> bool:
    return isinstance(node, (ast.List, ast.Tuple, ast.Set)) and any(_is_test(e) for e in node.elts)


def _is_test_path(value: str) -> bool:
    """A path segment (split on / or \\) equal to "test" or "test.<ext>"; a bare "test" is not a path."""
    segments = [part.strip().lower() for part in _SEPARATORS.split(value)]
    if len(segments) > 1 and "test" in segments:
        return True
    return any(_TEST_FILE.fullmatch(part) for part in segments)


def is_allowed(rel_path: str) -> bool:
    return any(rel_path == a or (a.endswith("/") and rel_path.startswith(a)) for a in ALLOWED)


def is_scanned(rel_path: str) -> bool:
    return rel_path.startswith(SCANNED)


class _Finder(ast.NodeVisitor):
    def __init__(self):
        self.hits: list[tuple[int, str]] = []
        self._refusal: set[int] = set()  # ids of Compare nodes in a raise-only `if` condition
        self._docstrings: set[int] = set()  # ids of constants that are bare expression statements

    def _hit(self, node, rule):
        self.hits.append((node.lineno, rule))

    def visit_If(self, node):
        if node.body and all(isinstance(s, ast.Raise) for s in node.body) and not node.orelse:
            self._refusal.update(id(n) for n in ast.walk(node.test) if isinstance(n, ast.Compare))
        self.generic_visit(node)

    def visit_Call(self, node):
        for arg in node.args:
            if _is_test(arg):
                self._hit(arg, "argument")
            elif isinstance(arg, ast.Constant) and isinstance(arg.value, str) and _QUERY.search(arg.value):
                self._hit(arg, "query")
        for keyword in node.keywords:
            if _is_test(keyword.value):
                self._hit(keyword.value, "keyword")
        if isinstance(node.func, ast.Attribute) and node.func.attr == "isin" and any(
            _literal_has_test(a) for a in node.args
        ):
            self._hit(node, "compare")
        self.generic_visit(node)

    def visit_Compare(self, node):
        if id(node) not in self._refusal:
            operands = [node.left, *node.comparators]
            for op, left, right in zip(node.ops, operands, operands[1:]):
                if isinstance(op, (ast.Eq, ast.Is)) and (_is_test(left) or _is_test(right)):
                    self._hit(node, "compare")
                if isinstance(op, ast.In) and _literal_has_test(right):
                    self._hit(node, "compare")
        self.generic_visit(node)

    def visit_Assign(self, node):
        if _is_test(node.value):
            self._hit(node, "assign")
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if node.value is not None and _is_test(node.value):
            self._hit(node, "assign")
        self.generic_visit(node)

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Div) and (_is_test(node.left) or _is_test(node.right)):
            self._hit(node, "path")
        self.generic_visit(node)

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant):
            self._docstrings.add(id(node.value))
        self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str) and id(node) not in self._docstrings and _is_test_path(node.value):
            self._hit(node, "path")

    def visit_Subscript(self, node):
        if _is_test(node.slice):
            self._hit(node, "subscript")
        self.generic_visit(node)

    def visit_Dict(self, node):
        for key, value in zip(node.keys, node.values):
            if (isinstance(key, ast.Constant) and isinstance(key.value, str)
                    and "split" in key.value.lower() and _is_test(value)):
                self._hit(key, "dict")
        self.generic_visit(node)

    def _sequence(self, node):
        elts = node.elts
        for first, second in zip(elts, elts[1:]):
            if (isinstance(first, ast.Constant) and first.value == "--split") and _is_test(second):
                self._hit(second, "cli")
        self.generic_visit(node)

    visit_List = visit_Tuple = _sequence


def python_hits(source: str) -> list[tuple[int, str]]:
    """(line, rule) for each test-split request in Python source; SyntaxError if it does not parse."""
    finder = _Finder()
    finder.visit(ast.parse(source))
    return finder.hits


def _notebook_code(text: str) -> list[tuple[int, str]]:
    """(cell number from 1, Python) per code cell; IPython ! and % lines become `pass` (same lines)."""
    cells = []
    for number, cell in enumerate(json.loads(text)["cells"], 1):
        if cell.get("cell_type") != "code":
            continue
        source = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        lines = []
        for line in source.splitlines():
            stripped = line.lstrip()
            indent = line[: len(line) - len(stripped)]
            lines.append(indent + "pass" if stripped.startswith(("!", "%")) else line)
        cells.append((number, "\n".join(lines)))
    return cells


def _config_values(node):
    if isinstance(node, dict):
        for value in node.values():  # keys (e.g. expected split counts "test: 2658") are not access
            yield from _config_values(value)
    elif isinstance(node, list):
        for value in node:
            yield from _config_values(value)
    else:
        yield node


def file_hits(rel: str, text: str) -> list[str]:
    """'path:line: rule: text' for each test-split request in one scanned file."""
    lines = text.splitlines()
    hits = [(n, "cli") for n, line in enumerate(lines, 1) if _CLI.search(line)]
    suffix = Path(rel).suffix.lower()
    try:
        if suffix == ".py":
            hits += python_hits(text)
        elif suffix == ".ipynb":
            for number, cell in _notebook_code(text):
                hits += [(0, f"{rule} (cell {number}, line {line})") for line, rule in python_hits(cell)]
        elif suffix in (".yaml", ".yml") or (suffix == ".json" and rel.startswith("configs/")):
            rule = "json" if suffix == ".json" else "yaml"
            data = json.loads(text) if suffix == ".json" else yaml.safe_load(text)
            hits += [(0, rule) for v in _config_values(data)
                     if isinstance(v, str) and v.strip().lower() == "test"]
    except (SyntaxError, ValueError, yaml.YAMLError) as error:
        # Fail closed: a file the guard cannot read is reported, never skipped.
        hits.append((0, f"cannot parse ({type(error).__name__}: {error})"))
    exempt = {(line, rule) for path, line, rule, _reason in ALLOWED_LINES if path == rel}
    out = []
    for n, rule in sorted(set(hits), key=lambda h: (h[0], h[1])):
        shown = lines[n - 1].strip() if 0 < n <= len(lines) else ""
        if n > 0 and (shown, rule) in exempt:  # same file, exact line text and that rule only
            continue
        out.append(f"{rel}:{n}: {rule}: {shown[:120]}")
    return out


def repo_files(root: Path) -> list[str]:
    """Tracked files plus untracked files that are not ignored, as POSIX paths relative to root."""
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout
    return [line for line in out.splitlines() if line]


def find_violations(root: Path, files: list[str]) -> list[str]:
    hits = []
    for rel in files:
        path = root / rel
        if not is_scanned(rel) or is_allowed(rel) or not path.is_file():
            continue
        hits += file_hits(rel, path.read_bytes().decode("utf-8", errors="replace"))
    return hits


def test_no_for_test_split_access_outside_allowed_files():
    violations = find_violations(REPO_ROOT, repo_files(REPO_ROOT))
    assert not violations, (
        "Code requests the FoR test split outside the reviewed runner (D31, D41; "
        "tests/governance/test_for_test_access.py):\n" + "\n".join(violations)
    )


def test_allow_list_is_empty():
    # Changing it is a reviewed step of the Stage 11 runner PR, never a side effect.
    assert ALLOWED == ()


PLANTED = {
    "src/a_keyword.py": 'cache = build_cache(root, manifest, profile, split="test", out=out)\n',
    "src/b_argument.py": "features = load_mfcc_cache(cache_dir, 'test', profile)\n",
    "src/c_compare.py": 'rows = manifest[manifest["split"] == "test"]\n',
    "src/d_isin.py": 'rows = manifest[manifest.split.isin(["val-B", "test"])]\n',
    "src/e_in.py": 'rows = [r for r in rows if r.split in ("test",)]\n',
    "src/f_assign.py": 'EVAL_SPLIT = "test"\n',
    "src/g_path.py": 'npz = Path(cache_dir) / "test" / "x.npz"\n',
    "src/h_file.py": 'meta = json.loads((cache_dir / "test.json").read_text())\n',
    "src/i_query.py": "rows = manifest.query(\"split == 'test'\")\n",
    "src/j_cli.py": 'subprocess.run(["python", "-m", "src.features.mfcc_cache", "--split", "test"])\n',
    "src/k_case.py": 'load(split=" Test ")\n',
    "src/l_assert.py": 'assert split == "test"\n',
    "src/m_ann.py": 'SPLIT: str = "test"\n',
    "src/n_broken.py": "def broken(:\n",
    "src/o_subscript.py": 'features = caches["test"]\n',
    "src/p_path_file.py": 'NPZ = load_npz("cache/test.npz")\n',
    "src/q_path_dir.py": 'root = Path("data/test")\n',
    "src/r_path_clip.py": 'clips = sorted(glob("splits/test/x.wav"))\n',
    "src/s_join.py": 'folder = os.path.join(root, "test")\n',
    "src/t_dict.py": 'run(**{"split": "test"})\n',
    "src/u_dict_key.py": 'CONFIG = {"eval_split": "test", "seed": 0}\n',
    "src/w_dict_case.py": 'CONFIG = {"Eval_Split": " Test "}\n',
    "configs/stage11.json": '{"evaluation": {"split": "test"}}\n',
    "configs/broken.json": '{"split": \n',
    "notebooks/x.ipynb": json.dumps({"cells": [
        {"cell_type": "markdown", "source": ['run(split="test")']},  # markdown: not code
        {"cell_type": "code", "source": ["!python -m src.features.logmel_cache --split=test --out o\n"]},
        {"cell_type": "code", "source": ["if True:\n", "    %time x = 1\n", "    run(split='test')\n"]},
    ]}),
    "configs/stage11.yaml": "evaluation:\n  split: test\n",
    "configs/list.yaml": "splits: [train, test]\n",
}

LOOKALIKES = {
    "src/r_refusal.py": (
        '"""The test split is refused before any file is read (D31): split="test" never runs."""\n'
        'CACHE_SPLITS = frozenset({"train", "val-A", "val-B"})\n'
        "SPLITS = {'train', 'val-A', 'val-B', 'test'}\n"
        'ALIASES = {"testing": "test", "test": "test"}\n'
        'FORBIDDEN_KEY_PARTS = ("val_b", "val-b", "test", "external")\n'
        "def check(split):\n"
        "    # never: load(split=\"test\")\n"
        '    if split == "test":\n'
        '        raise ValueError("the test split is evaluated once at Stage 11 (D31)")\n'
        '    if split == "test" or split not in CACHE_SPLITS:\n'
        '        raise ValueError(f"bad split {split!r}")\n'
        '    assert split != "test"\n'
        '    if split not in ("test",):\n'
        "        return split\n"
        '    ok = ratios.get(name) == 0.15\n'
        '    print("tests", "pytest", "testsuite", "testing", "test split")\n'
        "    return ok\n"
    ),
    "src/s_cli_like.py": 'parser.add_argument("--split", required=True, choices=sorted(CACHE_SPLITS))\n'
                         'HELP = "--split testing is not a split"\n',
    "notebooks/y.ipynb": json.dumps({"cells": [
        {"cell_type": "code", "source": [
            "test = subprocess.run(['python', '-m', 'pytest', '-q'])\n",
            "print(test.stdout)\n",
            "suites = [root] if root.tag == 'testsuite' else root.findall('testsuite')\n",
        ]},
    ]}),
    "configs/counts.yaml": "expected_splits:\n  train: 12404\n  test: 2658\nratios: {test: 0.15}\n",
    "configs/counts.json": '{"expected_splits": {"test": 2658, "train": 12404}}\n',
    "src/v_paths.py": (
        'def f():\n'
        '    """Splits are train/val-A/val-B/test."""\n'
        '    return ["tests/x.py", "pytest.ini", "data/testing/x", "test", "a/test_b.py", "test split"]\n'
    ),
    "notebooks/z.json": '{"split": "test"}\n',  # JSON is parsed only under configs/
    "tests/t.py": 'load(split="test")\n',  # tests/ and docs/ are not scanned
    "docs/n.md": "--split test\n",
}


def _plant(root, files):
    for rel, content in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(content, encoding="utf-8")


def test_guard_catches_planted_access(tmp_path):
    _plant(tmp_path, PLANTED)
    violations = find_violations(tmp_path, list(PLANTED))
    assert sorted({v.split(":", 1)[0] for v in violations}) == sorted(PLANTED)
    rules = "\n".join(violations)
    for rule in ("keyword", "argument", "compare", "assign", "path", "query", "cli", "yaml",
                 "subscript", "dict", "json", "cannot parse"):
        assert f": {rule}" in rules, rule
    notebook = [v for v in violations if v.startswith("notebooks/")]
    assert len(notebook) == 2  # the ! line (cli) and the cell under %time (keyword); not markdown


def test_guard_ignores_refusal_and_lookalikes(tmp_path):
    # Refusal code (raise-only if, !=, not in), split allow-lists, docstrings, comments, subscript
    # reads, YAML keys and files outside src/, notebooks/ and configs/ are not access.
    _plant(tmp_path, LOOKALIKES)
    assert find_violations(tmp_path, list(LOOKALIKES)) == []


def test_allowed_lines_are_pinned():
    # Each entry is reviewed: this test fails if one is added, removed or changed silently.
    assert ALLOWED_LINES == (
        (
            "src/splitting/for2sec.py",
            'if not np.isclose(ratios["train"], 0.70) or not np.isclose(ratios["test"], 0.15):',
            "subscript",
            "Stage 4 config check: reads the protocol's test-split ratio from configs/split.yaml, no clip",
        ),
    )


def test_allowed_lines_still_exist():
    # A stale entry (its line edited or moved away) would exempt nothing but hide that the line is
    # gone; each entry must match a line of its file and still produce its rule there.
    for rel, line, rule, _reason in ALLOWED_LINES:
        lines = [x.strip() for x in (REPO_ROOT / rel).read_text(encoding="utf-8").splitlines()]
        assert line in lines, (rel, line)
        number = lines.index(line) + 1
        assert (number, rule) in python_hits((REPO_ROOT / rel).read_text(encoding="utf-8")), (rel, rule)


_RATIO_BODY = '    raise SplitError("ratios")\n'


def test_allowed_line_exempts_only_its_exact_line(tmp_path):
    # Path AND exact stripped text: the same line elsewhere, or a one-character edit in the same
    # file, is caught again.
    rel, line, _rule, _reason = ALLOWED_LINES[0]
    _plant(tmp_path, {rel: f"{line}\n{_RATIO_BODY}", "src/other.py": f"{line}\n{_RATIO_BODY}"})
    assert [v.split(":", 1)[0] for v in find_violations(tmp_path, [rel, "src/other.py"])] == ["src/other.py"]
    _plant(tmp_path, {rel: f"{line.replace('0.15', '0.16')}\n{_RATIO_BODY}"})
    caught = find_violations(tmp_path, [rel])
    assert len(caught) == 1
    assert caught[0].startswith(f"{rel}:1: subscript: ")


def test_allowed_line_with_appended_access_is_reported(tmp_path):
    # Appending a real request to the allowed line changes its text, so nothing is exempt: the
    # new keyword hit and the subscript are both reported.
    rel, line, _rule, _reason = ALLOWED_LINES[0]
    assert line.endswith(":")
    changed = line[:-1] + ' or load(split="test"):'
    _plant(tmp_path, {rel: f"{changed}\n{_RATIO_BODY}"})
    rules = sorted(v.split(": ", 2)[1] for v in find_violations(tmp_path, [rel]))
    assert rules == ["keyword", "subscript"]


def test_allowed_line_exempts_only_its_rule(tmp_path, monkeypatch):
    # An entry exempts one rule on one line: another rule on the same, unchanged line is reported.
    line = 'ratio = ratios["test"] or load(split="test")'
    monkeypatch.setattr(
        sys.modules[__name__], "ALLOWED_LINES", (("src/x.py", line, "subscript", "synthetic"),)
    )
    _plant(tmp_path, {"src/x.py": f"{line}\n"})
    caught = find_violations(tmp_path, ["src/x.py"])
    assert len(caught) == 1
    assert caught[0].startswith("src/x.py:1: keyword: ")


def test_refusal_exemption_is_narrow():
    # Only a raise-only `if` is a refusal: an `if` that loads in its body, or one with an else
    # branch, still counts.
    assert python_hits('if split == "test":\n    raise ValueError("no")\n') == []
    assert python_hits('if split == "test":\n    rows = load(split)\n') == [(1, "compare")]
    assert python_hits('if split == "test":\n    raise ValueError("no")\nelse:\n    pass\n') == [(1, "compare")]


def test_allowed_and_scanned_prefixes():
    assert is_scanned("src/evaluation/clip_scores.py")
    assert is_scanned("notebooks/06_stage11.ipynb")
    assert is_scanned("configs/x.yaml")
    assert not is_scanned("tests/evaluation/test_clip_scores.py")
    assert not is_scanned("docs/evaluation_harness.md")
    assert not is_scanned("srcx/a.py")
    assert not is_allowed("src/evaluation/run_registry.py")
