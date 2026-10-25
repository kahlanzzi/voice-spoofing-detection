"""Governance guard for AGENTS.md hard rule 1 (D20): DEEP-VOICE paths stay in permitted places.

Before the single guarded external run, the only code allowed to know where DEEP-VOICE lives is the
metadata audit (src/audit/, configs/audit.yaml, the Stage 1 notebook). Documentation and tests may
mention the paths. Any other file that references the DEEP-VOICE root, the Kaggle archive or its
KAGGLE/ subfolder fails this test, so a stray loader or notebook is caught in CI.

This test reads repository text files only. It never touches a dataset.
"""
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Prefixes (directories end with "/") and exact files allowed to reference DEEP-VOICE paths.
ALLOWED = (
    "src/audit/",
    "configs/audit.yaml",
    "notebooks/01_stage1_dataset_audit.ipynb",
    "src/evaluation/external_run.py",
    "configs/external_run.yaml",
    "docs/",
    "tests/",
    "data/README.md",  # documents the Drive layout; no code
)

PATTERNS = (
    re.compile(r"deep_voice", re.IGNORECASE),                            # local / Drive root folder
    re.compile(r"deep-voice-deepfake-voice-recognition", re.IGNORECASE),  # Kaggle slug / zip name
    re.compile(r"birdy654", re.IGNORECASE),                              # Kaggle owner
    re.compile(r"\bKAGGLE\b"),                                           # audit subfolder KAGGLE/
)

BINARY_SUFFIXES = {".pdf", ".docx", ".png", ".jpg", ".wav", ".npy", ".pt", ".zip", ".tar"}


def is_allowed(rel_path: str) -> bool:
    return any(rel_path == a or (a.endswith("/") and rel_path.startswith(a)) for a in ALLOWED)


def repo_files(root: Path) -> list[str]:
    """Tracked files plus untracked files that are not ignored, as POSIX paths relative to root."""
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout
    return [line for line in out.splitlines() if line]


def find_violations(root: Path, files: list[str]) -> list[str]:
    """Return 'path:line: text' for every DEEP-VOICE path reference outside the allowed places."""
    hits = []
    for rel in files:
        path = root / rel
        if is_allowed(rel) or path.suffix.lower() in BINARY_SUFFIXES or not path.is_file():
            continue
        text = path.read_bytes().decode("utf-8", errors="replace")
        for n, line in enumerate(text.splitlines(), 1):
            if any(p.search(line) for p in PATTERNS):
                hits.append(f"{rel}:{n}: {line.strip()[:120]}")
    return hits


def test_no_deepvoice_paths_outside_allowed_files():
    violations = find_violations(REPO_ROOT, repo_files(REPO_ROOT))
    assert not violations, (
        "DEEP-VOICE paths referenced outside the permitted files (AGENTS.md hard rule 1, D20):\n"
        + "\n".join(violations)
    )


def test_guard_catches_planted_violation(tmp_path):
    files = {
        "src/models/train.py": 'DATA = "/content/data/deep_voice/KAGGLE/AUDIO"\n',
        "notebooks/02_explore.ipynb": '"kaggle datasets download birdy654/deep-voice-deepfake-voice-recognition"\n',
        "api/app.py": 'ROOT = Path(cfg.root) / "KAGGLE"\n',
        "src/audit/deepvoice_metadata_audit.py": 'root = "/content/data/deep_voice"\n',  # allowed
        "docs/notes.md": "raw/deep_voice/ lives in Drive\n",                         # allowed
        "src/preprocessing/env.py": 'token = os.environ["KAGGLE_API_TOKEN"]\n',      # not a path
    }
    for rel, content in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(content, encoding="utf-8")

    violations = find_violations(tmp_path, list(files))

    flagged = sorted({v.split(":", 1)[0] for v in violations})
    assert flagged == ["api/app.py", "notebooks/02_explore.ipynb", "src/models/train.py"]


def test_allowed_prefixes_do_not_leak():
    assert is_allowed("src/audit/common.py")
    assert is_allowed("configs/audit.yaml")
    assert is_allowed("src/evaluation/external_run.py")
    assert is_allowed("configs/external_run.yaml")
    assert not is_allowed("configs/audit_v2.yaml")
    assert not is_allowed("src/auditing/x.py")
    assert not is_allowed("notebooks/02_stage8_training.ipynb")
