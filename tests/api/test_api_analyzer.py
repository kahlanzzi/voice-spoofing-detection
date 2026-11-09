import ast
from pathlib import Path

import numpy as np

import api.analyzer
from api.analyzer import AnalysisOutcome, AnalyzerInfo, MockAnalyzer, file_sha256

SHAPES = {"synthetic": "HIGH", "uncertain": "MEDIUM", "authentic": "LOW"}
LOWEST = {"synthetic": 1.0, "uncertain": 0.50, "authentic": 0.73}


def random_files(tmp_path, n):
    rng = np.random.default_rng(7)
    for i in range(n):
        path = tmp_path / f"f{i}.bin"
        path.write_bytes(rng.bytes(100))
        yield path


def test_same_bytes_give_the_same_outcome(tmp_path):
    a, b = tmp_path / "a.wav", tmp_path / "b.flac"
    a.write_bytes(b"same bytes")
    b.write_bytes(b"same bytes")
    mock = MockAnalyzer(0.1)
    assert mock.analyze(a) == mock.analyze(b) == MockAnalyzer(0.5).analyze(a)


def test_outcomes_follow_risk_v1_shapes_and_all_verdicts_occur(tmp_path):
    mock = MockAnalyzer(0.1)
    seen = set()
    for path in random_files(tmp_path, 90):
        outcome = mock.analyze(path)
        assert isinstance(outcome, AnalysisOutcome)
        assert SHAPES[outcome.prediction] == outcome.risk_level
        assert LOWEST[outcome.prediction] <= outcome.confidence <= 1.0
        assert outcome.confidence == round(outcome.confidence, 2)
        assert (outcome.model_version, outcome.profile_version, outcome.policy_version) == \
            ("mock", "preprocessing-v1", "risk-v1")
        seen.add(outcome.prediction)
    assert seen == set(SHAPES)


def test_info_is_labelled_mock_with_the_configured_deferral():
    assert MockAnalyzer(0.0993).info() == AnalyzerInfo(
        ready=True, model_version="mock", profile_version="preprocessing-v1", policy_version="risk-v1",
        deferral_on_fit=0.0993, bundle_manifest_sha256=None, mock=True)


def test_file_sha256_reads_in_chunks(tmp_path):
    import hashlib
    path = tmp_path / "big.bin"
    data = bytes(range(256)) * 1000  # larger than one chunk
    path.write_bytes(data)
    assert file_sha256(path) == hashlib.sha256(data).digest()


def test_analyzer_module_never_imports_model_code():
    tree = ast.parse(Path(api.analyzer.__file__).read_text(encoding="utf-8"))
    names = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    names += [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not [n for n in names if n.split(".")[0] in {"torch", "src"}]
