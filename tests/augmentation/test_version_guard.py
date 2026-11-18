from importlib.metadata import PackageNotFoundError

import pytest

from src.augmentation.version_guard import require_versions, required_versions


def _requirements(tmp_path, text="numpy==2.5.3\nscipy==1.18.1\nsoundfile==0.14.0\n"):
    path = tmp_path / "requirements.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_reads_controlling_exact_pins_and_accepts_exact_environment(tmp_path):
    path = _requirements(tmp_path)
    expected = {"numpy": "2.5.3", "scipy": "1.18.1", "soundfile": "0.14.0"}
    assert required_versions(path) == expected
    assert require_versions(path, version=expected.__getitem__) == expected


def test_refuses_version_drift(tmp_path):
    path = _requirements(tmp_path)
    actual = {"numpy": "2.5.3", "scipy": "1.16.3", "soundfile": "0.14.0"}
    with pytest.raises(ValueError, match="scipy: expected 1.18.1, got 1.16.3"):
        require_versions(path, version=actual.__getitem__)


def test_refuses_missing_or_duplicate_pins(tmp_path):
    with pytest.raises(ValueError, match="missing exact D64 pin"):
        required_versions(_requirements(tmp_path, "numpy==2.5.3\nscipy==1.18.1\n"))
    with pytest.raises(ValueError, match="duplicate exact pin"):
        required_versions(_requirements(
            tmp_path,
            "numpy==2.5.3\nnumpy==2.5.3\nscipy==1.18.1\nsoundfile==0.14.0\n",
        ))


def test_refuses_missing_installed_package(tmp_path):
    def missing(_name):
        raise PackageNotFoundError

    with pytest.raises(ValueError, match="package is missing"):
        require_versions(_requirements(tmp_path), version=missing)
