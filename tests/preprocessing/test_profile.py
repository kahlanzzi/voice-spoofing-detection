"""Stage 6 profile-schema tests; synthetic configuration only."""
from __future__ import annotations

import copy

import pytest
import yaml

from src.preprocessing.profile import PreprocessingError, load_profile, parse_profile

# Fingerprint of configs/preprocessing.yaml, recorded as profile_fingerprint in the Stage 8
# checkpoint metadata; the ModelBundle cross-check (D59 (4)) compares against it.
PROFILE_FINGERPRINT = "5b4f5849a9bb63f9016dd79969ff5bfb9e37006acbff1e26bbfee4ee2bad68f2"


def _raw(profile_path):
    return yaml.safe_load(profile_path.read_text(encoding="utf-8"))


@pytest.fixture
def profile_path():
    from pathlib import Path
    return Path(__file__).resolve().parents[2] / "configs" / "preprocessing.yaml"


def test_load_profile_has_frozen_shapes_and_stable_fingerprint(profile_path):
    profile = load_profile(profile_path)
    assert profile.sample_rate == 16000
    assert profile.samples == 32000
    assert profile.expected_log_mel_frames == 197
    assert profile.features["fmax_hz"] == 7600.0
    assert profile.fingerprint() == load_profile(profile_path).fingerprint()
    assert len(profile.fingerprint()) == 64


def test_profile_fingerprint_is_pinned_and_parse_profile_agrees(profile_path):
    # The split of load_profile into read + parse_profile must not change the result: the file and
    # its text give the same profile and the same pinned fingerprint.
    profile = load_profile(profile_path)
    assert profile.fingerprint() == PROFILE_FINGERPRINT
    parsed = parse_profile(profile_path.read_text(encoding="utf-8"))
    assert parsed == profile
    assert parsed.fingerprint() == PROFILE_FINGERPRINT


def test_parse_profile_names_the_source_on_bad_yaml():
    with pytest.raises(PreprocessingError, match="Cannot read preprocessing profile bundle/x.yaml"):
        parse_profile("a: [unclosed", source="bundle/x.yaml")


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.update(extra=True), "unknown keys"),
        (lambda d: d.update(protocol_version="2.0"), "Protocol v1.0"),
        (lambda d: d.update(f2_ledger_sha256="0" * 64), "D28 F2"),
        (lambda d: d["waveform"].update(samples=16000), "waveform settings"),
        (lambda d: d["resampling"].update(method="kaiser_best"), "soxr_hq"),
        (lambda d: d["silence"].update(detector="silero"), "frame-RMS"),
        (lambda d: d["features"].update(n_mels=80), "feature settings"),
        (lambda d: d["features"].update(fmax_hz=8000.0), "feature settings"),
        (lambda d: d["benchmark"].update(mode="train"), "must not fit"),
    ],
)
def test_load_profile_rejects_changes(tmp_path, profile_path, mutate, message):
    data = copy.deepcopy(_raw(profile_path))
    mutate(data)
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(PreprocessingError, match=message):
        load_profile(path)


def test_profile_mappings_are_read_only(profile_path):
    profile = load_profile(profile_path)
    with pytest.raises(TypeError):
        profile.features["n_mels"] = 80
