"""Pinned FFmpeg boundary checks, including V1c's successful-verification cache."""
from __future__ import annotations

import pytest

from src.augmentation.codec import FFmpegCodecBackend
from src.augmentation.config import AugmentationError


def _settings(version="6.1", libraries=None):
    return {"executable": "ffmpeg", "expected_version": version,
            "expected_codec_libraries": libraries or {"libavcodec": "60.1"}}


def test_successful_pin_is_verified_once(monkeypatch):
    backend = FFmpegCodecBackend(_settings(), 16000)
    calls = []
    monkeypatch.setattr(backend, "version_text", lambda: calls.append(1) or
                        "ffmpeg version 6.1\nlibavcodec 60.1\n")
    backend.verify_pin()
    backend.verify_pin()
    assert calls == [1]


def test_failed_or_missing_pin_is_never_cached(monkeypatch):
    backend = FFmpegCodecBackend(_settings(version="wrong"), 16000)
    calls = []
    monkeypatch.setattr(backend, "version_text", lambda: calls.append(1) or
                        "ffmpeg version 6.1\nlibavcodec 60.1\n")
    for _ in range(2):
        with pytest.raises(AugmentationError, match="version mismatch"):
            backend.verify_pin()
    assert calls == [1, 1]
    with pytest.raises(AugmentationError, match="pins must be filled"):
        FFmpegCodecBackend(_settings(version=None, libraries={}), 16000).verify_pin()
