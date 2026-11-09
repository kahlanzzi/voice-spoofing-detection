"""The analyzer seam (docs/api_design.md §8): the only place the API calls model code.

analyze(path) returns the verdict part of the §11 response; info() returns what GET /api/status
reports. The API adds request_id and notice. MockAnalyzer is the stand-in until the frozen
ModelBundle (Stage 12): it never imports torch or src.models.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

MOCK_VERSIONS = {"model_version": "mock", "profile_version": "preprocessing-v1", "policy_version": "risk-v1"}
CHUNK_BYTES = 64 * 1024


@dataclass(frozen=True)
class AnalysisOutcome:
    prediction: str
    confidence: float
    risk_level: str
    model_version: str
    profile_version: str
    policy_version: str


@dataclass(frozen=True)
class AnalyzerInfo:
    ready: bool
    model_version: str
    profile_version: str
    policy_version: str
    deferral_on_fit: float
    bundle_manifest_sha256: str | None
    mock: bool


class AnalyzerUnavailable(Exception):
    """The analyzer is not loaded or refuses work: 503 analysis_unavailable."""


class Analyzer(Protocol):
    def analyze(self, path: Path) -> AnalysisOutcome: ...

    def info(self) -> AnalyzerInfo: ...


def file_sha256(path: Path) -> bytes:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.digest()


class MockAnalyzer:
    """Deterministic mock: the SHA-256 of the file bytes picks the outcome, so the same file always
    gives the same result. Outcomes follow risk-v1 shapes: synthetic 1.00 HIGH, uncertain
    0.50-1.00 MEDIUM, authentic 0.73-1.00 LOW."""

    def __init__(self, deferral_on_fit: float):
        self.deferral_on_fit = deferral_on_fit

    def analyze(self, path: Path) -> AnalysisOutcome:
        digest = file_sha256(path)
        u = int.from_bytes(digest[1:3], "big") / 0xFFFF  # in [0, 1]
        verdict = digest[0] % 3
        if verdict == 0:
            prediction, confidence, risk_level = "synthetic", 1.0, "HIGH"
        elif verdict == 1:
            prediction, confidence, risk_level = "uncertain", round(0.50 + 0.50 * u, 2), "MEDIUM"
        else:
            prediction, confidence, risk_level = "authentic", round(0.73 + 0.27 * u, 2), "LOW"
        return AnalysisOutcome(prediction=prediction, confidence=confidence, risk_level=risk_level,
                               **MOCK_VERSIONS)

    def info(self) -> AnalyzerInfo:
        return AnalyzerInfo(ready=True, deferral_on_fit=self.deferral_on_fit, bundle_manifest_sha256=None,
                            mock=True, **MOCK_VERSIONS)
