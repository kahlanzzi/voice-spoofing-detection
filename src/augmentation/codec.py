"""Pinned ffmpeg boundary for D50 codec round trips."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import soundfile as sf

from .config import AugmentationError


class FFmpegCodecBackend:
    """Round-trip audio through one of the seven D56 codec options."""

    def __init__(self, settings: Mapping[str, Any], sample_rate: int) -> None:
        self.executable = str(settings["executable"])
        self.expected_version = settings["expected_version"]
        self.expected_libraries = dict(settings["expected_codec_libraries"])
        self.sample_rate = int(sample_rate)

    def version_text(self) -> str:
        try:
            completed = subprocess.run(
                [self.executable, "-version"], capture_output=True, text=True,
                timeout=10, check=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise AugmentationError(f"cannot execute ffmpeg: {exc}") from exc
        return completed.stdout

    def verify_pin(self) -> None:
        if not self.expected_version or not self.expected_libraries:
            raise AugmentationError("ffmpeg version and codec-library pins must be filled by V1b before use")
        text = self.version_text()
        first = text.splitlines()[0] if text else ""
        if self.expected_version not in first:
            raise AugmentationError(
                f"ffmpeg version mismatch: expected {self.expected_version!r}, got {first!r}"
            )
        for library, version in self.expected_libraries.items():
            marker = f"{library} {version}"
            if marker not in text:
                raise AugmentationError(f"ffmpeg codec-library pin missing from runtime: {marker}")

    @staticmethod
    def _run(command: list[str]) -> None:
        try:
            subprocess.run(command, capture_output=True, timeout=60, check=True)
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode("utf-8", errors="replace")[-500:]
            raise AugmentationError(f"ffmpeg codec round trip failed: {detail}") from exc
        except (OSError, subprocess.SubprocessError) as exc:
            raise AugmentationError(f"ffmpeg codec round trip failed: {exc}") from exc

    def round_trip(self, waveform: np.ndarray, option: Mapping[str, Any]) -> np.ndarray:
        self.verify_pin()
        x = np.asarray(waveform, dtype=np.float32)
        if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
            raise AugmentationError("codec input must be one finite, non-empty waveform")
        kind, bitrate = option["kind"], option["bitrate_kbps"]
        with tempfile.TemporaryDirectory(prefix="vsd-codec-") as directory:
            root = Path(directory)
            source = root / "input.wav"
            encoded = root / {
                "mp3": "encoded.mp3", "opus": "encoded.opus",
                "narrowband_mulaw": "encoded.wav",
            }[kind]
            decoded = root / "decoded.wav"
            sf.write(source, x, self.sample_rate, subtype="FLOAT")
            common = [self.executable, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
            if kind == "mp3":
                encode = common + ["-i", str(source), "-ac", "1", "-ar", str(self.sample_rate),
                                   "-b:a", f"{bitrate}k", str(encoded)]
            elif kind == "opus":
                encode = common + ["-i", str(source), "-ac", "1", "-ar", str(self.sample_rate),
                                   "-c:a", "libopus", "-b:a", f"{bitrate}k", "-vbr", "off",
                                   "-application", "voip", str(encoded)]
            elif kind == "narrowband_mulaw":
                encode = common + ["-i", str(source), "-ac", "1", "-ar", "8000",
                                   "-c:a", "pcm_mulaw", str(encoded)]
            else:
                raise AugmentationError(f"unsupported codec kind: {kind!r}")
            decode = common + ["-i", str(encoded), "-ac", "1", "-ar", str(self.sample_rate),
                               "-c:a", "pcm_f32le", str(decoded)]
            self._run(encode)
            self._run(decode)
            result, sample_rate = sf.read(decoded, dtype="float32", always_2d=False)
        if sample_rate != self.sample_rate or np.asarray(result).ndim != 1:
            raise AugmentationError("codec output has the wrong sample rate or channel count")
        result = np.asarray(result, dtype=np.float32)
        if result.size < x.size:
            result = np.pad(result, (0, x.size - result.size))
        result = result[:x.size]
        if not np.isfinite(result).all():
            raise AugmentationError("codec output contains NaN or infinity")
        return result
