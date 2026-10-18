"""Per-source-speaker score tables (D4, D19); no audio or model access."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.evaluation.metrics import (
    BinaryMetrics,
    ThreeWayMetrics,
    binary_metrics,
    eer,
    roc_auc,
    three_way_decisions,
    three_way_metrics,
)
from src.risk.decisions import _validate


@dataclass(frozen=True)
class SpeakerMetrics:
    """One source speaker; undefined rates are NaN, never substituted with zero."""

    source_speaker: str
    n: int
    eer: float
    roc_auc: float
    binary: BinaryMetrics
    three_way: ThreeWayMetrics


def per_speaker_metrics(labels, scores, source_speakers, *, threshold, half_width):
    """Return SpeakerMetrics rows sorted by source-speaker ID.

    Scores must use the same scale as the supplied frozen threshold and band
    (for calibrated CNN decisions, supply calibrated p). No policy is fitted.
    Each aligned input row is one evaluation unit, normally one window.
    Select the filtered or unfiltered view before calling this function.

    Speaker IDs must be nonempty strings without surrounding whitespace.
    Single-class speakers retain their decision metrics but have NaN EER/AUC.
    The EER calculation's diagnostic threshold is discarded: every speaker
    uses the SAME supplied operating threshold and uncertain band.
    """
    y, s = _validate(labels, scores, require_both_classes=False)
    speakers = np.asarray(source_speakers, dtype=object)
    if speakers.ndim != 1 or len(speakers) != len(y):
        raise ValueError("source_speakers must be 1-D and match the labels length")
    for speaker in speakers.tolist():
        if not isinstance(speaker, str) or not speaker or speaker != speaker.strip():
            raise ValueError("source_speakers must contain nonempty, trimmed string IDs")

    # Validate and apply the shared band rule once, before grouping.
    predictions = three_way_decisions(s, threshold, half_width)
    rows = []
    for speaker in sorted(set(speakers.tolist())):
        selected = speakers == speaker
        speaker_y, speaker_s = y[selected], s[selected]
        both_classes = len(np.unique(speaker_y)) == 2
        rows.append(SpeakerMetrics(
            source_speaker=speaker,
            n=int(selected.sum()),
            eer=eer(speaker_y, speaker_s)[0] if both_classes else float("nan"),
            roc_auc=roc_auc(speaker_y, speaker_s) if both_classes else float("nan"),
            binary=binary_metrics(speaker_y, speaker_s, threshold),
            three_way=three_way_metrics(speaker_y, predictions[selected]),
        ))
    return tuple(rows)
