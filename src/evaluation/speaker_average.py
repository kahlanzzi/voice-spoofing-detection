"""Equal-weight summaries of speaker metric rows (D4, D19)."""
from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite, isnan
from numbers import Real

from src.evaluation.speakers import SpeakerMetrics


@dataclass(frozen=True)
class SpeakerMean:
    """A macro mean with explicit missingness; counts are numbers of speakers."""

    value: float
    n_speakers: int
    n_defined: int
    undefined_speakers: tuple[str, ...]


def speaker_average(rows, *, undefined_policy: str) -> dict[str, SpeakerMean]:
    """Average rates equally across distinct source speakers, never by window count.

    Caller MUST choose 'propagate' (NaN if any speaker is undefined) or 'raise'
    (reject such a summary). There is no default or silent exclusion of speakers.
    The external report's policy still needs team review before the real run.

    Input is the output of per_speaker_metrics for one common policy and view.
    Rates use dotted names, e.g. 'binary.accuracy' and 'non_deferred.f1'. Counts
    and confusion matrices remain in the individual rows and are not averaged.
    """
    if undefined_policy not in ("propagate", "raise"):
        raise ValueError("undefined_policy must be 'propagate' or 'raise'")
    try:
        rows = tuple(rows)
    except TypeError as exc:
        raise ValueError("rows must be an iterable of SpeakerMetrics") from exc
    if not rows or any(not isinstance(row, SpeakerMetrics) for row in rows):
        raise ValueError("rows must contain at least one SpeakerMetrics row")
    ids = [row.source_speaker for row in rows]
    if any(not isinstance(s, str) or not s or s != s.strip() for s in ids):
        raise ValueError("speaker IDs must be nonempty trimmed strings")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate source speaker: provide one row per speaker")
    rows = sorted(rows, key=lambda row: row.source_speaker)
    rates = ("accuracy", "precision", "recall", "f1", "fpr", "fnr")
    columns = {name: [getattr(row, name) for row in rows] for name in ("eer", "roc_auc")}
    for group in ("binary", "non_deferred"):
        for name in rates:
            columns[f"{group}.{name}"] = [
                getattr(row.binary if group == "binary" else row.three_way.non_deferred, name)
                for row in rows
            ]
    for name in ("coverage", "deferral_rate", "deferral_rate_bonafide", "deferral_rate_spoof"):
        columns[name] = [getattr(row.three_way, name) for row in rows]

    result = {}
    for name, values in columns.items():
        if any(isinstance(v, bool) or not isinstance(v, Real) or
               (not isnan(v) and (not isfinite(v) or not 0 <= v <= 1)) for v in values):
            raise ValueError(f"{name}: rates must be in [0, 1] or NaN")
        missing = tuple(row.source_speaker for row, v in zip(rows, values) if isnan(v))
        if missing and undefined_policy == "raise":
            raise ValueError(f"{name} is undefined for speakers {missing}")
        result[name] = SpeakerMean(
            value=float("nan") if missing else fsum(values) / len(rows),
            n_speakers=len(rows), n_defined=len(rows) - len(missing),
            undefined_speakers=missing,
        )
    return result
