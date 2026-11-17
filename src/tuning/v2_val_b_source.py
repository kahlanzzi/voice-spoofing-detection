"""F3 waveform val-B source for an aasist risk-v2 fit (PR V4c; Execution Plan v3.0 §9 V4; D45, D46 (7)).

v2_risk_run.run takes, for an aasist winner, a ValBWaveformSource: () -> (waveforms float32
(n, 32000), labels int8 (n,), paths unicode (n,)). val_b_waveform_source builds it. Called, the
source reads exactly the F2 val-B rows, sorted by path, through stage11_v2_run's own in-memory
input pass (compute_inputs): one read_verified_audio + extract_features call per clip, keeping
prepared.waveform, computed twice and refused if the two passes' hashes (the waveform hash is
v2_risk_run._sha256_arrays) differ. Labels and paths come from stage11_v2_run._rows_arrays. The
arrays are therefore the ones stage11_v2_run's rehearsal (part (c)) recomputes for aasist, so its
val-B content-hash check against risk-v2's val_b_input.content_sha256 can pass.

Why a module of its own: stage11_v2_run imports v2_risk_run, so v2_risk_run cannot import it at
module level (cycle); its main imports this module when the winner is aasist. Importing the
helpers, not copying them, keeps one definition of the val-B input for risk-v2 and Stage 11.
stage11_v2_run is one of the two files the FoR test-access guard allows; this module only imports
three helpers and two split-name constants from it, none of which is the test split (the guard's
import rule is about stage11_run.SPLIT only).

val-B only: no split argument, no other split, no test path. Nothing is cached or written to disk
(in memory, as D46 (7) for Stage 11). Nothing is fitted here.
"""
from __future__ import annotations

from pathlib import Path

from src.evaluation.stage11_v2_run import FIT_SPLIT, WAVEFORM, _rows_arrays, compute_inputs, split_rows
from src.preprocessing.profile import load_profile
from src.preprocessing.validation import load_f2_manifest
from src.tuning.v2_risk_run import ValBWaveformSource


def val_b_waveform_source(root, manifest_path, profile_path) -> ValBWaveformSource:
    """The aasist val-B source for v2_risk_run.run. Nothing is read until it is called."""
    root, manifest_path, profile_path = Path(root), Path(manifest_path), Path(profile_path)

    def source():
        manifest = load_f2_manifest(manifest_path)
        profile = load_profile(profile_path)
        rows = split_rows(manifest, FIT_SPLIT)  # the F2 val-B rows, sorted by path
        waveforms, _ = compute_inputs(root, rows, profile, WAVEFORM)  # two passes, equal hashes
        labels, paths = _rows_arrays(rows)
        return waveforms, labels, paths

    return source
