"""Integration tests using invented scores; no dataset paths, audio or model calls."""
import numpy as np
import pytest

from src.evaluation.bootstrap import bootstrap_ci
from src.evaluation.speaker_average import speaker_average
from src.evaluation.speakers import per_speaker_metrics


REAL_COUNTS = (50, 80, 120, 170, 230, 300, 400, 520)
ERROR_FRACTIONS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
IDS = tuple(f"synthetic-{i}" for i in range(8))
THRESHOLD, HALF_WIDTH = 0.5, 0.1  # Invented test policy, never a fitted policy.


def make_synthetic_windows(seed=2026):
    """Return aligned label/score/speaker/keep arrays with known error counts."""
    rng = np.random.default_rng(seed)
    labels, scores, speakers, keep = [], [], [], []
    for speaker, n_real, error_fraction in zip(IDS, REAL_COUNTS, ERROR_FRACTIONS):
        for label, n in ((0, n_real), (1, 7 * n_real)):
            n_wrong = int(n * error_fraction)
            # Fixed numbers of wrong and uncertain predictions yield a counting oracle.
            predicted = np.full(n, label)
            predicted[:n_wrong] = 1 - label
            values = np.where(predicted == 1, rng.uniform(0.75, 0.95, n),
                              rng.uniform(0.05, 0.25, n))
            values[n_wrong:n_wrong + n // 10] = 0.55 if label else 0.45
            selected = np.ones(n, dtype=bool)
            selected[::10] = False  # Invented filter flags, not a speech detector.
            labels.extend([label] * n)
            scores.extend(values)
            speakers.extend([speaker] * n)
            keep.extend(selected)
    order = rng.permutation(len(labels))
    return tuple(np.asarray(column)[order] for column in (labels, scores, speakers, keep))


@pytest.fixture(scope="module")
def windows():
    return make_synthetic_windows()


def test_synthetic_shape_and_seed(windows):
    y, p, speakers, keep = windows
    assert len(y) == 14960
    assert np.bincount(y).tolist() == [1870, 13090]
    assert np.mean(y) == 0.875
    assert np.all(np.isfinite(p) & (p >= 0) & (p <= 1))
    assert keep.dtype == np.bool_
    for sid, count in zip(IDS, REAL_COUNTS):
        assert np.bincount(y[speakers == sid]).tolist() == [count, 7 * count]
    for original, repeated in zip(windows, make_synthetic_windows()):
        np.testing.assert_array_equal(original, repeated)
    assert not np.array_equal(p, make_synthetic_windows(2027)[1])


def test_large_table_matches_known_errors_and_deferrals(windows):
    y, p, speakers, _ = windows
    rows = per_speaker_metrics(y, p, speakers, threshold=THRESHOLD, half_width=HALF_WIDTH)
    assert tuple(row.source_speaker for row in rows) == IDS
    for row, n0, error in zip(rows, REAL_COUNTS, ERROR_FRACTIONS):
        n1 = 7 * n0
        fp, fn = int(n0 * error), int(n1 * error)
        d0, d1 = n0 // 10, n1 // 10
        assert (row.binary.fp, row.binary.fn) == (fp, fn)
        assert row.binary.accuracy == pytest.approx(1 - (fp + fn) / (n0 + n1))
        assert row.three_way.confusion == ((n0 - fp - d0, fp, d0),
                                           (fn, n1 - fn - d1, d1))
        assert row.three_way.n_deferred == d0 + d1
        assert 0 <= row.eer <= 1 and 0 <= row.roc_auc <= 1
    assert rows[0].eer == 0 and rows[0].roc_auc == 1
    assert rows[-1].binary.accuracy == 0.5


def test_macro_mean_is_not_window_weighted(windows):
    y, p, speakers, _ = windows
    rows = per_speaker_metrics(y, p, speakers, threshold=THRESHOLD, half_width=HALF_WIDTH)
    # Explicit development option already supported by the API, not approval of D33.
    summary = speaker_average(rows, undefined_policy="raise")
    expected = [1 - (int(n * e) + int(7 * n * e)) / (8 * n)
                for n, e in zip(REAL_COUNTS, ERROR_FRACTIONS)]
    mean = summary["binary.accuracy"]
    assert mean.value == pytest.approx(sum(expected) / 8)
    assert (mean.n_speakers, mean.n_defined, mean.undefined_speakers) == (8, 8, ())
    assert abs(mean.value - np.mean((p >= THRESHOLD) == y)) > 0.1


def test_filter_counts_and_filtered_table_are_consistent(windows):
    y, p, speakers, keep = windows
    rows = per_speaker_metrics(y[keep], p[keep], speakers[keep],
                               threshold=THRESHOLD, half_width=HALF_WIDTH)
    assert tuple(row.source_speaker for row in rows) == IDS
    assert sum(row.n for row in rows) == int(keep.sum())
    for row, sid, n0 in zip(rows, IDS, REAL_COUNTS):
        for label, expected_total in ((0, n0), (1, 7 * n0)):
            group = (speakers == sid) & (y == label)
            dropped = int((group & ~keep).sum())
            retained = int((group & keep).sum())
            assert dropped == (expected_total + 9) // 10
            assert retained + dropped == expected_total
        selected = (speakers == sid) & keep
        # Filter removal is separate from uncertain predictions on retained windows.
        expected_deferred = int(((p[selected] >= 0.4) & (p[selected] < 0.6)).sum())
        assert row.three_way.n_deferred == expected_deferred


def test_missing_speaker_is_visible_in_roster_comparison(windows):
    y, p, speakers, keep = windows
    selected = keep & (speakers != IDS[-1])
    rows = per_speaker_metrics(y[selected], p[selected], speakers[selected],
                               threshold=THRESHOLD, half_width=HALF_WIDTH)
    # The existing helper only reports observed speakers. A future reporting layer must
    # check its expected roster; this test does not pretend the helper enforces eight.
    missing = set(IDS) - {row.source_speaker for row in rows}
    assert missing == {IDS[-1]}
    assert len(rows) == 7


def test_clustered_accuracy_interval_matches_speaker_count_oracle(windows):
    y, p, speakers, _ = windows
    correct = (p >= THRESHOLD) == y
    result = bootstrap_ci(lambda indices: float(correct[indices].mean()), len(y),
                          clusters=speakers, seed=2026)
    # Independent count-level oracle: repeated speaker draws retain their full sizes.
    # It catches row resampling and collapsing duplicate cluster draws.
    ordered_ids = list(dict.fromkeys(speakers.tolist()))
    sizes = np.array([8 * REAL_COUNTS[IDS.index(sid)] for sid in ordered_ids])
    successes = np.array([
        8 * REAL_COUNTS[IDS.index(sid)]
        - int(REAL_COUNTS[IDS.index(sid)] * ERROR_FRACTIONS[IDS.index(sid)])
        - int(7 * REAL_COUNTS[IDS.index(sid)] * ERROR_FRACTIONS[IDS.index(sid)])
        for sid in ordered_ids
    ])
    draws = np.random.default_rng(2026).integers(0, 8, size=(2000, 8))
    oracle = successes[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    low, high = np.percentile(oracle, [2.5, 97.5], method="linear")
    assert result.estimate == pytest.approx(successes.sum() / sizes.sum())
    assert (result.low, result.high) == pytest.approx((low, high))
    assert (result.n_resamples, result.n_invalid, result.confidence) == (2000, 0, 0.95)
