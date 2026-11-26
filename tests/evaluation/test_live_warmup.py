import pytest

from src.evaluation.live_warmup import warmup_checkpoints
from src.evaluation.live_warmup import plan_same_label_calls


def test_call_plan_preserves_every_id_once_and_ignores_input_order():
    ids = [f"clip-{i:03d}" for i in range(47)]
    plan = plan_same_label_calls(ids)
    assert plan == plan_same_label_calls(reversed(ids))
    assert len(plan.calls) == 3
    assert all(len(call) == 15 for call in plan.calls)
    assert len(plan.leftovers) == 2
    flattened = [clip for call in plan.calls for clip in call] + list(plan.leftovers)
    assert sorted(flattened) == ids


@pytest.mark.parametrize("count", [0, 1, 14, 15, 16, 30])
def test_call_plan_never_pads_or_oversamples(count):
    plan = plan_same_label_calls([str(i) for i in range(count)])
    assert len(plan.calls) == count // 15
    assert len(plan.leftovers) == count % 15


@pytest.mark.parametrize("ids", [["a", "a"], [""], ["  "], [None], [1]])
def test_call_plan_rejects_ambiguous_ids(ids):
    with pytest.raises(ValueError):
        plan_same_label_calls(ids)


def evaluate(values):
    return warmup_checkpoints(values, threshold=0.5, half_width=0.125)


def test_silence_delays_checkpoints_but_never_enters_mean():
    rows = evaluate([None, 0.25, None, 0.25, 0.25] + [0.25] * 10)
    assert [r.elapsed_windows for r in rows] == [5, 7, 12]
    assert [r.mean_p for r in rows] == [0.25] * 3
    assert [r.prediction for r in rows] == ["authentic"] * 3


def test_mean_includes_early_windows_not_just_last_five():
    rows = evaluate([1.0] * 5 + [0.0] * 10)
    assert rows[2].mean_p == 0.5
    assert rows[2].prediction == "uncertain"


@pytest.mark.parametrize("p,label", [(0.375, "uncertain"), (0.625, "synthetic")])
def test_exact_band_boundaries(p, label):
    assert all(r.prediction == label for r in evaluate([p] * 15))


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 5, 9])
def test_unreached_checkpoints_are_explicit(count):
    rows = evaluate([0.5] * count + [None] * (15 - count))
    for row in rows:
        assert row.reached == (count >= row.kept_windows)
        if not row.reached:
            assert row.mean_p is row.prediction is row.elapsed_windows is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1, True, "0.5"])
def test_rejects_invalid_probability_even_after_last_checkpoint(bad):
    with pytest.raises(ValueError, match="each p"):
        evaluate([0.5] * 14 + [bad])


@pytest.mark.parametrize("count", [0, 14, 16])
def test_rejects_non_30_second_sequence(count):
    with pytest.raises(ValueError, match="fifteen"):
        evaluate([0.5] * count)


def test_rejects_invalid_policy():
    with pytest.raises(ValueError):
        warmup_checkpoints([0.5] * 15, threshold=0.5, half_width=-0.1)
