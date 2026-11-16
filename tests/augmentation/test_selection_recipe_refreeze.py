import pandas as pd
import pytest

from src.augmentation.selection_recipe_refreeze import compare_recipes
from src.augmentation.selection_set_recipe import CONDITIONS, RECIPE_COLUMNS


def _recipes():
    rows = []
    for condition in CONDITIONS:
        resource = "noise.wav" if condition == "recorded_noise_10db" else (
            "rir.wav" if condition == "reverberation" else ""
        )
        rows.append({
            "source_path": "FAKE/one.wav",
            "label": "FAKE",
            "subgroup": "FAKE_other",
            "group": "1",
            "source_file_sha256": "1" * 64,
            "condition": condition,
            "condition_seed": "2026",
            "resource_path": resource,
            "resource_file_sha256": "2" * 64 if resource else "",
            "resource_channel": "0" if resource else "",
            "parameters_json": "{}",
            "sample_rate_hz": "16000",
            "samples": "32000",
            "waveform_float32_sha256": "a" * 64,
        })
    old = pd.DataFrame(rows, columns=RECIPE_COLUMNS)
    new = old.copy()
    new.loc[new["condition"] == "reverberation", "waveform_float32_sha256"] = "b" * 64
    return old, new


def _inventories():
    return (
        [{"rel_path": "noise.wav", "half": "selection"}],
        [{"rel_path": "rir.wav", "half": "selection"}],
    )


def test_comparison_accepts_only_hash_changes_and_lists_old_and_new():
    old, new = _recipes()
    noise, rirs = _inventories()
    result = compare_recipes(old, new, noise_rows=noise, rir_rows=rirs)
    assert result["result"] == "passed"
    assert result["row_count"] == 5
    assert result["changed_waveform_hashes_by_condition"] == {"reverberation": 1}
    assert result["changed_waveform_hashes"][0]["old_waveform_float32_sha256"] == "a" * 64
    assert result["changed_waveform_hashes"][0]["new_waveform_float32_sha256"] == "b" * 64
    assert result["resource_halves"] == {
        "recorded_noise_10db": "selection",
        "reverberation": "selection",
    }


@pytest.mark.parametrize("column", [
    "source_path", "condition_seed", "resource_path", "resource_channel", "parameters_json",
])
def test_comparison_refuses_invariant_changes(column):
    old, new = _recipes()
    noise, rirs = _inventories()
    new.loc[0, column] = str(new.loc[0, column]) + "changed"
    with pytest.raises(ValueError, match="changed|row set"):
        compare_recipes(old, new, noise_rows=noise, rir_rows=rirs)


def test_comparison_refuses_nonselection_resource_half():
    old, new = _recipes()
    noise, rirs = _inventories()
    noise[0]["half"] = "training"
    with pytest.raises(ValueError, match="selection half"):
        compare_recipes(old, new, noise_rows=noise, rir_rows=rirs)
