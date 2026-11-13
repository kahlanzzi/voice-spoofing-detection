from pathlib import Path

import pytest
import yaml

from src.augmentation import AugmentationError, load_augmentation_config


CONFIG = Path(__file__).resolve().parents[2] / "configs" / "augmentation.yaml"


def test_d50_d56_config_loads_with_incomplete_v1b_pins():
    config = load_augmentation_config(CONFIG)
    assert config.families == ("recorded_noise", "reverberation", "codec", "gain", "rawboost")
    assert [item["id"] for item in config.codec_options] == [
        "mp3_24", "mp3_48", "mp3_96", "opus_8", "opus_16", "opus_24",
        "narrowband_mulaw",
    ]
    assert not config.corpus_pins_complete
    assert not config.ffmpeg_pin_complete
    assert config.rawboost["algorithm"] == 1


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda c: c["training"].update({"apply_probability": 0.9}), "training policy"),
        (lambda c: c["training"]["families"].append("white_gaussian"), "training policy"),
        (lambda c: c["codec"]["options"][0].update({"bitrate_kbps": 32}), "exact seven"),
        (lambda c: c["rawboost"].update({"algorithm": 3}), "algorithm 1"),
        (lambda c: c["held_out_from_training"].update({"plain_low_pass": False}), "held-out"),
    ],
)
def test_governed_values_cannot_drift(tmp_path, mutate, message):
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "augmentation.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(AugmentationError, match=message):
        load_augmentation_config(path)
