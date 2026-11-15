from pathlib import Path

import pytest
import yaml

from src.augmentation import AugmentationError, load_augmentation_config


CONFIG = Path(__file__).resolve().parents[2] / "configs" / "augmentation.yaml"


def test_d50_d56_config_loads_with_completed_v1b_pins():
    config = load_augmentation_config(CONFIG)
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config.families == ("recorded_noise", "reverberation", "codec", "gain", "rawboost")
    assert [item["id"] for item in config.codec_options] == [
        "mp3_24", "mp3_48", "mp3_96", "opus_8", "opus_16", "opus_24",
        "narrowband_mulaw",
    ]
    assert config.corpus_pins_complete
    assert config.ffmpeg_pin_complete
    assert raw["corpora"]["noise"]["archive_sha256"] == "86d1061c7e15b5c9e906777685c519701df51bfde3001e1070dcc9ffac955ee1"
    assert raw["corpora"]["rir"]["archive_sha256"] == "3b50cfde915b3984738169b4beb341e9f6b8062ae4c2076146c5db71c2c05dc7"
    assert raw["ffmpeg"]["expected_version"] == "6.1.1-3ubuntu5"
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
