import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.evaluation import live_warmup_scoring as scoring
from src.evaluation.live_warmup_plan import export_call_plan


@pytest.fixture
def setup_run(tmp_path, monkeypatch):
    manifest = Path(__file__).resolve().parents[2] / 'data/manifests/for2sec_frozen_manifest.csv'
    path = tmp_path / 'plan.json'
    digest = export_call_plan(manifest, path, code_commit='a' * 40)
    events = []
    profile = SimpleNamespace(silence={'external_min_speech_fraction': 0.5})
    def transform(values):
        events.append('calibrate')
        assert values == [2.0]
        return [0.8]
    loaded = SimpleNamespace(bundle=SimpleNamespace(manifest_sha256='b' * 64,
        profile=profile, policy=SimpleNamespace(threshold=0.5, half_width=0.1,
        calibration=SimpleNamespace(transform=transform)),
        manifest=SimpleNamespace(components={})), architecture='cnn')
    monkeypatch.setitem(sys.modules, 'src.bundle.model_loader',
        SimpleNamespace(load_bundle_model=lambda _: loaded))
    def read(root, row):
        events.append('audio')
        return np.zeros(32000, dtype=np.float32), 16000
    monkeypatch.setattr(scoring, 'read_verified_audio', read)
    monkeypatch.setattr(scoring, 'preprocess_waveform', lambda w, r, p:
        SimpleNamespace(waveform=w, speech_fraction=0.5))
    monkeypatch.setattr(scoring, 'frozen_logit', lambda l, w: 2.0)
    kwargs = dict(expected_plan_sha256=digest, expected_bundle_sha256='b' * 64,
                  code_commit='c' * 40)
    def run():
        return scoring.score_call_plan(path, manifest, tmp_path, tmp_path, **kwargs)
    return run, kwargs, loaded, events


def test_scores_exact_membership_and_calibrates_once(setup_run):
    run, _, _, events = setup_run
    result = run()
    assert len(result['slots']) == 1320
    assert events.count('audio') == events.count('calibrate') == 1320
    assert all(slot['p'] == 0.8 and slot['kept'] for slot in result['slots'])
    assert result['provenance']['code_commit'] == 'c' * 40


@pytest.mark.parametrize('field', ['expected_plan_sha256', 'expected_bundle_sha256', 'code_commit'])
def test_bad_pin_stops_before_audio(setup_run, field):
    run, kwargs, _, events = setup_run
    kwargs[field] = '0' * (40 if field == 'code_commit' else 64) if field != 'code_commit' else 'bad'
    with pytest.raises(ValueError):
        run()
    assert events == []


def test_no_speech_never_scores_or_calibrates(setup_run, monkeypatch):
    run, _, _, events = setup_run
    monkeypatch.setattr(scoring, 'preprocess_waveform', lambda w, r, p:
        SimpleNamespace(waveform=w, speech_fraction=0.49))
    monkeypatch.setattr(scoring, 'frozen_logit', lambda *a: pytest.fail('no-speech scored'))
    result = run()
    assert events.count('calibrate') == 0
    assert all(slot['p'] is None for slot in result['slots'])
    assert all(row['reached'] == 0 for row in result['checkpoints'])


def test_wrong_duration_fails(setup_run, monkeypatch):
    run, _, _, _ = setup_run
    monkeypatch.setattr(scoring, 'read_verified_audio', lambda *a: (np.zeros(31999), 16000))
    with pytest.raises(ValueError, match='two seconds'):
        run()


def test_bad_calibration_fails(setup_run):
    run, _, loaded, _ = setup_run
    loaded.bundle.policy.calibration.transform = lambda _: [float('nan')]
    with pytest.raises(ValueError, match='calibration'):
        run()

@pytest.mark.parametrize('architecture', ['cnn', 'cnn-aug', 'crnn', 'aasist'])
def test_frozen_logit_uses_architecture_input_without_calibration(architecture, monkeypatch):
    from tests._torch import require_torch
    torch = require_torch()
    from src.preprocessing import features
    from src.training import normalization
    observed = []
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
        def forward(self, value):
            observed.append(tuple(value.shape))
            return torch.tensor([2.0])
    model = Model().eval()
    monkeypatch.setattr(features, 'log_mel_spectrogram', lambda w, p: np.zeros((1, 64, 197), dtype=np.float32))
    monkeypatch.setattr(normalization, 'apply_bin_stats', lambda x, m, s: x)
    loaded = SimpleNamespace(model=model, architecture=architecture,
        bundle=SimpleNamespace(profile=object()), mean=None, std=None)
    assert scoring.frozen_logit(loaded, np.zeros(32000, dtype=np.float32)) == 2.0
    assert observed == [(1, 32000) if architecture == 'aasist' else (1, 1, 64, 197)]
    model.train()
    with pytest.raises(ValueError, match='eval mode'):
        scoring.frozen_logit(loaded, np.zeros(32000, dtype=np.float32))
