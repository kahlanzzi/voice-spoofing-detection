"""Model v2 candidate C2 `crnn` (D49): shapes, layers, refusals, determinism. Synthetic input only."""
from __future__ import annotations

import pytest

from tests._torch import require_torch

torch = require_torch()
from torch import nn  # noqa: E402

from src.models.crnn import (  # noqa: E402
    CHANNELS,
    DROPOUT,
    FEATURE_MAP,
    INPUT_SHAPE,
    LSTM_HIDDEN,
    LSTM_INPUT,
    build_crnn,
)
from src.models.v2_checkpoint import trainable_parameters  # noqa: E402

CRNN_PARAMETERS = 319_585  # pinned: D49's tie rule compares trainable parameters (Entry 170)


def test_constants_match_d49():
    # D49 (C2): channels 16/32/64, 64 LSTM units per direction, dropout 0.3, the D39 log-Mel input.
    assert CHANNELS == (16, 32, 64)
    assert LSTM_HIDDEN == 64 and DROPOUT == 0.3
    assert INPUT_SHAPE == (1, 64, 197)
    # 64 Mel rows / 2^3 = 8, 197 frames // 2 // 2 // 2 = 24 (floor pooling); 64 channels x 8 rows.
    assert FEATURE_MAP == (64, 8, 24) and LSTM_INPUT == 512


@pytest.mark.parametrize("n", [1, 3])
def test_shapes_through_the_network(n):
    # The intermediate map, the LSTM input and the output are the shapes D49 describes,
    # for a batch of 1 (the API) and a batch of 3, in eval mode.
    torch.manual_seed(0)
    model = build_crnn().eval()
    x = torch.randn(n, *INPUT_SHAPE)
    with torch.no_grad():
        assert tuple(model.features(x).shape) == (n, *FEATURE_MAP)
        seq = model.sequence(x)
        assert tuple(seq.shape) == (n, 24, 512)
        out, _ = model.lstm(seq)
        assert tuple(out.shape) == (n, 24, 128)
        logits = model(x)
    assert tuple(logits.shape) == (n,) and logits.dtype == torch.float32


def test_sequence_flattens_channel_and_frequency_per_time_step():
    # Frame t of the sequence must be exactly the (channel, Mel) slice of the conv map at time t,
    # channel-major, so time is never mixed into the LSTM features.
    torch.manual_seed(0)
    model = build_crnn().eval()
    x = torch.randn(2, *INPUT_SHAPE)
    with torch.no_grad():
        z = model.features(x)
        seq = model.sequence(x)
    for t in (0, 11, 23):
        assert torch.equal(seq[:, t, :], z[:, :, :, t].reshape(2, -1))


def test_architecture_matches_d49():
    # Three D42 blocks (Conv 3x3 pad 1 -> BN -> ReLU -> MaxPool 2), one BiLSTM layer, Linear(128, 1).
    model = build_crnn()
    convs = [m for m in model.features if isinstance(m, nn.Conv2d)]
    assert [c.out_channels for c in convs] == list(CHANNELS)
    assert all(c.kernel_size == (3, 3) and c.padding == (1, 1) for c in convs)
    kinds = [type(m) for m in model.features]
    assert kinds == [nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d] * 3
    lstm = model.lstm
    assert (lstm.input_size, lstm.hidden_size, lstm.num_layers) == (512, 64, 1)
    assert lstm.bidirectional and lstm.batch_first
    assert model.dropout.p == DROPOUT
    assert (model.head.in_features, model.head.out_features) == (128, 1)


def test_trainable_parameter_count_is_pinned():
    assert trainable_parameters(build_crnn()) == CRNN_PARAMETERS


@pytest.mark.parametrize("shape", [(2, 64, 197), (2, 1, 64, 196), (2, 1, 63, 197), (2, 2, 64, 197)])
def test_wrong_input_shape_raises(shape):
    # A wrong shape would silently change the LSTM length or width; refuse it.
    with pytest.raises(ValueError, match="input must have shape"):
        build_crnn()(torch.zeros(shape))


def test_same_seed_same_weights_and_eval_outputs():
    # Determinism (D49 seeds): same seed -> identical initial weights and identical eval logits.
    x = torch.randn(3, *INPUT_SHAPE, generator=torch.Generator().manual_seed(7))
    outs, states = [], []
    for _ in range(2):
        torch.manual_seed(2026)
        model = build_crnn().eval()
        states.append(model.state_dict())
        with torch.no_grad():
            outs.append(model(x))
    assert all(torch.equal(states[0][k], states[1][k]) for k in states[0])
    assert torch.equal(outs[0], outs[1])
