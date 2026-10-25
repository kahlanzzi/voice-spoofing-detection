"""The D42 CNN: architecture, output shape and seeded initialization. Synthetic tensors only."""
from __future__ import annotations

import pytest

from src.training.determinism import set_seed
from tests._torch import require_torch

torch = require_torch()
from torch import nn  # noqa: E402

from src.models.cnn import CHANNELS, DROPOUT, INPUT_SHAPE, build_cnn  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_torch_flags():
    det = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_det, cudnn_bench = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
    yield
    torch.use_deterministic_algorithms(det, warn_only=warn)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn_det, cudnn_bench


def test_constants_match_d42():
    assert CHANNELS == (16, 32, 64, 128)
    assert DROPOUT == 0.3
    assert INPUT_SHAPE == (1, 64, 197)


@pytest.mark.parametrize("n_blocks", [3, 4])
def test_output_is_one_logit_per_clip(n_blocks):
    model = build_cnn(n_blocks).eval()
    with torch.no_grad():
        out = model(torch.randn(5, *INPUT_SHAPE))
    assert out.shape == (5,)
    assert out.dtype == torch.float32
    with torch.no_grad():
        assert model(torch.randn(1, *INPUT_SHAPE)).shape == (1,)


@pytest.mark.parametrize("bad", [2, 5, 0, True, "3", 3.0, None])
def test_only_3_or_4_blocks(bad):
    with pytest.raises(ValueError, match="n_blocks"):
        build_cnn(bad)


@pytest.mark.parametrize("shape", [(2, 64, 197), (2, 1, 64, 196), (2, 1, 32, 197), (2, 3, 64, 197)])
def test_wrong_input_shape_raises(shape):
    with pytest.raises(ValueError, match="input must have shape"):
        build_cnn(3)(torch.zeros(shape))


@pytest.mark.parametrize("n_blocks", [3, 4])
def test_architecture_matches_d42(n_blocks):
    model = build_cnn(n_blocks)
    layers = list(model.features)
    assert len(layers) == 4 * n_blocks
    in_ch = 1
    for i, out_ch in enumerate(CHANNELS[:n_blocks]):
        conv, bn, relu, pool = layers[4 * i: 4 * i + 4]
        assert isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d)
        assert isinstance(relu, nn.ReLU) and isinstance(pool, nn.MaxPool2d)
        assert (conv.in_channels, conv.out_channels) == (in_ch, out_ch)
        assert conv.kernel_size == (3, 3) and conv.padding == (1, 1) and conv.stride == (1, 1)
        assert bn.num_features == out_ch
        assert pool.kernel_size == 2
        in_ch = out_ch
    convs = [m.out_channels for m in model.modules() if isinstance(m, nn.Conv2d)]
    assert convs == list(CHANNELS[:n_blocks])
    assert isinstance(model.pool, nn.AdaptiveAvgPool2d) and model.pool.output_size == 1
    assert isinstance(model.dropout, nn.Dropout) and model.dropout.p == 0.3
    assert (model.head.in_features, model.head.out_features) == (CHANNELS[n_blocks - 1], 1)


@pytest.mark.parametrize("n_blocks", [3, 4])
def test_same_seed_same_initial_weights(n_blocks):
    def init(seed):
        set_seed(seed)
        return build_cnn(n_blocks).state_dict()

    a, b, c = init(0), init(0), init(1)
    assert a.keys() == b.keys() == c.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)
    # Conv and linear weights are drawn at random, so a different seed changes every one of them
    # (BatchNorm weights start at 1 whatever the seed).
    random_weights = [k for k in a if k.endswith("weight") and a[k].ndim > 1]
    assert random_weights
    assert all(not torch.equal(a[k], c[k]) for k in random_weights)
