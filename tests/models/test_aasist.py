"""Model v2 candidate C3 `aasist` (D49, D61): shapes, D61 changes, faithfulness to the reference.

The faithfulness test needs the reference repository cloned outside this repo at the pinned commit
(clovaai/aasist a04c9863f63d44471dde8a6abcb3b082b07cd1d1) and the env var AASIST_REF_DIR pointing to
it; otherwise it is skipped with the reason. Synthetic input only.
"""
from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path

import pytest

from tests._torch import require_torch

torch = require_torch()

from src.models.aasist import MODEL_CONFIG, NB_SAMP, build_aasist  # noqa: E402
from src.models.v2_checkpoint import trainable_parameters  # noqa: E402

AASIST_PARAMETERS = 297_705  # pinned: D49's tie rule compares trainable parameters (Entry 170)
REFERENCE_COMMIT = "a04c9863f63d44471dde8a6abcb3b082b07cd1d1"
# Reference config/AASIST.conf model_config at the pinned commit, written out by hand.
REFERENCE_CONFIG = {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}
# Ported and reference encoder/GAT outputs are compared with this tolerance. The port runs the
# same float32 operations in the same order, so they should agree bit for bit; the tolerance only
# absorbs kernel-level nondeterminism on other hardware.
ATOL = 1e-6
RTOL = 1e-5
SRC_FILE = Path(__file__).resolve().parents[2] / "src" / "models" / "aasist.py"


def _waveform(n: int, length: int = NB_SAMP, seed: int = 7):
    return torch.randn(n, length, generator=torch.Generator().manual_seed(seed)) * 0.1


def test_config_is_the_reference_with_nb_samp_32000():
    # D61 (1): reference model_config unchanged except nb_samp (D49: 2 s at 16 kHz).
    assert NB_SAMP == 32000
    assert MODEL_CONFIG == {**REFERENCE_CONFIG, "nb_samp": 32000}


@pytest.mark.parametrize("n", [1, 3])
def test_output_is_one_logit_per_clip(n):
    # D61 (2): forward returns (n,) logits only, for a batch of 1 (the API) and of 3, eval mode.
    torch.manual_seed(0)
    model = build_aasist().eval()
    with torch.no_grad():
        out = model(_waveform(n))
    assert isinstance(out, torch.Tensor)  # not the reference (embedding, output) tuple
    assert tuple(out.shape) == (n,) and out.dtype == torch.float32


def test_out_layer_is_one_logit():
    # D61 (2): Linear(5 * gat_dims[1], 1) = Linear(160, 1).
    layer = build_aasist().out_layer
    assert (layer.in_features, layer.out_features) == (160, 1)


@pytest.mark.parametrize("x", [
    torch.zeros(2, 64600),            # reference length: refused, D49 fixes 32,000
    torch.zeros(2, 31999),
    torch.zeros(2, 1, 32000),         # channel axis: the reference adds it itself
    torch.zeros(32000),
    torch.zeros(2, 32000, dtype=torch.float64),
])
def test_wrong_input_shape_or_dtype_raises(x):
    with pytest.raises(ValueError, match="float32 with shape"):
        build_aasist()(x)


def test_forward_takes_no_freq_aug_and_conv_has_no_mask():
    # D61 (4): the Freq_aug option and the frequency-mask path are removed, not just switched off.
    model = build_aasist()
    with pytest.raises(TypeError):
        model(_waveform(1), Freq_aug=True)
    with pytest.raises(TypeError):
        model.conv_time(_waveform(1).unsqueeze(1), mask=True)
    tree = ast.parse(SRC_FILE.read_text(encoding="utf-8"))
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert "random" not in imported  # the mask path was the only user of `random`


def test_trainable_parameter_count_is_pinned():
    # 297,866 in the reference with 2 outputs; one logit removes 160 weights + 1 bias.
    assert trainable_parameters(build_aasist()) == AASIST_PARAMETERS


def test_same_seed_same_weights_and_eval_outputs():
    # Determinism: same seed -> identical initial weights and identical eval logits.
    x = _waveform(2)
    outs, states = [], []
    for _ in range(2):
        torch.manual_seed(2026)
        model = build_aasist().eval()
        states.append(model.state_dict())
        with torch.no_grad():
            outs.append(model(x))
    assert all(torch.equal(states[0][k], states[1][k]) for k in states[0])
    assert torch.equal(outs[0], outs[1])


def test_build_does_not_share_the_module_config():
    # build_aasist copies MODEL_CONFIG, so a model cannot change the config of the next one.
    model = build_aasist()
    model.d_args["filts"][1][1] = 999
    assert MODEL_CONFIG["filts"][1] == [1, 32]


# Faithfulness to the reference -----------------------------------------------------------------

def _reference_module():
    ref_dir = os.environ.get("AASIST_REF_DIR")
    if not ref_dir:
        pytest.skip("AASIST_REF_DIR not set: clone clovaai/aasist at "
                    f"{REFERENCE_COMMIT} outside the repo and point AASIST_REF_DIR to it")
    path = Path(ref_dir) / "models" / "AASIST.py"
    if not path.is_file():
        pytest.skip(f"AASIST_REF_DIR={ref_dir} has no models/AASIST.py")
    head = Path(ref_dir) / ".git" / "HEAD"
    if head.is_file() and head.read_text(encoding="ascii").strip() != REFERENCE_COMMIT:
        pytest.fail(f"{ref_dir} must be checked out at {REFERENCE_COMMIT} (detached HEAD)")
    spec = importlib.util.spec_from_file_location("aasist_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_port_matches_reference_before_the_final_layer():
    # Same weights, same input -> the 160-d embedding before out_layer (the encoder and GAT part)
    # must equal the reference's last_hidden; only out_layer differs (D61 (2)), so it is skipped.
    ref_module = _reference_module()
    torch.manual_seed(2026)
    reference = ref_module.Model(dict(REFERENCE_CONFIG))
    # Non-default BatchNorm running statistics, so the comparison covers the buffers too.
    reference.train()
    with torch.no_grad():
        for seed in (1, 2):
            reference(_waveform(4, seed=seed))
    reference.eval()

    port = build_aasist().eval()
    ref_state = {k: v for k, v in reference.state_dict().items() if not k.startswith("out_layer.")}
    result = port.load_state_dict(ref_state, strict=False)
    assert sorted(result.missing_keys) == ["out_layer.bias", "out_layer.weight"]
    assert result.unexpected_keys == []
    assert set(reference.state_dict()) == set(port.state_dict())  # same names, same structure

    captured = {}
    port.out_layer.register_forward_hook(lambda m, inp, out: captured.setdefault("h", inp[0]))
    x = _waveform(3, seed=11)
    with torch.no_grad():
        ref_hidden, ref_out = reference(x)
        logits = port(x)
    assert tuple(ref_hidden.shape) == tuple(captured["h"].shape) == (3, 160)
    assert tuple(ref_out.shape) == (3, 2)
    max_diff = (ref_hidden - captured["h"]).abs().max().item()
    print(f"AASIST faithfulness: max |port - reference| before out_layer = {max_diff:.3e}")
    torch.testing.assert_close(captured["h"], ref_hidden, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(logits, port.out_layer(captured["h"]).squeeze(1))

    # The reference itself runs at its own length too; the port's encoder does not depend on it.
    with torch.no_grad():
        hidden_64600, _ = reference(_waveform(1, length=64600))
    assert tuple(hidden_64600.shape) == (1, 160)
