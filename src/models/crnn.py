"""Model v2 candidate C2 `crnn`: three D42 conv blocks, a bidirectional LSTM over time, one logit (D49).

Input (n, 1, 64, 197): the F3 log-Mel (D39) after D42's per-bin standardization, here with the
statistics of the clean FoR train log-Mel (D61 (5)).

Layers and shapes (D49 (C2), exactly):
- three D42 blocks, Conv2d 3x3 (padding 1) -> BatchNorm2d -> ReLU -> MaxPool2d 2, channels 16, 32,
  64: (n, 1, 64, 197) -> (n, 64, 8, 24) (floor pooling: 64 -> 8 Mel rows, 197 -> 24 frames);
- per time step, channel x frequency flattened: (n, 24, 64 * 8) = (n, 24, 512);
- one bidirectional LSTM layer, 64 units per direction, batch_first: (n, 24, 128);
- mean over time: (n, 128); Dropout(0.3); Linear(128, 1); forward returns (n,) logits.

The block is the D42 block of src/models/cnn.py, restated here (4 lines) so cnn.py, part of F4,
stays unchanged. Layering: no import of src.evaluation or src.tuning.
"""
from __future__ import annotations

from torch import nn

CHANNELS = (16, 32, 64)  # D49 (C2): three D42 blocks
LSTM_HIDDEN = 64  # D49 (C2): units per direction
DROPOUT = 0.3  # D49 (C2)
INPUT_SHAPE = (1, 64, 197)  # D39 log-Mel per clip
FEATURE_MAP = (64, 8, 24)  # channels, Mel rows, frames after the three blocks
LSTM_INPUT = FEATURE_MAP[0] * FEATURE_MAP[1]  # 512


def _block(in_ch: int, out_ch: int) -> list[nn.Module]:
    """The D42 block (as in src/models/cnn.py)."""
    return [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.MaxPool2d(2),
    ]


class CRNN(nn.Module):
    """features (3 conv blocks) -> per-frame flatten -> BiLSTM -> mean over time -> dropout -> linear."""

    def __init__(self) -> None:
        super().__init__()
        layers, in_ch = [], INPUT_SHAPE[0]
        for out_ch in CHANNELS:
            layers += _block(in_ch, out_ch)
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.lstm = nn.LSTM(LSTM_INPUT, LSTM_HIDDEN, num_layers=1, batch_first=True,
                            bidirectional=True)
        self.dropout = nn.Dropout(DROPOUT)
        self.head = nn.Linear(2 * LSTM_HIDDEN, 1)

    def sequence(self, x):
        """The LSTM input (n, 24, 512): conv features with channel x frequency flattened per frame."""
        if x.ndim != 4 or tuple(x.shape[1:]) != INPUT_SHAPE:
            raise ValueError(f"input must have shape (n, {', '.join(map(str, INPUT_SHAPE))}), "
                             f"got {tuple(x.shape)}")
        z = self.features(x)  # (n, 64, 8, 24)
        n, c, f, t = z.shape
        return z.permute(0, 3, 1, 2).reshape(n, t, c * f)

    def forward(self, x):
        out, _ = self.lstm(self.sequence(x))  # (n, 24, 128)
        return self.head(self.dropout(out.mean(dim=1))).squeeze(1)


def build_crnn() -> CRNN:
    """The D49 (C2) CRNN; it has no options."""
    return CRNN()
