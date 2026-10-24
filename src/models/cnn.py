"""The D42 CNN: 3 or 4 conv blocks on the standardized F3 log-Mel, one logit per clip (D30, D42).

Block: Conv2d 3x3 (padding 1) -> BatchNorm2d -> ReLU -> MaxPool2d 2x2. Channels 16, 32, 64 and 128
for the 4th block: the network the Stage 6 compute benchmark measured at batch 256
(docs/audit_evidence/stage6_preprocessing/benchmark.json), with the BatchNorm D42 adds. Head:
global average pooling -> Dropout(0.3) -> Linear(C, 1). The logit is the raw score of score(audio)
(D26 C->D; higher = more likely spoof). D30 searches n_blocks in {3, 4}; nothing else is built.

Input (n, 1, 64, 197): after 3 blocks the map is 8 x 24, after 4 blocks 4 x 12 (floor pooling).
"""
from __future__ import annotations

from torch import nn

CHANNELS = (16, 32, 64, 128)  # output channels of blocks 1..4 (D42)
DROPOUT = 0.3  # D42 head dropout
N_BLOCKS = (3, 4)  # D30 search space
INPUT_SHAPE = (1, 64, 197)  # D39 log-Mel per clip


class CNN(nn.Module):
    """features (conv blocks) -> global average pooling -> dropout -> linear; forward gives (n,) logits."""

    def __init__(self, n_blocks: int) -> None:
        super().__init__()
        if isinstance(n_blocks, bool) or not isinstance(n_blocks, int) or n_blocks not in N_BLOCKS:
            raise ValueError(f"n_blocks must be one of {N_BLOCKS} (D30), got {n_blocks!r}")
        self.n_blocks = n_blocks
        layers, in_ch = [], INPUT_SHAPE[0]
        for out_ch in CHANNELS[:n_blocks]:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(DROPOUT)
        self.head = nn.Linear(in_ch, 1)

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != INPUT_SHAPE:
            raise ValueError(f"input must have shape (n, {', '.join(map(str, INPUT_SHAPE))}), "
                             f"got {tuple(x.shape)}")
        z = self.pool(self.features(x)).flatten(1)
        return self.head(self.dropout(z)).squeeze(1)


def build_cnn(n_blocks: int) -> CNN:
    """The D42 CNN with n_blocks in {3, 4}; any other value raises ValueError."""
    return CNN(n_blocks)
