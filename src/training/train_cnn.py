"""Train one D42 CNN configuration on FoR train with D42 early stopping on val-A (Stage 8; D30, D42).

train_one sees only train and val-A: there is no parameter for val-B, test or external data (val-B is
not scored until Stage 10, D9, D42). Uses eer() and roc_auc() from the evaluation harness, so it lives
in the research tier; the checkpoint module never imports it.

Order of one run (D42):
1. set_seed(seed) first, so the weight initialization and the data order are seeded.
2. mean, std = fit_bin_stats(train_x), on train only; train and val-A standardized with them.
3. build_cnn(n_blocks); Adam(learning_rate, weight_decay=0), constant rate; BCEWithLogitsLoss
   without class weights; float32.
4. Each epoch: one pass over all of train (the last batch may be smaller), shuffled by a
   torch.Generator seeded with seed; then model.eval() and no_grad scoring of val-A in batches;
   val-A EER and ROC-AUC of the logits. epoch_seconds covers training and val-A scoring.
5. Early stopping (D42): the best starts at +inf; an epoch improves only if its val-A EER is
   strictly lower; then it becomes the best epoch and a copy of its state_dict is kept. Otherwise
   the counter of non-improving epochs increases, and training stops when it reaches patience, or
   after max_epochs. A tie never replaces the best, so the earliest epoch wins; the returned
   state_dict is the best epoch's, not the last.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.evaluation.metrics import eer, roc_auc
from src.models.cnn import build_cnn
from src.risk.decisions import _validate_labels
from src.training.determinism import set_seed
from src.training.normalization import apply_bin_stats, fit_bin_stats


@dataclass(frozen=True)
class EpochRecord:
    """One epoch of history (D42 W&B fields)."""

    epoch: int
    train_loss: float
    val_a_eer: float
    val_a_auc: float
    epoch_seconds: float


@dataclass(frozen=True)
class TrainResult:
    """The best epoch's weights (CPU tensors), the train-only statistics and the run record."""

    state_dict: dict
    mean: np.ndarray
    std: np.ndarray
    best_epoch: int
    best_val_a_eer: float
    epochs_run: int
    history: tuple[EpochRecord, ...]
    determinism: dict
    peak_gpu_memory_bytes: int | None


def _positive_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an int >= 1, got {value!r}")
    return value


def _labels(name: str, labels, n: int) -> np.ndarray:
    y = _validate_labels(labels)  # 0/1, both classes present
    if y.shape != (n,):
        raise ValueError(f"{name} must have one label per clip ({n}), got shape {y.shape}")
    return y


def _cpu_copy(model: nn.Module) -> dict:
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


def _score(model: nn.Module, x: torch.Tensor, batch_size: int, device: torch.device) -> np.ndarray:
    """Logits of every clip in x, in order, with the model in eval mode and no gradients."""
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            out.append(model(x[start:start + batch_size].to(device)).cpu())
    return torch.cat(out).numpy().astype(np.float64)


def train_one(train_x, train_y, val_a_x, val_a_y, *, n_blocks, learning_rate, seed, max_epochs,
              patience, batch_size, logger, device="cpu") -> TrainResult:
    """Train one configuration (module docstring, steps 1-5); log each epoch and the summary."""
    determinism = set_seed(seed)
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) \
            or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError(f"learning_rate must be a finite number > 0, got {learning_rate!r}")
    max_epochs = _positive_int("max_epochs", max_epochs)
    patience = _positive_int("patience", patience)
    batch_size = _positive_int("batch_size", batch_size)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"device {device} requested but CUDA is not available")

    mean, std = fit_bin_stats(train_x)  # train only (D42)
    x_train = torch.from_numpy(apply_bin_stats(train_x, mean, std))
    x_val = torch.from_numpy(apply_bin_stats(val_a_x, mean, std))
    y_train_np = _labels("train_y", train_y, len(x_train))
    y_val = _labels("val_a_y", val_a_y, len(x_val))
    y_train = torch.from_numpy(y_train_np.astype(np.float32))

    model = build_cnn(n_blocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
    loss_fn = nn.BCEWithLogitsLoss()
    order_gen = torch.Generator().manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best, best_epoch, best_state, bad = math.inf, None, None, 0
    history = []
    n = len(x_train)
    for epoch in range(1, max_epochs + 1):
        start_time = time.perf_counter()
        model.train()
        order = torch.randperm(n, generator=order_gen)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            xb, yb = x_train[idx].to(device), y_train[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(idx)
        logits = _score(model, x_val, batch_size, device)
        val_eer, _ = eer(y_val, logits)
        val_auc = roc_auc(y_val, logits)
        record = EpochRecord(epoch=epoch, train_loss=total / n, val_a_eer=float(val_eer),
                             val_a_auc=float(val_auc),
                             epoch_seconds=time.perf_counter() - start_time)
        history.append(record)
        if record.val_a_eer < best:  # strictly lower (D42); a tie keeps the earlier epoch
            best, best_epoch, best_state, bad = record.val_a_eer, epoch, _cpu_copy(model), 0
        else:
            bad += 1
        logger.log_epoch(record.epoch, record.train_loss, record.val_a_eer, record.val_a_auc,
                         record.epoch_seconds)
        if bad >= patience:
            break

    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    logger.log_summary(best_epoch, best, len(history), peak)
    return TrainResult(
        state_dict=best_state, mean=mean, std=std, best_epoch=best_epoch, best_val_a_eer=best,
        epochs_run=len(history), history=tuple(history), determinism=determinism,
        peak_gpu_memory_bytes=peak,
    )
