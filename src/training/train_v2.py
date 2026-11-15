"""Train one model v2 run: FoR train (raw cache, D50 augmentation) and val-A through S only (D49).

Generalizes train_one (D42 structure) to the three D49 candidates. There is no parameter for val-B,
test or external data (D9): only the raw train cache and the selection set S (built from val-A).

Order of one run:
1. set_seed(run_seed) first; build_model(candidate); optimizer from configs/v2_candidates.yaml:
   cnn-aug / crnn Adam(lr 1e-3, weight_decay 0), constant; aasist Adam(lr 1e-4, betas (0.9, 0.999),
   weight_decay 1e-4, amsgrad False) with cosine_lr(step, 30 x steps_per_epoch, 1e-4, 5e-6) set
   before every optimizer step (global step from 0; D61 (3): the total uses 30 epochs, not the
   epochs actually run). BCEWithLogitsLoss without weights; float32. mean/std (clean-train
   statistics, D61 (5)) are required for the log-Mel models and refused for aasist.
2. Each epoch: one pass over all of train in the EpochSampler order (last batch may be smaller),
   items built by v2_data (augment -> F3 -> input); then eval mode, no_grad scoring of the five S
   conditions from the stored features (log-Mel S standardized once with the same clean stats);
   EER and ROC-AUC per condition on the raw logits; M = mean of the 5 EERs (D49).
3. Early stopping (D49 under D42): best starts at +inf; an epoch improves only if M is strictly
   lower; patience; max_epochs; a tie keeps the earliest epoch; the best epoch's state_dict (CPU
   copy) is returned, not the last.
4. History: one record per epoch with every logged field, including family and codec counts.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.evaluation.metrics import eer, roc_auc
from src.models.v2_checkpoint import INPUTS, build_model, trainable_parameters
from src.training.determinism import set_seed
from src.training.normalization import apply_bin_stats
from src.training.selection_set import CONDITIONS, SelectionSet
from src.training.train_cnn import _cpu_copy, _positive_int, _score
from src.training.v2_config import V2Config
from src.training.v2_data import (
    EpochSampler,
    V2TrainDataset,
    add_counts,
    empty_counts,
    make_loader,
)
from src.training.v2_schedule import cosine_lr

FORMS = {"logmel": "logmel", "waveform": "waveform"}  # input kind -> S form


@dataclass(frozen=True)
class V2TrainResult:
    """The best epoch's weights (CPU tensors) and the run record."""

    candidate: str
    state_dict: dict
    best_epoch: int
    best_selection_m: float
    epochs_run: int
    history: tuple[dict, ...]
    determinism: dict
    peak_gpu_memory_bytes: int | None
    trainable_parameters: int


def build_optimizer(model: nn.Module, settings: dict) -> torch.optim.Optimizer:
    """Adam with the candidate's configured arguments (configs/v2_candidates.yaml)."""
    opt = settings["optimizer"]
    if opt["name"] != "adam":
        raise ValueError(f"optimizer must be adam (D49, D61), got {opt['name']!r}")
    kwargs = {"lr": opt["learning_rate"], "weight_decay": opt["weight_decay"]}
    if "betas" in opt:
        kwargs["betas"] = tuple(opt["betas"])
    if "amsgrad" in opt:
        kwargs["amsgrad"] = opt["amsgrad"]
    return torch.optim.Adam(model.parameters(), **kwargs)


def learning_rate(settings: dict, step: int, steps_per_epoch: int) -> float:
    """The learning rate of global optimizer step `step` (from 0) for one candidate."""
    base = settings["optimizer"]["learning_rate"]
    scheduler = settings["scheduler"]
    if scheduler == "none":
        return float(base)
    if isinstance(scheduler, dict) and scheduler.get("name") == "cosine_per_step":
        total = int(scheduler["total_epochs"]) * _positive_int("steps_per_epoch", steps_per_epoch)
        return cosine_lr(step, total, base, scheduler["lr_min"])
    raise ValueError(f"unknown scheduler {scheduler!r}")


def selection_tensors(selection: SelectionSet, input_kind: str, mean, std) -> dict:
    """The five S inputs as tensors; the log-Mel standardized once with the clean-train stats."""
    if selection.form != FORMS[input_kind]:
        raise ValueError(f"S must be loaded with form {FORMS[input_kind]!r} for this candidate, "
                         f"got {selection.form!r}")
    out = {}
    for name in CONDITIONS:
        x = selection.features(name)
        if input_kind == "logmel":
            x = apply_bin_stats(x, mean, std)
        out[name] = torch.from_numpy(np.ascontiguousarray(x))
    return out


def evaluate_selection(model, tensors: dict, labels, batch_size: int, device) -> tuple[dict, dict]:
    """EER and ROC-AUC of the raw logits for every S condition, in CONDITIONS order."""
    eers, aucs = {}, {}
    for name in CONDITIONS:
        logits = _score(model, tensors[name], batch_size, device)
        eers[name] = float(eer(labels, logits)[0])
        aucs[name] = float(roc_auc(labels, logits))
    return eers, aucs


def selection_m(eers: dict) -> float:
    """M = mean of the five S EERs (D49)."""
    if list(eers) != list(CONDITIONS):
        raise ValueError(f"M needs the EERs of {list(CONDITIONS)} in order, got {list(eers)}")
    return math.fsum(eers.values()) / len(CONDITIONS)


def train_v2(candidate, *, config: V2Config, run_seed, raw_cache, selection, augmenter, profile,
             mean, std, max_epochs, patience, batch_size, num_workers, logger,
             device="cpu") -> V2TrainResult:
    """One run (module docstring, steps 1-4); log each epoch and the summary."""
    determinism = set_seed(run_seed)
    if candidate not in config.candidate_order:
        raise ValueError(f"candidate must be one of {config.candidate_order}, got {candidate!r}")
    settings = config.candidate(candidate)
    input_kind = INPUTS[candidate]
    max_epochs = _positive_int("max_epochs", max_epochs)
    patience = _positive_int("patience", patience)
    batch_size = _positive_int("batch_size", batch_size)
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError(f"num_workers must be an int >= 0, got {num_workers!r}")
    if max_epochs > config.settings["max_epochs"]:
        raise ValueError(f"max_epochs {max_epochs} exceeds the D49 maximum {config.settings['max_epochs']}")
    if input_kind == "waveform" and (mean is not None or std is not None):
        raise ValueError(f"{candidate} takes the waveform: mean and std must be None")
    if input_kind == "logmel" and (mean is None or std is None):
        raise ValueError(f"{candidate} takes the log-Mel: the clean-train mean and std are "
                         f"required (D61 (5))")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"device {device} requested but CUDA is not available")
    determinism = {**determinism, "num_workers": num_workers}

    tensors = selection_tensors(selection, input_kind, mean, std)
    labels = selection.labels
    model = build_model(candidate).to(device)
    optimizer = build_optimizer(model, settings)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(raw_cache)
    dataset = V2TrainDataset(raw_cache, augmenter, profile, input_kind, run_seed, mean, std)
    sampler = EpochSampler(n, run_seed)
    loader = make_loader(dataset, sampler, batch_size, num_workers)
    steps_per_epoch = math.ceil(n / batch_size)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best, best_epoch, best_state, bad, step = math.inf, None, None, 0, 0
    history = []
    for epoch in range(1, max_epochs + 1):
        start = time.perf_counter()
        model.train()
        counts, total, seen, lr = empty_counts(), 0.0, 0, None
        for xb, yb, families, options in loader:
            lr = learning_rate(settings, step, steps_per_epoch)
            for group in optimizer.param_groups:
                group["lr"] = lr
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            step += 1
            total += float(loss.item()) * len(yb)
            seen += len(yb)
            add_counts(counts, families, options)
        if sampler.epoch != epoch or seen != n:
            raise RuntimeError(f"epoch {epoch}: sampler epoch {sampler.epoch}, {seen} of {n} items")
        trained = time.perf_counter()
        eers, aucs = evaluate_selection(model, tensors, labels, batch_size, device)
        m = selection_m(eers)
        done = time.perf_counter()
        record = {
            "epoch": epoch, "train_loss": total / n, "selection_m": m,
            **{f"val_a_eer_{c}": eers[c] for c in CONDITIONS},
            **{f"val_a_auc_{c}": aucs[c] for c in CONDITIONS},
            "learning_rate": float(lr), "epoch_seconds": done - start,
            "train_seconds": trained - start, "selection_seconds": done - trained,
            **counts,
        }
        history.append(record)
        if m < best:  # strictly lower (D49 under D42); a tie keeps the earlier epoch
            best, best_epoch, best_state, bad = m, epoch, _cpu_copy(model), 0
        else:
            bad += 1
        logger.log_epoch(record)
        if bad >= patience:
            break

    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    logger.log_summary({"best_epoch": best_epoch, "best_selection_m": best,
                        "epochs_run": len(history), "peak_gpu_memory_bytes": peak})
    return V2TrainResult(
        candidate=candidate, state_dict=best_state, best_epoch=best_epoch, best_selection_m=best,
        epochs_run=len(history), history=tuple(history), determinism=determinism,
        peak_gpu_memory_bytes=peak, trainable_parameters=trainable_parameters(model),
    )
