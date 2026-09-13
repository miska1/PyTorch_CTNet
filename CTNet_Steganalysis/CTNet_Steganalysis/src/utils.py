"""Shared helpers for training and testing CTNet."""

import logging
import os
import random
from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


class AverageMeter:
    """Keeps the latest value and running average of a metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_logger(log_path=None, mode="a"):
    """Log to console and (optionally) to a file."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)

    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode=mode, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(file_handler)
    return logger


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name="auto"):
    """'auto' -> cuda if available, then Apple mps, else cpu."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def lr_at_epoch(epoch, base_lr, milestones, gamma=0.1):
    """Step schedule identical to the original MultiStepLR usage
    (scheduler.step() at the start of every epoch): the LR is multiplied
    by ``gamma`` from each milestone epoch on. E.g. base 0.01,
    milestones [80, 130, 170]: epochs 1-79 -> 1e-2, 80-129 -> 1e-3,
    130-169 -> 1e-4, 170+ -> 1e-5. Computing it directly also makes
    resuming from a checkpoint use the correct LR."""
    return base_lr * gamma ** bisect_right(sorted(milestones), epoch)


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def save_checkpoint(path, model, optimizer, epoch, **extra):
    """Uses the original key names ('original_state', 'optimizer_state', 'epoch')."""
    state = {
        "original_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
    }
    state.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)        # atomic: no corrupted checkpoint if interrupted


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    """Load a checkpoint produced by this project OR by the original CTNet.py.

    Accepts {'original_state': ...}, {'state_dict': ...} or a bare state_dict,
    and strips a 'module.' prefix left by DataParallel. Returns the full dict.
    """
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:            # torch < 1.13 has no weights_only
        ckpt = torch.load(path, map_location=map_location)

    if isinstance(ckpt, dict) and "original_state" in ckpt:
        state = ckpt["original_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state, ckpt = ckpt, {"original_state": ckpt}

    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(state)

    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


def count_parameters(model, trainable_only=True):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def flatten_pair_batch(sample, device):
    """(B, 2, C, H, W) paired batch -> (2B, C, H, W) images and (2B,) labels on device."""
    data, label = sample["data"], sample["label"]
    data = data.reshape(-1, *data.shape[2:])
    label = label.reshape(-1)
    non_blocking = device.type == "cuda"
    return data.to(device, non_blocking=non_blocking), label.to(device, non_blocking=non_blocking)
