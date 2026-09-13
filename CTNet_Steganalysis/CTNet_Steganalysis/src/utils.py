"""Shared helpers for training and testing CTNet."""

import csv
import json
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


# --------------------------------------------------------------------------- #
#  Metrics                                                                     #
# --------------------------------------------------------------------------- #
# Convention: positive class = stego (label 1), negative class = cover (label 0).
#   TP = stego detected          FN = stego missed
#   TN = cover accepted          FP = cover falsely flagged

def roc_auc(labels, scores):
    """Area under the ROC curve via the Mann-Whitney U statistic (ties get the
    average rank, so it is correct even when many images share a score).
    Returns None if one of the two classes is absent."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    _, inv, counts = np.unique(scores[order], return_inverse=True, return_counts=True)
    avg_rank = np.cumsum(counts) - (counts - 1) / 2.0
    ranks = np.empty(len(scores))
    ranks[order] = avg_rank[inv]
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_curve(labels, scores):
    """ROC points for every distinct threshold: (fpr, tpr, thresholds) as arrays,
    starting at (0, 0) and ending at (1, 1). Returns None if a class is absent."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if (labels == 1).sum() == 0 or (labels == 0).sum() == 0:
        return None

    order = np.argsort(-scores, kind="mergesort")
    y, s = labels[order], scores[order]
    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]      # keep one point per distinct score
    tpr = np.r_[0.0, tps[last] / tps[-1]]
    fpr = np.r_[0.0, fps[last] / fps[-1]]
    thr = np.r_[np.inf, s[last]]
    return fpr, tpr, thr


def compute_metrics(labels, scores, threshold=0.5, loss=None):
    """Full classification report for stego detection.

    labels    : 0 = cover, 1 = stego
    scores    : P(stego) per image
    threshold : score >= threshold -> predicted stego

    Rates that are undefined (empty class, or precision with no positive
    prediction) are returned as None and serialise to null in JSON.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    preds = (scores >= threshold).astype(int)

    tn = int(((labels == 0) & (preds == 0)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum())
    tp = int(((labels == 1) & (preds == 1)).sum())
    n = tn + fp + fn + tp
    n_cover, n_stego, n_flagged = tn + fp, tp + fn, tp + fp

    tpr = tp / n_stego if n_stego else None                 # recall / sensitivity / 1 - P_MD
    tnr = tn / n_cover if n_cover else None                 # specificity / cover accuracy
    p_fa = fp / n_cover if n_cover else None                # false alarm rate (FPR)
    p_md = fn / n_stego if n_stego else None                # missed detection rate (FNR)
    precision = tp / n_flagged if n_flagged else None       # PPV
    balanced_acc = None if tpr is None or tnr is None else 0.5 * (tpr + tnr)
    f1 = None if precision is None or tpr is None or precision + tpr == 0 \
        else 2 * precision * tpr / (precision + tpr)

    metrics = {
        "images": n,
        "cover_images": n_cover,
        "stego_images": n_stego,
        "threshold": threshold,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "accuracy": (tp + tn) / n if n else None,
        "balanced_accuracy": balanced_acc,
        "precision": precision,
        "recall_TPR": tpr,
        "f1_score": f1,
        "specificity_TNR": tnr,
        "false_alarm_rate_FPR": p_fa,
        "missed_detection_rate_FNR": p_md,
        "P_E": None if p_fa is None or p_md is None else 0.5 * (p_fa + p_md),
        "AUC": roc_auc(labels, scores),
    }
    if loss is not None:
        metrics["loss"] = float(loss)
    # kept for readability of the log / backwards compatibility
    metrics["cover_accuracy"] = tnr
    metrics["stego_accuracy"] = tpr
    return metrics


def fmt(v, nd=4):
    return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def format_metrics(metrics, title="Results"):
    """Multi-line human-readable report (returned as a list of lines)."""
    m = metrics
    return [
        "-" * 62,
        f"{title}  ({m['images']} images: {m['cover_images']} cover, {m['stego_images']} stego)",
        "-" * 62,
        "Confusion matrix (rows = true, cols = predicted):",
        "                 cover     stego",
        f"   cover   {m['TN']:>9} {m['FP']:>9}   (TN, FP)",
        f"   stego   {m['FN']:>9} {m['TP']:>9}   (FN, TP)",
        "",
        f"  Accuracy                  : {fmt(m['accuracy'])}",
        f"  Balanced accuracy         : {fmt(m['balanced_accuracy'])}",
        f"  Precision                 : {fmt(m['precision'])}",
        f"  Recall / TPR              : {fmt(m['recall_TPR'])}",
        f"  F1-score                  : {fmt(m['f1_score'])}",
        f"  Specificity / TNR         : {fmt(m['specificity_TNR'])}",
        f"  False-alarm rate (FPR)    : {fmt(m['false_alarm_rate_FPR'])}",
        f"  Missed-detection rate(FNR): {fmt(m['missed_detection_rate_FNR'])}",
        f"  P_E = (FPR + FNR) / 2     : {fmt(m['P_E'])}",
        f"  ROC AUC                   : {fmt(m['AUC'])}",
        *([f"  Loss                      : {fmt(m['loss'])}"] if "loss" in m else []),
        f"  Decision threshold        : {fmt(m['threshold'], 2)}",
        "-" * 62,
    ]


def log_metrics(metrics, title="Results"):
    for line in format_metrics(metrics, title):
        logging.info(line)


def save_metrics(path, metrics, **extra):
    """Write the metrics dict to JSON (extra fields first, e.g. checkpoint name)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({**extra, **metrics}, f, indent=2)
    return path


def plot_roc(labels, scores, path, title="ROC curve", auc=None):
    """Save a ROC curve plot (PNG/PDF/SVG by extension). Returns the path,
    or None if matplotlib is missing or a class is absent."""
    curve = roc_curve(labels, scores)
    if curve is None:
        logging.warning("ROC curve needs both cover and stego images - plot skipped.")
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")                     # no display needed (servers, SSH)
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib is not installed - ROC plot skipped "
                        "(pip install matplotlib).")
        return None

    fpr, tpr, thr = curve
    if auc is None:
        auc = roc_auc(labels, scores)

    # mark the operating point closest to the top-left corner (Youden's J)
    j = int(np.argmax(tpr - fpr))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=150)
    ax.plot(fpr, tpr, lw=2, label=f"CTNet (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="random guess (AUC = 0.5)")
    ax.plot(fpr[j], tpr[j], "o", ms=6, color="crimson",
            label=f"best trade-off @ {thr[j]:.3f}\n(FPR {fpr[j]:.3f}, TPR {tpr[j]:.3f})")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("False-alarm rate  (cover classified as stego)")
    ax.set_ylabel("True-positive rate  (stego detected)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_roc_points(path, labels, scores):
    """Write the ROC points (threshold, FPR, TPR) to CSV for plotting elsewhere."""
    curve = roc_curve(labels, scores)
    if curve is None:
        return None
    fpr, tpr, thr = curve
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "false_alarm_rate_FPR", "recall_TPR"])
        for t, x, y in zip(thr, fpr, tpr):
            w.writerow([f"{t:.6f}" if np.isfinite(t) else "inf", f"{x:.6f}", f"{y:.6f}"])
    return path


def count_parameters(model, trainable_only=True):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def flatten_pair_batch(sample, device):
    """(B, 2, C, H, W) paired batch -> (2B, C, H, W) images and (2B,) labels on device."""
    data, label = sample["data"], sample["label"]
    data = data.reshape(-1, *data.shape[2:])
    label = label.reshape(-1)
    non_blocking = device.type == "cuda"
    return data.to(device, non_blocking=non_blocking), label.to(device, non_blocking=non_blocking)
