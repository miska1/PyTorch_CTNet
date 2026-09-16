#!/usr/bin/env python3
"""
Train CTNet on pre-split cover/stego data.

    python src/train.py                                   # defaults: data/train, data/val, data/test
    python src/train.py --epochs 100 --batch-size 8 --run-name suniward_04
    python src/train.py --resume outputs/suniward_04/last.pt

Training procedure (same as the original CTNet.py):
  * paired mini-batches: B cover images + their B stego versions,
  * random rot90 + flip augmentation applied identically to each pair,
  * SGD (momentum 0.9), weight decay 5e-4 on weights only (not on biases / BN),
  * LR 0.01, x0.1 at epochs 80, 130, 170; 200 epochs,
  * before each evaluation BN running statistics are refreshed with one
    no-grad pass over the training set ("adjust_bn_stats"),
  * best-on-validation checkpoint is only tracked after epoch 170,
  * at the end the best checkpoint is evaluated on the test split.
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ctnet import build_model
from datasets import CoverStegoPairDataset, DatasetError, eval_transform, find_pairs, train_transform
from utils import (DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, AverageMeter, compute_metrics,
                   count_parameters, flatten_pair_batch, fmt, format_duration, get_device,
                   load_checkpoint, log_metrics, lr_at_epoch, plot_roc, save_checkpoint,
                   save_metrics, save_roc_points, set_logger, set_lr, set_seed, sync)


# --------------------------------------------------------------------------- #
#  One epoch of training / evaluation                                          #
# --------------------------------------------------------------------------- #
def train_one_epoch(model, device, loader, optimizer, criterion, epoch, epochs, print_freq):
    """One training epoch. Returns (loss, accuracy, seconds, data_wait_seconds).

    Timing notes:
      * 'data wait' is the time the training loop spent waiting for the
        DataLoader; a high share means disk/CPU is the bottleneck, so more
        --workers help. It is part of the epoch time, not extra.
      * on GPU the timers are taken after a synchronisation, so they measure
        finished work rather than queued work.
    """
    losses, accs = AverageMeter(), AverageMeter()
    n_batches = len(loader)
    model.train()

    t_epoch = time.time()
    data_wait = 0.0
    images = 0
    end = time.time()

    for i, sample in enumerate(loader):
        data_wait += time.time() - end
        data, label = flatten_pair_batch(sample, device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        n = data.size(0)
        images += n
        losses.update(loss.item(), n)
        accs.update((output.argmax(1) == label).float().mean().item(), n)

        if i % print_freq == 0 or i == n_batches - 1:
            sync(device)
            elapsed = time.time() - t_epoch
            eta = elapsed / (i + 1) * (n_batches - 1 - i)
            logging.info(f"  epoch {epoch}/{epochs}  batch {i + 1}/{n_batches}  "
                         f"loss {losses.val:.4f} (avg {losses.avg:.4f})  acc {accs.avg:.4f}  "
                         f"| {images / max(elapsed, 1e-9):.1f} img/s  "
                         f"elapsed {format_duration(elapsed)}  ETA {format_duration(eta)}")
        end = time.time()

    sync(device)
    return losses.avg, accs.avg, time.time() - t_epoch, data_wait


@torch.no_grad()
def adjust_bn_stats(model, device, loader):
    """Refresh BatchNorm running statistics with a forward pass over the training set."""
    model.train()
    for sample in loader:
        data, _ = flatten_pair_batch(sample, device)
        model(data)


@torch.no_grad()
def evaluate(model, device, loader, criterion, threshold=0.5):
    """Run the model over a paired loader.

    Returns (metrics, labels, scores) where scores are the P(stego) of every
    image, so the caller can also plot a ROC curve.
    """
    model.eval()
    loss_sum, n = 0.0, 0
    all_labels, all_scores = [], []

    for sample in loader:
        data, label = flatten_pair_batch(sample, device)
        output = model(data)
        loss_sum += criterion(output, label).item() * data.size(0)
        n += data.size(0)
        all_labels.append(label.cpu().numpy())
        all_scores.append(torch.softmax(output, dim=1)[:, 1].cpu().numpy())

    labels = np.concatenate(all_labels) if all_labels else np.array([])
    scores = np.concatenate(all_scores) if all_scores else np.array([])
    metrics = compute_metrics(labels, scores, threshold, loss=loss_sum / max(n, 1))
    return metrics, labels, scores


def log_val_summary(epoch, m):
    """One compact line per validation (the full report is printed for the test set)."""
    logging.info("-" * 8)
    logging.info(f"Val (epoch {epoch}): acc {fmt(m['accuracy'])} | loss {fmt(m['loss'])} | "
                 f"bal acc {fmt(m['balanced_accuracy'])} | F1 {fmt(m['f1_score'])} | "
                 f"P_E {fmt(m['P_E'])} | AUC {fmt(m['AUC'])} | {m['images']} images")


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #
def build_optimizer(model, lr, momentum, weight_decay):
    """Weight decay only on multi-dimensional params (conv / linear weights), as in the original."""
    params_wd, params_rest = [], []
    for p in model.parameters():
        if p.requires_grad:
            (params_wd if p.dim() != 1 else params_rest).append(p)
    groups = [{"params": params_wd, "weight_decay": weight_decay},
              {"params": params_rest, "weight_decay": 0.0}]
    return optim.SGD(groups, lr=lr, momentum=momentum)


def make_loader(dataset, batch_size, shuffle, workers, device):
    workers = max(0, min(workers, os.cpu_count() or 1))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=device.type == "cuda", persistent_workers=workers > 0)


def main(args):
    set_seed(args.seed)
    device = get_device(args.device)

    set_logger(None)                       # console only until the run folder exists

    # ---- data (checked first, so a bad path doesn't leave an empty run folder) ----
    # expanduser() makes '~/datasets/alaska' work; resolve() turns every path into an
    # absolute one, so what is logged is exactly what is read.
    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.is_dir() and not (args.train_dir and args.val_dir):
        raise DatasetError(f"--data-dir does not exist: {data_dir}\n"
                           f"  (given as '{args.data_dir}', working directory is {Path.cwd()})")
    train_dir = Path(args.train_dir).expanduser().resolve() if args.train_dir else data_dir / "train"
    val_dir = Path(args.val_dir).expanduser().resolve() if args.val_dir else data_dir / "val"
    test_dir = Path(args.test_dir).expanduser().resolve() if args.test_dir else data_dir / "test"

    train_set = CoverStegoPairDataset(train_dir, train_transform())
    val_set = CoverStegoPairDataset(val_dir, eval_transform())
    train_loader = make_loader(train_set, args.batch_size, True, args.workers, device)
    val_loader = make_loader(val_set, args.batch_size, False, args.workers, device)

    # ---- output directory -------------------------------------------------
    if args.run_name:
        out_dir = Path(args.output_dir).expanduser() / args.run_name
    elif args.resume:
        out_dir = Path(args.resume).expanduser().resolve().parent
    else:
        out_dir = Path(args.output_dir).expanduser() / datetime.now().strftime("ctnet_%Y%m%d-%H%M%S")
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = out_dir / "best.pt", out_dir / "last.pt"
    set_logger(out_dir / "train.log", mode="a" if args.resume else "w")

    if args.best_after >= args.epochs:
        logging.warning(f"--best-after ({args.best_after}) >= --epochs ({args.epochs}); "
                        f"tracking best model from epoch 1 instead.")
        args.best_after = 0

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    logging.info(f"Output directory: {out_dir}")
    logging.info(f"Device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    if not (args.train_dir and args.val_dir and args.test_dir):
        logging.info(f"Data directory: {data_dir}")
    logging.info(f"  train: {len(train_set)} pairs  <- {train_dir}")
    logging.info(f"  val:   {len(val_set)} pairs  <- {val_dir}")
    logging.info(f"  test:  {test_dir}" + ("" if test_dir.is_dir() else "   (missing)"))
    logging.info(f"Images per step: {2 * args.batch_size} ({args.batch_size} cover + {args.batch_size} stego)")

    # ---- model / optimizer ------------------------------------------------
    model = build_model(device)
    optimizer = build_optimizer(model, args.lr, args.momentum, args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    logging.info(f"CTNet trainable parameters: {count_parameters(model):,}")

    start_epoch, best_acc, previous_time = 1, 0.0, 0.0
    if args.resume:
        ckpt = load_checkpoint(Path(args.resume).expanduser(), model, optimizer, map_location=device)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", 0.0)
        previous_time = ckpt.get("elapsed_seconds", 0.0)      # time spent in earlier runs
        logging.info(f"Resumed from {args.resume} (epoch {start_epoch - 1}, "
                     f"best val acc {best_acc:.4f}, {format_duration(previous_time)} trained so far)")

    # ---- training loop ----------------------------------------------------
    run_start = time.time()
    epoch_times = []

    for epoch in range(start_epoch, args.epochs + 1):
        lr = lr_at_epoch(epoch, args.lr, args.milestones, args.gamma)
        set_lr(optimizer, lr)
        logging.info(f"===== Epoch {epoch}/{args.epochs}  lr={lr:g} =====")
        t_epoch = time.time()

        _, _, train_time, data_wait = train_one_epoch(
            model, device, train_loader, optimizer, criterion, epoch, args.epochs, args.print_freq)
        parts = [f"train {format_duration(train_time)}"]

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            if not args.no_bn_adjust:
                t0 = time.time()
                adjust_bn_stats(model, device, train_loader)
                sync(device)
                parts.append(f"BN refresh {format_duration(time.time() - t0)}")

            t0 = time.time()
            m, _, _ = evaluate(model, device, val_loader, criterion, args.threshold)
            sync(device)
            parts.append(f"val {format_duration(time.time() - t0)}")
            log_val_summary(epoch, m)

            if m["accuracy"] > best_acc and epoch > args.best_after:
                best_acc = m["accuracy"]
                save_checkpoint(best_path, model, optimizer, epoch, best_acc=best_acc, val_metrics=m,
                                elapsed_seconds=previous_time + time.time() - run_start)
                logging.info(f"New best model saved -> {best_path}")
            logging.info(f"Best val accuracy: {best_acc:.4f}")

        epoch_time = time.time() - t_epoch
        epoch_times.append(epoch_time)
        save_checkpoint(last_path, model, optimizer, epoch, best_acc=best_acc,
                        elapsed_seconds=previous_time + time.time() - run_start)

        # ---- timing summary of this epoch ---------------------------------
        mean_epoch = sum(epoch_times) / len(epoch_times)
        remaining = (args.epochs - epoch) * mean_epoch
        logging.info(f"Epoch {epoch}/{args.epochs} took {format_duration(epoch_time)} "
                     f"({', '.join(parts)}; waiting for data {100 * data_wait / max(train_time, 1e-9):.0f}% "
                     f"of the training part)")
        logging.info(f"Elapsed {format_duration(previous_time + time.time() - run_start)} | "
                     f"average epoch {format_duration(mean_epoch)} | "
                     f"{args.epochs - epoch} epoch(s) left, ETA {format_duration(remaining)} "
                     f"({(datetime.now() + timedelta(seconds=remaining)).strftime('%Y-%m-%d %H:%M')})")

    # ---- total training time ----------------------------------------------
    run_time = time.time() - run_start
    total_time = previous_time + run_time
    logging.info("")
    if epoch_times:
        logging.info(f"Training finished: {len(epoch_times)} epoch(s) in {format_duration(run_time)} "
                     f"(average {format_duration(sum(epoch_times) / len(epoch_times))} per epoch, "
                     f"fastest {format_duration(min(epoch_times))}, slowest {format_duration(max(epoch_times))})")
    else:
        logging.info(f"No epoch was run: --epochs is {args.epochs}, but training would start "
                     f"at epoch {start_epoch}.")
    if previous_time:
        logging.info(f"Total training time including earlier runs: {format_duration(total_time)}")

    # ---- final test with the best model ----------------------------------
    if args.no_test:
        return
    try:
        find_pairs(test_dir)
    except DatasetError as e:
        logging.warning(f"Skipping final test: {e}")
        return

    ckpt_path = best_path if best_path.exists() else last_path
    logging.info(f"\nTesting {ckpt_path.name} on {test_dir}")
    load_checkpoint(ckpt_path, model, map_location=device)
    if not args.no_bn_adjust:
        adjust_bn_stats(model, device, train_loader)

    test_set = CoverStegoPairDataset(test_dir, eval_transform())
    test_loader = make_loader(test_set, args.batch_size, False, args.workers, device)
    t0 = time.time()
    m, labels, scores = evaluate(model, device, test_loader, criterion, args.threshold)
    sync(device)
    test_time = time.time() - t0
    log_metrics(m, f"Test set - {ckpt_path.name}")
    logging.info(f"Test finished in {format_duration(test_time)} "
                 f"({m['images'] / max(test_time, 1e-9):.1f} img/s)")

    save_metrics(out_dir / "test_metrics.json", m,
                 checkpoint=str(ckpt_path), split=str(test_dir))
    logging.info(f"Metrics written to {out_dir / 'test_metrics.json'}")
    if not args.no_roc:
        roc_png = plot_roc(labels, scores, out_dir / "test_roc.png",
                           title=f"ROC - CTNet test set ({out_dir.name})", auc=m["AUC"])
        save_roc_points(out_dir / "test_roc_points.csv", labels, scores)
        if roc_png:
            logging.info(f"ROC curve written to {roc_png}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train CTNet (colour spatial steganalysis).",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("data")
    g.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="root with train/ val/ test/ subfolders")
    g.add_argument("--train-dir", default=None, help="override <data-dir>/train")
    g.add_argument("--val-dir", default=None, help="override <data-dir>/val")
    g.add_argument("--test-dir", default=None, help="override <data-dir>/test")
    g.add_argument("--workers", type=int, default=4, help="DataLoader worker processes (0 = main process)")

    g = p.add_argument_group("optimisation")
    g.add_argument("--epochs", type=int, default=200)
    g.add_argument("--batch-size", type=int, default=16, help="cover/stego PAIRS per step (images = 2x)")
    g.add_argument("--lr", type=float, default=0.01)
    g.add_argument("--momentum", type=float, default=0.9)
    g.add_argument("--weight-decay", type=float, default=5e-4)
    g.add_argument("--milestones", type=int, nargs="+", default=[80, 130, 170], help="LR decay epochs")
    g.add_argument("--gamma", type=float, default=0.1, help="LR decay factor")
    g.add_argument("--seed", type=int, default=None)

    g = p.add_argument_group("evaluation / checkpoints")
    g.add_argument("--eval-every", type=int, default=1, help="validate every N epochs")
    g.add_argument("--best-after", type=int, default=170,
                   help="only epochs > this value can become the 'best' checkpoint (original: 170)")
    g.add_argument("--no-bn-adjust", action="store_true",
                   help="skip BN-statistics refresh over the train set before evaluation (faster)")
    g.add_argument("--no-test", action="store_true", help="skip final evaluation on the test split")
    g.add_argument("--threshold", type=float, default=0.5,
                   help="P(stego) >= threshold -> stego (used for all reported metrics)")
    g.add_argument("--no-roc", action="store_true",
                   help="do not save test_roc.png / test_roc_points.csv")
    g.add_argument("--print-freq", type=int, default=100)
    g.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    g.add_argument("--run-name", default=None, help="sub-folder of --output-dir (default: timestamp)")
    g.add_argument("--resume", default=None, help="checkpoint to resume from (e.g. outputs/run/last.pt)")
    g.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:1 | mps")
    return p.parse_args(argv)


if __name__ == "__main__":
    try:
        main(parse_args())
    except DatasetError as e:
        raise SystemExit(f"Data error: {e}")
