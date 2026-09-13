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
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ctnet import build_model
from datasets import CoverStegoPairDataset, DatasetError, eval_transform, find_pairs, train_transform
from utils import (DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, AverageMeter, count_parameters,
                   flatten_pair_batch, get_device, load_checkpoint, lr_at_epoch,
                   save_checkpoint, set_logger, set_lr, set_seed)


# --------------------------------------------------------------------------- #
#  One epoch of training / evaluation                                          #
# --------------------------------------------------------------------------- #
def train_one_epoch(model, device, loader, optimizer, criterion, epoch, print_freq):
    batch_time, data_time, losses, accs = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    model.train()
    end = time.time()

    for i, sample in enumerate(loader):
        data_time.update(time.time() - end)
        data, label = flatten_pair_batch(sample, device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        n = data.size(0)
        losses.update(loss.item(), n)
        accs.update((output.argmax(1) == label).float().mean().item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0 or i == len(loader) - 1:
            logging.info(f"Epoch: [{epoch}][{i}/{len(loader)}]\t"
                         f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                         f"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                         f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                         f"Acc {accs.avg:.4f}")
    return losses.avg, accs.avg


@torch.no_grad()
def adjust_bn_stats(model, device, loader):
    """Refresh BatchNorm running statistics with a forward pass over the training set."""
    model.train()
    for sample in loader:
        data, _ = flatten_pair_batch(sample, device)
        model(data)


@torch.no_grad()
def evaluate(model, device, loader, criterion):
    """Returns dict with accuracy, loss and per-class accuracies on a paired loader."""
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    correct_cls, total_cls = [0, 0], [0, 0]

    for sample in loader:
        data, label = flatten_pair_batch(sample, device)
        output = model(data)
        loss_sum += criterion(output, label).item() * data.size(0)
        pred = output.argmax(1)
        correct += (pred == label).sum().item()
        total += label.numel()
        for c in (0, 1):
            m = label == c
            correct_cls[c] += (pred[m] == c).sum().item()
            total_cls[c] += m.sum().item()

    cover_acc = correct_cls[0] / max(total_cls[0], 1)
    stego_acc = correct_cls[1] / max(total_cls[1], 1)
    return {
        "accuracy": correct / max(total, 1),
        "loss": loss_sum / max(total, 1),
        "cover_accuracy": cover_acc,             # 1 - false-alarm rate
        "stego_accuracy": stego_acc,             # 1 - missed-detection rate
        "P_E": 0.5 * ((1 - cover_acc) + (1 - stego_acc)),
        "images": total,
    }


def log_metrics(name, m):
    logging.info("-" * 8)
    logging.info(f"{name}: acc {m['accuracy']:.4f} | loss {m['loss']:.4f} | "
                 f"cover acc {m['cover_accuracy']:.4f} | stego acc {m['stego_accuracy']:.4f} | "
                 f"P_E {m['P_E']:.4f} | {m['images']} images")


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
    data_dir = Path(args.data_dir)
    train_dir = Path(args.train_dir or data_dir / "train")
    val_dir = Path(args.val_dir or data_dir / "val")
    test_dir = Path(args.test_dir or data_dir / "test")

    train_set = CoverStegoPairDataset(train_dir, train_transform())
    val_set = CoverStegoPairDataset(val_dir, eval_transform())
    train_loader = make_loader(train_set, args.batch_size, True, args.workers, device)
    val_loader = make_loader(val_set, args.batch_size, False, args.workers, device)

    # ---- output directory -------------------------------------------------
    if args.run_name:
        out_dir = Path(args.output_dir) / args.run_name
    elif args.resume:
        out_dir = Path(args.resume).resolve().parent
    else:
        out_dir = Path(args.output_dir) / datetime.now().strftime("ctnet_%Y%m%d-%H%M%S")
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
    logging.info(f"Train pairs: {len(train_set)} ({train_dir})")
    logging.info(f"Val pairs:   {len(val_set)} ({val_dir})")
    logging.info(f"Images per step: {2 * args.batch_size} ({args.batch_size} cover + {args.batch_size} stego)")

    # ---- model / optimizer ------------------------------------------------
    model = build_model(device)
    optimizer = build_optimizer(model, args.lr, args.momentum, args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    logging.info(f"CTNet trainable parameters: {count_parameters(model):,}")

    start_epoch, best_acc = 1, 0.0
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, map_location=device)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", 0.0)
        logging.info(f"Resumed from {args.resume} (epoch {start_epoch - 1}, best val acc {best_acc:.4f})")

    # ---- training loop ----------------------------------------------------
    for epoch in range(start_epoch, args.epochs + 1):
        lr = lr_at_epoch(epoch, args.lr, args.milestones, args.gamma)
        set_lr(optimizer, lr)
        logging.info(f"===== Epoch {epoch}/{args.epochs}  lr={lr:g} =====")

        train_one_epoch(model, device, train_loader, optimizer, criterion, epoch, args.print_freq)

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            if not args.no_bn_adjust:
                adjust_bn_stats(model, device, train_loader)
            m = evaluate(model, device, val_loader, criterion)
            log_metrics(f"Val (epoch {epoch})", m)

            if m["accuracy"] > best_acc and epoch > args.best_after:
                best_acc = m["accuracy"]
                save_checkpoint(best_path, model, optimizer, epoch, best_acc=best_acc, val_metrics=m)
                logging.info(f"New best model saved -> {best_path}")
            logging.info(f"Best val accuracy: {best_acc:.4f}")

        save_checkpoint(last_path, model, optimizer, epoch, best_acc=best_acc)

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
    m = evaluate(model, device, test_loader, criterion)
    log_metrics("Test", m)
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"checkpoint": str(ckpt_path), **m}, f, indent=2)


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
