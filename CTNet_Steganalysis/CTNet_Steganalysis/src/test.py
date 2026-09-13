#!/usr/bin/env python3
"""
Test a trained CTNet on images.

Labelled evaluation (folder with cover/ and stego/ sub-folders):
    python src/test.py --weights outputs/run/best.pt                      # default: data/test
    python src/test.py --weights outputs/run/best.pt --input data/val

Prediction on arbitrary images / folders (no labels needed):
    python src/test.py --weights outputs/run/best.pt --input photo.png some_folder/
    python src/test.py --weights outputs/run/best.pt --input big_images/ --tile-mode center

Save every prediction to CSV:
    python src/test.py --weights outputs/run/best.pt --input data/test --output results.csv

Images larger than 256x256 are split into non-overlapping 256x256 tiles
(or centre-cropped with --tile-mode center) and the per-tile stego
probabilities are aggregated. Images are never resized: interpolation
would wipe out the embedding noise the detector relies on.
"""

import argparse
import csv
import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ctnet import CLASS_NAMES, IMAGE_SIZE, CTNet
from datasets import (COVER_DIR, STEGO_DIR, InferenceImageDataset, collect_labeled_images,
                      inference_collate, is_labeled_dir, list_images)
from utils import DEFAULT_DATA_DIR, get_device, load_checkpoint, set_logger

try:
    from tqdm import tqdm
except ImportError:                      # tqdm is optional
    def tqdm(x, **_):
        return x


# --------------------------------------------------------------------------- #
#  Input collection                                                            #
# --------------------------------------------------------------------------- #
def collect_inputs(inputs):
    """Returns list of (path, label or None).

    * a folder with cover/ and stego/ sub-folders -> labelled (0 / 1)
    * a folder named 'cover' or 'stego'           -> labelled by its name
    * a file whose parent folder is 'cover'/'stego' -> labelled by that name
    * any other folder / file                     -> unlabelled
    """
    def label_from_name(name):
        name = name.lower()
        return 0 if name == COVER_DIR else 1 if name == STEGO_DIR else None

    items, seen = [], set()
    for inp in inputs:
        p = Path(inp)
        if p.is_file():
            new = [(p, label_from_name(p.parent.name))]
        elif is_labeled_dir(p):
            new = collect_labeled_images(p)
        elif p.is_dir():
            label = label_from_name(p.name)
            new = [(f, label) for f in list_images(p)]
        else:
            logging.warning(f"Input not found: {p}")
            continue
        for path, label in new:
            key = path.resolve()
            if key not in seen:
                seen.add(key)
                items.append((path, label))
    return items


# --------------------------------------------------------------------------- #
#  Inference                                                                   #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model, device, paths, batch_size=32, tile_mode="tile", aggregate="mean", workers=2):
    """Returns per-image list of dicts {p_stego, n_tiles, error}."""
    dataset = InferenceImageDataset(paths, IMAGE_SIZE, tile_mode)
    workers = max(0, min(workers, os.cpu_count() or 1))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers,
                        collate_fn=inference_collate)

    tile_probs = [[] for _ in paths]
    errors = [""] * len(paths)
    buf_tiles, buf_owner = [], []

    def flush():
        if not buf_tiles:
            return
        x = torch.stack(buf_tiles).to(device)
        probs = torch.softmax(model(x), dim=1)[:, 1].cpu().tolist()
        for owner, pr in zip(buf_owner, probs):
            tile_probs[owner].append(pr)
        buf_tiles.clear()
        buf_owner.clear()

    for batch in tqdm(loader, desc="Testing", unit="img"):
        for tiles, idx, err in batch:
            if tiles is None:
                errors[idx] = err
                continue
            for t in tiles:
                buf_tiles.append(t)
                buf_owner.append(idx)
                if len(buf_tiles) >= batch_size:
                    flush()
    flush()

    results = []
    for probs, err in zip(tile_probs, errors):
        if err or not probs:
            results.append({"p_stego": None, "n_tiles": 0, "error": err or "no tiles"})
        else:
            p = float(np.max(probs)) if aggregate == "max" else float(np.mean(probs))
            results.append({"p_stego": p, "n_tiles": len(probs), "error": ""})
    return results


# --------------------------------------------------------------------------- #
#  Metrics                                                                     #
# --------------------------------------------------------------------------- #
def roc_auc(labels, scores):
    """Area under ROC via the Mann-Whitney U statistic (ties get average rank)."""
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


def compute_metrics(labels, preds, scores):
    labels, preds = np.asarray(labels), np.asarray(preds)
    tn = int(((labels == 0) & (preds == 0)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum())
    tp = int(((labels == 1) & (preds == 1)).sum())
    n_cover, n_stego = tn + fp, tp + fn
    p_fa = fp / n_cover if n_cover else None          # false alarm (cover called stego)
    p_md = fn / n_stego if n_stego else None          # missed detection
    return {
        "images": len(labels), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": (tn + tp) / max(len(labels), 1),
        "cover_accuracy": None if p_fa is None else 1 - p_fa,
        "stego_accuracy": None if p_md is None else 1 - p_md,
        "P_FA": p_fa, "P_MD": p_md,
        "P_E": None if p_fa is None or p_md is None else 0.5 * (p_fa + p_md),
        "AUC": roc_auc(labels, scores),
    }


def fmt(v):
    return "n/a" if v is None else f"{v:.4f}"


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #
def main(args):
    set_logger(None)
    device = get_device(args.device)

    items = collect_inputs(args.input)
    if not items:
        raise SystemExit(f"No images found in: {', '.join(map(str, args.input))}")

    model = CTNet().to(device)
    ckpt = load_checkpoint(args.weights, model, map_location=device)
    model.eval()
    logging.info(f"Loaded {args.weights} (epoch {ckpt.get('epoch', '?')}) on {device}")
    logging.info(f"{len(items)} image(s); tile mode: {args.tile_mode}; aggregation: {args.aggregate}")

    paths = [p for p, _ in items]
    results = predict(model, device, paths, args.batch_size, args.tile_mode, args.aggregate, args.workers)

    rows = []
    for (path, label), r in zip(items, results):
        pred = None if r["p_stego"] is None else int(r["p_stego"] >= args.threshold)
        rows.append({"path": str(path), "label": label, "pred": pred, **r})

    ok = [r for r in rows if r["pred"] is not None]
    failed = [r for r in rows if r["pred"] is None]
    labeled = [r for r in ok if r["label"] is not None]

    # ---- per-image output --------------------------------------------------
    # unlabelled images are always listed; labelled ones only with --verbose
    to_print = [r for r in ok if r["label"] is None or args.verbose]
    if to_print:
        logging.info("")
        logging.info(f"{'prediction':<10} {'P(stego)':>9} {'tiles':>5}  {'true':<6} file")
        for r in to_print:
            true = "" if r["label"] is None else CLASS_NAMES[r["label"]]
            mark = "" if r["label"] is None else ("  OK" if r["label"] == r["pred"] else "  WRONG")
            logging.info(f"{CLASS_NAMES[r['pred']]:<10} {r['p_stego']:>9.4f} {r['n_tiles']:>5}  "
                         f"{true:<6} {r['path']}{mark}")

    for r in failed:
        logging.warning(f"SKIPPED {r['path']}: {r['error']}")

    # ---- summary -------------------------------------------------------------
    n_stego = sum(r["pred"] for r in ok)
    logging.info("")
    logging.info(f"Predicted: {len(ok) - n_stego} cover, {n_stego} stego  "
                 f"(threshold {args.threshold}, {len(failed)} skipped)")

    if labeled:
        m = compute_metrics([r["label"] for r in labeled], [r["pred"] for r in labeled],
                            [r["p_stego"] for r in labeled])
        logging.info("-" * 60)
        logging.info(f"Labelled images : {m['images']}")
        logging.info(f"Accuracy        : {fmt(m['accuracy'])}")
        logging.info(f"Cover accuracy  : {fmt(m['cover_accuracy'])}   (P_FA = {fmt(m['P_FA'])})")
        logging.info(f"Stego accuracy  : {fmt(m['stego_accuracy'])}   (P_MD = {fmt(m['P_MD'])})")
        logging.info(f"P_E             : {fmt(m['P_E'])}   (0.5 * (P_FA + P_MD))")
        logging.info(f"ROC AUC         : {fmt(m['AUC'])}")
        logging.info("Confusion matrix (rows = true, cols = predicted):")
        logging.info(f"            cover   stego")
        logging.info(f"   cover {m['tn']:>7} {m['fp']:>7}")
        logging.info(f"   stego {m['fn']:>7} {m['tp']:>7}")
        logging.info("-" * 60)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "true_label", "prediction", "p_stego", "n_tiles", "error"])
            for r in rows:
                w.writerow([r["path"],
                            "" if r["label"] is None else CLASS_NAMES[r["label"]],
                            "" if r["pred"] is None else CLASS_NAMES[r["pred"]],
                            "" if r["p_stego"] is None else f"{r['p_stego']:.6f}",
                            r["n_tiles"], r["error"]])
        logging.info(f"Results written to {out}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Test CTNet on images or labelled cover/stego folders.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--weights", "-w", required=True, help="checkpoint (.pt) from train.py or original CTNet.py")
    p.add_argument("--input", "-i", nargs="+", default=[str(DEFAULT_DATA_DIR / "test")],
                   help="image file(s) and/or folder(s)")
    p.add_argument("--tile-mode", choices=["tile", "center"], default="tile",
                   help="how to handle images larger than 256x256")
    p.add_argument("--aggregate", choices=["mean", "max"], default="mean",
                   help="how to combine tile probabilities into one image score")
    p.add_argument("--threshold", type=float, default=0.5, help="P(stego) >= threshold -> stego")
    p.add_argument("--batch-size", type=int, default=32, help="tiles per forward pass")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:1 | mps")
    p.add_argument("--output", "-o", default=None, help="optional CSV file with all predictions")
    p.add_argument("--verbose", "-v", action="store_true", help="print every image, also in labelled mode")
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
