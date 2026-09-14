#!/usr/bin/env python3
"""
Test a trained CTNet on images.

Folders are searched recursively and every image is labelled by the nearest
'cover' or 'stego' folder above it, so a whole dataset tree can be tested at
once. All images found are reported together as one result, regardless of the
sub-folder they came from:

    python src/test.py --weights outputs/run/best.pt --input data
        -> one report over data/{train,val,test}/{cover,stego}

    python src/test.py --weights outputs/run/best.pt                  # data/test only
    python src/test.py --weights outputs/run/best.pt --input data/val data/test

Prediction on images without labels (nothing called cover/ or stego/ above them):

    python src/test.py --weights outputs/run/best.pt --input photo.png some_folder/
    python src/test.py --weights outputs/run/best.pt --input big_images/ --tile-mode center

Save the predictions, the metric report and the ROC curve:

    python src/test.py --weights outputs/run/best.pt --input data \
                       -o predictions.csv -m metrics.json --roc roc.png

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
from datasets import InferenceImageDataset, inference_collate, label_from_path, list_images
from utils import (DEFAULT_DATA_DIR, compute_metrics, fmt, get_device, load_checkpoint,
                   log_metrics, plot_roc, save_metrics, save_roc_points, set_logger)

try:
    from tqdm import tqdm
except ImportError:                      # tqdm is optional
    def tqdm(x, **_):
        return x


# --------------------------------------------------------------------------- #
#  Input collection                                                            #
# --------------------------------------------------------------------------- #
def collect_inputs(inputs):
    """Collect every image below the given files/folders.

    Folders are searched recursively, so a whole dataset tree can be passed at
    once (``--input data`` finds data/train/cover, data/train/stego,
    data/val/..., data/test/... and any other sub-folder).

    Every image gets a label from the nearest 'cover' (0) or 'stego' (1) folder
    above it, or None when there is none. All labelled images are reported
    together, no matter which sub-folder they came from.
    """
    items, seen = [], set()
    for inp in inputs:
        p = Path(inp)
        if p.is_file():
            found = [p]
        elif p.is_dir():
            found = list_images(p)                    # recursive
            if not found:
                logging.warning(f"No images found in: {p}")
        else:
            logging.warning(f"Input not found: {p}")
            continue
        for path in found:
            key = path.resolve()
            if key not in seen:
                seen.add(key)
                items.append({"path": path, "label": label_from_path(path)})
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

    results = predict(model, device, [it["path"] for it in items],
                      args.batch_size, args.tile_mode, args.aggregate, args.workers)

    rows = []
    for it, r in zip(items, results):
        pred = None if r["p_stego"] is None else int(r["p_stego"] >= args.threshold)
        rows.append({**it, "path": str(it["path"]), "pred": pred, **r})

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

    # ---- summary -----------------------------------------------------------
    n_stego = sum(r["pred"] for r in ok)
    logging.info("")
    logging.info(f"Predicted: {len(ok) - n_stego} cover, {n_stego} stego  "
                 f"(threshold {args.threshold}, {len(failed)} skipped)")

    if not labeled:
        if args.metrics or args.roc:
            logging.warning("No labelled images (none of them is inside a 'cover' or 'stego' "
                            "folder) - metrics file and ROC curve skipped.")
        write_predictions_csv(args.output, rows)
        return

    labels = [r["label"] for r in labeled]
    scores = [r["p_stego"] for r in labeled]
    overall = compute_metrics(labels, scores, args.threshold)

    logging.info("")
    log_metrics(overall, f"All labelled images - {Path(args.weights).name}")

    # ---- files -------------------------------------------------------------
    if args.metrics:
        save_metrics(args.metrics, overall, checkpoint=str(args.weights),
                     input=[str(i) for i in args.input],
                     tile_mode=args.tile_mode, aggregate=args.aggregate)
        logging.info(f"Metrics written to {args.metrics}")

    if args.roc:
        png = plot_roc(labels, scores, args.roc, auc=overall["AUC"],
                       title=f"ROC - CTNet ({Path(args.weights).stem})")
        points = save_roc_points(Path(args.roc).with_suffix(".csv"), labels, scores)
        if png:
            logging.info(f"ROC curve written to {png} (points: {points})")

    write_predictions_csv(args.output, rows)


def write_predictions_csv(path, rows):
    if not path:
        return
    out = Path(path)
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
                   help="image file(s) and/or folder(s); folders are searched recursively, "
                        "so a whole dataset tree such as 'data' can be given at once")
    p.add_argument("--tile-mode", choices=["tile", "center"], default="tile",
                   help="how to handle images larger than 256x256")
    p.add_argument("--aggregate", choices=["mean", "max"], default="mean",
                   help="how to combine tile probabilities into one image score")
    p.add_argument("--threshold", type=float, default=0.5, help="P(stego) >= threshold -> stego")
    p.add_argument("--batch-size", type=int, default=32, help="tiles per forward pass")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:1 | mps")
    p.add_argument("--output", "-o", default=None, help="CSV file with the prediction for every image")
    p.add_argument("--metrics", "-m", default=None,
                   help="JSON file with the full metric report (labelled input only)")
    p.add_argument("--roc", default=None,
                   help="save the ROC curve here, e.g. roc.png (labelled input only); "
                        "the raw points are written next to it as .csv")
    p.add_argument("--verbose", "-v", action="store_true", help="print every image, also in labelled mode")
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
