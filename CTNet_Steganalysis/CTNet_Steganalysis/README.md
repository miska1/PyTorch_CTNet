# CTNet – Convolutional Transformer Network for Colour Image Steganalysis

PyTorch implementation of **CTNet**, a deep-learning steganalyser that decides whether a colour (RGB)
image is a clean **cover** or a **stego** image carrying a hidden message embedded in the spatial domain
(e.g. S-UNIWARD, HILL). The network combines fixed high-pass residual filters, a CNN group for local
features, a Transformer for global features and iSQRT-COV global covariance pooling.

> Kang-Kang Wei, Wei-Qi Luo, Shun-Quan Tan, Ji-Wu Huang. *CTNet: A Convolutional Transformer Network for
> Color Image Steganalysis.* Journal of Computer Science and Technology (JCST), 40(2): 413–427, 2025.
> DOI: [10.1007/s11390-023-3006-3](https://doi.org/10.1007/s11390-023-3006-3)

This repository is a restructured version of the original single-file `CTNet.py`: the model is separated
from the training code, data are read from ready-made `cover/` and `stego/` folders, and there is a
standalone script for testing the trained model on any images. The network and training procedure are
unchanged (see [Remarks](#remarks)).

---

## Project structure

```
CTNet_Steganalysis/
├── data/
│   ├── train/
│   │   ├── cover/          # training cover images
│   │   └── stego/          # training stego images (same file names as cover/)
│   ├── val/
│   │   ├── cover/
│   │   └── stego/
│   └── test/
│       ├── cover/
│       └── stego/
├── src/
│   ├── ctnet.py            # CTNet model only (+ bundled iSQRT-COV pooling)
│   ├── srm_filter_kernel.py# 30 SRM high-pass kernels used by the pre-processing layer
│   ├── datasets.py         # image reading, cover/stego pairing, augmentation, tiling
│   ├── train.py            # training / validation / final test
│   ├── test.py             # testing on labelled folders or arbitrary images
│   └── utils.py            # logging, checkpoints, LR schedule, metrics, ROC plot
├── outputs/                # created by train.py (checkpoints, logs, metrics, ROC curve)
├── requirements.txt
└── README.md
```

---

## Installation

Python 3.9 or newer is recommended.

```bash
# 1. create and activate a virtual environment
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# 2. (GPU only) install the CUDA build of PyTorch first – pick the command for your
#    CUDA version at https://pytorch.org/get-started/locally/ , for example:
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 3. install the remaining packages
pip install -r requirements.txt

# 4. quick check – builds the network and runs a forward pass
python src/ctnet.py
```

Without step 2 the CPU build of PyTorch is installed. It works, but training CTNet on a CPU is
impractically slow; use it only for testing a trained model.

---

## Preparing the data

Put your images into the three splits. Each split must contain a `cover/` and a `stego/` folder:

```
data/train/cover/00001.ppm      data/train/stego/00001.ppm
data/train/cover/00002.ppm      data/train/stego/00002.ppm
...
```

Requirements:

| Requirement | Why |
|---|---|
| **Cover and its stego version share the same file name** (extension may differ, sub-folders allowed) | Training uses *paired* batches: every cover goes into the same batch as its stego version, with identical augmentation. Files without a partner are ignored with a warning. |
| **256 × 256 pixels** for train / val | The architecture is fixed to 256 × 256 input (the Transformer dimension 1024 = 32 × 32 after three poolings). |
| **8-bit RGB** | Grayscale files are replicated to 3 channels; alpha channels are dropped; 16-bit images are rejected. |
| **Lossless formats** (`.ppm`, `.png`, `.bmp`, `.tif`, `.pgm`) | Lossy re-compression (JPEG/WebP) destroys the ±1 embedding changes the network learns to detect. |

For reference, the original experiments used ALASKA II colour images (256 × 256) with
14 000 / 1 000 / 5 000 train / validation / test pairs and one steganographic algorithm at a fixed payload
(e.g. S-UNIWARD 0.4 bpc).

---

## Training

```bash
# default: data/train, data/val, data/test – 200 epochs, 16 pairs per batch
python src/train.py --run-name suniward_0.4

# smaller GPU, quicker experiment
python src/train.py --run-name quick --batch-size 8 --epochs 60 --milestones 30 45 55 --best-after 0

# resume an interrupted run (continues in the same folder)
python src/train.py --resume outputs/suniward_0.4/last.pt

# data stored elsewhere (relative, absolute or ~ paths all work)
python src/train.py --data-dir D:\datasets\alaska_suniward04
python src/train.py --data-dir ~/datasets/alaska_suniward04
python src/train.py --data-dir ../shared/alaska        # relative to where you run the command

# splits in unrelated places
python src/train.py --train-dir /mnt/fast/train --val-dir /mnt/slow/val --test-dir /mnt/slow/test
```

The folders actually used are printed as absolute paths at the start of every run, so it is easy to
confirm that the right data was picked up:

```
Data directory: /home/user/datasets/alaska_suniward04
  train: 14000 pairs  <- /home/user/datasets/alaska_suniward04/train
  val:   1000 pairs  <- /home/user/datasets/alaska_suniward04/val
  test:  /home/user/datasets/alaska_suniward04/test
```

Main options (`python src/train.py -h` lists all):

| Option | Default | Meaning |
|---|---|---|
| `--data-dir` | `data` | folder containing `train/`, `val/`, `test/`; may be relative, absolute or start with `~` |
| `--train-dir / --val-dir / --test-dir` | – | override individual splits (any location) |
| `--epochs` | 200 | number of epochs |
| `--batch-size` | 16 | cover/stego **pairs** per step → 32 images |
| `--lr` | 0.01 | initial SGD learning rate (momentum 0.9) |
| `--milestones` | 80 130 170 | epochs where LR is multiplied by `--gamma` (0.1) |
| `--weight-decay` | 5e-4 | applied to weights only, not to biases/BN |
| `--best-after` | 170 | only epochs after this can become `best.pt` (as in the original) |
| `--eval-every` | 1 | validate every N epochs |
| `--no-bn-adjust` | off | skip the BN-statistics refresh pass before evaluation (≈2× faster epochs) |
| `--no-test` | off | skip the final test-set evaluation |
| `--threshold` | 0.5 | P(stego) ≥ threshold → stego, for all reported metrics |
| `--no-roc` | off | do not save the ROC curve of the final test |
| `--workers` | 4 | DataLoader worker processes (use `0` if you hit problems on Windows) |
| `--device` | auto | `auto`, `cpu`, `cuda`, `cuda:1`, `mps` |
| `--seed` | none | random seed for reproducibility |

While training runs, each epoch prints its progress and then its timing:

```
===== Epoch 12/200  lr=0.01 =====
  epoch 12/200  batch 100/875  loss 0.6412 (avg 0.6534)  acc 0.6231  | 41.7 img/s  elapsed 1m 17s  ETA 9m 58s
  ...
Val (epoch 12): acc 0.7180 | loss 0.5731 | bal acc 0.7180 | F1 0.7003 | P_E 0.2820 | AUC 0.7902 | 2000 images
Epoch 12/200 took 15m 41s (train 11m 12s, BN refresh 3m 40s, val 49s; waiting for data 4% of the training part)
Elapsed 3h 08m 22s | average epoch 15m 42s | 188 epoch(s) left, ETA 2d 01h 12m (2026-09-17 08:41)
```

and at the end:

```
Training finished: 200 epoch(s) in 2d 04h 20m (average 15m 41s per epoch, fastest 14m 02s, slowest 18m 10s)
```

"waiting for data" is the share of the training part spent waiting for the DataLoader instead of
computing. If it is high (say above 20 %), raise `--workers`; if it is near zero, the GPU is the
bottleneck. On GPU the timers synchronise first, so they measure finished work, not queued work. The
elapsed time is stored in the checkpoint, so after `--resume` the total also counts the earlier runs.

Everything for one run is written to `outputs/<run-name>/`:

| File | Content |
|---|---|
| `best.pt` | checkpoint with the best validation accuracy (after `--best-after`) |
| `last.pt` | checkpoint after the latest epoch – use it with `--resume` |
| `train.log` | full training log |
| `config.json` | all arguments of the run |
| `test_metrics.json` | full metric report on the test set (see [Metrics](#metrics)) |
| `test_roc.png` | ROC curve of the test set |
| `test_roc_points.csv` | the ROC points (threshold, FPR, TPR) for plotting elsewhere |

---

## Testing

`src/test.py` loads a checkpoint and evaluates it.

**Folders are searched recursively**, and every image is labelled by the nearest folder called `cover`
(0) or `stego` (1) *above* it. So you can point the script at a single folder, at several folders, or at
the whole dataset tree – no pairing and no fixed depth required:

```bash
python src/test.py --weights outputs/suniward_0.4/best.pt                     # data/test (default)
python src/test.py --weights outputs/suniward_0.4/best.pt --input data        # the whole tree
python src/test.py --weights outputs/suniward_0.4/best.pt --input data/val data/test
python src/test.py --weights outputs/suniward_0.4/best.pt --input data/test/cover
```

With `--input data` all six folders (`data/{train,val,test}/{cover,stego}`) are evaluated in one run and
reported **together as a single result** – one confusion matrix, one set of metrics, one ROC curve over
every image found. Sub-folders inside `cover/` or `stego/` are fine, the label comes from the nearest
matching ancestor. Images with no `cover`/`stego` folder above them are still predicted, just not scored,
and duplicate paths are only evaluated once.

If you ever do need a breakdown, run the script once per folder, or export the per-image CSV with `-o`
and group it yourself.

Save the predictions, the report and the ROC curve as files:

```bash
python src/test.py --weights outputs/suniward_0.4/best.pt --input data \
                   -o predictions.csv -m metrics.json --roc roc.png
```

`-o` writes one row per image, `-m` the metric report as JSON, `--roc` the curve as an image (plus
`roc.csv` with the raw points next to it). The console report over all labelled images looks like this:

```
--------------------------------------------------------------
All labelled images - best.pt  (10000 images: 5000 cover, 5000 stego)
--------------------------------------------------------------
Confusion matrix (rows = true, cols = predicted):
                 cover     stego
   cover        4402       598   (TN, FP)
   stego         731      4269   (FN, TP)

  Accuracy                  : 0.8671
  Balanced accuracy         : 0.8671
  Precision                 : 0.8772
  Recall / TPR              : 0.8538
  F1-score                  : 0.8653
  Specificity / TNR         : 0.8804
  False-alarm rate (FPR)    : 0.1196
  Missed-detection rate(FNR): 0.1462
  P_E = (FPR + FNR) / 2     : 0.1329
  ROC AUC                   : 0.9387
  Decision threshold        : 0.50
--------------------------------------------------------------
```

(The numbers above are only an illustration of the format.)

Images without labels are simply listed with their prediction:

```bash
python src/test.py --weights outputs/suniward_0.4/best.pt --input photo.png folder_with_images/
```

```
prediction  P(stego) tiles  true   file
cover         0.0312     1         folder_with_images/a.png
stego         0.9741     6         photo.png
```

Handling of image sizes in `test.py`:

* exactly 256 × 256 → used directly;
* larger → split into non-overlapping 256 × 256 tiles (`--tile-mode tile`, default) and the tile
  probabilities are averaged (`--aggregate mean`, or `max` to flag an image if *any* tile looks stego);
  `--tile-mode center` uses a single centre crop instead;
* smaller than 256 in any dimension → skipped with a message.

Images are **never resized** – interpolation would erase the stego signal.

Other options: `--threshold` (P(stego) decision threshold, default 0.5), `--batch-size` (tiles per
forward pass – lower it if you run out of memory), `--output/-o` (CSV with every prediction),
`--metrics/-m` (JSON report), `--roc` (ROC curve image), `--verbose/-v` (list every image, also the
labelled ones), `--device`.

`test.py` also accepts checkpoints produced by the original `CTNet.py`.

---

## Metrics

Both `train.py` (final test) and `test.py` (`--metrics`) write the same JSON report. Positive class =
**stego**, negative class = **cover**.

```json
{
  "checkpoint": "outputs/suniward_0.4/best.pt",
  "split": "data/test",
  "images": 10000,
  "cover_images": 5000,
  "stego_images": 5000,
  "threshold": 0.5,
  "TP": 4269, "TN": 4402, "FP": 598, "FN": 731,
  "accuracy": 0.8671,
  "balanced_accuracy": 0.8671,
  "precision": 0.8772,
  "recall_TPR": 0.8538,
  "f1_score": 0.8653,
  "specificity_TNR": 0.8804,
  "false_alarm_rate_FPR": 0.1196,
  "missed_detection_rate_FNR": 0.1462,
  "P_E": 0.1329,
  "AUC": 0.9387,
  "loss": 0.3187,
  "cover_accuracy": 0.8804,
  "stego_accuracy": 0.8538
}
```

| Field | Definition | Meaning here |
|---|---|---|
| `TP` | stego predicted stego | hidden message correctly found |
| `TN` | cover predicted cover | clean image correctly cleared |
| `FP` | cover predicted stego | false accusation |
| `FN` | stego predicted cover | hidden message missed |
| `accuracy` | (TP+TN)/N | fraction correct; only meaningful when the classes are balanced |
| `balanced_accuracy` | (TPR+TNR)/2 | accuracy corrected for unbalanced classes; equals `1 - P_E` |
| `precision` | TP/(TP+FP) | of the images flagged as stego, how many really are |
| `recall_TPR` | TP/(TP+FN) | of the stego images, how many were found (detection rate) |
| `f1_score` | 2·P·R/(P+R) | harmonic mean of precision and recall |
| `specificity_TNR` | TN/(TN+FP) | of the cover images, how many were cleared |
| `false_alarm_rate_FPR` | FP/(TN+FP) | cover images wrongly flagged = `1 - specificity` |
| `missed_detection_rate_FNR` | FN/(TP+FN) | stego images missed = `1 - recall` |
| `P_E` | (FPR+FNR)/2 | the standard steganalysis error; 0.5 = useless, 0 = perfect |
| `AUC` | area under the ROC curve | threshold-independent ranking quality |
| `loss` | cross-entropy | only in the report written by `train.py` |

`cover_accuracy` / `stego_accuracy` are duplicates of `specificity_TNR` / `recall_TPR`, kept because those
names are common in the steganalysis literature. Rates that are undefined (e.g. `precision` when nothing
was flagged, or anything needing a class that is absent) are written as `null`.

Everything except `AUC` depends on `--threshold`; only the counts and rates change with it, `AUC` does not.
Raise the threshold if false alarms are more costly than misses, and lower it in the opposite case.

### ROC curve

The ROC curve plots the detection rate (TPR) against the false-alarm rate (FPR) for **every** possible
threshold, so it shows the whole trade-off instead of a single operating point. The diagonal is random
guessing; the further the curve bulges toward the top-left corner, the better the detector. The marked
point is the best trade-off in the Youden sense (largest TPR − FPR), which is a reasonable default
threshold when false alarms and misses are equally bad.

`train.py` writes `test_roc.png` + `test_roc_points.csv` at the end of training (disable with `--no-roc`);
`test.py` writes them with `--roc <file>`. The image format follows the extension (`.png`, `.pdf`, `.svg`).
The CSV holds every (threshold, FPR, TPR) triple, so the curve can be re-plotted or compared with other
detectors in Excel, gnuplot or matplotlib. A ROC needs both classes, so it is only produced for labelled
input. If matplotlib is not installed, only the CSV is written and a warning is shown.

---

## Model overview

| Stage | Output (per image) |
|---|---|
| RGB input (values 0–255, not normalised) | 3 × 256 × 256 |
| **Pre-processing**: the same fixed bank of 62 high-pass filters (30 SRM + 32 Gabor, 5 × 5) applied to each colour channel, TLU clamp to [-5, 5], concatenation | 186 × 256 × 256 |
| `group2`: 2 × (conv 3×3 → BN → ReLU) → avg-pool /2 | 64 × 128 × 128 |
| `group3`: 2 × (conv 3×3 → BN → ReLU) → avg-pool /2 | 128 × 64 × 64 |
| `group4`: conv 3×3 → BN → ReLU → avg-pool /2 | 256 × 32 × 32 |
| **Transformer** (1 layer, 2 heads, head dim 4, MLP 4096) over 256 channel-tokens of length 1024 | 256 × 1024 → 256 × 32 × 32 |
| **iSQRT-COV pooling**: covariance → matrix square root (5 Newton–Schulz iterations) → upper triangle | 32 896 |
| FC 2048 → Dropout 0.7 → FC 2 | logits (cover, stego) |

≈ 76.5 M trainable parameters, of which ≈ 67 M are in the first fully-connected layer.

---

## Remarks

### What changed compared with the original `CTNet.py`

* **Structure** – one file was split into `ctnet.py` (model only), `datasets.py`, `train.py`, `test.py`
  and `utils.py`. Hard-coded absolute dataset paths and `.npy` index lists were replaced by the
  `data/{train,val,test}/{cover,stego}` folder layout.
* **Missing dependency bundled** – the original imported `MPNCOV` (iSQRT-COV pooling), which was not in the
  repository. It is re-implemented in `ctnet.py` with standard differentiable PyTorch operations. It was
  checked against the original MPNCOV code: forward output and gradients agree to floating-point precision
  (~1e-15 in float64), and the complete CTNet gives the same outputs as the original network with the
  same weights.
* **Fewer dependencies** – unused/obsolete imports were removed (`distutils` no longer exists in
  Python ≥ 3.12, `scipy`, `torchvision` only provided `Compose`).
* **Checkpoint compatibility** – parameter names and checkpoint keys (`original_state`,
  `optimizer_state`, `epoch`) are unchanged, so old checkpoints load in both `train.py --resume` and
  `test.py`.
* **Device** – automatic CPU / CUDA / Apple-MPS selection instead of hard-coded CUDA.
* **Robustness fixes** (training logic itself unchanged):
  * the learning rate is computed from the epoch number – same schedule as the original
    (drop at epochs 80, 130, 170), but also correct after resuming (the original restarted the scheduler);
  * the best validation accuracy is stored in the checkpoint and restored on resume (was reset to 0);
  * evaluating the test set no longer overwrites checkpoints (the original's `evaluate()` also saved
    models when called on the test set);
  * if no `best.pt` exists (e.g. training shorter than `--best-after`), the final test falls back to `last.pt`,
    and `--best-after` is automatically lowered when it is ≥ `--epochs`;
  * checkpoints are written atomically, so an interrupted save cannot corrupt them;
  * images are read via `cv2.imdecode`, so paths with non-ASCII characters work on Windows.
* **Flexible test input** – `test.py` takes any mix of files and folders, searches folders recursively and
  labels each image by the nearest `cover`/`stego` folder above it, so the whole `data` tree can be
  evaluated in one combined run. The original had no inference script at all.
* **Timing** – the original printed a per-batch time that was hard to interpret. Each epoch now reports
  throughput and ETA while it runs, then its own duration split into training / BN refresh / validation,
  the elapsed time and the projected finish; the end of the run reports the total and the average epoch.
* **Extra metrics** – the original only printed accuracy. Both scripts now report the confusion matrix
  (TP/TN/FP/FN), accuracy, balanced accuracy, precision, recall/TPR, F1, specificity, false-alarm rate,
  missed-detection rate, P_E and ROC AUC, save them as JSON, and plot the ROC curve; `test.py` can also
  export a per-image CSV.

### Original behaviour kept on purpose

These are properties of the published model; they were preserved so that results remain comparable:

* **`group5` is an identity.** In the original code the 1×1 conv block computes its output but returns the
  input. Its layers are still defined (for checkpoint compatibility) but not executed – this changes neither
  outputs nor gradients. `PointwiseBlock(use_conv=True)` enables it for experiments.
* **Channels are RGB**, although the original variable names suggest YUV.
* **Pixel values are not normalised** (0–255); the high-pass filters remove the image content anyway.
* **BN-statistics refresh** – before every validation (and before the final test) the whole training set is
  passed through the network in train mode without gradients, to update BatchNorm running statistics. This
  roughly doubles the time per epoch; `--no-bn-adjust` disables it.
* **Best model only after epoch 170** (`--best-after`), i.e. after the last LR drop.
* `--batch-size` counts **pairs**, so the default 16 means 32 images per step.

### Practical notes and limitations

* **A detector is only as general as its training data.** CTNet learns the traces of one steganographic
  algorithm, one payload and one image source. On images from a different camera/processing pipeline
  ("cover-source mismatch") or with another embedding algorithm, accuracy can drop sharply. A high
  P(stego) on a random image from the internet is a hint, not proof.
* **Spatial domain only.** CTNet targets spatial-domain embedding in lossless images. Stego images that were
  later JPEG-compressed, resized or filtered have lost most of the signal, and JPEG-domain steganography
  (J-UNIWARD, UERD, …) needs a differently configured detector.
* **Fixed input size.** Training requires 256 × 256 images. Crop (do not resize) larger images before
  creating stego versions; `test.py` tiles large images automatically.
* **Memory.** Most memory goes into the full-resolution 186-channel residual maps. Measured on CPU, a
  training step needs roughly 0.2 GB per image on top of ≈ 1.5 GB for the model and optimiser state, so the
  default 32 images per step need a GPU with a lot of memory (GPU usage is usually somewhat higher than the
  CPU measurement). Reduce `--batch-size` on `CUDA out of memory`.
* **Training time.** With 14 000 pairs and 200 epochs (plus the BN refresh pass), expect days on a single
  GPU. For quick experiments use fewer epochs with proportionally earlier `--milestones`.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `No cover/stego pairs found` | Check the folder layout and that cover/stego files have identical names. |
| `... size WxH, but CTNet is trained on 256x256 images` | Crop the dataset to 256 × 256 (training/validation only). |
| `CUDA out of memory` | Lower `--batch-size` (e.g. 8 or 4). |
| DataLoader hangs / errors on Windows | Run with `--workers 0`. |
| Training much slower than expected | Check the "waiting for data" share in the epoch line: high means raise `--workers`, near zero means the GPU is the limit. |
| `torch.cuda.is_available()` is `False` | The CPU build of PyTorch is installed – reinstall PyTorch with the CUDA command from pytorch.org. |
| No ROC image, only the `.csv` | matplotlib is missing: `pip install matplotlib`. |
| Validation accuracy stays at 0.5 | Normal during the first epochs; if it persists, check that stego images really differ from their covers and are stored losslessly. Lower payloads take much longer to learn (curriculum training from a higher payload helps). |

---

## Citation

```bibtex
@article{wei2025ctnet,
  title   = {CTNet: A Convolutional Transformer Network for Color Image Steganalysis},
  author  = {Wei, Kang-Kang and Luo, Wei-Qi and Tan, Shun-Quan and Huang, Ji-Wu},
  journal = {Journal of Computer Science and Technology},
  volume  = {40},
  number  = {2},
  pages   = {413--427},
  year    = {2025},
  doi     = {10.1007/s11390-023-3006-3}
}
```

The iSQRT-COV pooling follows: P. Li, J. Xie, Q. Wang, Z. Gao. *Towards Faster Training of Global
Covariance Pooling Networks by Iterative Matrix Square Root Normalization.* CVPR 2018.
