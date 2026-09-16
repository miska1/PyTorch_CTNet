"""
Data loading for CTNet.

Expected layout (each split already separated into cover / stego):

    data/
      train/  cover/  stego/
      val/    cover/  stego/
      test/   cover/  stego/

Training and validation use *paired* samples, exactly like the original
code: a cover image and its stego version (same file name) always go into
the same mini-batch and receive the same augmentation. Pairs are matched by
relative path without extension, so ``cover/00001.ppm`` pairs with
``stego/00001.ppm`` (or ``stego/00001.png``).
"""

import logging
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTENSIONS = {".ppm", ".pgm", ".pnm", ".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg", ".webp"}
LOSSY_EXTENSIONS = {".jpg", ".jpeg", ".webp"}
COVER_DIR = "cover"
STEGO_DIR = "stego"


class DatasetError(RuntimeError):
    """Raised for problems with the data folders (missing dirs, no pairs, ...)."""


# --------------------------------------------------------------------------- #
#  Image I/O                                                                   #
# --------------------------------------------------------------------------- #
def read_image(path):
    """Read an image as uint8 RGB, channel-first: (3, H, W).

    * uses cv2.imdecode so non-ASCII (e.g. Windows) paths work,
    * grayscale images are replicated to 3 channels,
    * an alpha channel is dropped,
    * pixel values stay in [0, 255] (no normalisation, as in the original).
    """
    path = Path(path)
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Could not read image: {path}")
    if img.dtype != np.uint8:
        raise ValueError(f"Only 8-bit images are supported, got {img.dtype} for {path}")

    if img.ndim == 2:                                   # grayscale
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 4:                             # BGRA -> BGR
        img = img[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(np.transpose(img, (2, 0, 1)))


def list_images(folder, recursive=True):
    """Sorted list of image files inside ``folder``."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    it = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS)


def _key(path, root):
    """Pairing key: relative path without extension (posix style)."""
    return path.relative_to(root).with_suffix("").as_posix()


def find_pairs(split_dir):
    """Match cover/stego files of one split. Returns list of (cover_path, stego_path)."""
    split_dir = Path(split_dir)
    if not split_dir.is_dir():
        raise DatasetError(f"Folder not found: {split_dir}")
    cover_root, stego_root = split_dir / COVER_DIR, split_dir / STEGO_DIR
    missing = [d.name for d in (cover_root, stego_root) if not d.is_dir()]
    if missing:
        have = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
        raise DatasetError(f"{split_dir} has no {' and no '.join(missing)} sub-folder "
                           f"(sub-folders found: {', '.join(have) if have else 'none'})")

    covers = {_key(p, cover_root): p for p in list_images(cover_root)}
    stegos = {_key(p, stego_root): p for p in list_images(stego_root)}
    common = sorted(covers.keys() & stegos.keys())

    n_cover_only = len(covers) - len(common)
    n_stego_only = len(stegos) - len(common)
    if n_cover_only or n_stego_only:
        logging.warning(f"[{split_dir.name}] {n_cover_only} cover and {n_stego_only} stego files have no "
                        f"partner with the same name and are ignored.")
    if not common:
        raise DatasetError(
            f"No cover/stego pairs found in {split_dir}. Cover and stego images must share file names "
            f"(e.g. cover/00001.ppm <-> stego/00001.ppm). Found {len(covers)} cover, {len(stegos)} stego files.")

    lossy = [covers[k] for k in common[:50] if covers[k].suffix.lower() in LOSSY_EXTENSIONS]
    if lossy:
        logging.warning(f"[{split_dir.name}] lossy files (e.g. {lossy[0].name}) detected. CTNet is a "
                        f"spatial-domain detector; re-compressed stego images lose their embedding signal.")
    return [(covers[k], stegos[k]) for k in common]


def label_from_path(path):
    """Label an image by the nearest folder called 'cover' (0) or 'stego' (1)
    anywhere above it; None if there is no such folder.

        data/test/cover/a.png            -> 0
        data/train/stego/sub/b.png       -> 1
        photos/holiday.png               -> None
    """
    p = Path(path).resolve()
    for parent in p.parents:
        name = parent.name.lower()
        if name == COVER_DIR:
            return 0
        if name == STEGO_DIR:
            return 1
    return None


# --------------------------------------------------------------------------- #
#  Transforms (original AugData / ToTensor)                                   #
# --------------------------------------------------------------------------- #
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class AugData:
    """Random 90-degree rotation + random vertical flip, applied identically
    to the cover and stego image of a pair. data shape: (2, C, H, W)."""

    def __call__(self, sample):
        data, label = sample["data"], sample["label"]
        rot = random.randint(0, 3)
        data = np.rot90(data, rot, axes=[2, 3]).copy()
        if random.random() < 0.5:
            data = np.flip(data, axis=2).copy()
        return {"data": data, "label": label}


class ToTensor:
    """uint8 numpy -> float32 tensor (values kept in [0, 255], no scaling)."""

    def __call__(self, sample):
        data, label = sample["data"], sample["label"]
        return {
            "data": torch.from_numpy(data.astype(np.float32)),
            "label": torch.from_numpy(label).long(),
        }


def train_transform():
    return Compose([AugData(), ToTensor()])


def eval_transform():
    return Compose([ToTensor()])


# --------------------------------------------------------------------------- #
#  Datasets                                                                    #
# --------------------------------------------------------------------------- #
class CoverStegoPairDataset(Dataset):
    """Paired cover/stego dataset for one split directory (data/train, data/val, data/test).

    Each item is a dict:
        data  : tensor (2, 3, H, W)   -> [cover, stego]
        label : tensor (2,)           -> [0, 1]
    A DataLoader batch of size B therefore holds 2*B images; reshape with
    ``data.reshape(-1, *data.shape[2:])`` and ``label.reshape(-1)``.
    """

    def __init__(self, split_dir, transform=None, image_size=256):
        self.split_dir = Path(split_dir)
        self.pairs = find_pairs(self.split_dir)
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        cover_path, stego_path = self.pairs[idx]
        cover = read_image(cover_path)
        stego = read_image(stego_path)

        for p, img in ((cover_path, cover), (stego_path, stego)):
            if img.shape[1:] != (self.image_size, self.image_size):
                raise ValueError(f"{p}: size {img.shape[2]}x{img.shape[1]}, but CTNet is trained on "
                                 f"{self.image_size}x{self.image_size} images. Crop your dataset first.")

        sample = {"data": np.stack([cover, stego]), "label": np.array([0, 1], dtype="int32")}
        if self.transform:
            sample = self.transform(sample)
        return sample


def make_tiles(img, size=256, mode="tile"):
    """Split a (3, H, W) image into (T, 3, size, size) patches for inference.

    mode = "tile"   : all non-overlapping size x size tiles (border remainder is dropped)
    mode = "center" : a single centre crop
    Images smaller than ``size`` in any dimension raise ValueError (no resizing:
    interpolation would destroy the stego noise the network looks for).
    """
    _, h, w = img.shape
    if h < size or w < size:
        raise ValueError(f"image is {w}x{h}, smaller than the required {size}x{size}")

    if mode == "center" or (h == size and w == size):
        top, left = (h - size) // 2, (w - size) // 2
        return img[None, :, top:top + size, left:left + size]

    if mode != "tile":
        raise ValueError(f"unknown tile mode: {mode}")
    tiles = [img[:, y:y + size, x:x + size]
             for y in range(0, h - size + 1, size)
             for x in range(0, w - size + 1, size)]
    return np.stack(tiles)


class InferenceImageDataset(Dataset):
    """Arbitrary images for test.py. Item: (tiles float tensor (T,3,S,S), index) or (None, index) on error."""

    def __init__(self, paths, image_size=256, tile_mode="tile"):
        self.paths = list(paths)
        self.image_size = image_size
        self.tile_mode = tile_mode

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = read_image(self.paths[idx])
            tiles = make_tiles(img, self.image_size, self.tile_mode)
            return torch.from_numpy(tiles.astype(np.float32)), idx, ""
        except Exception as e:                      # keep going on bad files
            return None, idx, str(e)


def inference_collate(batch):
    """Keep variable number of tiles per image: return the list unchanged."""
    return batch
