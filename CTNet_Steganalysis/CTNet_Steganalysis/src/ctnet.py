"""
CTNet: CNN + Transformer network for colour-image spatial steganalysis.

This module contains ONLY the model. Training, data loading and evaluation
live in train.py / datasets.py / test.py.

The architecture is a 1:1 port of the original ``CTNet.py``:

    RGB 3x256x256
      -> per-channel fixed high-pass bank (30 SRM + 32 Gabor = 62 filters) + TLU(5)
      -> concat                                   186 x 256 x 256
      -> group2  (2x conv3x3 + BN + ReLU, avgpool) 64 x 128 x 128
      -> group3  (2x conv3x3 + BN + ReLU, avgpool) 128 x 64 x 64
      -> group4  (1x conv3x3 + BN + ReLU, avgpool) 256 x 32 x 32
      -> flatten spatial -> Transformer (256 tokens of dim 1024)
      -> reshape back                               256 x 32 x 32
      -> group5  (identity, see note in PointwiseBlock)
      -> iSQRT-COV global covariance pooling (5 Newton-Schulz iterations)
      -> upper-triangle vector (256*257/2 = 32896)
      -> FC 2048 -> Dropout(0.7) -> FC 2   (logits: 0 = cover, 1 = stego)

Sub-module attribute names (group1..group5, transformer, fc1, dropout, fc2)
are kept identical to the original so checkpoints produced by the original
script load with ``model.load_state_dict(ckpt["original_state"])``.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from srm_filter_kernel import all_normalized_hpf_list

# The architecture is tied to 256x256 input:
#   after 3 stride-2 poolings the map is 32x32 -> 1024 tokens-dim for the
#   transformer, which is then folded back into a (1024 / DIM) x DIM map.
IMAGE_SIZE = 256
DIM = 32
NUM_CLASSES = 2          # 0 = cover, 1 = stego
CLASS_NAMES = ("cover", "stego")


# --------------------------------------------------------------------------- #
#  Pre-processing: fixed high-pass filter bank                                #
# --------------------------------------------------------------------------- #
class TLU(nn.Module):
    """Truncated Linear Unit: clamps activations to [-threshold, threshold]."""

    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def forward(self, x):
        return torch.clamp(x, min=-self.threshold, max=self.threshold)


def build_filters():
    """30 normalised SRM kernels (zero-padded to 5x5) + 32 Gabor kernels."""
    filters = []
    ksize = 5
    sigma = [0.5, 1.0]
    phi = [0, np.pi / 2]

    for hpf_item in all_normalized_hpf_list:
        row_1 = int((5 - hpf_item.shape[0]) / 2)
        row_2 = int((5 - hpf_item.shape[0]) - row_1)
        col_1 = int((5 - hpf_item.shape[1]) / 2)
        col_2 = int((5 - hpf_item.shape[1]) - col_1)
        hpf_item = np.pad(hpf_item, pad_width=((row_1, row_2), (col_1, col_2)), mode="constant")
        filters.append(hpf_item.astype(np.float32))

    # Gabor: 8 orientations (0, 22.5, ..., 157.5 deg) x 2 sigmas x 2 phases
    for theta in np.arange(0, np.pi, np.pi / 8):
        for k in range(2):
            for j in range(2):
                kern = cv2.getGaborKernel((ksize, ksize), sigma[k], theta, sigma[k] / 0.56,
                                          0.5, phi[j], ktype=cv2.CV_32F)
                filters.append(kern.astype(np.float32))
    return filters


class HPF(nn.Module):
    """Non-trainable 62-filter high-pass bank applied to a single channel."""

    def __init__(self):
        super().__init__()
        filt = np.stack(build_filters()).astype(np.float32)           # (62, 5, 5)
        hpf_weight = nn.Parameter(torch.from_numpy(filt).view(62, 1, 5, 5), requires_grad=False)

        self.hpf = nn.Conv2d(1, 62, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = hpf_weight
        self.tlu = TLU(5.0)

    def forward(self, x):
        return self.tlu(self.hpf(x))


# --------------------------------------------------------------------------- #
#  Convolutional blocks                                                        #
# --------------------------------------------------------------------------- #
class DoubleConvPoolBlock(nn.Module):
    """(original: block1) 2x [conv3x3 -> BN -> ReLU] -> AvgPool(3, s=2)."""

    def __init__(self, inchannel, outchannel):
        super().__init__()
        self.basic = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
        )

    def forward(self, x):
        return self.basic(x)


class ConvPoolBlock(nn.Module):
    """(original: block2) conv3x3 -> BN -> ReLU -> AvgPool(3, s=2)."""

    def __init__(self, inchannel, outchannel):
        super().__init__()
        self.basic = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
        )

    def forward(self, x):
        return self.basic(x)


class PointwiseBlock(nn.Module):
    """(original: block3)

    NOTE: In the original code this block computes ``self.basic(x)`` but
    returns the *input* unchanged, i.e. it acts as an identity. That
    behaviour is preserved here so results match the published model.
    The layers are still registered so original checkpoints load cleanly;
    the dead branch is simply not executed (its output was discarded anyway,
    so this changes neither the output nor the gradients).
    Set ``use_conv=True`` to actually apply the 1x1 conv (an experiment,
    NOT the original behaviour).
    """

    def __init__(self, inchannel, outchannel, use_conv=False):
        super().__init__()
        self.use_conv = use_conv
        self.basic = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.basic(x) if self.use_conv else x


# --------------------------------------------------------------------------- #
#  Transformer (ViT-style, pre-norm)                                          #
# --------------------------------------------------------------------------- #
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# --------------------------------------------------------------------------- #
#  iSQRT-COV global covariance pooling (replaces external MPNCOV package)     #
# --------------------------------------------------------------------------- #
# The original script imported ``from MPNCOV.python import MPNCOV``, which was
# not part of the repository. The three functions below reproduce the same
# forward computation (Li et al., "Towards Faster Training of Global
# Covariance Pooling Networks by Iterative Matrix Square Root Normalization",
# CVPR 2018). They are written with plain differentiable tensor ops, so
# autograd yields the exact gradient of the same Newton-Schulz iteration that
# MPNCOV back-propagates by hand.

def covpool(x):
    """(B, C, H, W) -> (B, C, C) sample covariance (1/M normalised, M = H*W).

    Equivalent to MPNCOV's X @ I_hat @ X^T with I_hat = (1/M)(I - 1/M * 11^T),
    but without materialising the M x M centering matrix.
    """
    b, c, h, w = x.shape
    m = h * w
    x = x.reshape(b, c, m)
    x = x - x.mean(dim=2, keepdim=True)
    return x.bmm(x.transpose(1, 2)) / m


def sqrtm_newton_schulz(x, iter_n):
    """Approximate matrix square root via coupled Newton-Schulz iterations
    with pre-normalisation by trace and post-compensation by sqrt(trace)."""
    b, dim, _ = x.shape
    eye = torch.eye(dim, device=x.device, dtype=x.dtype).unsqueeze(0).expand(b, dim, dim)
    i3 = 3.0 * eye

    norm_a = x.diagonal(dim1=1, dim2=2).sum(-1)               # trace, (B,)
    a = x / norm_a.view(b, 1, 1)

    if iter_n < 2:
        zy = 0.5 * (i3 - a)
        yzy = a.bmm(zy)
    else:
        zy = 0.5 * (i3 - a)
        y = a.bmm(zy)
        z = zy
        for _ in range(1, iter_n - 1):
            zy = 0.5 * (i3 - z.bmm(y))
            y, z = y.bmm(zy), zy.bmm(z)
        yzy = 0.5 * y.bmm(i3 - z.bmm(y))

    return yzy * torch.sqrt(norm_a).view(b, 1, 1)


def triuvec(x):
    """(B, C, C) -> (B, C*(C+1)/2), row-major upper triangle (same order as MPNCOV)."""
    dim = x.shape[1]
    rows, cols = torch.triu_indices(dim, dim, device=x.device)
    return x[:, rows, cols]


# --------------------------------------------------------------------------- #
#  CTNet                                                                       #
# --------------------------------------------------------------------------- #
class CTNet(nn.Module):
    def __init__(self, sqrt_iters=5, dropout=0.7):
        super().__init__()
        self.sqrt_iters = sqrt_iters

        self.group1 = HPF()                                   # shared across R, G, B
        self.group2 = DoubleConvPoolBlock(186, 64)            # 3 * 62 = 186
        self.group3 = DoubleConvPoolBlock(64, 128)
        self.group4 = ConvPoolBlock(128, 256)
        self.group5 = PointwiseBlock(256, 256)

        self.transformer = Transformer(dim=1024, depth=1, heads=2, dim_head=4, mlp_dim=1024 * 4)

        self.fc1 = nn.Linear(int(256 * (256 + 1) / 2), 2048)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(2048, NUM_CLASSES)

    def forward(self, x):
        if x.dim() != 4 or x.shape[1] != 3 or x.shape[2] != IMAGE_SIZE or x.shape[3] != IMAGE_SIZE:
            raise ValueError(
                f"CTNet expects input of shape (B, 3, {IMAGE_SIZE}, {IMAGE_SIZE}), got {tuple(x.shape)}")

        # Same HPF bank applied to each colour channel separately
        # (the original code names them y/u/v, but the data is RGB).
        c0 = self.group1(x[:, 0:1, :, :])
        c1 = self.group1(x[:, 1:2, :, :])
        c2 = self.group1(x[:, 2:3, :, :])
        out = torch.cat([c0, c1, c2], dim=1)

        out = self.group2(out)
        out = self.group3(out)
        out = self.group4(out)                                # (B, 256, 32, 32)

        out = out.reshape(out.shape[0], out.shape[1], out.shape[2] * out.shape[3])  # (B, 256, 1024)
        out = self.transformer(out)
        out = out.unsqueeze(3)
        out = out.reshape(out.shape[0], out.shape[1],
                          int(out.shape[2] / DIM), int(out.shape[3] * DIM))       # (B, 256, 32, 32)

        out = self.group5(out)

        out = covpool(out)
        out = sqrtm_newton_schulz(out, self.sqrt_iters)
        out = triuvec(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


# Backwards-compatible alias for the original class name
Net = CTNet


def init_weights(module):
    """Original initialisation: Kaiming-normal for trainable convs, N(0, 0.01) for linears."""
    if type(module) == nn.Conv2d:
        if module.weight.requires_grad:
            nn.init.kaiming_normal_(module.weight.data, mode="fan_in", nonlinearity="relu")
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight.data, mean=0, std=0.01)


def build_model(device="cpu", init=True):
    """Create a CTNet, optionally with the original weight initialisation."""
    model = CTNet()
    if init:
        model.apply(init_weights)
    return model.to(device)


if __name__ == "__main__":
    # Quick sanity check: python src/ctnet.py
    net = build_model()
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"CTNet trainable parameters: {n_train:,}")
    with torch.no_grad():
        y = net.eval()(torch.randint(0, 256, (2, 3, IMAGE_SIZE, IMAGE_SIZE)).float())
    print("output shape:", tuple(y.shape))
