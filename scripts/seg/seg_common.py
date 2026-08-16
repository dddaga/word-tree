"""Shared pieces for the coarse-grid drone-surveillance segmentation line (seg_step0xx).

Reframe (user, 2026-08-14): object detection as COARSE semantic segmentation.
Input 128x128, output stride 8 -> 16x16 grid, C event classes x K channels.
Localisation accuracy is deliberately sacrificed; the grid heat-map plus the
per-class channel group is what a drone operator needs ("something happened
THERE, and it looks like THIS kind of event").

Supervision needs NO pixel masks and NO download: a cached torchvision detector
is run once over imagenette images and its boxes are rasterised onto the 16x16
grid as soft targets (same distillation trick as det_step002 / Branch C).

COCO ids here are the 91-category ids torchvision detectors emit.
"""
from __future__ import annotations
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models import vgg16
from torchvision.models.detection import (fasterrcnn_mobilenet_v3_large_fpn,
                                          FasterRCNN_MobileNet_V3_Large_FPN_Weights)

ROOT = Path(__file__).parent.parent.parent
EVENT_CLASSES = ["person", "vehicle", "animal", "object"]
C = len(EVENT_CLASSES)

_VEHICLE = {2, 3, 4, 6, 7, 8, 9}                      # bicycle car motorcycle bus train truck boat
_ANIMAL = set(range(16, 26))                          # bird..giraffe
COCO_TO_EVENT = {1: 0, **{i: 1 for i in _VEHICLE}, **{i: 2 for i in _ANIMAL}}

# VGG16 weights live in a sibling account's torch cache on this machine; use it
# read-only rather than re-downloading half a gigabyte.
VGG16_LOCAL = [Path.home() / ".cache/torch/hub/checkpoints/vgg16-397923af.pth",
               Path("/Users/indra/.cache/torch/hub/checkpoints/vgg16-397923af.pth")]


def event_of(coco_id: int) -> int:
    """Map a COCO category id onto our 4-way event taxonomy ('object' is the catch-all)."""
    return COCO_TO_EVENT.get(int(coco_id), 3)


def list_images(split: str = "train", limit: int = 2000) -> list[Path]:
    root = ROOT / "data" / "imagenette2-320" / split
    paths = sorted(p for wnid in sorted(root.iterdir()) if wnid.is_dir()
                   for p in sorted(wnid.glob("*.JPEG")))
    return paths[::max(1, len(paths) // limit)][:limit]


def load_image(path: Path, res: int) -> torch.Tensor:
    im = Image.open(path).convert("RGB").resize((res, res), Image.BILINEAR)
    x = torch.frombuffer(im.tobytes(), dtype=torch.uint8).float().view(res, res, 3) / 255.0
    x = x.permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (x - mean) / std


def rasterize(boxes, labels, scores, res: int, grid: int, thr: float = 0.3) -> torch.Tensor:
    """Boxes -> [C, grid, grid] soft target. Cell value = max detector score of any
    surviving detection of that event class overlapping the cell. Overlap is treated
    as binary: at 16x16 a cell is 8x8 px, so partial-coverage weighting is noise."""
    t = torch.zeros(C, grid, grid)
    s = res / grid
    for b, l, sc in zip(boxes, labels, scores):
        if sc < thr:
            continue
        e = event_of(l)
        x0, y0, x1, y1 = (b / s).tolist()
        i0, i1 = int(max(0, y0)), int(min(grid - 1, y1))
        j0, j1 = int(max(0, x0)), int(min(grid - 1, x1))
        t[e, i0:i1 + 1, j0:j1 + 1] = torch.maximum(t[e, i0:i1 + 1, j0:j1 + 1],
                                                   torch.tensor(float(sc)))
    return t


def build_teacher(device):
    m = fasterrcnn_mobilenet_v3_large_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1)
    return m.eval().to(device)


def _vgg_backbone():
    net = vgg16(weights=None)
    for p in VGG16_LOCAL:
        if p.exists():
            net.load_state_dict(torch.load(p, map_location="cpu"))
            break
    else:
        raise FileNotFoundError(f"no cached vgg16 weights in {VGG16_LOCAL}")
    return net.features[:23]          # through block4_conv3+relu -> stride 8


BRANCHES = ("pool", "d1", "d4", "d8", "d16")
RATES = {"d1": 1, "d4": 4, "d8": 8, "d16": 16}


class MFPM(nn.Module):
    """FgSegNet_v2's cascaded dilated pyramid, read off its source. Each branch sees the
    concat of the input and the PREVIOUS branch, so rates compound; concatenation adds
    rank where a multiplicative gate would not. At a 16x16 grid the rate-16 branch already
    covers the whole map, i.e. global context for ~7% of encoder MACs.

    `branches` selects a SUBSET (seg_step022's ablation). Dropping a dilated branch rewires the
    cascade onto the previous SURVIVING one, which is the only way to drop a stage without also
    severing the chain below it -- so a dropped middle branch is 'stage removed', not 'chain cut'.
    Attribute names and therefore state_dict keys are unchanged, so checkpoints written before
    this argument existed still load. The default reproduces the original module exactly:
    same submodules, same construction order (hence same RNG draws), byte-identical output.

    `gap` adds a TRUE global branch (seg_step023). The `pool` branch above is a 1x1 conv on a
    3x3 max-pool -- local, despite the name -- so before this there was no image-level vector
    anywhere in the net. 'cat' concatenates it (adds rank); 'se' multiplies the concat by a
    sigmoid of it (classic squeeze-excite). 'se' exists ONLY as the pre-registered control for
    the gate-death theorem (steps 873-916) and GLAM's mul-hurts/add-neutral T0 result: those
    predict 'cat' >= 'se'. If 'se' wins, both are falsified on this task and that is the finding.
    """

    def __init__(self, c_in: int = 512, c: int = 64, branches=BRANCHES, gap: str = "none"):
        super().__init__()
        self.branches = tuple(b for b in BRANCHES if b in branches)
        self.gap_mode = gap
        if not self.branches:
            raise ValueError("MFPM needs at least one branch")
        if "pool" in self.branches:
            self.pool = nn.Conv2d(c_in, c, 1)
        self.dilated, prev = [b for b in self.branches if b != "pool"], None
        for b in self.dilated:
            setattr(self, b, nn.Conv2d(c_in if prev is None else c_in + c, c, 3,
                                       padding=RATES[b], dilation=RATES[b]))
            prev = b
        self.out_ch = (len(self.branches) + (gap == "cat")) * c
        if gap != "none":
            self.g = nn.Conv2d(c_in, self.out_ch if gap == "se" else c, 1)
        self.norm = nn.GroupNorm(self.out_ch, self.out_ch)        # InstanceNorm equivalent
        self.drop = nn.Dropout2d(0.25)

    def forward(self, x):
        outs, prev = [], None
        if "pool" in self.branches:
            outs.append(self.pool(F.max_pool2d(x, 3, stride=1, padding=1)))
        for b in self.dilated:
            prev = getattr(self, b)(x if prev is None else F.relu(torch.cat([x, prev], 1)))
            outs.append(prev)
        if self.gap_mode == "cat":
            outs.append(self.g(x.mean((2, 3), keepdim=True)).expand(-1, -1, *x.shape[2:]))
        z = torch.cat(outs, 1)
        if self.gap_mode == "se":
            z = z * torch.sigmoid(self.g(x.mean((2, 3), keepdim=True)))
        return self.drop(F.relu(self.norm(z)))


class CoarseSegNet(nn.Module):
    """VGG16 b1-b4 (frozen except block4) -> M_FPM -> 1x1 head of C*K channels.

    K channels per class exist for explainability: confidence = max over K, and the
    argmax channel names WHICH sub-pattern of the class fired. They collapse to K
    copies unless pushed apart, which is what the decorrelation penalty is for."""

    def __init__(self, k: int = 2, mfpm: bool = True, c_mid: int = 64,
                 branches=BRANCHES, gap: str = "none"):
        super().__init__()
        self.k, self.enc = k, _vgg_backbone()
        for i, m in enumerate(self.enc):
            if isinstance(m, nn.Conv2d) and i < 17:   # block4 convs (17,19,21) stay trainable
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)
        self.mfpm = MFPM(512, c_mid, branches, gap) if mfpm else None
        self.head = nn.Conv2d(self.mfpm.out_ch if mfpm else 512, C * k, 1)

    def forward(self, x):
        z = self.enc(x)
        if self.mfpm is not None:
            z = self.mfpm(z)
        return self.head(z)                            # [B, C*K, G, G] logits

    def class_logits(self, out):
        """Collapse the K channels of each class by max -> [B, C, G, G]."""
        b, _, h, w = out.shape
        return out.view(b, C, self.k, h, w).amax(2)


def decor_penalty(out, k: int) -> torch.Tensor:
    """Anti-Hebbian: push the K channels within a class group apart. Inert in every
    prior test (A4b ~ A4, step003 S2 -0.30pp) because there was nothing it needed to
    separate; here the K-channel groups are exactly that target."""
    if k < 2:
        return out.new_zeros(())
    b, _, h, w = out.shape
    g = out.view(b, C, k, h * w)
    g = g - g.mean(-1, keepdim=True)
    g = g / (g.norm(dim=-1, keepdim=True) + 1e-6)
    corr = g @ g.transpose(-1, -2)                     # [B, C, k, k]
    off = corr - torch.diag_embed(torch.diagonal(corr, dim1=-2, dim2=-1))
    return off.pow(2).sum() / (b * C * k * (k - 1))
