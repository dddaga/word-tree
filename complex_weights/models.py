# TinySegNet — compact FCN for binary foreground segmentation.
# Baseline is all real convs. A single conv can be swapped for PhaseConv2d
# (multiply -> phase-addition) to isolate the science question:
#   does phase-addition preserve information at iso-parameter budget?
import torch
import torch.nn as nn
from phaseconv import PhaseConv2d, encode_phase


class RealBlock(nn.Module):
    def __init__(self, i, o, k=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(i, o, k, stride, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(o)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class PhaseBlock(nn.Module):
    """Drop-in for RealBlock: real in -> phase-encode -> PhaseConv -> BN -> real out.
    Transparent at the tensor boundary, so exactly one layer is 'complex'."""
    def __init__(self, i, o, k=3, stride=1, encode="wrap", variant="real", n_roots=0):
        super().__init__()
        self.conv = PhaseConv2d(i, o, k, stride, k // 2, variant=variant, n_roots=n_roots)
        self.bn = nn.BatchNorm2d(o)
        self.act = nn.ReLU(inplace=True)
        self.encode = encode

    def forward(self, x):
        phi = encode_phase(x, self.encode)
        return self.act(self.bn(self.conv(phi)))


def make_block(i, o, k, stride, phase, encode, variant, n_roots):
    return PhaseBlock(i, o, k, stride, encode, variant, n_roots) if phase else RealBlock(i, o, k, stride)


class TinySegNet(nn.Module):
    """~6-conv encoder-decoder, 2 downsamples. swap_layers: set of layer indices
    (0..5) to realize as PhaseConv instead of real conv."""
    LAYERS = 6

    def __init__(self, in_ch=3, width=16, swap_layers=(), encode="wrap", variant="real", n_roots=0):
        super().__init__()
        swap = set(swap_layers)
        w, w2 = width, width * 2
        mk = lambda idx, i, o, s: make_block(i, o, 3, s, idx in swap, encode, variant, n_roots)
        self.e0 = mk(0, in_ch, w, 1)
        self.e1 = mk(1, w, w, 2)     # /2
        self.e2 = mk(2, w, w2, 2)    # /4
        self.mid = mk(3, w2, w2, 1)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.d1 = mk(4, w2, w, 1)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.d2 = mk(5, w, w, 1)
        self.head = nn.Conv2d(w, 1, 1)

    def forward(self, x):
        x = self.e0(x)
        x = self.e1(x)
        x = self.e2(x)
        x = self.mid(x)
        x = self.d1(self.up1(x))
        x = self.d2(self.up2(x))
        return self.head(x)


def count_params(m):
    return sum(p.numel() for p in m.parameters())
