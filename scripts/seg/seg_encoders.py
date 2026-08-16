"""Encoder variants for seg_step002 — the only lever that can hit the drone budget.

seg_step001 CONFIRMED (forward-hook MAC count) that the head is free (~4 MMAC) and the
VGG16 b1-b4 encoder is 4.56 GMAC at 128px, i.e. ~5x over the 1 GMAC/frame ceiling. The
MACs are spread almost evenly across the four blocks (0.63 / 0.91 / 1.51 / 1.51 G), so
no single block is "the" cost — each lever has to attack the whole stack.

Every encoder here ends at stride 8 so the 16x16 grid and the head are unchanged; only
the encoder differs. Levers are kept ORTHOGONAL and are run one at a time (Compounding
Rule): depth (drop a block), width, conv factorisation, and input downsampling.

Pretrained weights only exist for the unmodified VGG shapes, so width/separable/stem
arms are randomly initialised. That is a real confound (architecture AND init change
together), which is why seg_step002 also runs a randomly-initialised full-VGG control:
scratch arms are read against THAT, never against the pretrained number.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from scripts.seg.seg_common import BRANCHES, C, MFPM, VGG16_LOCAL

CHANS = [64, 128, 256, 512]
REPS = [2, 2, 3, 3]


def _sep(c_in: int, c_out: int) -> nn.Sequential:
    """Depthwise 3x3 + pointwise 1x1. Cuts a 3x3 conv from 9*c_in*c_out to
    9*c_in + c_in*c_out MACs/px — ~8x at these widths."""
    return nn.Sequential(nn.Conv2d(c_in, c_in, 3, padding=1, groups=c_in),
                         nn.Conv2d(c_in, c_out, 1))


def build_vgg_shaped(width: float = 1.0, sep: bool = False, stem_stride: int = 1,
                     n_blocks: int = 4, reps: list[int] | None = None) -> tuple[nn.Sequential, int]:
    """VGG-shaped stack that always ends at stride 8.

    Three downsamples are needed for stride 8; a stride-2 stem spends one of them, so
    the remaining pools shift the deep (expensive) blocks to a coarser grid.

    `reps` overrides the per-block conv count (seg_step016). seg_step015 CONFIRMED that
    63% of E7's batch-1 latency is fixed per-launch overhead, so conv COUNT is a latency
    lever independent of MACs — trading depth for width at iso-MAC should be faster.
    """
    chans = [max(8, int(c * width)) for c in CHANS[:n_blocks]]
    reps = REPS if reps is None else reps
    n_pool = 3 - (1 if stem_stride == 2 else 0)
    layers, c_in, first = [], 3, True
    for bi in range(n_blocks):
        for _ in range(reps[bi]):
            if first or sep is False:
                layers += [nn.Conv2d(c_in, chans[bi], 3, padding=1,
                                     stride=stem_stride if first else 1)]
            else:
                layers += [_sep(c_in, chans[bi])]
            layers += [nn.ReLU(inplace=True)]
            c_in, first = chans[bi], False
        if bi < n_pool:
            layers += [nn.MaxPool2d(2, 2)]
    return nn.Sequential(*layers), c_in


def build_vgg_sliced(width: float = 0.5, n_blocks: int = 4,
                     reps: list[int] | None = None) -> tuple[nn.Sequential, int]:
    """Narrow VGG whose filters are SLICED out of the pretrained VGG16.

    seg_step002 CONFIRMED the ImageNet init is worth +9.4pp — 10-30x any architecture
    lever — while halving the width is free. Those two facts only combine if a narrow
    net can inherit the init, which random re-initialisation throws away. Slicing takes
    the first `width` fraction of each conv's output filters (and the matching input
    channels of the next conv), so the student starts from real ImageNet features at
    E3's cost instead of from noise. Filter order in VGG is arbitrary, so "first k" is
    an unbiased choice, not a selection heuristic.
    """
    net, _ = build_vgg_shaped(width=width, n_blocks=n_blocks, reps=reps)
    src, _ = build_vgg_pretrained(n_blocks=n_blocks)
    dst_convs = [m for m in net if isinstance(m, nn.Conv2d)]
    src_convs = [m for m in src if isinstance(m, nn.Conv2d)]

    # Pair BY BLOCK, not by position. With the default reps the two are identical, but once
    # `reps` shrinks a block (seg_step016) a positional zip would hand dst block-2's conv the
    # weights of src block-1's, whose channel counts do not even fit. Within a block, dst conv j
    # takes src conv j; a dst block deeper than src's is clamped to src's last conv, which has
    # the c->c shape every repeat after the first uses.
    r_dst = REPS[:n_blocks] if reps is None else reps
    src_base = [sum(REPS[:b]) for b in range(n_blocks)]
    pairs, d_i = [], 0
    for b in range(n_blocks):
        for j in range(r_dst[b]):
            pairs.append((dst_convs[d_i], src_convs[src_base[b] + min(j, REPS[b] - 1)]))
            d_i += 1

    with torch.no_grad():
        for d, s in pairs:
            o, i = d.weight.shape[:2]
            if o > s.weight.shape[0] or i > s.weight.shape[1]:
                raise ValueError(f"cannot slice {tuple(d.weight.shape)} out of "
                                 f"{tuple(s.weight.shape)} — width/reps combination too wide")
            d.weight.copy_(s.weight[:o, :i])
            d.bias.copy_(s.bias[:o])
    return net, dst_convs[-1].out_channels


def build_vgg_pretrained(n_blocks: int = 4) -> tuple[nn.Sequential, int]:
    """Real VGG16 features with the cached ImageNet weights.

    n_blocks=4 -> features[:23] (block4 convs, 512ch, stride 8, matches seg_step001).
    n_blocks=3 -> features[:17] (through pool3, 256ch, also stride 8) — the depth lever.
    """
    from torchvision.models import vgg16
    net = vgg16(weights=None)
    for p in VGG16_LOCAL:
        if p.exists():
            net.load_state_dict(torch.load(p, map_location="cpu"))
            break
    else:
        raise FileNotFoundError(f"no cached vgg16 weights in {VGG16_LOCAL}")
    cut, ch = (23, 512) if n_blocks == 4 else (17, 256)
    return net.features[:cut], ch


# arm -> (kind, kwargs, freeze_below, pretrained). freeze_below is an index into the
# encoder's module list; scratch arms train everything since there is nothing to keep.
ARMS = {
    "E0": ("pre", dict(n_blocks=4), 17, True),    # seg_step001 B1 config — reference
    "E1": ("shaped", dict(), 0, False),           # same shape, random init — scratch control
    "E2": ("pre", dict(n_blocks=3), 10, True),    # DEPTH: drop block4
    "E3": ("shaped", dict(width=0.5), 0, False),  # WIDTH: half channels
    "E4": ("shaped", dict(sep=True), 0, False),   # FACTORISATION: depthwise-separable
    "E5": ("shaped", dict(stem_stride=2), 0, False),  # RESOLUTION: stride-2 stem
    # seg_step003 — inherit the ImageNet init at a cheap width (the +9.4pp that E3 lost)
    "E6": ("sliced", dict(width=0.5), 0, True),                 # width 0.5x, sliced init
    "E7": ("sliced", dict(width=0.5, n_blocks=3), 0, True),     # + drop block4 (compound)
    # E8 is E7's exact architecture with a RANDOM init — the seg_step012 control that removes the
    # teacher from the student's weights entirely. seg_step011 falsified channel alignment by
    # shuffling channels that still came from the teacher; this removes them.
    "E8": ("shaped", dict(width=0.5, n_blocks=3), 0, False),
    # seg_step013 slice-ratio sweep: same lever as E7 at other widths. The hint needs no change —
    # it targets t_enc[:, :C_student], which tracks whatever width the student has.
    "E9": ("sliced", dict(width=0.25, n_blocks=3), 0, True),
    "E10": ("sliced", dict(width=0.75, n_blocks=3), 0, True),
    # seg_step016 — trade DEPTH for WIDTH at roughly iso-MAC against E7 (769M encoder MACs, 7 convs).
    # seg_step015 CONFIRMED 63% of E7's batch-1 latency is fixed per-launch cost, so fewer/fatter
    # kernels should win time that narrowing cannot. E11 = 4 convs, E12 = 3 convs; both still end at
    # stride 8, so the head and the 16x16 grid are untouched and only the encoder differs.
    "E11": ("sliced", dict(width=0.75, n_blocks=3, reps=[1, 1, 2]), 0, True),
    "E12": ("sliced", dict(width=1.0, n_blocks=3, reps=[1, 1, 1]), 0, True),
}


class SegNet(nn.Module):
    """Swappable encoder -> optional M_FPM -> 1x1 head of C*K channels.

    Identical to seg_step001's CoarseSegNet apart from the encoder, so any delta is
    attributable to the encoder alone.
    """

    def __init__(self, arm: str, k: int = 2, mfpm: bool = True, c_mid: int = 64,
                 branches=BRANCHES, gap: str = "none"):
        super().__init__()
        kind, kw, freeze_below, _ = ARMS[arm]
        self.k = k
        builder = {"pre": build_vgg_pretrained, "sliced": build_vgg_sliced}.get(
            kind, build_vgg_shaped)
        self.enc, c_out = builder(**kw)
        for i, m in enumerate(self.enc):
            if isinstance(m, nn.Conv2d) and i < freeze_below:
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)
        self.mfpm = MFPM(c_out, c_mid, branches, gap) if mfpm else None
        self.head = nn.Conv2d(self.mfpm.out_ch if mfpm else c_out, C * k, 1)

    def forward(self, x):
        z = self.enc(x)
        if self.mfpm is not None:
            z = self.mfpm(z)
        return self.head(z)

    def class_logits(self, out):
        b, _, h, w = out.shape
        return out.view(b, C, self.k, h, w).amax(2)
