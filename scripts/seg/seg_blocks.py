"""Block families for seg_step024 — how to spend a fixed MAC budget, not how much to spend.

seg_step022 closed the pyramid question: `lean` (M_FPM without pool/d1) is the default student at
731,624p / 0.8450 GMAC, and no head-side change cleared the bar. The remaining design choice inside
the drone ceiling is the ENCODER BLOCK ITSELF — every seg arm so far has been a VGG-shaped 3x3 stack,
which is an inherited constraint, not a measured one.

THE CONFOUND THAT DICTATES THE DESIGN. seg_step003 CONFIRMED the sliced ImageNet init is worth
+4.26pp. None of the families here can inherit that slice (an inverted-residual or 7x7-depthwise
kernel has no VGG counterpart to slice from), so a family read against `lean` would be measuring
family AND init at once, with init the larger term. Every arm built here is therefore SCRATCH and is
read against E8 — the same architecture at random init — exactly as seg_encoders.py's docstring
requires. This step answers "which family is best at scratch"; finding a winner an init is a separate
question, not one this step can answer.

HELD FIXED so only the block type moves: 3 stages, reps [2,2,3], three 2x pools -> stride 8, and the
final stage pinned to 128 channels so the seg_step010 feature hint (which reads t_enc[:, :C_student])
is byte-identical across cells. Interior width is the only free variable and is solved per family to
land on the same GMAC as the VGG baseline — see solve_width() in the seg_step024 driver.

Residual note: the first block of each stage changes channel count and so runs WITHOUT a skip; the
repeats after it are c->c and carry one. That is the standard arrangement and keeps the parameter
cost of the family attributable to the block, not to projection shortcuts.
"""
from __future__ import annotations

import torch.nn as nn

STAGE_CHANS = [32, 64, 128]      # E7/E8 at width 0.5, n_blocks=3; stage 3 is PINNED (hint contract)
STAGE_REPS = [2, 2, 3]


class VggBlock(nn.Module):
    """Plain 3x3 conv + ReLU — the incumbent, reproduced here so all families share one harness."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(inplace=True))
        self.res = c_in == c_out

    def forward(self, x):
        y = self.body(x)
        return x + y if self.res else y


class SepBlock(nn.Module):
    """Depthwise 3x3 -> pointwise 1x1. ~8x cheaper than a dense 3x3 at these widths, which is
    exactly the budget this step is trying to re-spend on width."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c_in, c_in, 3, padding=1, groups=c_in),
            nn.Conv2d(c_in, c_out, 1),
            nn.ReLU(inplace=True))
        self.res = c_in == c_out

    def forward(self, x):
        y = self.body(x)
        return x + y if self.res else y


class IRBlock(nn.Module):
    """MobileNetV2 inverted residual: 1x1 expand -> depthwise 3x3 -> 1x1 project, no activation on
    the projection. Expansion is INSIDE the block, so at iso-GMAC it buys nonlinear width without
    widening the tensors that flow between stages."""

    def __init__(self, c_in: int, c_out: int, expand: int = 4):
        super().__init__()
        c_h = max(8, c_in * expand)
        self.body = nn.Sequential(
            nn.Conv2d(c_in, c_h, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c_h, c_h, 3, padding=1, groups=c_h), nn.ReLU(inplace=True),
            nn.Conv2d(c_h, c_out, 1))
        self.res = c_in == c_out

    def forward(self, x):
        y = self.body(x)
        return x + y if self.res else y


class ConvNeXtBlock(nn.Module):
    """ConvNeXt: depthwise 7x7 -> norm -> 1x1 expand 4x -> GELU -> 1x1 project.

    The 7x7 depthwise kernel is the point — at a 16x16 output grid a large receptive field per layer
    is cheap in MACs precisely because it is depthwise, which is the opposite trade to the dilated
    pyramid seg_step022 just measured. GroupNorm(1, c) stands in for LayerNorm here: it is the same
    normalisation over the channel dimension without the two permutes, which matter at this size.
    """

    def __init__(self, c_in: int, c_out: int, expand: int = 4):
        super().__init__()
        c_h = max(8, c_out * expand)
        self.body = nn.Sequential(
            nn.Conv2d(c_in, c_out, 7, padding=3, groups=c_in if c_in == c_out else 1),
            nn.GroupNorm(1, c_out),
            nn.Conv2d(c_out, c_h, 1), nn.GELU(),
            nn.Conv2d(c_h, c_out, 1))
        self.res = c_in == c_out

    def forward(self, x):
        y = self.body(x)
        return x + y if self.res else y


FAMILIES = {"vgg": VggBlock, "sep": SepBlock, "ir": IRBlock, "cnx": ConvNeXtBlock}


def build_block_stack(family: str, width: float = 1.0) -> tuple[nn.Sequential, int]:
    """Stride-8 stack of `family` blocks, final stage pinned to STAGE_CHANS[-1].

    `width` scales the two INTERIOR stages only. Stage 3 is fixed because the feature hint reads the
    encoder's output channels; letting it float would change the supervision along with the family
    and make the cells unreadable against each other.
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; have {list(FAMILIES)}")
    block = FAMILIES[family]
    chans = [max(8, int(round(c * width))) for c in STAGE_CHANS[:-1]] + [STAGE_CHANS[-1]]

    # A dense 3x3 stem for every family: the first conv sees 3 input channels, where depthwise and
    # inverted-residual variants are degenerate (groups=3) rather than cheap. Holding it fixed keeps
    # the delta attributable to the stages.
    layers: list[nn.Module] = [nn.Conv2d(3, chans[0], 3, padding=1), nn.ReLU(inplace=True)]
    c_in = chans[0]
    for si, c in enumerate(chans):
        for _ in range(STAGE_REPS[si]):
            layers.append(block(c_in, c))
            c_in = c
        layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers), c_in
