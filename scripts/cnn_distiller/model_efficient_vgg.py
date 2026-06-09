"""EfficientVGG: Efficient CNN distillation of VGG16's conv+pool layers.

Configurable architecture — channels, kernel size, expansion, side branch, CReLU.
Always outputs [B, 25088] features (512×7×7, C4=512 fixed) for teacher feature
cosine distillation compatibility, plus [B, 10] logits.

All pooling ops use MaxPool (no AvgPool).

Spatial flow (input 3×224×224):
  Stem     : 3→C1, stride 2 + MaxPool2 → [B, C1, 56, 56]
  Block1   : ConvNeXt(C1) → [B, C1, 56, 56]
  SideBranch (opt): Block1 → [B, C2] channel attention for Block2
  DS1      : C1→C2, stride 2 → [B, C2, 28, 28]
  Block2   : ConvNeXt(C2) + optional attn → [B, C2, 28, 28]
  DS2      : C2→C3, stride 2 → [B, C3, 14, 14]
  Block3   : ConvNeXt(C3) ± CReLU → [B, C3, 14, 14]
  DS3      : C3→512, stride 2 → [B, 512, 7, 7]
  Flatten  : → [B, 25088]
  Head     : AdaptiveMaxPool + Linear(512→10)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def crelu(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    """Concatenated Leaky ReLU — doubles channel dim, captures pos+neg signal."""
    return torch.cat([F.leaky_relu(x, negative_slope), F.leaky_relu(-x, negative_slope)], dim=1)


class ChanLN(nn.Module):
    """Per-channel LayerNorm for [B, C, H, W] tensors (ConvNeXt convention)."""

    def __init__(self, C: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class DSConv(nn.Module):
    """Depthwise-separable conv for downsampling: DW(k×k) → ChanLN → PW."""

    def __init__(self, C_in: int, C_out: int, k: int = 3,
                 stride: int = 1, padding: int = 1) -> None:
        super().__init__()
        self.dw = nn.Conv2d(C_in, C_in, k, stride=stride, padding=padding,
                            groups=C_in, bias=False)
        self.ln = ChanLN(C_in)
        self.pw = nn.Conv2d(C_in, C_out, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.ln(self.dw(x)))


# ---------------------------------------------------------------------------
# ConvNeXt block: DW(k×k), ChanLN, inverted bottleneck (×expansion), GELU, residual
# ---------------------------------------------------------------------------

class ConvNeXtBlock(nn.Module):
    def __init__(self, C: int, expansion: int = 2, dw_kernel: int = 7) -> None:
        super().__init__()
        C_mid = C * expansion
        pad = dw_kernel // 2
        self.dw  = nn.Conv2d(C, C, dw_kernel, padding=pad, groups=C, bias=False)
        self.ln  = ChanLN(C)
        self.pw1 = nn.Conv2d(C, C_mid, 1, bias=False)
        self.pw2 = nn.Conv2d(C_mid, C, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pw2(F.gelu(self.pw1(self.ln(self.dw(x)))))


# ---------------------------------------------------------------------------
# ConvNeXt + CReLU: compress→CReLU(2×)→project, residual
# ---------------------------------------------------------------------------

class ConvNeXtBlockCReLU(nn.Module):
    def __init__(self, C: int, dw_kernel: int = 7) -> None:
        super().__init__()
        C_half = max(1, C // 2)
        pad = dw_kernel // 2
        self.dw  = nn.Conv2d(C, C, dw_kernel, padding=pad, groups=C, bias=False)
        self.ln  = ChanLN(C)
        self.pw1 = nn.Conv2d(C, C_half, 1, bias=False)    # compress
        self.pw2 = nn.Conv2d(C, C, 1, bias=False)          # project (CReLU doubles C_half→C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pw2(crelu(self.pw1(self.ln(self.dw(x)))))


# ---------------------------------------------------------------------------
# Side branch: Block1 out [B, C1, 56, 56] → [B, C2] sigmoid channel attention
# Applied as multiplicative gate on Block2 output (C2 channels).
# ---------------------------------------------------------------------------

class SideBranch(nn.Module):
    def __init__(self, C_in: int, C_att: int, dw_kernel: int = 7) -> None:
        super().__init__()
        C_half = max(1, C_att // 2)
        pad = dw_kernel // 2
        self.pool = nn.MaxPool2d(4)                                      # 56→14
        self.dw   = nn.Conv2d(C_in, C_in, dw_kernel, padding=pad,
                               groups=C_in, bias=False)
        self.ln   = ChanLN(C_in)
        self.pw   = nn.Conv2d(C_in * 2, C_in, 1, bias=False)            # merge after CReLU
        self.gap  = nn.AdaptiveMaxPool2d(1)
        self.fc   = nn.Linear(C_in, C_half)
        self.proj = nn.Linear(C_att, C_att)                              # after CReLU: C_half*2=C_att

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h    = self.pool(x)                        # [B, C_in, 14, 14]
        skip = h
        h    = crelu(self.ln(self.dw(h)))          # [B, C_in*2, 14, 14]
        h    = self.pw(h) + skip                   # [B, C_in, 14, 14]
        h    = self.gap(h).flatten(1)              # [B, C_in]
        h    = crelu(F.gelu(self.fc(h)))           # [B, C_att]
        return torch.sigmoid(self.proj(h))          # [B, C_att]


# ---------------------------------------------------------------------------
# EfficientVGG
# ---------------------------------------------------------------------------

class EfficientVGG(nn.Module):
    """EfficientVGG — configurable CNN distillation student for VGG16.

    Parameters
    ----------
    channels : tuple[int, int, int]
        (C1, C2, C3) intermediate channel counts. C4 = 512 fixed.
    dw_kernel : int
        Depthwise conv kernel size (3, 5, or 7).
    expansion : int
        ConvNeXt inverted bottleneck expansion factor.
    use_side_branch : bool
        Enable side-branch channel attention on Block2.
    use_crelu_block3 : bool
        Use CReLU in Block3 instead of standard GELU inverted bottleneck.
    """

    C4 = 512  # fixed — ensures features = 512×7×7 = 25088

    def __init__(
        self,
        channels: tuple[int, int, int] = (64, 128, 256),
        dw_kernel: int = 7,
        expansion: int = 2,
        use_side_branch: bool = True,
        use_crelu_block3: bool = True,
    ) -> None:
        super().__init__()
        C1, C2, C3 = channels
        C4 = self.C4
        self.use_side_branch  = use_side_branch
        self.use_crelu_block3 = use_crelu_block3

        # Stem: 3→C1, stride 2 + MaxPool → 56×56
        n_groups = max(1, C1 // 8)
        self.stem = nn.Sequential(
            nn.Conv2d(3, C1, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(n_groups, C1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
        )

        # Block1
        self.block1 = ConvNeXtBlock(C1, expansion=expansion, dw_kernel=dw_kernel)

        # Optional side branch
        if use_side_branch:
            self.side = SideBranch(C_in=C1, C_att=C2, dw_kernel=dw_kernel)

        # DS1: C1→C2, stride 2 → 28×28
        self.ds1 = DSConv(C1, C2, k=3, stride=2, padding=1)

        # Block2
        self.block2 = ConvNeXtBlock(C2, expansion=expansion, dw_kernel=dw_kernel)

        # DS2: C2→C3, stride 2 → 14×14
        self.ds2 = DSConv(C2, C3, k=3, stride=2, padding=1)

        # Block3 — CReLU if C3 >= 4 and flag set
        if use_crelu_block3 and C3 >= 4:
            self.block3 = ConvNeXtBlockCReLU(C3, dw_kernel=dw_kernel)
        else:
            self.block3 = ConvNeXtBlock(C3, expansion=expansion, dw_kernel=dw_kernel)

        # DS3: C3→512, stride 2 → 7×7
        self.ds3 = DSConv(C3, C4, k=3, stride=2, padding=1)

        # Head: MaxPool + FC
        self.head = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(C4, 10),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logits [B, 10], features [B, 25088])."""
        x = self.stem(x)                   # [B, C1, 56, 56]
        x = self.block1(x)                 # [B, C1, 56, 56]

        if self.use_side_branch:
            att = self.side(x)             # [B, C2]

        x = self.ds1(x)                    # [B, C2, 28, 28]
        x = self.block2(x)                 # [B, C2, 28, 28]

        if self.use_side_branch:
            x = x * att.unsqueeze(-1).unsqueeze(-1)

        x = self.ds2(x)                    # [B, C3, 14, 14]
        x = self.block3(x)                 # [B, C3, 14, 14]
        x = self.ds3(x)                    # [B, 512, 7, 7]

        features = x.flatten(1)            # [B, 25088]
        logits   = self.head(x)            # [B, 10]
        return logits, features


# ---------------------------------------------------------------------------
# FLOPs counter (hook-based, counts MACs)
# ---------------------------------------------------------------------------

def count_macs(model: nn.Module, img_size: int = 224) -> int:
    """Count multiply-accumulate ops for a single image forward pass."""
    macs = [0]

    def _hook(module, inp, out):
        if isinstance(module, nn.Conv2d):
            _, C_in, H_in, W_in = inp[0].shape
            _, C_out, H_out, W_out = out.shape
            k_h, k_w = module.kernel_size
            g = module.groups
            macs[0] += int(C_out * H_out * W_out * (C_in // g) * k_h * k_w)
        elif isinstance(module, nn.Linear):
            macs[0] += int(module.in_features * module.out_features)

    hooks = [m.register_forward_hook(_hook)
             for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        dev = next(model.parameters()).device
        model(torch.zeros(1, 3, img_size, img_size, device=dev))
    for h in hooks:
        h.remove()
    return macs[0]
