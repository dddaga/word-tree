"""MultiScaleCNN: parallel-dilation DW-sep CNN for VGG16 distillation.

Each stage runs N parallel depthwise branches at different dilation rates
(multi-scale receptive fields), merged by 1×1 conv. CReLU optional per block.

Spatial flow (input 3×224×224):
  Stem  : 3→C1, stride2 + MaxPool2 → [B, C1, 56, 56]
  Block1: MultiScaleBlock(C1) → [B, C1, 56, 56]
  DS1   : C1→C2, stride2     → [B, C2, 28, 28]
  Block2: MultiScaleBlock(C2) → [B, C2, 28, 28]
  DS2   : C2→C3, stride2     → [B, C3, 14, 14]
  Block3: MultiScaleBlock(C3) → [B, C3, 14, 14]
  DS3   : C3→512, stride2    → [B, 512, 7, 7]
  Out   : (logits [B,10], features [B,25088])
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def crelu(x: torch.Tensor) -> torch.Tensor:
    """Concat pos+neg half-wave rectifications — doubles channel dim."""
    return torch.cat([F.relu(x), F.relu(-x)], dim=1)


class ChanLN(nn.Module):
    """Per-channel LayerNorm for [B, C, H, W] (ConvNeXt convention)."""
    def __init__(self, C: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class DSDown(nn.Module):
    """DW-sep stride-2 downsampler: DW(3×3) → ChanLN → PW."""
    def __init__(self, C_in: int, C_out: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(C_in, C_in, 3, stride=2, padding=1, groups=C_in, bias=False)
        self.ln = ChanLN(C_in)
        self.pw = nn.Conv2d(C_in, C_out, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.ln(self.dw(x)))


class DilBranch(nn.Module):
    """Dilated DW-sep residual branch with GELU or CReLU inverted bottleneck.

    GELU path: DW → LN → PW(C→C*exp) → GELU → PW(→C)
    CReLU path: DW → LN → PW(C→C//2) → CReLU(→C) → PW(C→C)
    Both: residual add (no channels change).
    """
    def __init__(self, C: int, dil: int = 1, expansion: int = 2,
                 use_crelu: bool = False, dw_k: int = 7) -> None:
        super().__init__()
        pad = dil * (dw_k // 2)
        self.use_crelu = use_crelu
        self.dw = nn.Conv2d(C, C, dw_k, padding=pad, dilation=dil, groups=C, bias=False)
        self.ln = ChanLN(C)
        if use_crelu:
            C_half = max(1, C // 2)
            self.pw1 = nn.Conv2d(C, C_half, 1, bias=False)
            self.pw2 = nn.Conv2d(C, C, 1, bias=False)       # C_half*2 → C after crelu
        else:
            C_mid = C * expansion
            self.pw1 = nn.Conv2d(C, C_mid, 1, bias=False)
            self.pw2 = nn.Conv2d(C_mid, C, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(self.dw(x))
        if self.use_crelu:
            return x + self.pw2(crelu(self.pw1(h)))
        return x + self.pw2(F.gelu(self.pw1(h)))


class MultiScaleBlock(nn.Module):
    """N parallel DilBranch at different dilation rates, concatenated then merged."""
    def __init__(self, C: int, dil_rates: tuple, expansion: int = 2,
                 use_crelu: bool = False) -> None:
        super().__init__()
        self.branches = nn.ModuleList([
            DilBranch(C, d, expansion, use_crelu) for d in dil_rates
        ])
        n = len(dil_rates)
        self.merge = nn.Conv2d(C * n, C, 1, bias=False) if n > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [b(x) for b in self.branches]
        if len(outs) == 1:
            return outs[0]
        return self.merge(torch.cat(outs, dim=1))


class MultiScaleCNN(nn.Module):
    """Configurable multi-scale CNN distillation student for VGG16.

    Parameters
    ----------
    channels : (C1, C2, C3) — intermediate channel counts
    dil_rates : tuple of ints — dilation rates for parallel branches (e.g. (1, 2, 4))
    expansion : ConvNeXt inverted bottleneck expansion (GELU path only)
    crelu_mask : (b1, b2, b3) — which blocks use CReLU instead of GELU
    """
    C4 = 512  # fixed: features = 512×7×7 = 25088

    def __init__(
        self,
        channels: tuple = (64, 128, 256),
        dil_rates: tuple = (1,),
        expansion: int = 2,
        crelu_mask: tuple = (False, False, True),
    ) -> None:
        super().__init__()
        C1, C2, C3 = channels
        C4 = self.C4
        ng = max(1, C1 // 8)

        self.stem = nn.Sequential(
            nn.Conv2d(3, C1, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(ng, C1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
        )
        self.b1  = MultiScaleBlock(C1, dil_rates, expansion, bool(crelu_mask[0]))
        self.ds1 = DSDown(C1, C2)
        self.b2  = MultiScaleBlock(C2, dil_rates, expansion, bool(crelu_mask[1]))
        self.ds2 = DSDown(C2, C3)
        self.b3  = MultiScaleBlock(C3, dil_rates, expansion, bool(crelu_mask[2]))
        self.ds3 = DSDown(C3, C4)
        self.head = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(C4, 10),
        )

    def forward(self, x: torch.Tensor) -> tuple:
        x = self.stem(x)
        x = self.b1(x)
        x = self.ds1(x)
        x = self.b2(x)
        x = self.ds2(x)
        x = self.b3(x)
        x = self.ds3(x)
        features = x.flatten(1)       # [B, 25088]
        logits   = self.head(x)       # [B, 10]
        return logits, features


def count_macs(model: nn.Module, img_size: int = 224) -> int:
    """Count multiply-accumulate ops (MACs) for a single image forward pass."""
    macs = [0]

    def _hook(module, inp, out):
        if isinstance(module, nn.Conv2d):
            _, C_in, _, _ = inp[0].shape
            _, C_out, H_out, W_out = out.shape
            kh, kw = module.kernel_size
            macs[0] += int(C_out * H_out * W_out * (C_in // module.groups) * kh * kw)
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
