"""cnnc_step001 models — multi-branch full-context CNNs (Imagenette, 224px).

Configs (param-matched ~0.5M +/-10%):
  Ref           single-branch plain downsampling stack
  A_global      Ref-like local branch + global-context branch
                (16x avgpool -> convs -> channel-wise FC, Pathak et al. 2016)
  B_multibranch 3 branches: full-res local / 4x-down mid / 16x-down global
  C_crelu       B + CReLU in first 2 conv layers of each branch
                (Shang et al. 2016: halve filters, concat(relu(x), relu(-x)))
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(max(1, c // 8), c)


class ConvBlock(nn.Module):
    """3x3 conv + GN + GELU. crelu=True: conv emits c_out//2 filters, then
    concat(relu(h), relu(-h)) restores c_out (Shang et al. 2016)."""

    def __init__(self, c_in: int, c_out: int, stride: int = 1, crelu: bool = False):
        super().__init__()
        self.crelu = crelu
        c_conv = c_out // 2 if crelu else c_out
        self.conv = nn.Conv2d(c_in, c_conv, 3, stride=stride, padding=1, bias=False)
        self.gn = _gn(c_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.gn(self.conv(x))
        if self.crelu:
            return torch.cat([F.relu(h), F.relu(-h)], dim=1)
        return F.gelu(h)


class ChannelWiseFC(nn.Module):
    """Per-channel spatial FC (Context Encoders, Pathak et al. 2016).
    Each channel: full HWxHW dense map over spatial locations.
    No parameters connecting different feature maps. Params = C*HW*HW."""

    def __init__(self, c: int, hw: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(c, hw, hw) * hw ** -0.5)
        self.b = nn.Parameter(torch.zeros(c, hw))
        self.hw = hw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        v = x.flatten(2)                                    # [B, C, HW]
        v = torch.einsum("bci,cij->bcj", v, self.w) + self.b
        return v.view(B, C, H, W)


class Branch(nn.Module):
    """Plain conv stack. spec = [(c_out, stride), ...].
    pre_pool: avg-pool input by this factor first (4x mid / 16x global).
    stem_pool: MaxPool2 after first conv (full-res local branches).
    crelu_first_n: apply CReLU to the first n ConvBlocks."""

    def __init__(self, spec, in_ch: int = 3, pre_pool: int = 1,
                 stem_pool: bool = False, crelu_first_n: int = 0):
        super().__init__()
        self.pre_pool = pre_pool
        layers: list[nn.Module] = []
        c = in_ch
        for i, (c_out, stride) in enumerate(spec):
            layers.append(ConvBlock(c, c_out, stride, crelu=i < crelu_first_n))
            if stem_pool and i == 0:
                layers.append(nn.MaxPool2d(2))
            c = c_out
        self.net = nn.Sequential(*layers)
        self.out_ch = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_pool > 1:
            x = F.avg_pool2d(x, self.pre_pool)
        return self.net(x)


class GlobalContextBranch(nn.Module):
    """16x avgpool (224->14) -> conv -> conv s2 (->7x7) -> channel-wise FC
    -> 1x1 conv to propagate info across channels (Pathak et al. 2016)."""

    def __init__(self, c1: int = 32, c2: int = 40, crelu: bool = False):
        super().__init__()
        self.b1 = ConvBlock(3, c1, 1, crelu=crelu)
        self.b2 = ConvBlock(c1, c2, 2)
        self.cwfc = ChannelWiseFC(c2, 49)
        self.gn = _gn(c2)
        self.mix = nn.Conv2d(c2, c2, 1, bias=False)
        self.out_ch = c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.avg_pool2d(x, 16)            # 224 -> 14
        x = self.b2(self.b1(x))            # [B, c2, 7, 7]
        x = F.gelu(self.gn(self.cwfc(x)))
        return self.mix(x)


class MultiBranchNet(nn.Module):
    """GAP each branch -> concat -> linear head. 1..3 branches."""

    def __init__(self, branches: list[nn.Module], n_classes: int = 10):
        super().__init__()
        self.branches = nn.ModuleList(branches)
        self.head = nn.Linear(sum(b.out_ch for b in branches), n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [F.adaptive_avg_pool2d(b(x), 1).flatten(1) for b in self.branches]
        return self.head(torch.cat(feats, dim=1))


# ---------------------------------------------------------------------------
# Config builders — widths tuned for ~0.5M params each (+/-10%)
# ---------------------------------------------------------------------------

_LOCAL_B = [(16, 2), (24, 1), (24, 2), (32, 1), (32, 2), (56, 1)]
_MID_B   = [(32, 1), (64, 2), (64, 1), (96, 2), (96, 1), (144, 2)]


def build_model(name: str, n_classes: int = 10) -> nn.Module:
    if name == "Ref":
        ref = Branch([(32, 2), (64, 1), (64, 2), (128, 1), (128, 2), (176, 1)],
                     stem_pool=True)
        return MultiBranchNet([ref], n_classes)
    if name == "A_global":
        local = Branch([(32, 2), (64, 1), (64, 2), (128, 1), (128, 2), (112, 1)],
                       stem_pool=True)
        return MultiBranchNet([local, GlobalContextBranch(32, 36)], n_classes)
    if name == "B_multibranch":
        local = Branch(_LOCAL_B, stem_pool=True)
        mid = Branch(_MID_B, pre_pool=4)
        return MultiBranchNet([local, mid, GlobalContextBranch(32, 40)], n_classes)
    if name == "C_crelu":
        local = Branch(_LOCAL_B, stem_pool=True, crelu_first_n=2)
        mid = Branch(_MID_B, pre_pool=4, crelu_first_n=2)
        glob = GlobalContextBranch(32, 40, crelu=True)
        return MultiBranchNet([local, mid, glob], n_classes)
    raise ValueError(f"unknown config: {name}")


CONFIG_DESCS = {
    "Ref":           "single-branch plain downsampling stack",
    "A_global":      "Ref + 16x-down global-context branch (channel-wise FC)",
    "B_multibranch": "3 branches: full-res / 4x-down / 16x-down global",
    "C_crelu":       "B + CReLU in first 2 convs of every branch",
}


# ---------------------------------------------------------------------------
# MAC counter (hook-based, single 224x224 image)
# ---------------------------------------------------------------------------

def count_macs(model: nn.Module, img_size: int = 224) -> int:
    macs = [0]

    def _hook(m, inp, out):
        if isinstance(m, nn.Conv2d):
            _, C_out, H, W = out.shape
            k_h, k_w = m.kernel_size
            macs[0] += int(C_out * H * W * (m.in_channels // m.groups) * k_h * k_w)
        elif isinstance(m, nn.Linear):
            macs[0] += int(m.in_features * m.out_features)
        elif isinstance(m, ChannelWiseFC):
            macs[0] += int(m.w.shape[0] * m.hw * m.hw)

    hooks = [m.register_forward_hook(_hook) for m in model.modules()
             if isinstance(m, (nn.Conv2d, nn.Linear, ChannelWiseFC))]
    with torch.no_grad():
        dev = next(model.parameters()).device
        model(torch.zeros(1, 3, img_size, img_size, device=dev))
    for h in hooks:
        h.remove()
    return macs[0]
