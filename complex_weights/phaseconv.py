# PhaseConv2d — multiply-as-phase-addition conv (Phase 0 reference impl).
# See README.md for the science/systems split.
#
# Idea (Euler): unit-magnitude complex weight w = exp(i*theta), input lifted to
# phasor z = exp(i*phi). Then w*z = exp(i*(theta+phi)) -> the "multiply" is a
# pure PHASE ADDITION. Accumulate over the receptive field (Cartesian sum, the
# unavoidable accumulate), then project to real:  y = Re( sum exp(i(theta+phi)) )
#                                                   = sum cos(theta + phi).
# So cosine is not an ad-hoc activation — it is Re() of the phasor sum.
#
# HONESTY NOTE (CLAUDE.md tenet: measure goal, not proxy):
#   This reference computes Re(S) EXACTLY, but via the identity
#     Re(S) = conv(cos phi, cos theta) - conv(sin phi, sin theta)
#   which uses real MULTIPLIES under the hood. That is fine: this file proves
#   the FUNCTION (accuracy). The energy win (add instead of mul, LUT phases)
#   is a separate kernel claim measured in Phase 2 — never claimed from here.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseConv2d(nn.Module):
    """Phasor convolution. theta (weight phases) is the only conv parameter —
    same count as a real conv's weights, so comparisons are iso-parameter.

    variant:
      'real'  -> output = sum cos(theta+phi)         (real activation, re-encoded next layer)
      'phase' -> output = angle(S), magnitude |S| kept as gate (phase-native, ablation)
    """

    def __init__(self, in_ch, out_ch, k=3, stride=1, padding=1,
                 variant="real", phase_bias=True, n_roots=0):
        super().__init__()
        self.in_ch, self.out_ch, self.k = in_ch, out_ch, k
        self.stride, self.padding, self.variant = stride, padding, variant
        self.n_roots = n_roots  # 0 = continuous phase; R>0 = quantize theta to R roots of unity
        # weight phases, init ~ small so exp(i*theta) starts near coherent
        self.theta = nn.Parameter(torch.empty(out_ch, in_ch, k, k).uniform_(-math.pi, math.pi))
        self.bias = nn.Parameter(torch.zeros(out_ch)) if phase_bias else None

    def _theta_q(self):
        # Quantize weight phase to nearest of R roots of unity (2*pi*k/R), straight-through
        # estimator so gradients flow to the continuous theta. R roots -> log2(R) bits/weight.
        if not self.n_roots:
            return self.theta
        step = 2 * math.pi / self.n_roots
        q = torch.round(self.theta / step) * step
        return self.theta + (q - self.theta).detach()

    def _phasor_sum(self, phi):
        # phi: [B, in_ch, H, W] real angles. Returns real S_re, S_im [B,out,H',W'].
        theta = self._theta_q()
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        cphi, sphi = torch.cos(phi), torch.sin(phi)
        # S = sum exp(i(theta+phi)) = (cos t*cos phi - sin t*sin phi) + i(sin t*cos phi + cos t*sin phi)
        s_re = F.conv2d(cphi, cos_t, None, self.stride, self.padding) \
             - F.conv2d(sphi, sin_t, None, self.stride, self.padding)
        s_im = F.conv2d(cphi, sin_t, None, self.stride, self.padding) \
             + F.conv2d(sphi, cos_t, None, self.stride, self.padding)
        return s_re, s_im

    def forward(self, phi):
        s_re, s_im = self._phasor_sum(phi)
        if self.variant == "real":
            y = s_re
            if self.bias is not None:
                y = y + self.bias.view(1, -1, 1, 1)
            return y
        elif self.variant == "phase":
            ang = torch.atan2(s_im, s_re)
            if self.bias is not None:
                ang = ang + self.bias.view(1, -1, 1, 1)
            return ang
        raise ValueError(self.variant)


def encode_phase(x, mode="wrap"):
    """Lift real activations to angles. 'wrap' embraces overflow-rotation:
    values beyond [-pi,pi] wrap (mod 2pi) — the free-rotation-on-overflow idea,
    emulated in float here (integer-overflow version is a deploy kernel)."""
    if mode == "wrap":
        return torch.remainder(x + math.pi, 2 * math.pi) - math.pi
    if mode == "tanh":
        return math.pi * torch.tanh(x)
    if mode == "identity":
        return x
    raise ValueError(mode)


def real_conv_flops(in_ch, out_ch, k, H, W):
    """MAC count for one standard conv output map (for the Pareto table)."""
    return out_ch * in_ch * k * k * H * W
