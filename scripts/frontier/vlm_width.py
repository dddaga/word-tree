"""MLP-width slicing helpers for the vlm_step02X frontier line.

LIBRARY ONLY -- no argparse, no I/O at import, so step scripts can import it. Extracted at
vlm_step023 because that script crossed the repo's 200-line limit; the slicing logic is the
reusable half and the experiment's gates/verdicts are the half that must stay in the step script.
"""
from __future__ import annotations
import copy

import torch
import torch.nn as nn


def slice_mlp(layer, m):
    """Return a copy of `layer` with its MLP intermediate width cut to m channels.

    Channels are ranked by ||fc1 row|| * ||fc2 column||: a unit is important only if it both fires
    (large input weights) and is read (large output weights). Keeping the top-m is the width
    analogue of depth truncation -- the teacher's own most-used units, warm-started from its
    weights -- rather than a random init, which would confound 'width is too small' with 'the
    student started from nothing'. No adapters are needed: fc1 loses rows, fc2 loses the matching
    columns, and the residual stream stays at hidden_size, so the tower's input and output
    interfaces (patch embeddings in, post_layernorm + connector out) are untouched.
    """
    l = copy.deepcopy(layer)
    fc1, fc2 = l.mlp.fc1, l.mlp.fc2
    score = fc1.weight.norm(dim=1) * fc2.weight.norm(dim=0)
    keep = torch.topk(score, m).indices.sort().values
    n1 = nn.Linear(fc1.in_features, m, bias=fc1.bias is not None)
    n2 = nn.Linear(m, fc2.out_features, bias=fc2.bias is not None)
    with torch.no_grad():
        n1.weight.copy_(fc1.weight[keep]); n2.weight.copy_(fc2.weight[:, keep])
        if fc1.bias is not None: n1.bias.copy_(fc1.bias[keep])
        if fc2.bias is not None: n2.bias.copy_(fc2.bias)
    l.mlp.fc1, l.mlp.fc2 = n1, n2
    return l


def build_width_student(ev, depth, m, device):
    """A trainable ModuleList of the first `depth` teacher layers, each MLP-sliced to width m."""
    return nn.ModuleList([slice_mlp(l, m) for l in ev.full[:depth]]).to(device)


def n_params(mod):
    return sum(p.numel() for p in mod.parameters())
