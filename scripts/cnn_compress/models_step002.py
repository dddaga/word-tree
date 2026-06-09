"""cnnc_step002 models — iso-MAC variants of the step001 winners.

cnnc_step001 (T0): B_multibranch (110.9M MACs) and C_crelu (94.1M) lose only
~2pp to Ref (224.0M MACs) at iso-params. Question here: at MATCHED MACs
(~224M +/-10%), do they beat Ref? Params are free to exceed Ref's 483K —
MACs are the controlled variable; both dims reported.

Configs (MACs targeted at 224M +/-10%; Ref itself not rerun — step001
number 0.7557 reused, identical seed/protocol):
  B_isomac   B_multibranch widths scaled ~1.44x (MACs ~ width^2)
  C_isomac   B_isomac widths + CReLU in first 2 convs of every branch,
             CReLU MAC savings reinvested as +1 conv on local & mid branches
             (depth, per PLAN: spend budget on depth not width)
  Ref_deep   control: Ref MAC budget spent on depth — 9 convs, narrower

Reuses Branch / GlobalContextBranch / MultiBranchNet / count_macs from
models_step001 (imported, not copied).
"""
from __future__ import annotations

import torch.nn as nn

from scripts.cnn_compress.models_step001 import (   # noqa: F401 (count_macs re-export)
    Branch, GlobalContextBranch, MultiBranchNet, count_macs)

# --- B_multibranch widths x ~1.44 (step001: local 16..56, mid 32..144) -----
_LOCAL_ISO = [(24, 2), (32, 1), (32, 2), (48, 1), (48, 2), (80, 1)]
_MID_ISO   = [(48, 1), (96, 2), (96, 1), (144, 2), (144, 1), (208, 2)]

# --- C: same widths + CReLU(first 2) + extra depth at final resolution -----
_LOCAL_ISO_C = _LOCAL_ISO + [(80, 1)]    # extra 14x14 conv
_MID_ISO_C   = _MID_ISO + [(208, 1)]     # extra 7x7 conv

# --- Ref_deep: Ref's 224M MAC budget spent on depth (9 convs vs 6) ---------
_REF_DEEP = [(32, 2), (48, 1), (48, 2), (80, 1), (80, 1), (80, 2),
             (112, 1), (112, 1), (144, 1)]


def build_model(name: str, n_classes: int = 10) -> nn.Module:
    if name == "B_isomac":
        local = Branch(_LOCAL_ISO, stem_pool=True)
        mid = Branch(_MID_ISO, pre_pool=4)
        return MultiBranchNet([local, mid, GlobalContextBranch(48, 56)], n_classes)
    if name == "C_isomac":
        local = Branch(_LOCAL_ISO_C, stem_pool=True, crelu_first_n=2)
        mid = Branch(_MID_ISO_C, pre_pool=4, crelu_first_n=2)
        glob = GlobalContextBranch(48, 56, crelu=True)
        return MultiBranchNet([local, mid, glob], n_classes)
    if name == "Ref_deep":
        return MultiBranchNet([Branch(_REF_DEEP, stem_pool=True)], n_classes)
    raise ValueError(f"unknown config: {name}")


CONFIG_DESCS = {
    "B_isomac": "B_multibranch widths x1.44 -> ~224M MACs (3 branches)",
    "C_isomac": "B_isomac + CReLU first 2 convs + 1 extra conv/branch",
    "Ref_deep": "Ref MAC budget on depth: 9-conv single branch",
}
