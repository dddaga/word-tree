"""Resolution-variable eval harness for the vlm_step02X line.

LIBRARY ONLY -- no argparse, no I/O at import, so step scripts can import it. Extracted at
vlm_step021 because `vlm_step020_resolution_ladder.py` parses argv at module scope (as every step
script does), so importing ResEval from it would run step020's parser against step021's flags.
step020 keeps its inline copy deliberately: it is a DONE experiment and its script must stay
reproducible exactly as run -- the same rule that left vlm_step002/003's eval copies in place.
"""
from __future__ import annotations
import time

import torch
import torch.nn.functional as F

from scripts.frontier.vlm_eval import VLMEval, sync


class ResEval(VLMEval):
    """VLMEval with a settable pixel grid. Set `.res`; 512 is the stock grid.

    Two overrides, both minimal. `encode` downsamples the processor's normalised pixels to
    self.res before the tower runs -- that is the whole intervention, and it is done model-side
    because the processor's own `size={'longest_edge': N}` knob is INERT (verified: every N in
    512/384/256/192/128 returns a 512x512 tensor). Downsampling the processor's output keeps
    mean/std preprocessing identical across resolutions, so the pixel grid is the only thing that
    changes. SigLIP interpolates its position embeddings, so every grid runs.

    `merge` splices the resulting T' features into the FIRST T' image slots and drops the remaining
    (64 - T') slots, reusing the drop-mask shape of the parent's `prune`. The prompt is built by
    the processor and always carries 64 image placeholders, so without this the parent's
    `assert len(pos) == len(feats)` fires at every resolution below 512.
    """

    res = 512

    def encode(self, b, grad=False):
        sync(self.dev); t0 = time.perf_counter()
        px = b["pixel_values"]
        if self.res != px.shape[-1]:
            sh = px.shape
            px = F.interpolate(px.reshape(-1, *sh[-3:]), size=(self.res, self.res),
                               mode="bilinear", align_corners=False, antialias=True)
            px = px.reshape(*sh[:-3], *px.shape[-3:])
        with torch.set_grad_enabled(grad):
            lh = self.m.model.get_image_features(pixel_values=px).last_hidden_state
            feats = self.m.model.connector(lh)
        sync(self.dev)
        return feats.reshape(-1, feats.shape[-1]), (time.perf_counter() - t0) * 1000

    def merge(self, b, feats):
        ids = b["input_ids"]
        emb = self.m.get_input_embeddings()(ids).clone()
        pos = (ids[0] == self.img_id).nonzero(as_tuple=True)[0]
        t = len(feats)
        assert t <= len(pos), f"{t} feats > {len(pos)} slots"
        emb[0, pos[:t]] = feats.to(emb.dtype)
        if t == len(pos): return emb, pos
        keep = torch.ones(emb.shape[1], dtype=torch.bool, device=self.dev)
        keep[pos[t:]] = False
        return emb[:, keep], pos[:t]
