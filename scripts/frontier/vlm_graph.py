"""CUDA-graph-safe wrapper around VLMEval. LIBRARY ONLY -- no argparse, no I/O at import.

Extracted at vlm_step033 rather than inlined because `vlm_step033_cudagraph.py` is already at the
repo's 200-line limit, and because any later arm that captures graphs needs exactly this wrapper.

WHY IT EXISTS: inductor's `mode="reduce-overhead"` path (cudagraph trees) allocates model outputs
inside graph-owned memory, which a subsequent replay overwrites. The constrained-choice protocol in
`VLMEval` deliberately REUSES one prefill KV cache across 10 teacher-forced label passes through the
same text_model, so every one of those passes invalidates the cache the next one reads. PyTorch
raises rather than corrupting silently:

    RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent
    run ... To prevent overwriting, clone the tensor outside of torch.compile() or call
    torch.compiler.cudagraph_mark_step_begin() before each model invocation.

Both remedies are applied here: mark a step boundary before each invocation, and copy the cache out
of graph memory immediately after it.

MEASUREMENT NOTE, load-bearing: the clone sits OUTSIDE the timed prefill window (after the post-
prefill sync), so it does not flatter or penalise the prefill number. It is an artifact of this
EVAL protocol, not of deployment -- a deployed pipeline decodes inside the same captured region and
never hands a cache back to Python. Do not report it as a cudagraph overhead.
"""
from __future__ import annotations

import time

import torch

from scripts.frontier.vlm_eval import VLMEval, sync


def clone_cache(cache):
    """Copy a DynamicCache's tensors out of cudagraph-owned storage, in place. Returns the cache."""
    for lay in getattr(cache, "layers", []):
        if getattr(lay, "keys", None) is not None:
            lay.keys = lay.keys.clone()
            lay.values = lay.values.clone()
    return cache


class GraphEval(VLMEval):
    """VLMEval whose text_model invocations are safe under cudagraph trees.

    Overrides only the two methods that touch the reused KV cache. `encode` needs no change: the
    vision tower's output is consumed by `merge` before any further invocation, and `merge` clones
    the embedding table lookup already.
    """

    def forward(self, emb):
        """Prefill (TIMED, identical window to VLMEval.forward) + constrained choice (untimed)."""
        mask = torch.ones(emb.shape[:2], dtype=torch.long, device=self.dev)
        torch.compiler.cudagraph_mark_step_begin()
        sync(self.dev)
        t0 = time.perf_counter()
        with torch.no_grad():
            o = self.m.model.text_model(inputs_embeds=emb, attention_mask=mask, use_cache=True)
            logits = self.m.lm_head(o.last_hidden_state[:, -1])[0].float()
        sync(self.dev)
        ms = (time.perf_counter() - t0) * 1000
        past = clone_cache(o.past_key_values)
        return logits, self.choose(logits, past, emb.shape[1]), ms

    def choose(self, logits, past, plen):
        """As VLMEval.choose, but each label pass marks a step and copies the cache back out."""
        lp0, out = logits.log_softmax(-1), []
        for ids in self.label_ids:
            torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad():
                o = self.m.model.text_model(
                    inputs_embeds=self.m.get_input_embeddings()(ids[None]), past_key_values=past,
                    attention_mask=torch.ones(1, plen + len(ids), dtype=torch.long,
                                              device=self.dev), use_cache=True)
                lg = self.m.lm_head(o.last_hidden_state)[0].float().log_softmax(-1)
            s = lp0[ids[0]] + sum(lg[i, ids[i + 1]] for i in range(len(ids) - 1))
            out.append(float(s) / len(ids))
            clone_cache(past).crop(plen)
        return int(torch.tensor(out).argmax())
