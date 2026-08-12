"""llm_step001: GPT-2-small FFN layer-importance profile (Branch A, first-level).

WHY (builds on step989, does NOT repeat it):
  step989 KILLED "SGNNET replicates layer-6 FFN via isolated MSE regression"
  (cos_sim 0.19 vs 0.5 thresh; even a dense 394K MLP only hit 0.58). That test
  blindly picked layer 6 and used frozen-teacher MSE — a NARROW single-layer probe.

  Before spending GPU on an end-to-end KD replacement (llm_step002), first answer
  the cheap high-info question step989 never asked: WHICH FFN layers matter least?
  Ablate each block's MLP branch (residual kept) and measure perplexity increase on
  wikitext-2. A layer whose FFN can be zeroed for little ppl cost is a SOFT target
  for replacement. Forward-only, CPU-fine.

Output: results/frontier/llm_step001_ffn_importance__{SLOT}.json
  baseline_ppl, per-layer {ppl, delta_ppl, delta_pct}, ranked softest->hardest.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--model", default="gpt2", help="HF model id (gpt2 = 124M small)")
parser.add_argument("--n_windows", type=int, default=160, help="128-tok windows for ppl")
parser.add_argument("--seq_len", type=int, default=128)
parser.add_argument("--smoke_test", action="store_true")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

SLOT = os.environ.get("SGN_SLOT", "local")
OUT = ROOT / "results" / "frontier" / f"llm_step001_ffn_importance__{SLOT}.json"


def load_model_and_data():
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    from datasets import load_dataset
    tok = GPT2TokenizerFast.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model).to(DEVICE).eval()
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    L = args.seq_len
    n = min(args.n_windows, len(ids) // L)
    windows = torch.stack([ids[i * L:(i + 1) * L] for i in range(n)])  # (n, L)
    return model, windows


@torch.no_grad()
def perplexity(model, windows) -> float:
    """Mean token CE over fixed-length windows -> exp(). Lower = better."""
    tot_loss, tot_tok = 0.0, 0
    for i in range(0, len(windows), 8):
        batch = windows[i:i + 8].to(DEVICE)
        out = model(batch, labels=batch)          # shifted-LM CE, mean over batch
        n_tok = batch.shape[0] * (batch.shape[1] - 1)
        tot_loss += out.loss.item() * n_tok
        tot_tok += n_tok
    return math.exp(tot_loss / tot_tok)


class _ZeroMLP:
    """Context manager: replace block[layer].mlp.forward with a zero-return (ablate FFN)."""
    def __init__(self, model, layer):
        self.mlp = model.transformer.h[layer].mlp
        self.orig = self.mlp.forward

    def __enter__(self):
        self.mlp.forward = lambda hs, *a, **k: torch.zeros_like(hs)
        return self

    def __exit__(self, *exc):
        self.mlp.forward = self.orig


def main():
    if args.smoke_test:
        # shape-only: confirm zero-ablation hook flips one layer's MLP output to 0
        from transformers import GPT2LMHeadModel
        m = GPT2LMHeadModel.from_pretrained(args.model)
        x = torch.randint(0, 100, (1, 16))
        base = m(x).logits
        with _ZeroMLP(m, 0):
            abl = m(x).logits
        changed = not torch.allclose(base, abl)
        print(f"  smoke: ablation changes logits={changed}")
        sys.exit(0 if changed else 1)

    print(f"{'='*66}\nllm_step001 FFN-importance profile  model={args.model}  device={DEVICE}")
    t0 = time.time()
    model, windows = load_model_and_data()
    n_layers = model.config.n_layer
    print(f"  {n_layers} layers  {len(windows)} windows x {args.seq_len} tok  [{time.time()-t0:.0f}s load]")

    base_ppl = perplexity(model, windows)
    print(f"  baseline ppl = {base_ppl:.3f}")

    rows = []
    for L in range(n_layers):
        with _ZeroMLP(model, L):
            ppl = perplexity(model, windows)
        d = ppl - base_ppl
        rows.append({"layer": L, "ppl_ffn_zeroed": round(ppl, 4),
                     "delta_ppl": round(d, 4), "delta_pct": round(100 * d / base_ppl, 2)})
        print(f"  layer {L:2d}: FFN-zeroed ppl={ppl:8.3f}  Δ={d:+8.3f} ({100*d/base_ppl:+6.1f}%)", flush=True)

    ranked = sorted(rows, key=lambda r: r["delta_ppl"])   # softest (smallest Δ) first
    res = {"step": "llm_step001", "model": args.model, "device": str(DEVICE),
           "n_windows": len(windows), "seq_len": args.seq_len,
           "baseline_ppl": round(base_ppl, 4), "per_layer": rows,
           "softest_layers": [r["layer"] for r in ranked[:4]],
           "hardest_layers": [r["layer"] for r in ranked[-4:]],
           "elapsed_s": round(time.time() - t0, 1)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\n  softest FFN layers (cheapest to replace): {res['softest_layers']}")
    print(f"  hardest FFN layers: {res['hardest_layers']}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
