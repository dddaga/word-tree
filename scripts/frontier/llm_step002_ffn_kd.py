"""llm_step002: end-to-end logit-KD FFN replacement in GPT-2-small (Branch A, load-bearing).

THE TEST step989 NEVER RAN.
  step989 killed "SGNNET replicates GPT-2 FFN" using ISOLATED layer-6 MSE regression
  (frozen teacher, x_ffn->y_ffn, no downstream loss) — even a dense 394K MLP only hit
  cos_sim 0.58. But exact reconstruction is the WRONG target: the residual stream is
  over-specified and the network can COMPENSATE if the rest stays live.

  Here: replace ONE block's FFN (layer picked by llm_step001 importance profile) with a
  compact module, FREEZE all other weights, and train ONLY the replacement to match the
  full teacher's LOGITS (KL divergence) on real text. This is behavioural distillation,
  not activation reconstruction. Question: can end-to-end KD recover perplexity that
  isolated MSE could not? If a ~1-5%-param compact FFN recovers near-baseline ppl -> the
  direction reopens. If even a full-capacity reinit can't -> the layer is genuinely
  load-bearing (a STRONGER kill than step989).

Configs (all trained via KD, rest of net frozen):
  zero_floor  : FFN removed, no train (perplexity floor, ref)
  lr64        : LowRank 768->64->768   (~0.10M, ~2% of dense FFN)
  lr256       : LowRank 768->256->768  (~0.39M, ~8%)
  sparse256k32: top-k sparse hidden (r=256,k=32) — SGNNET sparse-routing analog
  dense_reinit: full 768->3072->768 reinit (ceiling of the KD method)

Output: results/frontier/llm_step002_ffn_kd_L{layer}__{SLOT}.json
"""
from __future__ import annotations
import argparse, copy, json, math, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--model", default="gpt2")
parser.add_argument("--layer", type=int, default=-1, help="FFN layer to replace; -1 = auto from step001 json")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--n_train", type=int, default=800, help="128-tok train windows")
parser.add_argument("--n_eval", type=int, default=160)
parser.add_argument("--kd_temp", type=float, default=2.0)
parser.add_argument("--lm_weight", type=float, default=0.1, help="aux LM-CE weight alongside KD")
parser.add_argument("--seq_len", type=int, default=128)
parser.add_argument("--smoke_test", action="store_true")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

SLOT = os.environ.get("SGN_SLOT", "local")
DENSE_FFN_PARAMS = 768 * 3072 * 2 + 3072 + 768        # GPT-2 small block FFN = 4,722,432


class LowRankFFN(nn.Module):
    def __init__(self, d, r):
        super().__init__()
        self.up = nn.Linear(d, r); self.down = nn.Linear(r, d)

    def forward(self, x, *a, **k):
        return self.down(F.gelu(self.up(x)))


class SparseTopKFFN(nn.Module):
    """SGNNET analog: only top-k of r hidden units active per token (hard sparse routing)."""
    def __init__(self, d, r, k):
        super().__init__()
        self.up = nn.Linear(d, r); self.down = nn.Linear(r, d); self.k = k

    def forward(self, x, *a, **k):
        h = F.gelu(self.up(x))
        kth = h.topk(self.k, dim=-1).values[..., -1:]     # per-token k-th largest
        return self.down(h * (h >= kth))                   # zero all but top-k


class DenseFFN(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.c_fc = nn.Linear(d, h); self.c_proj = nn.Linear(h, d)

    def forward(self, x, *a, **k):
        return self.c_proj(F.gelu(self.c_fc(x)))


class ZeroFFN(nn.Module):
    """FFN branch removed (returns zeros) — perplexity floor, no params."""
    def forward(self, x, *a, **k):
        return torch.zeros_like(x)


def build_module(name):
    if name == "lr64":          return LowRankFFN(768, 64)
    if name == "lr256":         return LowRankFFN(768, 256)
    if name == "sparse256k32":  return SparseTopKFFN(768, 256, 32)
    if name == "dense_reinit":  return DenseFFN(768, 3072)
    raise ValueError(name)


def load():
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    from datasets import load_dataset
    tok = GPT2TokenizerFast.from_pretrained(args.model)
    teacher = GPT2LMHeadModel.from_pretrained(args.model).to(DEVICE).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    L = args.seq_len

    def windows(split, n):
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        ids = tok("\n\n".join(t for t in ds["text"] if t.strip()), return_tensors="pt").input_ids[0]
        m = min(n, len(ids) // L)
        return torch.stack([ids[i * L:(i + 1) * L] for i in range(m)])

    return teacher, windows("train", args.n_train), windows("test", args.n_eval)


@torch.no_grad()
def perplexity(model, w):
    tl, tt = 0.0, 0
    for i in range(0, len(w), 8):
        b = w[i:i + 8].to(DEVICE)
        n = b.shape[0] * (b.shape[1] - 1)
        tl += model(b, labels=b).loss.item() * n; tt += n
    return math.exp(tl / tt)


def train_config(name, teacher, tr, ev, target_layer, base_ppl):
    student = copy.deepcopy(teacher)
    for p in student.parameters():
        p.requires_grad_(False)
    mod = build_module(name).to(DEVICE)
    student.transformer.h[target_layer].mlp = mod       # swap FFN; only `mod` trainable
    n_p = sum(p.numel() for p in mod.parameters())
    opt = torch.optim.AdamW(mod.parameters(), lr=1e-3, weight_decay=0.0)
    T = args.kd_temp
    student.train()
    t0 = time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(len(tr))
        for i in range(0, len(tr), 8):
            b = tr[perm[i:i + 8]].to(DEVICE)
            with torch.no_grad():
                t_logits = teacher(b).logits
            s_logits = student(b).logits
            kd = F.kl_div(F.log_softmax(s_logits / T, -1), F.softmax(t_logits / T, -1),
                          reduction="batchmean") * T * T
            lm = F.cross_entropy(s_logits[:, :-1].reshape(-1, s_logits.size(-1)),
                                 b[:, 1:].reshape(-1))
            loss = kd + args.lm_weight * lm
            opt.zero_grad(); loss.backward(); opt.step()
        ppl = perplexity(student, ev)
        print(f"    {name:<12} ep{ep+1}/{args.epochs} ppl={ppl:.3f} kd={kd.item():.4f} [{time.time()-t0:.0f}s]", flush=True)
    ppl = perplexity(student, ev)
    return {"config": name, "layer": target_layer, "n_params": n_p,
            "pct_dense_ffn": round(100 * n_p / DENSE_FFN_PARAMS, 2),
            "final_ppl": round(ppl, 4), "recovery_vs_baseline": round(base_ppl / ppl, 4),
            "elapsed_s": round(time.time() - t0, 1)}


def main():
    if args.smoke_test:
        for nm in ["lr64", "lr256", "sparse256k32", "dense_reinit"]:
            m = build_module(nm); out = m(torch.randn(2, 8, 768))
            print(f"  {nm:<12} params={sum(p.numel() for p in m.parameters()):,} out={tuple(out.shape)}")
        sys.exit(0)

    target = args.layer
    if target < 0:                                       # auto: softest FFN from step001
        for slot in ("local", "mini_cpu", "5060ti_cuda", "mini_mps"):
            f = ROOT / "results" / "frontier" / f"llm_step001_ffn_importance__{slot}.json"
            if f.exists():
                target = json.loads(f.read_text())["softest_layers"][0]; break
        if target < 0:
            target = 6                                   # fallback (step989's layer)

    print(f"{'='*66}\nllm_step002 FFN KD-replacement  layer={target}  device={DEVICE}")
    teacher, tr, ev = load()
    base_ppl = perplexity(teacher, ev)
    with_zero = copy.deepcopy(teacher)
    with_zero.transformer.h[target].mlp = ZeroFFN().to(DEVICE)
    zero_ppl = perplexity(with_zero, ev)
    print(f"  teacher ppl={base_ppl:.3f}  zero-FFN floor ppl={zero_ppl:.3f}  (layer {target})")

    res = {"step": "llm_step002", "model": args.model, "layer": target, "device": str(DEVICE),
           "teacher_ppl": round(base_ppl, 4), "zero_floor_ppl": round(zero_ppl, 4),
           "dense_ffn_params": DENSE_FFN_PARAMS, "configs": {}}
    OUT = ROOT / "results" / "frontier" / f"llm_step002_ffn_kd_L{target}__{SLOT}.json"
    for nm in ["lr64", "lr256", "sparse256k32", "dense_reinit"]:
        res["configs"][nm] = train_config(nm, teacher, tr, ev, target, base_ppl)
        OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(res, indent=2))
    print(f"\n  teacher={base_ppl:.2f}  zero-floor={zero_ppl:.2f}")
    for nm, r in res["configs"].items():
        print(f"  {nm:<12} ppl={r['final_ppl']:8.3f}  {r['n_params']:>9,}p ({r['pct_dense_ffn']:5.1f}% FFN)")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
