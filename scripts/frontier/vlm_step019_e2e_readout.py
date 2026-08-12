"""vlm_step019: what does the int8 readout actually save IN THE HARNESS, per image?

Part 8 SS30 found every kernel number in part 7 was measured at T=64/128/576 while `vlm_eval` calls
`lm_head` at T=1 (forward) and T=1-3 (choose, ten times per image, OUTSIDE the timed region), and
that the microbenchmark win at the deployed shape is 3.05x. It then said quoting that end-to-end
would be the composition error SS24.1 warned of, one level up. So: measure it rather than assert it.

The kernel is imported verbatim from step017 -- ONE definition, no re-typed variant to drift. What
is new is that it runs inside the real model, on real activations, with the real call pattern, and
that the `choose()` calls the harness never timed are counted. FIVE ARMS, each one step apart:
  fp32       stock `lm_head`. Harness baseline and VALIDITY GATE: must reproduce step006's 0.711.
  fp16       same math, half weights -- the honest DEPLOYMENT baseline. SS30 note 3: an fp32
             comparison flatters int8, so this arm keeps the headline ratio off fp32.
  w8a8_fake  per-row weight scales + per-token act scales, quantized then dequantized, fp32 matmul.
             Same scheme as `vlm_quant.ARMS['w8a8_row']`: the ACCURACY, none of the kernel.
  w8a8_tri   same scheme, real Triton int8 kernel -> int32 + a SEPARATE dequant pass.
  w8a8_trif  same kernel, dequant FUSED in the epilogue. SS30 called fusion unnecessary at 1.01x
             in isolation; the separate pass also costs a launch + a 49280-wide output round-trip.

So tri-vs-fake is a one-variable measurement of the kernel, trif-vs-tri a one-variable measurement of
fusion, and tri-vs-fp16 the one-variable measurement of the deployment swap itself.

PRE-REGISTERED GATES (written before the run):
  1. KERNEL FIDELITY: agree(w8a8_tri, w8a8_fake) >= 0.99. Below that the kernel is not computing the
     scheme step016 measured accuracy under, and no latency number from it may be quoted.
  2. VALIDITY: d12 fp32 top-1 within +-2.0pp of 0.711 (vlm_step006), the same tolerance step010 used.
  3. Latency is per-image and SPLIT into forward (1 call) and choose (10 calls) -- `prefill_ms`
     alone cannot show the readout cost, because 10 of the 11 calls happen outside it.
Depth is 12 (stock tower): this measures the READOUT and vision depth cannot touch `lm_head`
latency, so nothing here depends on a distillation checkpoint.
Output: results/frontier/vlm_step019_e2e_readout__{SLOT}.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(ROOT := Path(__file__).parent.parent.parent))
if (_LIBS := ROOT / "vlm_libs").is_dir(): sys.path.insert(0, str(_LIBS))

import torch
import torch.nn as nn
from PIL import Image

from scripts.frontier.vlm_eval import LABELS, WNID2LABEL, VLMEval, sample_images, sync
from scripts.frontier.vlm_step017_fused_dequant import run as tri_run

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
parser.add_argument("--device", default="cuda")
parser.add_argument("--arms", nargs="+",
                    default=["fp32", "fp16", "w8a8_fake", "w8a8_tri", "w8a8_trif"])
parser.add_argument("--n_eval", type=int, default=3900, help="seed 42 -> step006's 3859 images")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--cfg", type=int, nargs=5, default=[16, 128, 64, 4, 4],
                    help="step018's best config for every T <= 16, which is every T seen here")
parser.add_argument("--smoke_test", action="store_true")
args = parser.parse_args()
if args.smoke_test: args.n_eval = 20

DEVICE = torch.device(args.device)
SLOT = os.environ.get("SGN_SLOT", "local")
OUT = ROOT / "results" / "frontier" / f"vlm_step019_e2e_readout__{SLOT}.json"
DATA = ROOT / "data" / "imagenette2-320"
STEP006_TOP1 = 0.711


def q_rows(w):
    """Symmetric per-row int8. Returns (int8 [N,K], fp32 scale [N]). Same rule as vlm_quant."""
    s = w.abs().amax(1).clamp(min=1e-8) / 127.0
    return (w / s[:, None]).round().clamp(-127, 127).to(torch.int8), s.float()


class Head(nn.Module):
    """One module for every arm; `mode` switches only the arithmetic. Weights, scales and the
    activation quantization rule are shared code, so an accuracy difference between arms cannot come
    from a different quantizer -- only from the path the numbers take."""

    def __init__(self, lin, mode, cfg):
        super().__init__()
        self.mode, self.cfg, self.log = mode, tuple(cfg), []
        w = lin.weight.data                                  # [N=49280, K=576]
        self.w32, self.bias = w, lin.bias
        self.w16 = w.half()
        w8, sw = q_rows(w)
        self.w8t = w8.t().contiguous()                       # [K, N] for the kernel
        self.sw = sw
        self.wdq = (w8.float() * sw[:, None])                # fake-quant weights, dequantized

    def _mm(self, x2):
        if self.mode == "fp32": return x2 @ self.w32.t()
        if self.mode == "fp16": return (x2.half() @ self.w16.t()).float()
        sa = x2.abs().amax(1).clamp(min=1e-8) / 127.0        # per-token activation scale
        x8 = (x2 / sa[:, None]).round().clamp(-127, 127)
        if self.mode == "w8a8_fake":
            return (x8 * sa[:, None]) @ self.wdq.t()
        xq = x8.to(torch.int8)
        if self.mode == "w8a8_trif":                         # fused epilogue -> fp16, one kernel
            return tri_run(xq, self.w8t, sa, self.sw, self.cfg, 1).float()
        acc = tri_run(xq, self.w8t, sa, self.sw, self.cfg, 0)         # int32, unfused
        return acc.float() * sa[:, None] * self.sw[None, :]  # separate dequant -- SS30's choice

    def forward(self, x):
        sync(x.device); t0 = time.perf_counter()
        out = self._mm(x.reshape(-1, x.shape[-1]))
        if self.bias is not None: out = out + self.bias
        sync(x.device); self.log.append((time.perf_counter() - t0) * 1000)
        return out.reshape(*x.shape[:-1], -1)

    def bytes(self):
        n, k = self.w32.shape
        if self.mode == "fp32": return n * k * 4
        if self.mode == "fp16": return n * k * 2
        return n * k + n * 4                                 # int8 weights + one fp32 scale per row


def evaluate(ev, head, val):
    """One pass at a fixed arm. Splits lm_head time into the forward call (1/image, inside the
    harness's `prefill_ms`) and the choose calls (10/image, which the harness never timed)."""
    ev.m.lm_head = head
    ev.run(Image.open(val[0][0]).convert("RGB"))              # warm up: Triton compiles on call 1
    head.log.clear()
    preds, fwd, chz, pre, vis, t0 = [], 0.0, 0.0, 0.0, 0.0, time.perf_counter()
    for i, (path, _) in enumerate(val):
        c0 = len(head.log)
        lg, pred, vm, pm = ev.run(Image.open(path).convert("RGB"))
        # Call 1 of this image is VLMEval.forward; calls 2.. are the ten choose() calls. The
        # boundary is the image, not a call count, so a label whose token run changes nothing.
        ts = head.log[c0:]
        fwd += ts[0]; chz += sum(ts[1:])
        preds.append(pred); vis += vm; pre += pm
        if (i + 1) % 500 == 0:
            el = time.perf_counter() - t0
            print(f"    [{i+1}/{len(val)}] {el:.0f}s eta {el/(i+1)*(len(val)-i-1):.0f}s", flush=True)
    n = len(val)
    return preds, (fwd + chz) / n, fwd / n, len(head.log) / n, vis / n, pre / n


def main():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    print("=" * 82)
    print(f"vlm_step019 e2e readout  device={DEVICE}  arms={args.arms}  cfg={tuple(args.cfg)}")
    m = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float32).eval().to(DEVICE)
    m.requires_grad_(False)
    ev = VLMEval(m, AutoProcessor.from_pretrained(args.model), DEVICE)
    stock = ev.m.lm_head
    val = sample_images(DATA / "val", args.n_eval, seed=args.seed)
    gold = [LABELS.index(WNID2LABEL[w]) for _, w in val]
    print(f"  {len(val)} images  lm_head {tuple(stock.weight.shape)}  bias={stock.bias is not None}")

    out, hb = {}, {}
    for arm in args.arms:
        print(f"  --- {arm} ---", flush=True)
        head = Head(stock, arm, args.cfg)
        hb[arm] = head.bytes()
        out[arm] = evaluate(ev, head, val)
        del head; torch.cuda.empty_cache()   # each arm holds a private 28M-weight copy
    ev.m.lm_head = stock                                      # leave the model as we found it

    n = len(val)
    res = {"step": "vlm_step019", "device": str(DEVICE), "n_eval": n, "seed": args.seed,
           "cfg": list(args.cfg), "arms": args.arms, "cells": {}, "gates": {}}
    print(f"\n  {'arm':<11} {'top1':>6} {'agr_fp32':>9} {'head_ms':>8} {'fwd_ms':>7} {'chz_ms':>7} "
          f"{'calls':>6} {'pre_ms':>7} {'head_MB':>8}")
    for arm in args.arms:
        preds, hms, fms, calls, vis, pre = out[arm]
        row = {"top1": sum(int(a == b) for a, b in zip(preds, gold)) / n,
               "agree_fp32": sum(int(a == b) for a, b in zip(preds, out[args.arms[0]][0])) / n,
               "head_ms": round(hms, 4), "forward_ms": round(fms, 4),
               "choose_ms": round(hms - fms, 4), "calls_per_image": calls,
               "vision_ms": round(vis, 2), "prefill_ms": round(pre, 2), "head_bytes": hb[arm]}
        res["cells"][arm] = row
        print(f"  {arm:<11} {row['top1']:>6.3f} {row['agree_fp32']:>9.3f} {hms:>8.4f} {fms:>7.4f} "
              f"{hms-fms:>7.4f} {calls:>6.1f} {pre:>7.2f} {row['head_bytes']/1e6:>8.1f}")

    if "w8a8_tri" in out and "w8a8_fake" in out:
        a = sum(int(x == y) for x, y in zip(out["w8a8_tri"][0], out["w8a8_fake"][0])) / n
        res["gates"]["kernel_fidelity"] = {"agree": a, "threshold": 0.99,
                                           "verdict": "PASS" if a >= 0.99 else "FAIL"}
        print(f"\n  GATE kernel fidelity  agree(tri, fake) = {a:.4f} vs 0.99 -> "
              f"{res['gates']['kernel_fidelity']['verdict']}")
    if "fp32" in out:
        d = res["cells"]["fp32"]["top1"] - STEP006_TOP1
        res["gates"]["validity"] = {"top1": res["cells"]["fp32"]["top1"], "step006": STEP006_TOP1,
                                    "delta": d, "verdict": "PASS" if abs(d) <= 0.02 else "FAIL"}
        print(f"  GATE validity  d12 fp32 top1 {res['cells']['fp32']['top1']:.3f} vs step006 "
              f"{STEP006_TOP1} (d {d:+.4f}) -> {res['gates']['validity']['verdict']}")
    for base in ("fp32", "fp16"):
        for tgt in [a for a in ("w8a8_tri", "w8a8_trif") if a in out and base in out]:
            b, t = res["cells"][base], res["cells"][tgt]
            print(f"  {tgt} vs {base}: head {b['head_ms']/t['head_ms']:.2f}x  "
                  f"forward {b['forward_ms']/t['forward_ms']:.2f}x  "
                  f"bytes {b['head_bytes']/t['head_bytes']:.2f}x  "
                  f"| prefill_ms {b['prefill_ms']:.2f} -> {t['prefill_ms']:.2f}")
    print("  head_ms is ALL 11 calls; the harness's prefill_ms contains only forward_ms of them.")
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(res, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
