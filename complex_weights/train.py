# Phase-0 sanity trainer. Compares real baseline vs one-layer PhaseConv swap
# on the synthetic foreground task. Reports F-measure + params (iso-param check).
# Usage: python train.py --swap 3 --steps 400
import argparse
import math
import torch
import torch.nn.functional as F
from models import TinySegNet, count_params
from data import batch, f_measure


def run(swap_layers, steps, width, encode, variant, seed, device, gap=0.40, noise=0.05, n_roots=0, quiet=False):
    torch.manual_seed(seed)
    net = TinySegNet(width=width, swap_layers=swap_layers, encode=encode, variant=variant, n_roots=n_roots).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    net.train()
    for s in range(steps):
        img, mask = batch(B=32, seed=s, device=device, gap=gap, noise=noise)
        logits = net(img)
        loss = F.binary_cross_entropy_with_logits(logits, mask)
        opt.zero_grad(); loss.backward(); opt.step()
        if not quiet and (s % max(1, steps // 5) == 0 or s == steps - 1):
            print(f"  step {s:4d}  loss {loss.item():.4f}")
    # eval on held-out seeds
    net.eval()
    fs = []
    with torch.no_grad():
        for i in range(8):
            img, mask = batch(B=64, seed=10000 + i, device=device, gap=gap, noise=noise)
            fs.append(f_measure(net(img), mask))
    fmean = sum(fs) / len(fs)
    return fmean, count_params(net)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swap", type=int, nargs="*", default=None,
                    help="layer indices 0..5 to make PhaseConv; omit = real baseline")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--encode", default="wrap")
    ap.add_argument("--variant", default="real")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gap", type=float, default=0.40)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--roots", type=int, default=0, help="quantize theta to R roots of unity (0=continuous)")
    ap.add_argument("--sweep", action="store_true", help="baseline + each single-layer swap")
    ap.add_argument("--full", action="store_true", help="baseline vs all-layers-phase")
    ap.add_argument("--qsweep", action="store_true", help="full-phase, sweep root count R (bits)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} encode={args.encode} variant={args.variant} steps={args.steps}")

    common = dict(steps=args.steps, width=args.width, encode=args.encode,
                  variant=args.variant, seed=args.seed, device=device,
                  gap=args.gap, noise=args.noise, n_roots=args.roots)
    print(f"gap={args.gap} noise={args.noise} roots={args.roots}")
    allL = tuple(range(TinySegNet.LAYERS))
    if args.qsweep:
        base_f, base_p = run((), quiet=True, **{k: v for k, v in common.items() if k != "n_roots"}, n_roots=0)
        print(f"BASELINE (all real)      F={base_f:.4f}")
        cont_f, _ = run(allL, quiet=True, **{k: v for k, v in common.items() if k != "n_roots"}, n_roots=0)
        print(f"FULL PHASE continuous    F={cont_f:.4f}  dF={cont_f-base_f:+.4f}")
        for R in (16, 8, 4, 2):
            f, _ = run(allL, quiet=True, **{k: v for k, v in common.items() if k != "n_roots"}, n_roots=R)
            print(f"FULL PHASE R={R:<2d} ({math.log2(R):.0f}b) F={f:.4f}  dF={f-base_f:+.4f}")
        return
    if args.sweep:
        base_f, base_p = run((), quiet=True, **common)
        print(f"BASELINE (all real)      F={base_f:.4f}  params={base_p}")
        for L in range(TinySegNet.LAYERS):
            f, p = run((L,), quiet=True, **common)
            print(f"swap L{L}                   F={f:.4f}  params={p}  dF={f-base_f:+.4f}")
    elif args.full:
        base_f, base_p = run((), quiet=True, **common)
        print(f"BASELINE (all real)      F={base_f:.4f}  params={base_p}")
        allL = tuple(range(TinySegNet.LAYERS))
        f, p = run(allL, quiet=True, **common)
        print(f"FULL PHASE (all 6)       F={f:.4f}  params={p}  dF={f-base_f:+.4f}")
    else:
        swap = tuple(args.swap) if args.swap else ()
        f, p = run(swap, **common)
        tag = f"swap={swap}" if swap else "baseline(real)"
        print(f"RESULT {tag}  F={f:.4f}  params={p}")


if __name__ == "__main__":
    main()
