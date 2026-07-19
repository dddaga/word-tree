"""CNN step037 — Wall-time vs MACs reality check for the EfficientVGG distiller.

Motivation (meditation 005): the entire CNN GA line (cnn_step006→036) optimized
*MACs* and converged on GA5 (48,96,384; dw_kernel=3; 141.3M MACs). But a 30-line POC
showed depthwise-separable convs are memory-bound: 8.2× fewer MACs → 2.7× SLOWER wall-time
than dense on mac. So the CNN Pareto story (MAC-based) may not hold in real latency.

This bench times a forward pass for: GA5 (depthwise, low-MAC), a dense-conv baseline
(high-MAC), and reports MACs vs measured ms/inf side by side. If the low-MAC GA5 is not
also the fastest, the paper's CNN efficiency claim must be reframed around wall-time.

Usage:
    d_env/bin/python3 scripts/cnn_distiller/bench_cnn_step037_walltime.py --device mps --repeats 100
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

parser = argparse.ArgumentParser(description="CNN step037 wall-time vs MACs bench")
parser.add_argument("--device",  default="auto")
parser.add_argument("--repeats", type=int, default=100)
parser.add_argument("--warmup",  type=int, default=20)
parser.add_argument("--batch",   type=int, default=32)
parser.add_argument("--slot",    default="mini_mps")
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[2]


def pick_device(torch):
    if args.device != "auto":
        return torch.device(args.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def time_model(torch, name, model, x, device, macs):
    model = model.to(device).eval()
    with torch.no_grad():
        for _ in range(args.warmup):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    if device.type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.repeats):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    if device.type == "mps":
        torch.mps.synchronize()
    ms = (time.perf_counter() - t0) / args.repeats * 1e3
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {name:22s} MACs={macs/1e6:7.1f}M  params={n_params:8,d}  wall={ms:7.3f} ms/inf")
    return dict(name=name, macs=macs, params=n_params, ms_per_inf=ms)


def main():
    import torch
    sys.path.insert(0, str(ROOT))
    from scripts.cnn_distiller.model_efficient_vgg import EfficientVGG, count_macs
    device = pick_device(torch)
    x = torch.randn(args.batch, 3, 224, 224, device=device)
    print(f"CNN step037 wall-time bench — device={device} batch={args.batch} repeats={args.repeats}")

    # GA5 optimum: low-MAC depthwise-heavy
    ga5 = EfficientVGG(channels=(48, 96, 384), dw_kernel=3, expansion=2,
                       use_side_branch=False, use_crelu_block3=True)
    # Dense control: wider channels, standard convs — HIGHER MACs, tests memory-bound thesis
    dense = EfficientVGG(channels=(64, 128, 256), dw_kernel=3, expansion=2,
                         use_side_branch=True, use_crelu_block3=False)
    results = [
        time_model(torch, "GA5 (dw, low-MAC)", ga5, x, device, count_macs(ga5)),
        time_model(torch, "Dense-ctrl (hi-MAC)", dense, x, device, count_macs(dense)),
    ]
    # Verdict: does lower MACs → lower wall-time?
    ga5_r, dense_r = results[0], results[1]
    mac_ratio = dense_r["macs"] / ga5_r["macs"]
    wall_ratio = dense_r["ms_per_inf"] / ga5_r["ms_per_inf"]
    verdict = "MACs PREDICT latency" if wall_ratio > 1.0 else "MACs MISLEAD (GA5 slower despite fewer MACs)"
    print(f"  MAC ratio dense/GA5 = {mac_ratio:.2f}x  →  wall ratio = {wall_ratio:.2f}x  →  {verdict}")
    out = ROOT / "results" / f"bench_cnn_step037_walltime__{args.slot}.json"
    out.write_text(json.dumps(dict(config=vars(args), results=results,
                                   mac_ratio=mac_ratio, wall_ratio=wall_ratio,
                                   verdict=verdict), indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
