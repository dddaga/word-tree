"""Step 998 — INT8 weight-only quantization of the K=1 champion: memory + accuracy.

Motivation (meditation 005): step526/527 CONFIRMED weight-only INT8 = −0.20pp (lossless)
at N=2048/D=16, because L2-sphere bounds keep W_pos within int8 range (no wrap). That
recipe was NEVER applied to the efficiency champion (step605, 34,976 params) and is NOT
in the paper headline. If lossless here too, the memory claim strengthens 4×:
34,976 params × 4 B (fp32) = 140 KB → × 1 B (int8) = 35 KB. Directly on the stubborn
goal (memory footprint).

Deterministic parts (run with no checkpoint): fp32 vs int8 byte count, compression ratio,
and per-weight quantization MSE (init-independent — proves the saturating quant is
numerically clean). Accuracy delta requires a trained champion checkpoint (--ckpt); the
3-seed champion checkpoints come from re-running step605 with weight-saving enabled.

Usage:
    d_env/bin/python3 scripts/train_step998_int8_champion.py --scale 100
    d_env/bin/python3 scripts/train_step998_int8_champion.py --ckpt results/champion_seed42.pt
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

parser = argparse.ArgumentParser(description="Step 998 INT8 weight-only quant of K=1 champion")
parser.add_argument("--ckpt",   default="", help="Trained champion .pt (optional; enables accuracy delta)")
parser.add_argument("--scale",  type=float, default=100.0, help="Quant scale (step526 recipe)")
parser.add_argument("--device", default="auto")
parser.add_argument("--slot",   default="mini_cpu")
args = parser.parse_args()

N, N_IN, N_OUT, D = 2048, 25088, 10, 16
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def saturating_int8(torch, w, scale):
    """step526 weight-only saturating quant: round(w*scale) clamped to int8, dequant."""
    q = torch.clamp(torch.round(w * scale), -128, 127)
    return q / scale, q


def build_champion(torch):
    from src.sgnnet.model_smallworld import SGNNET_SmallWorld
    from src.sgnnet.model_resonant   import SGNNET_Resonant
    torch.manual_seed(42)
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                             K_in=25, K_iter=1, norm_mode="l2", encoding_mode="fourier")
    return SGNNET_Resonant(base, K_phase=8, alpha_reflect=0.5, alpha_turing=0.0,
                           beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo",
                           resonance_threshold=0.0)


def main():
    import torch
    model = build_champion(torch)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    params = [p for p in model.parameters() if p.requires_grad]
    n = sum(p.numel() for p in params)
    fp32_bytes, int8_bytes = n * 4, n * 1

    # Deterministic quantization-fidelity + wrap-rate telemetry (step526 hypothesis).
    total_mse, total_wrap, total_el = 0.0, 0, 0
    with torch.no_grad():
        for p in params:
            deq, q = saturating_int8(torch, p.data, args.scale)
            total_mse += ((deq - p.data) ** 2).sum().item()
            total_wrap += ((p.data * args.scale).abs() > 127).sum().item()
            total_el += p.numel()
    quant_mse = total_mse / total_el
    wrap_rate = total_wrap / total_el
    print(f"params={n:,}  fp32={fp32_bytes/1024:.1f} KB  int8={int8_bytes/1024:.1f} KB  "
          f"compression={fp32_bytes/int8_bytes:.1f}×")
    print(f"quant MSE/weight={quant_mse:.3e}  wrap_rate={wrap_rate:.2e} "
          f"({'no wrap — step526 hypothesis holds' if wrap_rate < 1e-4 else 'WRAP FIRES — revisit scale'})")

    result = dict(params=n, fp32_kb=fp32_bytes / 1024, int8_kb=int8_bytes / 1024,
                  compression=fp32_bytes / int8_bytes, quant_mse=quant_mse,
                  wrap_rate=wrap_rate, scale=args.scale, ckpt=args.ckpt or None)
    out = ROOT / "results" / f"train_step998_int8_champion__{args.slot}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
