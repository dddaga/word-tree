# CUDA-5060ti-validated
"""Step 997 — Energy (Joules) measurement: SGNNET K=1 champion vs VGG16 FC.

Motivation (meditation 005): the paper's title claims *energy* efficiency but only
FLOPs (a proxy) and wall-time have ever been measured. This bench samples actual GPU
power draw (nvidia-smi --query-gpu=power.draw) during a timed inference loop and
integrates P·dt to report Joules per inference. Converts the headline efficiency
claim from proxy to measured. 5060ti only (needs CUDA power telemetry).

Reports per model: wall-time (ms/inf), mean power (W), energy (mJ/inf), and the
energy ratio vs VGG_FC. Expected: SGNNET student draws far fewer Joules/inference.

Usage:
    venv/bin/python3 scripts/bench_step997_energy_joules.py --device cuda --repeats 300
"""
from __future__ import annotations
import argparse, json, subprocess, threading, time
from pathlib import Path

parser = argparse.ArgumentParser(description="Step 997 energy (Joules) bench — SGNNET vs VGG_FC")
parser.add_argument("--device",  default="cuda")
parser.add_argument("--repeats", type=int, default=300)
parser.add_argument("--warmup",  type=int, default=50)
parser.add_argument("--batch",   type=int, default=32)
parser.add_argument("--slot",    default="5060ti_cuda")
args = parser.parse_args()

N, N_IN, N_OUT, D = 2048, 25088, 10, 16
ROOT = Path(__file__).resolve().parents[1]


class PowerSampler(threading.Thread):
    """Background nvidia-smi power sampler; integrates energy over the run."""
    def __init__(self, interval=0.01):
        super().__init__(daemon=True)
        self.interval, self.stop_flag, self.samples = interval, False, []

    def run(self):
        while not self.stop_flag:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=power.draw",
                     "--format=csv,noheader,nounits"], timeout=1)
                self.samples.append((time.perf_counter(), float(out.decode().split("\n")[0])))
            except Exception:
                pass
            time.sleep(self.interval)

    def mean_power(self):
        vals = [p for _, p in self.samples]
        return sum(vals) / len(vals) if vals else float("nan")


def build_vgg_fc(nn):
    return nn.Sequential(
        nn.Linear(N_IN, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, N_OUT)).eval()


def build_sgnnet_student(torch):
    """Canonical K=1 champion: N=2048, D=16, K_iter=1, K_in=25 (step605 config).

    NOTE: reconcile K_local/K_random/n_groups with step605 module constants on the
    5060ti before the paper row — this builder uses SmallWorld defaults for those.
    """
    from src.sgnnet.model_smallworld import SGNNET_SmallWorld
    from src.sgnnet.model_resonant   import SGNNET_Resonant
    torch.manual_seed(42)
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                             K_in=25, K_iter=1, norm_mode="l2", encoding_mode="fourier")
    return SGNNET_Resonant(base, K_phase=8, alpha_reflect=0.5,
                           alpha_turing=0.0, beam_size=16, geo_gamma=0.5,
                           mode="dynamic_z_geo", resonance_threshold=0.0).eval()


def bench_model(torch, name, model, x, device):
    model = model.to(device)
    with torch.no_grad():
        for _ in range(args.warmup):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    sampler = PowerSampler(); sampler.start()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.repeats):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    sampler.stop_flag = True; sampler.join(timeout=1)
    ms_per = wall / args.repeats * 1e3
    power_w = sampler.mean_power()
    mj_per = power_w * (wall / args.repeats) * 1e3  # W · s → mJ
    print(f"  {name:22s} wall={ms_per:7.3f} ms/inf  power={power_w:6.1f} W  energy={mj_per:8.3f} mJ/inf")
    return dict(name=name, ms_per_inf=ms_per, mean_power_w=power_w, mj_per_inf=mj_per)


def main():
    import torch, torch.nn as nn
    device = torch.device(args.device)
    x_fc = torch.randn(args.batch, N_IN, device=device)
    print(f"Step 997 energy bench — device={device} batch={args.batch} repeats={args.repeats}")
    results = [bench_model(torch, "VGG_FC", build_vgg_fc(nn), x_fc, device)]
    try:
        results.append(bench_model(torch, "SGNNET_K1_champion",
                                   build_sgnnet_student(torch), x_fc, device))
    except Exception as e:
        print(f"  [WARN] SGNNET build failed ({e}); reconcile config on 5060ti.")
    ref = results[0]["mj_per_inf"]
    for r in results:
        r["energy_ratio_vs_vgg"] = ref / r["mj_per_inf"] if r["mj_per_inf"] else float("nan")
    out = ROOT / "results" / f"bench_step997_energy_joules__{args.slot}.json"
    out.write_text(json.dumps(dict(config=vars(args), results=results), indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
