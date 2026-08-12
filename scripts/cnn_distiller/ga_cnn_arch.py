"""GA architecture search for MultiScaleCNN.

Searches: channel widths, dilation rates, CReLU placement, expansion factor.
Fitness: val_acc * (REF_MACS / model_macs)^0.1  — rewards efficiency.
Fast eval: EVAL_EPOCHS epochs on EVAL_FRAC of training data (GA pre-filter only).
Top configs from GA should be re-evaluated at full T0 (20ep, 50% data).

REF_MACS = 183.2M (EfficientVGG Ref baseline).
"""
from __future__ import annotations
import math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from scripts.cnn_distiller.model_multiscale import MultiScaleCNN, count_macs

REF_MACS   = 183.2e6
MAX_MACS   = 1_200e6   # hard cap: 1.2B MACs

ARCH_SPACE: dict = {
    "C1":       {"type": "discrete", "values": [32, 48, 64, 96]},
    "C2":       {"type": "discrete", "values": [64, 96, 128, 192]},
    "C3":       {"type": "discrete", "values": [128, 192, 256, 384]},
    "n_br":     {"type": "discrete", "values": [1, 2, 3]},
    "max_dil":  {"type": "discrete", "values": [1, 2, 4]},
    "crelu_b1": {"type": "discrete", "values": [0, 1]},
    "crelu_b2": {"type": "discrete", "values": [0, 1]},
    "crelu_b3": {"type": "discrete", "values": [0, 1]},
    "expansion":{"type": "discrete", "values": [1, 2, 4]},
}


def _dil_rates(n_br: int, max_dil: int) -> tuple:
    if n_br == 1 or max_dil == 1:
        return (1,) * n_br
    if n_br == 2:
        return (1, max_dil)
    return (1, max(1, max_dil // 2), max_dil)


def _build(cfg: dict) -> MultiScaleCNN:
    return MultiScaleCNN(
        channels=(cfg["C1"], cfg["C2"], cfg["C3"]),
        dil_rates=_dil_rates(cfg["n_br"], cfg["max_dil"]),
        expansion=cfg["expansion"],
        crelu_mask=(bool(cfg["crelu_b1"]), bool(cfg["crelu_b2"]), bool(cfg["crelu_b3"])),
    )


def _sample() -> dict:
    return {k: random.choice(s["values"]) for k, s in ARCH_SPACE.items()}


def _mutate(cfg: dict, p: float = 0.35) -> dict:
    return {
        k: random.choice(ARCH_SPACE[k]["values"]) if random.random() < p else cfg[k]
        for k in ARCH_SPACE
    }


class GACNNSearch:
    """Population-based GA for MultiScaleCNN architecture search."""

    def __init__(
        self,
        train_ds,
        val_ds,
        population: int = 10,
        generations: int = 6,
        top_k: int = 4,
        eval_fraction: float = 0.10,
        eval_epochs: int = 5,
        batch: int = 32,
        device: str = "mps",
    ) -> None:
        self.train_ds     = train_ds
        self.val_ds       = val_ds
        self.population   = population
        self.generations  = generations
        self.top_k        = top_k
        self.eval_fraction= eval_fraction
        self.eval_epochs  = eval_epochs
        self.batch        = batch
        self.device       = torch.device(device)

    # ------------------------------------------------------------------ #
    def _loaders(self):
        n  = len(self.train_ds)
        k  = max(64, int(n * self.eval_fraction))
        idx = torch.randperm(n)[:k].tolist()
        tr = DataLoader(Subset(self.train_ds, idx), batch_size=self.batch,
                        shuffle=True, num_workers=0)
        vl = DataLoader(self.val_ds, batch_size=self.batch,
                        shuffle=False, num_workers=0)
        return tr, vl

    def _fitness(self, cfg: dict) -> tuple[float, int, int]:
        """Return (fitness, val_correct, n_macs).  -1.0 fitness on failure."""
        try:
            model = _build(cfg).to(self.device)
        except Exception:
            return -1.0, 0, 0

        macs = count_macs(model)
        if macs > MAX_MACS:
            return -1.0, 0, macs

        n_params = sum(p.numel() for p in model.parameters())
        if n_params > 5_000_000:
            return -1.0, 0, macs

        opt  = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        tr, vl = self._loaders()

        model.train()
        for _ in range(self.eval_epochs):
            for batch in tr:
                imgs, _feat, soft, labels = batch
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                logits, _ = model(imgs)
                loss = F.cross_entropy(logits, labels)
                opt.zero_grad(); loss.backward(); opt.step()

        model.eval(); correct = total = 0
        with torch.no_grad():
            for batch in vl:
                imgs, _feat, soft, labels = batch
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                logits, _ = model(imgs)
                correct += (logits.argmax(1) == labels).sum().item()
                total   += len(labels)

        val_acc = correct / total if total else 0.0
        fitness = val_acc * (REF_MACS / macs) ** 0.10
        return fitness, correct, macs

    # ------------------------------------------------------------------ #
    def run(self) -> list[dict]:
        """Run GA. Returns list of top_k configs sorted by fitness (best first)."""
        pop = [_sample() for _ in range(self.population)]
        scored: list[tuple[dict, float, int]] = []   # (cfg, fitness, macs)

        for gen in range(self.generations):
            gen_scored = []
            for cfg in pop:
                fit, _, macs = self._fitness(cfg)
                gen_scored.append((cfg, fit, macs))
                macs_m = macs / 1e6
                print(f"  gen{gen} fit={fit:.4f} MACs={macs_m:.1f}M "
                      f"C=({cfg['C1']},{cfg['C2']},{cfg['C3']}) "
                      f"br={cfg['n_br']} dil={cfg['max_dil']} "
                      f"crelu=({cfg['crelu_b1']},{cfg['crelu_b2']},{cfg['crelu_b3']}) "
                      f"exp={cfg['expansion']}", flush=True)

            scored.extend(gen_scored)
            ranked = sorted(gen_scored, key=lambda x: x[1], reverse=True)
            parents = [cfg for cfg, _, _ in ranked[:self.top_k]]
            pop = list(parents)
            while len(pop) < self.population:
                pop.append(_mutate(random.choice(parents)))
            print(f"Gen {gen} best fit={ranked[0][1]:.4f} MACs={ranked[0][2]/1e6:.1f}M")

        # Deduplicate by config key, return top_k unique
        seen, result = set(), []
        for cfg, fit, macs in sorted(scored, key=lambda x: x[1], reverse=True):
            key = str(sorted(cfg.items()))
            if key not in seen:
                seen.add(key)
                result.append({"config": cfg, "ga_fitness": fit, "macs_M": macs / 1e6})
            if len(result) == self.top_k:
                break
        return result
