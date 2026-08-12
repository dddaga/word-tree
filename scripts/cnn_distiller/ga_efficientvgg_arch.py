"""GA architecture search for EfficientVGG.

Motivation: MultiScaleCNN GA search (step006) found configs capping at 37-42% T1.
EfficientVGG (step003) reaches 73.17% T1 with same distillation setup.
Search over EfficientVGG space finds efficient variants of the architecture that works.

Search space:
  C1∈[32,48,64,96], C2∈[64,96,128,192], C3∈[128,192,256,384]
  dw_kernel∈[3,5,7], expansion∈[1,2,4]
  use_side_branch∈[0,1], use_crelu_block3∈[0,1]

Fitness: val_acc * (REF_MACS / model_macs)^0.1
REF_MACS = 183.2M (EfficientVGG Ref — step003 config).
"""
from __future__ import annotations
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from scripts.cnn_distiller.model_efficient_vgg import EfficientVGG, count_macs

REF_MACS = 183.2e6
MAX_MACS = 800e6
MAX_PARAMS = 5_000_000

ARCH_SPACE: dict = {
    "C1":              {"values": [32, 48, 64, 96]},
    "C2":              {"values": [64, 96, 128, 192]},
    "C3":              {"values": [128, 192, 256, 384]},
    "dw_kernel":       {"values": [3, 5, 7]},
    "expansion":       {"values": [1, 2, 4]},
    "use_side_branch": {"values": [0, 1]},
    "use_crelu_block3":{"values": [0, 1]},
}


def _build(cfg: dict) -> EfficientVGG:
    return EfficientVGG(
        channels=(cfg["C1"], cfg["C2"], cfg["C3"]),
        dw_kernel=cfg["dw_kernel"],
        expansion=cfg["expansion"],
        use_side_branch=bool(cfg["use_side_branch"]),
        use_crelu_block3=bool(cfg["use_crelu_block3"]),
    )


def _sample() -> dict:
    return {k: random.choice(s["values"]) for k, s in ARCH_SPACE.items()}


def _mutate(cfg: dict, p: float = 0.30) -> dict:
    return {
        k: random.choice(ARCH_SPACE[k]["values"]) if random.random() < p else cfg[k]
        for k in ARCH_SPACE
    }


class GAEfficientVGGSearch:
    """Population-based GA for EfficientVGG architecture search."""

    def __init__(
        self,
        train_ds,
        val_ds,
        population: int = 12,
        generations: int = 6,
        top_k: int = 5,
        eval_fraction: float = 0.10,
        eval_epochs: int = 5,
        batch: int = 32,
        device: str = "mps",
    ) -> None:
        self.train_ds      = train_ds
        self.val_ds        = val_ds
        self.population    = population
        self.generations   = generations
        self.top_k         = top_k
        self.eval_fraction = eval_fraction
        self.eval_epochs   = eval_epochs
        self.batch         = batch
        self.device        = torch.device(device)

    def _loaders(self):
        n   = len(self.train_ds)
        k   = max(64, int(n * self.eval_fraction))
        idx = torch.randperm(n)[:k].tolist()
        tr  = DataLoader(Subset(self.train_ds, idx), batch_size=self.batch,
                         shuffle=True, num_workers=0)
        vl  = DataLoader(self.val_ds, batch_size=self.batch,
                         shuffle=False, num_workers=0)
        return tr, vl

    def _fitness(self, cfg: dict) -> tuple[float, float, int]:
        """Return (fitness, val_acc, macs). -1.0 on invalid/oversized config."""
        try:
            model = _build(cfg).to(self.device)
        except Exception as e:
            print(f"    [build error] {e}")
            return -1.0, 0.0, 0

        macs   = count_macs(model)
        params = sum(p.numel() for p in model.parameters())
        if macs > MAX_MACS or params > MAX_PARAMS:
            return -1.0, 0.0, macs

        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
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
        return fitness, val_acc, macs

    def run(self) -> list[dict]:
        """Run GA. Returns top_k unique configs sorted by fitness."""
        pop    = [_sample() for _ in range(self.population)]
        scored: list[tuple[dict, float, float, int]] = []

        for gen in range(self.generations):
            gen_scored = []
            for cfg in pop:
                fit, acc, macs = self._fitness(cfg)
                gen_scored.append((cfg, fit, acc, macs))
                print(
                    f"  gen{gen:02d}  fit={fit:.4f}  acc={acc:.4f}  MACs={macs/1e6:.1f}M"
                    f"  C=({cfg['C1']},{cfg['C2']},{cfg['C3']})"
                    f"  k={cfg['dw_kernel']}  exp={cfg['expansion']}"
                    f"  side={cfg['use_side_branch']}  crelu={cfg['use_crelu_block3']}",
                    flush=True,
                )
            scored.extend(gen_scored)
            ranked  = sorted(gen_scored, key=lambda x: x[1], reverse=True)
            parents = [cfg for cfg, _, _, _ in ranked[:self.top_k]]
            pop     = list(parents)
            while len(pop) < self.population:
                pop.append(_mutate(random.choice(parents)))
            best = ranked[0]
            print(
                f"Gen {gen:02d} BEST  fit={best[1]:.4f}  acc={best[2]:.4f}"
                f"  MACs={best[3]/1e6:.1f}M"
                f"  C=({best[0]['C1']},{best[0]['C2']},{best[0]['C3']})"
                f"  k={best[0]['dw_kernel']}  crelu={best[0]['use_crelu_block3']}",
                flush=True,
            )

        seen, result = set(), []
        for cfg, fit, acc, macs in sorted(scored, key=lambda x: x[1], reverse=True):
            key = str(sorted(cfg.items()))
            if key not in seen:
                seen.add(key)
                result.append({
                    "config": cfg, "ga_fitness": fit,
                    "ga_acc": acc, "macs_M": macs / 1e6,
                    "params": sum(p.numel() for p in _build(cfg).parameters()),
                })
            if len(result) == self.top_k:
                break
        return result
