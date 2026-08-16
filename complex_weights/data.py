# Synthetic foreground-segmentation task — fast proxy to iterate PhaseConv before CDnet.
# Background: low-contrast noise texture. Foreground: brighter blobs (random ellipses).
# GT mask = blob support. Cheap, deterministic per-seed, runs in seconds on GPU.
import torch


def _blobs(B, H, W, g, n_max=3):
    mask = torch.zeros(B, 1, H, W, device=g.device)
    ys = torch.linspace(-1, 1, H, device=g.device).view(1, 1, H, 1)
    xs = torch.linspace(-1, 1, W, device=g.device).view(1, 1, 1, W)
    for _ in range(n_max):
        cy = (torch.rand(B, 1, 1, 1, generator=g, device=g.device) * 1.4 - 0.7)
        cx = (torch.rand(B, 1, 1, 1, generator=g, device=g.device) * 1.4 - 0.7)
        ry = 0.12 + 0.20 * torch.rand(B, 1, 1, 1, generator=g, device=g.device)
        rx = 0.12 + 0.20 * torch.rand(B, 1, 1, 1, generator=g, device=g.device)
        on = (torch.rand(B, 1, 1, 1, generator=g, device=g.device) > 0.25).float()
        d = ((ys - cy) / ry) ** 2 + ((xs - cx) / rx) ** 2
        mask = torch.maximum(mask, on * (d < 1.0).float())
    return mask


def batch(B=32, H=64, W=64, seed=0, device="cuda", gap=0.40, noise=0.05):
    # gap = fg/bg mean separation (smaller = harder). noise = sensor sigma.
    g = torch.Generator(device=device).manual_seed(seed)
    mask = _blobs(B, H, W, g)
    lo = 0.5 - gap / 2
    bg = lo + 0.15 * torch.randn(B, 3, H, W, generator=g, device=device)
    fg = lo + gap + 0.15 * torch.randn(B, 3, H, W, generator=g, device=device)
    img = bg * (1 - mask) + fg * mask
    img = img + noise * torch.randn(B, 3, H, W, generator=g, device=device)
    return img.clamp(0, 1), mask


def f_measure(logits, mask, thr=0.5):
    pred = (torch.sigmoid(logits) > thr).float()
    tp = (pred * mask).sum()
    fp = (pred * (1 - mask)).sum()
    fn = ((1 - pred) * mask).sum()
    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    return (2 * prec * rec / (prec + rec + 1e-6)).item()
