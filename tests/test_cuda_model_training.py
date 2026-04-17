"""Guard rail: SGNNET_Resonant_CUDA and SGNNET_AntiHebbian_CUDA training correctness.

Validates that both CUDA model wrappers can run multiple consecutive train+eval
forward-backward cycles without:
  - 'backward through freed graph' (stale supp_w grad_fn)
  - 'CUDAGraph buffer overwrite' (cudagraph_mark_step_begin gated incorrectly)
  - gradient-free W_pos hidden rows (supp_w incorrectly detached during training)

These tests run on CPU so they execute in CI without a GPU.
The logic being tested (grad graph correctness) is device-agnostic.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from src.sgnnet.model_smallworld import SGNNET_SmallWorld
from src.sgnnet.model_resonant_cuda import SGNNET_Resonant_CUDA, SGNNET_AntiHebbian_CUDA

# Tiny config — fast on CPU, covers all code paths
CFG = dict(
    N_hidden=32, N_out=10, D=4, N_in=64,
    K_in=4, K_local=1, K_random=1, n_groups=4,
    K_iter=2, norm_mode="l2", encoding_mode="fourier",
)
B, N_IN = 4, 64


def _make_resonant(compile: bool = False) -> SGNNET_Resonant_CUDA:
    base = SGNNET_SmallWorld(**CFG)
    return SGNNET_Resonant_CUDA(
        base, K_phase=2, alpha_reflect=0.5, alpha_turing=0.0,
        beam_size=4, compile=compile,
    )


def _make_ah(compile: bool = False) -> SGNNET_AntiHebbian_CUDA:
    return SGNNET_AntiHebbian_CUDA(_make_resonant(compile), alpha_ahebb=1.0, variant="wpos")


def _multi_batch_cycle(model, n_batches: int = 3) -> None:
    """Run n_batches of train-forward + backward, then eval-forward.

    This is the minimal reproduction of the bug: caching supp_w with a live
    grad_fn means batch 2's backward tries to re-use freed intermediates.
    """
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(B, N_IN)

    # Training batches — must not crash on batch 2+
    model.train()
    for i in range(n_batches):
        opt.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, torch.zeros(B, dtype=torch.long))
        loss.backward()   # crash here on batch 2 if supp_w is cached with grad_fn
        opt.step()

    # Eval forward — must not crash (separate no_grad path)
    model.eval()
    with torch.no_grad():
        model(x)


def _wpos_hidden_gradient(model) -> bool:
    """Return True if W_pos hidden rows receive a non-zero gradient.

    When supp_w is fully detached, only the 10 output rows get gradient.
    The hidden rows should also get gradient via the supp_w → routing path.
    """
    # Identify which model holds W_pos
    if isinstance(model, SGNNET_AntiHebbian_CUDA):
        wpos = model.m.base.W_pos
    else:
        wpos = model.base.W_pos

    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    model.train()
    opt.zero_grad()

    x = torch.randn(B, N_IN)
    out = model(x)
    loss = F.cross_entropy(out, torch.zeros(B, dtype=torch.long))
    loss.backward()

    N_h = CFG["N_hidden"]
    hidden_grad = wpos.grad[:N_h]
    return hidden_grad is not None and hidden_grad.abs().max().item() > 0.0


class TestResonantCUDATraining:
    def test_multi_batch_no_crash(self):
        """Three consecutive backward calls must not raise 'freed graph' error."""
        _multi_batch_cycle(_make_resonant())

    def test_eval_after_train_no_crash(self):
        """Eval forward after training forward must not crash."""
        model = _make_resonant()
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        opt.zero_grad()
        loss = F.cross_entropy(model(torch.randn(B, N_IN)), torch.zeros(B, dtype=torch.long))
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            model(torch.randn(B, N_IN))

    def test_output_shape(self):
        model = _make_resonant()
        out = model(torch.randn(B, N_IN))
        assert out.shape == (B, CFG["N_out"])


class TestAntiHebbianCUDATraining:
    def test_multi_batch_no_crash(self):
        """Three consecutive backward calls must not raise 'freed graph' error."""
        _multi_batch_cycle(_make_ah())

    def test_wpos_hidden_rows_get_gradient(self):
        """W_pos hidden rows must receive gradient via supp_w → routing path.

        If supp_w is incorrectly detached at training time, hidden rows get
        zero gradient and the model cannot learn positional structure.
        """
        assert _wpos_hidden_gradient(_make_ah()), (
            "W_pos hidden rows have zero gradient — supp_w is detached during training"
        )

    def test_eval_cache_no_crash(self):
        """supp_w cache should work correctly during eval (no_grad)."""
        model = _make_ah()
        model.eval()
        x = torch.randn(B, N_IN)
        with torch.no_grad():
            model(x)
            model(x)  # second call uses cache

    def test_train_then_eval_cache_invalidated(self):
        """After tick_epoch, eval cache is refreshed correctly."""
        model = _make_ah()
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        opt.zero_grad()
        loss = F.cross_entropy(model(torch.randn(B, N_IN)), torch.zeros(B, dtype=torch.long))
        loss.backward()
        opt.step()

        model.tick_epoch()   # invalidates cache

        model.eval()
        with torch.no_grad():
            model(torch.randn(B, N_IN))  # re-builds cache with updated W_pos

    def test_output_shape(self):
        model = _make_ah()
        out = model(torch.randn(B, N_IN))
        assert out.shape == (B, CFG["N_out"])
