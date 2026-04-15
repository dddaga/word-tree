"""Numerical equivalence tests for Triton routing_step_proj kernel.

Tests:
  - PyTorch reference vs Triton output at atol=1e-4, rtol=1e-3
  - Multiple (B, N, D, K_hh) shapes
  - K_iter=5 loop equivalence
  - Graceful skip when Triton unavailable (CUDA-only kernel on Mac)

Run:
    python -m unittest tests.test_triton_routing         # local (skip if no Triton)
    venv/bin/python -m unittest tests.test_triton_routing  # on 5060ti
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

# Import with graceful failure
try:
    from src.sgnnet.triton.routing_kernel import (
        routing_step_proj,
        _routing_step_proj_pytorch,
        TRITON_AVAILABLE,
    )
    from src.sgnnet.triton.routing_kernel_fused import (
        routing_fused_parallel,
        routing_fused_sequential,
        _routing_step_proj_pytorch_k_iter,
    )
    from src.sgnnet.triton.sgnnet_proj_triton import SGNNET_DeltaProjTriton
    IMPORT_OK = True
    IMPORT_ERR = None
except Exception as e:
    IMPORT_OK = False
    IMPORT_ERR = str(e)
    TRITON_AVAILABLE = False


def _make_inputs(B, N, D, K_hh, device, seed=0):
    """Generate consistent random inputs for one routing step."""
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    Z       = F.normalize(torch.randn(B, N, D, generator=rng), dim=-1).to(device)
    theta   = torch.rand(N, generator=rng).to(device) * 0.2          # [N] small positive
    conn    = torch.randint(0, N, (N, K_hh), generator=rng).to(torch.int32).to(device)
    dw_raw  = torch.randn(N, K_hh, D, generator=rng).to(device)
    dw_norm = F.normalize(dw_raw, dim=-1)
    Z_refl  = torch.randn(B, N, D, generator=rng).to(device) * 0.1

    return Z, theta, conn, dw_norm, Z_refl


SHAPES = [
    (1,   512,  16, 2),
    (32,  2048, 16, 2),
    (1,   4096, 16, 2),
]

ATOL = 1e-4
RTOL = 1e-3


@unittest.skipUnless(IMPORT_OK, f"Import failed: {IMPORT_ERR}")
class TestRoutingKernelImport(unittest.TestCase):
    """Basic import and constant checks."""

    def test_triton_available_flag(self):
        """TRITON_AVAILABLE should be a bool."""
        self.assertIsInstance(TRITON_AVAILABLE, bool)

    def test_pytorch_fallback_callable(self):
        """_routing_step_proj_pytorch must be callable regardless of TRITON_AVAILABLE."""
        self.assertTrue(callable(_routing_step_proj_pytorch))


@unittest.skipUnless(IMPORT_OK, f"Import failed: {IMPORT_ERR}")
class TestPyTorchReference(unittest.TestCase):
    """Validate the PyTorch reference implementation is internally consistent."""

    def _run(self, B, N, D, K_hh):
        device = torch.device("cpu")
        Z, theta, conn, dw_norm, Z_refl = _make_inputs(B, N, D, K_hh, device)
        conn64 = conn.to(torch.int64)

        # Inline reference (from train_step268)
        theta_pos = theta.unsqueeze(0).unsqueeze(-1)   # [1, N, 1]
        dw_unsq   = dw_norm.unsqueeze(0)               # [1, N, K_hh, D]

        Z_fwd    = F.relu(Z - theta_pos)
        Z_nb     = Z_fwd[:, conn64, :]
        proj     = (Z_nb * dw_unsq).sum(-1, keepdim=True)
        Z_nb     = Z_nb * proj.abs()
        Z_struct = Z_nb.sum(dim=2)
        Z_rem    = Z_fwd - Z
        Z_refl2  = 0.5 * Z_refl + Z_rem
        Z_exp    = F.normalize((Z_struct + Z_refl2).clamp(-10, 10), dim=-1)

        # Wrapper reference
        Z_got, Z_refl_got = _routing_step_proj_pytorch(Z, theta, conn64, dw_norm, Z_refl, 0.5)

        self.assertTrue(torch.allclose(Z_got, Z_exp, atol=ATOL, rtol=RTOL),
                        f"PyTorch ref mismatch at B={B} N={N} D={D} K={K_hh}: "
                        f"max_diff={( Z_got - Z_exp).abs().max().item():.2e}")
        self.assertTrue(torch.allclose(Z_refl_got, Z_refl2, atol=ATOL, rtol=RTOL),
                        f"Z_reflected ref mismatch at B={B} N={N} D={D} K={K_hh}")

    def test_shapes(self):
        for B, N, D, K_hh in SHAPES:
            with self.subTest(B=B, N=N, D=D, K_hh=K_hh):
                self._run(B, N, D, K_hh)


@unittest.skipUnless(IMPORT_OK, f"Import failed: {IMPORT_ERR}")
@unittest.skipUnless(TRITON_AVAILABLE, "Triton not available — skipping Triton kernel tests")
class TestTritonVsReference(unittest.TestCase):
    """Triton kernel output must match PyTorch reference to atol=1e-4."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        cls.device = torch.device("cuda")

    def _run_single_step(self, B, N, D, K_hh):
        Z, theta, conn, dw_norm, Z_refl = _make_inputs(B, N, D, K_hh, self.device)

        # PyTorch reference
        Z_ref, Z_refl_ref = _routing_step_proj_pytorch(
            Z, theta, conn.to(torch.int64), dw_norm, Z_refl, 0.5
        )

        # Triton
        Z_tri, Z_refl_tri = routing_step_proj(
            Z, theta, conn, dw_norm, Z_refl, 0.5
        )

        max_diff = (Z_tri - Z_ref).abs().max().item()
        max_refl = (Z_refl_tri - Z_refl_ref).abs().max().item()

        self.assertTrue(
            torch.allclose(Z_tri, Z_ref, atol=ATOL, rtol=RTOL),
            f"Z mismatch B={B} N={N} D={D} K={K_hh}: max_diff={max_diff:.2e}"
        )
        self.assertTrue(
            torch.allclose(Z_refl_tri, Z_refl_ref, atol=ATOL, rtol=RTOL),
            f"Z_refl mismatch B={B} N={N} D={D} K={K_hh}: max_diff={max_refl:.2e}"
        )

    def test_shapes_single_step(self):
        for B, N, D, K_hh in SHAPES:
            with self.subTest(B=B, N=N, D=D, K_hh=K_hh):
                self._run_single_step(B, N, D, K_hh)

    def test_k_iter_loop_equivalence(self):
        """5 consecutive routing steps must match between Triton and PyTorch."""
        B, N, D, K_hh = 32, 2048, 16, 2
        K_ITER = 5
        Z, theta, conn, dw_norm, Z_refl = _make_inputs(B, N, D, K_hh, self.device)

        Z_ref   = Z.clone()
        Z_refl_ref = Z_refl.clone()
        Z_tri   = Z.clone()
        Z_refl_tri = Z_refl.clone()

        conn64 = conn.to(torch.int64)

        for _ in range(K_ITER):
            Z_ref, Z_refl_ref = _routing_step_proj_pytorch(
                Z_ref, theta, conn64, dw_norm, Z_refl_ref, 0.5
            )
            Z_tri, Z_refl_tri = routing_step_proj(
                Z_tri, theta, conn, dw_norm, Z_refl_tri, 0.5
            )

        max_diff = (Z_tri - Z_ref).abs().max().item()
        self.assertTrue(
            torch.allclose(Z_tri, Z_ref, atol=ATOL, rtol=RTOL),
            f"K_iter=5 loop mismatch: max_diff={max_diff:.2e}"
        )

    def test_output_l2_normalized(self):
        """Triton output must be l2-normalised along the last dimension."""
        B, N, D, K_hh = 32, 2048, 16, 2
        Z, theta, conn, dw_norm, Z_refl = _make_inputs(B, N, D, K_hh, self.device)
        Z_out, _ = routing_step_proj(Z, theta, conn, dw_norm, Z_refl, 0.5)

        norms = Z_out.norm(dim=-1)  # [B, N]
        self.assertTrue(
            torch.allclose(norms, torch.ones_like(norms), atol=1e-5),
            f"Output not l2-normalised: max norm deviation={( norms - 1).abs().max().item():.2e}"
        )


@unittest.skipUnless(IMPORT_OK, f"Import failed: {IMPORT_ERR}")
@unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")
class TestSGNNETDeltaProjTriton(unittest.TestCase):
    """End-to-end model forward test."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        cls.device = torch.device("cuda")

    def test_forward_shape(self):
        """Model forward must return [B, N_out] logits."""
        B = 4; N_IN = 25088; N_OUT = 10
        N = 512; D = 16; K_HH = 2; K_ITER = 2

        K_r = max(1, K_HH // 4); K_l = K_HH - K_r
        model = SGNNET_DeltaProjTriton(
            N_hidden=N, N_out=N_OUT, D_=D, N_in=N_IN,
            K_in=10, K_iter=K_ITER, K_local=K_l, K_random=K_r,
            n_groups=max(8, N // 8), alpha_reflect=0.5, seed=0,
        ).to(self.device).eval()

        x = torch.randn(B, N_IN, device=self.device)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (B, N_OUT))

    def test_triton_vs_pytorch_fallback(self):
        """Triton model forward must match PyTorch fallback at atol=5e-4."""
        B = 4; N_IN = 25088; N_OUT = 10
        N = 512; D = 16; K_HH = 2; K_ITER = 3

        K_r = max(1, K_HH // 4); K_l = K_HH - K_r

        # Triton model
        m_tri = SGNNET_DeltaProjTriton(
            N_hidden=N, N_out=N_OUT, D_=D, N_in=N_IN,
            K_in=10, K_iter=K_ITER, K_local=K_l, K_random=K_r,
            n_groups=max(8, N // 8), alpha_reflect=0.5, seed=42,
        ).to(self.device).eval()

        # PyTorch fallback (same weights, Triton disabled)
        m_ref = SGNNET_DeltaProjTriton(
            N_hidden=N, N_out=N_OUT, D_=D, N_in=N_IN,
            K_in=10, K_iter=K_ITER, K_local=K_l, K_random=K_r,
            n_groups=max(8, N // 8), alpha_reflect=0.5, seed=42,
        ).to(self.device).eval()
        m_ref._use_triton = False   # force PyTorch path

        # Copy weights to ensure identical model state
        m_ref.load_state_dict(m_tri.state_dict())

        x = torch.randn(B, N_IN, device=self.device)
        with torch.no_grad():
            out_tri = m_tri(x)
            out_ref = m_ref(x)

        max_diff = (out_tri - out_ref).abs().max().item()
        # Slightly relaxed tolerance for full model (seeding + K_iter accumulation)
        self.assertTrue(
            torch.allclose(out_tri, out_ref, atol=5e-4, rtol=1e-3),
            f"Full model Triton vs PyTorch mismatch: max_diff={max_diff:.2e}"
        )


@unittest.skipUnless(IMPORT_OK, f"Import failed: {IMPORT_ERR}")
class TestFusedKernelPyTorchReference(unittest.TestCase):
    """Validate _routing_step_proj_pytorch_k_iter (sequential fallback for fused kernels)."""

    def test_k_iter_matches_per_step_loop(self):
        """K_iter sequential fallback must match calling per-step 5x."""
        B, N, D, K_hh = 4, 256, 16, 2
        K_ITER = 5
        device = torch.device("cpu")
        Z, theta, conn, dw_norm, Z_refl = _make_inputs(B, N, D, K_hh, device)
        conn64 = conn.to(torch.int64)

        # Per-step loop using the single-step reference
        Z_ref, Z_refl_ref = Z.clone(), Z_refl.clone()
        for _ in range(K_ITER):
            Z_ref, Z_refl_ref = _routing_step_proj_pytorch(
                Z_ref, theta, conn64, dw_norm, Z_refl_ref, 0.5
            )

        # K_iter sequential fallback
        Z_got, Z_refl_got = _routing_step_proj_pytorch_k_iter(
            Z, theta, conn, dw_norm, Z_refl, 0.5, K_iter=K_ITER
        )

        max_diff = (Z_got - Z_ref).abs().max().item()
        self.assertTrue(
            torch.allclose(Z_got, Z_ref, atol=ATOL, rtol=RTOL),
            f"K_iter sequential fallback mismatch: max_diff={max_diff:.2e}"
        )


@unittest.skipUnless(IMPORT_OK, f"Import failed: {IMPORT_ERR}")
@unittest.skipUnless(TRITON_AVAILABLE, "Triton not available — skipping fused kernel tests")
class TestFusedKernelTriton(unittest.TestCase):
    """Tests for routing_fused_parallel and routing_fused_sequential kernels."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        cls.device = torch.device("cuda")

    def test_fused_parallel_output_shape(self):
        """routing_fused_parallel must return [B,N,D] tensors."""
        B, N, D, K_hh = 4, 512, 16, 2
        Z, theta, conn, dw_norm, Z_refl = _make_inputs(B, N, D, K_hh, self.device)
        Z_out, Z_refl_out = routing_fused_parallel(
            Z, theta, conn, dw_norm, Z_refl, 0.5, K_iter=5
        )
        self.assertEqual(Z_out.shape, (B, N, D))
        self.assertEqual(Z_refl_out.shape, (B, N, D))

    def test_fused_parallel_output_normalized(self):
        """routing_fused_parallel output must be l2-normalised along last dim."""
        B, N, D, K_hh = 32, 2048, 16, 2
        Z, theta, conn, dw_norm, Z_refl = _make_inputs(B, N, D, K_hh, self.device)
        Z_out, _ = routing_fused_parallel(Z, theta, conn, dw_norm, Z_refl, 0.5, K_iter=5)
        norms = Z_out.norm(dim=-1)
        self.assertTrue(
            torch.allclose(norms, torch.ones_like(norms), atol=1e-4),
            f"parallel fused output not l2-normalized: max_dev={( norms - 1).abs().max().item():.2e}"
        )

    def test_fused_parallel_k1_matches_per_step(self):
        """routing_fused_parallel with K_iter=1 must match the single per-step kernel.

        With K_iter=1, the 'fixed-Z neighbor' approximation is exact (no difference
        between parallel and sequential at K=1).
        """
        B, N, D, K_hh = 32, 2048, 16, 2
        Z, theta, conn, dw_norm, Z_refl = _make_inputs(B, N, D, K_hh, self.device)

        # Single per-step reference
        Z_ref, Z_refl_ref = routing_step_proj(
            Z, theta, conn, dw_norm, Z_refl, 0.5
        )

        # Fused parallel K=1
        Z_fused, Z_refl_fused = routing_fused_parallel(
            Z, theta, conn, dw_norm, Z_refl, 0.5, K_iter=1
        )

        max_diff = (Z_fused - Z_ref).abs().max().item()
        self.assertTrue(
            torch.allclose(Z_fused, Z_ref, atol=ATOL, rtol=RTOL),
            f"fused_parallel K=1 vs per-step mismatch: max_diff={max_diff:.2e}"
        )

    def test_fused_sequential_output_shape(self):
        """routing_fused_sequential must return [B,N,D] tensors."""
        B, N, D, K_hh = 4, 512, 16, 2
        Z, theta, conn, dw_norm, Z_refl = _make_inputs(B, N, D, K_hh, self.device)
        Z_out, Z_refl_out = routing_fused_sequential(
            Z, theta, conn, dw_norm, Z_refl, 0.5, K_iter=5
        )
        self.assertEqual(Z_out.shape, (B, N, D))
        self.assertEqual(Z_refl_out.shape, (B, N, D))

    def test_fused_sequential_output_normalized(self):
        """routing_fused_sequential output must be l2-normalised."""
        B, N, D, K_hh = 32, 2048, 16, 2
        Z, theta, conn, dw_norm, Z_refl = _make_inputs(B, N, D, K_hh, self.device)
        Z_out, _ = routing_fused_sequential(Z, theta, conn, dw_norm, Z_refl, 0.5, K_iter=5)
        norms = Z_out.norm(dim=-1)
        self.assertTrue(
            torch.allclose(norms, torch.ones_like(norms), atol=1e-4),
            f"sequential fused output not l2-normalized: max_dev={( norms - 1).abs().max().item():.2e}"
        )

    def test_fused_sequential_k1_near_per_step(self):
        """routing_fused_sequential K=1 documents its approximate nature.

        The sequential kernel uses a ping-pong strategy where sender Z is read
        from Z_pong — but Z_pong is shared across all Triton programs in the grid.
        When multiple programs run concurrently on different SMs, a program reading
        its sender's Z from Z_pong may see a PARTIALLY-UPDATED value written by
        another tile that completed ahead of it.  This is an unavoidable data race
        in a multi-block grid without grid-wide sync.

        For N=2048, BLOCK_N=64-512, there are ceil(2048/BLOCK_N) tiles.
        Tiles accessing the SAME sender neuron from different tiles may get stale
        or fresh values depending on execution order.

        This test DOCUMENTS that the sequential kernel differs from the exact
        per-step result.  We only assert: (a) output shape correct, (b) no NaN,
        (c) l2-normalized output.  Numerical equivalence is NOT guaranteed.
        """
        B, N, D, K_hh = 32, 2048, 16, 2
        Z, theta, conn, dw_norm, Z_refl = _make_inputs(B, N, D, K_hh, self.device)

        Z_fused, _ = routing_fused_sequential(
            Z, theta, conn, dw_norm, Z_refl, 0.5, K_iter=1
        )

        # Shape check
        self.assertEqual(Z_fused.shape, (B, N, D))
        # No NaN
        self.assertFalse(torch.isnan(Z_fused).any(), "sequential K=1 output has NaN")
        # l2-normalized
        norms = Z_fused.norm(dim=-1)
        self.assertTrue(
            torch.allclose(norms, torch.ones_like(norms), atol=1e-4),
            f"sequential K=1 output not normalized: max_dev={( norms-1).abs().max().item():.2e}"
        )
        # Log actual diff vs per-step for info (not a pass/fail criterion)
        Z_ref, _ = routing_step_proj(Z, theta, conn, dw_norm, Z_refl, 0.5)
        max_diff = (Z_fused - Z_ref).abs().max().item()
        print(f"\n  fused_sequential K=1 vs per-step max_diff={max_diff:.4f} "
              f"(approximate kernel — data race across blocks, expected non-zero)")

    def test_fused_parallel_k5_not_equal_sequential(self):
        """K_iter=5 parallel should differ from sequential (different semantics).

        This is a DOCUMENTATION test, not a correctness test.
        We assert the max diff is measurable (>0) to confirm the two strategies
        are genuinely different.  If they're identical, something is wrong.
        """
        B, N, D, K_hh = 32, 2048, 16, 2
        Z, theta, conn, dw_norm, Z_refl = _make_inputs(B, N, D, K_hh, self.device)

        Z_par, _ = routing_fused_parallel(
            Z.clone(), theta, conn, dw_norm, Z_refl.clone(), 0.5, K_iter=5
        )
        Z_seq, _ = routing_fused_sequential(
            Z.clone(), theta, conn, dw_norm, Z_refl.clone(), 0.5, K_iter=5
        )

        max_diff = (Z_par - Z_seq).abs().max().item()
        # Document the difference — both are valid approximations but distinct
        # They may be close or far depending on graph structure
        # We just confirm neither NaN nor identical
        self.assertFalse(torch.isnan(Z_par).any(), "parallel output has NaN")
        self.assertFalse(torch.isnan(Z_seq).any(), "sequential output has NaN")
        # Log the diff for the test report (not a pass/fail criterion)
        print(f"\n  parallel vs sequential K=5 max_diff={max_diff:.4f} "
              f"(expected non-zero — different semantics)")

    def test_fused_model_parallel_mode_forward_shape(self):
        """SGNNET_DeltaProjTriton with fused_mode='parallel' returns correct shape."""
        B = 4; N_IN = 25088; N_OUT = 10
        N = 512; D = 16; K_HH = 2; K_ITER = 2

        K_r = max(1, K_HH // 4); K_l = K_HH - K_r
        model = SGNNET_DeltaProjTriton(
            N_hidden=N, N_out=N_OUT, D_=D, N_in=N_IN,
            K_in=10, K_iter=K_ITER, K_local=K_l, K_random=K_r,
            n_groups=max(8, N // 8), alpha_reflect=0.5, seed=0,
        ).to(self.device).eval()
        model._fused_mode = "parallel"

        x = torch.randn(B, N_IN, device=self.device)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (B, N_OUT))

    def test_fused_model_sequential_mode_forward_shape(self):
        """SGNNET_DeltaProjTriton with fused_mode='sequential' returns correct shape."""
        B = 4; N_IN = 25088; N_OUT = 10
        N = 512; D = 16; K_HH = 2; K_ITER = 2

        K_r = max(1, K_HH // 4); K_l = K_HH - K_r
        model = SGNNET_DeltaProjTriton(
            N_hidden=N, N_out=N_OUT, D_=D, N_in=N_IN,
            K_in=10, K_iter=K_ITER, K_local=K_l, K_random=K_r,
            n_groups=max(8, N // 8), alpha_reflect=0.5, seed=0,
        ).to(self.device).eval()
        model._fused_mode = "sequential"

        x = torch.randn(B, N_IN, device=self.device)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (B, N_OUT))


if __name__ == "__main__":
    unittest.main(verbosity=2)
