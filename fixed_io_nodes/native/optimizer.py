import torch
import torch.nn as nn
from typing import Tuple

from .layer import NativeNeurographLayer


class _GradAccumulator:
    """
    Per-node gradient accumulation with vectorized Adam.
    Simplified from core.gradient_accumulator.UnquantizedGradientAccumulator:
    no shared memory locks, no phase_values cache, no version tracking.
    """

    def __init__(self, node_store, lr: float, accumulation_steps: int,
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 device: str = "cpu"):
        self.node_store = node_store
        self.lr = lr
        self.accumulation_steps = accumulation_steps
        self.betas = betas
        self.eps = eps
        self.device = device

        N = node_store.total_nodes
        V = node_store.vector_dim

        self.phase_grads = torch.zeros((N, V), device=device, dtype=torch.float32)
        self.mag_grads = torch.zeros_like(self.phase_grads)
        self.phase_grad_counts = torch.zeros(N, dtype=torch.int32, device=device)
        self.mag_grad_counts = torch.zeros_like(self.phase_grad_counts)

        # Per-node Adam state
        self.phase_exp_avg = torch.zeros((N, V), device=device, dtype=torch.float32)
        self.phase_exp_avg_sq = torch.zeros_like(self.phase_exp_avg)
        self.phase_state_steps = torch.zeros(N, dtype=torch.int32, device=device)

        self.mag_exp_avg = torch.zeros((N, V), device=device, dtype=torch.float32)
        self.mag_exp_avg_sq = torch.zeros_like(self.mag_exp_avg)
        self.mag_state_steps = torch.zeros(N, dtype=torch.int32, device=device)

    def receive_gradients(self, active_indices: torch.Tensor,
                          phase_grads: torch.Tensor, mag_grads: torch.Tensor):
        if active_indices is None or active_indices.numel() == 0:
            return
        active_indices = active_indices.to(self.device)
        phase_grads = phase_grads.to(self.device)
        mag_grads = mag_grads.to(self.device)

        self.phase_grads.index_add_(0, active_indices, phase_grads)
        self.mag_grads.index_add_(0, active_indices, mag_grads)
        ones = torch.ones_like(active_indices, dtype=self.phase_grad_counts.dtype)
        self.phase_grad_counts.index_add_(0, active_indices, ones)
        self.mag_grad_counts.index_add_(0, active_indices, ones)

    @torch.no_grad()
    def step(self):
        """Apply vectorized Adam to nodes that reached accumulation_steps threshold."""
        phase_ready = self.phase_grad_counts >= self.accumulation_steps
        mag_ready = self.mag_grad_counts >= self.accumulation_steps

        ready_p = phase_ready.nonzero(as_tuple=True)[0]
        ready_m = mag_ready.nonzero(as_tuple=True)[0]

        if ready_p.numel() > 0:
            avg_grad = self.phase_grads[ready_p] / self.phase_grad_counts[ready_p].unsqueeze(-1).float()
            self._vectorized_adam(
                self.node_store.phase_weight.data, ready_p, avg_grad,
                self.phase_exp_avg, self.phase_exp_avg_sq, self.phase_state_steps,
            )
            self.phase_grads[ready_p] = 0.0
            self.phase_grad_counts[ready_p] = 0

        if ready_m.numel() > 0:
            avg_grad = self.mag_grads[ready_m] / self.mag_grad_counts[ready_m].unsqueeze(-1).float()
            self._vectorized_adam(
                self.node_store.mag_weight.data, ready_m, avg_grad,
                self.mag_exp_avg, self.mag_exp_avg_sq, self.mag_state_steps,
            )
            self.mag_grads[ready_m] = 0.0
            self.mag_grad_counts[ready_m] = 0

    def _vectorized_adam(self, param_data, idx, grad, exp_avg, exp_avg_sq, state_steps):
        """
        In-place vectorized Adam on selected rows.
        Same math as UnquantizedGradientAccumulator._vectorized_adam.
        """
        beta1, beta2 = self.betas

        # Extract rows
        p = param_data[idx]
        ea = exp_avg[idx]
        eas = exp_avg_sq[idx]
        ss = state_steps[idx]

        ss += 1
        ea.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        eas.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        step_2d = ss.unsqueeze(-1).float()
        bc1 = 1.0 - beta1 ** step_2d
        bc2 = 1.0 - beta2 ** step_2d

        denom = (eas.sqrt() / torch.sqrt(bc2)).add_(self.eps)
        step_size = self.lr / bc1
        p.sub_(step_size * (ea / denom))

        # Write back
        param_data[idx] = p
        exp_avg[idx] = ea
        exp_avg_sq[idx] = eas
        state_steps[idx] = ss


class NativeGNNOptimizer:
    """
    Dual-path optimizer:
    - Head params (linear layers): standard Adam, updated every batch
    - GNN params (phase_weight, mag_weight): per-node gradient accumulation
      with Adam, updated only when a node reaches accumulation_steps
    """

    def __init__(self, model: nn.Module, lr: float, accumulation_steps: int,
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        # Discover GNN layers
        gnn_layers = [m for m in model.modules() if isinstance(m, NativeNeurographLayer)]
        if not gnn_layers:
            raise ValueError("Model has no NativeNeurographLayer instances")

        self._gnn_layers = gnn_layers

        # Collect GNN param IDs to exclude from head optimizer
        gnn_param_ids = set()
        self._accumulators = []
        for layer in gnn_layers:
            ns = layer._node_store
            gnn_param_ids.add(id(ns.phase_weight))
            gnn_param_ids.add(id(ns.mag_weight))
            device = ns.phase_weight.device
            self._accumulators.append(
                _GradAccumulator(ns, lr, accumulation_steps, betas, eps, device=str(device))
            )

        head_params = [p for p in model.parameters() if id(p) not in gnn_param_ids]
        self._head_optimizer = torch.optim.Adam(head_params, lr=lr, betas=betas, eps=eps)

    def zero_grad(self):
        self._head_optimizer.zero_grad()
        for layer in self._gnn_layers:
            ns = layer._node_store
            if ns.phase_weight.grad is not None:
                ns.phase_weight.grad = None
            if ns.mag_weight.grad is not None:
                ns.mag_weight.grad = None

    def step(self):
        # 1. Extract GNN gradients, feed to accumulators, clear
        for layer, acc in zip(self._gnn_layers, self._accumulators):
            ns = layer._node_store
            pg = ns.phase_weight.grad
            mg = ns.mag_weight.grad

            if pg is not None:
                norms = pg.norm(dim=1)
                active_idx = (norms > 1e-10).nonzero(as_tuple=True)[0]
                if active_idx.numel() > 0:
                    phase_g = pg[active_idx]
                    mag_g = mg[active_idx] if mg is not None else torch.zeros_like(phase_g)
                    acc.receive_gradients(active_idx, phase_g, mag_g)

            acc.step()

            # Clear GNN grads so head Adam doesn't touch them
            ns.phase_weight.grad = None
            ns.mag_weight.grad = None

        # 2. Standard Adam for head params
        self._head_optimizer.step()
