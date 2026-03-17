# DistributedNeurographStack and custom autograd Function.
# Uses RPC for forward and backward; workers run local backward when sent gradient slices.
# Per-layer accumulators consume phase/mag grads from workers.

from ._imports import *  # noqa: F401, F403
try:
    from ..main import load_config, get_weights_save_path, gradient_accumulator_process_fn
except ImportError:
    from main import load_config, get_weights_save_path, gradient_accumulator_process_fn

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import os

try:
    from torch.distributed.rpc import init_rpc as rpc_init_rpc, shutdown as rpc_shutdown
    import torch.distributed.rpc as rpc
except (ImportError, AttributeError):
    try:
        from torch.distributed.rpc.api import init_rpc as rpc_init_rpc, shutdown as rpc_shutdown
        import torch.distributed.rpc as rpc
    except (ImportError, AttributeError):
        rpc_init_rpc = rpc_shutdown = None
        rpc = None

import copy
import time
import atexit
from collections import defaultdict

from . import worker as worker_mod
from .gnn_grad_sink import GNNGradientSink
from .worker_pool import WorkerPool
from ._config_utils import get_config, get_node_store_from_config
try:
    from ..core.gradient_accumulator import UnquantizedGradientAccumulator
except ImportError:
    from core.gradient_accumulator import UnquantizedGradientAccumulator


def _normalize_config_list(configs):
    """Ensure configs is a list of dicts; load from paths if strings."""
    out = []
    for c in configs:
        if isinstance(c, str):
            out.append(load_config(c))
        else:
            out.append(c)
    return out


class _DistributedNeurographFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, worker_names, gradient_queue, num_workers, layer_id=0):
        # x: (B, input_node_count, vector_dim). Plain RPC forward; no dist_autograd context.
        B = x.shape[0]
        results = []
        for i in range(B):
            w = i % num_workers
            h1_i = x[i].detach().cpu().requires_grad_(True)
            out_i = rpc.rpc_sync(
                worker_names[w], worker_mod.gnn_forward_distributed, (h1_i, layer_id)
            )
            results.append(out_i)
        stack_out = torch.stack(results)
        ctx.worker_names = worker_names
        ctx.gradient_queue = gradient_queue
        ctx.num_workers = num_workers
        ctx.B = B
        ctx.layer_id = layer_id
        return stack_out.to(x.device)

    @staticmethod
    def backward(ctx, grad_output):
        co = grad_output.cpu()
        if co.shape[0] != ctx.B:
            co = co.expand(ctx.B, *co.shape[1:]).contiguous()
        lid = ctx.layer_id
        # Send each worker its gradient slice; worker runs local backward on stored outputs.
        for w in range(ctx.num_workers):
            grad_slice = [co[i] for i in range(w, ctx.B, ctx.num_workers)]
            if grad_slice:
                rpc.rpc_sync(ctx.worker_names[w], worker_mod.run_backward_distributed, (grad_slice, lid))
        phase_acc = defaultdict(lambda: None)
        mag_acc = defaultdict(lambda: None)
        for w in range(ctx.num_workers):
            p, m = rpc.rpc_sync(ctx.worker_names[w], worker_mod.gnn_get_grads_distributed, (lid,))
            for k, v in (p or {}).items():
                if v is not None:
                    phase_acc[k] = v if phase_acc[k] is None else phase_acc[k] + v
            for k, v in (m or {}).items():
                if v is not None:
                    mag_acc[k] = v if mag_acc[k] is None else mag_acc[k] + v
        phase_dict = dict(phase_acc) if phase_acc else {}
        mag_dict = dict(mag_acc) if mag_acc else {}
        ctx.gradient_queue.put((phase_dict, mag_dict))
        grad_parts = [None] * ctx.B
        for w in range(ctx.num_workers):
            lst = rpc.rpc_sync(ctx.worker_names[w], worker_mod.get_input_grads_distributed, (lid,))
            for j, g in enumerate(lst):
                idx = w + j * ctx.num_workers
                if idx < ctx.B:
                    grad_parts[idx] = g
        grad_input = torch.stack(grad_parts)
        return grad_input.to(grad_output.device), None, None, None, None


class _PooledWorkerFunction(torch.autograd.Function):
    """
    Forward/backward via a long-lived worker pool. Main pushes weights each batch,
    then forward tasks; backward pushes grad_output and collects phase/mag/input grads.
    When B > num_workers, processes in chunks of num_workers.
    """

    @staticmethod
    def forward(ctx, x, config, node_store, gradient_sink, pool, layer_ref):
        B = x.shape[0]
        nw = pool.num_workers
        state_dict = node_store.get_custom_state()
        outputs = [None] * B
        scattering_prob = (
            layer_ref._current_scattering_prob
            if layer_ref._current_scattering_prob is not None
            else layer_ref._scattering_prob_base
        )
        
        # Submit weights ONCE per batch, not per chunk
        pool.submit_weights(state_dict)
        
        for chunk_start in range(0, B, nw):
            chunk_end = min(chunk_start + nw, B)
            for j in range(chunk_end - chunk_start):
                i = chunk_start + j
                xi = x[i].detach().cpu().requires_grad_(True)
                pool.submit_forward(j, xi, scattering_prob)
            for j in range(chunk_end - chunk_start):
                i = chunk_start + j
                outputs[i] = pool.get_forward_result(j)
        stack_out = torch.stack(outputs)
        ctx.x = x
        ctx.B = B
        ctx.gradient_sink = gradient_sink
        ctx.pool = pool
        return stack_out.to(x.device)

    @staticmethod
    def backward(ctx, grad_output):
        co = grad_output.cpu()
        if co.shape[0] != ctx.B:
            co = co.expand(ctx.B, *co.shape[1:]).contiguous()
        pool = ctx.pool
        nw = pool.num_workers
        by_i = [None] * ctx.B
        
        all_idx, all_pg, all_mg = [], [], []
        
        for chunk_start in range(0, ctx.B, nw):
            chunk_end = min(chunk_start + nw, ctx.B)
            for j in range(chunk_end - chunk_start):
                i = chunk_start + j
                clean_grad = co[i].detach().clone().contiguous()
                pool.submit_backward(j, clean_grad)
            for j in range(chunk_end - chunk_start):
                i = chunk_start + j
                idx, pg, mg, ig = pool.get_backward_result(j)
                by_i[i] = ig
                
                # Collect valid tensors for concatenation
                if idx is not None and idx.numel() > 0:
                    all_idx.append(idx)
                    all_pg.append(pg)
                    all_mg.append(mg)

        # Concatenate lists into single batch tensors. 
        # If Node 5 was activated 3 times in the batch, its ID will appear 3 times in cat_idx!
        if ctx.gradient_sink is not None and all_idx:
            cat_idx = torch.cat(all_idx)
            cat_pg = torch.cat(all_pg)
            cat_mg = torch.cat(all_mg)
            ctx.gradient_sink.add(cat_idx, cat_pg, cat_mg)

        grad_input = torch.stack(by_i)
        grad_input = grad_input.to(dtype=ctx.x.dtype, device=grad_output.device)
        return grad_input, None, None, None, None, None



class _StackLayerView(nn.Module):
    """
    View of one GNN layer in a DistributedNeurographStack.
    forward(x, context=None): context is optional and ignored in the RPC path; reserved for cross-layer use.
    """

    def __init__(self, layer_id, worker_names, gradient_queue, num_workers):
        super().__init__()
        self._layer_id = layer_id
        self._worker_names = worker_names
        self._gradient_queue = gradient_queue
        self._num_workers = num_workers

    def forward(self, x, context=None):
        # context accepted for API compatibility; RPC path does not use it yet
        return _DistributedNeurographFunction.apply(
            x, self._worker_names, self._gradient_queue, self._num_workers, self._layer_id
        )


class DistributedNeurographStack(nn.Module):
    """
    Stack of one or more distributed GNN layers sharing one RPC world and one worker pool.
    Single-layer: DistributedNeurographStack([cfg]) then stack.layers[0].

    - configs: list of config dicts or paths (one per GNN layer).
    - stack.layers[i](x, context=None): run i-th GNN layer.
    - stack.forward(x, context=None): run all layers in order and update context (outputs, layer_index).
    - stack.shutdown(): stop accumulators, RPC, and workers.
    """

    def __init__(self, configs):
        super().__init__()
        configs = _normalize_config_list(configs)
        if not configs:
            raise ValueError("DistributedNeurographStack requires at least one config.")
        self._configs = configs
        self._num_layers = len(configs)
        worker_count = configs[0]["training"].get("worker_count", 1)
        for i, c in enumerate(configs):
            wc = c["training"].get("worker_count", 1)
            if wc != worker_count:
                raise ValueError(
                    f"All configs must use the same training.worker_count; config[{i}] has {wc}, config[0] has {worker_count}."
                )
        self._num_workers = worker_count
        self._world_size = 1 + worker_count

        if rpc_init_rpc is None or rpc is None:
            raise RuntimeError(
                "PyTorch distributed RPC is not available. Install PyTorch with RPC support."
            )
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")

        self._gradient_queues = []
        self._accumulator_processes = []
        for i, cfg in enumerate(configs):
            q = mp.Queue()
            self._gradient_queues.append(q)
            save_path = None
            try:
                log_path = cfg["system"]["logging"]["log_path"]
                save_path = get_weights_save_path(log_path, cfg["qdrant"]["collection_name"])
            except (KeyError, Exception):
                pass
            proc = mp.Process(
                target=gradient_accumulator_process_fn,
                args=(q, cfg, None, None, save_path),
                name=f"accumulator_layer_{i}",
            )
            proc.start()
            self._accumulator_processes.append(proc)
        time.sleep(2)

        self._worker_processes = []
        for k in range(self._num_workers):
            rank = 1 + k
            p = mp.Process(
                target=worker_mod.run_gnn_worker_multilayer,
                args=(rank, self._world_size, configs),
                name=f"worker_{rank}",
            )
            p.start()
            self._worker_processes.append(p)

        rpc_init_rpc("master", rank=0, world_size=self._world_size)
        self._worker_names = [f"worker_{1 + k}" for k in range(self._num_workers)]

        self.layers = nn.ModuleList([
            _StackLayerView(i, self._worker_names, self._gradient_queues[i], self._num_workers)
            for i in range(self._num_layers)
        ])

    def forward(self, x, context=None):
        """
        Run all GNN layers in order. If context is None, a new dict is created.
        Updates context["layer_index"] and context.setdefault("outputs", []) with each layer output (detached).
        Returns the final layer output.
        """
        if context is None:
            context = {}
        for i in range(self._num_layers):
            context["layer_index"] = i
            out = self.layers[i](x)
            context.setdefault("outputs", []).append(out.detach())
            x = out
        return x

    def shutdown(self):
        """Put None into each gradient queue, join accumulators, shutdown RPC, join workers."""
        for q in self._gradient_queues:
            if q is not None:
                q.put(None)
        for proc in self._accumulator_processes:
            if proc is not None and proc.is_alive():
                proc.join(timeout=120)
        try:
            if rpc_shutdown is not None:
                rpc_shutdown()
        except Exception:
            pass
        for p in self._worker_processes:
            if p.is_alive():
                p.terminate()
            p.join(timeout=5)
        self._worker_processes.clear()
        self._gradient_queues = []
        self._accumulator_processes = []


class DistributedNeurographLayer(nn.Module):
    """
    Single distributed GNN layer: one process per sample.
    No RPC; each sample runs in its own process so active_nodes never needs clearing.
    Collates outputs after all sample processes finish, then in backward sends grads
    and collates (phase_grads, mag_grads, input_grad) from all.
    Owns gradient_sink, node_store, and accumulator; GNNAdam discovers these from the model.
    """

    def __init__(self, config):
        super().__init__()
        cfg = get_config(path=config) if isinstance(config, str) else get_config(overrides=config)
        self._config = cfg
        self._gradient_sink = GNNGradientSink()
        self._node_store = get_node_store_from_config(cfg)
        device = cfg["system"]["device"]
        self._accumulator = UnquantizedGradientAccumulator(
            node_store=self._node_store,
            lr=cfg["training"]["lr"],
            accumulation_steps=cfg["training"]["accumulation_steps"],
            betas=tuple(cfg["training"].get("betas", (0.9, 0.999))),
            eps=cfg["training"].get("eps", 1e-8),
            verbose=cfg["system"]["logging"].get("verbose", False),
            device=device,
            save_path=cfg.get("system", {}).get("weights_save_path"),
            save_interval=cfg.get("system", {}).get("weights_save_interval"),
        )
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        # Workers run GNN on CPU so we don't put num_workers copies of the model on GPU (OOM).
        # Main process keeps config device (e.g. cuda) for encoder and accumulator.
        worker_config = copy.deepcopy(cfg)
        worker_config.setdefault("system", {})
        worker_config["system"] = dict(worker_config["system"])
        worker_config["system"]["device"] = "cpu"
        self._pool = WorkerPool(worker_config)
        self._scattering_prob_base = self._config.get("model", {}).get("scattering_prob", 0.0)
        self._current_scattering_prob = None
        self._stochastic_radiation_duration = self._config.get("model", {}).get("stochastic_radiation_duration", 0.5)

        #Called automatically when the program exits
        atexit.register(self.shutdown)

    def set_training_progress(self, epoch: int, total_epochs: int) -> None:
        duration = self._stochastic_radiation_duration
        if self._scattering_prob_base == 0:
            self._current_scattering_prob = 0.0
            return
        if epoch >= total_epochs / 2:
            self._current_scattering_prob = 0.0
        else:
            self._current_scattering_prob = self._scattering_prob_base * max(
                0.0, 1.0 -  epoch / (total_epochs * duration)
            )

    @property
    def gradient_sink(self):
        """Gradient sink for this layer; GNNAdam discovers this from the model."""
        return self._gradient_sink

    @property
    def accumulator(self):
        """Gradient accumulator for this layer; GNNAdam discovers this from the model."""
        return self._accumulator

    def forward(self, x):
        x = torch.tanh(x) * torch.pi
        return _PooledWorkerFunction.apply(
            x, self._config, self._node_store, self._gradient_sink, self._pool, self
        )

    def shutdown(self):
        """Shut down the worker pool. Call when done training (e.g. process exit)."""
        if getattr(self, "_pool", None) is not None:
            self._pool.shutdown()
            self._pool = None
