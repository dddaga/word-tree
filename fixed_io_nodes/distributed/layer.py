# DistributedNeurographStack and custom autograd Function.
# Uses RPC for forward and backward; workers run local backward when sent gradient slices.
# Per-layer accumulators consume phase/mag grads from workers.

from ._imports import *  # noqa: F401, F403
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

import time
from collections import defaultdict

from . import worker as worker_mod


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


class _OneProcessPerSampleFunction(torch.autograd.Function):
    """
    One process per sample; forward and backward each start all B processes in parallel,
    then collect results and join. active_nodes never needs clearing per process.
    Gradients reach upstream: we detach only the copy sent to the worker; backward
    returns grad_input = d(loss)/d(x) from workers, so autograd assigns it to x.grad
    and continues back through the layers that produced x.
    """

    @staticmethod
    def forward(ctx, x, config, gradient_queue):
        B = x.shape[0]
        output_queue = mp.Queue()
        procs = []
        for i in range(B):
            xi = x[i].detach().cpu().requires_grad_(True)
            p = mp.Process(
                target=worker_mod.run_one_sample_forward_only,
                args=(i, xi, config, output_queue),
                name=f"sample_{i}",
            )
            p.start()
            procs.append(p)
        by_i = {}
        for _ in range(B):
            sidx, out_np, err = output_queue.get()
            if err is not None:
                for p in procs:
                    p.join(timeout=5)
                raise RuntimeError(f"Sample {sidx} forward failed") from err
            by_i[sidx] = torch.from_numpy(out_np.copy())
        for p in procs:
            p.join(timeout=120)
        stack_out = torch.stack([by_i[i] for i in range(B)])
        ctx.x = x
        ctx.config = config
        ctx.B = B
        ctx.gradient_queue = gradient_queue
        return stack_out.to(x.device)

    @staticmethod
    def backward(ctx, grad_output):
        co = grad_output.cpu()
        if co.shape[0] != ctx.B:
            co = co.expand(ctx.B, *co.shape[1:]).contiguous()
        x, config = ctx.x, ctx.config
        output_queue = mp.Queue()
        by_i = {}
        def np_to_tensor_dict(d):
            if not d:
                return d
            return {k: torch.from_numpy(v.copy()) for k, v in d.items() if v is not None}

        procs = []
        for i in range(ctx.B):
            xi = x[i].detach().cpu().requires_grad_(True)
            p = mp.Process(
                target=worker_mod.run_one_sample_backward_only,
                args=(i, xi, config, co[i], output_queue),
                name=f"back_{i}",
            )
            p.start()
            procs.append(p)
        for _ in range(ctx.B):
            sidx, pg_np, mg_np, ig_np, err = output_queue.get()
            if err is not None:
                for p in procs:
                    p.join(timeout=5)
                raise RuntimeError(f"Sample {sidx} backward failed") from err
            pg = np_to_tensor_dict(pg_np or {})
            mg = np_to_tensor_dict(mg_np or {})
            ig = torch.from_numpy(ig_np.copy())
            by_i[sidx] = (pg, mg, ig)
        for p in procs:
            p.join(timeout=120)
        phase_acc = defaultdict(lambda: None)
        mag_acc = defaultdict(lambda: None)
        for i in range(ctx.B):
            pg, mg, _ = by_i[i]
            for k, v in (pg or {}).items():
                if v is not None:
                    phase_acc[k] = v if phase_acc[k] is None else phase_acc[k] + v
            for k, v in (mg or {}).items():
                if v is not None:
                    mag_acc[k] = v if mag_acc[k] is None else mag_acc[k] + v
        ctx.gradient_queue.put((dict(phase_acc), dict(mag_acc)))
        grad_input = torch.stack([by_i[i][2] for i in range(ctx.B)])
        # ensure same dtype/device as ctx.x so autograd assigns to x.grad correctly
        grad_input = grad_input.to(dtype=ctx.x.dtype, device=grad_output.device)
        return grad_input, None, None


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
    Single distributed GNN layer: one process per sample, one accumulator.
    No RPC; each sample runs in its own process so active_nodes never needs clearing.
    Collates outputs after all sample processes finish, then in backward sends grads
    and collates (phase_grads, mag_grads, input_grad) from all.
    """

    def __init__(self, config):
        super().__init__()
        cfg = load_config(config) if isinstance(config, str) else config
        self._config = cfg
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        self._gradient_queue = mp.Queue()
        save_path = None
        try:
            log_path = cfg["system"]["logging"]["log_path"]
            save_path = get_weights_save_path(log_path, cfg["qdrant"]["collection_name"])
        except (KeyError, Exception):
            pass
        self._accumulator_process = mp.Process(
            target=gradient_accumulator_process_fn,
            args=(self._gradient_queue, cfg, None, None, save_path),
            name="accumulator",
        )
        self._accumulator_process.start()
        time.sleep(2)

    def forward(self, x):
        return _OneProcessPerSampleFunction.apply(x, self._config, self._gradient_queue)

    def shutdown(self):
        if getattr(self, "_gradient_queue", None) is not None:
            self._gradient_queue.put(None)
        if getattr(self, "_accumulator_process", None) is not None and self._accumulator_process.is_alive():
            self._accumulator_process.join(timeout=120)
        self._gradient_queue = None
        self._accumulator_process = None
