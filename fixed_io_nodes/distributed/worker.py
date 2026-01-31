# Worker and RPC-invokable functions for DistributedNeurographStack.
# Process-local state: _worker_models (one per layer), _batch_inputs[layer_id].

from ._imports import *  # noqa: F401, F403
from main import get_qdrant_params
from core import initialize_model_and_nodestore

import torch

_worker_models = []  # list of models, one per layer
_batch_inputs = {}   # layer_id -> list of inputs for that layer
_batch_outputs = {}  # layer_id -> list of output tensors (on device) for local backward


def _get_model(layer_id: int):
    if not _worker_models or layer_id < 0 or layer_id >= len(_worker_models):
        raise RuntimeError(
            "Worker models not set or invalid layer_id; run_gnn_worker_multilayer/distributed must be called in this process."
            "must be called in this process."
        )
    return _worker_models[layer_id]


def _ensure_batch_inputs(layer_id: int):
    if layer_id not in _batch_inputs:
        _batch_inputs[layer_id] = []
    return _batch_inputs[layer_id]


def _ensure_batch_outputs(layer_id: int):
    if layer_id not in _batch_outputs:
        _batch_outputs[layer_id] = []
    return _batch_outputs[layer_id]


def gnn_forward_distributed(h1: torch.Tensor, layer_id: int = 0) -> torch.Tensor:
    """RPC callable. Runs model forward for the given layer_id; stores input and output for backward."""
    model = _get_model(layer_id)
    device = next(model.parameters()).device
    h1 = h1.to(device)
    if not h1.requires_grad:
        h1 = h1.requires_grad_(True)
    buf_out = _ensure_batch_outputs(layer_id)
    if len(buf_out) == 0:
        model.reset()
    _ensure_batch_inputs(layer_id).append(h1)
    model.gnn.sync_weights()
    out = model(h1)
    buf_out.append(out)
    return out.cpu()


def run_backward_distributed(grad_list: list, layer_id: int = 0):
    """RPC callable. Runs local backward with grad_list on stored outputs for layer_id; then clears outputs."""
    model = _get_model(layer_id)
    device = next(model.parameters()).device
    outputs = _ensure_batch_outputs(layer_id)
    if len(outputs) != len(grad_list):
        raise RuntimeError(f"run_backward_distributed: layer {layer_id} has {len(outputs)} outputs but got {len(grad_list)} grads")
    for i, (out, g) in enumerate(zip(outputs, grad_list)):
        g_dev = g.to(device) if isinstance(g, torch.Tensor) else torch.tensor(g, device=device)
        out.backward(g_dev, retain_graph=(i < len(outputs) - 1))
    _batch_outputs[layer_id] = []


def gnn_get_grads_distributed(layer_id: int = 0):
    """RPC callable. Returns (phase_grads, mag_grads) from model.gnn.get_grads() for the given layer_id, CPU dicts."""
    model = _get_model(layer_id)
    phase_grads, mag_grads = model.gnn.get_grads()
    to_cpu = lambda d: {k: v.detach().cpu().clone() if v is not None else None for k, v in d.items()}
    return to_cpu(phase_grads), to_cpu(mag_grads)


def get_input_grads_distributed(layer_id: int = 0):
    """RPC callable. Returns list of gradients w.r.t. each input stored this batch for layer_id; then clears buffer."""
    buf = _ensure_batch_inputs(layer_id)
    out = []
    for inp in buf:
        if inp.grad is not None:
            out.append(inp.grad.detach().cpu().clone())
        else:
            out.append(torch.zeros_like(inp, device="cpu"))
    _batch_inputs[layer_id] = []
    return out


def run_gnn_worker_multilayer(rank: int, world_size: int, config_list: list):
    """
    Process entry for each RPC worker used by DistributedNeurographStack.
    config_list: list of config dicts (one per GNN layer). Builds one model per config.
    """
    global _worker_models, _batch_inputs, _batch_outputs
    import torch.distributed.rpc as rpc

    _worker_models = []
    _batch_inputs = {}
    _batch_outputs = {}

    for cfg in config_list:
        qdrant_params = get_qdrant_params(cfg)
        model, _ = initialize_model_and_nodestore(
            qdrant_url=cfg["qdrant"]["url"],
            collection_name=cfg["qdrant"]["collection_name"],
            total_nodes=cfg["graph"]["total_nodes"],
            input_nodes=cfg["graph"]["input_nodes"],
            output_nodes=cfg["graph"]["output_nodes"],
            cardinality=cfg["graph"]["cardinality"],
            radiation_targets=cfg["graph"]["radiation_targets"],
            vector_dim=cfg["model"]["vector_dim"],
            phase_bins=cfg["model"]["phase_bins"],
            mag_bins=cfg["model"]["mag_bins"],
            iterations=cfg["model"]["iterations"],
            activation_threshold=cfg["model"]["activation_threshold"],
            gamma=cfg["model"]["gamma"],
            device=cfg["system"]["device"],
            temporal_decay=cfg["model"].get("temporal_decay", 1.0),
            radiation_similarity_threshold=cfg["model"].get("radiation_similarity_threshold", 0.0),
            qdrant_params=qdrant_params or {},
            verbose=cfg["system"].get("logging", {}).get("verbose", False)
        )
        model = model.to(cfg["system"]["device"])
        _worker_models.append(model)

    name = f"worker_{rank}"
    rpc.init_rpc(name, rank=rank, world_size=world_size)
    torch.distributed.rpc.shutdown()


def run_gnn_worker_distributed(rank: int, world_size: int, config: dict):
    """Process entry for each RPC worker used by DistributedNeurographLayer (single-layer)."""
    run_gnn_worker_multilayer(rank, world_size, [config])


def run_gnn_worker(rank: int, world_size: int, config_list: list):
    """Alias for run_gnn_worker_multilayer; accepts single config or list for backward compat."""
    if isinstance(config_list, dict):
        config_list = [config_list]
    run_gnn_worker_multilayer(rank, world_size, config_list)


def _load_model_for_config(config):
    model, _ = initialize_model_and_nodestore(
        qdrant_url=config["qdrant"]["url"],
        collection_name=config["qdrant"]["collection_name"],
        total_nodes=config["graph"]["total_nodes"],
        input_nodes=config["graph"]["input_nodes"],
        output_nodes=config["graph"]["output_nodes"],
        cardinality=config["graph"]["cardinality"],
        radiation_targets=config["graph"]["radiation_targets"],
        vector_dim=config["model"]["vector_dim"],
        phase_bins=config["model"]["phase_bins"],
        mag_bins=config["model"]["mag_bins"],
        iterations=config["model"]["iterations"],
        activation_threshold=config["model"]["activation_threshold"],
        gamma=config["model"]["gamma"],
        device=config["system"]["device"],
        temporal_decay=config["model"].get("temporal_decay", 1.0),
        radiation_similarity_threshold=config["model"].get("radiation_similarity_threshold", 0.0),
        qdrant_params=get_qdrant_params(config) or {},
        verbose=config["system"].get("logging", {}).get("verbose", False)
    )
    device = config["system"]["device"]
    if isinstance(device, str) and "cuda" in device:
        model = model.to(device)
    return model


def run_one_sample_forward_only(sample_idx, x_i, config, output_queue):
    """Process target: load model, forward one sample, put (sample_idx, out_np), exit. One process at a time."""
    try:
        model = _load_model_for_config(config)
        model.gnn.sync_weights()
        device = next(model.parameters()).device
        x_i = x_i.to(device)
        if not x_i.requires_grad:
            x_i = x_i.requires_grad_(True)
        out = model(x_i)
        out_np = out.cpu().detach().numpy().copy()
        output_queue.put((sample_idx, out_np, None))
    except Exception as e:
        output_queue.put((sample_idx, None, e))


def run_one_sample_backward_only(sample_idx, x_i, config, grad_i, output_queue):
    """Process target: load model, forward then backward one sample; put numpy/simple types to avoid shared-mem unpickle."""
    try:
        model = _load_model_for_config(config)
        model.gnn.sync_weights()
        device = next(model.parameters()).device
        x_i = x_i.to(device).detach().requires_grad_(True)
        out = model(x_i)
        grad_i = grad_i.to(device)
        out.backward(grad_i)
        phase_grads, mag_grads = model.gnn.get_grads()
        to_cpu = lambda d: {k: v.detach().cpu().clone() if v is not None else None for k, v in d.items()}
        pg = to_cpu(phase_grads)
        mg = to_cpu(mag_grads)
        if x_i.grad is None:
            ig = torch.zeros_like(x_i, device="cpu")
            import sys
            print(f"[worker back sample {sample_idx}] x_i.grad is None → sending zeros", file=sys.stderr)
        else:
            ig = x_i.grad.detach().cpu().clone()
            if ig.abs().sum().item() == 0:
                print(f"[worker back sample {sample_idx}] x_i.grad is all zeros", file=sys.stderr)
        pg_np = {k: v.numpy().copy() if v is not None else None for k, v in pg.items()}
        mg_np = {k: v.numpy().copy() if v is not None else None for k, v in mg.items()}
        ig_np = ig.numpy().copy()
        output_queue.put((sample_idx, pg_np, mg_np, ig_np, None))
    except Exception as e:
        output_queue.put((sample_idx, None, None, None, e))
