"""
Long-lived worker pool for DistributedNeurographLayer.
Workers are started once and reuse model; they receive updated weights each batch.
"""

import torch
import torch.multiprocessing as mp
from typing import Any, Dict, List, Optional, Tuple

from . import worker as worker_mod


class WorkerPool:
    """
    Pool of num_workers long-lived processes. Each worker runs worker_pool_loop,
    builds the model once, then handles weights/forward/backward commands.
    """

    def __init__(self, config: dict, num_workers: Optional[int] = None):
        if num_workers is None:
            num_workers = config["training"].get("accumulation_steps", 4)
        self._num_workers = max(1, num_workers)
        self._config = config
        self._task_queues: List[mp.Queue] = []
        self._result_queues: List[mp.Queue] = []
        self._processes: List[mp.Process] = []
        self._start()

    def _start(self):
        for i in range(self._num_workers):
            tq = mp.Queue()
            rq = mp.Queue()
            self._task_queues.append(tq)
            self._result_queues.append(rq)
            p = mp.Process(
                target=worker_mod.worker_pool_loop,
                args=(i, tq, rq, self._config),
                name=f"pool_worker_{i}",
                daemon=True,
            )
            p.start()
            self._processes.append(p)

    def submit_weights(self, state_dict: dict):
        for i in range(self._num_workers):
            self._task_queues[i].put(("weights", state_dict))

    def submit_forward(self, worker_id: int, x_i: torch.Tensor, scattering_prob=None):
        payload = (x_i, scattering_prob) if scattering_prob is not None else x_i
        self._task_queues[worker_id].put(("forward", payload))

    def get_forward_result(self, worker_id: int) -> torch.Tensor:
        kind, payload = self._result_queues[worker_id].get()
        if kind != "out":
            raise RuntimeError(f"Expected 'out', got {kind}")
        if isinstance(payload, torch.Tensor):
            return payload
        import numpy as np
        if isinstance(payload, np.ndarray):
            return torch.from_numpy(payload.copy())
        return torch.from_numpy(payload.copy())

    def reset_workers(self):
        """Reset GNN state in all workers (e.g. after eval passes that did not run backward)."""
        for i in range(self._num_workers):
            self._task_queues[i].put(("reset", None))
        for i in range(self._num_workers):
            kind, _ = self._result_queues[i].get()
            if kind != "reset_ack":
                raise RuntimeError(f"Worker {i} returned {kind!r}, expected reset_ack")

    def submit_backward(self, worker_id: int, grad_i: torch.Tensor):
        self._task_queues[worker_id].put(("backward", grad_i))

    def get_backward_result(
        self, worker_id: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        kind, payload = self._result_queues[worker_id].get()
        if kind != "grads":
            raise RuntimeError(f"Expected 'grads', got {kind}")
        indices_np, pg_np, mg_np, ig_np = payload
        indices = torch.from_numpy(indices_np) if indices_np is not None else None
        pg = torch.from_numpy(pg_np) if pg_np is not None else None
        mg = torch.from_numpy(mg_np) if mg_np is not None else None
        ig = torch.from_numpy(ig_np.copy()) if ig_np is not None else None
        return indices, pg, mg, ig

    def shutdown(self):
        for i in range(self._num_workers):
            try:
                self._task_queues[i].put(("shutdown",))
            except Exception:
                pass
        for p in self._processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)
        self._processes.clear()
        self._task_queues.clear()
        self._result_queues.clear()

    @property
    def num_workers(self) -> int:
        return self._num_workers
