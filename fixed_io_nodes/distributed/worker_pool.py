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

    def submit_forward(self, worker_id: int, x_i: torch.Tensor):
        self._task_queues[worker_id].put(("forward", x_i))

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

    def submit_backward(self, worker_id: int, grad_i: torch.Tensor):
        self._task_queues[worker_id].put(("backward", grad_i))

    def get_backward_result(
        self, worker_id: int
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], torch.Tensor]:
        kind, payload = self._result_queues[worker_id].get()
        if kind != "grads":
            raise RuntimeError(f"Expected 'grads', got {kind}")
        pg_np, mg_np, ig_np = payload
        pg = (
            {k: torch.from_numpy(v.copy() if hasattr(v, "copy") else v) for k, v in (pg_np or {}).items() if v is not None}
            if pg_np
            else {}
        )
        mg = (
            {k: torch.from_numpy(v.copy() if hasattr(v, "copy") else v) for k, v in (mg_np or {}).items() if v is not None}
            if mg_np
            else {}
        )
        ig = torch.from_numpy(ig_np.copy()) if ig_np is not None else None
        return pg, mg, ig

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
