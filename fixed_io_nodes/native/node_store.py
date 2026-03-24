import random
import numpy as np
import torch
import torch.nn as nn
from typing import List, Set, Tuple, Union

from core.nodestore import SimpleCosineSearch


class NativeNodeStore(nn.Module):
    """
    Single-process node store with nn.Parameters for phase/mag weights.
    No shared memory, no process locks. Graph topology is deterministic from seed.
    """

    def __init__(
        self,
        total_nodes: int,
        input_nodes: int,
        output_nodes: int,
        cardinality: int,
        vector_dim: int,
        seed: int = 42,
        device: str = "cpu",
    ):
        super().__init__()
        self.total_nodes = total_nodes
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.cardinality = cardinality
        self.vector_dim = vector_dim

        self.phase_weight = nn.Parameter(torch.empty(total_nodes, vector_dim))
        self.mag_weight = nn.Parameter(torch.empty(total_nodes, vector_dim))

        self.search_backend = SimpleCosineSearch()
        self.connections = {}

        self._initialize_graph(seed)

    # ------------------------------------------------------------------
    # Graph initialization (mirrors PytorchNodeStore logic)
    # ------------------------------------------------------------------

    def _initialize_graph(self, seed: int):
        rng_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        node_ids = list(range(self.total_nodes))
        self.input_nodeids = set(node_ids[: self.input_nodes])
        self.output_nodeids = set(node_ids[-self.output_nodes :])

        if self.total_nodes - (self.input_nodes + self.output_nodes) < 0.1 * self.total_nodes:
            raise ValueError("Total nodes must be sufficiently larger than input + output nodes")

        # Build connections
        graph = {nid: {"incoming": [], "outgoing": []} for nid in node_ids}
        all_ids = set(node_ids)
        for n in node_ids:
            if n in self.input_nodeids:
                continue
            possible = list(all_ids - self.output_nodeids - {n})
            count = random.randint(1, self.cardinality)
            incoming = random.choices(possible, k=count)
            graph[n]["incoming"] = incoming
            for src in incoming:
                graph[src]["outgoing"].append(n)
        self.connections = graph

        # Initialize weights
        with torch.no_grad():
            for nid in node_ids:
                self.phase_weight.data[nid] = torch.from_numpy(
                    np.random.uniform(0, 2 * np.pi, self.vector_dim).astype(np.float32)
                )
                self.mag_weight.data[nid] = torch.from_numpy(
                    np.random.normal(1.0, 0.1, self.vector_dim).astype(np.float32)
                )

        self._build_edge_indices()

        # Restore RNG state so we don't affect external randomness
        random.setstate(rng_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)

    def _build_edge_indices(self):
        """Build (2, num_edges) static edge index from connections + self-loops."""
        sources, dests = [], []
        for nid in range(self.total_nodes):
            for dest in self.connections.get(nid, {}).get("outgoing", []):
                sources.append(nid)
                dests.append(dest)
            # Self-loop
            sources.append(nid)
            dests.append(nid)
        self.register_buffer(
            "edge_indices", torch.tensor([sources, dests], dtype=torch.long)
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_nodes_batch(
        self,
        query_vectors: torch.Tensor,
        vector_name: str = "phase",
        limit: int = 3,
        with_payload: bool = False,
        with_vectors: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cosine search using current phase_weight values (computed on-the-fly)."""
        if vector_name != "phase":
            raise NotImplementedError(f"Vector name '{vector_name}' not implemented")

        # Compute phase_values from current parameters (detached — search is non-differentiable)
        pw = self.phase_weight.data
        phase_values = torch.cat([torch.cos(pw), torch.sin(pw)], dim=-1)

        # Build query in conjugate form
        if not isinstance(query_vectors, torch.Tensor):
            query_vectors = torch.stack(query_vectors)
        query_batch = torch.cat(
            [torch.cos(query_vectors), -torch.sin(query_vectors)], dim=-1
        )

        return self.search_backend.search_batch(query_batch, phase_values, limit)

    def sample_random_nodeids(self, count: int) -> List[int]:
        if count > self.total_nodes:
            raise ValueError("Count cannot be greater than total nodes")
        return random.sample(range(self.total_nodes), count)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def get_custom_state(self) -> dict:
        return {
            "phase_weight": self.phase_weight.data.cpu().clone(),
            "mag_weight": self.mag_weight.data.cpu().clone(),
            "connections": dict(self.connections),
            "metadata": {
                "total_nodes": self.total_nodes,
                "vector_dim": self.vector_dim,
                "input_nodeids": list(self.input_nodeids),
                "output_nodeids": list(self.output_nodeids),
                "input_nodes": self.input_nodes,
                "output_nodes": self.output_nodes,
                "cardinality": self.cardinality,
            },
        }

    @torch.no_grad()
    def load_custom_state(self, state_dict: dict):
        meta = state_dict["metadata"]
        if meta["total_nodes"] != self.total_nodes:
            raise ValueError(f"Total nodes mismatch: saved={meta['total_nodes']}, current={self.total_nodes}")
        if meta["vector_dim"] != self.vector_dim:
            raise ValueError(f"Vector dim mismatch: saved={meta['vector_dim']}, current={self.vector_dim}")

        self.phase_weight.data.copy_(state_dict["phase_weight"])
        self.mag_weight.data.copy_(state_dict["mag_weight"])
        self.connections = state_dict["connections"]
        self.input_nodeids = set(meta["input_nodeids"])
        self.output_nodeids = set(meta["output_nodeids"])
        self._build_edge_indices()
