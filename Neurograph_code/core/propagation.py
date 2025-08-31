# core/propagation.py

import torch
from core.modular_cell import ModularPhaseCell
from core.node_store import NodeStore
from core.radiation import get_radiation_neighbors

def propagate_step(
    active_nodes: dict,
    node_store: NodeStore,
    phase_cell: ModularPhaseCell,
    graph_df,
    lookup_table,
    use_radiation: bool,
    top_k_neighbors: int,
    radiation_batch_size: int,
    phase_bins: int,
    device='cpu'
):
    """
    One timestep of hybrid forward propagation (static + radiation).

    Args:
        active_nodes (dict): node_id → (context_phase_idx, context_mag_idx)
        node_store (NodeStore): Provides self phase/mag index vectors
        phase_cell (PhaseCell): Lookup + signal unit
        graph_df (pd.DataFrame): Static topology
        lookup_table: LookupTableModule with cos/exp
        use_radiation (bool): Whether to apply dynamic phase-based propagation
        top_k_neighbors (int): Number of dynamic neighbors
        device: CPU or CUDA

    Returns:
        updates: List of (target_node_id, new_phase_idx, new_mag_idx, activation_strength)
    """
    updates = []

    for source_node, (ctx_phase_idx, ctx_mag_idx) in active_nodes.items():
        ctx_phase_idx = ctx_phase_idx.to(device)
        ctx_mag_idx = ctx_mag_idx.to(device)

        # --- Static conduction ---
        static_targets = graph_df.loc[
            graph_df["node_id"] == source_node, "input_connections"
        ].values[0]

        all_targets = list(static_targets)

        # --- Dynamic radiation ---
        if use_radiation:
            dynamic_targets = get_radiation_neighbors(
                current_node_id=source_node,
                ctx_phase_idx=ctx_phase_idx,
                node_store=node_store,
                graph_df=graph_df,
                lookup_table=lookup_table,
                top_k=top_k_neighbors,
                phase_bins=phase_bins,
                batch_size=radiation_batch_size
            )
            all_targets += dynamic_targets

        # --- Propagate to all targets ---
        for target_node in all_targets:
            self_phase_idx = node_store.get_phase(target_node).to(device)
            self_mag_idx = node_store.get_mag(target_node).to(device)

            phase_out, mag_out, signal, strength, grad_phase, grad_mag = phase_cell(
                ctx_phase_idx, ctx_mag_idx, self_phase_idx, self_mag_idx
            )

            updates.append((target_node, phase_out, mag_out, strength.item()))

    return updates
