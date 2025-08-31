# core/backward.py

import torch

def backward_pass(
    activation_table,
    node_store,
    phase_cell,
    lookup_table,
    target_context,       # dict: node_id → (target_phase_idx, target_mag_idx)
    output_nodes: set,    # e.g., from node_store.output_nodes
    learning_rate: float = 0.01,
    include_all_outputs: bool = False,
    device='cpu',
    verbose=False
):
    """
    Performs manual backward pass using signal gradients.

    Args:
        activation_table: Final ActivationTable after forward pass
        node_store: NodeStore object
        phase_cell: PhaseCell module (returns gradient-aware signal)
        lookup_table: ExtendedLookupTableModule
        target_context: dict of node_id → (target_phase_idx, target_mag_idx)
        output_nodes: set of node_ids that are designated outputs
        learning_rate: step size
        include_all_outputs: include inactive output nodes (for warm-up)
    """
    updated_nodes = 0

    for node_id in output_nodes:
        # Skip if not active unless warm-up
        if not include_all_outputs and not activation_table.is_active(node_id):
            continue
        if node_id not in target_context:
            continue

        # Get predicted context from activation table
        pred_phase, pred_mag = activation_table.table.get(node_id, (None, None, None))[:2]
        if pred_phase is None or pred_mag is None:
            continue  # Skip silently if missing

        # Get target
        tgt_phase, tgt_mag = target_context[node_id]
        tgt_phase = tgt_phase.to(device)
        tgt_mag   = tgt_mag.to(device)

        # Get current node parameters
        self_phase = node_store.get_phase(node_id).to(device)
        self_mag   = node_store.get_mag(node_id).to(device)

        # Run PhaseCell to get gradients
        _, _, _, _, grad_phase, grad_mag = phase_cell(
            ctx_phase_idx=tgt_phase,
            ctx_mag_idx=tgt_mag,
            self_phase_idx=self_phase,
            self_mag_idx=self_mag
        )

        # Update with learning rate, then clamp or wrap
        new_phase = (self_phase.float() - learning_rate * grad_phase).round().long() % lookup_table.N
        new_mag   = (self_mag.float()   - learning_rate * grad_mag).round().long() % lookup_table.M

        # Write back
        node_store.phase_table[node_id].data = new_phase
        node_store.mag_table[node_id].data   = new_mag
        updated_nodes += 1

        if verbose:
            print(f"🔧 Updated node {node_id}: Δphase={grad_phase.mean():.4f}, Δmag={grad_mag.mean():.4f}")

    if verbose:
        print(f"\n🔁 Backward pass completed. Updated {updated_nodes} nodes.")
