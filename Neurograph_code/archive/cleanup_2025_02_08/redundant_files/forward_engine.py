# core/enhanced_forward_engine.py

from typing import List, Tuple, Optional
from core.activation_table import ActivationTable
from core.propagation import propagate_step
from utils.activation_balancer import ActivationBalancer, MultiOutputLossStrategy

def is_output_node_active(active_nodes: List[str], output_nodes: set) -> bool:
    """Check if any output nodes are currently active."""
    return any(node_id in output_nodes for node_id in active_nodes)

def run_enhanced_forward(
    graph_df,
    node_store,
    phase_cell,
    lookup_table,
    input_context: dict,  # node_id → (phase_idx, mag_idx)
    vector_dim: int,
    phase_bins: int,
    mag_bins: int,
    decay_factor: float,
    min_strength: float,
    max_timesteps: int,
    use_radiation: bool = True,
    top_k_neighbors: int = 4,
    min_output_activation_timesteps: int = 2,
    device='cpu',
    verbose=False,
    activation_tracker=None,
    activation_balancer: Optional[ActivationBalancer] = None,
    multi_output_strategy: Optional[MultiOutputLossStrategy] = None
):
    """
    Enhanced event-driven forward pass with activation balancing and multi-output support.
    
    Args:
        graph_df: Static topology DataFrame
        node_store: NodeStore object containing output_nodes
        phase_cell: PhaseCell module (gradient-aware)
        lookup_table: ExtendedLookupTableModule
        input_context: dict of node_id → (phase_idx [D], mag_idx [D])
        vector_dim: Dimensionality of phase/mag vectors
        decay_factor: Scalar decay per step
        min_strength: Minimum activation strength to keep a node active
        max_timesteps: Maximum timesteps (safety limit)
        use_radiation: Enable phase-based dynamic neighbors
        top_k_neighbors: Top-K neighbors for radiation
        min_output_activation_timesteps: Minimum timesteps before checking for output activation
        device: 'cpu' or 'cuda'
        verbose: Verbose logging per timestep
        activation_tracker: Optional activation frequency tracker
        activation_balancer: Optional activation balancer for risk mitigation
        multi_output_strategy: Optional multi-output loss strategy

    Returns:
        ActivationTable: Final activation state
    """
    activation = ActivationTable(
        vector_dim=vector_dim,
        phase_bins=phase_bins,
        mag_bins=mag_bins,
        decay_factor=decay_factor,
        min_strength=min_strength,
        device=device
    )

    # Get output nodes from node_store
    output_nodes = set(node_store.output_nodes)
    
    # Tracking variables for activation analysis
    output_activation_timesteps = {}  # node_id -> first timestep it activated
    first_active_output = None
    first_activation_timestep = None
    timesteps_since_first_activation = 0
    
    # Inject initial input context
    for node_id, (p, m) in input_context.items():
        activation.inject(node_id, p.to(device), m.to(device), strength=1.0)

    timestep = 0
    while timestep < max_timesteps:
        ctx = activation.get_active_context()
        
        # Track output node activations for the first time
        current_active_outputs = []
        for node_id in ctx.keys():
            if node_id in output_nodes:
                current_active_outputs.append(node_id)
                if node_id not in output_activation_timesteps:
                    output_activation_timesteps[node_id] = timestep
                    if first_active_output is None:
                        first_active_output = node_id
                        first_activation_timestep = timestep
        
        # Update timesteps since first activation
        if first_activation_timestep is not None:
            timesteps_since_first_activation = timestep - first_activation_timestep
        
        # Check termination conditions
        should_terminate = False
        
        # Standard early termination check
        if (timestep >= min_output_activation_timesteps and 
            is_output_node_active(list(ctx.keys()), output_nodes)):
            
            # Multi-output strategy: continue for more outputs
            if multi_output_strategy is not None:
                if not multi_output_strategy.should_continue_forward(
                    timesteps_since_first_activation, current_active_outputs):
                    should_terminate = True
            else:
                # Standard behavior: terminate on first activation
                should_terminate = True
        
        if should_terminate:
            if verbose:
                print(f"✅ Output activation detected at timestep {timestep}")
                print(f"🎯 Active output nodes: {current_active_outputs}")
                if multi_output_strategy:
                    print(f"🔄 Multi-output: Trained {len(current_active_outputs)} outputs")
            break
            
        # Check for dead network (no active nodes)
        if not ctx:
            if verbose:
                print(f"💀 Network died at timestep {timestep} - no active nodes")
            break

        if verbose:
            print(f"\n⏱️ Timestep {timestep}")
            print(f"🔹 Active nodes: {list(ctx.keys())}")
            if current_active_outputs:
                print(f"🎯 Active outputs: {current_active_outputs}")
            else:
                print(f"🎯 Waiting for output activation...")

        # Propagation step
        updates = propagate_step(
            active_nodes=ctx,
            node_store=node_store,
            phase_cell=phase_cell,
            graph_df=graph_df,
            lookup_table=lookup_table,
            use_radiation=use_radiation,
            top_k_neighbors=top_k_neighbors,
            device=device
        )

        # Apply activation balancing if enabled
        if activation_balancer is not None:
            # Check if we should force activation of underutilized nodes
            should_force, target_node = activation_balancer.should_force_activation(
                current_active_outputs
            )
            
            if should_force and target_node:
                # Force activation by injecting signal to target node
                # Use random phase/magnitude within valid ranges
                import torch
                forced_phase = torch.randint(0, phase_bins, (vector_dim,), device=device)
                forced_mag = torch.randint(0, mag_bins, (vector_dim,), device=device)
                
                # Add forced activation to updates
                updates.append((target_node, forced_phase, forced_mag, 0.8))  # Strong signal
                
                if verbose:
                    print(f"🔧 Forced activation: {target_node}")
            
            # Filter out suppressed activations
            filtered_updates = []
            for target_node, new_phase, new_mag, strength in updates:
                if (target_node in output_nodes and 
                    activation_balancer.should_suppress_activation(target_node)):
                    if verbose:
                        print(f"🚫 Suppressed activation: {target_node}")
                    continue
                filtered_updates.append((target_node, new_phase, new_mag, strength))
            
            updates = filtered_updates

        # Overwrite style: clear previous table before injecting next wave
        activation.clear()

        for target_node, new_phase, new_mag, strength in updates:
            activation.inject(target_node, new_phase, new_mag, strength)

        activation.decay_and_prune()
        timestep += 1

    # Get final activation context
    final_ctx = activation.get_active_context()
    active_outputs = [n for n in final_ctx.keys() if n in output_nodes]
    
    # Record activations in balancer
    if activation_balancer is not None:
        for node_id in active_outputs:
            activation_balancer.record_activation(node_id, forced=False)
    
    # Pass activation data to tracker if provided
    if activation_tracker is not None:
        activation_tracker.record_forward_pass(
            active_outputs=active_outputs,
            activation_timesteps=output_activation_timesteps,
            first_active_output=first_active_output,
            true_label=None  # Will be set by the training loop
        )
    
    if verbose:
        print(f"\n🏁 Forward pass completed after {timestep} timesteps")
        print(f"🎯 Final active output nodes: {active_outputs}")
        print(f"🔹 Total active nodes: {len(final_ctx)}")

    return activation
