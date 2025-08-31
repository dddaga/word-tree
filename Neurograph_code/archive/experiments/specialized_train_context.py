# train/specialized_train_context.py

import torch
import random
import os
from core.tables import ExtendedLookupTableModule
from core.graph import load_or_build_graph
from core.node_store import NodeStore
from core.cell import PhaseCell
from core.forward_engine import run_enhanced_forward
from core.backward import backward_pass
from utils.config import load_config

from modules.input_adapters import MNISTPCAAdapter
from modules.class_encoding import generate_digit_class_encodings
from modules.loss import signal_loss_from_lookup
from utils.activation_tracker import ActivationFrequencyTracker
from utils.activation_balancer import ActivationBalancer, MultiOutputLossStrategy


def specialized_train_context(config_path: str = "config/optimized.yaml"):
    """
    SPECIALIZED training context where each output node learns to recognize specific digits.
    This should resolve the issue where all nodes learn the same target.
    
    Key improvements:
    1. Each output node is assigned to specific digit classes
    2. Only the assigned node gets trained for each sample
    3. Prediction uses the most confident specialized node
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Tuple of (loss_log, activation_tracker, activation_balancer, node_specializations)
    """
    # Load config
    cfg = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🔧 Using SPECIALIZED training configuration: {config_path}")
    print(f"📊 Graph size: {cfg['total_nodes']} nodes ({cfg['num_input_nodes']} input, {cfg['num_output_nodes']} output)")

    # Load graph
    graph_keys = [
        "total_nodes", "num_input_nodes", "num_output_nodes",
        "vector_dim", "phase_bins", "mag_bins", "cardinality", "seed"
    ]
    graph_config = {k: cfg[k] for k in graph_keys}
    graph_df = load_or_build_graph(**graph_config, overwrite=False, 
                                   save_path=cfg.get("graph_path", "config/static_graph.pkl"))

    # Initialize graph components
    node_store = NodeStore(graph_df, cfg["vector_dim"], cfg["phase_bins"], cfg["mag_bins"])
    lookup = ExtendedLookupTableModule(cfg["phase_bins"], cfg["mag_bins"], device=device)
    phase_cell = PhaseCell(cfg["vector_dim"], lookup)

    # Setup training components
    output_nodes = list(node_store.output_nodes)  # Convert to list for indexing
    input_nodes = list(node_store.input_nodes)
    warmup_epochs = cfg.get("warmup_epochs", 20)
    learning_rate = cfg["learning_rate"]
    num_epochs = cfg.get("num_epochs", 80)
    samples_per_epoch = cfg.get("batch_size", 5)
    
    print(f"🎯 Output nodes: {len(output_nodes)} - {output_nodes}")
    print(f"📥 Input nodes: {len(input_nodes)} - {input_nodes}")
    
    # CRITICAL IMPROVEMENT: Assign each output node to specific digit classes
    node_specializations = {}
    digits_per_node = 10 // len(output_nodes)  # How many digits each node handles
    extra_digits = 10 % len(output_nodes)      # Remaining digits for first few nodes
    
    digit_idx = 0
    for i, node_id in enumerate(output_nodes):
        # First few nodes get one extra digit if there are remainders
        num_digits = digits_per_node + (1 if i < extra_digits else 0)
        assigned_digits = list(range(digit_idx, digit_idx + num_digits))
        node_specializations[node_id] = assigned_digits
        digit_idx += num_digits
        
        print(f"🎯 Node {node_id} specializes in digits: {assigned_digits}")
    
    print(f"🔄 Training mode: SPECIALIZED (samples per epoch: {samples_per_epoch})")

    # Setup MNIST + PCA injector
    adapter = MNISTPCAAdapter(
        vector_dim=cfg["vector_dim"],
        num_input_nodes=len(input_nodes),
        phase_bins=cfg["phase_bins"],
        mag_bins=cfg["mag_bins"],
        device=device
    )

    # Setup digit class encodings
    class_phase_encodings, class_mag_encodings = generate_digit_class_encodings(
        num_classes=10,
        vector_dim=cfg["vector_dim"],
        phase_bins=cfg["phase_bins"],
        mag_bins=cfg["mag_bins"],
        seed=cfg.get("seed", 42)
    )   

    # Initialize activation frequency tracker
    activation_tracker = ActivationFrequencyTracker(
        output_nodes=set(output_nodes),
        num_classes=10
    )

    # Initialize activation balancer if enabled
    activation_balancer = None
    if cfg.get("enable_activation_balancing", False):
        activation_balancer = ActivationBalancer(
            output_nodes=set(output_nodes),
            strategy=cfg.get("balancing_strategy", "quota"),
            max_activations_per_epoch=cfg.get("max_activations_per_epoch", 20),
            min_activations_per_epoch=cfg.get("min_activations_per_epoch", 2),
            force_activation_probability=cfg.get("force_activation_probability", 0.3)
        )
        print(f"⚖️  Activation balancing enabled: {cfg.get('balancing_strategy', 'quota')}")

    # Initialize multi-output loss strategy if enabled
    multi_output_strategy = None
    if cfg.get("enable_multi_output_loss", False):
        multi_output_strategy = MultiOutputLossStrategy(
            continue_timesteps=cfg.get("continue_timesteps_after_first", 2),
            max_outputs_to_train=cfg.get("max_outputs_to_train", 3)
        )
        print(f"🔄 Multi-output loss enabled: continue {cfg.get('continue_timesteps_after_first', 2)} timesteps")

    # Create log directory
    log_path = cfg.get("log_path", "logs/")
    os.makedirs(log_path, exist_ok=True)

    loss_log = []

    print(f"\n🚀 Starting SPECIALIZED training for {num_epochs} epochs...")
    print(f"🔥 Warmup period: {warmup_epochs} epochs")
    print(f"📈 Samples per epoch: {samples_per_epoch}")

    for epoch in range(num_epochs):
        include_all_outputs = epoch < warmup_epochs
        epoch_loss = 0.0
        epoch_samples = 0

        # Process samples individually with specialized training
        for sample_idx in range(samples_per_epoch):
            # Get single sample
            mnist_idx = random.randint(0, len(adapter.mnist) - 1)
            input_context, label = adapter.get_input_context(mnist_idx, input_nodes)

            # Run forward pass on single sample
            activation = run_enhanced_forward(
                graph_df=graph_df,
                node_store=node_store,
                phase_cell=phase_cell,
                lookup_table=lookup,
                input_context=input_context,
                vector_dim=cfg["vector_dim"],
                phase_bins=cfg["phase_bins"],
                mag_bins=cfg["mag_bins"],
                decay_factor=cfg["decay_factor"],
                min_strength=cfg["min_activation_strength"],
                max_timesteps=cfg["max_timesteps"],
                use_radiation=cfg["use_radiation"],
                top_k_neighbors=cfg["top_k_neighbors"],
                min_output_activation_timesteps=cfg.get("min_output_activation_timesteps", 2),
                device=device,
                verbose=(epoch % 20 == 0 and sample_idx == 0),
                activation_tracker=activation_tracker,
                activation_balancer=activation_balancer,
                multi_output_strategy=multi_output_strategy
            )
            
            # Record activation data
            active_outputs = list(activation.get_active_context().keys())
            active_output_nodes = [n for n in active_outputs if n in output_nodes]
            
            activation_tracker.record_forward_pass(
                active_outputs=active_output_nodes,
                activation_timesteps={},
                first_active_output=None,
                true_label=label
            )

            # CRITICAL IMPROVEMENT: Only train specialized nodes for this digit
            if include_all_outputs:
                # During warmup, train all nodes (original behavior)
                target_nodes = output_nodes
            else:
                # After warmup, only train nodes specialized for this digit
                target_nodes = []
                for node_id, assigned_digits in node_specializations.items():
                    if label in assigned_digits:
                        target_nodes.append(node_id)
                
                if not target_nodes:
                    # Fallback: if no specialized node, use all (shouldn't happen with proper assignment)
                    target_nodes = output_nodes

            # Create target context only for specialized nodes
            target_context = {}
            for node_id in target_nodes:
                target_context[node_id] = (
                    class_phase_encodings[label].clone(), 
                    class_mag_encodings[label].clone()
                )

            # Compute loss only for specialized nodes
            sample_loss = 0.0
            sample_counted_nodes = 0

            for node_id in target_nodes:
                if not include_all_outputs and node_id not in active_outputs:
                    continue

                pred_phase, pred_mag = activation.table.get(node_id, (None, None, None))[:2]
                if pred_phase is None:
                    continue

                tgt_phase, tgt_mag = target_context[node_id]

                pred_phase = pred_phase.to(torch.float32)
                tgt_phase = tgt_phase.to(torch.float32)
                pred_mag = pred_mag.to(torch.float32)
                tgt_mag = tgt_mag.to(torch.float32)

                loss = signal_loss_from_lookup(
                    pred_phase, pred_mag,
                    tgt_phase, tgt_mag,
                    lookup
                )
                sample_loss += loss.item()
                sample_counted_nodes += 1

            if sample_counted_nodes > 0:
                sample_loss /= sample_counted_nodes
                epoch_loss += sample_loss
                epoch_samples += 1

            # CRITICAL IMPROVEMENT: Backward pass only updates specialized nodes
            backward_pass(
                activation_table=activation,
                node_store=node_store,
                phase_cell=phase_cell,
                lookup_table=lookup,
                target_context=target_context,  # Only specialized nodes
                output_nodes=set(target_nodes),  # Only update specialized nodes
                learning_rate=learning_rate,
                include_all_outputs=include_all_outputs,
                verbose=(epoch % 20 == 0 and sample_idx == 0),
                device=device
            )

        # Average loss over epoch
        if epoch_samples > 0:
            epoch_loss /= epoch_samples
        loss_log.append(epoch_loss)

        print(f"[Epoch {epoch+1}] Avg Loss: {epoch_loss:.4f} | Samples: {epoch_samples}")
        
        # End epoch tracking
        activation_tracker.end_epoch()
        if activation_balancer is not None:
            activation_balancer.end_epoch()
        
        # Print periodic diagnostic reports
        if (epoch + 1) % 20 == 0:
            print(f"\n📊 Activation Summary after Epoch {epoch + 1}:")
            summary = activation_tracker.get_activation_summary()
            if "error" not in summary:
                print(f"   Active Output Nodes: {summary['unique_active_nodes']}/{len(output_nodes)}")
                print(f"   Dead Nodes: {summary['dead_nodes']}")
                print(f"   Dominant Nodes: {summary['dominant_nodes']}")
                
            if activation_balancer is not None:
                balance_summary = activation_balancer.get_balance_summary()
                if "error" not in balance_summary:
                    print(f"   Forced Activations: {balance_summary['forced_percentage']:.1f}%")
                    print(f"   Underutilized: {balance_summary['underutilized_nodes']}")
                    print(f"   Overutilized: {balance_summary['overutilized_nodes']}")

    # Final diagnostic reports
    print("\n" + "="*80)
    print("🔍 FINAL SPECIALIZED TRAINING ANALYSIS")
    print("="*80)
    
    activation_tracker.print_diagnostic_report()
    
    if activation_balancer is not None:
        activation_balancer.print_balance_report()
    
    # Print specialization summary
    print("\n" + "="*60)
    print("🎯 NODE SPECIALIZATION SUMMARY")
    print("="*60)
    for node_id, assigned_digits in node_specializations.items():
        print(f"   {node_id}: digits {assigned_digits}")
    
    # Generate plots
    try:
        activation_tracker.plot_activation_frequency(save_path=os.path.join(log_path, "specialized_activation_frequency.png"))
        activation_tracker.plot_epoch_evolution(save_path=os.path.join(log_path, "specialized_epoch_evolution.png"))
        print(f"📊 Plots saved to {log_path}")
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")

    return loss_log, activation_tracker, activation_balancer, node_specializations
