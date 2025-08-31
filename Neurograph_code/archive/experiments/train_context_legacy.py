# train/enhanced_train_context.py

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


def enhanced_train_context(config_path: str = "config/default.yaml"):
    """
    Enhanced training context with activation balancing and multi-output loss support.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Tuple of (loss_log, activation_tracker, activation_balancer)
    """
    # Load config
    cfg = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🔧 Using configuration: {config_path}")
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
    output_nodes = node_store.output_nodes
    input_nodes = list(node_store.input_nodes)
    warmup_epochs = cfg.get("warmup_epochs", 5)
    learning_rate = cfg["learning_rate"]
    num_epochs = cfg.get("num_epochs", 50)
    batch_size = cfg.get("batch_size", 5)

    print(f"🎯 Output nodes: {len(output_nodes)} - {output_nodes}")
    print(f"📥 Input nodes: {len(input_nodes)} - {input_nodes}")

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
        output_nodes=output_nodes,
        num_classes=10
    )

    # Initialize activation balancer if enabled
    activation_balancer = None
    if cfg.get("enable_activation_balancing", False):
        activation_balancer = ActivationBalancer(
            output_nodes=output_nodes,
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

    print(f"\n🚀 Starting enhanced training for {num_epochs} epochs...")
    print(f"🔥 Warmup period: {warmup_epochs} epochs")

    for epoch in range(num_epochs):
        include_all_outputs = epoch < warmup_epochs

        # Batch processing: collect multiple samples
        batch_input_contexts = []
        batch_labels = []
        
        # Sample batch_size examples
        for _ in range(batch_size):
            sample_idx = random.randint(0, len(adapter.mnist) - 1)
            input_context, label = adapter.get_input_context(sample_idx, input_nodes)
            batch_input_contexts.append(input_context)
            batch_labels.append(label)

        # Merge all input contexts into one dictionary (assuming no node conflicts)
        merged_input_context = {}
        for input_context in batch_input_contexts:
            merged_input_context.update(input_context)

        # Run enhanced forward pass
        activation = run_enhanced_forward(
            graph_df=graph_df,
            node_store=node_store,
            phase_cell=phase_cell,
            lookup_table=lookup,
            input_context=merged_input_context,
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
            verbose=(epoch % 10 == 0),
            activation_tracker=activation_tracker,
            activation_balancer=activation_balancer,
            multi_output_strategy=multi_output_strategy
        )
        
        # Record individual forward passes for each sample in the batch
        active_outputs = list(activation.get_active_context().keys())
        active_output_nodes = [n for n in active_outputs if n in output_nodes]
        
        # Record activation data for each sample in the batch
        for batch_idx, label in enumerate(batch_labels):
            activation_tracker.record_forward_pass(
                active_outputs=active_output_nodes,
                activation_timesteps={},  # Filled by enhanced forward engine
                first_active_output=None,  # Filled by enhanced forward engine
                true_label=label
            )

        # Compute loss: reconstruct target_context from individual labels
        total_loss = 0.0
        active_outputs = activation.get_active_context().keys()
        total_counted_nodes = 0
        
        # Average loss over all samples in the batch
        for batch_idx in range(batch_size):
            label = batch_labels[batch_idx]
            class_phase = class_phase_encodings[label]
            class_mag = class_mag_encodings[label]

            # Create target context for this sample
            sample_target_context = {
                node_id: (class_phase.clone(), class_mag.clone())
                for node_id in output_nodes
            }

            # Compute loss for this sample
            sample_loss = 0.0
            sample_counted_nodes = 0

            for node_id in output_nodes:
                if not include_all_outputs and node_id not in active_outputs:
                    continue

                pred_phase, pred_mag = activation.table.get(node_id, (None, None, None))[:2]
                if pred_phase is None:
                    continue

                tgt_phase, tgt_mag = sample_target_context[node_id]

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
                total_loss += sample_loss
                total_counted_nodes += 1

        # Average loss over the batch
        if total_counted_nodes > 0:
            total_loss /= total_counted_nodes
        loss_log.append(total_loss)

        print(f"[Epoch {epoch+1}] Batch Loss: {total_loss:.4f} | Active Outputs: {len(active_output_nodes)}")

        # Backward pass - use the last sample's target context for backward pass
        final_label = batch_labels[-1]
        final_target_context = {
            node_id: (class_phase_encodings[final_label].clone(), class_mag_encodings[final_label].clone())
            for node_id in output_nodes
        }

        backward_pass(
            activation_table=activation,
            node_store=node_store,
            phase_cell=phase_cell,
            lookup_table=lookup,
            target_context=final_target_context,
            output_nodes=output_nodes,
            learning_rate=learning_rate,
            include_all_outputs=include_all_outputs,
            verbose=(epoch % 10 == 0),
            device=device
        )
        
        # End epoch tracking
        activation_tracker.end_epoch()
        if activation_balancer is not None:
            activation_balancer.end_epoch()
        
        # Print periodic diagnostic reports
        if (epoch + 1) % 10 == 0:
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
    print("🔍 FINAL ENHANCED TRAINING ANALYSIS")
    print("="*80)
    
    activation_tracker.print_diagnostic_report()
    
    if activation_balancer is not None:
        activation_balancer.print_balance_report()
    
    # Generate plots
    try:
        activation_tracker.plot_activation_frequency(save_path=os.path.join(log_path, "activation_frequency.png"))
        activation_tracker.plot_epoch_evolution(save_path=os.path.join(log_path, "epoch_evolution.png"))
        print(f"📊 Plots saved to {log_path}")
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")

    return loss_log, activation_tracker, activation_balancer
