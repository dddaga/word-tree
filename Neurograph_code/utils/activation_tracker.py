# utils/activation_tracker.py

import torch
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Set, Tuple, Optional

class ActivationFrequencyTracker:
    """
    Tracks activation patterns of output nodes to diagnose training imbalances
    and identify risks from early output activation-based loss computation.
    """
    
    def __init__(self, output_nodes: List[str], num_classes: int = 10):
        """
        Initialize the activation tracker.
        
        Args:
            output_nodes: List of output node IDs
            num_classes: Number of classes (for MNIST = 10)
        """
        self.output_nodes = set(output_nodes)
        self.num_classes = num_classes
        
        # Core tracking data structures
        self.total_activations = Counter()  # node_id -> total activation count
        self.first_activations = Counter()  # node_id -> count of times it activated first
        self.activation_timesteps = defaultdict(list)  # node_id -> [timestep1, timestep2, ...]
        self.epoch_history = defaultdict(lambda: Counter())  # epoch -> {node_id: count}
        
        # Per-class tracking (if labels are provided)
        self.class_activations = defaultdict(lambda: Counter())  # class -> {node_id: count}
        self.class_first_activations = defaultdict(lambda: Counter())  # class -> {node_id: count}
        
        # Risk detection
        self.dead_nodes = set()
        self.dominant_nodes = set()
        
        # Current epoch tracking
        self.current_epoch = 0
        self.current_epoch_activations = Counter()
        
    def record_forward_pass(self, 
                          active_outputs: List[str], 
                          activation_timesteps: Dict[str, int],
                          first_active_output: Optional[str] = None,
                          true_label: Optional[int] = None):
        """
        Record activation data from a single forward pass.
        
        Args:
            active_outputs: List of output nodes that activated
            activation_timesteps: Dict mapping node_id -> timestep when it first activated
            first_active_output: The first output node to activate (if known)
            true_label: Ground truth class label (0-9 for MNIST)
        """
        # Record total activations
        for node_id in active_outputs:
            if node_id in self.output_nodes:
                self.total_activations[node_id] += 1
                self.current_epoch_activations[node_id] += 1
                
                # Record activation timestep
                if node_id in activation_timesteps:
                    self.activation_timesteps[node_id].append(activation_timesteps[node_id])
                
                # Record per-class activations if label provided
                if true_label is not None:
                    self.class_activations[true_label][node_id] += 1
        
        # Record first activation
        if first_active_output and first_active_output in self.output_nodes:
            self.first_activations[first_active_output] += 1
            if true_label is not None:
                self.class_first_activations[true_label][first_active_output] += 1
    
    def end_epoch(self):
        """Call at the end of each epoch to update epoch history."""
        self.epoch_history[self.current_epoch] = self.current_epoch_activations.copy()
        self.current_epoch_activations.clear()
        self.current_epoch += 1
        
        # Update risk detection
        self._update_risk_detection()
    
    def _update_risk_detection(self):
        """Update dead and dominant node detection."""
        # Dead nodes: never activated
        self.dead_nodes = self.output_nodes - set(self.total_activations.keys())
        
        # Dominant nodes: activated significantly more than average
        if self.total_activations:
            total_activations = sum(self.total_activations.values())
            avg_activations = total_activations / len(self.output_nodes)
            threshold = avg_activations * 2.0  # 2x average = dominant
            
            self.dominant_nodes = {
                node_id for node_id, count in self.total_activations.items()
                if count > threshold
            }
    
    def get_activation_summary(self) -> Dict:
        """Get comprehensive activation statistics."""
        if not self.total_activations:
            return {"error": "No activation data recorded"}
        
        activation_counts = list(self.total_activations.values())
        
        return {
            "total_forward_passes": sum(activation_counts),
            "total_activations": sum(activation_counts),
            "unique_active_nodes": len(self.total_activations),
            "dead_nodes": len(self.dead_nodes),
            "dominant_nodes": len(self.dominant_nodes),
            "activation_stats": {
                "mean": np.mean(activation_counts),
                "std": np.std(activation_counts),
                "min": np.min(activation_counts),
                "max": np.max(activation_counts),
                "median": np.median(activation_counts)
            },
            "dead_node_list": list(self.dead_nodes),
            "dominant_node_list": list(self.dominant_nodes)
        }
    
    def get_first_activation_summary(self) -> Dict:
        """Get statistics about which nodes activate first."""
        if not self.first_activations:
            return {"error": "No first activation data recorded"}
        
        total_first = sum(self.first_activations.values())
        
        return {
            "total_first_activations": total_first,
            "nodes_that_activated_first": len(self.first_activations),
            "first_activation_distribution": dict(self.first_activations),
            "first_activation_percentages": {
                node_id: (count / total_first) * 100
                for node_id, count in self.first_activations.items()
            }
        }
    
    def print_diagnostic_report(self):
        """Print a comprehensive diagnostic report."""
        print("\n" + "="*60)
        print("🔍 ACTIVATION FREQUENCY DIAGNOSTIC REPORT")
        print("="*60)
        
        # Overall statistics
        summary = self.get_activation_summary()
        if "error" not in summary:
            print(f"\n📊 OVERALL STATISTICS:")
            print(f"   Total Forward Passes: {summary['total_forward_passes']}")
            print(f"   Active Output Nodes: {summary['unique_active_nodes']}/{len(self.output_nodes)}")
            print(f"   Dead Nodes: {summary['dead_nodes']}")
            print(f"   Dominant Nodes: {summary['dominant_nodes']}")
            
            stats = summary['activation_stats']
            print(f"\n📈 ACTIVATION DISTRIBUTION:")
            print(f"   Mean: {stats['mean']:.2f}")
            print(f"   Std:  {stats['std']:.2f}")
            print(f"   Min:  {stats['min']}")
            print(f"   Max:  {stats['max']}")
            print(f"   Median: {stats['median']:.2f}")
        
        # Risk assessment
        print(f"\n⚠️  RISK ASSESSMENT:")
        if self.dead_nodes:
            print(f"   💀 DEAD NODES ({len(self.dead_nodes)}): {sorted(list(self.dead_nodes))}")
        else:
            print(f"   ✅ No dead nodes detected")
            
        if self.dominant_nodes:
            print(f"   🎯 DOMINANT NODES ({len(self.dominant_nodes)}): {sorted(list(self.dominant_nodes))}")
        else:
            print(f"   ✅ No dominant nodes detected")
        
        # First activation analysis
        first_summary = self.get_first_activation_summary()
        if "error" not in first_summary:
            print(f"\n🥇 FIRST ACTIVATION ANALYSIS:")
            print(f"   Nodes that activated first: {first_summary['nodes_that_activated_first']}/{len(self.output_nodes)}")
            
            # Show top first activators
            sorted_first = sorted(
                first_summary['first_activation_percentages'].items(),
                key=lambda x: x[1], reverse=True
            )
            print(f"   Top first activators:")
            for node_id, percentage in sorted_first[:5]:
                print(f"     {node_id}: {percentage:.1f}%")
        
        # Per-node detailed breakdown
        print(f"\n📋 PER-NODE BREAKDOWN:")
        sorted_nodes = sorted(self.total_activations.items(), key=lambda x: x[1], reverse=True)
        for node_id, count in sorted_nodes:
            first_count = self.first_activations.get(node_id, 0)
            avg_timestep = np.mean(self.activation_timesteps[node_id]) if self.activation_timesteps[node_id] else 0
            print(f"   {node_id}: {count} total, {first_count} first, avg_timestep: {avg_timestep:.1f}")
        
        print("="*60)
    
    def plot_activation_frequency(self, save_path: Optional[str] = None):
        """Plot activation frequency distribution."""
        if not self.total_activations:
            print("No activation data to plot")
            return
        
        # Prepare data
        all_nodes = sorted(list(self.output_nodes))
        activation_counts = [self.total_activations.get(node, 0) for node in all_nodes]
        first_counts = [self.first_activations.get(node, 0) for node in all_nodes]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Total activations
        bars1 = ax1.bar(range(len(all_nodes)), activation_counts, alpha=0.7, color='skyblue')
        ax1.set_title('Total Activation Frequency per Output Node')
        ax1.set_xlabel('Output Node')
        ax1.set_ylabel('Total Activations')
        ax1.set_xticks(range(len(all_nodes)))
        ax1.set_xticklabels(all_nodes, rotation=45)
        
        # Highlight dead and dominant nodes
        for i, node in enumerate(all_nodes):
            if node in self.dead_nodes:
                bars1[i].set_color('red')
                bars1[i].set_alpha(0.8)
            elif node in self.dominant_nodes:
                bars1[i].set_color('orange')
                bars1[i].set_alpha(0.8)
        
        # Add legend
        ax1.legend(['Normal', 'Dead', 'Dominant'])
        
        # Plot 2: First activations
        bars2 = ax2.bar(range(len(all_nodes)), first_counts, alpha=0.7, color='lightgreen')
        ax2.set_title('First Activation Frequency per Output Node')
        ax2.set_xlabel('Output Node')
        ax2.set_ylabel('First Activations')
        ax2.set_xticks(range(len(all_nodes)))
        ax2.set_xticklabels(all_nodes, rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Activation frequency plot saved to: {save_path}")
        else:
            plt.show()
    
    def plot_epoch_evolution(self, save_path: Optional[str] = None):
        """Plot how activation patterns evolve over epochs."""
        if not self.epoch_history:
            print("No epoch history to plot")
            return
        
        # Prepare data
        epochs = sorted(self.epoch_history.keys())
        all_nodes = sorted(list(self.output_nodes))
        
        # Create activation matrix: epochs x nodes
        activation_matrix = []
        for epoch in epochs:
            epoch_data = [self.epoch_history[epoch].get(node, 0) for node in all_nodes]
            activation_matrix.append(epoch_data)
        
        activation_matrix = np.array(activation_matrix)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        plt.imshow(activation_matrix.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        plt.colorbar(label='Activations per Epoch')
        plt.title('Activation Frequency Evolution Across Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Output Node')
        plt.yticks(range(len(all_nodes)), all_nodes)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Epoch evolution plot saved to: {save_path}")
        else:
            plt.show()
