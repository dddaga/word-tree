# utils/activation_balancer.py

import torch
from collections import defaultdict, Counter
from typing import Dict, List, Set, Optional, Tuple
import random

class ActivationBalancer:
    """
    Implements strategies to balance activation patterns across output nodes
    to mitigate risks from early output activation-based loss computation.
    """
    
    def __init__(self, 
                 output_nodes: List[str], 
                 strategy: str = "quota",
                 max_activations_per_epoch: int = 20,
                 min_activations_per_epoch: int = 2,
                 force_activation_probability: float = 0.3):
        """
        Initialize the activation balancer.
        
        Args:
            output_nodes: List of output node IDs
            strategy: Balancing strategy ("quota", "round_robin", "weighted", "penalty")
            max_activations_per_epoch: Maximum activations allowed per node per epoch
            min_activations_per_epoch: Minimum activations required per node per epoch
            force_activation_probability: Probability of forcing underutilized nodes
        """
        self.output_nodes = set(output_nodes)
        self.strategy = strategy
        self.max_activations_per_epoch = max_activations_per_epoch
        self.min_activations_per_epoch = min_activations_per_epoch
        self.force_activation_probability = force_activation_probability
        
        # Tracking per epoch
        self.current_epoch = 0
        self.epoch_activations = Counter()  # node_id -> count this epoch
        self.epoch_forced_activations = Counter()  # node_id -> forced count this epoch
        
        # Historical tracking
        self.total_activations = Counter()  # node_id -> total count
        self.total_forced_activations = Counter()  # node_id -> total forced count
        self.underutilized_nodes = set()
        self.overutilized_nodes = set()
        
    def should_force_activation(self, 
                              current_active_outputs: List[str],
                              true_label: Optional[int] = None) -> Tuple[bool, Optional[str]]:
        """
        Determine if we should force activation of an underutilized node.
        
        Args:
            current_active_outputs: Currently active output nodes
            true_label: Ground truth label (for targeted forcing)
            
        Returns:
            (should_force, target_node_id)
        """
        if self.strategy == "none":
            return False, None
            
        # Find underutilized nodes this epoch
        underutilized = []
        for node_id in self.output_nodes:
            if self.epoch_activations[node_id] < self.min_activations_per_epoch:
                underutilized.append(node_id)
        
        if not underutilized:
            return False, None
            
        # Strategy-specific logic
        if self.strategy == "quota":
            # Force activation if we have quota violations
            if random.random() < self.force_activation_probability:
                target_node = random.choice(underutilized)
                return True, target_node
                
        elif self.strategy == "round_robin":
            # Always force the most underutilized node
            min_activations = min(self.epoch_activations[node] for node in underutilized)
            candidates = [node for node in underutilized 
                         if self.epoch_activations[node] == min_activations]
            target_node = random.choice(candidates)
            return True, target_node
            
        elif self.strategy == "weighted":
            # Force based on inverse activation frequency
            weights = []
            for node_id in underutilized:
                # Higher weight for less activated nodes
                weight = 1.0 / (self.epoch_activations[node_id] + 1)
                weights.append(weight)
            
            if weights and random.random() < self.force_activation_probability:
                # Weighted random selection
                total_weight = sum(weights)
                r = random.random() * total_weight
                cumsum = 0
                for i, weight in enumerate(weights):
                    cumsum += weight
                    if r <= cumsum:
                        return True, underutilized[i]
        
        return False, None
    
    def should_suppress_activation(self, node_id: str) -> bool:
        """
        Determine if we should suppress activation of an overutilized node.
        
        Args:
            node_id: Node to check for suppression
            
        Returns:
            True if activation should be suppressed
        """
        if self.strategy == "none":
            return False
            
        if self.strategy in ["quota", "round_robin", "weighted"]:
            # Suppress if node has exceeded quota
            return self.epoch_activations[node_id] >= self.max_activations_per_epoch
            
        return False
    
    def record_activation(self, node_id: str, forced: bool = False):
        """Record an activation for tracking."""
        if node_id in self.output_nodes:
            self.epoch_activations[node_id] += 1
            self.total_activations[node_id] += 1
            
            if forced:
                self.epoch_forced_activations[node_id] += 1
                self.total_forced_activations[node_id] += 1
    
    def end_epoch(self):
        """Call at the end of each epoch to update tracking."""
        # Update utilization status
        self.underutilized_nodes.clear()
        self.overutilized_nodes.clear()
        
        for node_id in self.output_nodes:
            activations = self.epoch_activations[node_id]
            if activations < self.min_activations_per_epoch:
                self.underutilized_nodes.add(node_id)
            elif activations > self.max_activations_per_epoch:
                self.overutilized_nodes.add(node_id)
        
        # Reset epoch counters
        self.epoch_activations.clear()
        self.epoch_forced_activations.clear()
        self.current_epoch += 1
    
    def get_balance_summary(self) -> Dict:
        """Get comprehensive balancing statistics."""
        if not self.total_activations:
            return {"error": "No activation data recorded"}
        
        total_activations = sum(self.total_activations.values())
        total_forced = sum(self.total_forced_activations.values())
        
        return {
            "strategy": self.strategy,
            "total_activations": total_activations,
            "total_forced_activations": total_forced,
            "forced_percentage": (total_forced / total_activations * 100) if total_activations > 0 else 0,
            "current_epoch": self.current_epoch,
            "underutilized_nodes": len(self.underutilized_nodes),
            "overutilized_nodes": len(self.overutilized_nodes),
            "underutilized_list": list(self.underutilized_nodes),
            "overutilized_list": list(self.overutilized_nodes),
            "activation_distribution": dict(self.total_activations),
            "forced_distribution": dict(self.total_forced_activations)
        }
    
    def print_balance_report(self):
        """Print a comprehensive balancing report."""
        print("\n" + "="*60)
        print("⚖️  ACTIVATION BALANCING REPORT")
        print("="*60)
        
        summary = self.get_balance_summary()
        if "error" in summary:
            print(f"❌ {summary['error']}")
            return
        
        print(f"\n📊 BALANCING STATISTICS:")
        print(f"   Strategy: {summary['strategy']}")
        print(f"   Total Activations: {summary['total_activations']}")
        print(f"   Forced Activations: {summary['total_forced_activations']} ({summary['forced_percentage']:.1f}%)")
        print(f"   Current Epoch: {summary['current_epoch']}")
        
        print(f"\n⚖️  BALANCE STATUS:")
        print(f"   Underutilized Nodes: {summary['underutilized_nodes']}")
        if summary['underutilized_list']:
            print(f"     {summary['underutilized_list']}")
        print(f"   Overutilized Nodes: {summary['overutilized_nodes']}")
        if summary['overutilized_list']:
            print(f"     {summary['overutilized_list']}")
        
        print(f"\n📋 ACTIVATION DISTRIBUTION:")
        sorted_activations = sorted(summary['activation_distribution'].items(), 
                                  key=lambda x: x[1], reverse=True)
        for node_id, count in sorted_activations:
            forced_count = summary['forced_distribution'].get(node_id, 0)
            forced_pct = (forced_count / count * 100) if count > 0 else 0
            print(f"   {node_id}: {count} total ({forced_count} forced, {forced_pct:.1f}%)")
        
        print("="*60)


class MultiOutputLossStrategy:
    """
    Implements multi-output loss computation to train multiple nodes per forward pass
    instead of stopping at the first activation.
    """
    
    def __init__(self, 
                 continue_timesteps: int = 2,
                 max_outputs_to_train: int = 3):
        """
        Initialize multi-output loss strategy.
        
        Args:
            continue_timesteps: Additional timesteps to run after first activation
            max_outputs_to_train: Maximum number of output nodes to train per forward pass
        """
        self.continue_timesteps = continue_timesteps
        self.max_outputs_to_train = max_outputs_to_train
    
    def should_continue_forward(self, 
                              timesteps_since_first_activation: int,
                              current_active_outputs: List[str]) -> bool:
        """
        Determine if forward pass should continue after first activation.
        
        Args:
            timesteps_since_first_activation: Timesteps since first output activated
            current_active_outputs: Currently active output nodes
            
        Returns:
            True if forward pass should continue
        """
        # Continue for a few more timesteps to activate more outputs
        if timesteps_since_first_activation < self.continue_timesteps:
            return True
            
        # Stop if we have enough active outputs
        if len(current_active_outputs) >= self.max_outputs_to_train:
            return False
            
        return False
