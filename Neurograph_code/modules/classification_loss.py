"""
Classification Loss for Modular NeuroGraph
Implements categorical cross-entropy with cosine similarity logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from core.high_res_tables import HighResolutionLookupTables

class ClassificationLoss(nn.Module):
    """
    Categorical cross-entropy loss with cosine similarity logits.
    
    Features:
    - Interprets output nodes as class logits via cosine similarity
    - Supports temperature scaling for calibration
    - Label smoothing for regularization
    - Structured class encodings integration
    """
    
    def __init__(self, num_classes: int = 10, temperature: float = 1.0, 
                 label_smoothing: float = 0.0, reduction: str = 'mean'):
        """
        Initialize classification loss.
        
        Args:
            num_classes: Number of classes (10 for MNIST)
            temperature: Temperature for softmax scaling
            label_smoothing: Label smoothing factor [0, 1]
            reduction: Loss reduction ('mean', 'sum', 'none')
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        
        print(f"🔧 Initializing Classification Loss:")
        print(f"   📊 Classes: {num_classes}")
        print(f"   🌡️  Temperature: {temperature}")
        print(f"   🎯 Label smoothing: {label_smoothing}")
        print(f"   📉 Reduction: {reduction}")
    
    def compute_logits_from_signals(self, output_signals: Dict[int, torch.Tensor], 
                                   class_encodings: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
                                   lookup_tables: HighResolutionLookupTables) -> torch.Tensor:
        """
        Compute class logits from output node signals using cosine similarity.
        
        Args:
            output_signals: Dict mapping output_node_id -> signal_vector [vector_dim]
            class_encodings: Dict mapping class_id -> (phase_indices, mag_indices)
            lookup_tables: High-resolution lookup tables for signal computation
            
        Returns:
            Logits tensor of shape [num_classes]
        """
        # Convert class encodings to signal vectors
        class_signal_vectors = {}
        for class_id, (phase_idx, mag_idx) in class_encodings.items():
            class_signal_vectors[class_id] = lookup_tables.get_signal_vector(phase_idx, mag_idx)
        
        # Compute cosine similarities between output signals and class encodings
        logits = torch.zeros(self.num_classes, device=next(iter(output_signals.values())).device)
        
        for class_id in range(self.num_classes):
            if class_id not in class_signal_vectors:
                continue
                
            class_vector = class_signal_vectors[class_id]  # [vector_dim]
            similarities = []
            
            # Compute similarity with each output node
            for output_node_id, output_signal in output_signals.items():
                # Cosine similarity between output signal and class encoding
                similarity = F.cosine_similarity(
                    output_signal.unsqueeze(0), 
                    class_vector.unsqueeze(0)
                ).item()
                similarities.append(similarity)
            
            # Aggregate similarities (mean, max, or weighted combination)
            if similarities:
                logits[class_id] = torch.tensor(np.mean(similarities))
        
        return logits
    
    def compute_logits_from_activations(self, activation_table: Dict[int, Dict[str, torch.Tensor]], 
                                      output_node_ids: List[int],
                                      class_encodings: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
                                      lookup_tables: HighResolutionLookupTables) -> torch.Tensor:
        """
        Compute logits from activation table (alternative method).
        
        Args:
            activation_table: Activation table with node states
            output_node_ids: List of output node IDs
            class_encodings: Class encodings
            lookup_tables: Lookup tables
            
        Returns:
            Logits tensor of shape [num_classes]
        """
        device = next(iter(class_encodings.values()))[0].device
        logits = torch.zeros(self.num_classes, device=device)
        
        # Extract output node signals from activation table
        output_signals = {}
        for node_id in output_node_ids:
            if node_id in activation_table:
                # Get phase and magnitude indices from activation table
                phase_idx = activation_table[node_id].get('phase_idx')
                mag_idx = activation_table[node_id].get('mag_idx')
                
                if phase_idx is not None and mag_idx is not None:
                    # Convert to signal vector
                    signal = lookup_tables.get_signal_vector(phase_idx, mag_idx)
                    output_signals[node_id] = signal
        
        # Compute logits using signal-based method
        if output_signals:
            logits = self.compute_logits_from_signals(output_signals, class_encodings, lookup_tables)
        
        return logits
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute categorical cross-entropy loss.
        
        Args:
            logits: Predicted logits of shape [batch_size, num_classes] or [num_classes]
            targets: Target class indices of shape [batch_size] or scalar
            
        Returns:
            Loss tensor
        """
        # Handle single sample case
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if targets.dim() == 0:
            targets = targets.unsqueeze(0)
        
        # Apply temperature scaling
        scaled_logits = logits / self.temperature
        
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            loss = self._cross_entropy_with_label_smoothing(scaled_logits, targets)
        else:
            loss = F.cross_entropy(scaled_logits, targets, reduction=self.reduction)
        
        return loss
    
    def _cross_entropy_with_label_smoothing(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss with label smoothing.
        
        Args:
            logits: Predicted logits [batch_size, num_classes]
            targets: Target indices [batch_size]
            
        Returns:
            Smoothed loss tensor
        """
        # Create smoothed target distribution
        batch_size = logits.size(0)
        smoothed_targets = torch.zeros_like(logits)
        
        # Fill with uniform distribution for smoothing
        smoothed_targets.fill_(self.label_smoothing / (self.num_classes - 1))
        
        # Set correct class probability
        for i in range(batch_size):
            smoothed_targets[i, targets[i]] = 1.0 - self.label_smoothing
        
        # Compute KL divergence
        log_probs = F.log_softmax(logits, dim=1)
        loss = -torch.sum(smoothed_targets * log_probs, dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def compute_accuracy(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Compute classification accuracy.
        
        Args:
            logits: Predicted logits
            targets: Target class indices
            
        Returns:
            Accuracy as float
        """
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if targets.dim() == 0:
            targets = targets.unsqueeze(0)
        
        predictions = torch.argmax(logits, dim=1)
        correct = (predictions == targets).float()
        
        return correct.mean().item()
    
    def compute_top_k_accuracy(self, logits: torch.Tensor, targets: torch.Tensor, k: int = 3) -> float:
        """
        Compute top-k accuracy.
        
        Args:
            logits: Predicted logits
            targets: Target class indices
            k: Number of top predictions to consider
            
        Returns:
            Top-k accuracy as float
        """
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if targets.dim() == 0:
            targets = targets.unsqueeze(0)
        
        _, top_k_preds = torch.topk(logits, k, dim=1)
        targets_expanded = targets.unsqueeze(1).expand_as(top_k_preds)
        
        correct = (top_k_preds == targets_expanded).any(dim=1).float()
        
        return correct.mean().item()
    
    def get_class_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to class probabilities.
        
        Args:
            logits: Raw logits
            
        Returns:
            Probability distribution over classes
        """
        scaled_logits = logits / self.temperature
        return F.softmax(scaled_logits, dim=-1)
    
    def get_prediction_confidence(self, logits: torch.Tensor) -> Tuple[int, float]:
        """
        Get predicted class and confidence.
        
        Args:
            logits: Raw logits
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        probs = self.get_class_probabilities(logits)
        
        if probs.dim() > 1:
            probs = probs.squeeze(0)
        
        confidence, predicted_class = torch.max(probs, dim=0)
        
        return predicted_class.item(), confidence.item()

class AdaptiveClassificationLoss(ClassificationLoss):
    """
    Adaptive classification loss with dynamic temperature scaling.
    """
    
    def __init__(self, **kwargs):
        """Initialize adaptive loss."""
        super().__init__(**kwargs)
        
        # Track statistics for adaptive temperature
        self.register_buffer('running_confidence', torch.tensor(0.5))
        self.register_buffer('num_samples', torch.tensor(0, dtype=torch.long))
        
        self.target_confidence = 0.7  # Target confidence level
        self.temperature_lr = 0.01    # Learning rate for temperature adaptation
    
    def update_temperature(self, logits: torch.Tensor):
        """Update temperature based on prediction confidence."""
        if self.training:
            with torch.no_grad():
                probs = F.softmax(logits / self.temperature, dim=-1)
                max_probs = torch.max(probs, dim=-1)[0]
                current_confidence = max_probs.mean()
                
                # Update running confidence
                momentum = 0.1
                self.running_confidence = (1 - momentum) * self.running_confidence + momentum * current_confidence
                self.num_samples += 1
                
                # Adapt temperature
                if self.running_confidence > self.target_confidence:
                    # Too confident, increase temperature (soften predictions)
                    self.temperature += self.temperature_lr
                else:
                    # Not confident enough, decrease temperature (sharpen predictions)
                    self.temperature = max(0.1, self.temperature - self.temperature_lr)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive temperature."""
        self.update_temperature(logits)
        return super().forward(logits, targets)

class MultiOutputClassificationLoss(ClassificationLoss):
    """
    Classification loss for multiple output nodes with different specializations.
    """
    
    def __init__(self, output_node_weights: Optional[Dict[int, float]] = None, **kwargs):
        """
        Initialize multi-output loss.
        
        Args:
            output_node_weights: Weights for different output nodes
            **kwargs: Other arguments for base class
        """
        super().__init__(**kwargs)
        
        self.output_node_weights = output_node_weights or {}
    
    def compute_weighted_logits(self, output_signals: Dict[int, torch.Tensor], 
                               class_encodings: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
                               lookup_tables: HighResolutionLookupTables) -> torch.Tensor:
        """
        Compute weighted logits from multiple output nodes.
        
        Args:
            output_signals: Output node signals
            class_encodings: Class encodings
            lookup_tables: Lookup tables
            
        Returns:
            Weighted logits tensor
        """
        # Get individual logits for each output node
        individual_logits = {}
        
        for output_node_id, signal in output_signals.items():
            node_signals = {output_node_id: signal}
            node_logits = self.compute_logits_from_signals(node_signals, class_encodings, lookup_tables)
            individual_logits[output_node_id] = node_logits
        
        # Combine with weights
        if not individual_logits:
            device = next(iter(class_encodings.values()))[0].device
            return torch.zeros(self.num_classes, device=device)
        
        weighted_logits = torch.zeros_like(next(iter(individual_logits.values())))
        total_weight = 0.0
        
        for node_id, logits in individual_logits.items():
            weight = self.output_node_weights.get(node_id, 1.0)
            weighted_logits += weight * logits
            total_weight += weight
        
        if total_weight > 0:
            weighted_logits /= total_weight
        
        return weighted_logits

def create_classification_loss(loss_type: str = "standard", **kwargs) -> ClassificationLoss:
    """
    Factory function to create classification losses.
    
    Args:
        loss_type: Type of loss ("standard", "adaptive", "multi_output")
        **kwargs: Additional arguments for loss
        
    Returns:
        Classification loss instance
    """
    if loss_type == "standard":
        return ClassificationLoss(**kwargs)
    elif loss_type == "adaptive":
        return AdaptiveClassificationLoss(**kwargs)
    elif loss_type == "multi_output":
        return MultiOutputClassificationLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
