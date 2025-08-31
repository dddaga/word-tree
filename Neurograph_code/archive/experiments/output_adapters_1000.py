# modules/output_adapters_1000.py
# Output adapter for 1000-node network

import torch
import numpy as np
from modules.class_encoding import generate_digit_class_encodings

class OutputAdapter:
    def __init__(self, output_node_ids, num_classes, vector_dim, phase_bins, mag_bins, seed=42, device='cpu'):
        """
        Output adapter for converting between digit labels and output node contexts.
        
        Args:
            output_node_ids: List of output node IDs
            num_classes: Number of classes (10 for MNIST)
            vector_dim: Vector dimension per node
            phase_bins: Number of phase bins
            mag_bins: Number of magnitude bins
            seed: Random seed for class encodings
            device: Device to place tensors on ('cpu' or 'cuda')
        """
        self.output_node_ids = output_node_ids
        self.num_classes = num_classes
        self.vector_dim = vector_dim
        self.phase_bins = phase_bins
        self.mag_bins = mag_bins
        
        # Generate class encodings
        self.class_phase_encodings, self.class_mag_encodings = generate_digit_class_encodings(
            num_classes=num_classes,
            vector_dim=vector_dim,
            phase_bins=phase_bins,
            mag_bins=mag_bins,
            seed=seed
        )
        
        # Move encodings to the correct device
        self.device = device
        for i in range(num_classes):
            self.class_phase_encodings[i] = self.class_phase_encodings[i].to(device)
            self.class_mag_encodings[i] = self.class_mag_encodings[i].to(device)
        
        print(f"🎯 OutputAdapter initialized:")
        print(f"   📤 Output nodes: {len(output_node_ids)}")
        print(f"   🔢 Classes: {num_classes}")
        print(f"   📐 Vector dim: {vector_dim}")

    def get_output_context(self, label):
        """
        Get target output context for a given label.
        
        Args:
            label: Integer label (0-9 for MNIST)
            
        Returns:
            dict: {node_id: (phase_tensor, mag_tensor)}
        """
        if label < 0 or label >= self.num_classes:
            raise ValueError(f"Label {label} out of range [0, {self.num_classes-1}]")
        
        phase_encoding = self.class_phase_encodings[label].clone()
        mag_encoding = self.class_mag_encodings[label].clone()
        
        # Create target context for all output nodes
        target_context = {}
        for node_id in self.output_node_ids:
            target_context[node_id] = (phase_encoding, mag_encoding)
        
        return target_context

    def predict_label(self, final_activations):
        """
        Predict label from final activations using cosine similarity.
        
        Args:
            final_activations: dict {node_id: (phase_idx, mag_idx)}
            
        Returns:
            int: Predicted label (0-9) or -1 if no prediction possible
        """
        # Collect active output node activations
        active_outputs = []
        for node_id in self.output_node_ids:
            if node_id in final_activations:
                phase_idx, mag_idx = final_activations[node_id]
                active_outputs.append((phase_idx, mag_idx))
        
        if not active_outputs:
            return -1  # No active output nodes
        
        # Average the activations (simple ensemble)
        if len(active_outputs) == 1:
            avg_phase_idx, avg_mag_idx = active_outputs[0]
        else:
            # Average the indices
            phase_indices = torch.stack([act[0] for act in active_outputs])
            mag_indices = torch.stack([act[1] for act in active_outputs])
            avg_phase_idx = torch.round(torch.mean(phase_indices.float())).long()
            avg_mag_idx = torch.round(torch.mean(mag_indices.float())).long()
        
        # Compare with each class encoding using cosine similarity
        best_label = -1
        best_similarity = -1.0
        
        for label in range(self.num_classes):
            target_phase = self.class_phase_encodings[label]
            target_mag = self.class_mag_encodings[label]
            
            # Convert indices to float for similarity computation
            pred_phase_f = avg_phase_idx.float()
            pred_mag_f = avg_mag_idx.float()
            target_phase_f = target_phase.float()
            target_mag_f = target_mag.float()
            
            # Compute cosine similarity for phase and magnitude separately
            phase_sim = torch.cosine_similarity(pred_phase_f, target_phase_f, dim=0)
            mag_sim = torch.cosine_similarity(pred_mag_f, target_mag_f, dim=0)
            
            # Combined similarity (average of phase and magnitude)
            combined_sim = (phase_sim + mag_sim) / 2.0
            
            if combined_sim > best_similarity:
                best_similarity = combined_sim
                best_label = label
        
        return best_label

    def predict_label_detailed(self, final_activations):
        """
        Detailed prediction with confidence scores for debugging.
        
        Args:
            final_activations: dict {node_id: (phase_idx, mag_idx)}
            
        Returns:
            dict: Detailed prediction results
        """
        result = {
            'predicted_label': -1,
            'confidence': 0.0,
            'active_nodes': [],
            'class_similarities': {},
            'prediction_method': 'cosine_similarity'
        }
        
        # Collect active output nodes
        for node_id in self.output_node_ids:
            if node_id in final_activations:
                result['active_nodes'].append(node_id)
        
        if not result['active_nodes']:
            return result
        
        # Get prediction using main method
        predicted_label = self.predict_label(final_activations)
        result['predicted_label'] = predicted_label
        
        # Compute similarities for all classes
        active_outputs = [(final_activations[node_id]) for node_id in result['active_nodes']]
        
        if len(active_outputs) == 1:
            avg_phase_idx, avg_mag_idx = active_outputs[0]
        else:
            phase_indices = torch.stack([act[0] for act in active_outputs])
            mag_indices = torch.stack([act[1] for act in active_outputs])
            avg_phase_idx = torch.round(torch.mean(phase_indices.float())).long()
            avg_mag_idx = torch.round(torch.mean(mag_indices.float())).long()
        
        for label in range(self.num_classes):
            target_phase = self.class_phase_encodings[label]
            target_mag = self.class_mag_encodings[label]
            
            pred_phase_f = avg_phase_idx.float()
            pred_mag_f = avg_mag_idx.float()
            target_phase_f = target_phase.float()
            target_mag_f = target_mag.float()
            
            phase_sim = torch.cosine_similarity(pred_phase_f, target_phase_f, dim=0)
            mag_sim = torch.cosine_similarity(pred_mag_f, target_mag_f, dim=0)
            combined_sim = (phase_sim + mag_sim) / 2.0
            
            result['class_similarities'][label] = combined_sim.item()
        
        if predicted_label >= 0:
            result['confidence'] = result['class_similarities'][predicted_label]
        
        return result

    def get_class_encodings(self):
        """Get the class encodings for external use."""
        return {
            'phase': self.class_phase_encodings,
            'magnitude': self.class_mag_encodings
        }

    def analyze_encoding_similarity(self):
        """Analyze similarity between class encodings to detect potential confusion."""
        similarities = {}
        
        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
                phase_i = self.class_phase_encodings[i].float()
                phase_j = self.class_phase_encodings[j].float()
                mag_i = self.class_mag_encodings[i].float()
                mag_j = self.class_mag_encodings[j].float()
                
                phase_sim = torch.cosine_similarity(phase_i, phase_j, dim=0)
                mag_sim = torch.cosine_similarity(mag_i, mag_j, dim=0)
                combined_sim = (phase_sim + mag_sim) / 2.0
                
                similarities[(i, j)] = combined_sim.item()
        
        return similarities
