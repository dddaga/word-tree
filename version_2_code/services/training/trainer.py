import torch
import torch.nn.functional as F
from model.complex_tensor import ComplexTensor
import numpy as np

class GraphTrainer:
    def __init__(self, graph_ops, context_prop, loss_weight=0.5):
        """
        Initialize graph trainer
        Args:
            graph_ops: Graph operations instance
            context_prop: Context propagation instance
            loss_weight: Weight for primary loss (default: 0.5)
        """
        self.graph_ops = graph_ops
        self.context_prop = context_prop
        self.loss_weight = loss_weight
        self.eps = 1e-7  # Small epsilon for numerical stability

    def calculate_loss(self, primary_output, negative_outputs):
        """
        Calculate loss using binary cross entropy with numerical stability
        Args:
            primary_output: Tensor from target node
            negative_outputs: List of tensors from negative samples
        """
        # Clamp values for numerical stability
        primary_sum = primary_output.sum().clamp(min=-100, max=100)
        primary_prob = torch.sigmoid(primary_sum)
        
        # Primary loss: -log(sigmoid(x))
        primary_loss = F.binary_cross_entropy_with_logits(
            primary_sum, 
            torch.ones_like(primary_sum),
            reduction='mean'
        )

        # Contrastive loss: -log(1 - sigmoid(x))
        contrastive_loss = 0
        for neg_output in negative_outputs:
            neg_sum = neg_output.sum().clamp(min=-100, max=100)
            contrastive_loss += F.binary_cross_entropy_with_logits(
                neg_sum,
                torch.zeros_like(neg_sum),
                reduction='mean'
            )

        # Combine losses
        total_loss = self.loss_weight * primary_loss + contrastive_loss
        return total_loss, primary_prob

    def train_step(self, current_node, next_node, running_context, optimizer):
        """
        Perform a single training step
        """
        # Get connected nodes and their weights
        connected_nodes = self.graph_ops.get_connected_subwords(current_node)
        
        if next_node not in connected_nodes:
            return running_context, 0.0

        # Convert weights to ComplexTensor
        next_weights = ComplexTensor(init_tensor=torch.tensor(
            connected_nodes[next_node]['weight']))
        
        # Forward pass with target node
        primary_output = running_context.forward(next_weights.tensor)
        
        # Get negative samples
        negative_outputs = []
        for other_node, other_data in connected_nodes.items():
            if other_node != next_node:
                other_weights = ComplexTensor(init_tensor=torch.tensor(
                    other_data['weight']))
                neg_output = running_context.forward(other_weights.tensor)
                negative_outputs.append(neg_output)

        # Calculate loss
        loss, primary_prob = self.calculate_loss(primary_output, negative_outputs)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update running context with propagation
        propagation_result = self.context_prop.propagate_context(
            running_context.tensor,
            next_weights.tensor,
            [{'weight': w.tensor.detach(), 'subword': n} 
             for n, w in connected_nodes.items()]
        )

        if propagation_result:
            # Update running context
            running_context = ComplexTensor(init_tensor=propagation_result[0][1])
            
            # Update graph weights
            for node_subword, updated_context, _ in propagation_result:
                self.graph_ops.update_node(
                    node_subword,
                    updated_context.detach().numpy()
                )

        return running_context, loss.item()
