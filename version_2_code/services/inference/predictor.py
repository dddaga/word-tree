import torch
import torch.nn.functional as F
from model.complex_tensor import ComplexTensor
from typing import List, Tuple, Dict
import heapq

class GraphPredictor:
    def __init__(self, graph_ops, context_prop, beam_width=5, max_length=50):
        """
        Initialize graph predictor
        Args:
            graph_ops: Graph operations instance
            context_prop: Context propagation instance
            beam_width: Number of parallel paths to consider
            max_length: Maximum sequence length to generate
        """
        self.graph_ops = graph_ops
        self.context_prop = context_prop
        self.beam_width = beam_width
        self.max_length = max_length

    def _get_next_candidates(self, 
                           node: str, 
                           context: ComplexTensor, 
                           top_k: int = 5) -> List[Tuple[float, str]]:
        """
        Get top-k next candidates based on context alignment
        Returns: List of (probability, node) tuples
        """
        connected_nodes = self.graph_ops.get_connected_subwords(node)
        candidates = []
        
        for next_node, data in connected_nodes.items():
            # Convert to ComplexTensor
            next_weights = ComplexTensor(init_tensor=torch.tensor(data['weight']))
            
            # Calculate alignment score
            output = context.forward(next_weights.tensor)
            score = torch.sigmoid(output.sum()).item()
            
            candidates.append((-score, next_node))  # Negative for max-heap
        
        # Return top-k candidates
        return heapq.nsmallest(top_k, candidates)

    def beam_search(self, 
                   initial_subwords: List[str], 
                   temperature: float = 1.0) -> List[Tuple[float, List[str]]]:
        """
        Perform beam search to generate sequences
        Args:
            initial_subwords: Initial sequence of subwords
            temperature: Temperature for probability scaling
        Returns:
            List of (probability, sequence) tuples
        """
        if not initial_subwords:
            return []

        # Initialize beam with starting sequence
        running_context = ComplexTensor(N=self.context_prop.complex_tensor.N,
                                      steps=self.context_prop.complex_tensor.steps,
                                      max_clip_value=self.context_prop.complex_tensor.max_clip_value)
        
        beam = [(1.0, initial_subwords.copy(), running_context)]
        
        for _ in range(self.max_length - len(initial_subwords)):
            candidates = []
            
            # Expand each sequence in beam
            for prob, sequence, context in beam:
                next_candidates = self._get_next_candidates(sequence[-1], context)
                
                for neg_score, next_node in next_candidates:
                    score = -neg_score  # Convert back to positive
                    
                    # Apply temperature
                    scaled_prob = torch.sigmoid(torch.tensor(score / temperature)).item()
                    new_prob = prob * scaled_prob
                    
                    # Propagate context
                    node_weights = self.graph_ops.get_connected_subwords(sequence[-1])
                    next_weights = ComplexTensor(init_tensor=torch.tensor(
                        node_weights[next_node]['weight']))
                    
                    propagation_result = self.context_prop.propagate_context(
                        context.tensor,
                        next_weights.tensor,
                        [{'weight': next_weights.tensor, 'subword': next_node}]
                    )
                    
                    if propagation_result:
                        new_context = ComplexTensor(init_tensor=propagation_result[0][1])
                        new_sequence = sequence + [next_node]
                        candidates.append((new_prob, new_sequence, new_context))
            
            # Select top-k candidates for next iteration
            beam = heapq.nlargest(self.beam_width, candidates, key=lambda x: x[0])
            
            # Early stopping if all sequences have converged
            if all(len(set(seq)) == 1 for _, seq, _ in beam):
                break
        
        # Return final sequences with their probabilities
        return [(prob, seq) for prob, seq, _ in beam]
