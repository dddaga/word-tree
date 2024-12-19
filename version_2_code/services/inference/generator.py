from typing import List, Optional, Dict
import torch
import numpy as np

class SequenceGenerator:
    def __init__(self, tokenizer, predictor):
        """
        Initialize sequence generator
        Args:
            tokenizer: Subword tokenizer instance
            predictor: Graph predictor instance
        """
        self.tokenizer = tokenizer
        self.predictor = predictor

    def generate_continuation(self, 
                            input_text: str, 
                            max_length: Optional[int] = None,
                            temperature: float = 1.0,
                            top_p: float = 0.9) -> List[Dict[str, any]]:
        """
        Generate continuations for input text
        Args:
            input_text: Input text to continue
            max_length: Maximum length of generated sequence
            temperature: Temperature for sampling (higher = more diverse)
            top_p: Nucleus sampling threshold
        Returns:
            List of dictionaries containing generated sequences and their metadata
        """
        # Tokenize input
        initial_subwords = self.tokenizer.tokenize(input_text)
        
        # Generate candidates using beam search
        candidates = self.predictor.beam_search(
            initial_subwords, 
            temperature=temperature
        )
        
        # Apply nucleus sampling
        if top_p < 1.0:
            probs = np.array([prob for prob, _ in candidates])
            sorted_indices = np.argsort(probs)[::-1]
            cumsum_probs = np.cumsum(probs[sorted_indices])
            cutoff_index = np.argmax(cumsum_probs > top_p)
            candidates = [candidates[i] for i in sorted_indices[:cutoff_index + 1]]
        
        # Format results
        results = []
        for prob, sequence in candidates:
            # Convert subwords back to text
            generated_text = self.tokenizer.detokenize(sequence)
            
            results.append({
                'text': generated_text,
                'probability': float(prob),
                'subwords': sequence,
                'length': len(sequence)
            })
        
        return results

    def generate_alternatives(self, 
                            input_text: str,
                            n_alternatives: int = 5) -> List[Dict[str, any]]:
        """
        Generate alternative completions for input text
        Args:
            input_text: Input text
            n_alternatives: Number of alternatives to generate
        Returns:
            List of alternative completions with metadata
        """
        results = self.generate_continuation(
            input_text,
            temperature=1.2,  # Higher temperature for more diversity
            top_p=0.9
        )
        
        # Sort by probability and take top-n
        sorted_results = sorted(results, 
                              key=lambda x: x['probability'], 
                              reverse=True)
        
        return sorted_results[:n_alternatives]
