"""
Orthogonal Class Encodings for Modular NeuroGraph
Generates structured, orthogonal vectors for digit classes (0-9)
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, Tuple, Optional, List
from scipy.linalg import qr
from sklearn.preprocessing import normalize

class OrthogonalClassEncoder:
    """
    Generates orthogonal class encodings for improved class separation.
    
    Features:
    - True orthogonal vectors (dot product = 0)
    - Quantized to discrete phase-magnitude indices
    - Configurable encoding dimension
    - Validation of orthogonality constraints
    """
    
    def __init__(self, num_classes: int, encoding_dim: int,
                 phase_bins: int, mag_bins: int,
                 orthogonality_threshold: float = 0.1, device: str = 'cpu',
                 cache_encodings: bool = True):
        """
        Initialize orthogonal class encoder.
        
        Args:
            num_classes: Number of classes (10 for MNIST digits)
            encoding_dim: Dimension of encoding vectors
            phase_bins: Number of discrete phase bins
            mag_bins: Number of discrete magnitude bins
            orthogonality_threshold: Maximum allowed dot product between classes
            device: Computation device
            cache_encodings: Whether to cache encodings to avoid regeneration
        """
        self.num_classes = num_classes
        self.encoding_dim = encoding_dim
        self.phase_bins = phase_bins
        self.mag_bins = mag_bins
        self.orthogonality_threshold = orthogonality_threshold
        self.device = device
        self.cache_encodings = cache_encodings
        
        print(f"🔧 Initializing Orthogonal Class Encoder:")
        print(f"   📊 Classes: {num_classes}")
        print(f"   📐 Encoding dimension: {encoding_dim}")
        print(f"   🎯 Orthogonality threshold: {orthogonality_threshold}")
        print(f"   📈 Resolution: {phase_bins}×{mag_bins}")
        print(f"   💾 Caching: {cache_encodings}")
        
        # Try to load cached encodings first
        cache_path = self._get_cache_path()
        if cache_encodings and self._load_cached_encodings(cache_path):
            print(f"📂 Loaded cached encodings from {cache_path}")
            print(f"✅ Orthogonal class encodings ready (cached)")
        else:
            # Generate new encodings
            print(f"🔄 Generating new orthogonal encodings...")
            self.class_encodings = self.generate_orthogonal_encodings()
            self.validate_orthogonality()
            
            # Save to cache if enabled
            if cache_encodings:
                self._save_cached_encodings(cache_path)
                print(f"💾 Encodings cached to {cache_path}")
            
            print(f"✅ Orthogonal class encodings generated")
    
    def generate_orthogonal_encodings(self) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate orthogonal class encodings with improved discrete space preservation.
        
        Returns:
            Dictionary mapping class_id -> (phase_indices, mag_indices)
        """
        print(f"🔄 Generating new orthogonal encodings...")
        
        # Method 1: Generate continuous orthogonal vectors
        if self.encoding_dim >= self.num_classes:
            continuous_encodings = self._generate_qr_orthogonal()
        else:
            continuous_encodings = self._generate_gram_schmidt_orthogonal()
        
        # Method 2: IMPROVED - Convert to discrete space with orthogonality preservation
        class_encodings = self._optimize_discrete_orthogonality(continuous_encodings)
        
        return class_encodings
    
    def _generate_qr_orthogonal(self) -> Dict[int, np.ndarray]:
        """Generate orthogonal vectors using QR decomposition."""
        # Create random matrix and apply QR decomposition
        np.random.seed(42)  # For reproducibility
        random_matrix = np.random.randn(self.encoding_dim, self.num_classes)
        
        # QR decomposition gives orthogonal columns
        Q, R = qr(random_matrix)
        
        # Extract orthogonal vectors
        encodings = {}
        for i in range(self.num_classes):
            encodings[i] = Q[:, i]
        
        return encodings
    
    def _generate_gram_schmidt_orthogonal(self) -> Dict[int, np.ndarray]:
        """Generate orthogonal vectors using Gram-Schmidt process."""
        np.random.seed(42)
        vectors = []
        
        for i in range(self.num_classes):
            # Start with random vector
            if i == 0:
                v = np.random.randn(self.encoding_dim)
            else:
                # Generate vector orthogonal to previous ones
                v = np.random.randn(self.encoding_dim)
                
                # Gram-Schmidt orthogonalization
                for prev_v in vectors:
                    v = v - np.dot(v, prev_v) * prev_v
            
            # Normalize
            v = v / np.linalg.norm(v)
            vectors.append(v)
        
        # Convert to dictionary
        encodings = {}
        for i, v in enumerate(vectors):
            encodings[i] = v
        
        return encodings
    
    def _optimize_discrete_orthogonality(self, continuous_encodings: Dict[int, np.ndarray]) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        IMPROVED: Convert continuous orthogonal vectors to discrete space while preserving orthogonality.
        
        This method uses iterative optimization to find discrete encodings that maintain
        maximum orthogonality in the actual signal space used by the lookup tables.
        
        Args:
            continuous_encodings: Dictionary of continuous orthogonal vectors
            
        Returns:
            Dictionary mapping class_id -> (phase_indices, mag_indices)
        """
        from core.high_res_tables import HighResolutionLookupTables
        
        # Initialize lookup tables for signal computation
        lookup = HighResolutionLookupTables(self.phase_bins, self.mag_bins, self.device)
        
        # Step 1: Initial quantization using improved method
        class_encodings = {}
        for class_id, continuous_vector in continuous_encodings.items():
            phase_indices, mag_indices = self._improved_vectorize_to_phase_mag(continuous_vector)
            class_encodings[class_id] = (phase_indices, mag_indices)
        
        # Step 2: Iterative refinement to improve orthogonality
        max_iterations = 50
        best_orthogonality_score = -1.0
        best_encodings = class_encodings.copy()
        
        for iteration in range(max_iterations):
            # Compute current orthogonality score
            current_score = self._compute_discrete_orthogonality_score(class_encodings, lookup)
            
            if current_score > best_orthogonality_score:
                best_orthogonality_score = current_score
                best_encodings = {k: (v[0].clone(), v[1].clone()) for k, v in class_encodings.items()}
            
            # Try small perturbations to improve orthogonality
            improved = False
            for class_id in range(self.num_classes):
                phase_idx, mag_idx = class_encodings[class_id]
                
                # Try small perturbations
                for dim in range(min(3, self.encoding_dim)):  # Only perturb first few dimensions
                    for delta in [-1, 1]:
                        # Try phase perturbation
                        new_phase_idx = phase_idx.clone()
                        new_phase_idx[dim] = (new_phase_idx[dim] + delta) % self.phase_bins
                        
                        # Test this perturbation
                        test_encodings = class_encodings.copy()
                        test_encodings[class_id] = (new_phase_idx, mag_idx)
                        
                        test_score = self._compute_discrete_orthogonality_score(test_encodings, lookup)
                        if test_score > current_score:
                            class_encodings[class_id] = (new_phase_idx, mag_idx)
                            current_score = test_score
                            improved = True
                            break
                        
                        # Try magnitude perturbation
                        new_mag_idx = mag_idx.clone()
                        new_mag_idx[dim] = torch.clamp(new_mag_idx[dim] + delta, 0, self.mag_bins - 1)
                        
                        test_encodings = class_encodings.copy()
                        test_encodings[class_id] = (phase_idx, new_mag_idx)
                        
                        test_score = self._compute_discrete_orthogonality_score(test_encodings, lookup)
                        if test_score > current_score:
                            class_encodings[class_id] = (phase_idx, new_mag_idx)
                            current_score = test_score
                            improved = True
                            break
                    
                    if improved:
                        break
                
                if improved:
                    break
            
            # Early stopping if no improvement
            if not improved:
                break
        
        print(f"   🔧 Discrete optimization: {iteration+1} iterations, "
              f"orthogonality score: {best_orthogonality_score:.4f}")
        
        return best_encodings
    
    def _improved_vectorize_to_phase_mag(self, continuous_vector: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        IMPROVED: Convert continuous vector to discrete indices with better orthogonality preservation.
        
        Args:
            continuous_vector: Numpy array of shape [encoding_dim]
            
        Returns:
            Tuple of (phase_indices, mag_indices) as LongTensors
        """
        # Method 1: Use polar coordinate transformation for better separation
        # Convert to complex representation for each dimension pair
        phase_indices = []
        mag_indices = []
        
        for i in range(self.encoding_dim):
            if i < len(continuous_vector):
                val = continuous_vector[i]
            else:
                val = 0.0
            
            # Phase: map value to phase using arctangent-like function
            # This provides better distribution across phase space
            phase_val = (np.arctan(val * 2) + np.pi/2) / np.pi * 2 * np.pi  # [0, 2π]
            phase_idx = int(phase_val / (2 * np.pi) * self.phase_bins)
            phase_idx = np.clip(phase_idx, 0, self.phase_bins - 1)
            phase_indices.append(phase_idx)
            
            # Magnitude: use sigmoid-like mapping for better distribution
            mag_val = 1 / (1 + np.exp(-val * 3))  # Sigmoid: [0, 1]
            mag_idx = int(mag_val * self.mag_bins)
            mag_idx = np.clip(mag_idx, 0, self.mag_bins - 1)
            mag_indices.append(mag_idx)
        
        return (torch.tensor(phase_indices, dtype=torch.long, device=self.device),
                torch.tensor(mag_indices, dtype=torch.long, device=self.device))
    
    def _compute_discrete_orthogonality_score(self, class_encodings: Dict[int, Tuple[torch.Tensor, torch.Tensor]], 
                                            lookup: 'HighResolutionLookupTables') -> float:
        """
        Compute orthogonality score for discrete encodings using actual signal vectors.
        
        Args:
            class_encodings: Dictionary of discrete encodings
            lookup: Lookup tables for signal computation
            
        Returns:
            Orthogonality score (higher is better, 1.0 is perfect)
        """
        # Compute signal vectors for all classes
        signal_vectors = {}
        for class_id, (phase_idx, mag_idx) in class_encodings.items():
            signal_vectors[class_id] = lookup.get_signal_vector(phase_idx, mag_idx)
        
        # Compute pairwise cosine similarities
        similarities = []
        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
                vec_i = signal_vectors[i]
                vec_j = signal_vectors[j]
                
                # Normalize vectors
                vec_i_norm = vec_i / (torch.norm(vec_i) + 1e-8)
                vec_j_norm = vec_j / (torch.norm(vec_j) + 1e-8)
                
                # Compute cosine similarity
                similarity = torch.dot(vec_i_norm, vec_j_norm).abs().item()
                similarities.append(similarity)
        
        # Orthogonality score: 1.0 - mean absolute similarity
        # Perfect orthogonality (all similarities = 0) gives score = 1.0
        mean_similarity = np.mean(similarities)
        orthogonality_score = 1.0 - mean_similarity
        
        return orthogonality_score
    
    def _vectorize_to_phase_mag(self, continuous_vector: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert continuous vector to discrete phase-magnitude indices.
        
        Args:
            continuous_vector: Numpy array of shape [encoding_dim]
            
        Returns:
            Tuple of (phase_indices, mag_indices) as LongTensors
        """
        # Split vector into phase and magnitude components
        # Use first half for phase, second half for magnitude
        mid_point = len(continuous_vector) // 2
        
        if len(continuous_vector) % 2 == 1:
            # Odd dimension: use middle element for both
            phase_part = continuous_vector[:mid_point + 1]
            mag_part = continuous_vector[mid_point:]
        else:
            # Even dimension: split evenly
            phase_part = continuous_vector[:mid_point]
            mag_part = continuous_vector[mid_point:]
        
        # Pad to encoding_dim if needed
        if len(phase_part) < self.encoding_dim:
            phase_part = np.pad(phase_part, (0, self.encoding_dim - len(phase_part)))
        if len(mag_part) < self.encoding_dim:
            mag_part = np.pad(mag_part, (0, self.encoding_dim - len(mag_part)))
        
        # Truncate to encoding_dim if needed
        phase_part = phase_part[:self.encoding_dim]
        mag_part = mag_part[:self.encoding_dim]
        
        # Convert to phase indices [0, phase_bins-1]
        # Map from [-1, 1] to [0, 2π] then quantize
        phase_normalized = (phase_part + 1) / 2  # [-1,1] -> [0,1]
        phase_radians = phase_normalized * 2 * math.pi  # [0,1] -> [0,2π]
        phase_indices = np.floor(phase_radians / (2 * math.pi) * self.phase_bins).astype(int)
        phase_indices = np.clip(phase_indices, 0, self.phase_bins - 1)
        
        # Convert to magnitude indices [0, mag_bins-1]
        # Map from [-1, 1] to [-3, 3] then quantize
        mag_scaled = mag_part * 3  # [-1,1] -> [-3,3]
        mag_normalized = (mag_scaled + 3) / 6  # [-3,3] -> [0,1]
        mag_indices = np.floor(mag_normalized * self.mag_bins).astype(int)
        mag_indices = np.clip(mag_indices, 0, self.mag_bins - 1)
        
        return (torch.tensor(phase_indices, dtype=torch.long, device=self.device),
                torch.tensor(mag_indices, dtype=torch.long, device=self.device))
    
    def validate_orthogonality(self):
        """Validate that generated encodings are sufficiently orthogonal."""
        print(f"\n🔍 Validating Orthogonality:")
        
        # Reconstruct continuous vectors for validation
        continuous_encodings = {}
        for class_id, (phase_idx, mag_idx) in self.class_encodings.items():
            # Convert back to continuous for dot product computation
            phase_vals = phase_idx.float() / self.phase_bins * 2 * math.pi
            mag_vals = (mag_idx.float() / self.mag_bins * 6) - 3
            
            # Combine phase and magnitude (simple concatenation for validation)
            continuous_encodings[class_id] = torch.cat([
                torch.cos(phase_vals), torch.sin(phase_vals), mag_vals
            ])
        
        # Compute pairwise dot products
        max_dot_product = 0.0
        worst_pair = (0, 1)
        
        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
                vec_i = continuous_encodings[i]
                vec_j = continuous_encodings[j]
                
                # Normalize vectors
                vec_i_norm = vec_i / torch.norm(vec_i)
                vec_j_norm = vec_j / torch.norm(vec_j)
                
                # Compute dot product
                dot_product = torch.dot(vec_i_norm, vec_j_norm).abs().item()
                
                if dot_product > max_dot_product:
                    max_dot_product = dot_product
                    worst_pair = (i, j)
                
                print(f"   Classes {i}-{j}: dot product = {dot_product:.4f}")
        
        # Check orthogonality threshold
        if max_dot_product <= self.orthogonality_threshold:
            print(f"✅ Orthogonality validated: max dot product = {max_dot_product:.4f}")
        else:
            print(f"⚠️  Orthogonality warning: max dot product = {max_dot_product:.4f} > {self.orthogonality_threshold}")
            print(f"   Worst pair: classes {worst_pair[0]}-{worst_pair[1]}")
    
    def get_class_encoding(self, class_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get phase-magnitude encoding for a specific class.
        
        Args:
            class_id: Class identifier (0-9 for MNIST)
            
        Returns:
            Tuple of (phase_indices, mag_indices)
        """
        if class_id not in self.class_encodings:
            raise ValueError(f"Class {class_id} not found in encodings")
        
        return self.class_encodings[class_id]
    
    def get_all_encodings(self) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """Get all class encodings."""
        return self.class_encodings.copy()
    
    def compute_similarity_matrix(self) -> torch.Tensor:
        """
        Compute similarity matrix between all class encodings.
        
        Returns:
            Tensor of shape [num_classes, num_classes] with cosine similarities
        """
        from core.high_res_tables import HighResolutionLookupTables
        
        # Initialize lookup tables for signal computation
        lookup = HighResolutionLookupTables(self.phase_bins, self.mag_bins, self.device)
        
        # Compute signal vectors for all classes
        signal_vectors = {}
        for class_id, (phase_idx, mag_idx) in self.class_encodings.items():
            signal_vectors[class_id] = lookup.get_signal_vector(phase_idx, mag_idx)
        
        # Compute similarity matrix
        similarity_matrix = torch.zeros(self.num_classes, self.num_classes, device=self.device)
        
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                vec_i = signal_vectors[i]
                vec_j = signal_vectors[j]
                
                # Cosine similarity
                similarity = torch.cosine_similarity(vec_i.unsqueeze(0), vec_j.unsqueeze(0))
                similarity_matrix[i, j] = similarity.item()
        
        return similarity_matrix
    
    def get_encoding_stats(self) -> Dict[str, float]:
        """Get statistics about the encodings."""
        similarity_matrix = self.compute_similarity_matrix()
        
        # Remove diagonal (self-similarity = 1.0)
        off_diagonal = similarity_matrix[~torch.eye(self.num_classes, dtype=bool)]
        
        stats = {
            'mean_similarity': off_diagonal.mean().item(),
            'max_similarity': off_diagonal.max().item(),
            'min_similarity': off_diagonal.min().item(),
            'std_similarity': off_diagonal.std().item(),
            'orthogonality_score': 1.0 - off_diagonal.abs().mean().item()  # Higher is better
        }
        
        return stats
    
    def save_encodings(self, filepath: str):
        """Save encodings to file."""
        torch.save({
            'class_encodings': self.class_encodings,
            'num_classes': self.num_classes,
            'encoding_dim': self.encoding_dim,
            'phase_bins': self.phase_bins,
            'mag_bins': self.mag_bins,
            'orthogonality_threshold': self.orthogonality_threshold
        }, filepath)
        print(f"💾 Encodings saved to {filepath}")
    
    def load_encodings(self, filepath: str):
        """Load encodings from file."""
        data = torch.load(filepath, map_location=self.device)
        self.class_encodings = data['class_encodings']
        self.num_classes = data['num_classes']
        self.encoding_dim = data['encoding_dim']
        self.phase_bins = data['phase_bins']
        self.mag_bins = data['mag_bins']
        self.orthogonality_threshold = data['orthogonality_threshold']
        print(f"📂 Encodings loaded from {filepath}")
    
    def _get_cache_path(self) -> str:
        """Get cache file path based on configuration."""
        import os
        cache_dir = "cache/encodings"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create unique filename based on parameters
        filename = f"orthogonal_{self.num_classes}c_{self.encoding_dim}d_{self.phase_bins}p_{self.mag_bins}m.pt"
        return os.path.join(cache_dir, filename)
    
    def _save_cached_encodings(self, filepath: str):
        """Save encodings to cache."""
        try:
            torch.save({
                'class_encodings': self.class_encodings,
                'num_classes': self.num_classes,
                'encoding_dim': self.encoding_dim,
                'phase_bins': self.phase_bins,
                'mag_bins': self.mag_bins,
                'orthogonality_threshold': self.orthogonality_threshold,
                'cache_version': '1.0'
            }, filepath)
        except Exception as e:
            print(f"⚠️  Failed to save cache: {e}")
    
    def _load_cached_encodings(self, filepath: str) -> bool:
        """Load encodings from cache if available and valid."""
        try:
            import os
            if not os.path.exists(filepath):
                return False
            
            data = torch.load(filepath, map_location=self.device)
            
            # Validate cache compatibility
            if (data.get('num_classes') != self.num_classes or
                data.get('encoding_dim') != self.encoding_dim or
                data.get('phase_bins') != self.phase_bins or
                data.get('mag_bins') != self.mag_bins):
                print(f"⚠️  Cache parameters mismatch, regenerating...")
                return False
            
            # Load cached data
            self.class_encodings = data['class_encodings']
            
            # Move tensors to correct device
            for class_id in self.class_encodings:
                phase_idx, mag_idx = self.class_encodings[class_id]
                self.class_encodings[class_id] = (
                    phase_idx.to(self.device),
                    mag_idx.to(self.device)
                )
            
            return True
            
        except Exception as e:
            print(f"⚠️  Failed to load cache: {e}")
            return False

class StructuredClassEncoder:
    """Alternative structured encoding using mathematical patterns."""
    
    def __init__(self, num_classes: int, encoding_dim: int,
                 phase_bins: int, mag_bins: int, device: str = 'cpu'):
        """Initialize structured encoder with mathematical patterns."""
        self.num_classes = num_classes
        self.encoding_dim = encoding_dim
        self.phase_bins = phase_bins
        self.mag_bins = mag_bins
        self.device = device
        
        self.class_encodings = self._generate_structured_encodings()
    
    def _generate_structured_encodings(self) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """Generate structured encodings using mathematical patterns."""
        encodings = {}
        
        for class_id in range(self.num_classes):
            # Use trigonometric patterns for structure
            phase_pattern = []
            mag_pattern = []
            
            for dim in range(self.encoding_dim):
                # Phase: use class-specific frequency patterns
                phase_val = (class_id * math.pi / self.num_classes + 
                           dim * 2 * math.pi / self.encoding_dim) % (2 * math.pi)
                phase_idx = int(phase_val / (2 * math.pi) * self.phase_bins)
                phase_pattern.append(min(phase_idx, self.phase_bins - 1))
                
                # Magnitude: use class-specific amplitude patterns
                mag_val = math.sin(class_id * math.pi / self.num_classes + dim * math.pi / 2)
                mag_normalized = (mag_val + 1) / 2  # [-1,1] -> [0,1]
                mag_idx = int(mag_normalized * self.mag_bins)
                mag_pattern.append(min(mag_idx, self.mag_bins - 1))
            
            encodings[class_id] = (
                torch.tensor(phase_pattern, dtype=torch.long, device=self.device),
                torch.tensor(mag_pattern, dtype=torch.long, device=self.device)
            )
        
        return encodings
    
    def get_class_encoding(self, class_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get encoding for specific class."""
        return self.class_encodings[class_id]

def create_class_encoder(encoding_type: str = "orthogonal", **kwargs) -> OrthogonalClassEncoder:
    """
    Factory function to create class encoders.
    
    Args:
        encoding_type: Type of encoder ("orthogonal" or "structured")
        **kwargs: Additional arguments for encoder
        
    Returns:
        Class encoder instance
    """
    if encoding_type == "orthogonal":
        return OrthogonalClassEncoder(**kwargs)
    elif encoding_type == "structured":
        return StructuredClassEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")
