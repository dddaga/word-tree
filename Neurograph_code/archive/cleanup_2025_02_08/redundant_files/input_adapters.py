# input_adapters_1000.py
# Enhanced input adapter for 1000-node network with 200 input nodes

import torch
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
import numpy as np

class MNISTPCAAdapter1000:
    def __init__(self, vector_dim, num_input_nodes, phase_bins, mag_bins, device='cpu'):
        """
        Enhanced MNIST adapter for 1000-node network with 200 input nodes.
        Uses PCA with higher dimensions (2000) for richer feature representation.
        
        Args:
            vector_dim (int): Dimensionality of phase/mag vectors per node (typically 5)
            num_input_nodes (int): Number of input nodes (200 for large network)
            phase_bins (int): Number of discrete phase bins
            mag_bins (int): Number of discrete magnitude bins
        """
        self.vector_dim = vector_dim
        self.num_input_nodes = num_input_nodes
        # Calculate total dimensions needed, but cap at available features
        desired_dim = 2 * vector_dim * num_input_nodes
        # MNIST has 784 features, so we can't extract more than 784 PCA components
        self.total_dim = min(desired_dim, 784)
        self.phase_bins = phase_bins
        self.mag_bins = mag_bins
        self.device = device

        print(f"🔧 Initializing Enhanced MNIST Adapter:")
        print(f"   📊 Input nodes: {num_input_nodes}")
        print(f"   📐 Vector dim per node: {vector_dim}")
        print(f"   🎯 Total PCA dimensions: {self.total_dim}")
        print(f"   📈 Feature capacity increase: {self.total_dim/50:.1f}x vs previous")

        # Load MNIST as flat 784 vectors
        self.mnist = datasets.MNIST(
            root="data", train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(-1))  # flatten: [784]
            ])
        )

        print(f"📚 Loading MNIST dataset: {len(self.mnist)} samples")
        
        # Fit PCA on full dataset with much higher dimensions
        print(f"🔄 Computing PCA: 784 → {self.total_dim} dimensions...")
        X = torch.stack([self.mnist[i][0] for i in range(len(self.mnist))])  # shape: [N, 784]
        
        # Use higher n_components for richer feature extraction
        pca = PCA(n_components=self.total_dim)
        self.pca_data = pca.fit_transform(X.numpy())  # [N, total_dim]
        
        # Report PCA statistics
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        print(f"✅ PCA completed:")
        print(f"   📊 Explained variance (first 100 components): {cumulative_variance[99]:.3f}")
        print(f"   📊 Explained variance (first 500 components): {cumulative_variance[499]:.3f}")
        print(f"   📊 Explained variance (all {self.total_dim} components): {cumulative_variance[-1]:.3f}")
        print(f"   🎯 Feature richness: {self.total_dim} vs previous 50 dimensions")

    def encode_to_phase_mag(self, vec):
        """
        Quantizes a vector into discrete phase and magnitude indices.
        Enhanced for larger dimensional vectors.
        """
        # Normalize to [0, 1] range
        vec_min, vec_max = vec.min(), vec.max()
        if vec_max > vec_min:
            vec = (vec - vec_min) / (vec_max - vec_min)
        else:
            vec = np.zeros_like(vec)  # Handle constant vectors
        
        # Split into phase and magnitude components
        mid_point = len(vec) // 2
        phase_raw = vec[:mid_point]
        mag_raw = vec[mid_point:]

        # Quantize to discrete bins
        phase_idx = torch.floor(torch.tensor(phase_raw * self.phase_bins, device=self.device)).long().clamp(0, self.phase_bins - 1)
        mag_idx = torch.floor(torch.tensor(mag_raw * self.mag_bins, device=self.device)).long().clamp(0, self.mag_bins - 1)

        return phase_idx, mag_idx

    def get_input_context(self, idx, input_node_ids):
        """
        Returns input context for 200 input nodes with rich PCA features.
        
        Returns: 
            input_context: dict[node_id → (phase_idx [vector_dim], mag_idx [vector_dim])]
            label: int (digit class 0-9)
        """
        if idx >= len(self.pca_data):
            raise IndexError(f"Index {idx} out of range for dataset size {len(self.pca_data)}")
        
        vec = self.pca_data[idx]  # [total_dim = 784]
        phase_idx, mag_idx = self.encode_to_phase_mag(vec)

        # Distribute across input nodes
        # We have total_dim/2 elements each for phase and magnitude
        # Need to distribute across num_input_nodes, each getting vector_dim elements
        phase_per_node = len(phase_idx) // self.num_input_nodes
        mag_per_node = len(mag_idx) // self.num_input_nodes
        
        input_context = {}
        for i, node_id in enumerate(input_node_ids):
            if i >= self.num_input_nodes:
                break
                
            # Calculate indices for this node
            phase_start = i * phase_per_node
            phase_end = min((i + 1) * phase_per_node, len(phase_idx))
            mag_start = i * mag_per_node  
            mag_end = min((i + 1) * mag_per_node, len(mag_idx))
            
            # Get the slice for this node
            p_slice = phase_idx[phase_start:phase_end]
            m_slice = mag_idx[mag_start:mag_end]
            
            # Pad or truncate to exactly vector_dim elements
            if len(p_slice) < self.vector_dim:
                # Pad with zeros if needed
                p_pad = torch.zeros(self.vector_dim - len(p_slice), dtype=p_slice.dtype, device=self.device)
                m_pad = torch.zeros(self.vector_dim - len(m_slice), dtype=m_slice.dtype, device=self.device)
                p = torch.cat([p_slice, p_pad])
                m = torch.cat([m_slice, m_pad])
            else:
                # Truncate if needed
                p = p_slice[:self.vector_dim]
                m = m_slice[:self.vector_dim]
            
            input_context[node_id] = (p, m)

        return input_context, self.mnist[idx][1]  # return context and label

    def get_dataset_info(self):
        """Returns information about the dataset and PCA transformation."""
        return {
            'dataset_size': len(self.mnist),
            'original_dims': 784,
            'pca_dims': self.total_dim,
            'input_nodes': self.num_input_nodes,
            'vector_dim_per_node': self.vector_dim,
            'capacity_increase': self.total_dim / 50  # vs previous 50 dims
        }

# Backward compatibility alias
MNISTPCAAdapter = MNISTPCAAdapter1000
