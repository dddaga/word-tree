# input_adapters.py

import torch
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
import numpy as np

class MNISTPCAAdapter:
    def __init__(self, vector_dim, num_input_nodes, phase_bins, mag_bins, device='cpu'):
        """
        Converts MNIST images to input_context dict for NeuroGraph.
        
        Args:
            vector_dim (int): Dimensionality of phase/mag vectors per node
            num_input_nodes (int): Number of input nodes
            phase_bins (int): Number of discrete phase bins
            mag_bins (int): Number of discrete magnitude bins
        """
        self.vector_dim = vector_dim
        self.num_input_nodes = num_input_nodes
        self.total_dim = 2 * vector_dim * num_input_nodes
        self.phase_bins = phase_bins
        self.mag_bins = mag_bins
        self.device = device

        # Load MNIST as flat 784 vectors
        self.mnist = datasets.MNIST(
            root="data", train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(-1))  # flatten: [784]
            ])
        )

        # Fit PCA on full dataset (first time only)
        X = torch.stack([self.mnist[i][0] for i in range(len(self.mnist))])  # shape: [N, 784]
        pca = PCA(n_components=self.total_dim)
        self.pca_data = pca.fit_transform(X.numpy())  # [N, total_dim]

    def encode_to_phase_mag(self, vec):
        """
        Quantizes a vector into discrete phase and magnitude indices.
        """
        vec = (vec - vec.min()) / (vec.max() - vec.min() + 1e-6)
        phase_raw, mag_raw = np.split(vec, 2)

        phase_idx = torch.floor(torch.tensor(phase_raw * self.phase_bins)).long().clamp(0, self.phase_bins - 1)
        mag_idx   = torch.floor(torch.tensor(mag_raw * self.mag_bins)).long().clamp(0, self.mag_bins - 1)

        return phase_idx.to(self.device), mag_idx.to(self.device)

    def get_input_context(self, idx, input_node_ids):
        """
        Returns: dict[node_id → (phase_idx [D], mag_idx [D])]
        """
        vec = self.pca_data[idx]  # [total_dim]
        phase_idx, mag_idx = self.encode_to_phase_mag(vec)

        input_context = {}
        for i, node_id in enumerate(input_node_ids):
            p = phase_idx[i * self.vector_dim: (i + 1) * self.vector_dim]
            m = mag_idx[i * self.vector_dim: (i + 1) * self.vector_dim]
            input_context[node_id] = (p, m)

        return input_context, self.mnist[idx][1]  # also return label
