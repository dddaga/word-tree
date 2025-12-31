"""
Synthetic time-series data generator for testing the GNN architecture.
Generates combinations of sine and cosine waves with different frequencies.
"""
import torch
import numpy as np
from typing import Tuple, List


class SyntheticSignalDataset:
    """
    Generate synthetic signals as combinations of sine and cosine waves.
    Each signal class has a unique frequency combination.
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        seq_length: int = 50,
        num_samples_per_class: int = 200,
        noise_level: float = 0.1,
        seed: int = 42
    ):
        """
        num_classes: Number of different signal types
        seq_length: Length of each signal sequence
        num_samples_per_class: Number of samples per signal class
        noise_level: Standard deviation of Gaussian noise to add
        seed: Random seed for reproducibility
        """
        self.num_classes = num_classes
        self.seq_length = seq_length
        self.num_samples_per_class = num_samples_per_class
        self.noise_level = noise_level
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Define frequency combinations for each class
        # Each class is a different combination of sine/cosine frequencies
        self.signal_configs = self._generate_signal_configs()
        
        # Pre-generate all data
        self.data, self.labels = self._generate_dataset()
        
    def _generate_signal_configs(self) -> List[dict]:
        """Generate unique frequency configurations for each signal class."""
        configs = []
        for i in range(self.num_classes):
            config = {
                'freq1': 0.5 + i * 0.3,  # Varying frequencies
                'freq2': 1.0 + i * 0.4,
                'phase1': np.random.uniform(0, 2*np.pi),
                'phase2': np.random.uniform(0, 2*np.pi),
                'weight1': np.random.uniform(0.5, 1.0),
                'weight2': np.random.uniform(0.5, 1.0),
            }
            configs.append(config)
        return configs
    
    def _generate_signal(self, class_idx: int) -> np.ndarray:
        """Generate a single signal instance for a given class."""
        config = self.signal_configs[class_idx]
        
        t = np.linspace(0, 4*np.pi, self.seq_length)
        
        # Combination of sine and cosine with different frequencies
        signal = (
            config['weight1'] * np.sin(config['freq1'] * t + config['phase1']) +
            config['weight2'] * np.cos(config['freq2'] * t + config['phase2'])
        )
        
        # Normalize to [-1, 1] range
        signal = signal / (config['weight1'] + config['weight2'])
        
        # Add noise
        if self.noise_level > 0:
            signal += np.random.normal(0, self.noise_level, self.seq_length)
        
        return signal
    
    def _generate_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate the complete dataset."""
        all_signals = []
        all_labels = []
        
        for class_idx in range(self.num_classes):
            for _ in range(self.num_samples_per_class):
                signal = self._generate_signal(class_idx)
                all_signals.append(signal)
                all_labels.append(class_idx)
        
        # Convert to tensors and shuffle
        signals = torch.tensor(np.array(all_signals), dtype=torch.float32)
        labels = torch.tensor(all_labels, dtype=torch.long)
        
        # Shuffle
        indices = torch.randperm(len(signals))
        signals = signals[indices]
        labels = labels[indices]
        
        return signals, labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def get_sample_signals(self, num_per_class: int = 1) -> dict:
        """Get sample signals from each class for visualization."""
        samples = {}
        for class_idx in range(self.num_classes):
            class_samples = []
            for _ in range(num_per_class):
                signal = self._generate_signal(class_idx)
                class_samples.append(signal)
            samples[f'class_{class_idx}'] = np.array(class_samples)
        return samples


class TemporalSignalDataset:
    """
    Temporal dataset that feeds signals sequentially, timestep by timestep.
    """
    
    def __init__(
        self,
        base_dataset: SyntheticSignalDataset,
        input_window: int = 10,
        predict_ahead: int = 1
    ):
        """
        base_dataset: The base synthetic signal dataset
        input_window: Number of past timesteps to use as input
        predict_ahead: Number of timesteps ahead to predict
        """
        self.base_dataset = base_dataset
        self.input_window = input_window
        self.predict_ahead = predict_ahead
        
        # Create temporal windows
        self.temporal_data, self.temporal_labels = self._create_temporal_windows()
    
    def _create_temporal_windows(self):
        """Create sliding windows over the signals."""
        all_windows = []
        all_labels = []
        
        for signal, label in zip(self.base_dataset.data, self.base_dataset.labels):
            # Create sliding windows
            for i in range(len(signal) - self.input_window - self.predict_ahead + 1):
                window = signal[i:i + self.input_window]
                all_windows.append(window)
                all_labels.append(label)
        
        return torch.stack(all_windows), torch.tensor(all_labels)
    
    def __len__(self):
        return len(self.temporal_data)
    
    def __getitem__(self, idx):
        return self.temporal_data[idx], self.temporal_labels[idx]


if __name__ == "__main__":
    # Example usage
    print("Creating synthetic signal dataset...")
    dataset = SyntheticSignalDataset(
        num_classes=5,
        seq_length=50,
        num_samples_per_class=200,
        noise_level=0.1
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Signal shape: {dataset.data.shape}")
    print(f"Labels shape: {dataset.labels.shape}")
    
    # Get a sample
    signal, label = dataset[0]
    print(f"\nSample signal shape: {signal.shape}")
    print(f"Sample label: {label}")
    print(f"Signal range: [{signal.min():.3f}, {signal.max():.3f}]")
    
    # Create temporal dataset
    print("\nCreating temporal dataset...")
    temporal_dataset = TemporalSignalDataset(
        dataset, 
        input_window=10, 
        predict_ahead=1
    )
    print(f"Temporal dataset size: {len(temporal_dataset)}")
    window, label = temporal_dataset[0]
    print(f"Window shape: {window.shape}")
    print(f"Window label: {label}")


