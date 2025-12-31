#!/usr/bin/env python3
"""
Training Framework for Neurograph Networks.

This module provides a flexible training system with:
- Multiple dataset types (patterns, sequences, classification)
- Configuration-based experiments
- Metrics tracking and logging
- Result export
"""

import json
import time
import math
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Callable, Any, Tuple
from pathlib import Path
import numpy as np

try:
    import requests
except ImportError:
    requests = None  # Will use direct import if available

# Try to import manager directly for faster training
try:
    from manager import VizSession, NetworkConfig
    DIRECT_MODE = True
except ImportError:
    DIRECT_MODE = False


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    epoch: int
    loss: float
    avg_gradient: float
    active_nodes: int
    output_phase_error: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class TrainingResult:
    """Complete training result."""
    config_name: str
    network_config: Dict
    training_config: Dict
    metrics_history: List[TrainingMetrics]
    final_loss: float
    total_epochs: int
    training_time: float
    converged: bool
    final_state: Optional[Dict] = None


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    type: str  # "pattern", "temporal", "classification", "sine", "custom"
    patterns: List[List[float]] = field(default_factory=list)
    targets: List[float] = field(default_factory=list)
    sequence_length: int = 10
    prediction_horizon: int = 1
    num_classes: int = 2
    noise_level: float = 0.0
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'DatasetConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass  
class ExperimentConfig:
    """Full experiment configuration."""
    name: str
    network: Dict
    training: Dict
    dataset: DatasetConfig
    
    @classmethod
    def from_dict(cls, name: str, d: Dict) -> 'ExperimentConfig':
        return cls(
            name=name,
            network=d.get('network', {}),
            training=d.get('training', {}),
            dataset=DatasetConfig.from_dict(d.get('dataset', {}))
        )


class Dataset:
    """Base class for datasets."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.data: List[Tuple[List[float], List[float]]] = []
        self._generate()
    
    def _generate(self):
        """Generate dataset - override in subclasses."""
        pass
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[List[float], List[float]]:
        return self.data[idx]
    
    def __iter__(self):
        return iter(self.data)


class PatternDataset(Dataset):
    """Dataset of fixed input-output patterns."""
    
    def _generate(self):
        patterns = self.config.patterns
        targets = self.config.targets
        
        if not patterns:
            # Default XOR pattern
            patterns = [[0, 0], [0, 1], [1, 0], [1, 1]]
            targets = [0, math.pi, math.pi, 0]
        
        for pattern, target in zip(patterns, targets):
            # Add noise if configured
            if self.config.noise_level > 0:
                pattern = [p + random.gauss(0, self.config.noise_level) for p in pattern]
            
            # Targets are phase values
            if isinstance(target, (int, float)):
                target = [target]
            
            self.data.append((pattern, target))


class TemporalDataset(Dataset):
    """Dataset for temporal sequence prediction."""
    
    def _generate(self):
        seq_len = self.config.sequence_length
        horizon = self.config.prediction_horizon
        
        # Generate random sequences
        for _ in range(100):  # 100 sequences
            sequence = [random.random() for _ in range(seq_len + horizon)]
            
            # Input: first seq_len elements
            # Target: next horizon elements (as phase)
            for i in range(seq_len - horizon):
                inputs = sequence[i:i+seq_len]
                targets = [v * 2 * math.pi for v in sequence[i+seq_len:i+seq_len+horizon]]
                self.data.append((inputs, targets))


class SineDataset(Dataset):
    """Dataset for sine wave approximation."""
    
    def _generate(self):
        # Generate sine wave samples
        for _ in range(200):
            x = random.uniform(0, 2 * math.pi)
            y = math.sin(x)
            
            # Input: x value (normalized)
            # Target: y value (as phase)
            inputs = [x / (2 * math.pi)]  # Normalize to [0, 1]
            targets = [(y + 1) * math.pi]  # Map [-1, 1] to [0, 2π]
            
            self.data.append((inputs, targets))


class ClassificationDataset(Dataset):
    """Dataset for multi-class classification."""
    
    def _generate(self):
        num_classes = self.config.num_classes
        patterns_per_class = 50
        
        for class_idx in range(num_classes):
            # Generate clustered patterns for each class
            center = [random.random() for _ in range(4)]  # 4-dimensional input
            
            for _ in range(patterns_per_class):
                # Add noise around center
                pattern = [c + random.gauss(0, 0.2) for c in center]
                
                # Target: class encoded as phase
                target_phase = (class_idx / num_classes) * 2 * math.pi
                
                self.data.append((pattern, [target_phase]))
        
        # Shuffle
        random.shuffle(self.data)


def create_dataset(config: DatasetConfig) -> Dataset:
    """Factory function to create datasets."""
    dataset_types = {
        'pattern': PatternDataset,
        'temporal': TemporalDataset,
        'sine': SineDataset,
        'classification': ClassificationDataset,
    }
    
    dataset_class = dataset_types.get(config.type, PatternDataset)
    return dataset_class(config)


class NeurographTrainer:
    """Trainer for Neurograph networks."""
    
    def __init__(self, base_url: str = "http://localhost:8765", use_direct: bool = True):
        self.base_url = base_url
        self.use_direct = use_direct and DIRECT_MODE
        self.session = None
        
        if self.use_direct:
            self.session = VizSession()
        elif requests is None:
            raise ImportError("requests library required for HTTP mode")
    
    def init_network(self, config: Dict) -> Dict:
        """Initialize network."""
        if self.use_direct:
            from manager import NetworkConfig
            cfg = NetworkConfig(**config)
            self.session.network = None
            return asdict(self.session.init_network(config))
        else:
            response = requests.post(f"{self.base_url}/api/init", json=config)
            response.raise_for_status()
            return response.json()
    
    def step_forward(self) -> Dict:
        """Execute forward step."""
        if self.use_direct:
            result = self.session.step_forward()
            return self._convert_result(result)
        else:
            response = requests.post(f"{self.base_url}/api/step")
            response.raise_for_status()
            return response.json()
    
    def step_backward(self) -> Dict:
        """Execute backward step."""
        if self.use_direct:
            result = self.session.step_backward()
            return self._convert_result(result)
        else:
            response = requests.post(f"{self.base_url}/api/backprop")
            response.raise_for_status()
            return response.json()
    
    def inject_sequence(self, sequence: List[float], timestep: int = 0) -> Dict:
        """Inject temporal sequence."""
        if self.use_direct:
            self.session.inject_temporal_sequence(sequence, timestep)
            return {"status": "injected", "timestep": timestep}
        else:
            response = requests.post(
                f"{self.base_url}/api/inject_sequence",
                json={"sequence": sequence, "timestep": timestep}
            )
            response.raise_for_status()
            return response.json()
    
    def _convert_result(self, result) -> Dict:
        """Convert StepResult to dict."""
        if hasattr(result, '__dict__'):
            return {
                'nodes': result.nodes,
                'edges': result.edges,
                'active_signals': result.active_signals,
                'radiation_paths': result.radiation_paths,
                'step_number': result.step_number,
                'mode': result.mode
            }
        return result
    
    def compute_loss(self, state: Dict) -> float:
        """Compute phase error loss."""
        total_error = 0.0
        count = 0
        
        for node_id, node in state["nodes"].items():
            if node["role"] == "output" and node.get("target_phase") is not None:
                error = abs(node["target_phase"] - node["phase"])
                # Circular distance
                if error > math.pi:
                    error = 2 * math.pi - error
                total_error += error
                count += 1
        
        return total_error / count if count > 0 else 0.0
    
    def get_active_count(self, state: Dict) -> int:
        """Count active nodes."""
        return sum(1 for n in state["nodes"].values() if n["active"])
    
    def get_avg_gradient(self, state: Dict) -> float:
        """Get average gradient magnitude."""
        grads = [abs(n["gradient"]) for n in state["nodes"].values()]
        return np.mean(grads) if grads else 0.0
    
    def train_epoch(self, dataset: Dataset, timestep_base: int = 0) -> TrainingMetrics:
        """Train one epoch over dataset."""
        total_loss = 0.0
        total_grad = 0.0
        total_active = 0
        total_error = 0.0
        
        for i, (inputs, targets) in enumerate(dataset):
            # Forward pass
            self.inject_sequence(inputs, timestep=timestep_base + i)
            state = self.step_forward()
            
            # Compute metrics
            loss = self.compute_loss(state)
            total_loss += loss
            total_active += self.get_active_count(state)
            
            # Backward pass
            state = self.step_backward()
            total_grad += self.get_avg_gradient(state)
            
            # Output phase error
            for node in state["nodes"].values():
                if node["role"] == "output":
                    total_error += abs(node.get("gradient", 0))
        
        n = len(dataset)
        return TrainingMetrics(
            epoch=0,  # Set by caller
            loss=total_loss / n,
            avg_gradient=total_grad / n,
            active_nodes=total_active // n,
            output_phase_error=total_error / n
        )
    
    def train(self, 
              experiment: ExperimentConfig,
              epochs: int = None,
              early_stopping_patience: int = 10,
              min_improvement: float = 0.001,
              verbose: bool = True,
              callback: Callable[[int, TrainingMetrics], None] = None) -> TrainingResult:
        """
        Train network on experiment configuration.
        
        Args:
            experiment: Full experiment configuration
            epochs: Number of epochs (overrides config)
            early_stopping_patience: Stop if no improvement for N epochs
            min_improvement: Minimum loss decrease to count as improvement
            verbose: Print progress
            callback: Called after each epoch with (epoch, metrics)
        
        Returns:
            TrainingResult with full history
        """
        # Get training parameters
        training_config = experiment.training
        epochs = epochs or training_config.get('epochs', 100)
        learning_rate = training_config.get('learning_rate', 0.05)
        
        # Initialize network
        network_config = {**experiment.network, 'learning_rate': learning_rate}
        self.init_network(network_config)
        
        # Create dataset
        dataset = create_dataset(experiment.dataset)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training: {experiment.name}")
            print(f"{'='*60}")
            print(f"Network: {len(dataset)} samples, {epochs} epochs")
            print(f"Learning rate: {learning_rate}")
        
        # Training loop
        start_time = time.time()
        metrics_history = []
        best_loss = float('inf')
        patience_counter = 0
        converged = False
        
        for epoch in range(epochs):
            metrics = self.train_epoch(dataset, timestep_base=epoch * len(dataset))
            metrics.epoch = epoch + 1
            metrics_history.append(metrics)
            
            # Check improvement
            if metrics.loss < best_loss - min_improvement:
                best_loss = metrics.loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch + 1}")
                converged = True
                break
            
            # Progress output
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"  Epoch {epoch+1:4d}: Loss={metrics.loss:.6f}, "
                      f"Grad={metrics.avg_gradient:.5f}, Active={metrics.active_nodes}")
            
            # Callback
            if callback:
                callback(epoch, metrics)
        
        training_time = time.time() - start_time
        
        # Get final state
        final_state = self.step_forward() if self.use_direct else None
        
        result = TrainingResult(
            config_name=experiment.name,
            network_config=network_config,
            training_config=training_config,
            metrics_history=metrics_history,
            final_loss=metrics_history[-1].loss if metrics_history else 0.0,
            total_epochs=len(metrics_history),
            training_time=training_time,
            converged=converged,
            final_state=final_state
        )
        
        if verbose:
            print(f"\nTraining complete in {training_time:.2f}s")
            print(f"Final loss: {result.final_loss:.6f}")
            print(f"Converged: {converged}")
        
        return result
    
    def compare_configs(self, 
                       experiments: List[ExperimentConfig],
                       epochs: int = 50,
                       verbose: bool = True) -> List[TrainingResult]:
        """Compare multiple configurations."""
        results = []
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Comparing {len(experiments)} configurations")
            print(f"{'='*60}")
        
        for exp in experiments:
            result = self.train(exp, epochs=epochs, verbose=verbose)
            results.append(result)
        
        # Summary
        if verbose:
            print(f"\n{'='*60}")
            print("Summary")
            print(f"{'='*60}")
            print(f"{'Config':<20} {'Final Loss':<12} {'Epochs':<8} {'Time':<10}")
            print("-" * 50)
            for r in results:
                print(f"{r.config_name:<20} {r.final_loss:<12.6f} {r.total_epochs:<8} {r.training_time:.2f}s")
        
        return results


def export_results(results: List[TrainingResult], path: str, format: str = 'json'):
    """Export training results."""
    path = Path(path)
    
    if format == 'json':
        data = []
        for r in results:
            data.append({
                'config_name': r.config_name,
                'network_config': r.network_config,
                'training_config': r.training_config,
                'final_loss': r.final_loss,
                'total_epochs': r.total_epochs,
                'training_time': r.training_time,
                'converged': r.converged,
                'metrics_history': [
                    {
                        'epoch': m.epoch,
                        'loss': m.loss,
                        'avg_gradient': m.avg_gradient,
                        'active_nodes': m.active_nodes,
                        'output_phase_error': m.output_phase_error
                    }
                    for m in r.metrics_history
                ]
            })
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    elif format == 'csv':
        import csv
        
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['config', 'epoch', 'loss', 'avg_gradient', 'active_nodes'])
            
            for r in results:
                for m in r.metrics_history:
                    writer.writerow([
                        r.config_name, m.epoch, m.loss, m.avg_gradient, m.active_nodes
                    ])
    
    print(f"Results exported to {path}")


# Predefined experiment configurations
PRESET_EXPERIMENTS = {
    'xor': ExperimentConfig(
        name='XOR Pattern',
        network={
            'num_input': 2,
            'num_input_connected': 4,
            'num_middle': 6,
            'num_output': 1,
            'vector_dim': 8,
            'beam_width': 10
        },
        training={
            'epochs': 100,
            'learning_rate': 0.05
        },
        dataset=DatasetConfig(
            type='pattern',
            patterns=[[0, 0], [0, 1], [1, 0], [1, 1]],
            targets=[0, math.pi, math.pi, 0]
        )
    ),
    
    'sine': ExperimentConfig(
        name='Sine Approximation',
        network={
            'num_input': 1,
            'num_input_connected': 4,
            'num_middle': 8,
            'num_output': 1,
            'vector_dim': 16
        },
        training={
            'epochs': 200,
            'learning_rate': 0.03
        },
        dataset=DatasetConfig(type='sine')
    ),
    
    'temporal': ExperimentConfig(
        name='Temporal Sequence',
        network={
            'num_input': 5,
            'num_input_connected': 8,
            'num_middle': 12,
            'num_output': 2,
            'vector_dim': 16,
            'beam_width': 20
        },
        training={
            'epochs': 50,
            'learning_rate': 0.03
        },
        dataset=DatasetConfig(
            type='temporal',
            sequence_length=5,
            prediction_horizon=1
        )
    ),
    
    'classification': ExperimentConfig(
        name='3-Class Classification',
        network={
            'num_input': 4,
            'num_input_connected': 6,
            'num_middle': 10,
            'num_output': 3,
            'vector_dim': 8
        },
        training={
            'epochs': 100,
            'learning_rate': 0.05
        },
        dataset=DatasetConfig(
            type='classification',
            num_classes=3
        )
    ),
    
    'small_fast': ExperimentConfig(
        name='Small Fast Network',
        network={
            'num_input': 2,
            'num_input_connected': 2,
            'num_middle': 4,
            'num_output': 1,
            'vector_dim': 4,
            'beam_width': 5
        },
        training={
            'epochs': 50,
            'learning_rate': 0.1
        },
        dataset=DatasetConfig(
            type='pattern',
            patterns=[[0, 0], [1, 1]],
            targets=[0, math.pi]
        )
    ),
    
    'large_complex': ExperimentConfig(
        name='Large Complex Network',
        network={
            'num_input': 4,
            'num_input_connected': 8,
            'num_middle': 16,
            'num_output': 4,
            'vector_dim': 32,
            'beam_width': 30,
            'radiation_k': 5
        },
        training={
            'epochs': 200,
            'learning_rate': 0.02
        },
        dataset=DatasetConfig(
            type='classification',
            num_classes=4
        )
    )
}


def get_preset(name: str) -> ExperimentConfig:
    """Get a preset experiment configuration."""
    if name not in PRESET_EXPERIMENTS:
        available = ', '.join(PRESET_EXPERIMENTS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return PRESET_EXPERIMENTS[name]


def list_presets() -> List[str]:
    """List available preset names."""
    return list(PRESET_EXPERIMENTS.keys())


if __name__ == "__main__":
    # Quick test
    trainer = NeurographTrainer(use_direct=True)
    
    # Run XOR experiment
    xor_config = get_preset('xor')
    result = trainer.train(xor_config, epochs=50, verbose=True)
    
    print(f"\nXOR Training Result:")
    print(f"  Final Loss: {result.final_loss:.6f}")
    print(f"  Epochs: {result.total_epochs}")
    print(f"  Time: {result.training_time:.2f}s")

