"""
Single sample training for GNN on MNIST.
Trains repeatedly on the same sample to verify learning dynamics.
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import yaml
import csv
import os
import time
from pathlib import Path
import numpy as np

os.environ["PYTHONWARNINGS"] = "ignore"

from gradient_accumulator import GradientAccumulator
from nodestore import NodeStore
from full_model import initialize_model_and_nodestore
from device_utils import get_best_device


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


class SingleSampleTrainer:
    """
    Trains repeatedly on a single MNIST sample.
    Useful for verifying gradient flow and convergence.
    """
    
    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = device
        
        # Initialize model and node store
        print(f"Initializing model on {device}...")
        self.model, self.node_store = initialize_model_and_nodestore(
            qdrant_url=config['qdrant']['url'],
            collection_name=config['qdrant']['collection_name'],
            total_nodes=config['graph']['total_nodes'],
            input_nodes=config['graph']['input_nodes'],
            output_nodes=config['graph']['output_nodes'],
            cardinality=config['graph']['cardinality'],
            radiation_targets=config['graph']['radiation_targets'],
            vector_dim=config['model']['vector_dim'],
            phase_bins=config['model'].get('phase_bins'),
            mag_bins=config['model'].get('mag_bins'),
            iterations=config['model']['iterations'],
            activation_threshold=config['model']['activation_threshold'],
            gamma=config['model']['gamma'],
            temporal_decay=config['model'].get('temporal_decay', 1.0),
            device=device,
        )
        
        # Initialize gradient accumulator
        self.accumulator = GradientAccumulator(
            node_store=self.node_store,
            lr=config['training']['lr'],
            batch_size=config['training'].get('batch_size', 1),
            momentum=config['training'].get('momentum', 0.9),
            verbose=True,
            device='cpu'  # Accumulator stays on CPU for Qdrant
        )
        
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # Setup logging
        self.log_path = config['system']['logging']['log_path']
        self._setup_logging()
        
        print("✅ Single sample trainer initialized")
    
    def _setup_logging(self):
        """Setup CSV logging."""
        path_obj = Path(self.log_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file exists
        file_exists = os.path.isfile(self.log_path)
        
        self.log_file = open(self.log_path, mode='a', newline='')
        fieldnames = ['step', 'loss', 'target', 'prediction', 'correct']
        self.csv_writer = csv.DictWriter(self.log_file, fieldnames=fieldnames)
        
        if not file_exists:
            self.csv_writer.writeheader()
            self.log_file.flush()
        
        print(f"📊 Logging to {self.log_path}")
    
    def train_step(self, data, target, step):
        """
        Execute one training step.
        """
        # Move data to device
        data = data.to(self.device)
        target = target.to(self.device)
        
        # Sync weights from Qdrant before forward pass
        self.model.gnn.sync_weights()
        
        # Forward pass
        out = self.model(data)
        loss = self.criterion(out.unsqueeze(0), target.unsqueeze(0))
        
        # Get prediction
        prediction = out.argmax().item()
        correct = prediction == target.item()
        
        # Validate loss before backward pass
        if torch.isinf(loss) or torch.isnan(loss):
            print(f"  ⚠️  Step {step}: Invalid loss ({loss.item()}), skipping")
            self.model.reset_activations()
            return float('nan'), prediction, correct
        
        # Backward pass
        loss.backward()
        
        # Extract gradients
        phase_grads, mag_grads = self.model.gnn.get_grads()
        
        # Clip and validate gradients
        max_grad_norm = 1.0  # Gradient clipping threshold
        valid_grads = True
        
        for grad in list(phase_grads.values()) + list(mag_grads.values()):
            if grad is not None:
                if torch.isinf(grad).any() or torch.isnan(grad).any():
                    valid_grads = False
                    break
        
        if not valid_grads:
            print(f"  ⚠️  Step {step}: Invalid gradients (NaN/Inf), skipping")
            self.model.reset_activations()
            return loss.item(), prediction, correct
        
        # Clip gradients to prevent explosion
        for grad in list(phase_grads.values()) + list(mag_grads.values()):
            if grad is not None:
                torch.nn.utils.clip_grad_norm_([grad], max_grad_norm)
        
        # Send gradients to accumulator (move to CPU first)
        clean_phase_grads = {k: v.detach().cpu().clone() for k, v in phase_grads.items() if v is not None}
        clean_mag_grads = {k: v.detach().cpu().clone() for k, v in mag_grads.items() if v is not None}
        
        self.accumulator.receive_gradients(clean_phase_grads, clean_mag_grads)
        self.accumulator.step()  # Update weights if batch is full
        
        # Reset activations for next step
        self.model.reset_activations()
        
        return loss.item(), prediction, correct
    
    def train(self, data, target, num_steps=1000, log_interval=10):
        """
        Train repeatedly on the same sample.
        """
        print(f"\n{'='*60}")
        print(f"🚀 Starting single sample training")
        print(f"{'='*60}")
        print(f"Target class: {target.item()}")
        print(f"Learning rate: {self.config['training']['lr']}")
        print(f"Iterations per forward: {self.config['model']['iterations']}")
        print(f"{'='*60}\n")
        
        losses = []
        correct_count = 0
        
        for step in range(num_steps):
            loss, prediction, correct = self.train_step(data, target, step)
            losses.append(loss)
            if correct:
                correct_count += 1
            
            # Log
            self.csv_writer.writerow({
                'step': step,
                'loss': loss,
                'target': target.item(),
                'prediction': prediction,
                'correct': correct
            })
            
            if step % log_interval == 0:
                self.log_file.flush()
                recent_losses = [l for l in losses[-log_interval:] if not np.isnan(l)]
                avg_loss = np.mean(recent_losses) if recent_losses else float('nan')
                recent_acc = sum(1 for l in losses[-log_interval:] if not np.isnan(l)) / log_interval * 100
                print(f"Step {step:4d}: loss={loss:.4f}, avg_loss={avg_loss:.4f}, pred={prediction}, target={target.item()}, correct={correct}")
        
        # Final statistics
        valid_losses = [l for l in losses if not np.isnan(l)]
        final_avg_loss = np.mean(valid_losses[-100:]) if len(valid_losses) >= 100 else np.mean(valid_losses)
        accuracy = correct_count / num_steps * 100
        
        print(f"\n{'='*60}")
        print(f"✅ Training complete!")
        print(f"Final avg loss (last 100): {final_avg_loss:.4f}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"{'='*60}\n")
        
        return losses
    
    def cleanup(self):
        """Close log file."""
        self.log_file.close()


def main():
    # Load config
    config = load_config('configs/single_sample_config.yaml')
    
    # Set device
    device = config['system']['device']
    if device == 'auto':
        device = get_best_device()
    print(f"🎯 Using device: {device}")
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load MNIST and get a single sample
    print("📊 Loading MNIST dataset...")
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((7, 7)),  # Downsample to 7x7 (49 pixels)
        transforms.Lambda(lambda x: x.flatten()),
    ])
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transformations)
    
    # Get the first sample
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    sample, target = next(iter(dataloader))
    sample = sample.squeeze()  # Remove batch dimension
    target = target.squeeze()
    
    print(f"✅ Loaded sample: shape={sample.shape}, target={target.item()}")
    
    # Create trainer
    trainer = SingleSampleTrainer(config, device=device)
    
    # Train
    try:
        trainer.train(sample, target, num_steps=500, log_interval=10)
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
