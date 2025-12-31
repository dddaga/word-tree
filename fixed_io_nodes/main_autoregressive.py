"""
Autoregressive temporal training for GNN.
Sequential feeding with temporal continuity across timesteps.
"""
import torch
import yaml
import csv
import os
import time
from pathlib import Path
import numpy as np

from gradient_accumulator import GradientAccumulator
from nodestore import NodeStore
from full_model import initialize_model_and_nodestore
from input_adapter import LinearInputAdapter
from device_utils import get_best_device
from synthetic_data import SyntheticSignalDataset


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


class AutoregressiveTrainer:
    """
    Trains the GNN in an autoregressive manner:
    1. Warm-up: Feed first N timesteps (forward only)
    2. Training: Feed one step, predict next, compute loss, backprop
    3. Activations persist across timesteps within a sequence
    """
    
    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = device
        self.window_size = config['training']['window_size']
        
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
            temporal_decay=config['model'].get('temporal_decay', 0.9),
            device=device,
        )
        
        # Input adapter transforms window to graph input
        # Each timestep in window → one input node's features
        adapter_input_dim = self.window_size  # Window of timesteps
        adapter_output_dim = config['graph']['input_nodes'] * config['model']['vector_dim']
        
        self.input_adapter = LinearInputAdapter(
            input_dim=adapter_input_dim,
            output_dim=adapter_output_dim,
            hidden_dims=config['model'].get('adapter_hidden_dims', [64, 32]),
            dropout=config['model'].get('adapter_dropout', 0.1),
        ).to(device)
        
        # Initialize gradient accumulator
        self.accumulator = GradientAccumulator(
            node_store=self.node_store,
            lr=config['training']['lr'],
            batch_size=config['training']['batch_size'],
            momentum=config['training'].get('momentum', 0.9),
            verbose=True,
            device='cpu'  # Accumulator stays on CPU for Qdrant
        )
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.warmup_steps = config['training']['warmup_steps']
        
        # Setup logging
        self.log_path = config['system']['logging']['log_path']
        self._setup_logging()
        
        print("✅ Autoregressive trainer initialized")
    
    def _setup_logging(self):
        """Setup CSV logging."""
        path_obj = Path(self.log_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        self.log_file = open(self.log_path, mode='a', newline='')
        fieldnames = ['sequence_id', 'timestep', 'phase', 'loss', 'target']
        self.csv_writer = csv.DictWriter(self.log_file, fieldnames=fieldnames)
        
        if not os.path.isfile(self.log_path):
            self.csv_writer.writeheader()
            self.log_file.flush()
        
        print(f"📊 Logging to {self.log_path}")
    
    def process_window(self, window, compute_gradients=True):
        """
        Process a sliding window through the model.
        
        window: tensor of shape [window_size] containing timestep values
        compute_gradients: if False, run forward only (warm-up phase)
        
        Returns: model output (class predictions)
        
        IMPORTANT: Activations persist across windows within a sequence.
        They are only reset at the start of a new sequence.
        """
        # Reshape for adapter: [window_size] → [1, window_size]
        window_input = window.unsqueeze(0).to(self.device)
        
        # Transform through adapter to get phase values for graph input
        with torch.set_grad_enabled(compute_gradients):
            adapted = self.input_adapter(window_input)  # Shape: [1, input_nodes * vector_dim]
            
            # Forward through GNN
            # Activations PERSIST across windows (temporal continuity)
            out = self.model(adapted)
        
        return out
    
    def train_sequence(self, sequence, target_class, sequence_id):
        """
        Train on a single sequence using sliding windows.
        
        sequence: 1D tensor of timesteps
        target_class: class label for this sequence
        sequence_id: ID for logging
        """
        seq_len = len(sequence)
        
        # Calculate number of windows we can create
        num_windows = seq_len - self.window_size + 1
        
        if num_windows <= 0:
            print(f"  ⚠️  Sequence too short (len={seq_len}, window={self.window_size}), skipping")
            return float('nan')
        
        # PHASE 1: WARM-UP - Feed first N windows (forward only, no gradients)
        warmup_windows = min(self.warmup_steps, num_windows)
        print(f"  Warm-up phase: {warmup_windows} windows")
        
        for w in range(warmup_windows):
            window = sequence[w:w + self.window_size]
            _ = self.process_window(window, compute_gradients=False)
            # Activations persist across windows (temporal continuity)
        
        # PHASE 2: TRAINING - Feed remaining windows with gradient computation
        training_windows = num_windows - warmup_windows
        print(f"  Training phase: {training_windows} windows")
        
        total_loss = 0.0
        num_training_steps = 0
        
        for w in range(warmup_windows, num_windows):
            window = sequence[w:w + self.window_size]
            window_start = w
            window_end = w + self.window_size - 1
            
            # Sync weights from Qdrant before forward pass
            self.model.gnn.sync_weights()
            
            # Forward pass with gradient tracking
            out = self.process_window(window, compute_gradients=True)
            
            # Compute loss
            # out shape: [5] (logits for 5 classes)
            # target shape: [] (scalar class index)
            target = torch.tensor(target_class, device=self.device)
            loss = self.criterion(out.unsqueeze(0), target.unsqueeze(0))  # Add batch dimension
            
            # Validate loss
            if torch.isinf(loss) or torch.isnan(loss):
                print(f"    ⚠️  Invalid loss at window [{window_start}:{window_end}], skipping")
                self.csv_writer.writerow({
                    'sequence_id': sequence_id,
                    'timestep': window_start,
                    'phase': 'training',
                    'loss': float('nan'),
                    'target': target_class
                })
                continue
            
            # Backward pass
            # Use retain_graph=True because activations persist across windows
            # and we need the computation graph for subsequent windows
            loss.backward(retain_graph=True)
            
            # Extract gradients from GNN
            phase_grads, mag_grads = self.model.gnn.get_grads()
            
            # Validate gradients
            valid_grads = True
            for grad in list(phase_grads.values()) + list(mag_grads.values()):
                if grad is not None and (torch.isinf(grad).any() or torch.isnan(grad).any()):
                    valid_grads = False
                    break
            
            if not valid_grads:
                print(f"    ⚠️  Invalid gradients at window [{window_start}:{window_end}], skipping")
                self.csv_writer.writerow({
                    'sequence_id': sequence_id,
                    'timestep': window_start,
                    'phase': 'training',
                    'loss': loss.item(),
                    'target': target_class
                })
                continue
            
            # Send gradients to accumulator (move to CPU first)
            clean_phase_grads = {k: v.detach().cpu().clone() for k, v in phase_grads.items() if v is not None}
            clean_mag_grads = {k: v.detach().cpu().clone() for k, v in mag_grads.items() if v is not None}
            
            self.accumulator.receive_gradients(clean_phase_grads, clean_mag_grads)
            self.accumulator.step()  # Update weights if batch is full
            
            # Activations persist across windows within sequence
            
            # Log
            self.csv_writer.writerow({
                'sequence_id': sequence_id,
                'timestep': window_start,
                'phase': 'training',
                'loss': loss.item(),
                'target': target_class
            })
            self.log_file.flush()
            
            total_loss += loss.item()
            num_training_steps += 1
            
            print(f"    window [{window_start}:{window_end}]: loss={loss.item():.4f}")
        
        avg_loss = total_loss / num_training_steps if num_training_steps > 0 else float('nan')
        print(f"  ✅ Sequence {sequence_id} complete. Avg loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def train(self, dataset, num_sequences=None):
        """
        Train on dataset autoregressively.
        
        dataset: SyntheticSignalDataset
        num_sequences: number of sequences to train on (None = all)
        """
        print(f"\n{'='*60}")
        print(f"🚀 Starting autoregressive training")
        print(f"{'='*60}")
        print(f"Warm-up steps: {self.warmup_steps}")
        print(f"Batch size: {self.config['training']['batch_size']}")
        print(f"Learning rate: {self.config['training']['lr']}")
        print(f"{'='*60}\n")
        
        num_sequences = num_sequences or len(dataset)
        
        for seq_idx in range(num_sequences):
            sequence, target_class = dataset[seq_idx]
            
            print(f"\n📊 Processing sequence {seq_idx+1}/{num_sequences} (class {target_class})")
            
            try:
                avg_loss = self.train_sequence(sequence, target_class.item(), seq_idx)
            except Exception as e:
                print(f"❌ Error processing sequence {seq_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n{'='*60}")
        print(f"✅ Training complete!")
        print(f"{'='*60}\n")
    
    def cleanup(self):
        """Close log file."""
        self.log_file.close()


def main():
    # Load config
    config = load_config('configs/config_autoregressive.yaml')
    
    # Set device
    device = config['system']['device']
    if device == 'auto':
        device = get_best_device()
    print(f"🎯 Using device: {device}")
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create dataset
    print("📊 Creating synthetic signal dataset...")
    dataset = SyntheticSignalDataset(
        num_classes=config['data']['num_classes'],
        seq_length=config['data']['seq_length'],
        num_samples_per_class=config['data']['num_samples_per_class'],
        noise_level=config['data']['noise_level'],
    )
    print(f"✅ Dataset created: {len(dataset)} sequences")
    
    # Create trainer
    trainer = AutoregressiveTrainer(config, device=device)
    
    # Train
    try:
        trainer.train(dataset, num_sequences=config['training'].get('num_sequences', 10))
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()

