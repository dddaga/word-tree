# train_context_1000.py
# Training pipeline for 1000-node network with 200 input nodes

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import random
from utils.config import load_config
from core.graph import load_or_build_graph
from core.node_store import NodeStore
from core.cell import PhaseCell
from core.tables import ExtendedLookupTableModule
from core.forward_engine import run_enhanced_forward
from core.backward import backward_pass
from modules.input_adapters_1000 import MNISTPCAAdapter1000
from modules.output_adapters_1000 import OutputAdapter
from modules.loss import signal_loss_from_lookup
from utils.activation_balancer import ActivationBalancer

class TrainContext1000:
    def __init__(self, config_path="config/large_1000_node.yaml"):
        """
        Enhanced training context for 1000-node network.
        Uses single-sample training methodology that proved effective.
        """
        print("🚀 Initializing 1000-Node Training Context")
        print("=" * 60)
        
        # Load configuration
        self.config = load_config(config_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"📋 Config loaded: {config_path}")
        print(f"   🏗️  Network: {self.config['total_nodes']} nodes ({self.config['num_input_nodes']} input, {self.config['num_output_nodes']} output)")
        print(f"   🎯 Training: {self.config['num_epochs']} epochs, LR={self.config['learning_rate']}")
        print(f"   💻 Device: {self.device}")
        
        # Build/load graph
        self.graph_df = load_or_build_graph(
            save_path=self.config['graph_path'],
            overwrite=False,
            total_nodes=self.config['total_nodes'],
            num_input_nodes=self.config['num_input_nodes'],
            num_output_nodes=self.config['num_output_nodes'],
            vector_dim=self.config['vector_dim'],
            phase_bins=self.config['phase_bins'],
            mag_bins=self.config['mag_bins'],
            cardinality=self.config['cardinality'],
            seed=self.config['seed']
        )
        
        # Initialize core components
        self.node_store = NodeStore(
            self.graph_df, 
            self.config['vector_dim'], 
            self.config['phase_bins'], 
            self.config['mag_bins']
        )
        
        self.lookup_table = ExtendedLookupTableModule(
            self.config['phase_bins'], 
            self.config['mag_bins'], 
            device=self.device
        )
        
        self.phase_cell = PhaseCell(self.config['vector_dim'], self.lookup_table)
        
        # Extract node information
        self.input_nodes = list(self.node_store.input_nodes)
        self.output_nodes = list(self.node_store.output_nodes)
        
        print(f"📊 Graph structure:")
        print(f"   📥 Input nodes: {len(self.input_nodes)} (first 5: {self.input_nodes[:5]})")
        print(f"   📤 Output nodes: {len(self.output_nodes)} ({self.output_nodes})")
        
        # Initialize adapters with enhanced capacity
        self.input_adapter = MNISTPCAAdapter1000(
            vector_dim=self.config['vector_dim'],
            num_input_nodes=self.config['num_input_nodes'],
            phase_bins=self.config['phase_bins'],
            mag_bins=self.config['mag_bins'],
            device=self.device
        )
        
        self.output_adapter = OutputAdapter(
            output_node_ids=self.output_nodes,
            num_classes=10,
            vector_dim=self.config['vector_dim'],
            phase_bins=self.config['phase_bins'],
            mag_bins=self.config['mag_bins'],
            device=self.device
        )
        
        # Initialize activation balancer for large network
        if self.config.get('enable_activation_balancing', False):
            self.activation_balancer = ActivationBalancer(
                output_nodes=self.output_nodes,
                strategy=self.config.get('balancing_strategy', 'round_robin'),
                max_activations_per_epoch=self.config.get('max_activations_per_epoch', 20),
                min_activations_per_epoch=self.config.get('min_activations_per_epoch', 5),
                force_activation_probability=self.config.get('force_activation_probability', 0.5)
            )
            print(f"⚖️  Activation balancing enabled: {self.config.get('balancing_strategy')}")
        else:
            self.activation_balancer = None
        
        print("✅ Training context initialized successfully")
        print("=" * 60)

    def train_single_sample(self, sample_idx, epoch):
        """
        Train on a single sample using proven methodology.
        This approach fixed the training-evaluation mismatch.
        """
        # Get input context for single sample (not merged batch)
        input_context, label = self.input_adapter.get_input_context(sample_idx, self.input_nodes)
        
        # Get target output context
        target_output_context = self.output_adapter.get_output_context(label)
        
        # Forward pass using the existing enhanced forward engine
        activation = run_enhanced_forward(
            graph_df=self.graph_df,
            node_store=self.node_store,
            phase_cell=self.phase_cell,
            lookup_table=self.lookup_table,
            input_context=input_context,
            vector_dim=self.config['vector_dim'],
            phase_bins=self.config['phase_bins'],
            mag_bins=self.config['mag_bins'],
            decay_factor=self.config['decay_factor'],
            min_strength=self.config['min_activation_strength'],
            max_timesteps=self.config['max_timesteps'],
            use_radiation=self.config['use_radiation'],
            top_k_neighbors=self.config['top_k_neighbors'],
            min_output_activation_timesteps=self.config.get('min_output_activation_timesteps', 3),
            device=self.device,
            verbose=False,
            activation_balancer=self.activation_balancer
        )
        
        # Compute loss
        loss = self.compute_loss(activation, target_output_context)
        
        # Backward pass using existing backward engine
        backward_pass(
            activation_table=activation,
            node_store=self.node_store,
            phase_cell=self.phase_cell,
            lookup_table=self.lookup_table,
            target_context=target_output_context,
            output_nodes=self.output_nodes,
            learning_rate=self.config['learning_rate'],
            include_all_outputs=True,  # Include all outputs for large network
            verbose=False,
            device=self.device
        )
        
        return loss, activation

    def compute_loss(self, activation, target_output_context):
        """Compute loss using existing signal loss function."""
        total_loss = 0.0
        num_outputs = 0
        
        for node_id in self.output_nodes:
            if node_id in target_output_context:
                # Get prediction from activation table
                pred_phase, pred_mag = activation.table.get(node_id, (None, None, None))[:2]
                if pred_phase is None:
                    continue
                
                # Get target
                target_phase, target_mag = target_output_context[node_id]
                
                # Convert to appropriate types
                pred_phase = pred_phase.to(torch.float32)
                pred_mag = pred_mag.to(torch.float32)
                target_phase = target_phase.to(torch.float32)
                target_mag = target_mag.to(torch.float32)
                
                # Compute loss using existing loss function
                loss = signal_loss_from_lookup(
                    pred_phase, pred_mag,
                    target_phase, target_mag,
                    self.lookup_table
                )
                
                total_loss += loss.item()
                num_outputs += 1
        
        return torch.tensor(total_loss / max(num_outputs, 1))

    def train_epoch(self, epoch):
        """Train for one epoch using single-sample methodology."""
        samples_per_epoch = self.config.get('batch_size', 5)  # Reuse batch_size as samples_per_epoch
        dataset_size = len(self.input_adapter.mnist)
        
        # Random sample selection for each epoch
        sample_indices = random.sample(range(dataset_size), min(samples_per_epoch, dataset_size))
        
        epoch_losses = []
        
        print(f"🔄 Epoch {epoch+1}: Training on {len(sample_indices)} samples")
        
        for i, sample_idx in enumerate(sample_indices):
            try:
                loss, activation = self.train_single_sample(sample_idx, epoch)
                epoch_losses.append(loss.item())
                
                if i % max(1, len(sample_indices) // 3) == 0:  # Progress updates
                    print(f"   📊 Sample {i+1}/{len(sample_indices)}: Loss = {loss.item():.4f}")
                    
            except Exception as e:
                print(f"⚠️  Error training sample {sample_idx}: {e}")
                continue
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
        print(f"✅ Epoch {epoch+1} completed: Avg Loss = {avg_loss:.4f}")
        
        return avg_loss

    def train(self):
        """Main training loop for 1000-node network."""
        print("🎯 Starting 1000-Node Network Training")
        print("=" * 60)
        
        num_epochs = self.config['num_epochs']
        warmup_epochs = self.config.get('warmup_epochs', 0)
        
        losses = []
        
        for epoch in range(num_epochs):
            is_warmup = epoch < warmup_epochs
            
            if is_warmup:
                print(f"🔥 WARMUP Epoch {epoch+1}/{warmup_epochs}")
            else:
                print(f"🚀 Training Epoch {epoch+1}/{num_epochs}")
            
            epoch_loss = self.train_epoch(epoch)
            losses.append(epoch_loss)
            
            # Progress reporting
            if (epoch + 1) % 10 == 0 or epoch < 5:
                print(f"📈 Progress: Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
                
                # Show recent loss trend
                if len(losses) >= 5:
                    recent_trend = np.mean(losses[-5:]) - np.mean(losses[-10:-5]) if len(losses) >= 10 else 0
                    trend_symbol = "📉" if recent_trend < 0 else "📈"
                    print(f"   {trend_symbol} Recent trend: {recent_trend:+.4f}")
        
        print("=" * 60)
        print("🎉 Training completed!")
        print(f"📊 Final loss: {losses[-1]:.4f}")
        print(f"📈 Loss improvement: {losses[0] - losses[-1]:+.4f}")
        
        return losses

    def evaluate_sample(self, sample_idx):
        """Evaluate a single sample (for testing)."""
        input_context, true_label = self.input_adapter.get_input_context(sample_idx, self.input_nodes)
        
        # Forward pass
        activation = run_enhanced_forward(
            graph_df=self.graph_df,
            node_store=self.node_store,
            phase_cell=self.phase_cell,
            lookup_table=self.lookup_table,
            input_context=input_context,
            vector_dim=self.config['vector_dim'],
            phase_bins=self.config['phase_bins'],
            mag_bins=self.config['mag_bins'],
            decay_factor=self.config['decay_factor'],
            min_strength=self.config['min_activation_strength'],
            max_timesteps=self.config['max_timesteps'],
            use_radiation=self.config['use_radiation'],
            top_k_neighbors=self.config['top_k_neighbors'],
            min_output_activation_timesteps=self.config.get('min_output_activation_timesteps', 3),
            device=self.device,
            verbose=False
        )
        
        # Convert activation to format expected by output adapter
        final_activations = {}
        for node_id in self.output_nodes:
            if node_id in activation.table:
                phase_idx, mag_idx = activation.table[node_id][:2]
                final_activations[node_id] = (phase_idx, mag_idx)
        
        predicted_label = self.output_adapter.predict_label(final_activations)
        
        return predicted_label, true_label

    def evaluate_accuracy(self, num_samples=100):
        """Evaluate accuracy on a subset of samples."""
        print(f"🧪 Evaluating accuracy on {num_samples} samples...")
        
        dataset_size = len(self.input_adapter.mnist)
        test_indices = random.sample(range(dataset_size), min(num_samples, dataset_size))
        
        correct = 0
        total = 0
        
        for idx in test_indices:
            try:
                pred_label, true_label = self.evaluate_sample(idx)
                if pred_label == true_label:
                    correct += 1
                total += 1
            except Exception as e:
                print(f"⚠️  Error evaluating sample {idx}: {e}")
                continue
        
        accuracy = correct / total if total > 0 else 0.0
        print(f"🎯 Accuracy: {correct}/{total} = {accuracy:.1%}")
        
        return accuracy

def main():
    """Test the 1000-node training pipeline."""
    print("🧪 Testing 1000-Node Training Pipeline")
    print("=" * 60)
    
    try:
        # Initialize training context
        trainer = TrainContext1000()
        
        # Quick test - train for a few epochs
        print("\n🔬 Quick training test (5 epochs)...")
        trainer.config['num_epochs'] = 5
        trainer.config['warmup_epochs'] = 2
        
        losses = trainer.train()
        
        # Evaluate accuracy
        accuracy = trainer.evaluate_accuracy(num_samples=50)
        
        print("\n📋 Test Results:")
        print(f"   ✅ Training completed successfully")
        print(f"   📊 Final loss: {losses[-1]:.4f}")
        print(f"   🎯 Accuracy: {accuracy:.1%}")
        print(f"   🚀 Ready for full training!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
