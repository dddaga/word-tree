"""
Modular Configuration System for NeuroGraph
Modern configuration loader for production use.
"""

import yaml
import os
from typing import Dict, Any, Optional
import torch

class ModularConfig:
    """Modern configuration loader for NeuroGraph."""
    
    def __init__(self, config_path: str = "config/production.yaml"):
        """
        Initialize modular configuration system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = {}
        self.mode = "modular"
        
        self.load_config()
        self.validate_config()
        self.setup_derived_parameters()
    
    def load_config(self):
        """Load configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            self.mode = self.config.get('system', {}).get('mode', 'modular')
            print(f"✅ Loaded {self.mode} configuration from {self.config_path}")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}")
    
    
    def validate_config(self):
        """Validate configuration parameters."""
        arch = self.config.get('architecture', {})
        resolution = self.config.get('resolution', {})
        
        # Validate architecture
        total_nodes = arch.get('total_nodes', 0)
        input_nodes = arch.get('input_nodes', 0)
        output_nodes = arch.get('output_nodes', 0)
        
        if total_nodes != input_nodes + output_nodes + arch.get('intermediate_nodes', 0):
            print(f"⚠️  Node count mismatch: {total_nodes} != {input_nodes} + {output_nodes} + {arch.get('intermediate_nodes', 0)}")
        
        # Validate resolution
        phase_bins = resolution.get('phase_bins', 8)
        mag_bins = resolution.get('mag_bins', 256)
        
        if phase_bins < 8 or mag_bins < 256:
            print(f"⚠️  Low resolution detected: {phase_bins} phase, {mag_bins} magnitude bins")
        
        # Validate gradient accumulation
        training = self.config.get('training', {})
        grad_accum = training.get('gradient_accumulation', {})
        
        if grad_accum.get('enabled', False):
            steps = grad_accum.get('accumulation_steps', 1)
            if steps < 2:
                print(f"⚠️  Gradient accumulation enabled but steps = {steps}")
    
    def setup_derived_parameters(self):
        """Setup derived parameters based on configuration."""
        arch = self.config['architecture']
        
        # Calculate intermediate nodes if not specified
        if 'intermediate_nodes' not in arch:
            arch['intermediate_nodes'] = (
                arch['total_nodes'] - 
                arch['input_nodes'] - 
                arch['output_nodes']
            )
        
        # Setup device
        self.config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Setup learning rate scaling
        training = self.config.get('training', {})
        grad_accum = training.get('gradient_accumulation', {})
        
        if grad_accum.get('enabled', False):
            base_lr = training['optimizer']['base_learning_rate']
            steps = grad_accum['accumulation_steps']
            scaling = grad_accum.get('lr_scaling', 'none')
            
            if scaling == 'sqrt':
                scaled_lr = base_lr * (steps ** 0.5)
            elif scaling == 'linear':
                scaled_lr = base_lr * steps
            else:
                scaled_lr = base_lr
            
            training['optimizer']['effective_learning_rate'] = scaled_lr
            
            print(f"📊 Learning rate scaling: {base_lr:.4f} → {scaled_lr:.4f} (√{steps} = {steps**0.5:.2f}x)")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path (e.g., 'architecture.total_nodes')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path
            value: Value to set
        """
        keys = key_path.split('.')
        config_ref = self.config
        
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        
        config_ref[keys[-1]] = value
    
    def is_modular(self) -> bool:
        """Check if running in modular mode."""
        return self.mode == "modular"
    
    def get_summary(self) -> str:
        """Get configuration summary."""
        arch = self.config['architecture']
        resolution = self.config['resolution']
        training = self.config['training']
        
        summary = f"""
NeuroGraph Configuration Summary ({self.mode.upper()} mode)
{'='*50}
Architecture:
  • Total nodes: {arch['total_nodes']}
  • Input nodes: {arch['input_nodes']}
  • Output nodes: {arch['output_nodes']}
  • Vector dimension: {arch['vector_dim']}

Resolution:
  • Phase bins: {resolution['phase_bins']}
  • Magnitude bins: {resolution['mag_bins']}
  • Resolution increase: {resolution.get('resolution_increase', 1)}x

Training:
  • Gradient accumulation: {training.get('gradient_accumulation', {}).get('enabled', False)}
  • Accumulation steps: {training.get('gradient_accumulation', {}).get('accumulation_steps', 1)}
  • Base learning rate: {training['optimizer']['base_learning_rate']}
  • Effective learning rate: {training['optimizer'].get('effective_learning_rate', 'N/A')}

Input Processing:
  • Adapter type: {self.config['input_processing']['adapter_type']}
  • Learnable: {self.config['input_processing']['learnable']}

Loss Function:
  • Type: {self.config['loss_function']['type']}
  • Class encoding: {self.config['class_encoding']['type']}
        """
        
        return summary.strip()
    
    def save_config(self, output_path: Optional[str] = None):
        """Save current configuration to file."""
        if output_path is None:
            output_path = self.config_path.replace('.yaml', '_saved.yaml')
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        print(f"� Configuration saved to {output_path}")

def load_modular_config(config_path: str = "config/modular_neurograph.yaml") -> ModularConfig:
    """Convenience function to load modular configuration."""
    return ModularConfig(config_path)
