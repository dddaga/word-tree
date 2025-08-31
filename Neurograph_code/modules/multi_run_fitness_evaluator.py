"""
Multi-Run Fitness Evaluator for Genetic Algorithm
Evaluates candidates using multiple stratified training runs to reduce variance
"""

import os
import sys
import yaml
import json
import tempfile
import shutil
import hashlib
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import logging

# Add current directory to path for NeuroGraph imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.stratified_data_manager import StratifiedDataManager, CustomDatasetAdapter
from train.modular_train_context import create_modular_train_context


class MultiRunFitnessEvaluator:
    """
    Multi-run fitness evaluator with stratified sampling and variance reduction.
    
    Features:
    - Multiple training runs per candidate (default: 5)
    - Stratified sampling ensuring class balance
    - Fixed test set for fair comparison
    - Dynamic epoch calculation based on batch size
    - Comprehensive statistics tracking
    """
    
    def __init__(self, num_runs: int = 5, 
                 training_samples_per_run: int = 500,
                 test_samples: int = 500,
                 samples_per_class: int = 50,
                 base_seed: int = 1000):
        """
        Initialize multi-run fitness evaluator.
        
        Args:
            num_runs: Number of training runs per candidate
            training_samples_per_run: Training samples per run
            test_samples: Fixed test set size
            samples_per_class: Samples per class for stratification
            base_seed: Base seed for reproducible sampling
        """
        self.num_runs = num_runs
        self.training_samples_per_run = training_samples_per_run
        self.test_samples = test_samples
        self.samples_per_class = samples_per_class
        self.base_seed = base_seed
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize stratified data manager
        self.data_manager = StratifiedDataManager(
            training_samples_per_run=training_samples_per_run,
            test_samples=test_samples,
            samples_per_class=samples_per_class,
            num_classes=10
        )
        
        # Statistics tracking
        self.evaluation_stats = {
            'total_evaluations': 0,
            'total_runs': 0,
            'run_times': [],
            'fitness_variances': [],
            'epoch_calculations': []
        }
        
        self.logger.info(f"MultiRunFitnessEvaluator initialized:")
        self.logger.info(f"  Number of runs per candidate: {num_runs}")
        self.logger.info(f"  Training samples per run: {training_samples_per_run}")
        self.logger.info(f"  Fixed test samples: {test_samples}")
        self.logger.info(f"  Samples per class: {samples_per_class}")
    
    def calculate_epochs(self, batch_size: int, total_samples: int = None) -> int:
        """
        Calculate epochs needed to train on exactly total_samples.
        
        Args:
            batch_size: Batch size from hyperparameters
            total_samples: Total training samples (default: training_samples_per_run)
            
        Returns:
            Number of epochs needed
        """
        if total_samples is None:
            total_samples = self.training_samples_per_run
        
        # Calculate epochs to process all samples at least once
        epochs = max(1, total_samples // batch_size)
        
        # Log epoch calculation for statistics
        self.evaluation_stats['epoch_calculations'].append({
            'batch_size': batch_size,
            'total_samples': total_samples,
            'calculated_epochs': epochs,
            'actual_samples_processed': epochs * batch_size
        })
        
        return epochs
    
    def create_neurograph_config_for_run(self, individual: Dict[str, Any], 
                                       run_id: int, 
                                       temp_dir: str) -> Dict[str, Any]:
        """
        Create NeuroGraph configuration for a specific run.
        
        Args:
            individual: Hyperparameter configuration
            run_id: Run identifier
            temp_dir: Temporary directory for this evaluation
            
        Returns:
            Complete NeuroGraph configuration
        """
        # Calculate epochs for this batch size
        epochs = self.calculate_epochs(individual['batch_size'])
        
        # Calculate derived parameters
        total_nodes = 200 + 10 + 790  # input + output + intermediate
        input_nodes = 200
        output_nodes = 10
        
        # Build complete configuration
        config = {
            'mode': 'modular',
            'device': 'cuda',
            
            # Architecture parameters
            'architecture': {
                'total_nodes': total_nodes,
                'input_nodes': input_nodes,
                'output_nodes': output_nodes,
                'vector_dim': individual['vector_dim'],
                'seed': 42
            },
            
            # Resolution parameters
            'resolution': {
                'phase_bins': individual['phase_bins'],
                'mag_bins': individual['mag_bins'],
                'resolution_increase': individual['phase_bins'] * individual['mag_bins'] // 64
            },
            
            # Graph structure
            'graph_structure': {
                'cardinality': individual['cardinality']
            },
            
            # Training parameters - UPDATED for stratified sampling
            'training': {
                'gradient_accumulation': {
                    'enabled': True,
                    'accumulation_steps': 8,  # Fixed
                    'lr_scaling': 'sqrt',
                    'buffer_size': 1500
                },
                'optimizer': {
                    'base_learning_rate': individual['learning_rate'],
                    'effective_learning_rate': individual['learning_rate'] * (8 ** 0.5),
                    'num_epochs': epochs,  # Dynamic based on batch size
                    'warmup_epochs': individual['warmup_epochs'],
                    'batch_size': individual['batch_size']
                }
            },
            
            # Forward pass parameters
            'forward_pass': {
                'max_timesteps': 50,
                'decay_factor': individual['decay_factor'],
                'min_activation_strength': individual['min_activation_strength'],
                'use_radiation': True,
                'top_k_neighbors': 4
            },
            
            # Input processing
            'input_processing': {
                'adapter_type': 'linear_projection',
                'input_dim': 784,
                'normalization': 'layer_norm',
                'dropout': 0.1,
                'learnable': True
            },
            
            # Class encoding
            'class_encoding': {
                'type': 'orthogonal',
                'num_classes': 10,
                'encoding_dim': individual['vector_dim'],
                'orthogonality_threshold': individual['orthogonality_threshold']
            },
            
            # Loss function
            'loss_function': {
                'type': 'categorical_crossentropy',
                'temperature': 1.0,
                'label_smoothing': 0.0
            },
            
            # Batch evaluation
            'batch_evaluation': {
                'enabled': True,
                'batch_size': 16,
                'streaming': True
            },
            
            # Debugging - UPDATED for stratified evaluation
            'debugging': {
                'evaluation_samples': self.test_samples,  # Fixed test set size
                'final_evaluation_samples': self.test_samples,
                'log_level': 'INFO'
            },
            
            # Stratified sampling configuration - NEW
            'stratified_sampling': {
                'enabled': True,
                'training_samples': self.training_samples_per_run,
                'test_samples': self.test_samples,
                'samples_per_class': self.samples_per_class,
                'run_id': run_id,
                'base_seed': self.base_seed
            },
            
            # Paths (will be set dynamically)
            'paths': {
                'graph_path': os.path.join(temp_dir, f"temp_graph_run_{run_id}.pkl"),
                'training_curves_path': os.path.join(temp_dir, f"curves_run_{run_id}.png"),
                'checkpoint_path': os.path.join(temp_dir, f"checkpoints_run_{run_id}/")
            }
        }
        
        return config
    
    def evaluate_single_run(self, individual: Dict[str, Any], 
                          run_id: int, 
                          temp_dir: str) -> Dict[str, Any]:
        """
        Evaluate a single training run for a candidate.
        
        Args:
            individual: Hyperparameter configuration
            run_id: Run identifier (0, 1, 2, ...)
            temp_dir: Temporary directory for this evaluation
            
        Returns:
            Dictionary with run results
        """
        start_time = datetime.now()
        
        try:
            # Get stratified training samples for this run
            training_indices = self.data_manager.get_training_samples(run_id, self.base_seed)
            test_indices = self.data_manager.get_fixed_test_set()
            
            # Validate stratification
            train_distribution = self.data_manager.validate_stratification(training_indices, use_test_set=False)
            test_distribution = self.data_manager.validate_stratification(test_indices, use_test_set=True)
            
            self.logger.info(f"Run {run_id}: Training distribution: {train_distribution}")
            self.logger.info(f"Run {run_id}: Test distribution: {test_distribution}")
            
            # Create NeuroGraph configuration for this run
            config = self.create_neurograph_config_for_run(individual, run_id, temp_dir)
            
            # Save temporary config file
            config_path = os.path.join(temp_dir, f"temp_config_run_{run_id}.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # Create and configure trainer
            trainer = create_modular_train_context(config_path)
            
            # Replace the trainer's dataset with our stratified samples
            # This is a bit of a hack, but necessary for stratified sampling
            training_dataset = CustomDatasetAdapter(
                self.data_manager.mnist_train, 
                training_indices
            )
            test_dataset = CustomDatasetAdapter(
                self.data_manager.mnist_test, 
                test_indices
            )
            
            # Update trainer's input adapter to use our stratified dataset
            trainer.input_adapter.mnist = training_dataset
            
            # Run training
            self.logger.info(f"Run {run_id}: Starting training with {len(training_indices)} samples, "
                           f"{config['training']['optimizer']['num_epochs']} epochs")
            
            losses = trainer.train()
            
            # Evaluate on fixed test set
            # Temporarily replace dataset for evaluation
            original_dataset = trainer.input_adapter.mnist
            trainer.input_adapter.mnist = test_dataset
            
            validation_accuracy = trainer.evaluate_accuracy(
                num_samples=self.test_samples,
                use_batch_evaluation=True
            )
            
            # Restore original dataset
            trainer.input_adapter.mnist = original_dataset
            
            # Calculate run time
            run_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare run results
            run_result = {
                'run_id': run_id,
                'fitness': validation_accuracy,
                'final_loss': losses[-1] if losses else 0.0,
                'training_samples': len(training_indices),
                'test_samples': len(test_indices),
                'epochs': config['training']['optimizer']['num_epochs'],
                'batch_size': individual['batch_size'],
                'run_time': run_time,
                'train_distribution': train_distribution,
                'test_distribution': test_distribution,
                'hyperparameters': individual.copy()
            }
            
            self.logger.info(f"Run {run_id}: Completed - Fitness: {validation_accuracy:.4f}, "
                           f"Time: {run_time:.1f}s")
            
            return run_result
            
        except Exception as e:
            self.logger.error(f"Run {run_id}: Error during evaluation: {e}")
            
            # Return failed run result
            run_time = (datetime.now() - start_time).total_seconds()
            return {
                'run_id': run_id,
                'fitness': 0.0,
                'error': str(e),
                'run_time': run_time,
                'failed': True
            }
    
    def evaluate_candidate_fitness(self, individual: Dict[str, Any]) -> float:
        """
        Evaluate candidate fitness using multiple stratified training runs.
        
        Args:
            individual: Hyperparameter configuration
            
        Returns:
            Average fitness across all runs
        """
        self.evaluation_stats['total_evaluations'] += 1
        evaluation_start = datetime.now()
        
        # Create temporary directory for this evaluation
        temp_dir = tempfile.mkdtemp(prefix="multi_run_eval_")
        
        try:
            self.logger.info(f"Evaluating candidate with {self.num_runs} runs:")
            param_str = ", ".join([f"{k}={v}" for k, v in individual.items()])
            self.logger.info(f"  Parameters: {param_str}")
            
            # Run multiple evaluations
            run_results = []
            for run_id in range(self.num_runs):
                self.logger.info(f"  Starting run {run_id + 1}/{self.num_runs}")
                
                run_result = self.evaluate_single_run(individual, run_id, temp_dir)
                run_results.append(run_result)
                
                self.evaluation_stats['total_runs'] += 1
            
            # Calculate statistics
            fitness_scores = [result['fitness'] for result in run_results if not result.get('failed', False)]
            
            if not fitness_scores:
                self.logger.error("All runs failed for this candidate")
                return 0.0
            
            # Calculate mean fitness and statistics
            mean_fitness = np.mean(fitness_scores)
            fitness_std = np.std(fitness_scores)
            fitness_min = np.min(fitness_scores)
            fitness_max = np.max(fitness_scores)
            
            # Track variance statistics
            self.evaluation_stats['fitness_variances'].append(fitness_std)
            
            # Calculate total evaluation time
            total_time = (datetime.now() - evaluation_start).total_seconds()
            self.evaluation_stats['run_times'].append(total_time)
            
            # Create evaluation report
            evaluation_report = self.data_manager.create_evaluation_report(run_results)
            
            # Log comprehensive results
            self.logger.info(f"Multi-run evaluation completed:")
            self.logger.info(f"  Mean fitness: {mean_fitness:.4f} ± {fitness_std:.4f}")
            self.logger.info(f"  Range: [{fitness_min:.4f}, {fitness_max:.4f}]")
            self.logger.info(f"  Successful runs: {len(fitness_scores)}/{self.num_runs}")
            self.logger.info(f"  Total time: {total_time:.1f}s")
            self.logger.info(f"  Coefficient of variation: {evaluation_report['variance_reduction']['coefficient_of_variation']:.3f}")
            
            return mean_fitness
            
        except Exception as e:
            self.logger.error(f"Error in multi-run evaluation: {e}")
            return 0.0
            
        finally:
            # Clean up temporary directory
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    self.logger.warning(f"Failed to clean up temp directory {temp_dir}: {e}")
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive evaluation statistics."""
        stats = self.evaluation_stats.copy()
        
        if stats['run_times']:
            stats['timing_statistics'] = {
                'mean_evaluation_time': np.mean(stats['run_times']),
                'total_evaluation_time': np.sum(stats['run_times']),
                'min_evaluation_time': np.min(stats['run_times']),
                'max_evaluation_time': np.max(stats['run_times'])
            }
        
        if stats['fitness_variances']:
            stats['variance_statistics'] = {
                'mean_fitness_std': np.mean(stats['fitness_variances']),
                'min_fitness_std': np.min(stats['fitness_variances']),
                'max_fitness_std': np.max(stats['fitness_variances'])
            }
        
        if stats['epoch_calculations']:
            batch_sizes = [calc['batch_size'] for calc in stats['epoch_calculations']]
            epochs = [calc['calculated_epochs'] for calc in stats['epoch_calculations']]
            
            stats['epoch_statistics'] = {
                'batch_size_range': [min(batch_sizes), max(batch_sizes)],
                'epoch_range': [min(epochs), max(epochs)],
                'mean_epochs': np.mean(epochs)
            }
        
        # Add data manager statistics
        stats['data_manager_stats'] = self.data_manager.get_statistics()
        
        return stats
    
    def reset_statistics(self):
        """Reset all evaluation statistics."""
        self.evaluation_stats = {
            'total_evaluations': 0,
            'total_runs': 0,
            'run_times': [],
            'fitness_variances': [],
            'epoch_calculations': []
        }
        self.logger.info("Evaluation statistics reset")


# Factory function for easy creation
def create_multi_run_fitness_evaluator(num_runs: int = 5,
                                     training_samples_per_run: int = 500,
                                     test_samples: int = 500,
                                     samples_per_class: int = 50) -> MultiRunFitnessEvaluator:
    """
    Factory function to create multi-run fitness evaluator.
    
    Args:
        num_runs: Number of runs per candidate
        training_samples_per_run: Training samples per run
        test_samples: Fixed test set size
        samples_per_class: Samples per class for stratification
        
    Returns:
        MultiRunFitnessEvaluator instance
    """
    return MultiRunFitnessEvaluator(
        num_runs=num_runs,
        training_samples_per_run=training_samples_per_run,
        test_samples=test_samples,
        samples_per_class=samples_per_class
    )
