"""
Stratified Data Manager for Genetic Algorithm Fitness Evaluation
Manages stratified sampling and fixed test sets for robust candidate evaluation
"""

import torch
import numpy as np
from torchvision import datasets, transforms
from typing import List, Dict, Tuple, Optional
import random
from collections import defaultdict
import logging

class StratifiedDataManager:
    """
    Manages stratified sampling for training and fixed test sets for evaluation.
    
    Features:
    - Stratified sampling ensuring equal class representation
    - Fixed test set for fair candidate comparison
    - Reproducible sampling with seeds
    - Efficient class-based indexing
    """
    
    def __init__(self, training_samples_per_run: int = 500, 
                 test_samples: int = 500, 
                 samples_per_class: int = 50,
                 num_classes: int = 10,
                 dataset_path: str = "data"):
        """
        Initialize stratified data manager.
        
        Args:
            training_samples_per_run: Total training samples per evaluation run
            test_samples: Total samples in fixed test set
            samples_per_class: Samples per class (should equal training_samples_per_run // num_classes)
            num_classes: Number of classes (10 for MNIST)
            dataset_path: Path to dataset storage
        """
        self.training_samples_per_run = training_samples_per_run
        self.test_samples = test_samples
        self.samples_per_class = samples_per_class
        self.num_classes = num_classes
        self.dataset_path = dataset_path
        
        # Validate configuration
        if training_samples_per_run != samples_per_class * num_classes:
            raise ValueError(f"training_samples_per_run ({training_samples_per_run}) must equal "
                           f"samples_per_class ({samples_per_class}) × num_classes ({num_classes})")
        
        if test_samples != samples_per_class * num_classes:
            raise ValueError(f"test_samples ({test_samples}) must equal "
                           f"samples_per_class ({samples_per_class}) × num_classes ({num_classes})")
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Load and organize dataset
        self.setup_dataset()
        self.organize_by_class()
        self.create_fixed_test_set()
        
        self.logger.info(f"StratifiedDataManager initialized:")
        self.logger.info(f"  Training samples per run: {training_samples_per_run}")
        self.logger.info(f"  Test samples (fixed): {test_samples}")
        self.logger.info(f"  Samples per class: {samples_per_class}")
        self.logger.info(f"  Classes available: {len(self.class_indices)}")
    
    def setup_dataset(self):
        """Setup MNIST dataset with proper transforms."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST normalization
            transforms.Lambda(lambda x: x.view(-1))  # Flatten to [784]
        ])
        
        # Load training dataset
        self.mnist_train = datasets.MNIST(
            root=self.dataset_path, train=True, download=True, transform=transform
        )
        
        # Load test dataset (for creating fixed test set)
        self.mnist_test = datasets.MNIST(
            root=self.dataset_path, train=False, download=True, transform=transform
        )
        
        self.logger.info(f"MNIST dataset loaded:")
        self.logger.info(f"  Training samples: {len(self.mnist_train)}")
        self.logger.info(f"  Test samples: {len(self.mnist_test)}")
    
    def organize_by_class(self):
        """Organize dataset indices by class for efficient stratified sampling."""
        self.class_indices = defaultdict(list)
        
        # Organize training data by class
        for idx, (_, label) in enumerate(self.mnist_train):
            self.class_indices[label].append(idx)
        
        # Log class distribution
        self.logger.info("Training data class distribution:")
        for class_id in range(self.num_classes):
            count = len(self.class_indices[class_id])
            self.logger.info(f"  Class {class_id}: {count} samples")
        
        # Verify we have enough samples per class
        min_samples = min(len(indices) for indices in self.class_indices.values())
        if min_samples < self.samples_per_class:
            raise ValueError(f"Not enough samples in smallest class. "
                           f"Need {self.samples_per_class}, have {min_samples}")
    
    def create_fixed_test_set(self):
        """Create a fixed test set with stratified sampling from test data."""
        # Organize test data by class
        test_class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(self.mnist_test):
            test_class_indices[label].append(idx)
        
        # Sample fixed test set with stratification
        self.fixed_test_indices = []
        test_seed = 42  # Fixed seed for reproducible test set
        rng = random.Random(test_seed)
        
        for class_id in range(self.num_classes):
            class_samples = test_class_indices[class_id]
            if len(class_samples) < self.samples_per_class:
                raise ValueError(f"Not enough test samples for class {class_id}. "
                               f"Need {self.samples_per_class}, have {len(class_samples)}")
            
            # Sample without replacement
            selected = rng.sample(class_samples, self.samples_per_class)
            self.fixed_test_indices.extend(selected)
        
        # Shuffle the final test set
        rng.shuffle(self.fixed_test_indices)
        
        self.logger.info(f"Fixed test set created: {len(self.fixed_test_indices)} samples")
        
        # Verify test set class distribution
        test_labels = [self.mnist_test[idx][1] for idx in self.fixed_test_indices]
        test_distribution = {i: test_labels.count(i) for i in range(self.num_classes)}
        self.logger.info(f"Test set class distribution: {test_distribution}")
    
    def get_training_samples(self, run_id: int, base_seed: int = 1000) -> List[int]:
        """
        Get stratified training samples for a specific run.
        
        Args:
            run_id: Run identifier (0, 1, 2, ...)
            base_seed: Base seed for reproducible sampling
            
        Returns:
            List of sample indices for training
        """
        # Create unique seed for this run
        run_seed = base_seed + run_id
        rng = random.Random(run_seed)
        
        training_indices = []
        
        # Sample from each class
        for class_id in range(self.num_classes):
            class_samples = self.class_indices[class_id]
            
            # Sample without replacement
            selected = rng.sample(class_samples, self.samples_per_class)
            training_indices.extend(selected)
        
        # Shuffle the final training set
        rng.shuffle(training_indices)
        
        return training_indices
    
    def get_fixed_test_set(self) -> List[int]:
        """
        Get the fixed test set indices.
        
        Returns:
            List of test sample indices (always the same)
        """
        return self.fixed_test_indices.copy()
    
    def get_sample_data(self, indices: List[int], use_test_set: bool = False) -> List[Tuple[torch.Tensor, int]]:
        """
        Get actual data samples for given indices.
        
        Args:
            indices: List of sample indices
            use_test_set: Whether to use test dataset or training dataset
            
        Returns:
            List of (image, label) tuples
        """
        dataset = self.mnist_test if use_test_set else self.mnist_train
        return [dataset[idx] for idx in indices]
    
    def validate_stratification(self, indices: List[int], use_test_set: bool = False) -> Dict[int, int]:
        """
        Validate that the given indices maintain proper stratification.
        
        Args:
            indices: Sample indices to validate
            use_test_set: Whether indices refer to test set
            
        Returns:
            Dictionary of class_id -> count
        """
        dataset = self.mnist_test if use_test_set else self.mnist_train
        labels = [dataset[idx][1] for idx in indices]
        
        distribution = {i: labels.count(i) for i in range(self.num_classes)}
        return distribution
    
    def get_statistics(self) -> Dict[str, any]:
        """Get statistics about the data manager."""
        return {
            'training_samples_per_run': self.training_samples_per_run,
            'test_samples': self.test_samples,
            'samples_per_class': self.samples_per_class,
            'num_classes': self.num_classes,
            'total_training_available': len(self.mnist_train),
            'total_test_available': len(self.mnist_test),
            'fixed_test_set_size': len(self.fixed_test_indices),
            'class_distribution': {
                class_id: len(indices) 
                for class_id, indices in self.class_indices.items()
            }
        }
    
    def verify_no_overlap(self, training_indices: List[int]) -> bool:
        """
        Verify that training indices don't overlap with fixed test set.
        Note: This is automatically true since we use train/test split from MNIST.
        
        Args:
            training_indices: Training sample indices
            
        Returns:
            True if no overlap (always True for MNIST train/test split)
        """
        # Since we use MNIST train/test split, there's no overlap by design
        # But we can implement this check for other datasets
        return True
    
    def create_evaluation_report(self, run_results: List[Dict]) -> Dict[str, any]:
        """
        Create a comprehensive evaluation report.
        
        Args:
            run_results: List of results from multiple runs
            
        Returns:
            Evaluation report with statistics
        """
        if not run_results:
            return {'error': 'No run results provided'}
        
        fitness_scores = [result.get('fitness', 0.0) for result in run_results]
        
        report = {
            'num_runs': len(run_results),
            'fitness_statistics': {
                'mean': np.mean(fitness_scores),
                'std': np.std(fitness_scores),
                'min': np.min(fitness_scores),
                'max': np.max(fitness_scores),
                'median': np.median(fitness_scores)
            },
            'data_configuration': {
                'training_samples_per_run': self.training_samples_per_run,
                'test_samples': self.test_samples,
                'samples_per_class': self.samples_per_class,
                'stratified': True
            },
            'variance_reduction': {
                'coefficient_of_variation': np.std(fitness_scores) / np.mean(fitness_scores) if np.mean(fitness_scores) > 0 else float('inf'),
                'fitness_range': np.max(fitness_scores) - np.min(fitness_scores)
            }
        }
        
        return report


class CustomDatasetAdapter:
    """
    Adapter for using stratified samples with NeuroGraph input adapters.
    Provides a dataset-like interface for specific sample indices.
    """
    
    def __init__(self, base_dataset, sample_indices: List[int]):
        """
        Initialize adapter with specific sample indices.
        
        Args:
            base_dataset: Original dataset (MNIST)
            sample_indices: List of indices to use
        """
        self.base_dataset = base_dataset
        self.sample_indices = sample_indices
    
    def __len__(self) -> int:
        """Return number of samples in this subset."""
        return len(self.sample_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item by subset index."""
        if idx >= len(self.sample_indices):
            raise IndexError(f"Index {idx} out of range for subset size {len(self.sample_indices)}")
        
        original_idx = self.sample_indices[idx]
        return self.base_dataset[original_idx]
    
    def get_original_index(self, subset_idx: int) -> int:
        """Get original dataset index for a subset index."""
        return self.sample_indices[subset_idx]


# Factory function for easy creation
def create_stratified_data_manager(training_samples: int = 500, 
                                 test_samples: int = 500,
                                 num_classes: int = 10) -> StratifiedDataManager:
    """
    Factory function to create stratified data manager.
    
    Args:
        training_samples: Training samples per run
        test_samples: Fixed test set size
        num_classes: Number of classes
        
    Returns:
        StratifiedDataManager instance
    """
    samples_per_class = training_samples // num_classes
    
    if training_samples != samples_per_class * num_classes:
        raise ValueError(f"training_samples ({training_samples}) must be divisible by "
                        f"num_classes ({num_classes})")
    
    return StratifiedDataManager(
        training_samples_per_run=training_samples,
        test_samples=test_samples,
        samples_per_class=samples_per_class,
        num_classes=num_classes
    )
