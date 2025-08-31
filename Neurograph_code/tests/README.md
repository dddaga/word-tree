# NeuroGraph Test Suite

## 🧪 Testing Overview

This directory contains comprehensive tests for the NeuroGraph project, organized by category for efficient testing and development.

## 📁 Test Directory Structure

### ⚡ Performance Tests (`tests/performance/`)
Tests focused on performance optimization, GPU acceleration, and system efficiency:

- **`test_batch_evaluation_performance.py`** - Batch evaluation performance testing
- **`test_gpu_optimizations.py`** - GPU optimization and CUDA acceleration tests
- **`test_performance_optimizations.py`** - General performance optimization tests

### 🧬 Genetic Algorithm Tests (`tests/genetic/`)
Tests for genetic algorithm implementations and hyperparameter optimization:

- **`test_fitness_caching.py`** - Fitness caching system tests
- **`test_genetic_tuner.py`** - Core genetic algorithm tuner tests
- **`test_stratified_genetic_tuner.py`** - Stratified genetic algorithm tests

### 🔄 Integration Tests (`tests/integration/`)
System integration and compatibility tests (currently empty - ready for future tests):

- *Ready for integration tests*
- *End-to-end system tests*
- *Component compatibility tests*

## 🚀 Running Tests

### Prerequisites
```bash
# Ensure you're in the NeuroGraph root directory
cd /path/to/Neurograph

# Activate your environment (if using conda/venv)
conda activate your-env  # or source venv/bin/activate
```

### Run All Tests
```bash
# Run all tests in the test suite
python -m pytest tests/ -v

# Run tests with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

### Run Specific Test Categories

#### Performance Tests
```bash
# Run all performance tests
python -m pytest tests/performance/ -v

# Run specific performance test
python tests/performance/test_gpu_optimizations.py
python tests/performance/test_batch_evaluation_performance.py
python tests/performance/test_performance_optimizations.py
```

#### Genetic Algorithm Tests
```bash
# Run all genetic algorithm tests
python -m pytest tests/genetic/ -v

# Run specific genetic tests
python tests/genetic/test_genetic_tuner.py
python tests/genetic/test_stratified_genetic_tuner.py
python tests/genetic/test_fitness_caching.py
```

#### Integration Tests
```bash
# Run integration tests (when available)
python -m pytest tests/integration/ -v
```

## 🎯 Test Categories Explained

### Performance Tests
These tests verify that the system performs efficiently and utilizes GPU resources optimally:

- **GPU Optimization**: Tests CUDA acceleration, memory usage, and vectorized operations
- **Batch Evaluation**: Tests batch processing performance and memory efficiency
- **Performance Benchmarks**: Tests overall system performance and optimization effectiveness

### Genetic Algorithm Tests
These tests ensure the genetic algorithm implementations work correctly:

- **Fitness Caching**: Tests the caching system for fitness evaluations
- **Genetic Tuner**: Tests core genetic algorithm functionality
- **Stratified Tuning**: Tests stratified sampling and multi-run evaluation

### Integration Tests
These tests verify that all components work together correctly:

- **End-to-End**: Complete system workflow tests
- **Component Integration**: Tests component compatibility and data flow
- **Configuration**: Tests configuration loading and system setup

## 🔧 Test Configuration

### GPU Testing
Tests automatically detect GPU availability:
- **CUDA Available**: Runs GPU-accelerated tests
- **CPU Only**: Falls back to CPU-based testing
- **Memory Management**: Tests respect GPU memory limits

### Test Data
Tests use:
- **Mock Data**: For unit tests and isolated component testing
- **MNIST Subset**: For integration and performance testing
- **Synthetic Data**: For genetic algorithm and optimization testing

## 📊 Performance Benchmarks

### Expected Performance (RTX 3050)
- **Forward Pass**: ~0.1-0.3 seconds per sample
- **Batch Evaluation**: ~0.05-0.1 seconds per sample
- **GPU Memory**: <10 MB for standard configurations
- **Initialization**: <3 seconds for full system setup

### Performance Test Thresholds
Tests verify performance meets minimum requirements:
- **GPU Utilization**: >50% during intensive operations
- **Memory Efficiency**: <20 MB total GPU memory usage
- **Throughput**: >2 samples/second for evaluation
- **Initialization**: <5 seconds for system startup

## 🐛 Debugging Tests

### Verbose Output
```bash
# Run tests with detailed output
python -m pytest tests/ -v -s

# Run with debugging information
python -m pytest tests/ --tb=long --capture=no
```

### GPU Debugging
```bash
# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Run GPU-specific tests only
python -m pytest tests/performance/test_gpu_optimizations.py -v
```

### Memory Debugging
```bash
# Run tests with memory profiling
python -m pytest tests/performance/ --profile-svg

# Monitor GPU memory during tests
nvidia-smi -l 1  # Run in separate terminal
```

## 📈 Test Coverage

### Current Coverage Areas
- ✅ **Performance**: GPU optimization, batch processing, memory efficiency
- ✅ **Genetic Algorithms**: Core functionality, caching, stratified sampling
- 🔄 **Integration**: Ready for implementation (currently empty)

### Future Test Areas
- **Training Pipeline**: End-to-end training tests
- **Configuration**: YAML configuration validation
- **Error Handling**: Robustness and error recovery tests
- **Compatibility**: Different GPU/CPU configurations

## 🤝 Contributing Tests

### Adding New Tests
1. **Choose Category**: Place in appropriate directory (`performance/`, `genetic/`, `integration/`)
2. **Follow Naming**: Use `test_*.py` naming convention
3. **Include Documentation**: Add docstrings and comments
4. **Update README**: Add test description to this file

### Test Structure
```python
import pytest
import torch
from your_module import YourClass

class TestYourFeature:
    """Test suite for YourFeature functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_basic_functionality(self):
        """Test basic functionality works correctly."""
        # Your test code here
        assert True
    
    def test_gpu_acceleration(self):
        """Test GPU acceleration if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        # Your GPU test code here
```

### Test Best Practices
- **Isolation**: Each test should be independent
- **Cleanup**: Clean up resources after tests
- **Assertions**: Use clear, descriptive assertions
- **Documentation**: Document test purpose and expected behavior
- **Performance**: Include performance benchmarks where relevant

## 🔍 Continuous Integration

### Automated Testing
Tests are designed to run in CI/CD environments:
- **GPU Optional**: Tests gracefully handle missing GPU
- **Fast Execution**: Core tests complete in <5 minutes
- **Resource Efficient**: Minimal memory and compute requirements
- **Clear Output**: Detailed reporting for debugging

### Test Matrix
Tests are validated across:
- **Python Versions**: 3.8, 3.9, 3.10, 3.11
- **PyTorch Versions**: Latest stable releases
- **Hardware**: CPU-only and GPU-enabled environments
- **Operating Systems**: Linux, Windows, macOS

---

**Last Updated**: February 2025  
**Status**: ✅ Organized and documented  
**Next Update**: Integration test implementation
